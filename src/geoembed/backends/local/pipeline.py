"""Local single-machine embedding pipeline with optimised COG streaming.

Implements findings from Microsoft's "Optimizing Cloud-to-GPU Throughput for Deep
Learning With Earth Observation Data" (Zaytar et al., 2025):

1. Tile-aligned read ordering — chips sorted by parent tile to minimise redundant
   decompression (up to 4x I/O reduction for 224px chips on 512px tiles)
2. persistent_workers=True — eliminates DataLoader worker spawn overhead
3. Configurable prefetch_factor — hides I/O latency (2 local, 8 remote)
4. pin_memory=True — enables async CPU→GPU transfer via DMA
5. Moderate num_workers (4-8) with no per-worker threading — paper found this
   outperforms fewer workers with many threads
"""

import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from geoembed.core.config import PipelineConfig, StorageConfig
from geoembed.io.readers.base import ChipReader
from geoembed.io.storage import create_writer
from geoembed.io.tile_sorting import sort_by_tile
from geoembed.models.base import Embedder


class TileGroupedChipDataset(Dataset):
    """
    PyTorch Dataset that reads chips on-the-fly from tile-sorted metadata.

    When metadata is sorted by parent tile (via `sort_by_tile`), sequential
    access through this dataset benefits from GDAL's block cache: adjacent
    chips sharing the same 512x512 tile are served from a single decompression.

    No binary materialisation — reads directly from source COGs.
    """

    def __init__(self, metadata: pd.DataFrame, reader: ChipReader):
        """
        Args:
            metadata: DataFrame with columns: chip_id, parent_path, col_off, row_off,
                      width, height. Should be pre-sorted by tile for optimal I/O.
            reader: ChipReader implementation for fetching pixels.
        """
        self.chip_ids = metadata["chip_id"].values
        self.parent_paths = metadata["parent_path"].values
        self.col_offs = metadata["col_off"].values.astype(np.int32)
        self.row_offs = metadata["row_off"].values.astype(np.int32)
        self.widths = metadata["width"].values.astype(np.int32)
        self.heights = metadata["height"].values.astype(np.int32)
        self.reader = reader

    def __len__(self) -> int:
        return len(self.chip_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor | None, str]:
        """
        Load a single chip.

        Returns:
            Tuple of (tensor or None on failure, chip_id string).
        """
        try:
            pixels = self.reader.read_chip(
                parent_path=str(self.parent_paths[index]),
                col_off=int(self.col_offs[index]),
                row_off=int(self.row_offs[index]),
                width=int(self.widths[index]),
                height=int(self.heights[index]),
            )
            # Ensure (C, H, W) and take first 3 bands
            if pixels.ndim == 3 and (pixels.shape[-1] == 3 or pixels.shape[-1] == 4):
                pixels = pixels.transpose(2, 0, 1)
            pixels = pixels[:3, :, :]
            tensor = torch.from_numpy(pixels.copy()).float()
        except Exception as e:
            print(f"Error reading chip {self.chip_ids[index]}: {e}")
            return None, str(self.chip_ids[index])

        return tensor, str(self.chip_ids[index])


def _collate_fn(batch: list[tuple[torch.Tensor | None, str]]) -> tuple[torch.Tensor, list[str]]:
    """Filter failed samples and stack into a batch tensor."""
    valid = [(img, cid) for img, cid in batch if img is not None]
    if not valid:
        return torch.empty(0), []
    images = torch.stack([img for img, _ in valid])
    ids = [cid for _, cid in valid]
    return images, ids


class LocalEmbeddingsPipeline:
    """
    Single-machine embedding pipeline optimised for COG streaming throughput.

    Key optimisations (from Microsoft paper):
    - Tile-aligned sorting: chips grouped by parent COG tile so GDAL block cache
      serves multiple chips per decompression (up to 4x fewer tile reads)
    - persistent_workers: workers stay alive between batches (no spawn overhead)
    - Aggressive prefetching: buffers multiple batches to hide I/O latency
    - pin_memory: enables async DMA transfer to GPU

    Usage:
        from geoembed.models import DofaEmbedder
        from geoembed.io.readers.rasterio_reader import RasterioChipReader
        from geoembed.core.config import PipelineConfig, StorageConfig

        pipeline = LocalEmbeddingsPipeline(
            embedder=DofaEmbedder(),
            reader=RasterioChipReader(),
            input_path="chip_metadata.parquet",
            output_storage=StorageConfig(path="embeddings.parquet"),
            config=PipelineConfig(batch_size=128, num_workers=4),
        )
        pipeline.run()
    """

    def __init__(
        self,
        embedder: Embedder,
        reader: ChipReader,
        input_path: str,
        output_storage: StorageConfig,
        config: PipelineConfig | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
    ):
        """
        Initialise the local pipeline.

        Args:
            embedder: Instantiated Embedder (e.g., DofaEmbedder).
            reader: ChipReader for fetching pixel data.
            input_path: Path to chip metadata Parquet file.
            output_storage: Where to write embeddings.
            config: Full pipeline config. If provided, overrides batch_size/num_workers.
            batch_size: DataLoader batch size. Default 128 (paper finds larger = higher throughput).
            num_workers: DataLoader worker count. Default 4 for local, 64 for remote.
        """
        self.embedder = embedder
        self.reader = reader
        self.input_path = input_path
        self.output_storage = output_storage

        if config is not None:
            self.batch_size = config.batch_size
            self.num_workers = config.num_workers
            self.prefetch_factor = config.prefetch_factor
            self.persistent_workers = config.persistent_workers
            self.tile_aligned = config.tile_aligned
        else:
            self.batch_size = batch_size or 128
            self.num_workers = num_workers or 4
            self.prefetch_factor = 2
            self.persistent_workers = True
            self.tile_aligned = True

    def _prepare_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Sort metadata by tile for optimal I/O, if tile_aligned is enabled."""
        if not self.tile_aligned:
            return metadata

        required_cols = {"parent_path", "col_off", "row_off"}
        if not required_cols.issubset(metadata.columns):
            print("[LocalEmbeddingsPipeline] WARNING: Missing columns for tile sorting, skipping.")
            return metadata

        sorted_meta = sort_by_tile(metadata, block_size=512)
        print(
            f"[LocalEmbeddingsPipeline] Sorted {len(sorted_meta)} chips by tile "
            f"({sorted_meta['parent_path'].nunique()} parent COGs)"
        )
        return sorted_meta

    def run(self) -> None:
        """Execute the local embedding pipeline with optimised COG streaming."""
        print("[LocalEmbeddingsPipeline] Starting...")
        start_time = time.monotonic()

        # Load and prepare metadata
        metadata = pd.read_parquet(self.input_path)
        print(f"[LocalEmbeddingsPipeline] Loaded {len(metadata)} chips from {self.input_path}")

        metadata = self._prepare_metadata(metadata)

        dataset = TileGroupedChipDataset(metadata, self.reader)

        prefetch = self.prefetch_factor if self.num_workers > 0 else None
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0 and self.persistent_workers),
            prefetch_factor=prefetch,
            shuffle=False,
        )

        writer = create_writer(self.output_storage)

        total_chips = 0
        for images, ids in tqdm(dataloader, desc="Generating embeddings"):
            if images.numel() == 0:
                continue

            embeddings = self.embedder.embed(images)
            emb_np = embeddings.float().cpu().numpy()
            writer.write_batch(ids, emb_np)
            total_chips += len(ids)

        writer.finalize()

        elapsed = time.monotonic() - start_time
        throughput = total_chips / elapsed if elapsed > 0 else 0
        print(
            f"[LocalEmbeddingsPipeline] Finished: {total_chips} chips in {elapsed:.1f}s "
            f"({throughput:.0f} chips/s)"
        )
