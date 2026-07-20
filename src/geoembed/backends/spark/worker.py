"""Full-image GPU worker — reads COG, tiles on GPU, runs inference.

Single Ray actor that processes one image at a time:
1. Read full COG to CPU (via rasterio) — in background thread for double-buffering
2. Transfer to GPU as float32 tensor
3. Tile via torch.unfold() (zero-copy views, ~0ms)
4. Batch inference with DOFA (mixed precision)
5. Compute spatial bounds from affine transform (pure arithmetic)
6. Return Arrow table with embeddings + spatial metadata

Double-buffering: reads the next image while GPU processes the current one,
hiding ~2.7s I/O latency behind ~3.2s GPU compute.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import rasterio
import torch
from rasterio.windows import Window

from geoembed.chipping.services import StrideService
from geoembed.models.base import Embedder


def _load_image(image_path: str) -> tuple[np.ndarray, dict]:
    """Read a full COG to CPU memory. Runs in background thread."""
    path = image_path.replace("dbfs:", "").replace("file:", "")
    with rasterio.open(path) as src:
        image_np = src.read()  # (C, H, W)
        metadata = {
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "parent_name": Path(path).stem,
            "parent_path": path,
        }
    return image_np, metadata


class FullImageGPUWorker:
    """
    Ray MapBatches actor: full-image read → GPU tile → inference → Arrow output.

    Processes images one at a time with double-buffered I/O. Each call receives
    a batch of image paths and returns an Arrow table with all chip embeddings
    and spatial bounds.
    """

    def __init__(
        self,
        chip_size: int = 224,
        batch_size: int = 256,
        mixed_precision: bool = True,
        model_name: str = "dofa",
    ):
        self.chip_size = chip_size
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder: Embedder | None = None
        self.executor = ThreadPoolExecutor(max_workers=1)

    def _setup_model(self) -> None:
        """Initialise model on first call (runs on GPU actor)."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        from geoembed.models.registry import get_embedder

        self.embedder = get_embedder(
            self.model_name,
            mixed_precision=self.mixed_precision,
            device=str(self.device),
        )

    def _tile_on_gpu(self, image_gpu: torch.Tensor) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Tile image on GPU using torch.unfold(). Returns patches + pixel offsets."""
        c, h, w = image_gpu.shape

        overlap_x, overlap_y = StrideService.calculate_optimal_overlap(
            (w, h), (self.chip_size, self.chip_size)
        )
        stride_x = max(1, self.chip_size - overlap_x)
        stride_y = max(1, self.chip_size - overlap_y)

        patches = image_gpu.unfold(1, self.chip_size, stride_y).unfold(2, self.chip_size, stride_x)
        n_rows, n_cols = patches.shape[1], patches.shape[2]

        patches = (
            patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, c, self.chip_size, self.chip_size)
        )

        offsets = [
            (c_idx * stride_x, r_idx * stride_y)
            for r_idx in range(n_rows)
            for c_idx in range(n_cols)
        ]

        return patches, offsets

    def _infer_batched(self, patches: torch.Tensor) -> np.ndarray:
        """Run inference in batches, return numpy embeddings."""
        if self.embedder is None:
            msg = "_setup_model() must be called before inference"
            raise RuntimeError(msg)
        results = []
        n = patches.shape[0]
        with torch.inference_mode():
            for i in range(0, n, self.batch_size):
                batch = patches[i : i + self.batch_size, :3, :, :]
                emb = self.embedder.embed(batch)
                results.append(emb.cpu())
        return torch.cat(results, dim=0).numpy()

    def _process_single_image(self, image_np: np.ndarray, metadata: dict) -> pa.Table:
        """Process one image: GPU transfer → tile → infer → Arrow table."""
        parent_name = metadata["parent_name"]
        parent_path = metadata["parent_path"]
        transform = metadata["transform"]

        image_gpu = torch.from_numpy(image_np).float().to(self.device, non_blocking=True)

        patches, offsets = self._tile_on_gpu(image_gpu)
        del image_gpu

        embeddings = self._infer_batched(patches)
        del patches

        n = len(offsets)
        chip_ids = [f"{parent_name}_{idx:06d}" for idx in range(n)]
        parent_ids = [parent_name] * n
        parent_paths = [parent_path] * n
        minxs = np.empty(n, dtype=np.float64)
        minys = np.empty(n, dtype=np.float64)
        maxxs = np.empty(n, dtype=np.float64)
        maxys = np.empty(n, dtype=np.float64)

        for idx, (col_off, row_off) in enumerate(offsets):
            window = Window(col_off, row_off, self.chip_size, self.chip_size)  # pyrefly: ignore
            minxs[idx], minys[idx], maxxs[idx], maxys[idx] = rasterio.windows.bounds(
                window, transform
            )

        col_offs = np.array([o[0] for o in offsets], dtype=np.int32)
        row_offs = np.array([o[1] for o in offsets], dtype=np.int32)

        return pa.table({
            "chip_id": pa.array(chip_ids, type=pa.string()),
            "parent_id": pa.array(parent_ids, type=pa.string()),
            "parent_path": pa.array(parent_paths, type=pa.string()),
            "col_off": pa.array(col_offs, type=pa.int32()),
            "row_off": pa.array(row_offs, type=pa.int32()),
            "embedding": pa.array(
                [embeddings[i].tolist() for i in range(n)], type=pa.list_(pa.float32())
            ),
            "minx": pa.array(minxs, type=pa.float64()),
            "miny": pa.array(minys, type=pa.float64()),
            "maxx": pa.array(maxxs, type=pa.float64()),
            "maxy": pa.array(maxys, type=pa.float64()),
        })

    def __call__(self, batch: dict[str, np.ndarray]) -> pa.Table:
        """
        Entry point for Ray map_batches.

        Receives a batch of image paths, processes each with double-buffered I/O,
        returns a combined Arrow table with all chip embeddings + spatial metadata.
        """
        if self.embedder is None:
            self._setup_model()

        image_paths = batch["image_path"]

        if len(image_paths) == 0:
            return pa.table({
                "chip_id": pa.array([], type=pa.string()),
                "parent_id": pa.array([], type=pa.string()),
                "parent_path": pa.array([], type=pa.string()),
                "embedding": pa.array([], type=pa.list_(pa.float32())),
                "minx": pa.array([], type=pa.float64()),
                "miny": pa.array([], type=pa.float64()),
                "maxx": pa.array([], type=pa.float64()),
                "maxy": pa.array([], type=pa.float64()),
            })

        tables: list[pa.Table] = []
        future = self.executor.submit(_load_image, str(image_paths[0]))

        for i in range(len(image_paths)):
            image_np, metadata = future.result()

            if i + 1 < len(image_paths):
                future = self.executor.submit(_load_image, str(image_paths[i + 1]))

            table = self._process_single_image(image_np, metadata)
            tables.append(table)
            del image_np

        return pa.concat_tables(tables)
