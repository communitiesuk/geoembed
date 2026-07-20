"""Configuration objects for the geoembed pipeline."""

from dataclasses import dataclass, field
from typing import Literal


def get_default_device() -> str:
    """
    Automatically selects the best available hardware accelerator.

    Priority: CUDA > MPS > CPU. Requires torch to be installed.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class StorageConfig:
    """
    Backend-agnostic storage configuration.

    Supports Delta Lake (via Spark), local Parquet (via PyArrow), and cloud
    object stores (via obstore).

    Attributes:
        backend: Storage format — "delta" for Unity Catalog/Delta Lake, "parquet" for files.
        path: Table name (delta) or file path (parquet).
        mode: Write mode — "append" or "overwrite".
        optimize_write: Enable write optimisation (Delta auto-compaction).
        cluster_columns: Columns for Delta liquid clustering (ALTER TABLE CLUSTER BY).
            Liquid clustering replaces Z-ordering — it's incremental, automatic, and
            doesn't require manual OPTIMIZE runs.
        store_type: Object store backend for obstore (local, az, s3, gcs).
        store_options: Backend-specific options (account keys, endpoints, etc.).
    """

    backend: Literal["delta", "parquet"] = "parquet"
    path: str = "./embeddings.parquet"
    mode: Literal["append", "overwrite"] = "append"
    optimize_write: bool = True
    cluster_columns: list[str] | None = None
    store_type: Literal["local", "az", "s3", "gcs"] = "local"
    store_options: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    """
    Top-level pipeline configuration.

    DataLoader tuning follows findings from Microsoft's "Optimizing Cloud-to-GPU
    Throughput for Deep Learning With Earth Observation Data" (Zaytar et al., 2025):
    - Block-aligned reads reduce redundant tile decompression
    - persistent_workers=True eliminates worker spawn overhead
    - prefetch_factor hides I/O latency (2 local, 8 remote)
    - num_workers scales with storage type (4 local, 64 remote)

    Attributes:
        model: Model name (from registry) or Embedder instance.
        backend: Compute backend — "spark" for distributed GPU, "local" for single-machine.
        chip_size: Spatial dimension of input chips (e.g., 224 for 224x224).
        batch_size: Number of chips per inference batch.
        mixed_precision: Use FP16/BF16 autocasting during inference.
        device: Explicit device string. If None, auto-detects.
        num_workers: DataLoader worker processes. Higher for remote (64), lower for local (4-8).
        prefetch_factor: Batches to prefetch per worker. Higher hides network latency (8 remote).
        persistent_workers: Keep workers alive between batches (always True for performance).
        tile_aligned: Sort chip reads to align with COG tile boundaries, reducing decompression.
        input_storage: Where to read chip metadata from.
        output_storage: Where to write embeddings to.
    """

    model: str = "dofa"
    backend: Literal["spark", "local"] = "spark"
    chip_size: int = 224
    batch_size: int = 128
    mixed_precision: bool = True
    device: str | None = None
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    tile_aligned: bool = True
    input_storage: StorageConfig = field(default_factory=StorageConfig)
    output_storage: StorageConfig = field(default_factory=StorageConfig)
