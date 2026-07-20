"""Storage configuration and writer factory."""

from geoembed.core.config import StorageConfig
from geoembed.io.writers.base import EmbeddingWriter


def create_writer(config: StorageConfig) -> EmbeddingWriter:
    """
    Factory function to create the appropriate writer from a StorageConfig.

    Args:
        config: Storage configuration specifying backend and options.

    Returns:
        An EmbeddingWriter instance configured per the StorageConfig.

    Raises:
        ValueError: If the backend is not supported.
    """
    if config.backend == "parquet":
        from geoembed.io.writers.parquet_writer import ParquetWriter

        return ParquetWriter(path=config.path, mode=config.mode)

    if config.backend == "delta":
        from geoembed.io.writers.delta_writer import DeltaWriter

        return DeltaWriter(
            table_name=config.path,
            mode=config.mode,
            optimize_write=config.optimize_write,
            cluster_columns=config.cluster_columns,
        )

    msg = f"Unsupported storage backend: '{config.backend}'. Use 'parquet' or 'delta'."
    raise ValueError(msg)
