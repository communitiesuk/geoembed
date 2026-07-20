"""High-level pipeline facade — the primary entry point for geoembed users."""

from __future__ import annotations

from geoembed.core.config import PipelineConfig, StorageConfig


class EmbeddingsPipeline:
    """
    Main entry point for generating geospatial embeddings.

    Wraps the complexity of backend selection, model initialisation, and
    storage configuration behind a simple interface.

    Usage:
        from geoembed.pipeline import EmbeddingsPipeline
        from geoembed.core.config import StorageConfig, PipelineConfig

        pipeline = EmbeddingsPipeline(
            config=PipelineConfig(
                model="dofa",
                backend="spark",
                batch_size=256,
            ),
            input_storage=StorageConfig(backend="delta", path="catalog.schema.metadata"),
            output_storage=StorageConfig(backend="delta", path="catalog.schema.embeddings"),
        )
        pipeline.run()
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        input_storage: StorageConfig | None = None,
        output_storage: StorageConfig | None = None,
    ):
        """
        Initialise the pipeline.

        Args:
            config: Pipeline configuration. Defaults to sensible values.
            input_storage: Where to read image metadata from.
            output_storage: Where to write embeddings to.
        """
        self.config = config or PipelineConfig()
        self.input_storage = input_storage or self.config.input_storage
        self.output_storage = output_storage or self.config.output_storage

    def run(self) -> None:
        """
        Execute the embedding pipeline using the configured backend.

        Dispatches to Spark GPU or Local backend based on config.backend.
        """
        if self.config.backend in ("spark", "ray"):
            self._run_spark()
        elif self.config.backend == "local":
            self._run_local()
        else:
            msg = f"Unsupported backend: '{self.config.backend}'. Use 'spark' or 'local'."
            raise ValueError(msg)

    def _run_spark(self) -> None:
        """Execute via the Spark GPU backend (full-image tiling)."""
        from geoembed.backends.spark.config import EmbeddingsConfig
        from geoembed.backends.spark.orchestrator import EmbeddingsPipeline as SparkPipeline

        spark_config = EmbeddingsConfig(
            batch_size=self.config.batch_size,
            mixed_precision=self.config.mixed_precision,
        )

        pipeline = SparkPipeline(
            config=spark_config,
            input_storage=self.input_storage,
            output_storage=self.output_storage,
        )
        pipeline.run()

    def _run_local(self) -> None:
        """Execute via the local single-machine backend."""
        from geoembed.backends.local.pipeline import LocalEmbeddingsPipeline
        from geoembed.io.readers.rasterio_reader import RasterioChipReader
        from geoembed.models.registry import get_embedder

        embedder = get_embedder(
            self.config.model,
            mixed_precision=self.config.mixed_precision,
            device=self.config.device,
        )
        reader = RasterioChipReader()

        pipeline = LocalEmbeddingsPipeline(
            embedder=embedder,
            reader=reader,
            input_path=self.input_storage.path,
            output_storage=self.output_storage,
            config=self.config,
        )
        pipeline.run()
