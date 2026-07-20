"""Spark-based full-image GPU embedding pipeline.

Distributes COG images across GPU workers via Spark mapInPandas.
Each worker reads a full image, tiles on GPU via torch.unfold(), runs batched
inference, and returns embeddings + spatial bounds.

Double-buffered I/O within each worker hides read latency behind GPU compute.
Uses Spark's native GPU resource scheduling (spark.task.resource.gpu.amount).
"""

from __future__ import annotations

import time
from typing import Any

from geoembed.backends.spark.config import EmbeddingsConfig
from geoembed.core.config import StorageConfig
from geoembed.core.naming import quote_table_name


class EmbeddingsPipeline:
    """
    Full-image GPU embedding pipeline distributed via Spark mapInPandas.

    Each Spark task:
    1. Receives a batch of image paths
    2. Reads full COGs, tiles on GPU, runs inference (with double-buffering)
    3. Returns embeddings + spatial bounds

    Spark handles multi-node GPU distribution via spark.task.resource.gpu.amount = 1.

    Usage:
        from geoembed.backends.spark.config import EmbeddingsConfig
        from geoembed.backends.spark.orchestrator import EmbeddingsPipeline
        from geoembed.core.config import StorageConfig

        config = EmbeddingsConfig(chip_size=224, batch_size=256, mixed_precision=True)
        pipeline = EmbeddingsPipeline(
            config=config,
            input_storage=StorageConfig(backend="delta", path="catalog.schema.metadata"),
            output_storage=StorageConfig(backend="delta", path="catalog.schema.embeddings"),
        )
        pipeline.run()
    """

    def __init__(
        self,
        config: EmbeddingsConfig,
        input_storage: StorageConfig,
        output_storage: StorageConfig,
    ):
        self.config = config
        self.input_storage = input_storage
        self.output_storage = output_storage

    def run(self) -> None:
        """Execute the full-image GPU embedding pipeline via Spark."""
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()

        print("[EmbeddingsPipeline] Starting full-image GPU pipeline...")
        start_time = time.monotonic()

        # Read metadata table — only need image_path
        quoted_input = quote_table_name(self.input_storage.path)
        source_df = (
            spark.read.table(quoted_input)
            .select("image_path")
            .filter("image_path IS NOT NULL AND id NOT LIKE 'ERROR_%'")
        )

        num_images = source_df.count()
        print(f"[EmbeddingsPipeline] {num_images} images to process.")

        worker = _SparkGPUWorker(
            chip_size=self.config.chip_size,
            batch_size=self.config.batch_size,
            mixed_precision=self.config.mixed_precision,
            model_name=self.config.model_name,
        )

        output_schema = (
            "chip_id STRING, parent_id STRING, parent_path STRING, "
            "col_off INT, row_off INT, "
            "embedding ARRAY<FLOAT>, "
            "minx DOUBLE, miny DOUBLE, maxx DOUBLE, maxy DOUBLE"
        )

        num_tasks = max(1, num_images // 4)
        result_df = source_df.repartition(num_tasks).mapInPandas(worker, schema=output_schema)

        quoted_output = quote_table_name(self.output_storage.path)
        writer = result_df.write.format("delta").mode(self.output_storage.mode)
        if self.output_storage.optimize_write:
            writer = writer.option("optimizeWrite", "true")
        writer.saveAsTable(quoted_output)

        if self.output_storage.cluster_columns:
            cols = ", ".join(self.output_storage.cluster_columns)
            spark.sql(f"ALTER TABLE {quoted_output} CLUSTER BY ({cols})")

        elapsed = time.monotonic() - start_time
        row_count = spark.read.table(quoted_output).count()
        print(
            f"[EmbeddingsPipeline] Complete: {row_count:,} embeddings "
            f"from {num_images} images in {elapsed:.1f}s"
        )


class _SparkGPUWorker:
    """
    Spark mapInPandas worker for full-image GPU inference.

    Pickled to GPU executors. Uses double-buffered I/O internally.
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

    def __call__(self, iterator: Any) -> Any:
        """Entry point for mapInPandas — processes batches of image paths."""
        import numpy as np

        from geoembed.backends.spark.worker import FullImageGPUWorker

        worker = FullImageGPUWorker(
            chip_size=self.chip_size,
            batch_size=self.batch_size,
            mixed_precision=self.mixed_precision,
            model_name=self.model_name,
        )

        for batch_df in iterator:
            if batch_df.empty:
                continue

            image_paths = batch_df["image_path"].values

            result_table = worker({"image_path": np.array(image_paths)})

            yield result_table.to_pandas()
