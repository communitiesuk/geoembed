"""Spark-based distributed chipping pipeline for Databricks.

Reads a metadata table, distributes chip geometry calculation across workers,
writes results to a Delta table with Databricks native geometry for spatial SQL.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from geoembed.core.naming import quote_table_name


@dataclass(frozen=True)
class SparkChippingConfig:
    """
    Configuration for the Spark chipping pipeline.

    Attributes:
        input_table: Source metadata table (must have `image_path` and `epsg` columns).
        output_table: Delta table to write chip metadata to.
        chip_size: Square chip dimension in pixels (default 224 for DOFA).
        images_per_task: Approximate images per Spark task (controls partition count).
    """

    input_table: str
    output_table: str
    chip_size: int = 224
    images_per_task: int = 10


class _ChipMetadataWorker:
    """
    Spark mapInPandas worker for distributed chip metadata generation.

    Initialises the chipping strategy once per partition, then generates virtual
    chip metadata (offsets, geometries) for each image in the batch.
    """

    def __init__(self, chip_size: int = 224):
        self.chip_size = chip_size

    def __call__(self, iterator) -> Any:
        from geoembed.chipping.strategies.sliding_window import SlidingWindowStrategy

        strategy = SlidingWindowStrategy(chip_size=self.chip_size)

        for batch_df in iterator:
            if batch_df.empty:
                continue

            results = []
            for _, row in batch_df.iterrows():
                path = str(row["image_path"])
                try:
                    results.extend(strategy.generate_chips(path))
                except (FileNotFoundError, OSError) as e:
                    print(f"SKIP: {e}")
                    continue

            if results:
                yield pd.DataFrame(results)


OUTPUT_SCHEMA = (
    "chip_id STRING, parent_id STRING, parent_path STRING, "
    "col_off INT, row_off INT, width INT, height INT, "
    "geometry_wkb BINARY, minx FLOAT, miny FLOAT, maxx FLOAT, maxy FLOAT"
)


class SparkChippingPipeline:
    """
    Distributed chipping pipeline for Databricks.

    Reads a metadata table, distributes chip geometry calculation across Spark
    workers using mapInPandas, adds a Databricks native geometry column, and
    writes results to a Delta table with liquid clustering on chip_id.

    Usage:
        from geoembed.chipping.spark import SparkChippingPipeline, SparkChippingConfig

        config = SparkChippingConfig(
            input_table="catalog.schema.metadata",
            output_table="catalog.schema.chip_metadata",
            chip_size=224,
        )
        pipeline = SparkChippingPipeline(spark, config)
        pipeline.run()
    """

    def __init__(self, spark: Any, config: SparkChippingConfig):
        """
        Args:
            spark: Active SparkSession.
            config: Chipping configuration.
        """
        self.spark = spark
        self.config = config

    def _resolve_srid(self, source_df: Any) -> int:
        """
        Derive the SRID from the source metadata table's `epsg` column.

        Takes the most common (mode) EPSG value from the table. Falls back to
        27700 (BNG) if the column is missing or all null.
        """
        if "epsg" not in source_df.columns:
            print("[SparkChippingPipeline] WARNING: No 'epsg' column found, defaulting to 27700")
            return 27700

        from pyspark.sql.functions import col

        row = (
            source_df.filter(col("epsg").isNotNull())
            .groupBy("epsg")
            .count()
            .orderBy("count", ascending=False)
            .first()
        )

        if row is None:
            print("[SparkChippingPipeline] WARNING: All EPSG values are null, defaulting to 27700")
            return 27700

        srid = int(row["epsg"])
        print(f"[SparkChippingPipeline] Derived SRID from metadata: EPSG:{srid}")
        return srid

    def run(self) -> None:
        """
        Execute the chipping pipeline.

        Reads source metadata, distributes chipping, adds native geometry
        (SRID derived from the metadata table's epsg column), writes to Delta,
        and applies liquid clustering.
        """
        start_time = time.monotonic()
        quoted_input = quote_table_name(self.config.input_table)
        quoted_output = quote_table_name(self.config.output_table)

        source_df = self.spark.read.table(quoted_input)
        num_images = source_df.count()
        num_partitions = max(1, math.ceil(num_images / self.config.images_per_task))

        srid = self._resolve_srid(source_df)

        print(f"[SparkChippingPipeline] {num_images} images -> {num_partitions} partitions")
        print(f"[SparkChippingPipeline] Chip size: {self.config.chip_size}px")

        worker = _ChipMetadataWorker(chip_size=self.config.chip_size)
        result_df = source_df.repartition(num_partitions, "image_path").mapInPandas(
            worker, schema=OUTPUT_SCHEMA
        )

        from pyspark.sql.functions import expr

        result_df = result_df.withColumn(
            "geometry",
            expr(f"ST_GeomFromWKB(geometry_wkb, {srid})"),
        )

        (
            result_df.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(quoted_output)
        )

        self.spark.sql(f"ALTER TABLE {quoted_output} CLUSTER BY (chip_id)")

        elapsed = time.monotonic() - start_time
        row_count = self.spark.read.table(quoted_output).count()
        print(f"[SparkChippingPipeline] Complete: {row_count:,} chips in {elapsed:.1f}s")
        print(f"[SparkChippingPipeline] Output: {self.config.output_table}")
