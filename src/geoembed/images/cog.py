"""Cloud Optimised GeoTIFF (COG) conversion pipeline.

Converts standard GeoTIFFs to COGs with internal tiling, compression, and
overviews for efficient cloud-based partial reads (HTTP range requests).

Includes a Spark-based pipeline for distributed conversion on Databricks.
"""

from __future__ import annotations

import os
import shutil
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from geoembed.core.naming import quote_table_name


class FileSystemAdapter(ABC):
    """Abstract filesystem operations for testability."""

    @abstractmethod
    def makedirs(self, path: str, exist_ok: bool = True) -> None: ...

    @abstractmethod
    def copy(self, src: str, dst: str) -> None: ...

    @abstractmethod
    def remove(self, path: str) -> None: ...

    @abstractmethod
    def rmtree(self, path: str) -> None: ...

    @abstractmethod
    def exists(self, path: str) -> bool: ...


class LocalFileSystem(FileSystemAdapter):
    """Concrete filesystem adapter for local/mounted paths."""

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        os.makedirs(path, exist_ok=exist_ok)

    def copy(self, src: str, dst: str) -> None:
        shutil.copy2(src, dst)

    def remove(self, path: str) -> None:
        os.remove(path)

    def rmtree(self, path: str) -> None:
        shutil.rmtree(path, ignore_errors=True)

    def exists(self, path: str) -> bool:
        return os.path.exists(path)


class ImageConverter(ABC):
    """Abstract interface for image format conversion."""

    @abstractmethod
    def convert(self, src: str, dst: str) -> None: ...

    @abstractmethod
    def validate(self, path: str) -> bool: ...


class RioCogConverter(ImageConverter):
    """
    Converts GeoTIFFs to COGs optimised for high-throughput GPU inference.

    Based on Microsoft's COG throughput findings (Zaytar et al., 2025):

    Compression choice depends on storage type:
    - LOCAL STORAGE: "none" (uncompressed) is 1.3-1.6x faster because CPU
      decompression is the bottleneck when reading from fast local disk/SSD.
    - REMOTE/CLOUD STORAGE: "lerc_zstd" is optimal — balances network transfer
      reduction (good compression ratio) with fast CPU decompression.
    - "deflate" is a safe middle ground for mixed local/remote workflows.

    Tiling:
    - 512x512 internal tiles enable tile-aligned reads. A 224px chip falls
      entirely within one tile, requiring exactly one decompression.
    - Misaligned reads force decompression of up to 4 adjacent tiles.

    Overviews:
    - Embedded reduced-resolution pyramids for fast wide-area previews.
    """

    def __init__(
        self,
        compression: str = "deflate",
        blocksize: int = 512,
        overview_resampling: str = "nearest",
    ):
        """
        Args:
            compression: Compression codec. Recommendations:
                - "none": Fastest local reads (no decompression overhead). Largest files.
                - "deflate": Good default. Moderate compression + decode speed.
                - "lerc_zstd": Best for remote/cloud (high compression, fast decode).
                - "lzw": Legacy option, similar to deflate.
            blocksize: Internal tile size. Must be 256 or 512 for tile-aligned reads.
                512 recommended — a 224px chip fits within one tile.
            overview_resampling: Resampling for overviews. "nearest" preserves
                spectral accuracy; "average" for smoother visual previews.
        """
        self.compression = compression
        self.blocksize = blocksize
        self.overview_resampling = overview_resampling

    def convert(self, src: str, dst: str) -> None:
        """
        Convert a GeoTIFF to a GPU-cloud-friendly COG.

        Generates a COG with internal 512x512 tiling, compression, and overviews.
        """
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles

        output_profile = cog_profiles.get(self.compression)
        output_profile.update({
            "blockxsize": self.blocksize,
            "blockysize": self.blocksize,
            "predictor": 2,
        })

        config = {
            # Use 2 threads per file — avoids thread starvation when multiple
            # Spark tasks share the same executor node.
            "GDAL_NUM_THREADS": "2",
            "GDAL_TIFF_INTERNAL_MASK": True,
            "GDAL_TIFF_OVR_BLOCKSIZE": self.blocksize,
        }

        cog_translate(
            src,
            dst,
            output_profile,
            config=config,
            overview_resampling=self.overview_resampling,  # pyrefly: ignore
            use_cog_driver=True,
        )

    def validate(self, path: str) -> bool:
        """Validate that a file is a valid COG."""
        from rio_cogeo.cogeo import cog_validate

        is_valid, _, _ = cog_validate(path)
        return is_valid


@dataclass(frozen=True)
class CogConfig:
    """
    Configuration for the COG conversion pipeline.

    Attributes:
        input_table: Source table with image metadata (path column required).
        output_table: Table to log conversion results.
        dst_root: Destination directory for COG files.
        compression: Compression codec.
        blocksize: Internal tile size.
        files_per_task: Files to process per Spark task.
    """

    input_table: str
    output_table: str
    dst_root: str
    compression: str = "deflate"
    blocksize: int = 512
    files_per_task: int = 10


class CogTaskHandler:
    """
    Handles COG conversion for a batch of files.

    Writes to a unique local temp directory (NVMe /local_disk0 on Databricks),
    then copies the validated COG to the UC Volume destination.
    """

    def __init__(
        self,
        converter: ImageConverter | None = None,
        fs: FileSystemAdapter | None = None,
        dst_root: str = "/tmp/cog_output",  # noqa: S108
    ):
        self.converter = converter or RioCogConverter()
        self.fs = fs or LocalFileSystem()
        self.dst_root = dst_root

    def process_file(self, src_path: str) -> dict[str, Any]:
        """
        Convert a single file to COG.

        Uses a unique temp directory per file to avoid collisions when multiple
        Spark tasks run concurrently on the same executor.

        Args:
            src_path: Path to the source GeoTIFF.

        Returns:
            Dict with keys: src, status, error.
        """
        filename = os.path.basename(src_path)
        # Use /local_disk0 on Databricks (fast NVMe) with unique dir per file
        temp_dir = f"/local_disk0/tmp/cog_{uuid.uuid4().hex[:8]}"
        temp_path = os.path.join(temp_dir, filename)
        final_path = os.path.join(self.dst_root, filename)

        try:
            os.makedirs(temp_dir, exist_ok=True)
            self.fs.makedirs(self.dst_root)

            # Convert to local temp (fast, NVMe)
            self.converter.convert(src_path, temp_path)

            # Validate locally (fast)
            if not self.converter.validate(temp_path):
                return {"src": src_path, "status": "invalid", "error": "COG validation failed"}

            # Copy validated COG to UC Volume (FUSE upload)
            self.fs.copy(temp_path, final_path)
        except Exception as e:
            return {"src": src_path, "status": "error", "error": str(e)}
        else:
            return {"src": src_path, "status": "success", "error": None}
        finally:
            # Always clean up local temp
            shutil.rmtree(temp_dir, ignore_errors=True)


class SparkCogWorker:
    """
    Spark mapInPandas worker for distributed COG conversion.

    Pickled to Spark executors. Each worker converts a batch of GeoTIFFs to COGs.
    """

    def __init__(self, config: CogConfig):
        self.config = config

    def __call__(self, iterator: Any) -> Any:
        import pandas as pd

        converter = RioCogConverter(
            compression=self.config.compression,
            blocksize=self.config.blocksize,
        )
        handler = CogTaskHandler(
            converter=converter,
            dst_root=self.config.dst_root,
        )

        for batch_df in iterator:
            if batch_df.empty:
                continue

            results = []
            paths = batch_df["path"].tolist()
            for src_path in paths:
                # Strip Spark's file: or dbfs: prefix if present
                clean_path = src_path.replace("dbfs:", "").replace("file:", "")
                result = handler.process_file(clean_path)
                results.append(result)

            if results:
                yield pd.DataFrame(results)


class SparkCogPipeline:
    """
    Distributed COG conversion pipeline for Databricks.

    Scans a volume for GeoTIFFs, distributes conversion across Spark workers
    using mapInPandas, and writes COGs to a destination volume.

    Usage:
        from geoembed.images.cog import SparkCogPipeline, CogConfig

        config = CogConfig(
            input_table="catalog.schema.imagery_metadata",
            output_table="catalog.schema.cog_log",
            dst_root="/Volumes/catalog/data/processed/COG",
            compression="deflate",
        )
        pipeline = SparkCogPipeline(spark, config)
        pipeline.run()
    """

    def __init__(self, spark: Any, config: CogConfig):
        """
        Args:
            spark: Active SparkSession.
            config: COG conversion configuration.
        """
        self.spark = spark
        self.config = config

    @property
    def _output_schema(self) -> str:
        return "src STRING, status STRING, error STRING"

    def run(self) -> None:
        """
        Run COG conversion from an existing metadata table.

        Reads image_path column from config.input_table.
        """
        from pyspark.sql.functions import col

        quoted_input = quote_table_name(self.config.input_table)
        print(f"[SparkCogPipeline] Reading source paths from {self.config.input_table}")
        df = self.spark.read.table(quoted_input).select(col("image_path").alias("path"))
        self._process(df)

    def run_from_volume(self, volume_path: str, glob_pattern: str = "*.tif") -> None:
        """
        Run COG conversion by scanning a Unity Catalog Volume for GeoTIFFs.

        Args:
            volume_path: Path to the UC Volume.
            glob_pattern: File glob pattern (default: "*.tif").
        """
        print(f"[SparkCogPipeline] Scanning {volume_path} for {glob_pattern}")
        df = (
            self.spark.read.format("binaryFile")
            .option("pathGlobFilter", glob_pattern)
            .option("recursiveFileLookup", "true")
            .load(volume_path)
            .select("path")
        )
        self._process(df)

    def _process(self, df: Any) -> None:
        """Core processing logic shared by run() and run_from_volume()."""
        import math

        total_files = df.count()
        num_partitions = max(1, math.ceil(total_files / self.config.files_per_task))
        quoted_output = quote_table_name(self.config.output_table)

        print(f"[SparkCogPipeline] {total_files} files -> {num_partitions} partitions")
        print(f"[SparkCogPipeline] Destination: {self.config.dst_root}")
        print(f"[SparkCogPipeline] Compression: {self.config.compression}")

        # Distribute evenly across workers
        df_repartitioned = df.repartition(num_partitions)

        worker = SparkCogWorker(self.config)
        result_df = df_repartitioned.mapInPandas(worker, schema=self._output_schema)

        # Write conversion log (quoted for hyphenated catalog names)
        (
            result_df.write.format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(quoted_output)
        )

        # Report summary
        log_df = self.spark.read.table(quoted_output)
        success_count = log_df.filter("status = 'success'").count()
        error_count = log_df.filter("status = 'error'").count()
        print(f"[SparkCogPipeline] Complete: {success_count} succeeded, {error_count} failed")
