"""Spark-based distributed metadata extraction for GeoTIFF imagery.

Scans a Unity Catalog Volume (or any path) for GeoTIFF files, extracts spatial
and temporal metadata from headers + XML sidecars in parallel across Spark workers,
and writes results to a Delta table.

This is typically the FIRST stage of the pipeline — it creates the metadata table
that subsequent stages (COG conversion, chipping) read from.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.sql import DataFrame

from geoembed.core.naming import quote_table_name


@dataclass(frozen=True)
class MetadataExtractionConfig:
    """
    Configuration for Spark-based metadata extraction.

    Attributes:
        volume_path: Path to scan for GeoTIFFs (Unity Catalog Volume or local).
        output_table: Delta table to write metadata to.
        glob_pattern: File pattern to match (default: "*.tif").
        recursive: Scan subdirectories recursively.
    """

    volume_path: str
    output_table: str
    glob_pattern: str = "*.tif"
    recursive: bool = True


class MetadataExtractionWorker:
    """
    Spark mapInPandas worker for distributed metadata extraction.

    Initialises GDAL optimisations and metadata readers once per partition,
    then processes batches of file paths. Extracts:
    - Spatial bounds and geometry (from GeoTIFF header via rasterio)
    - Temporal extent, EPSG, and provider (from XML sidecar if present)

    The worker is serialised to executors — keep it lightweight at init time.
    """

    def __call__(self, iterator) -> Any:
        from shapely.geometry import shape

        from geoembed.metadata.readers import GeotiffSpatialReader, XmlSidecarReader

        os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
        os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif,.tiff"
        os.environ["VSI_CACHE"] = "TRUE"
        os.environ["VSI_CACHE_SIZE"] = "5000000"

        spatial_reader = GeotiffSpatialReader()
        xml_reader = XmlSidecarReader()

        for pdf in iterator:
            results = []

            for path_str in pdf["path"]:
                try:
                    if path_str.startswith("dbfs:/"):
                        path_str = path_str.replace("dbfs:", "")
                    elif path_str.startswith("file:"):
                        path_str = path_str.replace("file:", "")

                    tif_path = Path(path_str)

                    spatial = spatial_reader.read(tif_path)
                    if not spatial:
                        results.append({
                            "id": f"ERROR_{tif_path.stem}",
                            "image_path": path_str,
                            "minx": None,
                            "miny": None,
                            "maxx": None,
                            "maxy": None,
                            "datetime": None,
                            "epsg": None,
                            "provider": "Read Failed",
                            "geometry_wkb": None,
                        })
                        continue

                    sidecar = xml_reader.read(tif_path)

                    bbox = spatial.get("bbox", [0.0, 0.0, 0.0, 0.0])
                    epsg = sidecar.get("epsg") or spatial.get("epsg")

                    footprint = spatial.get("footprint")
                    geometry_wkb = shape(footprint).wkb if footprint else None

                    results.append({
                        "id": tif_path.stem,
                        "image_path": str(tif_path),
                        "minx": float(bbox[0]),
                        "miny": float(bbox[1]),
                        "maxx": float(bbox[2]),
                        "maxy": float(bbox[3]),
                        "datetime": sidecar.get("datetime"),
                        "epsg": epsg,
                        "provider": sidecar.get("provider"),
                        "geometry_wkb": geometry_wkb,
                    })

                except Exception as e:
                    results.append({
                        "id": f"ERROR_{path_str}",
                        "image_path": str(e),
                        "minx": None,
                        "miny": None,
                        "maxx": None,
                        "maxy": None,
                        "datetime": None,
                        "epsg": None,
                        "provider": "Worker Exception",
                        "geometry_wkb": None,
                    })

            yield pd.DataFrame(results)


class SparkMetadataExtractor:
    """
    Distributed metadata extraction pipeline for Databricks.

    Scans a Unity Catalog Volume for GeoTIFFs, distributes header reading
    across Spark workers, and writes a Delta metadata table.

    Usage:
        from geoembed.metadata.spark import SparkMetadataExtractor, MetadataExtractionConfig

        config = MetadataExtractionConfig(
            volume_path="/Volumes/catalog/schema/raw_imagery",
            output_table="catalog.schema.imagery_metadata",
        )
        extractor = SparkMetadataExtractor(spark, config)
        extractor.run()
    """

    OUTPUT_SCHEMA = (
        "id STRING, image_path STRING, "
        "minx DOUBLE, miny DOUBLE, maxx DOUBLE, maxy DOUBLE, "
        "datetime TIMESTAMP, epsg INT, provider STRING, geometry_wkb BINARY"
    )

    def __init__(self, spark: Any, config: MetadataExtractionConfig):
        """
        Args:
            spark: Active SparkSession.
            config: Extraction configuration.
        """
        self.spark = spark
        self.config = config

    def _add_native_geometry(self, df: DataFrame) -> DataFrame:
        """
        Add Databricks native geometry column from WKB binary.

        Uses ST_GeomFromWKB to convert the geometry_wkb binary column to
        Databricks' native GEOMETRY type with the specified SRID.

        See: https://docs.databricks.com/sql/language-manual/sql-ref-st-geospatial-functions
        """
        from pyspark.sql.functions import expr

        # ST_GeomFromWKB(wkb, srid) creates a native GEOMETRY column
        # assumes dataframe has epsg column
        srid = getattr(self.config, "srid", df)
        return df.withColumn(
            "geometry",
            expr(f"ST_GeomFromWKB(geometry_wkb, {srid})"),
        )

    def run(self) -> None:
        """
        Execute the metadata extraction pipeline.

        Scans the volume, distributes extraction, writes to Delta with native geometry.
        """
        start_time = time.monotonic()
        quoted_table = quote_table_name(self.config.output_table)

        print(f"[SparkMetadataExtractor] Scanning {self.config.volume_path}...")
        path_df = (
            self.spark.read.format("binaryFile")
            .option("pathGlobFilter", self.config.glob_pattern)
            .option("recursiveFileLookup", str(self.config.recursive).lower())
            .load(self.config.volume_path)
            .select("path")
        )

        total_files = path_df.count()
        print(f"[SparkMetadataExtractor] Found {total_files} files")

        if total_files == 0:
            print("[SparkMetadataExtractor] No files found. Exiting.")
            return

        worker = MetadataExtractionWorker()
        result_df = path_df.mapInPandas(worker, schema=self.OUTPUT_SCHEMA)

        result_df = self._add_native_geometry(result_df)

        (
            result_df.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(quoted_table)
        )

        self.spark.sql(f"ALTER TABLE {quoted_table} CLUSTER BY (id)")

        elapsed = time.monotonic() - start_time
        row_count = self.spark.read.table(quoted_table).count()
        error_count = self.spark.read.table(quoted_table).filter("id LIKE 'ERROR_%'").count()

        print(
            f"[SparkMetadataExtractor] Complete: {row_count} rows "
            f"({error_count} errors) in {elapsed:.1f}s"
        )
        print(f"[SparkMetadataExtractor] Output: {self.config.output_table}")
