"""Create a chip metadata table with spatial geometry from an embeddings table.

The embeddings pipeline already computes chip_id, parent_id, parent_path,
col_off, row_off, minx, miny, maxx, maxy for every chip. This module creates
a proper spatial table (with geometry_wkb and native Databricks geometry columns)
from that information — no need to run the chipping pipeline separately.
"""

from __future__ import annotations

from typing import Any

from geoembed.core.naming import quote_table_name


def create_chip_table_from_embeddings(
    spark: Any,
    embeddings_table: str,
    output_table: str,
    metadata_table: str | None = None,
    srid: int | None = None,
) -> None:
    """
    Create a chip metadata table with spatial geometry from an embeddings table.

    The embeddings table (produced by EmbeddingsPipeline) already contains spatial
    bounds per chip. This function creates a proper geometry table for spatial queries.

    Args:
        spark: Active SparkSession.
        embeddings_table: Source embeddings table (three-part UC name).
        output_table: Destination chip metadata table (three-part UC name).
        metadata_table: Source imagery metadata table (used to derive SRID if not provided).
        srid: Spatial Reference ID. If None, derived from metadata_table's epsg column.
              Defaults to 27700 (BNG) if neither is provided.
    """

    quoted_input = quote_table_name(embeddings_table)
    quoted_output = quote_table_name(output_table)

    if srid is None and metadata_table is not None:
        quoted_meta = quote_table_name(metadata_table)
        row = (
            spark.read.table(quoted_meta)
            .filter("epsg IS NOT NULL")
            .groupBy("epsg")
            .count()
            .orderBy("count", ascending=False)
            .first()
        )
        srid = int(row["epsg"]) if row else 27700
    elif srid is None:
        srid = 27700

    print(f"[create_chip_table_from_embeddings] SRID: {srid}")
    print(f"[create_chip_table_from_embeddings] Source: {embeddings_table}")

    from pyspark.databricks.sql import functions as dbf
    from pyspark.sql.functions import col

    embeddings_df = spark.read.table(quoted_input)

    envelope = dbf.st_makeenvelope(col("minx"), col("miny"), col("maxx"), col("maxy"))

    chip_metadata = (
        embeddings_df.select(
            "chip_id",
            "parent_id",
            "parent_path",
            "col_off",
            "row_off",
            "minx",
            "miny",
            "maxx",
            "maxy",
        )
        .withColumn("geometry_wkb", dbf.st_asbinary(envelope))
        .withColumn("geometry", dbf.st_setsrid(envelope, srid))
    )

    (
        chip_metadata.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(quoted_output)
    )
    spark.sql(f"ALTER TABLE {quoted_output} CLUSTER BY (chip_id)")

    row_count = spark.read.table(quoted_output).count()
    print(
        f"[create_chip_table_from_embeddings] "
        f"Created {row_count:,} chip geometries -> {output_table}"
    )
