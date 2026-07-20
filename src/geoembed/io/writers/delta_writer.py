"""Delta Lake embedding writer for Databricks / Unity Catalog."""

import numpy as np
import pandas as pd

from geoembed.io.writers.base import EmbeddingWriter


class DeltaWriter(EmbeddingWriter):
    """
    Writes embeddings to a Delta Lake table via Spark.

    Designed for production use on Databricks with Unity Catalog governance.
    Supports append mode and optional liquid clustering.

    Requires: geoembed[spark] extras.
    """

    def __init__(
        self,
        table_name: str,
        mode: str = "append",
        optimize_write: bool = True,
        cluster_columns: list[str] | None = None,
    ):
        """
        Initialise the Delta writer.

        Args:
            table_name: Full Unity Catalog table name (e.g., "catalog.schema.table").
            mode: Spark write mode ("append" or "overwrite").
            optimize_write: Enable Delta's optimizeWrite for auto-compaction.
            cluster_columns: Columns for liquid clustering (e.g., ["chip_id"]).
                Liquid clustering is incremental and automatic — no manual
                OPTIMIZE runs needed after initial setup.
        """
        self.table_name = table_name
        self.mode = mode
        self.optimize_write = optimize_write
        self.cluster_columns = cluster_columns
        self._spark = None

    def _get_spark(self):
        """Lazily acquire the active SparkSession."""
        if self._spark is None:
            from pyspark.sql import SparkSession

            self._spark = SparkSession.builder.getOrCreate()
        return self._spark

    def write_batch(self, chip_ids: list[str], embeddings: np.ndarray) -> None:
        """Write a batch of embeddings directly to Delta."""
        from geoembed.core.naming import quote_table_name

        spark = self._get_spark()

        df = pd.DataFrame({
            "chip_id": chip_ids,
            "embedding": list(embeddings.astype(float)),
        })

        sdf = spark.createDataFrame(df)

        writer = sdf.write.format("delta").mode(self.mode)
        if self.optimize_write:
            writer = writer.option("optimizeWrite", "true")
        writer.saveAsTable(quote_table_name(self.table_name))

    def finalize(self) -> None:
        """Apply liquid clustering if configured."""
        if self.cluster_columns:
            from geoembed.core.naming import quote_table_name

            spark = self._get_spark()
            quoted = quote_table_name(self.table_name)
            cols = ", ".join(self.cluster_columns)
            spark.sql(f"ALTER TABLE {quoted} CLUSTER BY ({cols})")
            print(f"Liquid clustering enabled on {self.table_name} by ({cols})")
