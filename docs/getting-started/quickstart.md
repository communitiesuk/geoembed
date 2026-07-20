# Quick Start

## On Databricks (GPU Cluster)

Set Spark GPU & CPU config prior to starting cluster:

```text
spark.task.resource.gpu.amount 1
spark.task.cpus <check-cluster-for-CPU-allocation>
```

```python
%pip install torchgeo>=0.7.1 rasterio
dbutils.library.restartPython()

# Set Spark GPU configs
spark.conf.set("spark.task.resource.gpu.amount", "1")
spark.conf.set("spark.task.cpus", "4")

from geoembed.backends.spark.config import EmbeddingsConfig
from geoembed.backends.spark.orchestrator import EmbeddingsPipeline
from geoembed.core.config import StorageConfig

CATALOG = "your-catalog.schema"
METADATA_TABLE = f"{CATALOG}.imagery_metadata"
EMBEDDINGS_TABLE = f"{CATALOG}.embeddings"

config = EmbeddingsConfig(chip_size=224, batch_size=256, mixed_precision=True)

pipeline = EmbeddingsPipeline(
    config=config,
    input_storage=StorageConfig(backend="delta", path=METADATA_TABLE),
    output_storage=StorageConfig(backend="delta", path=EMBEDDINGS_TABLE, cluster_columns=["chip_id"]),
)
pipeline.run()
```

## Local (Single Image)

```python
from examples.gpu_full_image import process_single_cog

df = process_single_cog(
    cog_path="/path/to/image.tif",
    output_path="embeddings.parquet",
    batch_size=128,
)
```

## Full Pipeline

See the [Workflow Guide](.././workflow.md) for the complete end-to-end pipeline
including metadata extraction, COG conversion, and STAC catalog generation.
