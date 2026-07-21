# End-to-End Workflow: Raw Imagery to Embeddings on Databricks

This guide walks through the complete `geoembed` pipeline running on Azure Databricks, from raw GeoTIFFs in Unity Catalog Volumes through to 768-dimensional embeddings in Delta tables.

---

## Pipeline Overview

```
[Stage 0a]          [Stage 0b]          [Stage 0c]        [Stage 1]
Metadata Extract    Raw GeoTIFF → COG   STAC Catalog      Embed (GPU full-image)
(Spark, CPU)        (Spark, CPU)        (Driver)          (Spark, GPU)
```

| Stage | Purpose | Cluster Type |
|-------|---------|--------------|
| 0a | Metadata extraction | CPU workers (Standard_D3_v2) |
| 0b | COG conversion | CPU workers (Standard_D3_v2) |
| 0c | STAC catalog (optional) | Driver only |
| 1 | Embeddings (full-image GPU) | GPU node (NC4as_T4 or NV36ads_A10v5) |

> **Note:** the image chip vector boundaries can be generated during GPU inference or as a
> standalone function call via `SparkChippingPipeline`. If you need the image chip boundaries for
> downstream tasks (e.g. vector operations) then you can run the chipping pipeline.

---

## Prerequisites

### Package installation

Via init script (recommended — applied on cluster start):

```bash
# deps/init-scripts/init.sh installs `uv` and geoembed[spatial,cog,stac]
```

Or per-notebook:

```python
%pip install "/Workspace/path/to/geoembed-0.1.0.dev0-py3-none-any.whl[spatial,cog,stac]"
dbutils.library.restartPython()
```

For inference on GPU clusters, also install `torchgeo` (torch is pre-installed by DBR):

```python
%pip install torchgeo>=0.7.1
dbutils.library.restartPython()
```

### Unity Catalog paths

```python
CATALOG = "your_catalog"
SCHEMA = "your_schema"
VOLUME = "your_volume"
RAW_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/your_raw_geotiffs_folder"
COG_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/your_COG_folder"
METADATA_TABLE = f"{CATALOG}.{SCHEMA}.geoembed_metadata"
CHIP_TABLE = f"{CATALOG}.{SCHEMA}.geoembed_chip_metadata"
EMBEDDINGS_TABLE = f"{CATALOG}.{SCHEMA}.geoembed_embeddings"
```

---

## Stage 0a: Metadata Extraction

Scans a Unity Catalog Volume for GeoTIFF files and extracts spatial/temporal
metadata from headers and XML sidecars. Creates the metadata table that all
subsequent stages read from. Only reads file headers.

### Cluster: CPU (Standard_D3_v2)

```python
from geoembed.metadata.spark import SparkMetadataExtractor, MetadataExtractionConfig

config = MetadataExtractionConfig(
    volume_path=RAW_VOLUME,
    output_table=METADATA_TABLE,
    glob_pattern="*.tif",
    recursive=True,
)

SparkMetadataExtractor(spark, config).run()
```

### Output

Delta table with columns: `id`, `image_path`, `minx`, `miny`, `maxx`, `maxy`,
`datetime`, `epsg`, `provider`, `geometry_wkb`, `geometry` (native Databricks spatial type).

---

## Stage 0b: COG Conversion

**Skip if imagery is already COG format.**

Converts raw GeoTIFFs to Cloud Optimised GeoTIFFs with 512x512 internal tiling.

### Cluster: CPU (Standard_D3_v2)

```python
from geoembed.images.cog import SparkCogPipeline, CogConfig

config = CogConfig(
    input_table=METADATA_TABLE,
    output_table=f"{CATALOG}.data.geoembed_cog_log",
    dst_root=COG_VOLUME,
    compression="deflate",
    files_per_task=20,
)

SparkCogPipeline(spark, config).run()
```

After conversion, update metadata paths to point at the new COGs:

```python
from geoembed.core.naming import quote_table_name

spark.sql(f"""
    UPDATE {quote_table_name(METADATA_TABLE)}
    SET image_path = concat('{COG_VOLUME}/', id, '.tif')
    WHERE id NOT LIKE 'ERROR_%'
""")
```

---

## Stage 0c: STAC Catalog (Optional)

Generates a STAC Catalog + GeoParquet 1.1 index for spatial discovery.

```python
from geoembed.metadata.stac_catalog import (
    StacCatalogConfig,
    build_stac_items_from_metadata,
    generate_stac_catalog,
)

metadata_rows = (
    spark.read.table(quote_table_name(METADATA_TABLE))
    .filter("id NOT LIKE 'ERROR_%'")
    .toPandas()
    .to_dict("records")
)
item_dicts = build_stac_items_from_metadata(metadata_rows, cog_base_path=COG_VOLUME)

STAC_OUTPUT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/your_stac_catalog"
stac_config = StacCatalogConfig(
    catalog_id="some-id-value",
    catalog_title="This is your STAC Catalog",
    catalog_description="STAC Catalog for your imagery data.",
    collection_id="some-collection-id",
    collection_title="Title for Some Collection ID",
    collection_description="Description for Some Collection ID.",
    output_path=STAC_OUTPUT,
)
generate_stac_catalog(item_dicts, stac_config, lon_range=(-8.0, 2.0), lat_range=(49.0, 61.0))
```

---

## Chip Metadata (Spatial Vector Boundaries)

You can produce chip vector boundaries (with Databricks native geometry) in two ways:

### Option A: Before embeddings (standalone chipping)

Use this if you need chip boundaries for separate spatial processing **before**
running the embeddings pipeline. This runs on a CPU cluster.

```python
from geoembed.chipping.spark import SparkChippingPipeline, SparkChippingConfig

config = SparkChippingConfig(
    input_table=METADATA_TABLE,
    output_table=CHIP_TABLE,
    chip_size=224,
    images_per_task=10,
)

SparkChippingPipeline(spark, config).run()
```

### Option B: After embeddings (derived from embeddings table)

The embeddings pipeline already computes all spatial bounds per chip. Use this
to create a geometry table from the embeddings output — no separate chipping
stage needed.

```python
from geoembed.chipping.from_embeddings import create_chip_table_from_embeddings

create_chip_table_from_embeddings(
    spark=spark,
    embeddings_table=EMBEDDINGS_TABLE,
    output_table=CHIP_TABLE,
    metadata_table=METADATA_TABLE,  # Used to derive SRID (e.g., 27700 for British National Grid)
)
```

Both options produce the same schema:

| Column | Type | Description |
|--------|------|-------------|
| chip_id | string | e.g., "NT6200_000042" |
| parent_id | string | e.g., "NT6200" |
| parent_path | string | Full COG path |
| col_off | int | Pixel column offset in parent used to read chip |
| row_off | int | Pixel row offset in parent used to read chip |
| minx/miny/maxx/maxy | double | Bounding box coordinates |
| geometry_wkb | binary | WKB geometry (for non-Spark consumers) |
| geometry | geometry | Databricks native spatial type (with SRID) |

### Joining embeddings onto chip geometries

If you created chip boundaries separately (Option A) and want to add embeddings:

```python
from geoembed.core.naming import quote_table_name

chips_df = spark.read.table(quote_table_name(CHIP_TABLE)
embeddings_df = spark.read.table(quote_table_name(EMBEDDINGS_TABLE))

chips_with_embeddings = chips_df.join(
    embeddings_df.select("chip_id", "embedding"),
    on="chip_id",
    how="inner",
)

chips_with_embeddings.write.format("delta").mode("overwrite").saveAsTable(
    quote_table_name(f"{CATALOG}.{SCHEMA}.geoembed_chips_with_embeddings")
)
```

---

## Stage 1: Embedding Inference (Full-Image GPU Pipeline)

Reads **entire COGs** onto the GPU, tiles via `torch.unfold()`, runs batched
inference and `SparkChipperStrategy`, and computes spatial bounds — all in one pass per image.

Distributed across GPU workers via Spark `mapInPandas` with `spark.task.resource.gpu.amount = 1`. You
can also used [fractional values](https://docs.databricks.com/aws/en/compute/gpu#gpu-scheduling-for-ai-and-ml)
to specify the amount of GPU tasks, example cluster configs are below.

---

### How it works

```
For each COG image (per GPU worker):
1. Background thread reads next image to CPU
2. Transfer current image to GPU as float32 tensor
3. torch.unfold() tiles into 224x224 patches
4. Batched DOFA inference on all patches
5. Compute spatial bounds from affine transform
6. Yield results → Spark writes to Delta
```

---

### GPU Cluster Options

| Instance | GPU | VRAM | Node RAM | CPUs | GPUs/node | Best for |
|----------|-----|------|----------|------|-----------|----------|
| Standard_NV36ads_A10v5 | A10 | 24 GB | 440 GB | 36 | 1 | Production (high throughput) |
| Standard_NC4as_T4_v3 | T4 | 16 GB | 28 GB | 4 | 1 | Development (budget) |

---

### Databricks Cluster Configuration

**A10 cluster:**

| Setting | Value | Rationale |
|---------|-------|-----------|
| Driver | Standard_D3_v2 | Lightweight orchestration (no GPU needed) |
| Workers | 2x Standard_NV36ads_A10v5 | 2 GPUs for parallel processing |
| Runtime | DBR ML 17.3 LTS | Stable PyTorch + CUDA |
| `spark.task.resource.gpu.amount` | 1 | 1 full GPU per Spark task |
| `spark.executor.resource.gpu.amount` | 1 | Executor claims the node's GPU |
| `spark.task.cpus` | 4 | CPUs for I/O thread + data loading |
| `spark.executor.memory` | 200g | Holds full images in CPU memory |

**T4 cluster:**

| Setting | Value | Rationale |
|---------|-------|-----------|
| Driver | Standard_D4_v2 | Lightweight orchestration |
| Workers | 2x Standard_NC4as_T4_v3 | 2 GPUs for parallel processing |
| Runtime | DBR ML 17.3 LTS | Stable PyTorch + CUDA |
| `spark.task.resource.gpu.amount` | 1 | 1 full GPU per Spark task |
| `spark.executor.resource.gpu.amount` | 1 | Executor claims the node's GPU |
| `spark.task.cpus` | 4 | All CPUs for the task |
| `spark.executor.memory` | 16g | Limited RAM per node |

---

### Code

**Set Spark GPU resource configs** (do this before starting your Databricks cluster):

```text
# Required: tell Spark to allocate GPUs to tasks
spark.task.resource.gpu.amount 1
spark.task.cpus <check-cluster-for-CPU-allocation>
```

```python
# --- GPU cluster notebook (DBR ML 17.3 LTS) ---
%pip install torchgeo>=0.7.1 rasterio
dbutils.library.restartPython()
```

**Run the pipeline:**

```python
from geoembed.backends.spark.config import EmbeddingsConfig
from geoembed.backends.spark.orchestrator import EmbeddingsPipeline
from geoembed.core.config import StorageConfig

CATALOG = "catalog-sbx-uks-aaai-brownfieldai-001"
METADATA_TABLE = f"{CATALOG}.data.geoembed_metadata"
EMBEDDINGS_TABLE = f"{CATALOG}.data.geoembed_embeddings"

# A10 config
config = EmbeddingsConfig(chip_size=224, batch_size=512, mixed_precision=True)

# T4 config — uncomment if using T4
# config = EmbeddingsConfig(chip_size=224, batch_size=128, mixed_precision=True)

pipeline = EmbeddingsPipeline(
    config=config,
    input_storage=StorageConfig(backend="delta", path=METADATA_TABLE),
    output_storage=StorageConfig(
        backend="delta",
        path=EMBEDDINGS_TABLE,
        cluster_columns=["chip_id"],
    ),
)
pipeline.run()
```

Input is the **metadata table** (from Stage 0a), just needs `image_path` column.
No chip metadata table required.

---

### Output

| Column | Type | Description |
|--------|------|-------------|
| chip_id | string | e.g., "NT6200_000042" |
| parent_id | string | e.g., "NT6200" |
| parent_path | string | Full COG path |
| embedding | array\<float32\> | 768-dim DOFA embedding |
| minx | float64 | Chip bounding box (native CRS) |
| miny | float64 | |
| maxx | float64 | |
| maxy | float64 | |

---

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│             Delta Table: metadata (image_path column)           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ spark.read.table() → repartition
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Spark mapInPandas → FullImageGPUWorker             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Background thread: read next COG to CPU (rasterio)      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  For each image:                                                │
│  1. CPU → GPU transfer (float32, non_blocking)                  │
│  2. torch.unfold(dim=1, 224, stride).unfold(dim=2, 224, stride) │
│  3. Batched DofaEmbedder.embed() with inference_mode + FP16     │
│  4. rasterio.windows.bounds() for spatial metadata              │
│  5. Yield pandas DataFrame back to Spark                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Spark writes directly to Delta
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Delta Table: embeddings                                         │
│ chip_id | parent_id | embedding | minx | miny | maxx | maxy     │
│ Liquid clustered by chip_id                                     │
└─────────────────────────────────────────────────────────────────┘
```

Distribution is handled using Spark `mapInPandas`, but ideally we would use
[Ray](https://www.ray.io/)for running distributed geospatial inference - good examples include
[Wherobots](https://wherobots.com/blog/fields-of-the-world-rasterflow-geoai-pipeline/) and
[Xoople](https://www.anyscale.com/blog/ray-day-london-recap#how-xoople-runs-geospatial-foundation-model-inference-at-global-scale).

Separately, the process uses the CPU to read each image and pass it to the GPU. In the background, it
will fetch the next `X` images to try keep the GPU saturated. A further improvement would be reading
images directly to the GPU (there's an interesting
[GPU-native Xarray blog](https://xarray.dev/blog/gpu-pipeline?)); for COGs, there's promising libraries
like [`nvTIFF`](https://docs.nvidia.com/cuda/nvtiff/index.html) and Python bindings
[`cog3pio`](https://github.com/weiji14/cog3pio) which might enable this.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| CUDA OOM | `batch_size` too large or image too big | Reduce batch_size: 128 for T4, 256-512 for A10 |
| COG conversion hangs on last tasks | GDAL thread contention + FUSE writes | Use `files_per_task=20+` |
| `No module named 'torchgeo'` | Not installed on GPU cluster | `%pip install torchgeo>=0.7.1` |
| `sympy` / `torch` conflict | Init script installed torch over DBR's version | Don't include `[models]` in init script — use DBR's torch |
| `FileNotFoundError` on image reads | Metadata paths not updated after COG conversion | Run the UPDATE SQL in Stage 0b |
| Spark tasks stuck after cancellation | Orphaned worker processes | Restart the cluster |
| `INVALID_IDENTIFIER` on table names | Hyphenated catalog without backticks | `geoembed` handles this via `quote_table_name` |
| pandas/numpy version errors | DBR 18.x version conflicts | Use DBR ML 17.3 LTS |

---
