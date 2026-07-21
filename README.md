# geoembed

Generate vector embeddings from Cloud Optimised GeoTIFFs (COGs) using Databricks.

## Overview

`geoembed` is a Python library for generating embeddings from COGs using Geospatial Foundation
Models, such as [DOFA](https://github.com/zhu-xlab/DOFA).

The embeddings pipeline reads entire images onto the GPU, tiles them via `torch.unfold()` using a
chip size and stride calculation, and runs batched inference.

![Pipeline flowchart](docs/assets/pipeline_flowchart.png)

It runs on Databricks with GPU-accelerated inference via Spark.

It was developed as part of the Ministry of Housing, Communities and Local Government's (MHCLG) AAAI
lab to predict the potential of brownfield land - you can
[read more](https://mhclgdigital.blog.gov.uk/2026/07/16/from-pixels-to-policy-the-potential-of-geospatial-embeddings-for-mhclg/)
about the project.

The embeddings pipeline has been open-sourced in case it is useful for other Databricks' users. Note:
it had specific use case using aerial imagery within great Britain (EPSG:27700), so there are design
choices made around this.

## Architecture

```
COGs → [Spark mapInPandas: FullImageGPUWorker] → Delta Table
         ├─ Read full image (rasterio, background thread)
         ├─ Transfer to GPU (float32, non_blocking)
         ├─ Tile via torch.unfold() and a chip stride calculation
         ├─ Batched inference (DOFA ViT-Base, FP16)
         └─ Compute spatial bounds (affine transform)
```

Currently distributed via Spark `mapInPandas`, but ideally would use
[Ray](https://www.ray.io/)
for running distributed geospatial inference - good examples include
[Wherobots](https://wherobots.com/blog/fields-of-the-world-rasterflow-geoai-pipeline/) and
[Xoople](https://www.anyscale.com/blog/ray-day-london-recap#how-xoople-runs-geospatial-foundation-model-inference-at-global-scale).

## Installation

This was a lab project so it is not currently not published to PyPI. Install from source or wheel.

### From source (development)

From Git:

```bash
uv pip install git+https://github.com/communitiesuk/geoembed.git
```

Local:

```bash
git clone git+https://github.com/communitiesuk/geoembed.git
cd geoembed
uv sync --all-groups
```

### From wheel (Databricks)

```bash
uv build --wheel --out-dir dist/
```

Then on Databricks:

```python
%pip install "/Workspace/path/to/geoembed-0.1.0.dev0-py3-none-any.whl[spatial,cog,stac]"
dbutils.library.restartPython()
```

Alternatively, if you're using class compute, there's an example `init_example.sh` script which will
install `uv` and the Python wheel on cluster start up.

***Note***: due to Databricks Runtime conflicts you'll have to install `torchgeo` separately on a
GPU cluster:

```python
%pip install torchgeo>=0.7.1
dbutils.library.restartPython()
```

### Extras

A few extra modules can be included depending on your case. We found these were helpful during
initial development, so they are included as optional extras.

| Extra | Use case |
|-------|----------|
| `[spatial]` | Metadata extraction, chipping, spatial operations |
| `[cog]` | GeoTIFF → COG conversion |
| `[stac]` | STAC catalog + STAC GeoParquet generation |
| `[models]` | PyTorch models (local dev only, possibly conflicts with Databricks runtimes) |
| `[all]` | Everything above |

## GPU Cluster config

These are very dependent on individual use cases (e.g. size and complexity of your imagery). It will
likely take some trial and error to avoid hitting various out-of-memory errors: for example, each
task will load an image, a GeoFM model, and spawn various Python processes.

Below are some recommended configurations for a GPU cluster.

### Standard_NC4as_T4_v3 (development)

| Setting | Value | Rationale |
|---------|-------|-----------|
| Driver | Standard_D4_v2 | Lightweight orchestration |
| Workers | 2x Standard_NC4as_T4_v3 | 2 GPUs for parallel processing |
| Runtime | DBR ML 17.3 LTS | Stable PyTorch + CUDA |
| `spark.task.resource.gpu.amount` | 0.5 | 0.5 full GPU per Spark task |
| `spark.task.cpus` | 2 | All CPUs for the task |
| `spark.executor.memory` | 16g | Limited RAM per node |

### Standard_NV36ads_A10v5 (production)

| Setting | Value | Rationale |
|---------|-------|-----------|
| Driver | Standard_D4_v2 | Lightweight orchestration |
| Workers | 2x Standard_NV36ads_A10v5 | 2 GPUs for parallel processing |
| Runtime | DBR ML 17.3 LTS | Stable PyTorch + CUDA |
| `spark.task.resource.gpu.amount` | 0.5 | Half GPU per Spark task |
| `spark.task.cpus` | 18 | CPUs for I/O thread + data loading |
| `spark.executor.memory` | 100g | Holds full images in CPU memory |

For multiple GPUs on a single executor you can experiement with
`spark.executor.resource.gpu.amount`.

## Quickstart

Assuming you've set your GPU Spark config:

```python
from geoembed.backends.spark.config import EmbeddingsConfig
from geoembed.backends.spark.orchestrator import EmbeddingsPipeline
from geoembed.core.config import StorageConfig

config = EmbeddingsConfig(chip_size=224, batch_size=256, mixed_precision=True, model_name="dofa")

pipeline = EmbeddingsPipeline(
    config=config,
    input_storage=StorageConfig(backend="delta", path="catalog.schema.metadata"),
    output_storage=StorageConfig(backend="delta", path="catalog.schema.embeddings"),
)
pipeline.run()
```

Play around with the `batch_size` parameter, for example increasing to `512` or decreasing to `128`
may improve performance.

## Development

### Initial setup

```bash
make setup    # install dev deps + configure pre-commit
make setup-all # install all dependencies + configure pre-commit
```

### Running tests

```bash
make test         # fast tests (excludes slow/gpu)
make test-all     # full suite including slow tests
make test-cov     # with coverage report
```

### Code quality

```bash
make pre-commit   # run all hooks (ruff lint + format + pyrefly + detect-secrets)
make lint         # ruff linter only
make format       # ruff formatter only
make typecheck    # pyrefly type checker only
make check        # pre-commit + tests (full CI check)
```

### Nox (multi-Python testing)

```bash
make nox            # all default sessions (ruff, pyrefly, tests on 3.12 + 3.13)
make nox-lint       # lint session only
make nox-typecheck  # type-check session only
make nox-test       # tests on all Python versions
make nox-coverage   # coverage with HTML report
```

### Building

```bash
make build        # build wheel to dist/
make dbx-build    # build wheel to deps/whl/ for Databricks
make clean        # remove all build artifacts
```

### Documentation

```bash
make serve-docs   # local preview at localhost:8001
make build-docs   # static site to site/
```

Run `make help` to see all available commands.

## License

MIT
