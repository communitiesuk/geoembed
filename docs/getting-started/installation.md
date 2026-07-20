# Installation

`geoembed` is not published to PyPI. Install from source or from the built wheel.

## From Source (Development)

```bash
git clone https://github.com/communitiesuk/geoembed.git
cd geoembed
uv sync --group dev --group docs
```

## From Wheel (Databricks)

Build the wheel:

```bash
uv build --wheel --out-dir dist/
```

Install on a Databricks cluster via init script or `%pip`:

```python
%pip install "/Workspace/path/to/geoembed-0.1.0.dev0-py3-none-any.whl[spatial,cog,stac]"
dbutils.library.restartPython()
```

## Extras

| Extra | Packages | Use case |
|-------|----------|----------|
| `[spatial]` | rasterio, geopandas, shapely, osbng, pyproj | Metadata extraction, chipping, spatial ops |
| `[cog]` | rio-cogeo, rasterio | COG conversion |
| `[stac]` | pystac, stac-geoparquet, defusedxml, pyarrow | STAC catalog generation |
| `[models]` | torch, torchvision, torchgeo | Embedding models (local dev only) |
| `[all]` | All of the above | Full local development |

!!! warning "On Databricks"
    Do **not** install `[models]` on Databricks — it conflicts with the runtime's
    pre-installed PyTorch/sympy. Use the runtime's torch and install `torchgeo`
    separately: `%pip install torchgeo>=0.7.1`

## Init Script (Recommended for Clusters)

Amend `deps/init-scripts/init_example.sh` as a cluster init script. It installs the wheel
with `[spatial,cog,stac]` extras automatically.
