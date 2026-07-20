"""GeoParquet I/O utilities and STAC conversion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd


def from_geoparquet(path: str | Path) -> gpd.GeoDataFrame:
    """
    Read a GeoParquet file into a GeoDataFrame.

    Handles JSON-encoded 'assets' column if present (STAC convention).

    Args:
        path: Path to the GeoParquet file.

    Returns:
        GeoDataFrame with geometry column.
    """
    import json

    gdf = gpd.read_parquet(path)

    if "assets" in gdf.columns and gdf["assets"].dtype == "object":
        gdf["assets"] = gdf["assets"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    return gdf


def to_geoparquet(gdf: gpd.GeoDataFrame, path: str | Path) -> None:
    """
    Write a GeoDataFrame to GeoParquet format.

    JSON-encodes 'assets' column if present (STAC convention).

    Args:
        gdf: GeoDataFrame to write.
        path: Output file path.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_out = gdf.copy()
    if "assets" in df_out.columns:
        df_out["assets"] = df_out["assets"].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else x
        )

    df_out.to_parquet(path, index=False)


def create_geoparquet_chip_index(
    metadata_list: list[dict[str, Any]], output_path: str | Path
) -> Path:
    """
    Convert a list of STAC-like metadata dicts to a GeoParquet chip index.

    Args:
        metadata_list: List of chip metadata dictionaries (must contain 'geometry' key).
        output_path: Where to write the GeoParquet file.

    Returns:
        Path to the created file.
    """
    from shapely.geometry import shape

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    geometries = []
    for meta in metadata_list:
        geom = meta.pop("geometry", None)
        if geom is not None:
            geometries.append(shape(geom) if isinstance(geom, dict) else geom)
        else:
            geometries.append(None)
        records.append(meta)

    df = pd.DataFrame(records)
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:27700")
    gdf.to_parquet(output_path, index=False)

    return output_path
