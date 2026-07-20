"""STAC catalog generation with Z-order sorted GeoParquet output.

Builds a STAC Catalog → Collection → Items hierarchy and exports items as a
GeoParquet 1.1 file with bbox.covering metadata for efficient spatial queries
(DuckDB row group predicate pushdown).
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
from dataclasses import dataclass
from functools import cache
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pystac
from stac_geoparquet.arrow import parse_stac_items_to_arrow


@dataclass(frozen=True)
class StacCatalogConfig:
    """
    Configuration for STAC catalog generation.

    Attributes:
        catalog_id: Unique identifier for the catalog.
        catalog_title: Human-readable catalog title.
        catalog_description: Catalog description.
        collection_id: Unique identifier for the collection.
        collection_title: Human-readable collection title.
        collection_description: Collection description.
        output_path: Root directory for the STAC catalog output.
        parquet_compression: Compression codec for GeoParquet (zstd, snappy, etc.).
        row_group_size: Number of items per Parquet row group.
    """

    catalog_id: str
    catalog_title: str
    catalog_description: str
    collection_id: str
    collection_title: str
    collection_description: str
    output_path: str
    parquet_compression: str = "zstd"
    row_group_size: int = 1000


def compute_collection_extent(item_dicts: list[dict]) -> pystac.Extent:
    """
    Derive spatial and temporal extents from a list of STAC item dicts.

    Args:
        item_dicts: List of STAC item dictionaries.

    Returns:
        pystac.Extent covering all items.
    """
    bboxes = [d["bbox"] for d in item_dicts if d.get("bbox")]
    spatial_bbox = (
        [
            min(b[0] for b in bboxes),
            min(b[1] for b in bboxes),
            max(b[2] for b in bboxes),
            max(b[3] for b in bboxes),
        ]
        if bboxes
        else [-180.0, -90.0, 180.0, 90.0]
    )

    datetimes: list[datetime.datetime] = []
    for d in item_dicts:
        dt_str = d.get("properties", {}).get("datetime")
        if dt_str:
            with contextlib.suppress(ValueError):
                datetimes.append(datetime.datetime.fromisoformat(dt_str))

    start_dt = min(datetimes) if datetimes else None
    end_dt = max(datetimes) if datetimes else None

    return pystac.Extent(
        spatial=pystac.SpatialExtent(bboxes=[spatial_bbox]),
        temporal=pystac.TemporalExtent(intervals=[[start_dt, end_dt]]),
    )


def sort_by_morton(
    table: pa.Table,
    lon_range: tuple[float, float] = (-180.0, 180.0),
    lat_range: tuple[float, float] = (-90.0, 90.0),
) -> pa.Table:
    """
    Sort an Arrow table by Z-order (Morton code) on bbox centroids.

    Clusters spatially nearby items into the same Parquet row groups so that
    DuckDB can skip entire row groups when the query bbox doesn't overlap their
    extent (predicate pushdown via the GeoParquet 1.1 bbox.covering statistics).

    Args:
        table: Arrow table with a 'bbox' struct column.
        lon_range: Longitude normalisation range (min, max).
        lat_range: Latitude normalisation range (min, max).

    Returns:
        Spatially sorted Arrow table.
    """
    bbox = table.column("bbox").combine_chunks()
    xmin = bbox.field("xmin").to_pylist()
    xmax = bbox.field("xmax").to_pylist()
    ymin = bbox.field("ymin").to_pylist()
    ymax = bbox.field("ymax").to_pylist()

    cx = [(a + b) / 2 for a, b in zip(xmin, xmax, strict=False)]
    cy = [(a + b) / 2 for a, b in zip(ymin, ymax, strict=False)]

    N = 65535

    def _clamp(v: float, lo: float, hi: float) -> int:
        return max(0, min(N, int((v - lo) / (hi - lo) * N)))

    ix = [_clamp(x, *lon_range) for x in cx]
    iy = [_clamp(y, *lat_range) for y in cy]

    def _morton(x: int, y: int) -> int:
        """Interleave bits of two 16-bit integers (Z-order / Morton code)."""
        x &= 0xFFFF
        y &= 0xFFFF
        x = (x | (x << 8)) & 0x00FF00FF
        x = (x | (x << 4)) & 0x0F0F0F0F
        x = (x | (x << 2)) & 0x33333333
        x = (x | (x << 1)) & 0x55555555
        y = (y | (y << 8)) & 0x00FF00FF
        y = (y | (y << 4)) & 0x0F0F0F0F
        y = (y | (y << 2)) & 0x33333333
        y = (y | (y << 1)) & 0x55555555
        return x | (y << 1)

    codes = [_morton(x, y) for x, y in zip(ix, iy, strict=False)]
    order = sorted(range(len(codes)), key=lambda i: codes[i])
    return table.take(order)


def _ensure_utc(dt_val: Any) -> datetime.datetime:
    """Ensure a datetime value is timezone-aware (UTC)."""
    if dt_val is not None and hasattr(dt_val, "tzinfo") and dt_val.tzinfo is None:
        return dt_val.replace(tzinfo=datetime.UTC)
    if dt_val is None:
        return datetime.datetime.now(datetime.UTC)
    return dt_val


def _resolve_image_path(row: dict[str, Any], item_id: str, cog_base_path: str | None) -> str:
    """Resolve the COG asset href from the row metadata."""
    image_path = str(row.get("image_path", ""))
    if cog_base_path and not image_path.startswith("/"):
        return f"{cog_base_path}/{item_id}.tif"
    return image_path


def _build_single_stac_item(
    row: dict[str, Any],
    cog_base_path: str | None,
    reproject_fn: Any,
) -> dict | None:
    """
    Build a single STAC item dict from a metadata row.

    Args:
        row: Metadata dict with keys: id, image_path, geometry_wkb, epsg, datetime, provider.
        cog_base_path: Optional base path for COG assets.
        reproject_fn: Callable(epsg) -> transform function for reprojection to WGS84.

    Returns:
        STAC item dict, or None if the row is invalid.
    """
    from shapely.geometry import box, mapping
    from shapely.ops import transform as shapely_transform
    from shapely.wkb import loads as wkb_loads

    item_id = str(row["id"])
    if item_id.startswith("ERROR_"):
        return None

    geom_wkb = row.get("geometry_wkb")
    if geom_wkb is not None:
        geom = wkb_loads(bytes(geom_wkb))
    elif all(row.get(k) is not None for k in ("minx", "miny", "maxx", "maxy")):
        geom = box(float(row["minx"]), float(row["miny"]), float(row["maxx"]), float(row["maxy"]))
    else:
        return None

    epsg = row.get("epsg")
    if epsg and int(epsg) != 4326:
        geom = shapely_transform(reproject_fn(int(epsg)), geom)

    item = pystac.Item(
        id=item_id,
        geometry=mapping(geom),
        bbox=list(geom.bounds),
        datetime=_ensure_utc(row.get("datetime")),
        properties={},
    )

    if epsg:
        item.stac_extensions.append(
            "https://stac-extensions.github.io/projection/v1.0.0/schema.json"
        )
        item.properties["proj:epsg"] = int(epsg)

    if row.get("provider"):
        item.common_metadata.providers = [
            pystac.Provider(name=str(row["provider"]), roles=[pystac.ProviderRole.PRODUCER])
        ]

    item.stac_extensions.append("https://stac-extensions.github.io/eo/v1.0.0/schema.json")
    item.add_asset(
        "image",
        pystac.Asset(
            href=_resolve_image_path(row, item_id, cog_base_path),
            media_type=pystac.MediaType.COG,
            roles=["visual"],
            extra_fields={
                "eo:bands": [
                    {"name": "red", "common_name": "red"},
                    {"name": "green", "common_name": "green"},
                    {"name": "blue", "common_name": "blue"},
                ]
            },
        ),
    )
    return item.to_dict()


def build_stac_items_from_metadata(
    metadata_rows: list[dict[str, Any]],
    cog_base_path: str | None = None,
) -> list[dict]:
    """
    Convert metadata rows into STAC item dictionaries.

    Args:
        metadata_rows: List of dicts with keys: id, image_path, geometry_wkb, epsg,
                       datetime, provider.
        cog_base_path: Optional base path to prepend to image_path for COG assets.

    Returns:
        List of STAC item dicts (JSON-serialisable).
    """
    import pyproj

    @cache
    def get_transform(epsg: int):
        return pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform

    items = []
    for row in metadata_rows:
        try:
            item_dict = _build_single_stac_item(row, cog_base_path, get_transform)
            if item_dict is not None:
                items.append(item_dict)
        except Exception as e:
            print(f"Failed to build STAC item for {row.get('id', 'unknown')}: {e}")

    return items


def generate_stac_catalog(
    item_dicts: list[dict],
    config: StacCatalogConfig,
    lon_range: tuple[float, float] = (-180.0, 180.0),
    lat_range: tuple[float, float] = (-90.0, 90.0),
) -> str:
    """
    Generate a full STAC catalog with Z-order sorted GeoParquet.

    Creates:
    - STAC JSON catalog hierarchy (catalog.json, collection.json)
    - GeoParquet 1.1 file with bbox.covering metadata

    Args:
        item_dicts: List of STAC item dictionaries.
        config: Catalog configuration.
        lon_range: Longitude range for Morton code normalisation.
        lat_range: Latitude range for Morton code normalisation.

    Returns:
        Path to the generated GeoParquet file.
    """
    catalog = pystac.Catalog(
        id=config.catalog_id,
        title=config.catalog_title,
        description=config.catalog_description,
    )

    collection = pystac.Collection(
        id=config.collection_id,
        title=config.collection_title,
        description=config.collection_description,
        extent=compute_collection_extent(item_dicts),
    )
    catalog.add_child(collection)

    os.makedirs(config.output_path, exist_ok=True)
    catalog.normalize_hrefs(config.output_path)
    catalog.save(catalog_type=pystac.CatalogType.RELATIVE_PUBLISHED)

    collection_dir = os.path.join(config.output_path, config.collection_id)
    parquet_path = os.path.join(collection_dir, "items.parquet")

    reader = parse_stac_items_to_arrow(item_dicts)
    table = reader.read_all()
    table = sort_by_morton(table, lon_range=lon_range, lat_range=lat_range)

    geo_meta = {
        b"geo": json.dumps({
            "version": "1.1.0",
            "primary_column": "geometry",
            "columns": {
                "geometry": {
                    "encoding": "WKB",
                    "geometry_types": ["Polygon"],
                    "edges": "planar",
                    "covering": {
                        "bbox": {
                            "xmin": ["bbox", "xmin"],
                            "ymin": ["bbox", "ymin"],
                            "xmax": ["bbox", "xmax"],
                            "ymax": ["bbox", "ymax"],
                        }
                    },
                }
            },
        }).encode(),
    }
    table = table.replace_schema_metadata(geo_meta)
    pq.write_table(
        table,
        parquet_path,
        compression=config.parquet_compression,
        row_group_size=config.row_group_size,
    )
    print(
        f"GeoParquet saved to {parquet_path} "
        f"({len(table)} items, {os.path.getsize(parquet_path) / 1024 / 1024:.1f} MB)"
    )

    return parquet_path
