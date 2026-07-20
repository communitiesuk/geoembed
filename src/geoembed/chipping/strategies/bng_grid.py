"""British National Grid (BNG) chipping strategy — aligns chips to the OSGB grid."""

from __future__ import annotations

import os
from collections.abc import Iterator

import rasterio
from osbng.bng_reference import BNGReference
from shapely.geometry import box, shape
from shapely.wkb import dumps

from geoembed.chipping.strategies.base import SparkChipperStrategy


class BNGGridStrategy(SparkChipperStrategy):
    """
    Implements British National Grid (BNG) based chipping.

    Instead of arbitrary pixel windows, this strategy aligns chips to the OSGB
    grid system (e.g., 50m grid squares). This ensures chips nest cleanly within
    the 1km BNG grid used by the parent imagery.

    Useful when:
    - Working with UK Ordnance Survey / BNG-aligned imagery
    - Downstream analysis requires alignment to standard grid references
    - Comparing with other BNG-referenced datasets (e.g., AEF embeddings)
    """

    def __init__(self, resolution: str | int):
        """
        Args:
            resolution: BNG resolution string (e.g., "50m", "10m") or integer metres.
        """
        self.resolution = resolution

    def generate_chips(self, parent_path: str) -> Iterator[dict]:
        """
        Generate chips aligned to the BNG grid.

        Opens the COG, determines its BNG reference from the filename,
        enumerates child grid squares at the target resolution, and yields
        metadata for each.

        Args:
            parent_path: Path to the parent COG (filename must be a valid BNG reference).

        Yields:
            Metadata dict per chip with BNG-aligned geometry.
        """
        parent_name = os.path.splitext(os.path.basename(parent_path))[0]

        with rasterio.open(parent_path) as src:
            parent_ref = BNGReference(bng_ref_string=parent_name)
            children = parent_ref.bng_to_children(resolution=self.resolution)
            parent_bounds_geom = box(*src.bounds)

            for child in children:
                tile_geom = shape(child.__geo_interface__["geometry"])
                if not tile_geom.intersects(parent_bounds_geom):
                    continue

                geom_to_read = tile_geom.intersection(parent_bounds_geom)
                window = rasterio.windows.from_bounds(*geom_to_read.bounds, transform=src.transform)

                yield {
                    "chip_id": child.bng_ref_compact,
                    "parent_id": parent_name,
                    "parent_path": parent_path,
                    "col_off": int(window.col_off),
                    "row_off": int(window.row_off),
                    "width": int(window.width),
                    "height": int(window.height),
                    "geometry_wkb": dumps(tile_geom),
                    "minx": float(tile_geom.bounds[0]),
                    "miny": float(tile_geom.bounds[1]),
                    "maxx": float(tile_geom.bounds[2]),
                    "maxy": float(tile_geom.bounds[3]),
                }
