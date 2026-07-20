"""Sliding window chipping strategy — pixel-based tiling with optimal overlap."""

from __future__ import annotations

import os
from collections.abc import Iterator

import rasterio
from rasterio.windows import Window
from shapely.geometry import box
from shapely.wkb import dumps

from geoembed.chipping.services import StrideService
from geoembed.chipping.strategies.base import SparkChipperStrategy


class SlidingWindowStrategy(SparkChipperStrategy):
    """
    Implements pixel-based sliding window chipping.

    Uses StrideService to calculate optimal overlap that ensures the image is fully
    covered without gaps, adjusting the stride dynamically so the last chip aligns
    perfectly with the image edge.

    For 12.5cm imagery with chip_size=224:
    - Each chip = 28m x 28m physical footprint
    - Typical overlap = ~8 pixels (1m) for 8000x8000 parent images
    - Grid = 37x37 = 1,369 chips per parent image
    """

    def __init__(self, chip_size: int = 224):
        """
        Args:
            chip_size: Square chip dimension in pixels.
        """
        self.chip_size = chip_size

    @staticmethod
    def _clean_path(path: str) -> str:
        """Strip Spark/Databricks path prefixes (dbfs:, file:) for local FUSE access."""
        return path.replace("dbfs:", "").replace("file:", "")

    def generate_chips(self, parent_path: str) -> Iterator[dict]:
        """
        Generate virtual chip metadata for a parent image.

        Opens the COG to read dimensions and transform, then calculates all
        chip positions using optimal overlap. Only reads headers (no pixels).

        Args:
            parent_path: Path to the parent COG.

        Yields:
            Metadata dict per chip with geometry (WKB), offsets, and bounds.
        """
        parent_path = self._clean_path(parent_path)

        with rasterio.open(parent_path) as src:
            w, h = src.width, src.height
            transform = src.transform
            parent_name = os.path.splitext(os.path.basename(parent_path))[0]

            overlap_x, overlap_y = StrideService.calculate_optimal_overlap(
                (w, h), (self.chip_size, self.chip_size)
            )

            sx = max(1, self.chip_size - overlap_x)
            sy = max(1, self.chip_size - overlap_y)

            row_starts = range(0, h - self.chip_size + 1, sy)
            col_starts = range(0, w - self.chip_size + 1, sx)

            idx = 0
            for r in row_starts:
                for c in col_starts:
                    window = Window(c, r, self.chip_size, self.chip_size)  # pyrefly: ignore
                    minx, miny, maxx, maxy = rasterio.windows.bounds(window, transform)

                    yield {
                        "chip_id": f"{parent_name}_{idx:06d}",
                        "parent_id": parent_name,
                        "parent_path": parent_path,
                        "col_off": int(c),
                        "row_off": int(r),
                        "width": self.chip_size,
                        "height": self.chip_size,
                        "geometry_wkb": dumps(box(minx, miny, maxx, maxy)),
                        "minx": float(minx),
                        "miny": float(miny),
                        "maxx": float(maxx),
                        "maxy": float(maxy),
                    }
                    idx += 1
