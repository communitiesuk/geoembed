"""Synchronous chip reader using rasterio with tile-aware optimisations.

Implements the key I/O pattern from Microsoft's COG throughput paper:
- Open each parent file ONCE and read all chips within it sequentially
- GDAL block cache serves multiple chips per tile decompression
- Sequential row-major ordering within a tile minimises random I/O
"""

import os

import numpy as np
import rasterio
from rasterio.windows import Window

from geoembed.io.readers.base import ChipReader


class RasterioChipReader(ChipReader):
    """
    Reads chips from COGs/GeoTIFFs using rasterio with optimised GDAL settings.

    When chip metadata is sorted by tile (via `geoembed.io.tile_sorting.sort_by_tile`),
    this reader benefits from GDAL's internal block cache: adjacent chips from the
    same 512x512 COG tile share a single decompression, avoiding redundant I/O.

    Performance notes (from Microsoft paper, Zaytar et al. 2025):
    - Tile-aligned reads: 1 decompression per tile (up to 4 chips served)
    - Misaligned reads: up to 4x redundant decompression
    - Local storage: uncompressed COGs are 1.3-1.6x faster than Deflate
    - Remote storage: LERC-ZSTD balances network transfer vs decode time
    """

    def __init__(self) -> None:
        self._apply_gdal_optimisations()

    @staticmethod
    def _apply_gdal_optimisations() -> None:
        """Configure GDAL environment for optimal COG I/O.

        Key settings:
        - GDAL_CACHEMAX: Block cache size in MB. Larger cache retains more
          decompressed tiles, benefiting tile-grouped reads.
        - VSI_CACHE: Caches HTTP responses for cloud COGs.
        - GDAL_HTTP_MULTIPLEX: HTTP/2 multiplexing for concurrent range requests.
        """
        os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
        os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif,.tiff,.TIF,.TIFF"
        os.environ["GDAL_INGESTED_BYTES_AT_OPEN"] = "32768"
        os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
        os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"
        os.environ["GDAL_HTTP_VERSION"] = "2"
        os.environ["VSI_CACHE"] = "TRUE"
        os.environ["VSI_CACHE_SIZE"] = "25000000"
        os.environ["GDAL_CACHEMAX"] = "200"

    def read_chip(
        self,
        parent_path: str,
        col_off: int,
        row_off: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Read a single chip window from a GeoTIFF.

        For best throughput, ensure chip metadata is sorted by tile
        (parent_path → tile_row → tile_col) so the GDAL block cache
        serves multiple chips per tile decompression.

        Args:
            parent_path: Path to the parent COG/GeoTIFF.
            col_off: Column offset in pixels.
            row_off: Row offset in pixels.
            width: Chip width in pixels.
            height: Chip height in pixels.

        Returns:
            Numpy array of shape (C, H, W).
        """
        window = Window(col_off, row_off, width, height)  # pyrefly: ignore
        with rasterio.open(parent_path) as src:
            return src.read(window=window)

    def read_chips_from_parent(
        self,
        parent_path: str,
        windows: list[tuple[int, int, int, int]],
    ) -> list[np.ndarray]:
        """
        Read multiple chips from a single parent, opening the file once.

        This is the optimal access pattern for tile-sorted metadata: all chips
        from the same parent are read in a single file-open context, allowing
        the GDAL block cache to serve adjacent tiles without re-decompression.

        Args:
            parent_path: Path to the parent COG.
            windows: List of (col_off, row_off, width, height) tuples.

        Returns:
            List of numpy arrays, one per window, shape (C, H, W).
        """
        results = []
        with rasterio.open(parent_path) as src:
            for col_off, row_off, width, height in windows:
                window = Window(col_off, row_off, width, height)  # pyrefly: ignore
                results.append(src.read(window=window))
        return results
