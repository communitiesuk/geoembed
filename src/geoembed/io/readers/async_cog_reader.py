"""Async chip reader using async-geotiff and obstore — GDAL-free, cloud-native I/O."""

import asyncio
from typing import Any

import numpy as np

from geoembed.io.readers.base import ChipReader


class AsyncCogChipReader(ChipReader):
    """
    Reads chips from COGs using async-geotiff with obstore.

    This is the preferred reader for the Ray backend. It provides:
    - GDAL-free reads (no GDAL dependency on workers)
    - Async I/O for high concurrency
    - Connection caching for repeated reads from the same parent
    - Native support for local, Azure, S3, and GCS via obstore

    Requires: geoembed[ray] extras.
    """

    def __init__(self, store_path: str | None = None, store_options: dict[str, Any] | None = None):
        """
        Initialise the async reader.

        Args:
            store_path: Root path for the obstore LocalStore. If None, uses absolute paths.
            store_options: Options passed to the obstore backend.
        """
        try:
            from async_geotiff import GeoTIFF, Window  # noqa: F401
            from obstore.store import LocalStore  # noqa: F401
        except ImportError as e:
            msg = "async-geotiff and obstore required. Install with: pip install geoembed[ray]"
            raise ImportError(msg) from e

        self._store_path = store_path
        self._store_options = store_options or {}
        self._store = None
        self._geotiffs: dict[str, Any] = {}

    def _get_store(self) -> Any:
        """Lazily initialise the obstore backend."""
        if self._store is None:
            from obstore.store import LocalStore

            if self._store_path:
                self._store = LocalStore(self._store_path)
            else:
                self._store = LocalStore("/")
        return self._store

    async def _read_chip_async(
        self,
        parent_path: str,
        col_off: int,
        row_off: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Async implementation of chip reading."""
        from async_geotiff import GeoTIFF, Window

        if parent_path not in self._geotiffs:
            self._geotiffs[parent_path] = await GeoTIFF.open(parent_path, store=self._get_store())

        geotiff = self._geotiffs[parent_path]
        window = Window(col_off=col_off, row_off=row_off, width=width, height=height)
        array = await geotiff.read(window=window)
        return array.data

    def read_chip(
        self,
        parent_path: str,
        col_off: int,
        row_off: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Read a chip from a COG using async I/O (sync wrapper).

        For batch reads in the Ray backend, use `read_chips_async` directly.
        """
        return asyncio.run(self._read_chip_async(parent_path, col_off, row_off, width, height))

    async def read_chips_async(self, chip_metadata: list[dict[str, Any]]) -> list[np.ndarray]:
        """
        Read multiple chips concurrently.

        Args:
            chip_metadata: List of dicts with keys: parent_path, col_off, row_off, width, height.

        Returns:
            List of numpy arrays (C, H, W) in the same order as input.
        """
        tasks = [
            self._read_chip_async(
                m["parent_path"], m["col_off"], m["row_off"], m["width"], m["height"]
            )
            for m in chip_metadata
        ]
        return await asyncio.gather(*tasks)
