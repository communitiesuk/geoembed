"""Abstract base class for chip readers."""

from abc import ABC, abstractmethod

import numpy as np


class ChipReader(ABC):
    """
    Abstract interface for reading pixel data from imagery sources.

    Implementations handle different I/O backends (rasterio for sync local,
    async-geotiff for async cloud, etc.) while presenting a uniform interface
    to the embedding pipeline.
    """

    @abstractmethod
    def read_chip(
        self,
        parent_path: str,
        col_off: int,
        row_off: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Read a single chip from a parent image.

        Args:
            parent_path: Path to the parent COG/GeoTIFF.
            col_off: Column offset in pixels.
            row_off: Row offset in pixels.
            width: Chip width in pixels.
            height: Chip height in pixels.

        Returns:
            Numpy array of shape (C, H, W) with pixel values.
        """
        ...
