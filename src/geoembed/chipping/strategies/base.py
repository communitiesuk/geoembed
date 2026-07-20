"""Abstract base classes for chipping strategies."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChipResult:
    """
    Standardised output from a local chipping operation.

    Immutable data carrier linking generated metadata to its physical file.

    Attributes:
        metadata: STAC-compliant metadata dictionary for the chip.
        chip_path: Absolute file path to the generated GeoTIFF chip.
        parent_id: ID of the source image this chip was derived from.
    """

    metadata: dict[str, Any]
    chip_path: Path
    parent_id: str


class ChipperStrategy(ABC):
    """
    Strategy interface for local chipping — defines HOW to chip a raster.

    Implementations determine the tiling algorithm (e.g., sliding window, grid).
    Used with a JobExecutor for parallel local execution.
    """

    @abstractmethod
    def process_item(self, item: Any) -> list[ChipResult]:
        """
        Process a single source item to generate image chips.

        Args:
            item: A STAC item or similar metadata object representing the source raster.

        Returns:
            List of ChipResults containing metadata and paths for every generated chip.
        """
        ...


class SparkChipperStrategy(ABC):
    """
    Strategy interface for Spark-based chipping — generates chip metadata from a parent image.

    Unlike ChipperStrategy (which produces physical files), this generates only
    virtual chip metadata (offsets, geometries) for later materialisation or direct read.
    """

    @abstractmethod
    def generate_chips(self, parent_path: str) -> Iterator[dict]:
        """
        Generator yielding metadata dictionaries for each chip.

        Args:
            parent_path: Path to the source COG (e.g., Unity Catalog Volume path).

        Yields:
            dict with keys: chip_id, parent_id, parent_path, col_off, row_off,
            width, height, geometry_wkb, minx, miny, maxx, maxy.
        """
        ...
