"""Metadata readers — extract spatial and temporal information from imagery files.

Follows Single Responsibility Principle (SRP): each reader handles one data source.
"""

import datetime as dt
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as ET
import rasterio
from shapely.geometry import Polygon, mapping


class MetadataReader(ABC):
    """Abstract interface for file metadata readers."""

    @abstractmethod
    def read(self, file_path: Path) -> dict[str, Any]:
        """
        Extract metadata from a file.

        Args:
            file_path: Path to the file to read.

        Returns:
            Dictionary of extracted metadata fields.
        """
        ...


class XmlSidecarReader(MetadataReader):
    """
    Extracts temporal and provider metadata from ISO 19115 XML sidecars.

    Parses XML metadata files (.xml) that accompany GeoTIFFs to extract:
    - Capture datetime (from EX_TemporalExtent)
    - EPSG code (from referenceSystemIdentifier)
    - Provider/organisation name (from CI_ResponsibleParty)
    """

    def __init__(self) -> None:
        self.ns = {
            "gmd": "http://www.isotc211.org/2005/gmd",
            "gco": "http://www.isotc211.org/2005/gco",
            "gml": "http://www.opengis.net/gml/3.2",
            "gmx": "http://www.isotc211.org/2005/gmx",
            "xlink": "http://www.w3.org/1999/xlink",
        }

    def read(self, file_path: Path) -> dict[str, Any]:
        """
        Read metadata from the XML sidecar accompanying a GeoTIFF.

        Args:
            file_path: Path to the GeoTIFF (XML is assumed to be at same path with .xml extension).

        Returns:
            Dict with keys: xml_path, datetime, epsg, provider. Empty dict if no XML found.
        """
        xml_path = file_path.with_suffix(".xml")
        if not xml_path.exists():
            return {}

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            if root is None:
                return {}
            return {
                "xml_path": xml_path,
                "datetime": self._extract_datetime(root),
                "epsg": self._extract_epsg(root),
                "provider": self._find_text(
                    root, ".//gmd:CI_ResponsibleParty/gmd:organisationName/gco:CharacterString"
                ),
            }
        except Exception:
            return {}

    def _find_text(self, root: Element, xpath: str) -> str | None:
        el = root.find(xpath, self.ns)
        return el.text if el is not None else None

    def _extract_datetime(self, root: Element) -> dt.datetime | None:
        dt_str = self._find_text(root, ".//gmd:EX_TemporalExtent//gml:timePosition")
        if not dt_str:
            dt_str = self._find_text(root, ".//gmd:EX_TemporalExtent//gml:beginPosition")

        if dt_str:
            try:
                return dt.datetime.fromisoformat(dt_str).replace(tzinfo=dt.UTC)
            except ValueError:
                pass
        return None

    def _extract_epsg(self, root: Element) -> int | None:
        anchor = root.find(".//gmd:referenceSystemIdentifier//gmx:Anchor", self.ns)
        if (
            anchor is not None
            and (href := anchor.get(f"{{{self.ns['xlink']}}}href"))
            and (match := re.search(r"(\d+)$", href))
        ):
            return int(match.group(1))
        return None


class GeotiffSpatialReader(MetadataReader):
    """
    Extracts spatial bounds and geometry from GeoTIFF headers.

    Uses rasterio to read CRS and bounds without loading pixel data.
    Returns native-CRS bounds (not reprojected).
    """

    def read(self, file_path: Path) -> dict[str, Any]:
        """
        Read spatial metadata from a GeoTIFF.

        Args:
            file_path: Path to the GeoTIFF.

        Returns:
            Dict with keys: bbox (list[float]), footprint (GeoJSON), epsg (int|None).
        """
        with rasterio.open(file_path) as src:
            bounds = src.bounds
            crs = src.crs
            native_poly = Polygon.from_bounds(*bounds)
            epsg = crs.to_epsg() if crs else None
            return {"bbox": list(bounds), "footprint": mapping(native_poly), "epsg": epsg}
