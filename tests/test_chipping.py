"""Tests for chipping module — FSM: states, transitions, guards, invariants.

FSM for SlidingWindowStrategy:
  INIT → OPEN_COG → READ_METADATA → COMPUTE_OVERLAP → GENERATE_GRID → YIELD_CHIPS* → DONE

Key Invariants:
  1. Full image coverage (every pixel in at least one chip)
  2. No gaps between chips
  3. Geometry consistency (WKB matches bounds)
  4. Chip ID uniqueness
"""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("shapely")

import rasterio.transform
from shapely.wkb import loads as wkb_loads

from geoembed.chipping.services import StrideService


class TestStrideService:
    """StrideService invariant: overlap ensures full coverage, no gaps."""

    @pytest.mark.parametrize(
        "image_dims,chip_dims,expected",
        [
            ((8000, 8000), (224, 224), (8, 8)),
            ((448, 448), (224, 224), (0, 0)),
            ((100, 100), (224, 224), (0, 0)),
            ((1000, 1000), (224, 224), (30, 30)),
            ((224, 224), (224, 224), (0, 0)),
        ],
    )
    def test_overlap_values(self, image_dims, chip_dims, expected):
        assert StrideService.calculate_optimal_overlap(image_dims, chip_dims) == expected

    @pytest.mark.parametrize(
        "w,h",
        [(8000, 8000), (4000, 8000), (1000, 1000), (448, 448), (500, 700)],
    )
    def test_invariant_full_coverage(self, w, h):
        """INVARIANT: last chip must end exactly at image edge."""
        chip_size = 224
        if w < chip_size or h < chip_size:
            return  # Skip — image smaller than chip

        overlap_x, overlap_y = StrideService.calculate_optimal_overlap(
            (w, h), (chip_size, chip_size)
        )
        stride_x = chip_size - overlap_x
        stride_y = chip_size - overlap_y

        # Last valid start position + chip_size must equal image dimension
        last_col = max(range(0, w - chip_size + 1, stride_x))
        last_row = max(range(0, h - chip_size + 1, stride_y))
        assert last_col + chip_size == w, f"Width gap: {w} - ({last_col} + {chip_size})"
        assert last_row + chip_size == h, f"Height gap: {h} - ({last_row} + {chip_size})"

    def test_invariant_non_negative_overlap(self):
        """INVARIANT: overlap is always >= 0."""
        for w in [224, 500, 1000, 4000, 8000]:
            for h in [224, 500, 1000, 4000, 8000]:
                ox, oy = StrideService.calculate_optimal_overlap((w, h), (224, 224))
                assert ox >= 0, f"Negative overlap_x for ({w}, {h})"
                assert oy >= 0, f"Negative overlap_y for ({w}, {h})"


class TestSlidingWindowStrategy:
    """FSM tests for SlidingWindowStrategy.generate_chips()."""

    def _make_mock_src(self, width, height):
        """Helper: create mocked rasterio src."""
        mock_src = MagicMock()
        mock_src.width = width
        mock_src.height = height
        mock_src.transform = rasterio.transform.from_bounds(
            362000, 600000, 362000 + width * 0.125, 600000 + height * 0.125, width, height
        )
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)
        return mock_src

    def test_transition_init_to_done(self):
        """Full pipeline: INIT → OPEN → READ → OVERLAP → GRID → YIELD → DONE."""
        from geoembed.chipping.strategies.sliding_window import SlidingWindowStrategy

        strategy = SlidingWindowStrategy(chip_size=224)
        mock_src = self._make_mock_src(8000, 8000)

        with patch("rasterio.open", return_value=mock_src):
            chips = list(strategy.generate_chips("/fake/NT6200.tif"))

        assert len(chips) == 1369  # 37 x 37

        # Verify absolute coordinates for the first chip (top-left corner).
        # The mock transform is from_bounds(362000, 600000, ...) so minx should
        # start at 362000.0 and miny should be near the top of the image.
        first = chips[0]
        assert first["minx"] == pytest.approx(362000.0, abs=1e-3)
        # maxy corresponds to the top of the image (northing 600000 + height*res)
        expected_maxy = 600000 + 8000 * 0.125  # 601000.0
        assert first["maxy"] == pytest.approx(expected_maxy, abs=1e-3)

    def test_invariant_chip_id_uniqueness(self):
        """INVARIANT: all chip IDs are unique."""
        from geoembed.chipping.strategies.sliding_window import SlidingWindowStrategy

        strategy = SlidingWindowStrategy(chip_size=224)
        mock_src = self._make_mock_src(8000, 8000)

        with patch("rasterio.open", return_value=mock_src):
            chips = list(strategy.generate_chips("/fake/NT6200.tif"))

        ids = [c["chip_id"] for c in chips]
        assert len(ids) == len(set(ids)), "Duplicate chip IDs found"

    def test_invariant_geometry_consistency(self):
        """INVARIANT: WKB geometry matches minx/miny/maxx/maxy bounds."""
        from geoembed.chipping.strategies.sliding_window import SlidingWindowStrategy

        strategy = SlidingWindowStrategy(chip_size=224)
        mock_src = self._make_mock_src(448, 448)

        with patch("rasterio.open", return_value=mock_src):
            chips = list(strategy.generate_chips("/fake/img.tif"))

        for chip in chips:
            geom = wkb_loads(chip["geometry_wkb"])
            bounds = geom.bounds
            assert abs(bounds[0] - chip["minx"]) < 1e-6
            assert abs(bounds[1] - chip["miny"]) < 1e-6
            assert abs(bounds[2] - chip["maxx"]) < 1e-6
            assert abs(bounds[3] - chip["maxy"]) < 1e-6

    def test_invariant_bounds_ordering(self):
        """INVARIANT: minx < maxx AND miny < maxy for all chips."""
        from geoembed.chipping.strategies.sliding_window import SlidingWindowStrategy

        strategy = SlidingWindowStrategy(chip_size=224)
        mock_src = self._make_mock_src(8000, 8000)

        with patch("rasterio.open", return_value=mock_src):
            chips = list(strategy.generate_chips("/fake/NT6200.tif"))

        for chip in chips:
            assert chip["minx"] < chip["maxx"], f"minx >= maxx for {chip['chip_id']}"
            assert chip["miny"] < chip["maxy"], f"miny >= maxy for {chip['chip_id']}"

    def test_invariant_schema_completeness(self):
        """INVARIANT: every chip has all required keys."""
        from geoembed.chipping.strategies.sliding_window import SlidingWindowStrategy

        strategy = SlidingWindowStrategy(chip_size=224)
        mock_src = self._make_mock_src(448, 448)

        with patch("rasterio.open", return_value=mock_src):
            chips = list(strategy.generate_chips("/fake/img.tif"))

        required_keys = {
            "chip_id",
            "parent_id",
            "parent_path",
            "col_off",
            "row_off",
            "width",
            "height",
            "geometry_wkb",
            "minx",
            "miny",
            "maxx",
            "maxy",
        }
        for chip in chips:
            assert required_keys.issubset(chip.keys())

    def test_guard_path_cleaning(self):
        """GUARD: dbfs: and file: prefixes stripped before rasterio.open."""
        from geoembed.chipping.strategies.sliding_window import SlidingWindowStrategy

        strategy = SlidingWindowStrategy(chip_size=224)
        assert strategy._clean_path("dbfs:/Volumes/foo.tif") == "/Volumes/foo.tif"
        assert strategy._clean_path("file:/Volumes/foo.tif") == "/Volumes/foo.tif"
        assert strategy._clean_path("/Volumes/foo.tif") == "/Volumes/foo.tif"

    def test_edge_image_smaller_than_chip(self):
        """EDGE: image < chip_size produces zero chips."""
        from geoembed.chipping.strategies.sliding_window import SlidingWindowStrategy

        strategy = SlidingWindowStrategy(chip_size=224)
        mock_src = self._make_mock_src(100, 100)

        with patch("rasterio.open", return_value=mock_src):
            chips = list(strategy.generate_chips("/fake/small.tif"))

        assert len(chips) == 0

    def test_chip_id_format(self):
        """Chip ID = parent_stem_XXXXXX."""
        from geoembed.chipping.strategies.sliding_window import SlidingWindowStrategy

        strategy = SlidingWindowStrategy(chip_size=224)
        mock_src = self._make_mock_src(448, 448)

        with patch("rasterio.open", return_value=mock_src):
            chips = list(strategy.generate_chips("/fake/NT6200.tif"))

        assert chips[0]["chip_id"] == "NT6200_000000"
        assert chips[0]["parent_id"] == "NT6200"
        assert chips[0]["parent_path"] == "/fake/NT6200.tif"
