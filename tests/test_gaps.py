"""Gap tests: covers untested code paths in worker, config, delta writer, STAC, sorting."""

from __future__ import annotations

import dataclasses
import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

torch = pytest.importorskip("torch")

from geoembed.backends.spark.config import EmbeddingsConfig  # noqa: E402
from geoembed.backends.spark.worker import (  # noqa: E402
    FullImageGPUWorker,
    _load_image,
)
from geoembed.chipping.services import StrideService  # noqa: E402
from geoembed.io.tile_sorting import sort_by_tile  # noqa: E402

_has_pystac = True
try:
    from geoembed.metadata.stac_catalog import (
        _ensure_utc,
        _resolve_image_path,
    )
except ImportError:
    _has_pystac = False
    _ensure_utc = None  # type: ignore[assignment]
    _resolve_image_path = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# TestFullImageGPUWorkerTileOnGpu
# ---------------------------------------------------------------------------


class TestFullImageGPUWorkerTileOnGpu:
    """Test the actual FullImageGPUWorker._tile_on_gpu method."""

    @pytest.fixture
    def worker(self) -> FullImageGPUWorker:
        return FullImageGPUWorker(chip_size=224, batch_size=128, model_name="dofa")

    def test_448_produces_4_patches(self, worker: FullImageGPUWorker) -> None:
        """A 448x448 image should produce exactly 4 patches of 224x224."""
        image = torch.randn(3, 448, 448)
        patches, offsets = worker._tile_on_gpu(image)

        assert patches.shape == (4, 3, 224, 224)
        assert len(offsets) == 4

    def test_448_offsets_match_expected(self, worker: FullImageGPUWorker) -> None:
        """Offsets for 448x448 with chip_size=224 should be the four corners."""
        image = torch.randn(3, 448, 448)
        _, offsets = worker._tile_on_gpu(image)

        expected = [(0, 0), (224, 0), (0, 224), (224, 224)]
        assert offsets == expected

    def test_8000_produces_1369_patches(self, worker: FullImageGPUWorker) -> None:
        """An 8000x8000 image should produce 37x37 = 1369 patches."""
        image = torch.zeros(3, 8000, 8000)
        patches, offsets = worker._tile_on_gpu(image)

        assert patches.shape[0] == 1369
        assert len(offsets) == 1369

    def test_pixel_alignment(self, worker: FullImageGPUWorker) -> None:
        """First pixel of each patch must match image at that offset."""
        rng = np.random.default_rng(42)
        arr = rng.random((3, 448, 448), dtype=np.float32)
        image = torch.from_numpy(arr)

        patches, offsets = worker._tile_on_gpu(image)

        for i, (col_off, row_off) in enumerate(offsets):
            expected = image[:, row_off, col_off]
            actual = patches[i, :, 0, 0]
            torch.testing.assert_close(actual, expected)


# ---------------------------------------------------------------------------
# TestFullImageGPUWorkerEmptyBatch
# ---------------------------------------------------------------------------


class TestFullImageGPUWorkerEmptyBatch:
    """Test that empty input returns a 0-row table with correct schema."""

    def test_empty_paths_returns_empty_table(self) -> None:
        """Calling worker with empty image_path array returns 0-row table."""
        worker = FullImageGPUWorker(chip_size=224, batch_size=128, model_name="dofa")
        # Mock _setup_model so we don't need the model weights
        worker.embedder = MagicMock()

        result = worker({"image_path": np.array([])})

        assert isinstance(result, pa.Table)
        assert result.num_rows == 0
        expected_cols = {
            "chip_id",
            "parent_id",
            "parent_path",
            "embedding",
            "minx",
            "miny",
            "maxx",
            "maxy",
        }
        assert set(result.column_names) == expected_cols


# ---------------------------------------------------------------------------
# TestLoadImagePathCleaning
# ---------------------------------------------------------------------------


class TestLoadImagePathCleaning:
    """Test that _load_image strips dbfs: and file: prefixes."""

    @patch("geoembed.backends.spark.worker.rasterio.open")
    def test_dbfs_prefix_stripped(self, mock_open: MagicMock) -> None:
        """dbfs:/Volumes/path.tif should open /Volumes/path.tif."""
        mock_src = MagicMock()
        mock_src.read.return_value = np.zeros((3, 224, 224), dtype=np.uint8)
        mock_src.transform = [0.125, 0, 0, 0, -0.125, 0]
        mock_src.width = 224
        mock_src.height = 224
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_src

        _load_image("dbfs:/Volumes/path.tif")

        mock_open.assert_called_once_with("/Volumes/path.tif")

    @patch("geoembed.backends.spark.worker.rasterio.open")
    def test_file_prefix_stripped(self, mock_open: MagicMock) -> None:
        """file:/path.tif should open /path.tif."""
        mock_src = MagicMock()
        mock_src.read.return_value = np.zeros((3, 224, 224), dtype=np.uint8)
        mock_src.transform = [0.125, 0, 0, 0, -0.125, 0]
        mock_src.width = 224
        mock_src.height = 224
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_src

        _load_image("file:/path.tif")

        mock_open.assert_called_once_with("/path.tif")


# ---------------------------------------------------------------------------
# TestEmbeddingsConfigDefaults
# ---------------------------------------------------------------------------


class TestEmbeddingsConfigDefaults:
    """Test the EmbeddingsConfig frozen dataclass."""

    def test_default_values(self) -> None:
        """Defaults: chip_size=224, batch_size=256, mixed_precision=True, model=dofa."""
        config = EmbeddingsConfig()

        assert config.chip_size == 224
        assert config.batch_size == 256
        assert config.mixed_precision is True
        assert config.model_name == "dofa"

    def test_frozen_raises_on_mutation(self) -> None:
        """Frozen dataclass should raise on attribute mutation."""
        config = EmbeddingsConfig()

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.chip_size = 512  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestDeltaWriterMocked
# ---------------------------------------------------------------------------


class TestDeltaWriterMocked:
    """Test DeltaWriter with mocked Spark session."""

    @patch("geoembed.io.writers.delta_writer.DeltaWriter._get_spark")
    def test_write_batch_calls_create_and_save(self, mock_get_spark: MagicMock) -> None:
        """write_batch must call createDataFrame and saveAsTable."""
        from geoembed.io.writers.delta_writer import DeltaWriter

        mock_spark = MagicMock()
        mock_get_spark.return_value = mock_spark

        writer = DeltaWriter(table_name="cat.schema.table")
        embeddings = np.random.default_rng(0).random((3, 128)).astype(float)
        chip_ids = ["a", "b", "c"]

        writer.write_batch(chip_ids, embeddings)

        mock_spark.createDataFrame.assert_called_once()
        mock_sdf = mock_spark.createDataFrame.return_value
        mock_sdf.write.format.return_value.mode.return_value.option.return_value.saveAsTable.assert_called_once()

    @patch("geoembed.io.writers.delta_writer.DeltaWriter._get_spark")
    def test_finalize_with_cluster_columns_calls_sql(self, mock_get_spark: MagicMock) -> None:
        """finalize with cluster_columns should call spark.sql ALTER TABLE."""
        from geoembed.io.writers.delta_writer import DeltaWriter

        mock_spark = MagicMock()
        mock_get_spark.return_value = mock_spark

        writer = DeltaWriter(table_name="cat.schema.table", cluster_columns=["chip_id"])
        writer._spark = mock_spark

        writer.finalize()

        mock_spark.sql.assert_called_once()
        call_arg = mock_spark.sql.call_args[0][0]
        assert "ALTER TABLE" in call_arg
        assert "chip_id" in call_arg

    @patch("geoembed.io.writers.delta_writer.DeltaWriter._get_spark")
    def test_finalize_without_cluster_columns_no_sql(self, mock_get_spark: MagicMock) -> None:
        """finalize with cluster_columns=None should NOT call spark.sql."""
        from geoembed.io.writers.delta_writer import DeltaWriter

        mock_spark = MagicMock()
        mock_get_spark.return_value = mock_spark

        writer = DeltaWriter(table_name="cat.schema.table", cluster_columns=None)
        writer._spark = mock_spark

        writer.finalize()

        mock_spark.sql.assert_not_called()


# ---------------------------------------------------------------------------
# TestStacPureFunctions
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_pystac, reason="pystac not installed")
class TestStacPureFunctions:
    """Test STAC helper pure functions."""

    def test_ensure_utc_none_returns_datetime_with_tz(self) -> None:
        """_ensure_utc(None) returns a datetime with tzinfo set."""
        result = _ensure_utc(None)

        assert isinstance(result, datetime.datetime)
        assert result.tzinfo is not None

    def test_ensure_utc_naive_gets_utc(self) -> None:
        """_ensure_utc(naive_datetime) returns datetime with UTC tzinfo."""
        naive = datetime.datetime(2024, 1, 15, 12, 0, 0)
        result = _ensure_utc(naive)

        assert result.tzinfo is not None
        assert result.tzinfo == datetime.UTC

    def test_ensure_utc_aware_unchanged(self) -> None:
        """_ensure_utc(aware_datetime) returns it unchanged."""
        aware = datetime.datetime(2024, 1, 15, 12, 0, 0, tzinfo=datetime.UTC)
        result = _ensure_utc(aware)

        assert result is aware

    def test_resolve_image_path_absolute_stays(self) -> None:
        """Absolute image_path is returned unchanged."""
        row = {"image_path": "/abs/path.tif"}
        result = _resolve_image_path(row, "id", "/cog_base")

        assert result == "/abs/path.tif"

    def test_resolve_image_path_relative_uses_base(self) -> None:
        """Relative image_path uses cog_base_path + item_id."""
        row = {"image_path": "relative.tif"}
        result = _resolve_image_path(row, "img01", "/cog_base")

        assert result == "/cog_base/img01.tif"


# ---------------------------------------------------------------------------
# TestSortByTileEdgeCases
# ---------------------------------------------------------------------------


class TestSortByTileEdgeCases:
    """Edge cases for sort_by_tile."""

    def test_empty_dataframe_no_crash(self) -> None:
        """Sorting an empty DataFrame returns empty without crashing."""
        df = pd.DataFrame({"parent_path": [], "col_off": [], "row_off": []})
        result = sort_by_tile(df)

        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_input_not_mutated(self) -> None:
        """Original DataFrame is NOT modified after sorting."""
        df = pd.DataFrame({
            "parent_path": ["/a.tif", "/a.tif", "/b.tif"],
            "col_off": [512, 0, 256],
            "row_off": [512, 0, 256],
        })
        original = df.copy()

        sort_by_tile(df)

        pd.testing.assert_frame_equal(df, original)

    def test_idempotency(self) -> None:
        """Sorting an already-sorted DataFrame produces the same result."""
        df = pd.DataFrame({
            "parent_path": ["/a.tif", "/a.tif", "/a.tif", "/b.tif"],
            "col_off": [0, 224, 512, 0],
            "row_off": [0, 0, 512, 0],
        })
        first_sort = sort_by_tile(df)
        second_sort = sort_by_tile(first_sort)

        pd.testing.assert_frame_equal(first_sort, second_sort)


# ---------------------------------------------------------------------------
# TestStrideServicePropertyBased
# ---------------------------------------------------------------------------


class TestStrideServicePropertyBased:
    """Property-based tests for StrideService with many dimensions."""

    @pytest.mark.parametrize(
        "dimension",
        [224, 225, 300, 500, 700, 997, 1000, 2000, 4000, 8000],
    )
    def test_full_coverage_invariant(self, dimension: int) -> None:
        """last_start + chip_size == dimension (full coverage)."""
        chip_size = 224
        if dimension < chip_size:
            return

        overlap_x, _ = StrideService.calculate_optimal_overlap(
            (dimension, dimension), (chip_size, chip_size)
        )
        stride = chip_size - overlap_x
        assert stride > 0, f"stride must be positive for dimension={dimension}"

        starts = list(range(0, dimension - chip_size + 1, stride))
        last_start = starts[-1]

        assert last_start + chip_size == dimension, (
            f"Gap detected: last_start={last_start}, chip_size={chip_size}, dimension={dimension}"
        )

    def test_prime_dimension_overlap_non_negative(self) -> None:
        """For prime dimension (997), overlap must be non-negative and stride > 0."""
        chip_size = 224
        overlap_x, overlap_y = StrideService.calculate_optimal_overlap(
            (997, 997), (chip_size, chip_size)
        )

        assert overlap_x >= 0
        assert overlap_y >= 0

        stride_x = chip_size - overlap_x
        stride_y = chip_size - overlap_y
        assert stride_x > 0
        assert stride_y > 0
