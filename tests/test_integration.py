"""Integration tests for geoembed: GPU tiling, stride service, Parquet writer, pipeline dispatch."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

torch = pytest.importorskip("torch")

from geoembed.chipping.services import StrideService  # noqa: E402
from geoembed.core.config import PipelineConfig  # noqa: E402
from geoembed.io.writers.parquet_writer import ParquetWriter  # noqa: E402
from geoembed.pipeline import EmbeddingsPipeline  # noqa: E402

_has_pystac = True
try:
    from geoembed.metadata.stac_catalog import sort_by_morton
except ImportError:
    _has_pystac = False
    sort_by_morton = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sliding_window_count(
    image_h: int, image_w: int, chip_size: int
) -> tuple[int, list[tuple[int, int]]]:
    """
    Replicate the sliding window offset calculation from SlidingWindowStrategy.

    Returns (patch_count, offsets_list).
    """
    overlap_x, overlap_y = StrideService.calculate_optimal_overlap(
        (image_w, image_h), (chip_size, chip_size)
    )
    stride_x = max(1, chip_size - overlap_x)
    stride_y = max(1, chip_size - overlap_y)

    row_starts = list(range(0, image_h - chip_size + 1, stride_y))
    col_starts = list(range(0, image_w - chip_size + 1, stride_x))

    offsets = [(col, row) for row in row_starts for col in col_starts]
    return len(offsets), offsets


def _tile_on_gpu_helper(
    image: torch.Tensor, chip_size: int
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """
    Replicate _tile_on_gpu logic from FullImageGPUWorker for testing.

    Args:
        image: (C, H, W) tensor.
        chip_size: Square chip dimension.

    Returns:
        (patches, offsets) where patches is (N, C, chip_size, chip_size).
    """
    c, h, w = image.shape

    overlap_x, overlap_y = StrideService.calculate_optimal_overlap((w, h), (chip_size, chip_size))
    stride_x = max(1, chip_size - overlap_x)
    stride_y = max(1, chip_size - overlap_y)

    patches = image.unfold(1, chip_size, stride_y).unfold(2, chip_size, stride_x)
    n_rows, n_cols = patches.shape[1], patches.shape[2]

    patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, c, chip_size, chip_size)

    offsets = [
        (c_idx * stride_x, r_idx * stride_y) for r_idx in range(n_rows) for c_idx in range(n_cols)
    ]

    return patches, offsets


# ---------------------------------------------------------------------------
# TestTileOnGpuPixelCorrectness
# ---------------------------------------------------------------------------


class TestTileOnGpuPixelCorrectness:
    """Verify torch.unfold pixel correctness for GPU tiling."""

    @pytest.fixture
    def image_448(self) -> torch.Tensor:
        """Deterministic 3x448x448 float tensor."""
        rng = np.random.default_rng(123)
        arr = rng.random((3, 448, 448), dtype=np.float32)
        return torch.from_numpy(arr)

    def test_first_patch_matches_top_left_region(self, image_448: torch.Tensor) -> None:
        """First patch should equal the top-left 224x224 region of the image."""
        chip_size = 224
        patches, offsets = _tile_on_gpu_helper(image_448, chip_size)

        # 448/224 = 2 in each dimension, so 2x2 = 4 patches
        assert patches.shape[0] == 4

        expected = image_448[:, :chip_size, :chip_size]
        torch.testing.assert_close(patches[0], expected)

    def test_last_patch_matches_bottom_right_region(self, image_448: torch.Tensor) -> None:
        """Last patch should equal the bottom-right 224x224 region."""
        chip_size = 224
        patches, offsets = _tile_on_gpu_helper(image_448, chip_size)

        expected = image_448[:, 224:448, 224:448]
        torch.testing.assert_close(patches[-1], expected)

    def test_offset_list_matches_patch_order(self, image_448: torch.Tensor) -> None:
        """For each offset, patches[i][:, 0, 0] == image[:, row_off, col_off]."""
        chip_size = 224
        patches, offsets = _tile_on_gpu_helper(image_448, chip_size)

        for i, (col_off, row_off) in enumerate(offsets):
            expected_pixel = image_448[:, row_off, col_off]
            actual_pixel = patches[i, :, 0, 0]
            torch.testing.assert_close(actual_pixel, expected_pixel)


# ---------------------------------------------------------------------------
# TestStrideServiceUnfoldConsistency
# ---------------------------------------------------------------------------


class TestStrideServiceUnfoldConsistency:
    """Verify sliding window count == unfold count for various image sizes."""

    @pytest.mark.parametrize(
        "h,w",
        [(448, 448), (8000, 8000), (1000, 1000), (500, 700)],
        ids=["448x448", "8000x8000", "1000x1000", "500x700"],
    )
    def test_unfold_count_matches_sliding_window_count(self, h: int, w: int) -> None:
        """Unfold patch count must equal sliding window offset count."""
        chip_size = 224
        sw_count, _ = _sliding_window_count(h, w, chip_size)

        # Create a minimal tensor and unfold to get patch count
        image = torch.zeros(3, h, w)
        patches, offsets = _tile_on_gpu_helper(image, chip_size)

        assert patches.shape[0] == sw_count

    @pytest.mark.parametrize(
        "h,w",
        [(448, 448), (8000, 8000), (1000, 1000), (500, 700)],
        ids=["448x448", "8000x8000", "1000x1000", "500x700"],
    )
    def test_offset_list_length_matches_patch_count(self, h: int, w: int) -> None:
        """Offset list length must match the number of patches produced."""
        chip_size = 224
        image = torch.zeros(3, h, w)
        patches, offsets = _tile_on_gpu_helper(image, chip_size)

        assert len(offsets) == patches.shape[0]


# ---------------------------------------------------------------------------
# TestParquetWriterValueRoundTrip
# ---------------------------------------------------------------------------


class TestParquetWriterValueRoundTrip:
    """Verify VALUES not just counts survive Parquet round-trip."""

    def test_embedding_values_survive_round_trip(self, tmp_path) -> None:
        """Written embedding values must exactly match when read back."""
        rng = np.random.default_rng(42)
        embeddings = rng.random((5, 768)).astype(np.float32)
        chip_ids = [f"chip_{i:04d}" for i in range(5)]

        out_path = tmp_path / "embeddings.parquet"
        writer = ParquetWriter(path=out_path)
        writer.write_batch(chip_ids, embeddings)
        writer.finalize()

        df = pd.read_parquet(out_path)
        assert len(df) == 5

        for i in range(5):
            stored = np.array(df.iloc[i]["embedding"], dtype=np.float32)
            np.testing.assert_array_equal(stored, embeddings[i])

    def test_chip_ids_preserved_in_order(self, tmp_path) -> None:
        """Chip IDs must be preserved in insertion order."""
        rng = np.random.default_rng(42)
        embeddings = rng.random((3, 128)).astype(np.float32)
        chip_ids = ["alpha", "beta", "gamma"]

        out_path = tmp_path / "embeddings.parquet"
        writer = ParquetWriter(path=out_path)
        writer.write_batch(chip_ids, embeddings)
        writer.finalize()

        df = pd.read_parquet(out_path)
        assert list(df["chip_id"]) == chip_ids

    def test_multi_batch_order_preserved(self, tmp_path) -> None:
        """Multiple write_batch calls must preserve order across batches."""
        rng = np.random.default_rng(42)
        emb1 = rng.random((2, 64)).astype(np.float32)
        emb2 = rng.random((3, 64)).astype(np.float32)
        ids1 = ["batch1_a", "batch1_b"]
        ids2 = ["batch2_a", "batch2_b", "batch2_c"]

        out_path = tmp_path / "embeddings.parquet"
        writer = ParquetWriter(path=out_path)
        writer.write_batch(ids1, emb1)
        writer.write_batch(ids2, emb2)
        writer.finalize()

        df = pd.read_parquet(out_path)
        assert len(df) == 5
        assert list(df["chip_id"]) == ids1 + ids2

        # Verify values from second batch
        for i in range(3):
            stored = np.array(df.iloc[2 + i]["embedding"], dtype=np.float32)
            np.testing.assert_array_equal(stored, emb2[i])


# ---------------------------------------------------------------------------
# TestPipelineDispatch
# ---------------------------------------------------------------------------


class TestPipelineDispatch:
    """Test pipeline.py routing to correct backends."""

    def test_spark_backend_dispatches(self) -> None:
        """Spark backend config must dispatch to _run_spark."""
        config = PipelineConfig(backend="spark")
        pipeline = EmbeddingsPipeline(config=config)

        with patch.object(pipeline, "_run_spark") as mock_spark:
            pipeline.run()
            mock_spark.assert_called_once()

    def test_local_backend_dispatches(self) -> None:
        """Local backend config must dispatch to _run_local."""
        config = PipelineConfig(backend="local")
        pipeline = EmbeddingsPipeline(config=config)

        with patch.object(pipeline, "_run_local") as mock_local:
            pipeline.run()
            mock_local.assert_called_once()

    def test_invalid_backend_raises_valueerror(self) -> None:
        """Invalid backend string must raise ValueError."""
        config = PipelineConfig.__new__(PipelineConfig)
        object.__setattr__(config, "backend", "invalid_backend")
        object.__setattr__(config, "batch_size", 128)
        object.__setattr__(config, "mixed_precision", True)
        object.__setattr__(config, "model", "dofa")
        object.__setattr__(config, "chip_size", 224)
        object.__setattr__(config, "device", None)
        object.__setattr__(config, "num_workers", 4)
        object.__setattr__(config, "prefetch_factor", 2)
        object.__setattr__(config, "persistent_workers", True)
        object.__setattr__(config, "tile_aligned", True)
        object.__setattr__(config, "input_storage", None)
        object.__setattr__(config, "output_storage", None)

        pipeline = EmbeddingsPipeline(config=config)

        with pytest.raises(ValueError, match="Unsupported backend"):
            pipeline.run()


# ---------------------------------------------------------------------------
# TestSortByMorton
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_pystac, reason="pystac not installed")
class TestSortByMorton:
    """Test stac_catalog.sort_by_morton correctness."""

    @pytest.fixture
    def bbox_table(self) -> pa.Table:
        """Create a table with random bboxes for morton sorting tests."""
        rng = np.random.default_rng(99)
        n = 50

        # Generate random centroids, derive bboxes of width/height 0.1 degrees
        cx = rng.uniform(-180, 180, size=n)
        cy = rng.uniform(-90, 90, size=n)
        half = 0.05

        xmin = (cx - half).tolist()
        xmax = (cx + half).tolist()
        ymin = (cy - half).tolist()
        ymax = (cy + half).tolist()

        bbox_struct = pa.StructArray.from_arrays(
            [
                pa.array(xmin, type=pa.float64()),
                pa.array(ymin, type=pa.float64()),
                pa.array(xmax, type=pa.float64()),
                pa.array(ymax, type=pa.float64()),
            ],
            names=["xmin", "ymin", "xmax", "ymax"],
        )

        return pa.table({
            "id": pa.array([f"item_{i}" for i in range(n)]),
            "bbox": bbox_struct,
        })

    def test_monotonic_morton_codes(self, bbox_table: pa.Table) -> None:
        """After sorting, morton codes must be non-decreasing."""
        sorted_table = sort_by_morton(bbox_table)

        # Recompute morton codes from sorted table to verify ordering
        bbox = sorted_table.column("bbox").combine_chunks()
        xmin = bbox.field("xmin").to_pylist()
        xmax = bbox.field("xmax").to_pylist()
        ymin = bbox.field("ymin").to_pylist()
        ymax = bbox.field("ymax").to_pylist()

        cx = [(a + b) / 2 for a, b in zip(xmin, xmax, strict=False)]
        cy = [(a + b) / 2 for a, b in zip(ymin, ymax, strict=False)]

        N = 65535

        def _clamp(v: float, lo: float, hi: float) -> int:
            return max(0, min(N, int((v - lo) / (hi - lo) * N)))

        def _morton(x: int, y: int) -> int:
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

        codes = [
            _morton(_clamp(x, -180, 180), _clamp(y, -90, 90)) for x, y in zip(cx, cy, strict=False)
        ]

        for i in range(len(codes) - 1):
            assert codes[i] <= codes[i + 1], (
                f"Morton codes not monotonic at index {i}: {codes[i]} > {codes[i + 1]}"
            )

    def test_preserves_row_count(self, bbox_table: pa.Table) -> None:
        """Sorting must not add or remove rows."""
        sorted_table = sort_by_morton(bbox_table)
        assert sorted_table.num_rows == bbox_table.num_rows
