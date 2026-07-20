"""Tests for the I/O module — tile sorting, writers, storage factory."""

import numpy as np
import pandas as pd
import pytest

from geoembed.core.config import StorageConfig
from geoembed.io.storage import create_writer
from geoembed.io.tile_sorting import compute_tile_key, sort_by_tile
from geoembed.io.writers.parquet_writer import ParquetWriter


class TestTileSorting:
    """Tests for tile-aligned sorting utilities."""

    @pytest.mark.parametrize(
        "col_off,row_off,block_size,expected",
        [
            (0, 0, 512, (0, 0)),
            (100, 200, 512, (0, 0)),
            (512, 0, 512, (1, 0)),
            (600, 100, 512, (1, 0)),
            (0, 512, 512, (0, 1)),
            (1024, 1024, 512, (2, 2)),
            (511, 511, 512, (0, 0)),
        ],
    )
    def test_compute_tile_key(self, col_off, row_off, block_size, expected):
        assert compute_tile_key(col_off, row_off, block_size) == expected

    def test_sort_by_tile_groups_by_parent(self, sample_chip_metadata):
        """Chips from same parent should be contiguous after sorting."""
        sorted_df = sort_by_tile(sample_chip_metadata)

        # All parent A chips should be adjacent
        a_mask = sorted_df["parent_path"] == "/cogs/A.tif"
        a_indices = sorted_df[a_mask].index.tolist()
        assert a_indices == list(range(a_indices[0], a_indices[0] + len(a_indices)))

    def test_sort_by_tile_preserves_columns(self, sample_chip_metadata):
        """No columns added or removed."""
        sorted_df = sort_by_tile(sample_chip_metadata)
        assert set(sorted_df.columns) == set(sample_chip_metadata.columns)
        assert "_tile_col" not in sorted_df.columns
        assert "_tile_row" not in sorted_df.columns

    def test_sort_by_tile_preserves_row_count(self, sample_chip_metadata):
        """Same number of rows in, same number out."""
        sorted_df = sort_by_tile(sample_chip_metadata)
        assert len(sorted_df) == len(sample_chip_metadata)

    def test_sort_by_tile_orders_within_parent(self):
        """Within a parent, chips should be ordered by tile_row then tile_col."""
        df = pd.DataFrame({
            "chip_id": ["c1", "c2", "c3", "c4"],
            "parent_path": ["/a.tif"] * 4,
            "col_off": [512, 0, 512, 0],  # tile cols: 1, 0, 1, 0
            "row_off": [512, 512, 0, 0],  # tile rows: 1, 1, 0, 0
        })
        sorted_df = sort_by_tile(df)
        # Expected order: tile(0,0), tile(1,0), tile(0,1), tile(1,1)
        assert sorted_df.iloc[0]["col_off"] == 0
        assert sorted_df.iloc[0]["row_off"] == 0

    def test_sort_by_tile_custom_block_size(self):
        """Different block sizes should change grouping."""
        df = pd.DataFrame({
            "chip_id": ["c1", "c2"],
            "parent_path": ["/a.tif", "/a.tif"],
            "col_off": [256, 0],
            "row_off": [0, 0],
        })
        # With block_size=512, both are in tile (0, 0) — order unchanged
        sort_by_tile(df, block_size=512)
        # With block_size=128, first is in tile (2, 0), second in tile (0, 0)
        sorted_128 = sort_by_tile(df, block_size=128)
        # With smaller blocks, order should change
        assert sorted_128.iloc[0]["col_off"] == 0


class TestParquetWriter:
    """Tests for ParquetWriter."""

    def test_write_and_finalize(self, tmp_path):
        path = tmp_path / "embeddings.parquet"
        writer = ParquetWriter(path=path, mode="overwrite")

        writer.write_batch(["chip_a", "chip_b"], np.random.randn(2, 768))
        writer.write_batch(["chip_c"], np.random.randn(1, 768))
        writer.finalize()

        assert path.exists()
        df = pd.read_parquet(path)
        assert len(df) == 3
        assert list(df.columns) == ["chip_id", "embedding"]
        # Verify chip_id values match what was written
        assert list(df["chip_id"]) == ["chip_a", "chip_b", "chip_c"]

    def test_empty_write_does_nothing(self, tmp_path):
        path = tmp_path / "empty.parquet"
        writer = ParquetWriter(path=path, mode="overwrite")
        writer.finalize()
        assert not path.exists()

    def test_append_mode(self, tmp_path):
        path = tmp_path / "append.parquet"

        # First write
        writer = ParquetWriter(path=path, mode="overwrite")
        writer.write_batch(["a"], np.random.randn(1, 768))
        writer.finalize()

        # Append
        writer2 = ParquetWriter(path=path, mode="append")
        writer2.write_batch(["b"], np.random.randn(1, 768))
        writer2.finalize()

        df = pd.read_parquet(path)
        assert len(df) == 2

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "out.parquet"
        writer = ParquetWriter(path=path, mode="overwrite")
        writer.write_batch(["x"], np.random.randn(1, 10))
        writer.finalize()
        assert path.exists()


class TestStorageFactory:
    """Tests for the create_writer factory function."""

    def test_creates_parquet_writer(self, tmp_path):
        config = StorageConfig(backend="parquet", path=str(tmp_path / "test.parquet"))
        writer = create_writer(config)
        assert isinstance(writer, ParquetWriter)

    def test_creates_delta_writer(self):
        from geoembed.io.writers.delta_writer import DeltaWriter

        config = StorageConfig(backend="delta", path="catalog.schema.table")
        writer = create_writer(config)
        assert isinstance(writer, DeltaWriter)

    def test_unsupported_backend_raises(self):
        config = StorageConfig(backend="parquet", path="x")
        # Manually override to test error path
        object.__setattr__(config, "backend", "unknown")
        with pytest.raises(ValueError, match="Unsupported storage backend"):
            create_writer(config)
