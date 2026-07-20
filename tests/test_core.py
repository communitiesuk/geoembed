"""Basic tests for geoembed core functionality."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from geoembed.chipping.services import StrideService  # noqa: E402
from geoembed.core.config import PipelineConfig, StorageConfig, get_default_device  # noqa: E402
from geoembed.core.types import ChipMetadata, EmbeddingResult  # noqa: E402
from geoembed.io.storage import create_writer  # noqa: E402
from geoembed.io.writers.parquet_writer import ParquetWriter  # noqa: E402
from geoembed.models.base import Embedder  # noqa: E402
from geoembed.models.registry import get_embedder, register_model  # noqa: E402


class DummyEmbedder(Embedder):
    """Test embedder that returns random embeddings."""

    @property
    def embedding_dim(self) -> int:
        return 128

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        b = batch.shape[0]
        return torch.randn(b, self.embedding_dim)


def test_storage_config_defaults():
    config = StorageConfig()
    assert config.backend == "parquet"
    assert config.mode == "append"
    assert config.store_type == "local"


def test_pipeline_config_defaults():
    config = PipelineConfig()
    assert config.model == "dofa"
    assert config.backend == "spark"
    assert config.chip_size == 224
    assert config.batch_size == 128
    assert config.mixed_precision is True
    # Microsoft paper-recommended defaults
    assert config.num_workers == 4
    assert config.prefetch_factor == 2
    assert config.persistent_workers is True
    assert config.tile_aligned is True


def test_get_default_device():
    device = get_default_device()
    assert device in ("cuda", "mps", "cpu")


def test_chip_metadata_frozen():
    chip = ChipMetadata(
        chip_id="test_000001",
        parent_path="/path/to/image.tif",
        col_off=0,
        row_off=0,
        width=224,
        height=224,
    )
    assert chip.chip_id == "test_000001"
    assert chip.width == 224

    # Verify immutability — frozen dataclass should reject attribute assignment
    with pytest.raises((AttributeError, TypeError)):
        chip.chip_id = "mutated"  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        chip.width = 999  # type: ignore[misc]


def test_embedding_result():
    result = EmbeddingResult(chip_id="test_000001", embedding=[0.1, 0.2, 0.3])
    assert len(result.embedding) == 3


def test_stride_service_basic():
    overlap = StrideService.calculate_optimal_overlap((8000, 8000), (224, 224))
    assert overlap == (8, 8)


def test_stride_service_no_overlap_needed():
    # Image exactly divisible by chip size
    overlap = StrideService.calculate_optimal_overlap((448, 448), (224, 224))
    assert overlap == (0, 0)


def test_stride_service_single_chip():
    # Image smaller than chip
    overlap = StrideService.calculate_optimal_overlap((100, 100), (224, 224))
    assert overlap == (0, 0)


def test_model_registry_builtin():
    from geoembed.models.registry import _REGISTRY

    assert "dofa" in _REGISTRY


def test_model_registry_custom():
    register_model("dummy", DummyEmbedder)
    embedder = get_embedder("dummy")
    assert isinstance(embedder, DummyEmbedder)
    assert embedder.embedding_dim == 128

    # Test inference
    batch = torch.randn(4, 3, 224, 224)
    result = embedder.embed(batch)
    assert result.shape == (4, 128)


def test_create_writer_parquet(tmp_path):
    config = StorageConfig(backend="parquet", path=str(tmp_path / "test_embeddings.parquet"))
    writer = create_writer(config)
    assert isinstance(writer, ParquetWriter)


def test_parquet_writer(tmp_path):
    output_path = tmp_path / "embeddings.parquet"
    writer = ParquetWriter(path=output_path, mode="overwrite")

    # Write two batches
    writer.write_batch(["chip_a", "chip_b"], np.random.randn(2, 768))
    writer.write_batch(["chip_c"], np.random.randn(1, 768))
    writer.finalize()

    assert output_path.exists()

    import pandas as pd

    df = pd.read_parquet(output_path)
    assert len(df) == 3
    assert list(df.columns) == ["chip_id", "embedding"]
    # Verify actual chip_id values match what was written
    assert list(df["chip_id"]) == ["chip_a", "chip_b", "chip_c"]


def test_quote_table_name():
    """Backtick-quoting for UC table names with special characters."""
    import warnings

    import pytest

    from geoembed.core.naming import quote_table_name

    # Three-part name (standard UC)
    assert quote_table_name("catalog-foo-bar.schema.table") == "`catalog-foo-bar`.`schema`.`table`"
    # Clean identifiers still get quoted
    result = quote_table_name("my_catalog.my_schema.my_table")
    assert result == "`my_catalog`.`my_schema`.`my_table`"
    # Two-part name warns but works
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert quote_table_name("schema.table") == "`schema`.`table`"
    # Single part raises
    with pytest.raises(ValueError, match="must have at least 2 parts"):
        quote_table_name("just_a_table")
    # Too many parts raises
    with pytest.raises(ValueError, match="has 4 parts"):
        quote_table_name("a.b.c.d")


def test_tile_sorting():
    """Verify tile sorting groups chips by parent and tile position."""
    import pandas as pd

    from geoembed.io.tile_sorting import compute_tile_key, sort_by_tile

    # Chips scattered across 2 parents, various tile positions
    metadata = pd.DataFrame({
        "chip_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
        "parent_path": ["/a.tif", "/b.tif", "/a.tif", "/b.tif", "/a.tif", "/a.tif"],
        "col_off": [600, 100, 100, 600, 100, 600],  # tile 1, tile 0, tile 0, tile 1, tile 0, tile 1
        "row_off": [0, 512, 0, 0, 512, 512],  # tile 0, tile 1, tile 0, tile 0, tile 1, tile 1
        "width": [224, 224, 224, 224, 224, 224],
        "height": [224, 224, 224, 224, 224, 224],
    })

    sorted_df = sort_by_tile(metadata, block_size=512)

    # After sorting: grouped by parent, then by tile_row, then tile_col
    # Parent /a.tif should come before /b.tif (alphabetical)
    # Within /a.tif: tile (0,0) chips before tile (1,0) before tile (0,1) etc.
    assert len(sorted_df) == 6
    # All /a.tif chips should be contiguous
    a_indices = sorted_df[sorted_df["parent_path"] == "/a.tif"].index.tolist()
    assert a_indices == list(range(a_indices[0], a_indices[0] + len(a_indices)))

    # Test compute_tile_key
    assert compute_tile_key(100, 200, block_size=512) == (0, 0)
    assert compute_tile_key(600, 100, block_size=512) == (1, 0)
    assert compute_tile_key(100, 600, block_size=512) == (0, 1)


def test_tile_sorting_preserves_columns():
    """Tile sorting should not add or remove user columns."""
    import pandas as pd

    from geoembed.io.tile_sorting import sort_by_tile

    metadata = pd.DataFrame({
        "chip_id": ["c1", "c2"],
        "parent_path": ["/a.tif", "/a.tif"],
        "col_off": [0, 512],
        "row_off": [0, 0],
        "width": [224, 224],
        "height": [224, 224],
        "extra_col": ["x", "y"],
    })

    sorted_df = sort_by_tile(metadata)
    assert "extra_col" in sorted_df.columns
    assert "_tile_col" not in sorted_df.columns
    assert "_tile_row" not in sorted_df.columns
