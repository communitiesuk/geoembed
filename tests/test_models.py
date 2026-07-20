"""Tests for the models module — registry, embedder interface, GPU worker logic."""

import pytest

torch = pytest.importorskip("torch")

from geoembed.chipping.services import StrideService  # noqa: E402
from geoembed.models.base import Embedder  # noqa: E402
from geoembed.models.registry import get_embedder, register_model  # noqa: E402


class DummyEmbedder(Embedder):
    """Test embedder returning deterministic embeddings."""

    @property
    def embedding_dim(self) -> int:
        return 64

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        b = batch.shape[0]
        return torch.ones(b, self.embedding_dim)


class TestModelRegistry:
    """Tests for the model registry."""

    def test_register_and_retrieve(self):
        register_model("test_dummy", DummyEmbedder)
        embedder = get_embedder("test_dummy")
        assert isinstance(embedder, DummyEmbedder)
        assert embedder.embedding_dim == 64

    def test_case_insensitive_lookup(self):
        register_model("CaseMixed", DummyEmbedder)
        embedder = get_embedder("casemixed")
        assert isinstance(embedder, DummyEmbedder)

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            get_embedder("nonexistent_model_xyz")

    def test_register_non_embedder_raises(self):
        with pytest.raises(TypeError, match="Expected a subclass of Embedder"):
            register_model("bad", str)  # type: ignore[arg-type]

    def test_dofa_is_registered_by_default(self):
        from geoembed.models.registry import _REGISTRY

        assert "dofa" in _REGISTRY


class TestEmbedderInterface:
    """Tests for the Embedder ABC."""

    def test_device_auto_detection(self):
        embedder = DummyEmbedder()
        assert embedder.device in ("cuda", "mps", "cpu")

    def test_explicit_device(self):
        embedder = DummyEmbedder(device="cpu")
        assert embedder.device == "cpu"

    def test_embed_output_shape(self):
        embedder = DummyEmbedder(device="cpu")
        batch = torch.randn(8, 3, 224, 224)
        result = embedder.embed(batch)
        assert result.shape == (8, 64)

    def test_embed_single_image(self):
        embedder = DummyEmbedder(device="cpu")
        batch = torch.randn(1, 3, 224, 224)
        result = embedder.embed(batch)
        assert result.shape == (1, 64)


class TestGPUTilingLogic:
    """Tests for the torch.unfold tiling logic used in the GPU worker."""

    def test_unfold_produces_correct_chip_count(self):
        """8000x8000 image with chip_size=224 should produce 1369 patches."""
        image = torch.randn(3, 8000, 8000)
        chip_size = 224

        overlap_x, overlap_y = StrideService.calculate_optimal_overlap(
            (8000, 8000), (chip_size, chip_size)
        )
        stride_x = chip_size - overlap_x
        stride_y = chip_size - overlap_y

        patches = image.unfold(1, chip_size, stride_y).unfold(2, chip_size, stride_x)
        n_rows, n_cols = patches.shape[1], patches.shape[2]
        total = n_rows * n_cols

        assert total == 1369
        assert n_rows == 37
        assert n_cols == 37

    def test_unfold_patch_dimensions(self):
        """Each patch should be (3, 224, 224)."""
        image = torch.randn(3, 448, 448)
        chip_size = 224

        overlap_x, overlap_y = StrideService.calculate_optimal_overlap(
            (448, 448), (chip_size, chip_size)
        )
        stride_x = chip_size - overlap_x
        stride_y = chip_size - overlap_y

        patches = image.unfold(1, chip_size, stride_y).unfold(2, chip_size, stride_x)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, 3, chip_size, chip_size)

        assert patches.shape[1] == 3
        assert patches.shape[2] == 224
        assert patches.shape[3] == 224

    def test_unfold_pixel_correctness(self):
        """First patch should contain the same pixels as the top-left 224x224 of the image."""
        image = torch.arange(3 * 448 * 448, dtype=torch.float32).reshape(3, 448, 448)
        chip_size = 224

        patches = image.unfold(1, chip_size, 224).unfold(2, chip_size, 224)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, 3, chip_size, chip_size)

        # First patch should be top-left corner
        expected = image[:, :224, :224]
        assert torch.equal(patches[0], expected)

    def test_unfold_handles_non_square(self):
        """Non-square images should tile correctly."""
        image = torch.randn(3, 4000, 8000)
        chip_size = 224

        overlap_x, overlap_y = StrideService.calculate_optimal_overlap(
            (8000, 4000), (chip_size, chip_size)
        )
        stride_x = chip_size - overlap_x
        stride_y = chip_size - overlap_y

        patches = image.unfold(1, chip_size, stride_y).unfold(2, chip_size, stride_x)
        n_rows, n_cols = patches.shape[1], patches.shape[2]

        # Should have fewer rows than cols
        assert n_cols > n_rows
        assert n_rows > 0
        assert n_cols > 0


class TestDofaEmbedder:
    """Tests for DofaEmbedder (requires torchgeo)."""

    def test_embedding_dim(self):
        from geoembed.models.dofa import DofaEmbedder

        embedder = DofaEmbedder(device="cpu", mixed_precision=False)
        assert embedder.embedding_dim == 768

    def test_embed_output_shape(self):
        from geoembed.models.dofa import DofaEmbedder

        embedder = DofaEmbedder(device="cpu", mixed_precision=False)
        batch = torch.randn(2, 3, 224, 224)
        with torch.inference_mode():
            result = embedder.embed(batch)
        assert result.shape == (2, 768)

    def test_custom_wavelengths(self):
        from geoembed.models.dofa import DofaEmbedder

        wavelengths = [0.490, 0.560, 0.665]  # Sentinel-2 RGB
        embedder = DofaEmbedder(device="cpu", wavelengths=wavelengths)
        assert embedder.wavelengths == wavelengths

    def test_invalid_weights_raises(self):
        from geoembed.models.dofa import DofaEmbedder

        with pytest.raises(ValueError, match="Invalid DOFA weight name"):
            DofaEmbedder(weights="NONEXISTENT_WEIGHTS", device="cpu")
