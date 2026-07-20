"""DOFA (Dynamic One-For-All) embedding model implementation."""

import contextlib
from typing import Any, cast

import torch
from torchgeo.models import get_model
from torchvision.models._api import Weights

from geoembed.models.base import Embedder


class DofaEmbedder(Embedder):
    """
    Implementation of the DOFA (Dynamic One-For-All) architecture.

    DOFA is a Vision Transformer (ViT) pre-trained on massive amounts of satellite data.
    It supports dynamic wavelength selection, allowing it to adapt to different
    sensor configurations (Sentinel-2, Landsat, aerial RGB, etc.).

    Default wavelengths are for RGB aerial imagery:
    - Red:   0.640 um
    - Green: 0.538 um
    - Blue:  0.467 um

    Output: 768-dimensional embedding per chip (ViT-Base backbone).
    """

    def __init__(
        self,
        weights: str | Weights = "DOFA_MAE",
        wavelengths: list[float] | None = None,
        mixed_precision: bool = True,
        **kwargs: Any,
    ):
        """
        Initialise the DOFA model.

        Args:
            weights: Name of the weights to load (e.g. "DOFA_MAE") or Weights object.
            wavelengths: Sensor wavelengths in micrometres. Defaults to RGB [0.640, 0.538, 0.467].
            mixed_precision: If True, uses FP16/BF16 during inference for speed and lower VRAM.
            **kwargs: Additional arguments passed to the base Embedder (e.g., device).
        """
        super().__init__(**kwargs)

        if isinstance(weights, str):
            from torchgeo.models import DOFABase16_Weights

            try:
                self.weights = getattr(DOFABase16_Weights, weights)
            except AttributeError as err:
                available = list(DOFABase16_Weights.__members__.keys())
                msg = f"Invalid DOFA weight name: '{weights}'. Available: {available}"
                raise ValueError(msg) from err
        else:
            self.weights = weights

        self.model: Any = get_model("dofa_base_patch16_224", weights=self.weights)
        self.model = self.model.eval().to(self.device)

        with contextlib.suppress(Exception):
            self.model = torch.compile(self.model, mode="default")

        self.wavelengths = wavelengths or [0.640, 0.538, 0.467]
        self.mixed_precision = mixed_precision
        print(f"DOFA model loaded on {self.device} using weights: {self.weights}")

    @property
    def embedding_dim(self) -> int:
        return 768

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Generate DOFA embeddings for a batch of images.

        Handles GPU transfer, float conversion, AMP context, and wavelength injection.

        Args:
            batch: Input tensor of shape (B, 3, 224, 224).

        Returns:
            Embedding tensor of shape (B, 768).
        """
        batch = batch.to(self.device, non_blocking=True)
        batch = batch.float()

        with torch.inference_mode(), torch.amp.autocast(self.device, enabled=self.mixed_precision):
            return cast(
                torch.Tensor, self.model.forward_features(batch, wavelengths=self.wavelengths)
            )
