"""Abstract base class for all embedding models."""

from abc import ABC, abstractmethod

import torch

from geoembed.core.config import get_default_device


class Embedder(ABC):
    """
    Abstract base class for all embedding models.

    Ensures a consistent interface for the pipeline regardless of the underlying
    neural network architecture (e.g., ResNet, DOFA, Clay).

    Subclasses must implement the `embed` method with model-specific inference logic.
    """

    def __init__(self, device: str | None = None):
        """
        Initialise the embedder, automatically selecting the best
        available hardware (CUDA, MPS or CPU) if no device is specified.

        Args:
            device: Explicit device string (e.g., "cuda:0"). If None, auto-detects.
        """
        self.device = device or get_default_device()

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vector."""
        ...

    @abstractmethod
    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Run the model to generate embeddings.

        Args:
            batch: A tensor of image chips. Shape (B, C, H, W).
                   B = Batch Size
                   C = Channels (typically 3 for RGB)
                   H, W = Height, Width (typically 224)

        Returns:
            A tensor of embeddings. Shape (B, D).
            D = Embedding Dimension (e.g., 768 for ViT-Base).
        """
        ...
