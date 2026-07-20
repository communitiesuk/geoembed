"""Abstract base class for embedding writers."""

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingWriter(ABC):
    """
    Abstract interface for persisting embedding results.

    Implementations handle different storage backends (Parquet, Delta Lake, etc.)
    while presenting a uniform interface to the pipeline.
    """

    @abstractmethod
    def write_batch(self, chip_ids: list[str], embeddings: np.ndarray) -> None:
        """
        Write a batch of embeddings to storage.

        Args:
            chip_ids: List of chip identifiers.
            embeddings: Array of shape (N, D) where N = len(chip_ids) and D = embedding dim.
        """
        ...

    @abstractmethod
    def finalize(self) -> None:
        """
        Perform any cleanup or optimisation after all batches have been written.

        E.g., Z-ordering a Delta table, closing file handles, flushing buffers.
        """
        ...
