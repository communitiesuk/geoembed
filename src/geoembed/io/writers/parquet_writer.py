"""Parquet-based embedding writer for local and cloud storage."""

from pathlib import Path

import numpy as np
import pandas as pd

from geoembed.io.writers.base import EmbeddingWriter


class ParquetWriter(EmbeddingWriter):
    """
    Writes embeddings to a local Parquet file.

    Accumulates batches in memory and writes once on `finalize()`, or appends
    to an existing file if `mode="append"`.

    Suitable for local development and small-to-medium datasets.
    """

    def __init__(self, path: str | Path, mode: str = "overwrite"):
        """
        Initialise the Parquet writer.

        Args:
            path: Output file path.
            mode: "overwrite" replaces the file, "append" adds to existing.
        """
        self.path = Path(path)
        self.mode = mode
        self._batches: list[pd.DataFrame] = []

    def write_batch(self, chip_ids: list[str], embeddings: np.ndarray) -> None:
        """Buffer a batch of embeddings."""
        df = pd.DataFrame({
            "chip_id": chip_ids,
            "embedding": list(embeddings),
        })
        self._batches.append(df)

    def finalize(self) -> None:
        """Write all buffered batches to the Parquet file."""
        if not self._batches:
            return

        combined = pd.concat(self._batches, ignore_index=True)

        if self.mode == "append" and self.path.exists():
            existing = pd.read_parquet(self.path)
            combined = pd.concat([existing, combined], ignore_index=True)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(self.path, index=False)
        print(f"Wrote {len(combined)} embeddings to {self.path}")
        self._batches.clear()
