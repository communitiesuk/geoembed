"""Core type definitions for the geoembed pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChipMetadata:
    """
    Metadata for a single image chip (virtual — no pixel data).

    This is the standard input format for the embedding pipeline. It contains
    enough information to locate and read the chip from its parent COG.

    Attributes:
        chip_id: Unique identifier for this chip.
        parent_path: Path to the parent COG image.
        col_off: Column offset (pixels) within the parent image.
        row_off: Row offset (pixels) within the parent image.
        width: Chip width in pixels.
        height: Chip height in pixels.
    """

    chip_id: str
    parent_path: str
    col_off: int
    row_off: int
    width: int
    height: int


@dataclass(frozen=True)
class EmbeddingResult:
    """
    Output of the embedding pipeline for a single chip.

    Attributes:
        chip_id: The chip this embedding was generated from.
        embedding: The embedding vector (typically 768-dim for ViT-Base models).
    """

    chip_id: str
    embedding: list[float]
