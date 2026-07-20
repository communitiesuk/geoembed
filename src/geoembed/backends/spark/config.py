"""Configuration for the full-image GPU embedding pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingsConfig:
    """
    Configuration for the full-image GPU embedding pipeline.

    The pipeline reads entire COGs, tiles on GPU via torch.unfold(), and runs
    batched inference — no per-chip I/O.

    Attributes:
        chip_size: Square chip dimension in pixels (224 for DOFA).
        batch_size: Inference batch size within each image (tune for VRAM).
        mixed_precision: Use FP16 autocasting for faster inference.
        model_name: Registered model name (default "dofa").
    """

    chip_size: int = 224
    batch_size: int = 256
    mixed_precision: bool = True
    model_name: str = "dofa"
