"""Domain services for chipping logic — shared between all backends."""

import math


class StrideService:
    """
    Shared domain logic for calculating chip overlaps.

    Ensures chips cover the entire source image without gaps at the edges.
    Instead of padding (which adds fake data) or dropping edges (which loses data),
    this calculates the precise overlap so that the last chip aligns perfectly
    with the image edge.
    """

    @staticmethod
    def calculate_optimal_overlap(
        image_dims: tuple[int, int], chip_dims: tuple[int, int]
    ) -> tuple[int, int]:
        """
        Calculate the minimal overlap required for full image coverage.

        The algorithm determines the largest integer stride (step size) such that
        an integer number of steps exactly spans the remaining image dimension.

        Formula: Overlap = Chip_Size - Stride

        Args:
            image_dims: Dimensions of the source image (Width, Height).
            chip_dims: Dimensions of the desired chip (Width, Height).

        Returns:
            Tuple of (overlap_x, overlap_y) in pixels.
        """
        w, h = image_dims
        cw, ch = chip_dims

        def _get_stride(total_dim: int, chip_dim: int) -> int:
            target_start = total_dim - chip_dim
            if target_start <= 0:
                return 0

            steps = math.ceil(target_start / chip_dim)
            for i in range(steps, target_start + 1):
                stride = target_start / i
                if stride.is_integer():
                    return chip_dim - int(stride)
            return chip_dim

        return _get_stride(w, cw), _get_stride(h, ch)
