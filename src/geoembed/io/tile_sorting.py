"""Tile-aligned sorting for optimal COG I/O throughput.

Based on findings from Microsoft's "Optimizing Cloud-to-GPU Throughput for Deep
Learning With Earth Observation Data" (Zaytar et al., 2025):

- A misaligned window read forces decompression of up to 4x more tiles than needed
- Block-aligned reads decompress exactly one tile per chip (when chip <= tile)
- Sorting chips by parent tile groups reads that share the same decompressed data

For our 224px chips on 512px COG tiles: each tile contains up to 4 chips.
Grouping reads by parent tile means the tile is decompressed once and reused
for all chips within it, rather than re-opened and re-decompressed per chip.
"""

from __future__ import annotations

import pandas as pd


def compute_tile_key(col_off: int, row_off: int, block_size: int = 512) -> tuple[int, int]:
    """
    Compute which COG tile a chip falls into.

    Args:
        col_off: Chip column offset in pixels.
        row_off: Chip row offset in pixels.
        block_size: COG internal tile size (default 512).

    Returns:
        Tuple of (tile_col, tile_row) indices.
    """
    return col_off // block_size, row_off // block_size


def sort_by_tile(metadata: pd.DataFrame, block_size: int = 512) -> pd.DataFrame:
    """
    Sort chip metadata for optimal COG tile-aligned I/O.

    Groups chips by parent image, then by which internal COG tile they fall into.
    This ensures that when the DataLoader reads sequentially, chips from the same
    tile (and same parent) are adjacent — minimising redundant tile decompression.

    Sort order: parent_path → tile_row → tile_col → row_off → col_off

    This matches the rasterio sequential read pattern (row-major within a tile)
    and ensures the GDAL block cache serves multiple chips per decompression.

    Args:
        metadata: DataFrame with columns: parent_path, col_off, row_off.
        block_size: COG internal tile size (default 512px, matching our COG converter).

    Returns:
        Sorted DataFrame (new index, original order lost).
    """
    df = metadata.copy()
    df["_tile_col"] = df["col_off"] // block_size
    df["_tile_row"] = df["row_off"] // block_size

    df = df.sort_values(
        by=["parent_path", "_tile_row", "_tile_col", "row_off", "col_off"],
        ignore_index=True,
    )

    df = df.drop(columns=["_tile_col", "_tile_row"])
    return df
