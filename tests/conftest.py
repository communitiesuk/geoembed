"""Shared fixtures for the geoembed test suite."""

import pandas as pd
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring a GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests that are slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture
def sample_chip_metadata() -> pd.DataFrame:
    """Sample chip metadata DataFrame for testing tile sorting and pipelines."""
    return pd.DataFrame({
        "chip_id": ["A_000000", "A_000001", "A_000002", "B_000000", "B_000001"],
        "parent_path": ["/cogs/A.tif", "/cogs/A.tif", "/cogs/A.tif", "/cogs/B.tif", "/cogs/B.tif"],
        "parent_id": ["A", "A", "A", "B", "B"],
        "col_off": [0, 224, 512, 0, 224],
        "row_off": [0, 0, 512, 0, 0],
        "width": [224, 224, 224, 224, 224],
        "height": [224, 224, 224, 224, 224],
    })
