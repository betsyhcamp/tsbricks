"""Fixtures for runner submodules."""

import pandas as pd
import numpy as np
import pytest

# ---- Fixtures ----


@pytest.fixture
def panel_df() -> pd.DataFrame:
    """Synthetic panel with 2 series, strictly positive values."""
    n = 12
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    df_a = pd.DataFrame(
        {
            "unique_id": "A",
            "ds": dates,
            "y": np.arange(10, 10 + n, dtype=float),
        }
    )
    df_b = pd.DataFrame(
        {
            "unique_id": "B",
            "ds": dates,
            "y": np.arange(1, 1 + n, dtype=float) ** 2 * 50,
        }
    )
    return pd.concat([df_a, df_b], ignore_index=True)
