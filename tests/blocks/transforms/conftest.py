"""Fixtures for blocks.transforms tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def boxcox_panel_df() -> pd.DataFrame:
    """Synthetic panel with 2 series that have different distributions.

    Series A: linear ramp (values 10..19) — roughly log-linear.
    Series B: quadratic growth (values 100..900) — needs a stronger transform.
    Both are strictly positive, suitable for Box-Cox.
    """
    n = 10
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
            "y": np.arange(1, 1 + n, dtype=float) ** 2 * 100,
        }
    )
    return pd.concat([df_a, df_b], ignore_index=True)


@pytest.fixture
def workday_panel_df() -> pd.DataFrame:
    """Synthetic panel with 2 series and 4 monthly periods."""
    dates = pd.date_range("2024-01-01", periods=4, freq="MS")
    df_a = pd.DataFrame(
        {
            "unique_id": "A",
            "ds": dates,
            "y": [1000.0, 2000.0, 3000.0, 4000.0],
        }
    )
    df_b = pd.DataFrame(
        {
            "unique_id": "B",
            "ds": dates,
            "y": [500.0, 600.0, 700.0, 800.0],
        }
    )
    return pd.concat([df_a, df_b], ignore_index=True)


@pytest.fixture
def global_calendar() -> pd.DataFrame:
    """Global calendar with different n_workdays per month."""
    dates = pd.date_range("2024-01-01", periods=4, freq="MS")
    return pd.DataFrame(
        {
            "ds": dates,
            "n_workdays": [20.0, 19.0, 21.0, 22.0],
        }
    )


@pytest.fixture
def per_series_calendar() -> pd.DataFrame:
    """Per-series calendar where A and B have different n_workdays per month."""
    dates = pd.date_range("2024-01-01", periods=4, freq="MS")
    rows_a = pd.DataFrame(
        {
            "unique_id": "A",
            "ds": dates,
            "n_workdays": [20.0, 19.0, 21.0, 22.0],
        }
    )
    rows_b = pd.DataFrame(
        {
            "unique_id": "B",
            "ds": dates,
            "n_workdays": [18.0, 17.0, 19.0, 20.0],
        }
    )
    return pd.concat([rows_a, rows_b], ignore_index=True)
