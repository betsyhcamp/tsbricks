"""Shared fixtures for tsbricks.blocks tests."""

import subprocess

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fake_git_toplevel(tmp_path):
    """Mock subprocess.run that returns tmp_path as the git repo root."""

    def _run(cmd, **kwargs):
        result = subprocess.CompletedProcess(cmd, 0)
        result.stdout = str(tmp_path) + "\n"
        return result

    return _run


# ---- Fixtures for residual diagnostics (diagnostics.py) -------
@pytest.fixture
def diag_df():
    """Unsorted DataFrame with nonzero residual variance for diagnostics tests."""
    return pd.DataFrame(
        {
            "time": [3, 1, 2, 4, 5],
            "actual": [3.0, 1.5, 2.0, 4.5, 5.0],
            "fitted": [2.5, 1.0, 2.5, 4.0, 5.5],
        }
    )


# ---- Fixtures for ACF/PACF plotting (diagnostics.py) -------
@pytest.fixture
def acf_df_datetime():
    """DataFrame with datetime time_col for ACF/PACF tests."""
    return pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=10, freq="D"),
            "value": [1.0, 2.5, 1.5, 3.0, 2.0, 1.8, 3.5, 2.2, 2.8, 1.2],
        }
    )


@pytest.fixture
def acf_df_integer():
    """DataFrame with integer time_col for ACF/PACF tests."""
    return pd.DataFrame(
        {
            "time": [5, 3, 1, 4, 2, 8, 6, 10, 9, 7],
            "value": [1.0, 2.5, 1.5, 3.0, 2.0, 1.8, 3.5, 1.2, 2.8, 2.2],
        }
    )


@pytest.fixture
def acf_result():
    """Prebuilt AcfResult for ACF/PACF plot-layer tests."""
    from tsbricks.blocks.diagnostics import AcfResult

    return AcfResult(
        lags=np.arange(4),
        values=np.array([1.0, 0.6, -0.2, 0.1]),
        ci_lower=np.array([-0.4, -0.4, -0.4, -0.4]),
        ci_upper=np.array([0.4, 0.4, 0.4, 0.4]),
    )


@pytest.fixture
def sample_diag_data():
    """Prebuilt ResidualDiagnostics for plot-layer tests."""
    from tsbricks.blocks.diagnostics import ResidualDiagnostics

    return ResidualDiagnostics(
        timestamps=np.arange(5),
        actual=np.array([1.5, 2.0, 3.0, 4.5, 5.0]),
        fitted=np.array([1.0, 2.5, 2.5, 4.0, 5.5]),
        residuals=np.array([0.5, -0.5, 0.5, 0.5, -0.5]),
        acf_values=np.array([1.0, 0.2, -0.1]),
        conf_interval=0.5,
        kde_x=np.linspace(-1, 1, 200),
        kde_y=np.abs(np.sin(np.linspace(0, np.pi, 200))),
    )
