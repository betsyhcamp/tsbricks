from __future__ import annotations
import pytest  # noqa: F401
from typing import Any
import pandas as pd
import pyarrow as pa
from pytest_mock import MockerFixture
import numpy as np
import copy

from google.cloud import bigquery
from google.cloud.bigquery.job import QueryJob
from google.cloud.bigquery.table import RowIterator


@pytest.fixture
def mock_row_iterator(mocker: MockerFixture) -> Any:
    """Create mock RowIterator"""
    result = mocker.MagicMock(spec=RowIterator)
    result.total_rows = 100
    return result


@pytest.fixture
def mock_query_job(mocker: MockerFixture, mock_row_iterator: Any) -> Any:
    """
    Create mock QueryJob, wired to row iterator. Using `Any`
    to avoid importing unittest.mock for return type
    """
    job = mocker.MagicMock(spec=QueryJob)
    job.job_id = "test-job-123"
    job.total_bytes_processed = 1024
    job.total_bytes_billed = 2048
    job.cache_hit = False
    job.result.return_value = mock_row_iterator
    return job


@pytest.fixture
def mock_bq_client(mocker: MockerFixture, mock_query_job: Any) -> Any:
    """
    Create mock BigQuery client, wired to job. Using `Any`
    to avoid importing unittest.mock for return type
    """
    client = mocker.MagicMock(spec=bigquery.Client)
    client.query.return_value = mock_query_job
    return client


@pytest.fixture
def sample_pandas_df() -> pd.DataFrame:
    """Sample Pandas DataFrame for testing."""
    return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})


@pytest.fixture
def sample_arrow_table() -> pa.Table:
    """Sample Arrow table for testing."""
    return pa.table({"col1": [1, 2, 4], "col2": ["a", "b", "c"]})


# ------- Fixtures for testing backtest config Pydantic schemas ---------
@pytest.fixture
def valid_cfg() -> dict:
    """Minimal valid backtest config dict to reuse across tests."""
    return {
        "data": {"freq": "MS"},
        "cross_validation": {
            "mode": "explicit",
            "horizon": 6,
            "forecast_origins": ["2023-01-01", "2023-07-01"],
        },
        "transforms": [
            {
                "name": "box_cox",
                "class": "tsbricks.blocks.transforms.BoxCoxTransform",
                "scope": "per_series",
                "targets": ["y"],
                "perform_inverse_transform": True,
                "params": {"method": "guerrero", "season_length": 12},
            }
        ],
        "model": {
            "callable": "tsbricks._testing.dummy_models.forecast_only",
            "hyperparameters": {},
        },
        "metrics": {
            "definitions": [
                {
                    "name": "rmse",
                    "callable": "tsbricks.blocks.metrics.rmse",
                    "type": "simple",
                    "scope": "per_series",
                    "aggregation": "per_fold_mean",
                }
            ],
        },
    }


# support deletions in tests
DEL = object()


@pytest.fixture
def make_cfg(valid_cfg):
    """
    Deep-merge semantics:
      • dict+dict -> merge keys recursively
      • override with {} -> replace destination dict with {}
      • override with list/str/int/bool/None -> replace
      • override with DEL -> delete the key if present
    """

    def _deep_merge(base: dict, patch: dict) -> dict:
        out = copy.deepcopy(base)
        for k, v in patch.items():
            if v is DEL:
                out.pop(k, None)
                continue

            if isinstance(v, dict) and isinstance(out.get(k), dict):
                if not v:
                    out[k] = {}  # explicit clear
                else:
                    out[k] = _deep_merge(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
        return out

    def _factory(overrides: dict | None = None) -> dict:
        return _deep_merge(valid_cfg, overrides or {})

    return _factory


# ---- Fixtures for testing polars and/or pandas dataframes in checks/formatting -------


def _make_df(module_name, cols=("unique_id", "ds", "y"), rows=2):
    """Return a small DataFrame using either pandas or polars specifed as `module_name`."""
    data = {c: list(range(1, rows + 1)) for c in cols}
    return module_name.DataFrame(data)


@pytest.fixture
def pd_df():
    """Pandas dataframe factory fixture to create pandas dataframes"""
    pd = pytest.importorskip("pandas", reason="pandas is required for tests")

    def factory(cols=("unique_id", "ds", "y"), rows=2):
        return _make_df(pd, cols, rows)

    return factory


@pytest.fixture
def pl_df():
    """Polars dataframe factory fixture to create polars dataframes"""
    pl = pytest.importorskip("polars", reason="polars needed for tests")

    def factory(cols=("unique_id", "ds", "y"), rows=2):
        return _make_df(pl, cols, rows)

    return factory


@pytest.fixture
def pd_df_unsorted():
    """Unsorted pandas dataframe ot test ordering, indexing behavior"""
    pd = pytest.importorskip("pandas", reason="pandas is required for tests")

    def factory():
        return pd.DataFrame(
            {
                "unique_id": ["B", "A", "A", "B", "A"],
                "ds": [2, 3, 1, 1, 2],
                "y": [10, 11, 2, 30, 50],
            }
        )

    return factory


@pytest.fixture
def pl_df_unsorted_with_nulls():
    """Unsorted polars dataframe ot test ordering, indexing behavior"""
    pl = pytest.importorskip("polars", reason="polars is required for tests")

    def factory():
        return pl.DataFrame(
            # note that many forecasting algos cannot tolerate nulls, should have seperate check
            {
                "unique_id": ["B", "A", "A", "B", "A", "B"],
                "ds": [2, 3, 1, 1, None, 3],
                "y": [10, 11, 2, 30, 50, None],
            }
        )

    return factory


# ---- core constants / helpers for metrics.py
@pytest.fixture(scope="session")
def small_num_bound():
    """
    Source the guard from the module-under-test if available; otherwise
    fall back to the formula to avoid test import failures during refactors.
    """
    try:
        from tsbricks.blocks.metrics import _SMALL_NUM_BOUND

        return _SMALL_NUM_BOUND
    except ImportError:
        print(
            "Cannot import _SMALL_NUM_BOUND, using fallback small number test fixture"
        )
        # Fallback value; update if code changes
        return max(np.finfo(float).tiny, 1e-12)


@pytest.fixture(scope="session")
def eps_tiny(small_num_bound):
    # build an epsilon well below the guard so "near-zero" diffs trip unstable
    return small_num_bound * 1e-6


@pytest.fixture
def arr():
    """Factory to create float numpy arrays (keeps dtype uniform across tests)."""

    def _mk(x):
        return np.asarray(x, dtype=float)

    return _mk


# ---- canonical series for metrics.py
@pytest.fixture
def y_train_unit_diffs(arr):
    """0,1,2,3,... -> m=1 diffs are all 1 (stable, scale=1 for RMSSE; meanabs=1)."""
    return arr([0, 1, 2, 3])


@pytest.fixture
def y_train_double_diffs(arr):
    """0,2,4 -> m=1 diffs are all 2 (RMS=2, meanabs=2)."""
    return arr([0, 2, 4])


@pytest.fixture
def y_train_constant(arr):
    """Constant series -> zero diffs -> unstable scale."""
    return arr([5, 5, 5, 5])


@pytest.fixture
def y_train_too_short(arr):
    """len <= m -> invalid denominator."""
    return arr([7])  # for m=1 this is invalid


@pytest.fixture
def y_true_zero_error(arr):
    """Matches y_pred -> zero numerator."""
    return arr([1, 2, 3])


@pytest.fixture
def y_pred_zero_error(arr):
    return arr([1, 2, 3])


@pytest.fixture
def y_true_simple(arr):
    """Used in several tests where mean error or MSE is known."""
    return arr([2, 4])


@pytest.fixture
def y_pred_simple(arr):
    return arr([1, 3])


@pytest.fixture
def y_true_nonfinite(arr):
    return arr([1.0, np.nan, 3.0])


@pytest.fixture
def y_pred_nonfinite_pair(arr):
    return arr([1.0, 2.0, 3.0])


@pytest.fixture
def y_train_near_zero_diffs(arr, eps_tiny):
    """Tiny diffs << SMALL_NUM_BOUND -> unstable by guard."""
    return arr([0.0, eps_tiny, 2 * eps_tiny, 3 * eps_tiny])


@pytest.fixture
def fallback_scale_unit():
    return 1.0


@pytest.fixture
def y_true_shape_2d():
    # triggers shape check ValueError
    return np.array([[1.0, 2.0, 3.0]])


@pytest.fixture
def y_pred_shape_2d():
    return np.array([[1.0, 2.0, 3.0]])


@pytest.fixture
def y_pred_over(arr):
    """Overpredicts y_true_simple by +1 each → positive bias."""
    return arr([3, 5])


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
