"""Tests for metric evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsbricks.backtesting.evaluation import evaluate_metrics
from tsbricks.backtesting.schema import MetricDefinitionConfig, MetricsConfig


def _metrics_config(definitions: list[dict]) -> MetricsConfig:
    """Build a MetricsConfig from a list of definition dicts."""
    return MetricsConfig(definitions=[MetricDefinitionConfig(**d) for d in definitions])


RMSE_DEF = {
    "name": "rmse",
    "callable": "tsbricks.blocks.metrics.rmse",
    "type": "simple",
}

RMSSE_DEF = {
    "name": "rmsse",
    "callable": "tsbricks.blocks.metrics.rmsse",
    "type": "context_aware",
}


@pytest.fixture
def single_series_data():
    """Single series where RMSE is hand-computable.

    y_true=[2,4], y_pred=[1,3] → errors=[1,1] → MSE=1 → RMSE=1.0
    y_train=[0,1,2,3] → diffs=[1,1,1] → scale=1.0 → RMSSE=1.0
    """
    y_true = pd.DataFrame({"unique_id": ["A", "A"], "ds": [1, 2], "y": [2.0, 4.0]})
    y_pred = pd.DataFrame({"unique_id": ["A", "A"], "ds": [1, 2], "ypred": [1.0, 3.0]})
    y_train = pd.DataFrame(
        {
            "unique_id": ["A", "A", "A", "A"],
            "ds": [1, 2, 3, 4],
            "y": [0.0, 1.0, 2.0, 3.0],
        }
    )
    return y_true, y_pred, y_train


@pytest.fixture
def two_series_data():
    """Two-series panel. Series B has different errors.

    Series A: y_true=[2,4], y_pred=[1,3] → RMSE=1.0
    Series B: y_true=[10,20], y_pred=[10,20] → RMSE=0.0
    """
    y_true = pd.DataFrame(
        {
            "unique_id": ["A", "A", "B", "B"],
            "ds": [1, 2, 1, 2],
            "y": [2.0, 4.0, 10.0, 20.0],
        }
    )
    y_pred = pd.DataFrame(
        {
            "unique_id": ["A", "A", "B", "B"],
            "ds": [1, 2, 1, 2],
            "ypred": [1.0, 3.0, 10.0, 20.0],
        }
    )
    y_train = pd.DataFrame(
        {
            "unique_id": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "ds": [1, 2, 3, 4, 1, 2, 3, 4],
            "y": [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0],
        }
    )
    return y_true, y_pred, y_train


# ---- output schema ----


def test_output_has_expected_columns(single_series_data):
    """Output DataFrame has exactly the 5 spec columns."""
    y_true, y_pred, y_train = single_series_data
    result = evaluate_metrics(
        y_true, y_pred, y_train, _metrics_config([RMSE_DEF]), "fold_0"
    )

    assert list(result.columns) == [
        "metric_name",
        "unique_id",
        "fold",
        "aggregation",
        "value",
    ]


def test_fold_id_appears_in_output(single_series_data):
    """The fold_id parameter is reflected in every row's fold column."""
    y_true, y_pred, y_train = single_series_data
    result = evaluate_metrics(
        y_true, y_pred, y_train, _metrics_config([RMSE_DEF]), "fold_3"
    )

    assert (result["fold"] == "fold_3").all()


def test_aggregation_is_per_series(single_series_data):
    """All rows have aggregation='per_series' in V1."""
    y_true, y_pred, y_train = single_series_data
    result = evaluate_metrics(
        y_true, y_pred, y_train, _metrics_config([RMSE_DEF]), "fold_0"
    )

    assert (result["aggregation"] == "per_series").all()


# ---- simple metric values ----


def test_single_series_rmse_value(single_series_data):
    """RMSE for y_true=[2,4], y_pred=[1,3] is 1.0."""
    y_true, y_pred, y_train = single_series_data
    result = evaluate_metrics(
        y_true, y_pred, y_train, _metrics_config([RMSE_DEF]), "fold_0"
    )

    assert len(result) == 1
    assert result.iloc[0]["value"] == pytest.approx(1.0)


# ---- panel data ----


def test_panel_produces_row_per_series(two_series_data):
    """Two series × one metric = two rows."""
    y_true, y_pred, y_train = two_series_data
    result = evaluate_metrics(
        y_true, y_pred, y_train, _metrics_config([RMSE_DEF]), "fold_0"
    )

    assert len(result) == 2
    assert set(result["unique_id"]) == {"A", "B"}


def test_panel_series_values_are_independent(two_series_data):
    """Series A has RMSE=1.0, Series B has RMSE=0.0."""
    y_true, y_pred, y_train = two_series_data
    result = evaluate_metrics(
        y_true, y_pred, y_train, _metrics_config([RMSE_DEF]), "fold_0"
    )

    a_val = result.loc[result["unique_id"] == "A", "value"].iloc[0]
    b_val = result.loc[result["unique_id"] == "B", "value"].iloc[0]

    assert a_val == pytest.approx(1.0)
    assert b_val == pytest.approx(0.0)


# ---- context-aware metric ----


def test_context_aware_metric_returns_float(single_series_data):
    """RMSSE (context_aware, returns tuple) is handled and produces a float value."""
    y_true, y_pred, y_train = single_series_data
    result = evaluate_metrics(
        y_true, y_pred, y_train, _metrics_config([RMSSE_DEF]), "fold_0"
    )

    assert len(result) == 1
    assert np.isfinite(result.iloc[0]["value"])


def test_context_aware_rmsse_value(single_series_data):
    """RMSSE for known data: errors=[1,1], scale=1.0 → RMSSE=1.0."""
    y_true, y_pred, y_train = single_series_data
    result = evaluate_metrics(
        y_true, y_pred, y_train, _metrics_config([RMSSE_DEF]), "fold_0"
    )

    assert result.iloc[0]["value"] == pytest.approx(1.0)


# ---- multiple metrics ----


def test_multiple_metrics_produces_row_per_series_per_metric(single_series_data):
    """One series × two metrics = two rows."""
    y_true, y_pred, y_train = single_series_data
    result = evaluate_metrics(
        y_true, y_pred, y_train, _metrics_config([RMSE_DEF, RMSSE_DEF]), "fold_0"
    )

    assert len(result) == 2
    assert set(result["metric_name"]) == {"rmse", "rmsse"}
