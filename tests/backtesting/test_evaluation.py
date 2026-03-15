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


def _to_panel(arr, uid="A", col="y"):
    """Wrap a numpy array into a single-series DataFrame."""
    n = len(arr)
    return pd.DataFrame({"unique_id": [uid] * n, "ds": list(range(1, n + 1)), col: arr})


@pytest.fixture
def single_series_data(y_true_simple, y_pred_simple, y_train_unit_diffs):
    """Single series reusing root conftest canonical arrays.

    y_true=[2,4], y_pred=[1,3] → RMSE=1.0
    y_train=[0,1,2,3] → scale=1.0 → RMSSE=1.0
    """
    return (
        _to_panel(y_true_simple, col="y"),
        _to_panel(y_pred_simple, col="ypred"),
        _to_panel(y_train_unit_diffs, col="y"),
    )


@pytest.fixture
def two_series_data(y_true_simple, y_pred_simple, y_train_unit_diffs):
    """Two-series panel. Series A reuses root conftest arrays. Series B has zero error."""
    y_true_a = _to_panel(y_true_simple, uid="A", col="y")
    y_pred_a = _to_panel(y_pred_simple, uid="A", col="ypred")
    y_train_a = _to_panel(y_train_unit_diffs, uid="A", col="y")

    y_true_b = _to_panel([10.0, 20.0], uid="B", col="y")
    y_pred_b = _to_panel([10.0, 20.0], uid="B", col="ypred")
    y_train_b = _to_panel([5.0, 10.0, 15.0, 20.0], uid="B", col="y")

    return (
        pd.concat([y_true_a, y_true_b], ignore_index=True),
        pd.concat([y_pred_a, y_pred_b], ignore_index=True),
        pd.concat([y_train_a, y_train_b], ignore_index=True),
    )


# ---- output schema ----


def test_output_has_expected_columns(single_series_data):
    """Output DataFrame has exactly the 7 spec columns."""
    y_true, y_pred, y_train = single_series_data
    result = evaluate_metrics(
        y_true, y_pred, y_train, _metrics_config([RMSE_DEF]), "fold_0"
    )

    assert list(result.columns) == [
        "metric_name",
        "unique_id",
        "fold",
        "scope",
        "grouping_column_name",
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


def test_aggregation_is_per_fold_mean(single_series_data):
    """All rows have aggregation='per_fold_mean' by default."""
    y_true, y_pred, y_train = single_series_data
    result = evaluate_metrics(
        y_true, y_pred, y_train, _metrics_config([RMSE_DEF]), "fold_0"
    )

    assert (result["aggregation"] == "per_fold_mean").all()


def test_aggregation_pooled_propagates(single_series_data):
    """aggregation='pooled' on definition flows through to output rows."""
    y_true, y_pred, y_train = single_series_data
    pooled_def = {**RMSE_DEF, "aggregation": "pooled"}
    result = evaluate_metrics(
        y_true, y_pred, y_train, _metrics_config([pooled_def]), "fold_0"
    )

    assert (result["aggregation"] == "pooled").all()


def test_scope_is_per_series(single_series_data):
    """All rows have scope='per_series' for per-series metrics."""
    y_true, y_pred, y_train = single_series_data
    result = evaluate_metrics(
        y_true, y_pred, y_train, _metrics_config([RMSE_DEF]), "fold_0"
    )

    assert (result["scope"] == "per_series").all()


def test_grouping_column_name_is_none(single_series_data):
    """All rows have grouping_column_name=None for per-series."""
    y_true, y_pred, y_train = single_series_data
    result = evaluate_metrics(
        y_true, y_pred, y_train, _metrics_config([RMSE_DEF]), "fold_0"
    )

    assert result["grouping_column_name"].isna().all()


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


# ---- new params accepted ----


def test_evaluate_metrics_accepts_grouping_and_weights(single_series_data):
    """grouping_df and fold_weights are accepted without changing output."""
    y_true, y_pred, y_train = single_series_data
    grouping_df = pd.DataFrame({"unique_id": ["A"], "category": ["cat1"]})

    result_without = evaluate_metrics(
        y_true, y_pred, y_train, _metrics_config([RMSE_DEF]), "fold_0"
    )
    result_with = evaluate_metrics(
        y_true,
        y_pred,
        y_train,
        _metrics_config([RMSE_DEF]),
        "fold_0",
        grouping_df=grouping_df,
        fold_weights={"A": 1.0},
    )

    pd.testing.assert_frame_equal(result_with, result_without)
