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


# ---- group scope ----


def _four_series_data():
    """Four series (A, B, C, D) with known values for group scope tests.

    Series A: y_true=[2,4], y_pred=[1,3]  (errors=[1,1])
    Series B: y_true=[10,20], y_pred=[10,20]  (errors=[0,0])
    Series C: y_true=[5,5], y_pred=[3,3]  (errors=[2,2])
    Series D: y_true=[100,200], y_pred=[100,200]  (errors=[0,0])

    Groups: cat1={A,B}, cat2={C,D}
    cat1 concatenated: y_true=[2,4,10,20], y_pred=[1,3,10,20] -> RMSE=sqrt((1+1+0+0)/4)=sqrt(0.5)
    cat2 concatenated: y_true=[5,5,100,200], y_pred=[3,3,100,200] -> RMSE=sqrt((4+4+0+0)/4)=sqrt(2)
    """
    y_true = pd.DataFrame(
        {
            "unique_id": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "ds": [1, 2, 1, 2, 1, 2, 1, 2],
            "y": [2.0, 4.0, 10.0, 20.0, 5.0, 5.0, 100.0, 200.0],
        }
    )
    y_pred = pd.DataFrame(
        {
            "unique_id": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "ds": [1, 2, 1, 2, 1, 2, 1, 2],
            "ypred": [1.0, 3.0, 10.0, 20.0, 3.0, 3.0, 100.0, 200.0],
        }
    )
    y_train = pd.DataFrame(
        {
            "unique_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["D"] * 4,
            "ds": list(range(1, 5)) * 4,
            "y": [0.0, 1.0, 2.0, 3.0] * 4,
        }
    )
    grouping_df = pd.DataFrame(
        {
            "unique_id": ["A", "B", "C", "D"],
            "category": ["cat1", "cat1", "cat2", "cat2"],
        }
    )
    return y_true, y_pred, y_train, grouping_df


def _group_metrics_config(
    grouping_columns: list[str] | None = None,
    top_level_grouping_columns: list[str] | None = None,
    metric_type: str = "simple",
) -> MetricsConfig:
    """Build a MetricsConfig with a single group-scope metric."""
    metric_name = "rmse" if metric_type == "simple" else "rmsse"
    metric_callable = (
        "tsbricks.blocks.metrics.rmse"
        if metric_type == "simple"
        else "tsbricks.blocks.metrics.rmsse"
    )
    return MetricsConfig(
        definitions=[
            MetricDefinitionConfig(
                name=metric_name,
                callable=metric_callable,
                type=metric_type,
                scope="group",
                grouping_columns=grouping_columns,
            )
        ],
        grouping_columns=top_level_grouping_columns,
    )


def _group_scope_result():
    """Evaluate group-scope RMSE on _four_series_data; shared by multiple tests."""
    y_true, y_pred, y_train, grouping_df = _four_series_data()
    config = _group_metrics_config(grouping_columns=["category"])
    return evaluate_metrics(
        y_true, y_pred, y_train, config, "fold_0", grouping_df=grouping_df
    )


def test_group_scope_produces_row_per_group():
    """Group scope emits one row per group."""
    result = _group_scope_result()

    assert len(result) == 2
    assert set(result["unique_id"]) == {"cat1", "cat2"}


def test_group_scope_output_metadata():
    """Group scope rows have scope='group' and correct grouping_column_name."""
    result = _group_scope_result()

    assert (result["scope"] == "group").all()
    assert (result["grouping_column_name"] == "category").all()


def test_group_scope_simple_metric_values():
    """Group-scope RMSE values match manual computation on concatenated arrays."""
    result = _group_scope_result()

    cat1_val = result.loc[result["unique_id"] == "cat1", "value"].iloc[0]
    cat2_val = result.loc[result["unique_id"] == "cat2", "value"].iloc[0]
    assert cat1_val == pytest.approx(np.sqrt(0.5))
    assert cat2_val == pytest.approx(np.sqrt(2.0))


def _four_series_data_different_scales():
    """Like _four_series_data but with different training scales per group.

    y_true and y_pred are identical to _four_series_data.
    y_train differs: cat1 (A,B) has unit diffs, cat2 (C,D) has 10x diffs.

    cat1 concat y_train: [0,1,2,3, 0,1,2,3]
      diffs = [1,1,1,-3,1,1,1] → mean_sq = 15/7 → scale = sqrt(15/7)
    cat2 concat y_train: [0,10,20,30, 0,10,20,30]
      diffs = [10,10,10,-30,10,10,10] → mean_sq = 1500/7 → scale = sqrt(1500/7)

    cat1 RMSSE = sqrt(0.5) / sqrt(15/7) = sqrt(7/30)
    cat2 RMSSE = sqrt(2.0) / sqrt(1500/7) = sqrt(7/750)
    """
    y_true, y_pred, _, grouping_df = _four_series_data()
    y_train = pd.DataFrame(
        {
            "unique_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["D"] * 4,
            "ds": list(range(1, 5)) * 4,
            "y": (
                [0.0, 1.0, 2.0, 3.0]  # A: diffs = [1,1,1]
                + [0.0, 1.0, 2.0, 3.0]  # B: diffs = [1,1,1]
                + [0.0, 10.0, 20.0, 30.0]  # C: diffs = [10,10,10]
                + [0.0, 10.0, 20.0, 30.0]  # D: diffs = [10,10,10]
            ),
        }
    )
    return y_true, y_pred, y_train, grouping_df


def test_group_scope_context_aware_output_metadata():
    """Group scope with context-aware metric has correct output metadata."""
    y_true, y_pred, y_train, grouping_df = _four_series_data_different_scales()
    config = _group_metrics_config(
        grouping_columns=["category"], metric_type="context_aware"
    )

    result = evaluate_metrics(
        y_true, y_pred, y_train, config, "fold_0", grouping_df=grouping_df
    )

    assert len(result) == 2
    assert set(result["unique_id"]) == {"cat1", "cat2"}
    assert (result["scope"] == "group").all()


def test_group_scope_context_aware_metric_values():
    """Group-scope RMSSE uses group-filtered y_train, producing different scales."""
    y_true, y_pred, y_train, grouping_df = _four_series_data_different_scales()
    config = _group_metrics_config(
        grouping_columns=["category"], metric_type="context_aware"
    )

    result = evaluate_metrics(
        y_true, y_pred, y_train, config, "fold_0", grouping_df=grouping_df
    )

    # cat1 RMSSE = sqrt(0.5) / sqrt(15/7) = sqrt(7/30)
    # cat2 RMSSE = sqrt(2.0) / sqrt(1500/7) = sqrt(7/750)
    cat1_val = result.loc[result["unique_id"] == "cat1", "value"].iloc[0]
    cat2_val = result.loc[result["unique_id"] == "cat2", "value"].iloc[0]
    assert cat1_val == pytest.approx(np.sqrt(7.0 / 30.0))
    assert cat2_val == pytest.approx(np.sqrt(7.0 / 750.0))


def test_group_scope_falls_back_to_top_level_grouping_columns():
    """Group scope uses MetricsConfig.grouping_columns when definition has none."""
    y_true, y_pred, y_train, grouping_df = _four_series_data()
    config = _group_metrics_config(top_level_grouping_columns=["category"])

    result = evaluate_metrics(
        y_true, y_pred, y_train, config, "fold_0", grouping_df=grouping_df
    )

    assert len(result) == 2
    assert (result["grouping_column_name"] == "category").all()


def test_mixed_scope_row_counts():
    """Mixed per_series + group scope produces correct row counts."""
    y_true, y_pred, y_train, grouping_df = _four_series_data()
    config = MetricsConfig(
        definitions=[
            MetricDefinitionConfig(
                name="rmse_series",
                callable="tsbricks.blocks.metrics.rmse",
                type="simple",
            ),
            MetricDefinitionConfig(
                name="rmse_group",
                callable="tsbricks.blocks.metrics.rmse",
                type="simple",
                scope="group",
                grouping_columns=["category"],
            ),
        ],
    )

    result = evaluate_metrics(
        y_true, y_pred, y_train, config, "fold_0", grouping_df=grouping_df
    )

    per_series_rows = result[result["scope"] == "per_series"]
    group_rows = result[result["scope"] == "group"]
    assert len(per_series_rows) == 4
    assert len(group_rows) == 2


def test_mixed_scope_identifiers():
    """Mixed per_series + group scope produces correct identifiers per scope."""
    y_true, y_pred, y_train, grouping_df = _four_series_data()
    config = MetricsConfig(
        definitions=[
            MetricDefinitionConfig(
                name="rmse_series",
                callable="tsbricks.blocks.metrics.rmse",
                type="simple",
            ),
            MetricDefinitionConfig(
                name="rmse_group",
                callable="tsbricks.blocks.metrics.rmse",
                type="simple",
                scope="group",
                grouping_columns=["category"],
            ),
        ],
    )

    result = evaluate_metrics(
        y_true, y_pred, y_train, config, "fold_0", grouping_df=grouping_df
    )

    per_series_rows = result[result["scope"] == "per_series"]
    group_rows = result[result["scope"] == "group"]
    assert set(per_series_rows["unique_id"]) == {"A", "B", "C", "D"}
    assert set(group_rows["unique_id"]) == {"cat1", "cat2"}
