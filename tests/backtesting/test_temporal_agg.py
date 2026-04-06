"""Tests for temporal aggregation of backtest forecasts."""

from __future__ import annotations

import pandas as pd
import pytest

from tsbricks.backtesting.results import (
    AggregatedResults,
    BacktestResults,
    CVResults,
    TestResults,
)
from tsbricks.backtesting.schema import (
    AggregationConfig,
    EvaluationLevelConfig,
)
from tsbricks.backtesting.temporal_agg import aggregate_backtest


# ---- helpers ----


def _weekly_dates(start: str, periods: int) -> list[pd.Timestamp]:
    """Generate weekly Monday timestamps."""
    return list(pd.date_range(start, periods=periods, freq="W-MON"))


def _make_calendar(
    weeks: list[pd.Timestamp],
    month_map: dict[pd.Timestamp, str] | None = None,
) -> pd.DataFrame:
    """Build a calendar DataFrame mapping weekly timestamps to fiscal months.

    If month_map is None, uses the calendar month of each timestamp.
    """
    if month_map is None:
        month_map = {w: w.strftime("%Y-%m") for w in weeks}
    return pd.DataFrame({"ds": weeks, "fiscal_month": [month_map[w] for w in weeks]})


def _make_panel(
    unique_ids: list[str],
    dates: list[pd.Timestamp],
    value: float = 10.0,
) -> pd.DataFrame:
    """Build a panel DataFrame with constant values."""
    rows = []
    for uid in unique_ids:
        for ds in dates:
            rows.append({"unique_id": uid, "ds": ds, "y": value})
    return pd.DataFrame(rows)


def _make_forecast(
    unique_ids: list[str],
    dates: list[pd.Timestamp],
    value: float = 12.0,
) -> pd.DataFrame:
    """Build a forecast DataFrame with constant values."""
    rows = []
    for uid in unique_ids:
        for ds in dates:
            rows.append({"unique_id": uid, "ds": ds, "ypred": value})
    return pd.DataFrame(rows)


def _make_backtest_results(
    train_dates: list[pd.Timestamp],
    val_dates: list[pd.Timestamp],
    unique_ids: list[str],
    y_val: float = 10.0,
    y_pred: float = 12.0,
    y_train: float = 8.0,
    fold_ids: list[str] | None = None,
    include_test: bool = False,
    test_train_dates: list[pd.Timestamp] | None = None,
    test_dates: list[pd.Timestamp] | None = None,
) -> BacktestResults:
    """Build a minimal BacktestResults for testing aggregation."""
    if fold_ids is None:
        fold_ids = ["fold_0"]

    # One origin per fold, spaced 3 months apart
    base_origin = pd.Timestamp("2023-01-01")
    fold_origins = [
        base_origin + pd.DateOffset(months=3 * i) for i in range(len(fold_ids))
    ]

    forecasts_per_fold = {}
    train_val_splits = {}
    fold_id_to_origin = {}
    for i, fid in enumerate(fold_ids):
        forecasts_per_fold[fid] = _make_forecast(unique_ids, val_dates, y_pred)
        train_val_splits[fid] = {
            "train": _make_panel(unique_ids, train_dates, y_train),
            "val": _make_panel(unique_ids, val_dates, y_val),
        }
        fold_id_to_origin[fid] = fold_origins[i]

    cv = CVResults(
        forecasts_per_fold=forecasts_per_fold,
        metrics=pd.DataFrame(),
        fold_origins=fold_origins,
        fold_id_to_origin=fold_id_to_origin,
        train_val_splits_per_fold=train_val_splits,
    )

    test_results = None
    if include_test:
        t_train_dates = test_train_dates or train_dates
        t_dates = test_dates or val_dates
        test_results = TestResults(
            forecasts=_make_forecast(unique_ids, t_dates, y_pred),
            metrics=pd.DataFrame(),
            test_origin=pd.Timestamp("2023-06-01"),
            train_test_split={
                "train": _make_panel(unique_ids, t_train_dates, y_train),
                "test": _make_panel(unique_ids, t_dates, y_val),
            },
        )

    return BacktestResults(
        cv=cv,
        horizon=[(pd.Timestamp("2023-01-01"), 4)],
        config={},
        test=test_results,
    )


def _default_agg_config() -> AggregationConfig:
    """Standard aggregation config for tests."""
    return AggregationConfig(
        timestamp_col="ds",
        period_col="fiscal_month",
        agg_func="sum",
    )


def _default_eval_level() -> EvaluationLevelConfig:
    """Evaluation level with a single simple MAE metric."""
    return EvaluationLevelConfig(
        metrics={
            "definitions": [
                {
                    "name": "monthly_mae",
                    "callable": "tsbricks.blocks.metrics.mae",
                    "type": "simple",
                }
            ]
        }
    )


# ---- basic aggregation ----


def test_aggregate_backtest_produces_correct_shape() -> None:
    """Aggregated forecasts have one row per series per period."""
    # 4 weekly timestamps across 2 months
    weeks = _weekly_dates("2023-01-02", 4)
    train_weeks = _weekly_dates("2022-10-03", 8)
    train_calendar_weeks = train_weeks + weeks
    full_calendar = _make_calendar(
        train_calendar_weeks,
        {
            **{w: w.strftime("%Y-%m") for w in train_weeks},
            weeks[0]: "2023-01",
            weeks[1]: "2023-01",
            weeks[2]: "2023-02",
            weeks[3]: "2023-02",
        },
    )

    results = _make_backtest_results(
        train_dates=train_weeks,
        val_dates=weeks,
        unique_ids=["A", "B"],
    )

    agg = aggregate_backtest(
        results=results,
        aggregation_config=_default_agg_config(),
        evaluation_level_config=_default_eval_level(),
        calendar_df=full_calendar,
    )

    assert isinstance(agg, AggregatedResults)

    # 2 series x 2 months = 4 rows
    fold_forecast = agg.cv_forecasts["fold_0"]
    assert len(fold_forecast) == 4
    assert set(fold_forecast.columns) == {
        "unique_id",
        "fiscal_month",
        "ypred",
    }


def test_aggregate_backtest_sum_values() -> None:
    """Sum aggregation produces correct totals."""
    weeks = _weekly_dates("2023-01-02", 4)
    month_map = {
        weeks[0]: "2023-01",
        weeks[1]: "2023-01",
        weeks[2]: "2023-02",
        weeks[3]: "2023-02",
    }
    train_weeks = _weekly_dates("2022-10-03", 8)
    all_weeks = train_weeks + weeks
    full_calendar = _make_calendar(
        all_weeks,
        {
            **{w: w.strftime("%Y-%m") for w in train_weeks},
            **month_map,
        },
    )

    results = _make_backtest_results(
        train_dates=train_weeks,
        val_dates=weeks,
        unique_ids=["A"],
        y_pred=5.0,
    )

    agg = aggregate_backtest(
        results=results,
        aggregation_config=_default_agg_config(),
        evaluation_level_config=_default_eval_level(),
        calendar_df=full_calendar,
    )

    fold_forecast = agg.cv_forecasts["fold_0"]
    jan = fold_forecast[fold_forecast["fiscal_month"] == "2023-01"]
    # 2 weeks x 5.0 = 10.0
    assert jan["ypred"].iloc[0] == 10.0


def test_aggregate_backtest_metrics_evaluated() -> None:
    """Metrics are evaluated at the aggregated frequency."""
    weeks = _weekly_dates("2023-01-02", 4)
    month_map = {
        weeks[0]: "2023-01",
        weeks[1]: "2023-01",
        weeks[2]: "2023-02",
        weeks[3]: "2023-02",
    }
    train_weeks = _weekly_dates("2022-10-03", 8)
    all_weeks = train_weeks + weeks
    full_calendar = _make_calendar(
        all_weeks,
        {
            **{w: w.strftime("%Y-%m") for w in train_weeks},
            **month_map,
        },
    )

    results = _make_backtest_results(
        train_dates=train_weeks,
        val_dates=weeks,
        unique_ids=["A", "B"],
        y_val=10.0,
        y_pred=12.0,
    )

    agg = aggregate_backtest(
        results=results,
        aggregation_config=_default_agg_config(),
        evaluation_level_config=_default_eval_level(),
        calendar_df=full_calendar,
    )

    assert len(agg.cv_metrics) > 0
    assert "metric_name" in agg.cv_metrics.columns
    assert "monthly_mae" in agg.cv_metrics["metric_name"].values


def test_aggregate_backtest_reuses_fold_ids() -> None:
    """Aggregated results use the same fold IDs as native results."""
    weeks = _weekly_dates("2023-01-02", 4)
    train_weeks = _weekly_dates("2022-10-03", 8)
    all_weeks = train_weeks + weeks
    full_calendar = _make_calendar(all_weeks)

    results = _make_backtest_results(
        train_dates=train_weeks,
        val_dates=weeks,
        unique_ids=["A"],
        fold_ids=["fold_0", "fold_1"],
    )

    agg = aggregate_backtest(
        results=results,
        aggregation_config=_default_agg_config(),
        evaluation_level_config=_default_eval_level(),
        calendar_df=full_calendar,
    )

    assert set(agg.cv_forecasts.keys()) == {"fold_0", "fold_1"}


# ---- context-aware metrics ----


def test_aggregate_backtest_context_aware_metric() -> None:
    """Context-aware metrics work with aggregated y_train."""
    weeks = _weekly_dates("2023-01-02", 4)
    train_weeks = _weekly_dates("2022-10-03", 8)
    all_weeks = train_weeks + weeks
    full_calendar = _make_calendar(all_weeks)

    results = _make_backtest_results(
        train_dates=train_weeks,
        val_dates=weeks,
        unique_ids=["A"],
    )

    eval_level = EvaluationLevelConfig(
        metrics={
            "definitions": [
                {
                    "name": "monthly_rmsse",
                    "callable": "tsbricks.blocks.metrics.rmsse",
                    "type": "context_aware",
                }
            ]
        }
    )

    agg = aggregate_backtest(
        results=results,
        aggregation_config=_default_agg_config(),
        evaluation_level_config=eval_level,
        calendar_df=full_calendar,
    )

    assert "monthly_rmsse" in agg.cv_metrics["metric_name"].values


# ---- test fold aggregation ----


def test_aggregate_backtest_with_test_fold() -> None:
    """Test fold is aggregated when present."""
    weeks = _weekly_dates("2023-01-02", 4)
    train_weeks = _weekly_dates("2022-10-03", 8)
    all_weeks = train_weeks + weeks
    full_calendar = _make_calendar(all_weeks)

    results = _make_backtest_results(
        train_dates=train_weeks,
        val_dates=weeks,
        unique_ids=["A", "B"],
        include_test=True,
        test_train_dates=train_weeks,
        test_dates=weeks,
    )

    agg = aggregate_backtest(
        results=results,
        aggregation_config=_default_agg_config(),
        evaluation_level_config=_default_eval_level(),
        calendar_df=full_calendar,
    )

    assert agg.test_forecasts is not None
    assert agg.test_metrics is not None
    assert len(agg.test_forecasts) > 0
    assert len(agg.test_metrics) > 0


def test_aggregate_backtest_no_test_fold() -> None:
    """Test fold fields are None when test fold is absent."""
    weeks = _weekly_dates("2023-01-02", 4)
    train_weeks = _weekly_dates("2022-10-03", 8)
    all_weeks = train_weeks + weeks
    full_calendar = _make_calendar(all_weeks)

    results = _make_backtest_results(
        train_dates=train_weeks,
        val_dates=weeks,
        unique_ids=["A"],
        include_test=False,
    )

    agg = aggregate_backtest(
        results=results,
        aggregation_config=_default_agg_config(),
        evaluation_level_config=_default_eval_level(),
        calendar_df=full_calendar,
    )

    assert agg.test_forecasts is None
    assert agg.test_metrics is None


# ---- calendar validation ----


def test_calendar_missing_columns_raises() -> None:
    """calendar_df missing required columns raises ValueError."""
    weeks = _weekly_dates("2023-01-02", 4)
    train_weeks = _weekly_dates("2022-10-03", 8)

    # Calendar missing period_col
    bad_calendar = pd.DataFrame({"ds": train_weeks + weeks})

    results = _make_backtest_results(
        train_dates=train_weeks,
        val_dates=weeks,
        unique_ids=["A"],
    )

    with pytest.raises(ValueError, match="missing required columns"):
        aggregate_backtest(
            results=results,
            aggregation_config=_default_agg_config(),
            evaluation_level_config=_default_eval_level(),
            calendar_df=bad_calendar,
        )


def test_calendar_missing_timestamps_raises() -> None:
    """calendar_df missing forecast timestamps raises ValueError."""
    weeks = _weekly_dates("2023-01-02", 4)
    train_weeks = _weekly_dates("2022-10-03", 8)

    # Calendar only covers training weeks, not forecast weeks
    partial_calendar = _make_calendar(train_weeks)

    results = _make_backtest_results(
        train_dates=train_weeks,
        val_dates=weeks,
        unique_ids=["A"],
    )

    with pytest.raises(ValueError, match="missing timestamps"):
        aggregate_backtest(
            results=results,
            aggregation_config=_default_agg_config(),
            evaluation_level_config=_default_eval_level(),
            calendar_df=partial_calendar,
        )


def test_calendar_duplicate_timestamps_raises() -> None:
    """calendar_df with duplicate timestamps raises ValueError."""
    weeks = _weekly_dates("2023-01-02", 4)
    train_weeks = _weekly_dates("2022-10-03", 8)
    all_weeks = train_weeks + weeks

    # Create a calendar with a duplicate timestamp
    good_calendar = _make_calendar(all_weeks)
    dup_row = good_calendar.iloc[[0]]
    bad_calendar = pd.concat([good_calendar, dup_row], ignore_index=True)

    results = _make_backtest_results(
        train_dates=train_weeks,
        val_dates=weeks,
        unique_ids=["A"],
    )

    with pytest.raises(ValueError, match="duplicate timestamps"):
        aggregate_backtest(
            results=results,
            aggregation_config=_default_agg_config(),
            evaluation_level_config=_default_eval_level(),
            calendar_df=bad_calendar,
        )


def test_calendar_df_none_no_source_raises() -> None:
    """calendar_df=None with no calendar_source raises ValueError."""
    weeks = _weekly_dates("2023-01-02", 4)
    train_weeks = _weekly_dates("2022-10-03", 8)

    results = _make_backtest_results(
        train_dates=train_weeks,
        val_dates=weeks,
        unique_ids=["A"],
    )

    with pytest.raises(ValueError, match="calendar_df is required"):
        aggregate_backtest(
            results=results,
            aggregation_config=_default_agg_config(),
            evaluation_level_config=_default_eval_level(),
            calendar_df=None,
        )


# ---- metadata ----


def test_aggregate_backtest_metadata() -> None:
    """Metadata captures aggregation settings."""
    weeks = _weekly_dates("2023-01-02", 4)
    train_weeks = _weekly_dates("2022-10-03", 8)
    all_weeks = train_weeks + weeks
    full_calendar = _make_calendar(all_weeks)

    results = _make_backtest_results(
        train_dates=train_weeks,
        val_dates=weeks,
        unique_ids=["A"],
    )

    agg = aggregate_backtest(
        results=results,
        aggregation_config=_default_agg_config(),
        evaluation_level_config=_default_eval_level(),
        calendar_df=full_calendar,
    )

    assert agg.metadata["timestamp_col"] == "ds"
    assert agg.metadata["period_col"] == "fiscal_month"
    assert agg.metadata["agg_func"] == "sum"
    assert agg.metadata["calendar_source"] is None


# ---- fold_weights with skipped fold (regression) ----


def test_fold_weights_correct_after_skipped_fold() -> None:
    """Aggregated metrics use the correct origin's weights when an earlier fold is missing.

    Regression test: if fold_0 failed during run_backtest() and was
    removed from forecasts_per_fold, fold_1's weights must still be
    looked up by fold_1's origin — not fold_0's.

    Setup:
      - 2 weeks per month, 2 months → series A MAE=4.0, series B MAE=8.0
      - origin_0 weights: A=3.0, B=1.0 → weighted_mean = 5.0
      - origin_1 weights: A=1.0, B=3.0 → weighted_mean = 7.0
      - fold_0 is missing; only fold_1 survives
      - Assert the global metric equals 7.0 (origin_1 weights), not 5.0
    """
    weeks = _weekly_dates("2023-01-02", 4)
    train_weeks = _weekly_dates("2022-10-03", 8)
    all_weeks = train_weeks + weeks
    month_map = {
        weeks[0]: "2023-01",
        weeks[1]: "2023-01",
        weeks[2]: "2023-02",
        weeks[3]: "2023-02",
    }
    full_calendar = _make_calendar(
        all_weeks,
        {
            **{w: w.strftime("%Y-%m") for w in train_weeks},
            **month_map,
        },
    )

    # Series A predicts 12, series B predicts 14; both have actuals 10.
    # After sum aggregation (2 weeks/month):
    #   A: MAE = |20 - 24| = 4.0 per month → mean = 4.0
    #   B: MAE = |20 - 28| = 8.0 per month → mean = 8.0
    forecast_rows = []
    val_rows = []
    for ds in weeks:
        forecast_rows.append({"unique_id": "A", "ds": ds, "ypred": 12.0})
        forecast_rows.append({"unique_id": "B", "ds": ds, "ypred": 14.0})
        val_rows.append({"unique_id": "A", "ds": ds, "y": 10.0})
        val_rows.append({"unique_id": "B", "ds": ds, "y": 10.0})
    forecast_df = pd.DataFrame(forecast_rows)
    val_df = pd.DataFrame(val_rows)
    train_df = _make_panel(["A", "B"], train_weeks, 8.0)

    # Simulate: fold_0 failed, only fold_1 survived
    origin_0 = pd.Timestamp("2023-01-01")
    origin_1 = pd.Timestamp("2023-04-01")

    cv = CVResults(
        forecasts_per_fold={"fold_1": forecast_df},
        metrics=pd.DataFrame(),
        fold_origins=[origin_0, origin_1],
        fold_id_to_origin={
            "fold_0": origin_0,
            "fold_1": origin_1,
        },
        train_val_splits_per_fold={
            "fold_1": {"train": train_df, "val": val_df},
        },
    )

    results = BacktestResults(
        cv=cv,
        horizon=[(origin_0, 4), (origin_1, 4)],
        config={},
    )

    # Asymmetric weights: origin_0 and origin_1 produce
    # different weighted_mean values
    weights_df = pd.DataFrame(
        [
            {
                "unique_id": "A",
                "forecast_origin": origin_0,
                "raw_weight": 3.0,
            },
            {
                "unique_id": "B",
                "forecast_origin": origin_0,
                "raw_weight": 1.0,
            },
            {
                "unique_id": "A",
                "forecast_origin": origin_1,
                "raw_weight": 1.0,
            },
            {
                "unique_id": "B",
                "forecast_origin": origin_1,
                "raw_weight": 3.0,
            },
        ]
    )

    eval_level = EvaluationLevelConfig(
        metrics={
            "definitions": [
                {
                    "name": "global_mae",
                    "callable": "tsbricks.blocks.metrics.mae",
                    "type": "simple",
                    "scope": "global",
                    "aggregation_callable": (
                        "tsbricks.backtesting.aggregations.weighted_mean"
                    ),
                }
            ]
        }
    )

    agg = aggregate_backtest(
        results=results,
        aggregation_config=_default_agg_config(),
        evaluation_level_config=eval_level,
        calendar_df=full_calendar,
        weights_df=weights_df,
    )

    assert set(agg.cv_forecasts.keys()) == {"fold_1"}

    # Extract the global metric value
    global_rows = agg.cv_metrics[
        (agg.cv_metrics["metric_name"] == "global_mae")
        & (agg.cv_metrics["scope"] == "global")
    ]
    assert len(global_rows) == 1
    global_value = global_rows["value"].iloc[0]

    # With origin_1 weights (A=1, B=3):
    #   weighted_mean = (1*4 + 3*8) / (1+3) = 28/4 = 7.0
    # With origin_0 weights (A=3, B=1) it would be:
    #   weighted_mean = (3*4 + 1*8) / (3+1) = 20/4 = 5.0
    assert global_value == pytest.approx(7.0)
