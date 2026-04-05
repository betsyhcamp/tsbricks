"""Temporal aggregation of backtest forecasts to a coarser frequency."""

from __future__ import annotations

import pandas as pd

from tsbricks.backtesting.evaluation import evaluate_metrics
from tsbricks.backtesting.results import AggregatedResults, BacktestResults
from tsbricks.backtesting.schema import (
    AggregationConfig,
    EvaluationLevelConfig,
)


def _resolve_calendar_df(
    calendar_df: pd.DataFrame | str | None,
    aggregation_config: AggregationConfig,
) -> pd.DataFrame:
    """Resolve calendar_df from DataFrame, string path, or config source.

    Raises:
        ValueError: If calendar_df cannot be resolved.
    """
    if isinstance(calendar_df, pd.DataFrame):
        return calendar_df
    if isinstance(calendar_df, str):
        return pd.read_parquet(calendar_df)
    if aggregation_config.calendar_source is not None:
        return pd.read_parquet(aggregation_config.calendar_source)
    raise ValueError(
        "calendar_df is required for temporal aggregation but could "
        "not be resolved. Provide a DataFrame, a file path, or set "
        "aggregation.calendar_source in the config."
    )


def _validate_calendar_df(
    calendar_df: pd.DataFrame,
    aggregation_config: AggregationConfig,
) -> None:
    """Validate that calendar_df has the required columns.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {aggregation_config.timestamp_col, aggregation_config.period_col}
    missing = required - set(calendar_df.columns)
    if missing:
        raise ValueError(
            f"calendar_df is missing required columns: {sorted(missing)}. "
            f"Expected timestamp_col='{aggregation_config.timestamp_col}' "
            f"and period_col='{aggregation_config.period_col}'."
        )


def _validate_calendar_coverage(
    calendar_df: pd.DataFrame,
    timestamps: pd.Series,
    timestamp_col: str,
    context: str,
) -> None:
    """Validate that calendar_df covers all timestamps in the data.

    Args:
        calendar_df: Calendar DataFrame with timestamp_col.
        timestamps: Series of timestamps that must be covered.
        timestamp_col: Column name in calendar_df to check against.
        context: Description for error messages (e.g., "fold_0 forecasts").

    Raises:
        ValueError: If any timestamps are missing from the calendar.
    """
    calendar_timestamps = set(calendar_df[timestamp_col])
    data_timestamps = set(timestamps)
    missing = data_timestamps - calendar_timestamps
    if missing:
        raise ValueError(
            f"calendar_df is missing timestamps present in {context}: "
            f"{sorted(str(t) for t in missing)}."
        )


def _aggregate_df(
    df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    aggregation_config: AggregationConfig,
    value_col: str,
) -> pd.DataFrame:
    """Join a DataFrame with the calendar and aggregate to coarser frequency.

    Args:
        df: DataFrame with columns ``ds``, ``unique_id``, and *value_col*.
        calendar_df: Calendar mapping native timestamps to periods.
        aggregation_config: Aggregation settings.
        value_col: Name of the value column to aggregate (e.g., ``"ypred"``
            or ``"y"``).

    Returns:
        Aggregated DataFrame with columns ``unique_id``,
        ``{period_col}``, and *value_col*.
    """
    ts_col = aggregation_config.timestamp_col
    period_col = aggregation_config.period_col

    merged = df.merge(
        calendar_df[[ts_col, period_col]],
        left_on="ds",
        right_on=ts_col,
        how="left",
    )
    # Drop the join key if it differs from ds
    if ts_col != "ds":
        merged = merged.drop(columns=[ts_col])

    return merged.groupby(["unique_id", period_col], as_index=False)[value_col].agg(
        aggregation_config.agg_func
    )


def aggregate_backtest(
    results: BacktestResults,
    aggregation_config: AggregationConfig,
    evaluation_level_config: EvaluationLevelConfig,
    calendar_df: pd.DataFrame | str | None = None,
    grouping_df: pd.DataFrame | str | None = None,
    weights_df: pd.DataFrame | str | None = None,
) -> AggregatedResults:
    """Aggregate backtest forecasts to a coarser temporal frequency.

    Joins forecasts, actuals, and training data with a calendar
    DataFrame, groups by ``(unique_id, period_col)``, and applies
    ``agg_func``.  Optionally evaluates metrics at the aggregated
    frequency.

    This function follows the composable function pattern: it accepts
    parsed Pydantic models (from ``parse_config()``) rather than raw
    dicts or file paths.

    Args:
        results: Native-frequency backtest results from ``run_backtest()``.
        aggregation_config: Parsed ``AggregationConfig`` specifying
            ``timestamp_col``, ``period_col``, and ``agg_func``.
        evaluation_level_config: Parsed ``EvaluationLevelConfig`` for
            the aggregated level (``config.evaluation.aggregated``).
        calendar_df: Calendar DataFrame, file path, or ``None`` (loads
            from ``aggregation_config.calendar_source``).
        grouping_df: Series-to-group mapping DataFrame or file path for
            aggregated-level metrics.  Same resolution pattern as
            native-level ``grouping_df``.
        weights_df: Per-series weights DataFrame or file path for
            aggregated-level metrics.  Same resolution pattern as
            native-level ``weights_df``.

    Returns:
        :class:`AggregatedResults` with aggregated forecasts and metrics.

    Raises:
        ValueError: If ``calendar_df`` cannot be resolved, is missing
            required columns, or does not cover all forecast timestamps.
    """
    # ---- resolve calendar_df ----
    calendar_df = _resolve_calendar_df(calendar_df, aggregation_config)
    _validate_calendar_df(calendar_df, aggregation_config)

    period_col = aggregation_config.period_col

    # ---- resolve grouping_df ----
    if isinstance(grouping_df, str):
        grouping_df = pd.read_parquet(grouping_df)
    elif (
        grouping_df is None
        and evaluation_level_config.metrics.grouping_source is not None
    ):
        grouping_df = pd.read_parquet(evaluation_level_config.metrics.grouping_source)

    # ---- resolve weights_df ----
    if isinstance(weights_df, str):
        weights_df = pd.read_parquet(weights_df)
    elif (
        weights_df is None
        and evaluation_level_config.metrics.weights_source is not None
    ):
        weights_df = pd.read_parquet(evaluation_level_config.metrics.weights_source)

    # ---- aggregate CV folds ----
    agg_cv_forecasts: dict[str, pd.DataFrame] = {}
    all_agg_metrics: list[pd.DataFrame] = []

    for fold_id, forecast_df in results.cv.forecasts_per_fold.items():
        splits = results.cv.train_val_splits_per_fold[fold_id]
        val_df = splits["val"]
        train_df = splits["train"]

        # Validate calendar coverage for this fold
        _validate_calendar_coverage(
            calendar_df,
            forecast_df["ds"],
            aggregation_config.timestamp_col,
            f"{fold_id} forecasts",
        )
        _validate_calendar_coverage(
            calendar_df,
            val_df["ds"],
            aggregation_config.timestamp_col,
            f"{fold_id} actuals",
        )
        _validate_calendar_coverage(
            calendar_df,
            train_df["ds"],
            aggregation_config.timestamp_col,
            f"{fold_id} training data",
        )

        # Aggregate forecasts, actuals, and training data
        agg_pred = _aggregate_df(forecast_df, calendar_df, aggregation_config, "ypred")
        agg_true = _aggregate_df(val_df, calendar_df, aggregation_config, "y")
        agg_train = _aggregate_df(train_df, calendar_df, aggregation_config, "y")

        # Store public-facing forecasts (retain period_col name)
        agg_cv_forecasts[fold_id] = agg_pred

        # Rename period_col -> ds for evaluate_metrics()
        agg_pred_eval = agg_pred.rename(columns={period_col: "ds"})
        agg_true_eval = agg_true.rename(columns={period_col: "ds"})
        agg_train_eval = agg_train.rename(columns={period_col: "ds"})

        fold_metrics = evaluate_metrics(
            y_true=agg_true_eval,
            y_pred=agg_pred_eval,
            y_train=agg_train_eval,
            metrics_config=evaluation_level_config.metrics,
            fold_id=fold_id,
            grouping_df=grouping_df,
        )
        all_agg_metrics.append(fold_metrics)

    cv_metrics = pd.concat(all_agg_metrics, ignore_index=True)

    # ---- aggregate test fold ----
    agg_test_forecasts: pd.DataFrame | None = None
    agg_test_metrics: pd.DataFrame | None = None

    if results.test is not None:
        test_forecast_df = results.test.forecasts
        test_split = results.test.train_test_split
        test_test_df = test_split["test"]
        test_train_df = test_split["train"]

        _validate_calendar_coverage(
            calendar_df,
            test_forecast_df["ds"],
            aggregation_config.timestamp_col,
            "test forecasts",
        )
        _validate_calendar_coverage(
            calendar_df,
            test_test_df["ds"],
            aggregation_config.timestamp_col,
            "test actuals",
        )
        _validate_calendar_coverage(
            calendar_df,
            test_train_df["ds"],
            aggregation_config.timestamp_col,
            "test training data",
        )

        agg_test_pred = _aggregate_df(
            test_forecast_df, calendar_df, aggregation_config, "ypred"
        )
        agg_test_true = _aggregate_df(
            test_test_df, calendar_df, aggregation_config, "y"
        )
        agg_test_train = _aggregate_df(
            test_train_df, calendar_df, aggregation_config, "y"
        )

        agg_test_forecasts = agg_test_pred

        agg_test_pred_eval = agg_test_pred.rename(columns={period_col: "ds"})
        agg_test_true_eval = agg_test_true.rename(columns={period_col: "ds"})
        agg_test_train_eval = agg_test_train.rename(columns={period_col: "ds"})

        agg_test_metrics = evaluate_metrics(
            y_true=agg_test_true_eval,
            y_pred=agg_test_pred_eval,
            y_train=agg_test_train_eval,
            metrics_config=evaluation_level_config.metrics,
            fold_id="test",
            grouping_df=grouping_df,
        )

    metadata = {
        "timestamp_col": aggregation_config.timestamp_col,
        "period_col": aggregation_config.period_col,
        "agg_func": aggregation_config.agg_func,
        "calendar_source": aggregation_config.calendar_source,
    }

    return AggregatedResults(
        cv_forecasts=agg_cv_forecasts,
        cv_metrics=cv_metrics,
        test_forecasts=agg_test_forecasts,
        test_metrics=agg_test_metrics,
        metadata=metadata,
    )
