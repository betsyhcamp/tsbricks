"""Output dataclasses for the backtesting system."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class CVResults:
    """Cross-validation results used during model selection.

    Dict keys use the fold naming convention ``fold_0``, ``fold_1``, etc.

    Attributes:
        forecasts_per_fold: Fold ID to forecast DataFrame with columns
            ``ds``, ``unique_id``, ``ypred``. Values are on the original
            (untransformed) scale.
        metrics: Long-format DataFrame with columns ``metric_name``,
            ``unique_id``, ``fold``, ``aggregation``, ``value``.
            Contains raw per-fold, per-series values.
        fold_origins: Chronologically sorted forecast origins (timestamps
            for datetime ds, integers for integer ds), one per fold.
        train_val_splits_per_fold: Fold ID to ``{"train": df, "val": df}``
            on the original scale.
        fitted_values: Fold ID to in-sample fitted-value DataFrame
            (original scale). ``None`` when the model does not return
            fitted values.
        fitted_values_model_scale: Same as ``fitted_values`` but on the
            transformed scale. ``None`` when unavailable.
        transform_params: Fold ID to per-transform fitted parameters,
            e.g. ``{"fold_0": {"box_cox": {"A": {"lambda": 0.5}}}}``.
        metric_instability_flags: DataFrame flagging numerically unstable
            metric computations.
        metric_groups: Group name to grouped-metric DataFrame.
        fitted_models: Fold ID to serialized model bytes.
    """

    # Always present
    forecasts_per_fold: dict[str, pd.DataFrame]
    metrics: pd.DataFrame
    fold_origins: list[pd.Timestamp] | list[int]
    train_val_splits_per_fold: dict[str, dict[str, pd.DataFrame]]

    # Present depending on model/config
    fitted_values: dict[str, pd.DataFrame] | None = None
    fitted_values_model_scale: dict[str, pd.DataFrame] | None = None
    transform_params: dict[str, dict[str, dict]] | None = None
    metric_instability_flags: pd.DataFrame | None = None
    metric_groups: dict[str, pd.DataFrame] | None = None
    fitted_models: dict[str, bytes] | None = None


@dataclass(frozen=True)
class TestResults:
    """Test fold results. Structurally isolated from CV results.

    Unlike ``CVResults``, fields hold single DataFrames (not per-fold
    dicts) because there is exactly one test fold.

    Attributes:
        forecasts: Forecast DataFrame with columns ``ds``,
            ``unique_id``, ``ypred`` on the original scale.
        metrics: Long-format metrics DataFrame with the same schema
            as ``CVResults.metrics``.
        test_origin: Forecast origin for the test fold (timestamp for
            datetime ds, integer for integer ds).
        train_test_split: ``{"train": df, "test": df}`` on the
            original scale.
        fitted_values: In-sample fitted-value DataFrame (original
            scale). ``None`` when the model does not return fitted
            values.
        fitted_values_model_scale: Same as ``fitted_values`` but on
            the transformed scale. ``None`` when unavailable.
        transform_params: Per-transform fitted parameters,
            e.g. ``{"box_cox": {"A": {"lambda": 0.5}}}``.
        metric_instability_flags: DataFrame flagging numerically
            unstable metric computations.
        metric_groups: Grouped-metric DataFrame.
        fitted_model: Serialized model bytes.
    """

    # Always present
    forecasts: pd.DataFrame
    metrics: pd.DataFrame
    test_origin: pd.Timestamp | int
    train_test_split: dict[str, pd.DataFrame]

    # Present depending on model/config
    fitted_values: pd.DataFrame | None = None
    fitted_values_model_scale: pd.DataFrame | None = None
    transform_params: dict[str, dict] | None = None
    metric_instability_flags: pd.DataFrame | None = None
    metric_groups: pd.DataFrame | None = None
    fitted_model: bytes | None = None


@dataclass(frozen=True)
class AggregatedResults:
    """Results from temporal aggregation of backtest forecasts.

    Attributes:
        cv_forecasts: Fold ID to aggregated forecast DataFrame with
            columns ``unique_id``, ``{period_col}``, ``ypred``.
        cv_metrics: Long-format metrics DataFrame at the aggregated
            frequency.  Same schema as ``CVResults.metrics``.
        test_forecasts: Aggregated test forecast DataFrame.  ``None``
            when the test fold is absent.
        test_metrics: Aggregated test metrics DataFrame.  ``None``
            when the test fold is absent.
        metadata: Aggregation settings for logging
            (``timestamp_col``, ``period_col``, ``agg_func``,
            ``calendar_source``).
    """

    cv_forecasts: dict[str, pd.DataFrame]
    cv_metrics: pd.DataFrame
    test_forecasts: pd.DataFrame | None = None
    test_metrics: pd.DataFrame | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestResults:
    """Top-level results containing CV results, test results, and metadata.

    Attributes:
        cv: Cross-validation results.
        horizon: List of ``(origin, horizon)`` tuples in fold
            order.  Includes all CV origins and, when present,
            the test origin (appended last).  Duplicate origins
            with different horizons are preserved.
        config: Raw configuration dict that was passed to
            ``run_backtest``.
        git_hash: Full 40-character SHA of HEAD at run time.
            ``None`` if git is unavailable.
        uv_lock_info: Dict with ``path`` and ``sha256`` of the
            uv.lock file. ``None`` if uv.lock is not found.
        run_summary: Structured dict with ``"warnings"`` and
            ``"errors"`` lists capturing all issues during the run.
            Always populated (empty lists when no issues).
            See ``spec_backtest_warnings_run_summary.md`` §3.
        test: Test fold results. ``None`` when the test fold is
            disabled.
        aggregated: Temporally aggregated results. ``None`` when no
            aggregation is configured.
        extra: Escape-hatch dict for user-defined data.
    """

    # Required
    cv: CVResults
    horizon: list[tuple[pd.Timestamp | int, int]]
    config: dict

    # Metadata
    git_hash: str | None = None
    uv_lock_info: dict | None = None
    run_summary: dict = field(default_factory=lambda: {"warnings": [], "errors": []})

    # Test results (None when test fold disabled)
    test: TestResults | None = None

    # Aggregated results (None when no aggregation config or no calendar_df)
    aggregated: AggregatedResults | None = None

    # Escape hatch
    extra: dict | None = None
