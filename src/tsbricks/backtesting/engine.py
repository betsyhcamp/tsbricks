"""End-to-end backtest orchestrator."""

from __future__ import annotations

import pandas as pd

from tsbricks.blocks.metadata import get_git_hash, get_uv_lock_info
from tsbricks.backtesting.cross_validation import generate_folds
from tsbricks.backtesting.evaluation import evaluate_metrics
from tsbricks.backtesting.results import BacktestResults, CVResults, TestResults
from tsbricks.backtesting.schema import BacktestConfig, parse_config
from tsbricks.runner import (
    apply_transforms,
    fit_transforms,
    inverse_transforms,
    invoke_model,
)


def _validate_grouping_df(
    grouping_df: pd.DataFrame | None,
    backtest_config: BacktestConfig,
) -> None:
    """Validate grouping_df against the backtest config.

    Raises:
        ValueError: If a group-scope metric exists but grouping_df is
            missing, or if grouping_df is missing required columns.
    """
    has_group_scope = any(
        defn.scope == "group" for defn in backtest_config.metrics.definitions
    )

    if has_group_scope and grouping_df is None:
        raise ValueError(
            "grouping_df is required when any metric has scope='group', "
            "but none was provided (and metrics.grouping_source is not set)."
        )

    if grouping_df is None:
        return

    if "unique_id" not in grouping_df.columns:
        raise ValueError("grouping_df must contain a 'unique_id' column.")

    # Collect all referenced grouping columns
    required_cols: set[str] = set()
    top_level_cols = backtest_config.metrics.grouping_columns
    for defn in backtest_config.metrics.definitions:
        if defn.grouping_columns is not None:
            required_cols.update(defn.grouping_columns)
        elif top_level_cols is not None:
            required_cols.update(top_level_cols)

    missing = required_cols - set(grouping_df.columns)
    if missing:
        raise ValueError(
            f"grouping_df is missing required grouping columns: {sorted(missing)}."
        )


def _validate_weights_df(
    weights_df: pd.DataFrame | None,
    backtest_config: BacktestConfig,
) -> None:
    """Validate weights_df against the backtest config.

    Raises:
        ValueError: If a metric with ``aggregation_callable`` exists but
            weights_df is missing, or if weights_df is missing required
            columns or forecast origin coverage.
    """
    has_aggregation_callable = any(
        defn.aggregation_callable is not None
        for defn in backtest_config.metrics.definitions
    )

    if has_aggregation_callable and weights_df is None:
        raise ValueError(
            "weights_df is required when any metric has an aggregation_callable, "
            "but none was provided (and metrics.weights_source is not set)."
        )

    if weights_df is None:
        return

    required_cols = {"unique_id", "forecast_origin", "raw_weight"}
    missing = required_cols - set(weights_df.columns)
    if missing:
        raise ValueError(f"weights_df is missing required columns: {sorted(missing)}.")

    # Check forecast origin coverage
    expected_origins = set(backtest_config.cross_validation.forecast_origins)
    if backtest_config.test is not None:
        expected_origins.add(backtest_config.test.test_origin)

    covered_origins = set(weights_df["forecast_origin"].unique())
    missing_origins = expected_origins - covered_origins
    if missing_origins:
        raise ValueError(
            f"weights_df is missing rows for forecast origins: "
            f"{sorted(str(o) for o in missing_origins)}."
        )


def run_backtest(
    config_path: str | None = None,
    config: dict | None = None,
    df: pd.DataFrame | None = None,
    grouping_df: pd.DataFrame | str | None = None,
    weights_df: pd.DataFrame | str | None = None,
) -> BacktestResults:
    """Run a full cross-validated backtest.

    When the config contains a ``test`` block, an independent test fold is
    run after cross-validation.  The test fold fits transforms and the model
    from scratch on ``ds <= test_origin`` and evaluates over the next
    ``cross_validation.horizon`` periods.  There is no separate test horizon.

    Args:
        config_path: Path to a YAML configuration file.
        config: Configuration dict to parse directly.
        df: Input panel DataFrame with at least the columns specified in
            ``DataConfig`` (defaults: ``ds``, ``unique_id``, ``y``).
        grouping_df: Series-to-group mapping DataFrame or path to a
            parquet file.  Required when any metric has ``scope="group"``.
        weights_df: Per-series, per-origin weights DataFrame or path to
            a parquet file.  Required when any metric has an
            ``aggregation_callable``.

    Returns:
        A :class:`BacktestResults` containing CV metrics, forecasts,
        fold metadata, and optionally test fold results.

    Raises:
        ValueError: If *df* is ``None`` or if config arguments are invalid.
    """
    if df is None:
        raise ValueError("A DataFrame must be provided via the 'df' parameter.")

    # Collect environment metadata before any computation
    git_hash = get_git_hash()
    uv_lock_info = get_uv_lock_info()

    backtest_config = parse_config(config_path=config_path, config=config)

    # Rename user columns to standard names (no-op when defaults are used)
    col_map = {
        backtest_config.data.target_col: "y",
        backtest_config.data.date_col: "ds",
        backtest_config.data.id_col: "unique_id",
    }
    df = df.rename(columns=col_map)

    # ---- resolve grouping_df ----
    if isinstance(grouping_df, str):
        grouping_df = pd.read_parquet(grouping_df)
    elif grouping_df is None and backtest_config.metrics.grouping_source is not None:
        grouping_df = pd.read_parquet(backtest_config.metrics.grouping_source)

    _validate_grouping_df(grouping_df, backtest_config)

    # ---- resolve weights_df ----
    if isinstance(weights_df, str):
        weights_df = pd.read_parquet(weights_df)
    elif weights_df is None and backtest_config.metrics.weights_source is not None:
        weights_df = pd.read_parquet(backtest_config.metrics.weights_source)

    _validate_weights_df(weights_df, backtest_config)

    cv_folds, test_split = generate_folds(
        df,
        backtest_config.cross_validation,
        backtest_config.data,
        test_config=backtest_config.test,
    )

    forecasts_per_fold: dict[str, pd.DataFrame] = {}
    all_metrics: list[pd.DataFrame] = []

    for fold_id, splits in cv_folds.items():
        train_df = splits["train"]
        val_df = splits["val"]

        fitted_transforms, transformed_train = fit_transforms(
            train_df, backtest_config.transforms or []
        )
        apply_transforms(val_df, fitted_transforms)

        # Will need & use returned variables _variablename in a future version
        forecast_df, _fitted_values_df, _model_object = invoke_model(
            transformed_train,
            backtest_config.model,
            backtest_config.cross_validation.horizon,
        )

        forecast_original = inverse_transforms(forecast_df, fitted_transforms)

        fold_metrics = evaluate_metrics(
            y_true=val_df,
            y_pred=forecast_original,
            y_train=train_df,
            metrics_config=backtest_config.metrics,
            fold_id=fold_id,
            grouping_df=grouping_df,
        )

        forecasts_per_fold[fold_id] = forecast_original
        all_metrics.append(fold_metrics)

    metrics = pd.concat(all_metrics, ignore_index=True)

    if backtest_config.data.freq == 1:
        fold_origins = sorted(
            int(origin) for origin in backtest_config.cross_validation.forecast_origins
        )
    else:
        fold_origins = sorted(
            pd.Timestamp(origin)
            for origin in backtest_config.cross_validation.forecast_origins
        )

    cv_results = CVResults(
        forecasts_per_fold=forecasts_per_fold,
        metrics=metrics,
        fold_origins=fold_origins,
        train_val_splits_per_fold=cv_folds,
    )

    # ---- test fold ----
    test_results: TestResults | None = None
    if test_split is not None:
        test_train_df = test_split["train"]
        test_test_df = test_split["test"]

        fitted_transforms, transformed_train = fit_transforms(
            test_train_df, backtest_config.transforms or []
        )
        apply_transforms(test_test_df, fitted_transforms)

        forecast_df, _fitted_values_df, _model_object = invoke_model(
            transformed_train,
            backtest_config.model,
            backtest_config.cross_validation.horizon,
        )

        forecast_original = inverse_transforms(forecast_df, fitted_transforms)

        test_metrics = evaluate_metrics(
            y_true=test_test_df,
            y_pred=forecast_original,
            y_train=test_train_df,
            metrics_config=backtest_config.metrics,
            fold_id="test",
            grouping_df=grouping_df,
        )

        if backtest_config.data.freq == 1:
            test_origin_typed: pd.Timestamp | int = int(
                backtest_config.test.test_origin  # type: ignore[union-attr]
            )
        else:
            test_origin_typed = pd.Timestamp(
                backtest_config.test.test_origin  # type: ignore[union-attr]
            )

        test_results = TestResults(
            forecasts=forecast_original,
            metrics=test_metrics,
            test_origin=test_origin_typed,
            train_test_split=test_split,
        )

    raw_config: dict
    if config is not None:
        raw_config = config
    else:
        import yaml
        from pathlib import Path

        raw_config = yaml.safe_load(Path(config_path).read_text())  # type: ignore[arg-type]

    return BacktestResults(
        cv=cv_results,
        horizon=backtest_config.cross_validation.horizon,
        config=raw_config,
        git_hash=git_hash,
        uv_lock_info=uv_lock_info,
        test=test_results,
    )
