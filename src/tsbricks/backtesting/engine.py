"""End-to-end backtest orchestrator."""

from __future__ import annotations

import pandas as pd

from tsbricks.backtesting.cross_validation import generate_folds
from tsbricks.backtesting.evaluation import evaluate_metrics
from tsbricks.backtesting.results import BacktestResults, CVResults, TestResults
from tsbricks.backtesting.schema import parse_config
from tsbricks.runner import (
    apply_transforms,
    fit_transforms,
    inverse_transforms,
    invoke_model,
)


def run_backtest(
    config_path: str | None = None,
    config: dict | None = None,
    df: pd.DataFrame | None = None,
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

    Returns:
        A :class:`BacktestResults` containing CV metrics, forecasts,
        fold metadata, and optionally test fold results.

    Raises:
        ValueError: If *df* is ``None`` or if config arguments are invalid.
    """
    if df is None:
        raise ValueError("A DataFrame must be provided via the 'df' parameter.")

    backtest_config = parse_config(config_path=config_path, config=config)

    # Rename user columns to standard names (no-op when defaults are used)
    col_map = {
        backtest_config.data.target_col: "y",
        backtest_config.data.date_col: "ds",
        backtest_config.data.id_col: "unique_id",
    }
    df = df.rename(columns=col_map)

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
        test=test_results,
    )
