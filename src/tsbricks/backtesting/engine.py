"""End-to-end backtest orchestrator."""

from __future__ import annotations

import pandas as pd

from tsbricks.backtesting.cross_validation import generate_folds
from tsbricks.backtesting.evaluation import evaluate_metrics
from tsbricks.backtesting.results import BacktestResults, CVResults
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

    Args:
        config_path: Path to a YAML configuration file.
        config: Configuration dict to parse directly.
        df: Input panel DataFrame with at least the columns specified in
            ``DataConfig`` (defaults: ``ds``, ``unique_id``, ``y``).

    Returns:
        A :class:`BacktestResults` containing CV metrics, forecasts,
        and fold metadata.

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

    cv_folds, _ = generate_folds(
        df, backtest_config.cross_validation, backtest_config.data
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
    )
