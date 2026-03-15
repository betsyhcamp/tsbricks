"""Metric evaluation for a single backtest fold."""

from __future__ import annotations

import pandas as pd

from tsbricks.backtesting.schema import MetricsConfig
from tsbricks.runner._utils import dynamic_import


def evaluate_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    y_train: pd.DataFrame,
    metrics_config: MetricsConfig,
    fold_id: str,
) -> pd.DataFrame:
    """Compute all configured metrics for one fold, per series.

    Args:
        y_true: Actual values with columns ``ds``, ``unique_id``, ``y``.
        y_pred: Forecast values with columns ``ds``, ``unique_id``, ``ypred``.
        y_train: Training data with columns ``ds``, ``unique_id``, ``y``.
        metrics_config: Metric definitions from the backtest config.
        fold_id: Fold identifier (e.g. ``"fold_0"``).

    Returns:
        Long-format DataFrame with columns
        ``[metric_name, unique_id, fold, scope, grouping_column_name,
        aggregation, value]``.
    """
    resolved = [
        (defn, dynamic_import(defn.callable), defn.params or {})
        for defn in metrics_config.definitions
    ]

    rows: list[dict] = []

    for uid in y_pred["unique_id"].unique():
        y_true_arr = y_true.loc[y_true["unique_id"] == uid, "y"].to_numpy()
        y_pred_arr = y_pred.loc[y_pred["unique_id"] == uid, "ypred"].to_numpy()
        y_train_arr = y_train.loc[y_train["unique_id"] == uid, "y"].to_numpy()

        for defn, metric_fn, kwargs in resolved:
            if defn.type == "simple":
                result = metric_fn(y_true_arr, y_pred_arr, **kwargs)
            else:
                result = metric_fn(
                    y_true_arr, y_pred_arr, y_train=y_train_arr, **kwargs
                )

            value = result[0] if isinstance(result, tuple) else result

            rows.append(
                {
                    "metric_name": defn.name,
                    "unique_id": uid,
                    "fold": fold_id,
                    "scope": "per_series",
                    "grouping_column_name": None,
                    "aggregation": defn.aggregation,
                    "value": value,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "metric_name",
            "unique_id",
            "fold",
            "scope",
            "grouping_column_name",
            "aggregation",
            "value",
        ],
    )
