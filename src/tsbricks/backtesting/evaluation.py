"""Metric evaluation for a single backtest fold."""

from __future__ import annotations

import pandas as pd

from tsbricks.backtesting.schema import MetricsConfig
from tsbricks.runner._utils import dynamic_import


def _evaluate_group_scope(
    defn,
    metric_fn,
    kwargs: dict,
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    y_train: pd.DataFrame,
    grouping_df: pd.DataFrame | None,
    metrics_config: MetricsConfig,
    fold_id: str,
    rows: list[dict],
) -> None:
    """Compute a group-scope metric and append rows to *rows*.

    Series are partitioned by group using *grouping_df*.  Within each
    group the actual/predicted/training arrays are concatenated across
    all member series and the metric is computed once per group.
    """
    group_col = (defn.grouping_columns or metrics_config.grouping_columns)[0]
    group_map = grouping_df.groupby(group_col)["unique_id"].apply(set).to_dict()  # type: ignore[union-attr]

    for group_label, uid_set in group_map.items():
        y_true_arr = y_true.loc[y_true["unique_id"].isin(uid_set), "y"].to_numpy()
        y_pred_arr = y_pred.loc[y_pred["unique_id"].isin(uid_set), "ypred"].to_numpy()
        y_train_arr = y_train.loc[y_train["unique_id"].isin(uid_set), "y"].to_numpy()

        if defn.type == "simple":
            result = metric_fn(y_true_arr, y_pred_arr, **kwargs)
        else:
            result = metric_fn(y_true_arr, y_pred_arr, y_train=y_train_arr, **kwargs)

        value = result[0] if isinstance(result, tuple) else result

        rows.append(
            {
                "metric_name": defn.name,
                "unique_id": group_label,
                "fold": fold_id,
                "scope": "group",
                "grouping_column_name": group_col,
                "aggregation": defn.aggregation,
                "value": value,
            }
        )


def evaluate_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    y_train: pd.DataFrame,
    metrics_config: MetricsConfig,
    fold_id: str,
    grouping_df: pd.DataFrame | None = None,
    fold_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute all configured metrics for one fold, per series.

    Args:
        y_true: Actual values with columns ``ds``, ``unique_id``, ``y``.
        y_pred: Forecast values with columns ``ds``, ``unique_id``, ``ypred``.
        y_train: Training data with columns ``ds``, ``unique_id``, ``y``.
        metrics_config: Metric definitions from the backtest config.
        fold_id: Fold identifier (e.g. ``"fold_0"``).
        grouping_df: Series-to-group mapping.  Required when any metric
            has ``scope="group"``.
        fold_weights: Per-series weights for this fold (used by global
            scope aggregation in Phase 4).  Currently accepted but
            unused.

    Returns:
        Long-format DataFrame with columns
        ``[metric_name, unique_id, fold, scope, grouping_column_name,
        aggregation, value]``.
    """
    resolved = [
        (defn, dynamic_import(defn.callable), defn.params or {})
        for defn in metrics_config.definitions
    ]

    unique_ids = y_pred["unique_id"].unique()
    rows: list[dict] = []

    for defn, metric_fn, kwargs in resolved:
        if defn.scope == "per_series":
            for uid in unique_ids:
                y_true_arr = y_true.loc[y_true["unique_id"] == uid, "y"].to_numpy()
                y_pred_arr = y_pred.loc[y_pred["unique_id"] == uid, "ypred"].to_numpy()
                y_train_arr = y_train.loc[y_train["unique_id"] == uid, "y"].to_numpy()

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

        elif defn.scope == "group":
            _evaluate_group_scope(
                defn,
                metric_fn,
                kwargs,
                y_true,
                y_pred,
                y_train,
                grouping_df,
                metrics_config,
                fold_id,
                rows,
            )

        elif defn.scope == "global":
            pass  # Phase 4

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
