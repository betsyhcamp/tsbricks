"""Metric evaluation for a single backtest fold."""

from __future__ import annotations

import traceback as tb_module

import numpy as np
import pandas as pd

from tsbricks.backtesting.schema import MetricsConfig
from tsbricks.runner._utils import dynamic_import


def _resolve_params(
    defn,
    y_train: pd.DataFrame,
    grouping_df: pd.DataFrame | None,
) -> dict[str, dict]:
    """Resolve static per_series_params and fold-dependent param_resolvers.

    Returns a mapping ``{param_name: {unique_id: scalar}}``.
    """
    resolved: dict[str, dict] = {}

    if defn.per_series_params:
        for param_name, lookup in defn.per_series_params.items():
            resolved[param_name] = lookup

    if defn.param_resolvers:
        for param_name, resolver_config in defn.param_resolvers.items():
            if resolver_config.grouping_columns:
                if grouping_df is None:
                    raise ValueError(
                        f"Resolver for param '{param_name}' on metric "
                        f"'{defn.name}' declares grouping_columns "
                        f"{resolver_config.grouping_columns} but no "
                        f"grouping_df was provided."
                    )
                missing = set(resolver_config.grouping_columns) - set(
                    grouping_df.columns
                )
                if missing:
                    raise ValueError(
                        f"Resolver for param '{param_name}' on metric "
                        f"'{defn.name}' references grouping_columns "
                        f"{sorted(missing)} not found in grouping_df "
                        f"(available: {sorted(grouping_df.columns)})."
                    )

            resolver_fn = dynamic_import(resolver_config.callable)
            resolver_kwargs = resolver_config.params or {}
            result = resolver_fn(y_train, grouping_df=grouping_df, **resolver_kwargs)
            resolved[param_name] = result

    return resolved


def _compute_per_series_values(
    defn,
    metric_fn,
    kwargs: dict,
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    y_train: pd.DataFrame,
    unique_ids,
    resolved_params: dict[str, dict] | None = None,
    run_summary: dict | None = None,
    fold_id: str | None = None,
) -> dict[str, float]:
    """Compute per-series metric values and return as {unique_id: value} dict.

    Shared by global scope (all series) and group two-stage (group subset).
    """
    values: dict[str, float] = {}
    for uid in unique_ids:
        y_true_arr = y_true.loc[y_true["unique_id"] == uid, "y"].to_numpy()
        y_pred_arr = y_pred.loc[y_pred["unique_id"] == uid, "ypred"].to_numpy()
        y_train_arr = y_train.loc[y_train["unique_id"] == uid, "y"].to_numpy()

        per_uid_kwargs = {**kwargs}
        if resolved_params:
            for param_name, lookup in resolved_params.items():
                per_uid_kwargs[param_name] = lookup.get(uid, None)

        try:
            if defn.type == "simple":
                result = metric_fn(y_true_arr, y_pred_arr, **per_uid_kwargs)
            else:
                result = metric_fn(
                    y_true_arr, y_pred_arr, y_train=y_train_arr, **per_uid_kwargs
                )

            values[uid] = result[0] if isinstance(result, tuple) else result
        except Exception as exc:
            if run_summary is None:
                raise
            values[uid] = np.nan
            run_summary["errors"].append(
                {
                    "fold": fold_id,
                    "stage": "metric",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": tb_module.format_exc(),
                    "unique_id": uid,
                    "metric": defn.name,
                }
            )
    return values


def _evaluate_global_scope(
    defn,
    metric_fn,
    kwargs: dict,
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    y_train: pd.DataFrame,
    fold_id: str,
    fold_weights: dict[str, float] | None,
    rows: list[dict],
    resolved_params: dict[str, dict] | None = None,
    run_summary: dict | None = None,
) -> None:
    """Compute a global-scope metric via two-stage aggregation.

    Stage 1: compute per-series metric values.
    Stage 2: call the user-provided aggregation callable with per-series
    values and fold weights.
    """
    if fold_weights is None:
        raise ValueError(
            f"fold_weights is required for global-scope metric '{defn.name}' "
            f"(aggregation_callable={defn.aggregation_callable}), "
            f"but None was provided."
        )

    unique_ids = y_pred["unique_id"].unique()
    per_series_values = _compute_per_series_values(
        defn,
        metric_fn,
        kwargs,
        y_true,
        y_pred,
        y_train,
        unique_ids,
        resolved_params=resolved_params,
        run_summary=run_summary,
        fold_id=fold_id,
    )

    agg_fn = dynamic_import(defn.aggregation_callable)
    agg_kwargs = {**(defn.aggregation_params or {})}
    agg_kwargs["weights"] = fold_weights
    value = agg_fn(per_series_values, **agg_kwargs)

    rows.append(
        {
            "metric_name": defn.name,
            "unique_id": None,
            "fold": fold_id,
            "scope": "global",
            "grouping_column_name": None,
            "aggregation": defn.aggregation,
            "value": value,
        }
    )


def _evaluate_group_two_stage(
    defn,
    metric_fn,
    kwargs: dict,
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    y_train: pd.DataFrame,
    grouping_df: pd.DataFrame | None,
    metrics_config: MetricsConfig,
    fold_id: str,
    fold_weights: dict[str, float] | None,
    rows: list[dict],
    resolved_params: dict[str, dict] | None = None,
    run_summary: dict | None = None,
) -> None:
    """Compute a group-scope metric via two-stage aggregation.

    For each group: compute per-series metric values for group members,
    subset fold_weights to the group, then call the aggregation callable.
    """
    if fold_weights is None:
        raise ValueError(
            f"fold_weights is required for group-scope metric '{defn.name}' "
            f"with aggregation_callable={defn.aggregation_callable}, "
            f"but None was provided."
        )

    group_col = (defn.grouping_columns or metrics_config.grouping_columns)[0]
    group_map = grouping_df.groupby(group_col)["unique_id"].apply(set).to_dict()  # type: ignore[union-attr]

    for group_label, uid_set in group_map.items():
        per_series_values = _compute_per_series_values(
            defn,
            metric_fn,
            kwargs,
            y_true,
            y_pred,
            y_train,
            uid_set,
            resolved_params=resolved_params,
            run_summary=run_summary,
            fold_id=fold_id,
        )

        group_weights = {
            uid: fold_weights[uid] for uid in uid_set if uid in fold_weights
        }

        agg_fn = dynamic_import(defn.aggregation_callable)
        agg_kwargs = {**(defn.aggregation_params or {})}
        agg_kwargs["weights"] = group_weights
        value = agg_fn(per_series_values, **agg_kwargs)

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
    run_summary: dict | None = None,
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
        fold_weights: Per-series weights for this fold.  Required when
            any metric has ``scope="global"`` or ``scope="group"`` with
            an ``aggregation_callable``.

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
        if defn.scope == "group" and not defn.aggregation_callable:
            if defn.param_resolvers:
                raise ValueError(
                    f"Metric '{defn.name}' has scope='group' without "
                    f"aggregation_callable but defines param_resolvers. "
                    f"Param resolvers require per-series computation "
                    f"(use aggregation_callable for two-stage group metrics)."
                )
            if defn.per_series_params:
                raise ValueError(
                    f"Metric '{defn.name}' has scope='group' without "
                    f"aggregation_callable but defines per_series_params. "
                    f"Per-series params require per-series computation "
                    f"(use aggregation_callable for two-stage group metrics)."
                )

        resolved_params = _resolve_params(defn, y_train, grouping_df)

        if defn.scope == "per_series":
            for uid in unique_ids:
                y_true_arr = y_true.loc[y_true["unique_id"] == uid, "y"].to_numpy()
                y_pred_arr = y_pred.loc[y_pred["unique_id"] == uid, "ypred"].to_numpy()
                y_train_arr = y_train.loc[y_train["unique_id"] == uid, "y"].to_numpy()

                per_uid_kwargs = {**kwargs}
                if resolved_params:
                    for param_name, lookup in resolved_params.items():
                        per_uid_kwargs[param_name] = lookup.get(uid, None)

                try:
                    if defn.type == "simple":
                        result = metric_fn(y_true_arr, y_pred_arr, **per_uid_kwargs)
                    else:
                        result = metric_fn(
                            y_true_arr,
                            y_pred_arr,
                            y_train=y_train_arr,
                            **per_uid_kwargs,
                        )

                    value = result[0] if isinstance(result, tuple) else result
                except Exception as exc:
                    if run_summary is None:
                        raise
                    value = np.nan
                    run_summary["errors"].append(
                        {
                            "fold": fold_id,
                            "stage": "metric",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": tb_module.format_exc(),
                            "unique_id": uid,
                            "metric": defn.name,
                        }
                    )

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
            if defn.aggregation_callable:
                _evaluate_group_two_stage(
                    defn,
                    metric_fn,
                    kwargs,
                    y_true,
                    y_pred,
                    y_train,
                    grouping_df,
                    metrics_config,
                    fold_id,
                    fold_weights,
                    rows,
                    resolved_params=resolved_params,
                    run_summary=run_summary,
                )
            else:
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
            _evaluate_global_scope(
                defn,
                metric_fn,
                kwargs,
                y_true,
                y_pred,
                y_train,
                fold_id,
                fold_weights,
                rows,
                resolved_params=resolved_params,
                run_summary=run_summary,
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
