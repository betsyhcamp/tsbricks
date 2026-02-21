from __future__ import annotations

from typing import Any

import pandas as pd

from tsbricks.blocks.transforms.base import BaseTransform
from tsbricks.runner._utils import dynamic_import


def fit_transforms(
    train_df: pd.DataFrame,
    transform_configs: list[Any],
) -> tuple[list[BaseTransform], pd.DataFrame]:
    """Fit each transform on *train_df* and return the fitted chain.

    Iterates over *transform_configs* in order.  For each config:

    1. Dynamically imports the class from ``config.class_path``.
    2. Instantiates the transform.
    3. Calls ``fit_transform(df, target_col="y", **config.params)``.
    4. Stores ``config.perform_inverse_transform`` on the instance as
       ``_perform_inverse`` so :func:`inverse_transforms` can read it
       without needing the config again.

    Args:
        train_df: Training panel DataFrame (columns include ``y``,
            ``ds``, ``unique_id``).
        transform_configs: Sequence of config objects, each with at
            least ``class_path: str``, ``params: dict | None``, and
            ``perform_inverse_transform: bool``.

    Returns:
        ``(fitted_transforms, transformed_train_df)`` — the list of
        fitted transform instances and the fully-transformed training
        DataFrame.
    """
    fitted: list[BaseTransform] = []
    df = train_df

    for cfg in transform_configs:
        cls = dynamic_import(cfg.class_path)
        transform: BaseTransform = cls()
        transform._perform_inverse = cfg.perform_inverse_transform  # type: ignore[attr-defined]
        df = transform.fit_transform(df, target_col="y", **(cfg.params or {}))
        fitted.append(transform)

    return fitted, df


def apply_transforms(
    df: pd.DataFrame,
    fitted_transforms: list[BaseTransform],
) -> pd.DataFrame:
    """Apply an already-fitted transform chain to new data.

    Iterates over *fitted_transforms* in order, calling
    ``transform(df, target_col="y")`` on each.

    Args:
        df: Panel DataFrame to transform (e.g., a validation fold).
        fitted_transforms: Transforms previously returned by
            :func:`fit_transforms`.

    Returns:
        Transformed DataFrame.
    """
    for tx in fitted_transforms:
        df = tx.transform(df, target_col="y")
    return df


def inverse_transforms(
    forecast_df: pd.DataFrame,
    fitted_transforms: list[BaseTransform],
) -> pd.DataFrame:
    """Reverse the transform chain on forecast output.

    Iterates over *fitted_transforms* in **reverse** order.  For each
    transform whose ``_perform_inverse`` flag is ``True``, calls
    ``inverse_transform(df, target_col="ypred")``.

    The target column is ``ypred`` (not ``y``) because this operates on
    the forecast DataFrame, which contains predicted values.

    Args:
        forecast_df: Forecast DataFrame with a ``ypred`` column.
        fitted_transforms: Transforms previously returned by
            :func:`fit_transforms`.

    Returns:
        DataFrame with ``ypred`` on the original (untransformed) scale.
    """
    df = forecast_df
    for tx in reversed(fitted_transforms):
        if getattr(tx, "_perform_inverse", True):
            df = tx.inverse_transform(df, target_col="ypred")
    return df
