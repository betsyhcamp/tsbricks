from __future__ import annotations

from typing import Any

import pandas as pd

from tsbricks.runner._utils import dynamic_import


def resolve_model(model_config: Any) -> tuple[Any, dict]:
    """Resolve a model callable from config.

    Dynamically imports the function specified by
    ``model_config.callable`` and extracts hyperparameters.

    Args:
        model_config: Config object with at least ``callable: str``
            and ``hyperparameters: dict | None``.

    Returns:
        ``(model_fn, hyperparameters)`` — the callable and its kwargs.
    """
    model_fn = dynamic_import(model_config.callable)
    hyperparameters = model_config.hyperparameters or {}
    return model_fn, hyperparameters


def invoke_model(
    train_df: pd.DataFrame,
    model_config: Any,
    horizon: int,
    future_x_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, Any | None]:
    """Run a model callable and normalise its return value.

    The model callable must accept ``(train_df, horizon, **kwargs)``
    and may return one of three shapes:

    * ``DataFrame`` — forecast only.
    * ``(DataFrame, DataFrame)`` — forecast + fitted values.
    * ``(DataFrame, DataFrame, object)`` — forecast + fitted values
      + model object.

    This function normalises all three into a consistent 3-tuple.

    Args:
        train_df: Training panel DataFrame.
        model_config: Config object with ``callable`` and
            ``hyperparameters``.
        horizon: Number of forecast steps.
        future_x_df: Optional future exogenous DataFrame. If provided,
            passed as a keyword argument to the model callable.

    Returns:
        ``(forecast_df, fitted_values_df | None, model_object | None)``
    """
    model_fn, hyperparameters = resolve_model(model_config)

    kwargs: dict[str, Any] = {**hyperparameters}
    if future_x_df is not None:
        kwargs["future_x_df"] = future_x_df

    result = model_fn(train_df, horizon, **kwargs)

    if isinstance(result, pd.DataFrame):
        return result, None, None

    if isinstance(result, tuple):
        if len(result) == 2:
            return result[0], result[1], None
        if len(result) == 3:
            return result[0], result[1], result[2]

    raise TypeError(
        f"Model callable must return a DataFrame or a tuple of length 2-3, "
        f"got {type(result).__name__}"
    )
