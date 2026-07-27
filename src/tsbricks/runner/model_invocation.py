from __future__ import annotations

from typing import Any

import pandas as pd

from tsbricks.runner.utils import dynamic_import


def resolve_model(model_config: Any) -> tuple[Any, dict]:
    """Resolve a model callable and its config-derived kwargs.

    Dynamically imports ``model_config.callable`` and merges the
    fit-step and forecast-step parameters into a single kwargs dict.
    ``model_config.callable`` fits **and** forecasts, so it needs
    predict-time parameters as well as fit-time ones.

    ``hyperparameters`` wins on overlap: the fit-step declaration
    governs the combined call. The overlap itself is reported at
    config-parse time, not here -- it is a property of the
    configuration rather than of any one invocation.

    Args:
        model_config: Config object with at least ``callable: str`` and
            ``hyperparameters: dict | None``, and optionally
            ``predict_params: dict | None``.

    Returns:
        ``(model_fn, kwargs)`` the callable and its kwargs.

    Raises:
        ValueError: If either params dict contains the reserved key
            ``horizon``.
    """
    # hyperparameters directly, predict_params via getattr: the former
    # belongs to the pre-existing structural contract, so any config
    # object that ever worked exposes it. The latter postdates that
    # contract, so a duck-typed config cannot be assumed to carry it.
    hyperparameters = model_config.hyperparameters or {}
    predict_params = getattr(model_config, "predict_params", None) or {}

    for name, params in (
        ("hyperparameters", hyperparameters),
        ("predict_params", predict_params),
    ):
        if "horizon" in params:
            raise ValueError(
                f"model_config.{name} must not contain 'horizon': it "
                f"collides with the positional horizon argument."
            )

    model_fn = dynamic_import(model_config.callable)
    return model_fn, {**predict_params, **hyperparameters}


def resolve_predict(model_config: Any) -> tuple[Any, dict]:
    """Resolve a predict-only callable and its config-derived kwargs.

    Dynamically imports the function specified by
    ``model_config.predict_callable`` and normalizes ``predict_params``.
    ``hyperparameters`` are fit-time and are not read here -- they are
    baked into the fitted model the caller already holds.

    Args:
        model_config: Config object with ``predict_callable: str | None``
            and ``predict_params: dict | None``.

    Returns:
        ``(predict_fn, predict_params)`` the callable and its kwargs.

    Raises:
        ValueError: If ``predict_callable`` is absent or ``None``, or if
            ``predict_params`` contains the reserved key ``horizon``.
    """
    # getattr, not attribute access: predict_callable is optional and
    # postdates the structural contract, so a duck-typed config built
    # against the older shape has no such attribute. Absent and None are
    # one user mistake and get one exception naming one remedy.
    predict_callable = getattr(model_config, "predict_callable", None)
    if predict_callable is None:
        raise ValueError(
            "model_config has no 'predict_callable'. Set "
            "ModelConfig.predict_callable to the dotted path of a predict "
            "function (e.g. 'my_project.models.my_model_predict')."
        )

    predict_params = getattr(model_config, "predict_params", None) or {}
    if "horizon" in predict_params:
        raise ValueError(
            "model_config.predict_params must not contain 'horizon': it "
            "collides with the positional horizon argument."
        )

    # dict() so a caller mutating the returned kwargs cannot mutate the config
    return dynamic_import(predict_callable), dict(predict_params)


def invoke_model(
    train_df: pd.DataFrame,
    model_config: Any,
    horizon: int,
    future_x_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, Any | None]:
    """Run a model callable and normalize its return value.

    The model callable must accept ``(train_df, horizon, **kwargs)``
    and may return one of three shapes:

    * ``DataFrame``  forecast only.
    * ``(DataFrame, DataFrame)``  forecast + fitted values.
    * ``(DataFrame, DataFrame, object)``  forecast + fitted values
      + model object.

    This function normalizes all three into a consistent 3-tuple

    Args:
        train_df: Training panel DataFrame.
        model_config: Config object with ``callable`` and
            ``hyperparameters``, and optionally ``predict_params``.
        horizon: Number of forecast steps.
        future_x_df: Optional future exogenous DataFrame. If provided,
            passed as a keyword argument to the model callable,
            overriding any ``future_x_df`` key in the config params.

    Returns:
        ``(forecast_df, fitted_values_df | None, model_object | None)``

    .. note:: This function does not capture warnings internally.
       See ``PACKAGE_MAINTAINER_SPEC.md`` §9 for warning capture patterns.
    """
    model_fn, config_kwargs = resolve_model(model_config)

    kwargs: dict[str, Any] = {**config_kwargs}
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
