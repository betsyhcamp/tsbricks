from __future__ import annotations

from typing import Any

import pandas as pd

from tsbricks.runner.utils import dynamic_import


def resolve_model(model_config: Any) -> tuple[Any, dict]:
    """Resolve a model callable and its config-derived kwargs.

    Dynamically imports ``model_config.fit_predict_callable`` and merges
    the fit-step and forecast-step parameters into a single kwargs dict.
    That callable fits **and** forecasts, so it needs predict-time
    parameters as well as fit-time ones.

    ``hyperparameters`` wins on overlap: the fit-step declaration
    governs the combined call. The overlap itself is reported at
    config-parse time, not here -- it is a property of the
    configuration rather than of any one invocation.

    Args:
        model_config: Config object with at least
            ``fit_predict_callable: str`` and
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

    model_fn = dynamic_import(model_config.fit_predict_callable)
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
        model_config: Config object with ``fit_predict_callable`` and
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

    # Message skeleton shared with invoke_predict, kept parallel by
    # convention rather than a helper:
    #   <Role> callable '<dotted path>' must return <legal shape>, got <type>
    # Editing one should prompt an edit to the other. The engine collects
    # per-fold failures into run_summary["errors"], so this text is read
    # detached from its call site and must name the offending callable.
    raise TypeError(
        f"Model callable '{model_config.fit_predict_callable}' must return "
        f"a DataFrame or a tuple of length 2-3, got {type(result).__name__}"
    )


def invoke_predict(
    fitted_model: Any,
    model_config: Any,
    horizon: int,
    future_x_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run a model's predict-only callable against an already-fitted model.

    Mirrors ``invoke_model()``'s resolve-and-call shape for the
    predict-only half. The model is never refit, and never inspected or
    assumed to have a particular shape by this function.

    ``hyperparameters`` are deliberately not forwarded: they are
    fit-time and already baked into the fitted object, and passing
    e.g. ``num_leaves=31`` to a predict callable would raise
    ``TypeError``. Predict-time parameters travel in ``predict_params``.

    The return is a bare DataFrame rather than ``invoke_model``'s
    3-tuple, because predicting produces a forecast and nothing else --
    no fitted values, no new model object. A callable emitting
    prediction intervals still returns one DataFrame, with extra
    columns.

    Args:
        fitted_model: An already-fitted model object.
        model_config: Config object with ``predict_callable`` and
            optionally ``predict_params``.
        horizon: Number of forecast steps.
        future_x_df: Optional future exogenous DataFrame. If provided,
            passed as a keyword argument to the predict callable,
            overriding any ``future_x_df`` key in ``predict_params``.

    Returns:
        The predict callable's forecast DataFrame.

    Raises:
        ValueError: If ``model_config.predict_callable`` is absent or
            ``None``, or if ``predict_params`` contains ``horizon``.
        TypeError: If the predict callable does not return a DataFrame.

    .. note:: This function does not capture warnings internally.
       See ``PACKAGE_MAINTAINER_SPEC.md`` §9 for warning capture patterns.
    """
    predict_fn, predict_params = resolve_predict(model_config)

    kwargs: dict[str, Any] = {**predict_params}
    if future_x_df is not None:
        kwargs["future_x_df"] = future_x_df

    result = predict_fn(fitted_model, horizon, **kwargs)

    # Deliberate assertion, not dispatch: invoke_model inspects the shape
    # because it must normalize 1/2/3-tuples, but there is nothing to
    # dispatch on here. The realistic authoring mistake is pointing
    # predict_callable at a fit callable, whose (forecast, fitted) tuple
    # would otherwise travel into scoring and fail somewhere unrelated.
    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            f"Predict callable '{model_config.predict_callable}' must return "
            f"a DataFrame, got {type(result).__name__}"
        )

    return result
