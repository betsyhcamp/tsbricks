"""Dummy model callables for testing model invocation.

Two families live here:

* **Fit-and-forecast callables** ``(train_df, horizon, **kwargs)`` for
  ``invoke_model``: return-type detection and tuple unpacking.
* **Predict-only callables** ``(fitted_model, horizon, **kwargs)`` for
  ``invoke_predict``: these never refit; they receive an
  already-fitted object and produce a forecast from it.

Dotted paths to these callables are used verbatim in tests, which
resolve them through the real ``dynamic_import``. No test mocks
resolution, so renaming anything here breaks tests by design.
"""

from __future__ import annotations

import pandas as pd


def forecast_only(
    train_df: pd.DataFrame, horizon: int, **kwargs: object
) -> pd.DataFrame:
    """Return a forecast DataFrame only."""
    last_ds = train_df["ds"].max()
    is_integer_ds = pd.api.types.is_integer_dtype(train_df["ds"])
    rows = []
    for uid in train_df["unique_id"].unique():
        last_y = train_df.loc[train_df["unique_id"] == uid, "y"].iloc[-1]
        for h in range(1, horizon + 1):
            if is_integer_ds:
                future_ds = last_ds + h
            else:
                future_ds = last_ds + pd.DateOffset(months=h)
            rows.append(
                {
                    "unique_id": uid,
                    "ds": future_ds,
                    "ypred": float(last_y),
                }
            )
    return pd.DataFrame(rows)


def forecast_and_fitted(
    train_df: pd.DataFrame, horizon: int, **kwargs: object
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (forecast, fitted_values)."""
    forecast_df = forecast_only(train_df, horizon, **kwargs)
    fitted_df = train_df[["unique_id", "ds", "y"]].copy()
    fitted_df = fitted_df.rename(columns={"y": "ypred"})
    return forecast_df, fitted_df


def forecast_fitted_and_model(
    train_df: pd.DataFrame, horizon: int, **kwargs: object
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (forecast, fitted_values, model_object)."""
    forecast_df, fitted_df = forecast_and_fitted(train_df, horizon, **kwargs)
    model_object = {"name": "dummy", "hyperparameters": dict(kwargs)}
    return forecast_df, fitted_df, model_object


def forecast_with_exogenous(
    train_df: pd.DataFrame, horizon: int, **kwargs: object
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (forecast, fitted, model_object) with future_x_df echoed back."""
    forecast_df, fitted_df = forecast_and_fitted(train_df, horizon)
    model_object = {
        "name": "dummy_exog",
        "future_x_df": kwargs.get("future_x_df"),
    }
    return forecast_df, fitted_df, model_object


def forecast_with_warning(
    train_df: pd.DataFrame, horizon: int, **kwargs: object
) -> pd.DataFrame:
    """Emit a UserWarning then return a valid forecast."""
    import warnings

    warnings.warn("Model convergence issue", UserWarning, stacklevel=1)
    return forecast_only(train_df, horizon, **kwargs)


def always_fails(
    train_df: pd.DataFrame, horizon: int, **kwargs: object
) -> pd.DataFrame:
    """Always raises ValueError — for testing fold-level resilience."""
    raise ValueError("Intentional model failure for testing")


def returns_int(train_df: pd.DataFrame, horizon: int, **kwargs: object) -> int:
    """Return an int — invalid return type for testing."""
    return 42


def returns_tuple_of_one(
    train_df: pd.DataFrame, horizon: int, **kwargs: object
) -> tuple:
    """Return a 1-tuple — invalid arity for testing."""
    return (forecast_only(train_df, horizon, **kwargs),)


def returns_tuple_of_four(
    train_df: pd.DataFrame, horizon: int, **kwargs: object
) -> tuple:
    """Return a 4-tuple — invalid arity for testing."""
    forecast_df, fitted_df = forecast_and_fitted(train_df, horizon, **kwargs)
    return forecast_df, fitted_df, {"name": "dummy"}, "extra"


# ---- Predict-only callables (invoke_predict) ----


def predict_only(fitted_model: object, horizon: int, **kwargs: object) -> pd.DataFrame:
    """Echo everything received back as a one-row DataFrame.

    Deliberately shaped unlike a realistic predict callable. It ignores
    ``fitted_model`` rather than forecasting from it, and takes
    ``future_x_df`` through ``**kwargs`` instead of as a named
    parameter, so a test can tell "omitted" from "passed as ``None``" since
    ``has_future_x_df`` is False only in the former case. A named
    ``future_x_df=None`` parameter would collapse the two.

    Columns:
        fitted_model: the object received in slot 1, by identity.
        horizon: the value received in slot 2.
        has_future_x_df: whether ``future_x_df`` was passed at all.
        kwargs: every keyword argument received, in one cell -- so
            ``predict_params`` values and ``future_x_df`` can be
            asserted against by identity.
    """
    return pd.DataFrame(
        [
            {
                "fitted_model": fitted_model,
                "horizon": horizon,
                "has_future_x_df": "future_x_df" in kwargs,
                "kwargs": kwargs,
            }
        ]
    )


def predict_returns_int(fitted_model: object, horizon: int, **kwargs: object) -> int:
    """Return an int which is an invalid predict return type for testing."""
    return 6
