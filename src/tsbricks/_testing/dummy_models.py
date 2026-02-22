"""Dummy model callables for testing invoke_model return-type detection."""

from __future__ import annotations

import pandas as pd


def forecast_only(
    train_df: pd.DataFrame, horizon: int, **kwargs: object
) -> pd.DataFrame:
    """Return a forecast DataFrame only."""
    last_ds = train_df["ds"].max()
    rows = []
    for uid in train_df["unique_id"].unique():
        last_y = train_df.loc[train_df["unique_id"] == uid, "y"].iloc[-1]
        for h in range(1, horizon + 1):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": last_ds + pd.DateOffset(months=h),
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


def returns_int(train_df: pd.DataFrame, horizon: int, **kwargs: object) -> int:
    """Return an int — invalid return type for testing."""
    return 42
