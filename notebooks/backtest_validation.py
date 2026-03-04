# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv (3.11.11)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Backtest Validation Notebook
#
# Exercises the full `tsbricks` backtesting system with synthetic data and
# a statsforecast AutoETS model. Validates both the single-call `run_backtest`
# API and the composable step functions.

# %% [markdown]
# ## 1. Imports and Synthetic Data

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tsbricks.backtesting import (
    BacktestResults,
    evaluate_metrics,
    generate_folds,
    parse_config,
    run_backtest,
)
from tsbricks.runner import (
    apply_transforms,
    fit_transforms,
    inverse_transforms,
    invoke_model,
)

# Generate a 3-series monthly panel (48 months, strictly positive)
np.random.seed(1027)
dates = pd.date_range("2020-01-01", periods=48, freq="MS")

series_params = {
    "A": {"base": 100, "trend": 0.5, "amplitude": 10},
    "B": {"base": 200, "trend": 1.0, "amplitude": 20},
    "C": {"base": 50, "trend": 0.3, "amplitude": 5},
}

rows = []
for uid, params in series_params.items():
    for i, ds in enumerate(dates):
        value = (
            params["base"]
            + params["trend"] * i
            + params["amplitude"] * np.sin(2 * np.pi * i / 12)
            + np.random.normal(0, 2)
        )
        rows.append({"unique_id": uid, "ds": ds, "y": max(value, 1.0)})

df = pd.DataFrame(rows)
print(f"Shape: {df.shape}")
print(f"\nSeries counts:\n{df.groupby('unique_id').size()}")
df.head(10)

# %%
for uid in df['unique_id'].unique():
    temp=df[df['unique_id'] ==uid].copy()
    fig, ax = plt.subplots(figsize=(9,3))
    ax.plot(temp['ds'], temp['y'])
    ax.set_title(f'Series {uid}')
    ax.set_xlabel('Date')
    ax.set_ylabel('y')
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## 2. Define model callable: Statsforecast Model Wrapper

# %%
def statsforecast_ets(train_df, horizon, **kwargs):
    """Model callable: Thin wrapper around statsforecast AutoETS.

    Follows the tsbricks model callable convention:
    callable(train_df, horizon, **kwargs) -> DataFrame[ds, unique_id, ypred]
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS

    # get "season_length" and "freq" from kwargs and define defaults
    season_length = kwargs.get("season_length", 12)
    freq = kwargs.get("freq", "MS")

    sf = StatsForecast(
        models=[AutoETS(season_length=season_length)],
        freq=freq,
    )
    forecast = sf.forecast(df=train_df, h=horizon)
    forecast = forecast.reset_index()
    forecast = forecast.rename(columns={"AutoETS": "ypred"})
    return forecast[["unique_id", "ds", "ypred"]]


print("Model wrapper defined.")

# %% [markdown]
# ## 3. Build Config

# %%
cfg = {
    "data": {"freq": "MS"},
    "cross_validation": {
        "mode": "explicit",
        "horizon": 6,
        "forecast_origins": ["2023-01-01", "2023-06-01"],
    },
    "transforms": [
        {
            "name": "box_cox",
            "class": "tsbricks.blocks.transforms.BoxCoxTransform",
            "scope": "per_series",
            "targets": ["y"],
            "perform_inverse_transform": True,
            "params": {"method": "guerrero", "season_length": 12},
        }
    ],
    "model": {
        "callable": "__main__.statsforecast_ets",
        "hyperparameters": {"season_length": 12, "freq": "MS"},
    },
    "metrics": {
        "definitions": [
            {
                "name": "rmse",
                "callable": "tsbricks.blocks.metrics.rmse",
                "type": "simple",
            },
            {
                "name": "rmsse",
                "callable": "tsbricks.blocks.metrics.rmsse",
                "type": "context_aware",
            },
        ]
    },
}

cfg

# %% [markdown]
# ## 4. Run `run_backtest` and Inspect Results

# %%
results = run_backtest(config=cfg, df=df)

print(f"Type: {type(results).__name__}")
print(f"Horizon: {results.horizon}")
print(f"Fold origins: {results.cv.fold_origins}")
print(f"Folds: {list(results.cv.forecasts_per_fold.keys())}")
print(f"Test fold: {results.test}")
print(f"\nMetrics ({len(results.cv.metrics)} rows):")
results.cv.metrics

# %%
print("Sample forecast (fold_0):")
results.cv.forecasts_per_fold["fold_0"].head(10)

# %% [markdown]
# ## 5. Exercise Composable Step Functions

# %%
# Parse config and generate folds
backtest_config = parse_config(config=cfg)
cv_folds, _ = generate_folds(
    df, backtest_config.cross_validation, backtest_config.data
)

# verify cross val folds available
cv_folds.keys()

# %%
backtest_config.model.hyperparameters

# %%
backtest_config.model

# %%
cv_folds

# %%
# Do remaining inspection on fold_0
train_df = cv_folds["fold_0"]["train"]
val_df = cv_folds["fold_0"]["val"]
print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")

# %%
# Fit and apply transforms
fitted_transforms, transformed_train = fit_transforms(
    train_df, backtest_config.transforms
)
transformed_val = apply_transforms(val_df, fitted_transforms)

# %%
transformed_train

# %%

# %%
for uid in transformed_train['unique_id'].unique():
    temp = transformed_train[transformed_train['unique_id'] == uid].copy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(temp['ds'], temp['y'])
    ax.set_title(f"Series {uid}")
    ax.set_xlabel("Date")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.show()

# %%
fitted_transforms

# %%
# Invoke model on transformed data
forecast_df, _, _ = invoke_model(
    transformed_train,
    backtest_config.model,
    backtest_config.cross_validation.horizon,
)

# Inverse transform forecasts back to original scale
forecast_original_scale = inverse_transforms(forecast_df, fitted_transforms)

# %%
forecast_df.info()
forecast_df.head()

# %%
forecast_original_scale

# %%
# Evaluate metrics
fold_metrics = evaluate_metrics(
    y_true=val_df,
    y_pred=forecast_original_scale,
    y_train=train_df,
    metrics_config=backtest_config.metrics,
    fold_id="fold_0",
)

print("\nStep-function fold_0 metrics:")
fold_metrics

# %% [markdown]
# ## 6. Inspect Transform Parameters

# %%
params = fitted_transforms[0].get_fitted_params()
print("BoxCox fitted lambdas per series:")
for uid, p in params.items():
    print(f"  {uid}: lambda={p['lambda']:.4f} (type: {type(p['lambda']).__name__})")

# %% [markdown]
# ## 7. Verify Inverse Transform

# %%
print("Transformed scale (first 5):")
print(forecast_df["ypred"].head().values)

print("\nOriginal scale (first 5):")
print(forecast_original["ypred"].head().values)

assert not np.allclose(
    forecast_df["ypred"].values, forecast_original["ypred"].values
), "Inverse transform should change the values"
print("\nInverse transform verified: values differ between scales.")

# %% [markdown]
# ## 8. Sanity-Check Metrics

# %%
# Manually compute RMSE for series A in fold_0
uid = "A"
y_true_a = val_df.loc[val_df["unique_id"] == uid, "y"].to_numpy()
y_pred_a = forecast_original.loc[
    forecast_original["unique_id"] == uid, "ypred"
].to_numpy()

manual_rmse = np.sqrt(np.mean((y_true_a - y_pred_a) ** 2))

# Get the engine-computed RMSE for series A, fold_0
engine_rmse = results.cv.metrics.loc[
    (results.cv.metrics["metric_name"] == "rmse")
    & (results.cv.metrics["unique_id"] == uid)
    & (results.cv.metrics["fold"] == "fold_0"),
    "value",
].iloc[0]

print(f"Manual RMSE (series A, fold_0): {manual_rmse:.6f}")
print(f"Engine RMSE (series A, fold_0): {engine_rmse:.6f}")

assert np.isclose(manual_rmse, engine_rmse, rtol=1e-10), (
    f"RMSE mismatch: manual={manual_rmse}, engine={engine_rmse}"
)
print("\nManual RMSE matches engine RMSE.")
