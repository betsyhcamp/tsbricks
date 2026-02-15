# Forecast Backtest and Evaluation System — V1 Product Specification

## 1. Overview

This document specifies the V1 design of a forecast backtesting and evaluation system built for enterprise-scale time series forecasting workflows. The system enables data scientists to run cross-validated backtests with any forecast model, evaluate results using standard and custom metrics, apply preprocessing transforms, and produce structured artifacts suitable for experiment tracking via Vertex AI Experiments.

The system is model-agnostic, configuration-driven, and designed for use within Vertex AI pipelines, Vertex AI notebook instances, and local development environments.

The backtesting system is part of the `tsbricks` package and is accessed via `from tsbricks.backtesting import run_backtest`. Built-in transforms and metrics are provided in the `tsbricks.blocks` namespace (e.g., `tsbricks.blocks.transforms.BoxCoxTransform`, `tsbricks.blocks.metrics.rmsse`). See `PACKAGE_MAINTAINER_SPEC.md` for package architecture details.

______________________________________________________________________

## 2. Design Principles

- **Model-agnostic.** The system is decoupled from any specific forecasting library. Models are user-provided callables.
- **Configuration-driven.** A single YAML config file specifies the forecast horizon, cross-validation strategy, transforms, model hyperparameters, metrics, parallelization, and artifact storage.
- **One model per run.** Each config file defines a single model configuration. Comparisons across models happen via experiment tracking across runs.
- **User as adult.** The system does not impose minimum training sizes or other guardrails that restrict valid experimentation. Models that cannot handle the data will raise their own errors.
- **Resilient execution.** Failed series and folds are skipped and logged, not halted on. The system produces results for everything that succeeds.
- **Structured output.** Results are returned as a typed Python dataclass. The user controls how artifacts are logged to Vertex AI Experiments.
- **Reusable primitives for forward-looking forecasting.** The internal code must be decomposed into primitives that are shared between backtesting and a future `run_forecast` entry point. Config parsing, transform pipeline execution (fit, transform, inverse transform), model invocation, and model serialization are all building blocks that `run_backtest` calls per fold and `run_forecast` will call once on the full dataset. The fold loop, metric computation, and result dataclasses are the only backtest-specific logic. This architectural requirement ensures that the same YAML config that produces the best backtest results can directly drive a forward-looking forecast without any translation or re-specification.

______________________________________________________________________

## 3. API Surface

### 3.1 Two-Level API

The system exposes two levels of API:

**Convenience API.** `run_backtest` is a single function call that executes the entire backtesting pipeline — config parsing, fold generation, transform fitting, model invocation, inverse transforms, metric evaluation, and result assembly. This is the primary API for automated pipeline runs and standard backtests.

```python
from tsbricks.backtesting import run_backtest

results = run_backtest(config_path="config.yaml", df=df)
```

**Composable step function API.** Individual steps are exposed as public functions for power users who need to inspect intermediate state, branch between steps, or compose custom workflows. The step functions are the same primitives that `run_backtest` calls internally.

```python
from tsbricks.backtesting import parse_config, generate_folds, evaluate_metrics
from tsbricks.runner import (
    fit_transforms,
    apply_transforms,
    inverse_transforms,
    resolve_model,
    invoke_model,
    serialize_model,
)
```

### 3.2 Composable Step Functions

The following step functions are public API. Each is a pure function (or near-pure) that takes explicit inputs and returns explicit outputs, with no hidden shared state.

**`parse_config(config_path=None, config=None)`** — Parses and validates a YAML file or dict into a typed Pydantic config object. Mutually exclusive arguments. Returns the validated config.

```python
config = parse_config(config_path="config.yaml")
# or
config = parse_config(config=config_dict)
```

**`generate_folds(df, cv_config, test_config=None)`** — Produces CV fold splits (and optionally the test split) from the DataFrame and cross-validation config. Returns a tuple of `(cv_folds, test_split)` where `cv_folds` is a dict of fold splits (each containing training and validation DataFrames) and `test_split` is a single fold split dict or `None` if test is disabled. This tuple return enforces structural separation between CV and test at the API level.

```python
cv_folds, test_split = generate_folds(df, config.cross_validation, config.test)
# cv_folds = {"fold_0": {"train": df, "val": df}, "fold_1": ...}
# test_split = {"train": df, "val": df} or None
```

**`fit_transforms(train_df, transform_config)`** — Fits the transform chain on training data. Returns fitted transform objects and the transformed training DataFrame.

```python
fitted_transforms, transformed_train_df = fit_transforms(train_df, config.transforms)
```

**`apply_transforms(df, fitted_transforms)`** — Applies already-fitted transforms to new data (e.g., validation set, future exogenous data). Returns the transformed DataFrame.

```python
transformed_val_df = apply_transforms(val_df, fitted_transforms)
```

**`resolve_model(model_config)`** — Dynamically imports the model callable from the config's import path and extracts hyperparameters. Returns the callable and the hyperparameters dict. This is for power users who want to call the model directly with full control over arguments.

```python
model_fn, hyperparameters = resolve_model(config.model)
forecast_df = model_fn(transformed_train_df, horizon=12, **hyperparameters)
```

**`invoke_model(train_df, model_config, horizon, future_x_df=None)`** — Convenience wrapper that resolves the model callable, invokes it, and unpacks the return tuple into its three components (forecast, fitted values, model object), with `None` for missing elements. The `future_x_df` parameter passes future exogenous data through to the model callable. This is an explicit named parameter (not `**kwargs`) because future exogenous data is a runtime data input — like `train_df` — rather than an arbitrary config override. This gives the same behavior as `run_backtest`'s internal model invocation.

```python
forecast_df, fitted_values_df, model_object = invoke_model(
    transformed_train_df, config.model, config.cross_validation.horizon,
    future_x_df=future_x_df,
)
```

**`inverse_transforms(forecast_df, fitted_transforms)`** — Applies inverse transforms in reverse order to return forecasts to the original scale of `y`.

```python
forecast_original_scale = inverse_transforms(forecast_df, fitted_transforms)
```

**`evaluate_metrics(y_true, y_pred, y_train, metrics_config)`** — Computes all configured metrics. Handles simple and context-aware metrics, tuple returns with instability flags, and per-series/group/global scope routing.

```python
metrics_df = evaluate_metrics(
    y_true=val_df,
    y_pred=forecast_original_scale,
    y_train=train_df,
    metrics_config=config.metrics,
)
```

**`serialize_model(model_object, serialization_config, output_path)`** — Best-effort model serialization. Supports `pickle`, `cloudpickle`, `joblib`, `model_method`, and user-provided callables.

```python
serialize_model(model_object, config.model.serialization, output_path="model_fold_1.pkl")
```

### 3.3 Step Function Locations

Step functions are located according to the one-way dependency rule:

| Function             | Location               | Rationale                                    |
| -------------------- | ---------------------- | -------------------------------------------- |
| `parse_config`       | `tsbricks.backtesting` | Config model is backtest-specific in V1      |
| `generate_folds`     | `tsbricks.backtesting` | Fold generation is backtest-specific         |
| `evaluate_metrics`   | `tsbricks.backtesting` | Metric evaluation is backtest-specific in V1 |
| `fit_transforms`     | `tsbricks.runner`      | Shared between backtesting and forecasting   |
| `apply_transforms`   | `tsbricks.runner`      | Shared between backtesting and forecasting   |
| `inverse_transforms` | `tsbricks.runner`      | Shared between backtesting and forecasting   |
| `resolve_model`      | `tsbricks.runner`      | Shared between backtesting and forecasting   |
| `invoke_model`       | `tsbricks.runner`      | Shared between backtesting and forecasting   |
| `serialize_model`    | `tsbricks.runner`      | Shared between backtesting and forecasting   |

### 3.4 Full Power-User Example

```python
import pandas as pd
from tsbricks.backtesting import parse_config, generate_folds, evaluate_metrics
from tsbricks.runner import fit_transforms, apply_transforms, inverse_transforms, invoke_model

df = pd.read_parquet("gs://my-bucket/data/sales_with_features.parquet")
config = parse_config(config_path="config.yaml")

# generate_folds returns a tuple: (cv_folds dict, test_split or None)
cv_folds, test_split = generate_folds(df, config.cross_validation, config.test)

for fold_id, fold_split in cv_folds.items():
    train_df, val_df = fold_split["train"], fold_split["val"]

    # Fit transforms on training data
    fitted_transforms, transformed_train = fit_transforms(train_df, config.transforms)
    transformed_val = apply_transforms(val_df, fitted_transforms)

    # Inspect transformed data before modeling
    print(f"Fold {fold_id}: transformed_train shape = {transformed_train.shape}")
    print(f"Box-Cox lambda: {fitted_transforms[0].get_fitted_params()}")

    # Invoke model (future_x_df is optional, for models that use future exogenous data)
    forecast_df, fitted_values_df, model_object = invoke_model(
        transformed_train, config.model, config.cross_validation.horizon,
        future_x_df=transformed_val[["ds", "unique_id", "price", "promotions"]],
    )

    # Inverse transform forecasts to original scale
    forecast_original_scale = inverse_transforms(forecast_df, fitted_transforms)

    # Evaluate metrics
    metrics_df = evaluate_metrics(
        y_true=val_df, y_pred=forecast_original_scale,
        y_train=train_df, metrics_config=config.metrics
    )
    print(f"Fold {fold_id} RMSSE: {metrics_df.query('metric_name == \"rmsse\"')['value'].mean():.4f}")

# Process test fold separately if enabled
if test_split is not None:
    train_df, test_df = test_split["train"], test_split["val"]
    fitted_transforms, transformed_train = fit_transforms(train_df, config.transforms)
    forecast_df, fitted_values_df, model_object = invoke_model(
        transformed_train, config.model, config.cross_validation.horizon,
    )
    forecast_original_scale = inverse_transforms(forecast_df, fitted_transforms)
    test_metrics = evaluate_metrics(
        y_true=test_df, y_pred=forecast_original_scale,
        y_train=train_df, metrics_config=config.metrics
    )
    print(f"Test RMSSE: {test_metrics.query('metric_name == \"rmsse\"')['value'].mean():.4f}")
```

______________________________________________________________________

## 4. Data Format

### 4.1 Input Convention

The system accepts pandas DataFrames in Nixtla long format. Polars DataFrames are also accepted at the `run_backtest` entry point and are converted to pandas internally (see section 4.5). All internal processing uses pandas.

The user's DataFrame may use any column names for the target, date, and ID columns. The config's `data` section maps user column names to the internal standard names:

| Internal Name | Config Field | Description                            |
| ------------- | ------------ | -------------------------------------- |
| `y`           | `target_col` | Target variable                        |
| `ds`          | `date_col`   | Timestamp column                       |
| `unique_id`   | `id_col`     | Unique identifier for each time series |

`run_backtest` renames the user's columns to `y`, `ds`, `unique_id` at entry, immediately after Polars conversion (if applicable). **All internal processing, step functions, and output DataFrames use the standard names `y`, `ds`, `unique_id`.** Output DataFrames are not renamed back to the user's original column names — results consistently use the standard names.

**Step functions** (`fit_transforms`, `apply_transforms`, `invoke_model`, etc.) expect DataFrames with the standard column names `y`, `ds`, `unique_id`. A power user calling step functions directly is responsible for providing DataFrames with these column names.

Exogenous variables (`X_exogenous`) are provided as additional columns in the same DataFrame that contains `ds`, `unique_id`, and `y`. The config's `exogenous_columns` field specifies which columns are exogenous features. If no exogenous variables are used, `exogenous_columns` is `None` or an empty list. The user is responsible for joining exogenous data into the target DataFrame before passing it to the system.

### 4.2 Data Loading

The user loads data externally and passes DataFrames into the system programmatically. The system does not perform file I/O for data loading in V1.

The `run_backtest` function accepts the configuration as either a YAML file path or a Python dictionary. The two are mutually exclusive. **The YAML file is the preferred approach** for reproducibility and readability. The dictionary option is provided for programmatic use cases such as hyperparameter optimization with Optuna, where configs are generated dynamically.

```python
import pandas as pd
from tsbricks.backtesting import run_backtest

df = pd.read_parquet("gs://my-bucket/data/sales_with_features.parquet")

# Preferred: from YAML file
results = run_backtest(config_path="config.yaml", df=df)

# Alternative: from dict (e.g., generated programmatically by Optuna)
results = run_backtest(config=config_dict, df=df)
```

### 4.3 Panel and Single Series Support

The system supports both single time series and panel data (multiple `unique_id` values). The cross-validation, transform, and evaluation logic applies identically in both cases; a single series is treated as panel data with one `unique_id`.

### 4.4 Temporal Granularity

The system supports hourly, daily, weekly, and monthly time series. **The frequency must be specified explicitly in the config's `data` section via the `freq` field** (e.g., `freq: "MS"` for month-start, `freq: "D"` for daily). The system does not infer frequency from the data.

**Why explicit frequency is required.** `pd.infer_freq()` is unreliable in practice: it fails on series with gaps, returns `None` for short series, can produce ambiguous results for monthly data (`"M"` vs `"MS"` vs `"ME"`), and behaves inconsistently across pandas versions. In a panel setting, different series may have different gap patterns, making inference even less reliable. Requiring the user to specify `freq` eliminates this class of subtle bugs. The `freq` value is used for parametric fold origin computation and validation.

### 4.5 Polars Conversion

Polars DataFrames are accepted at the `run_backtest` entry point. The conversion to pandas happens at the very top of `run_backtest`, before column renaming and before any other processing. The step functions (`fit_transforms`, `invoke_model`, etc.) only accept pandas — they never see Polars. A power user calling step functions directly is responsible for providing pandas DataFrames.

This design keeps V1 simple while establishing the entry point where native Polars support can be added in V2 (see section 15).

______________________________________________________________________

## 5. Cross-Validation

### 5.1 Windowing Strategy

V1 uses an **expanding window** approach. For each fold, the training set is anchored at the start of the data and extends to the fold's forecast origin. The validation set begins immediately after the forecast origin and extends for the fixed forecast horizon.

Each time series, identified by a `unique_id`, may have a different initial (oldest) data point. However, the most recent data point must be the same across all time series in the panel. The shared most recent date is what the system uses when computing forecast origins in parametric mode.

If a computed forecast origin falls before a particular series' initial data point, that series is excluded from that fold with a logged error for that series and fold combination. This follows the resilient execution pattern described in section 13 — it does not halt the run.

The system is designed to accommodate sliding window in a future version.

### 5.2 Fold Specification Modes

The user specifies folds in one of two mutually exclusive modes:

**Explicit forecast origins.** The user provides a list of forecast origin dates. Each date defines where the training set ends and the forecast begins. These forecast origin dates are the date of the last and most recent data point in the cross-validation training sets.

```yaml
cross_validation:
  mode: explicit
  horizon: 12
  forecast_origins:
    - "2023-01-01"
    - "2023-04-01"
    - "2023-07-01"
    - "2023-10-01"
```

**Parametric.** The user specifies the number of folds and the step size (increment to slide the forecast origin). The system computes forecast origin dates by working backward from a boundary date. The boundary date depends on whether the test fold is enabled:

- **Test enabled:** The boundary is the `test_origin` date (the forecast origin of the test period).
- **Test disabled:** The boundary is the last date in the data (`data_end`).

The most recent CV fold origin is computed as `boundary - horizon` periods. Each subsequent fold steps backward by `step_size` periods. This ensures CV validation windows never overlap with the test period.

**Example.** Monthly data ending 2024-12-01, `horizon: 3`, `n_folds: 4`, `step_size: 2`, `test_origin: 2024-06-01`:

```
Boundary = test_origin = 2024-06-01 (test fold trains up to here)
Most recent CV origin = 2024-06-01 - 3 months = 2024-03-01
fold_3 origin: 2024-03-01 (validates 2024-04, 2024-05, 2024-06)
fold_2 origin: 2024-01-01 (validates 2024-02, 2024-03, 2024-04)
fold_1 origin: 2023-11-01 (validates 2023-12, 2024-01, 2024-02)
fold_0 origin: 2023-09-01 (validates 2023-10, 2023-11, 2023-12)
Test fold:     trains up to 2024-06-01, validates 2024-07, 2024-08, 2024-09
```

```yaml
cross_validation:
  mode: parametric
  horizon: 3
  n_folds: 4
  step_size: 2  # periods to shift origin between folds
```

### 5.3 Fold Naming Convention

CV fold dicts use zero-indexed, dynamically zero-padded keys. The padding width is determined by the maximum fold index: `fold_0`, `fold_1`, ..., `fold_9` for 1–10 folds (indices 0–9, single digit); `fold_00`, `fold_01`, ..., `fold_10` for 11 folds (indices 0–10, two digits needed); up to `fold_00`, ..., `fold_99` for 100 folds. Folds are ordered chronologically by origin date (fold_0 has the earliest origin).

### 5.4 Fixed Forecast Horizon

The forecast horizon is constant across all folds within a run, including the test fold. The forecast horizon is the number of time steps to forecast for each cross-validation fold. It is specified in the `cross_validation` section of the config.

### 5.5 Test Fold

The system supports a separate test fold that is structurally isolated from the cross-validation folds. The test fold is processed as a completely separate phase after cross-validation: the model is trained on all data up to the test origin, forecasts are generated for the same horizon, the same transforms and inverse transforms are applied, and the same metrics are computed. Test results are stored in a separate dataclass (`TestResults`) to prevent accidental mixing with CV results during model selection.

**`test_origin` is required when `test.enabled: true`.** The test origin is the forecast origin of the test period — the test set begins at the first time step after the test origin. This is consistent with the CV origin terminology: a "forecast origin" is always the last training observation, and the forecast begins immediately after it.

**`test_origin` serves as the CV/test boundary.** When test is enabled, parametric CV computes fold origins backward from `test_origin` (see section 5.2). This guarantees that no CV validation window overlaps with the test period.

**Validation:** `test_origin + horizon <= data_end`. The test fold's validation window must fit within the available data.

**Warning when `test_origin` provided but test disabled.** If `test.enabled: false` but `test_origin` is provided, the system logs a warning to alert the user that the test origin is being ignored. This is not an error — it may be intentional during iterative development.

The test fold is enabled by default but can be disabled by the user. When disabled, test result fields are `None`.

```yaml
test:
  enabled: true
  test_origin: "2024-06-01"
```

### 5.6 Internal Representation

Both the "explicit forecast origin" and the "parametric" modes produce the same internal representation: an ordered list of forecast origin dates. All downstream logic (splitting, fitting, evaluating) operates on this list without knowledge of which mode produced it. The test fold origin is handled separately from the CV fold origins.

______________________________________________________________________

## 6. Transforms

### 6.1 Overview

Transforms are a small, ordered chain of operations applied to the data before modeling and (for transforms on `y` with `perform_inverse: true`) inverted after forecasting and before metric evaluation. Transforms are applied sequentially in the order specified in the config.

### 6.2 Transform Object Interface

Each transform is a Python object that extends `BaseTransform` and implements the following methods. The interface operates on **DataFrames** with at least `ds`, `unique_id`, and the column being transformed. The column name is passed explicitly via `target_col`. Each call operates on one column at a time — if a transform config specifies `targets: [y, revenue]`, the system calls `fit_transform` once for `y` and once for `revenue`.

```python
from tsbricks.blocks.transforms import BaseTransform

class SomeTransform(BaseTransform):
    def fit_transform(self, df: pd.DataFrame, target_col: str, **params) -> pd.DataFrame:
        """Fit parameters from data and return DataFrame with target_col transformed.
        For fixed-parameter transforms, reads params without fitting.
        The transform fits independently per (unique_id, target_col) combination."""
        ...

    def transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Apply already-fitted parameters to new data (e.g., validation set).
        Returns DataFrame with target_col transformed."""
        ...

    def inverse_transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Reverse the transform. Used on forecasted y.
        Returns DataFrame with target_col inverse-transformed."""
        ...

    def get_fitted_params(self) -> dict:
        """Return dict of fitted or fixed parameters for artifact logging.
        Values must be plain Python types (float, int, str), not numpy scalars."""
        ...
```

**`BaseTransform`** is an abstract base class provided by `tsbricks.blocks.transforms`. It defines the four-method interface and provides a default `_map_per_series` helper method that isolates per-series iteration into a single swappable location. All built-in transforms extend `BaseTransform`. User-provided transforms should also extend `BaseTransform` but this is not strictly enforced at runtime.

The `fit_transform` method is called on each fold's training data. The `transform` method is called on the validation set using the parameters fitted from the training set. The `inverse_transform` method is applied to the model's forecasted `y` to return predictions to the original scale before metric evaluation.

Inverse transforms are only applied to `y`. Exogenous variables are model inputs and do not require inverse transformation after forecasting.

### 6.3 Scope

Each transform declares a scope:

- **`per_series`**: The transform fits independently for each `unique_id`. Fitted parameters may differ across series. The full panel DataFrame is passed to the transform; the transform uses `_map_per_series` internally to iterate over `unique_id` values. Examples: Box-Cox, StandardScaler.
- **`global`**: A single transform instance is shared across all series. The full panel DataFrame is passed directly — no splitting or concatenation is needed. The transform fits once on the entire panel. Examples: trading day normalization using a shared calendar, company working day normalization using company holidays, StandardScaler applied once over the entire panel dataset, or a transformation from nominal dollars to real dollars (V1 assumes all series share the same currency).

The transform scope implementation is designed to accommodate a future `per_group` scope (e.g., applying different transforms to subsets of series grouped by currency or product class) without architectural changes.

### 6.4 Targets and Multi-Target Handling

Each transform declares which columns it operates on via the `targets` field. The reserved keyword `y` refers to the target variable. Any other name refers to a specific column(s) corresponding to the exogenous variables. Different exogenous variables can receive different transforms.

When a transform specifies multiple targets (e.g., `targets: [y, revenue]`), the system calls `fit_transform` once for each target column. Each `(unique_id, target_col)` combination is fitted independently — there is no parameter sharing across columns. This one-column-per-call design keeps the transform interface simple and avoids ambiguity about which column's parameters are being referenced.

```yaml
transforms:
  - name: trading_day_normalization
    class: tsbricks.blocks.transforms.TradingDayNormalization
    scope: global
    targets: [y, revenue]
    perform_inverse_transform: true
    params:
      calendar: NYSE

  - name: log_transform
    class: tsbricks.blocks.transforms.NaturalLogTransform
    scope: per_series
    targets: [price]
    perform_inverse_transform: false
```

### 6.5 The `perform_inverse_transform` Flag

Each transform declares `perform_inverse_transform` in the config. This flag controls whether the inverse transform is applied to `y` after forecasting:

- **`perform_inverse_transform: true`**: The system applies `inverse_transform` to the model's forecasted `y` to return predictions to the original scale before metric evaluation. This is the typical case for transforms on `y`.
- **`perform_inverse_transform: false`**: The system skips inverse transformation. This is appropriate for transforms on exogenous columns (which are model inputs, not outputs) or for transforms where evaluation on the transformed scale is desired.

**Inverse transforms are only applied to `y`.** If a transform specifies `perform_inverse_transform: true` and targets both `y` and non-`y` columns (e.g., `targets: [y, revenue, price]`), the system logs a warning listing the non-`y` columns for which inverse transform will not be performed. The inverse is applied to `y` only; exogenous columns are model inputs and do not appear in the forecast output.

**V1 constraint:** `perform_inverse_transform` must be `true` for transforms targeting `y`. If the user sets `perform_inverse_transform: false` on a transform targeting `y`, the system raises `NotImplementedError` with a message explaining that evaluation on the transformed scale is deferred to a future version (see section 15). Transforms targeting only exogenous columns may have `perform_inverse_transform: false`.

### 6.6 Fixed vs. Fitted Parameters

Transforms support both modes:

- **Fitted parameters.** The `fit_transform` method learns parameters from the training data (e.g., Box-Cox lambda, StandardScaler mean and std). These parameters are refit on each fold's training data to avoid data leakage. For fitted parameters, the user may optionally provide an allowed range within which the fitted parameter must be found. For some fitted parameters, the user may also specify the objective for the parameter fitting such as for the Box Cox transform the `loglik` (log likelihood) or `guerrero` may be specified by the user.
- **Fixed parameters.** The user supplies a parameter value in the config (e.g., `power: 0.3333` for a cube root). The `fit_transform` method uses the supplied value without fitting.

There is also the ability to have overrides per-series where the user specifies the `unique_id` of the individual time series and the parameter overrides.

The transform object decides internally whether to fit or use a fixed value based on the `**params` it receives. The system treats all transforms identically, always calling `fit_transform` on each fold's training data.

```yaml
# Fitted from data
- name: box_cox
  class: tsbricks.blocks.transforms.BoxCoxTransform
  scope: per_series
  targets: [y]
  perform_inverse_transform: true
  params:
    lambda_range: [0, 2]

# Fixed cube root
- name: cube_root
  class: tsbricks.blocks.transforms.PowerTransform
  scope: per_series
  targets: [y]
  perform_inverse_transform: true
  params:
    power: 0.3333
    fixed: true

# Per-series overrides: fixed for some, fitted for others
- name: box_cox
  class: tsbricks.blocks.transforms.BoxCoxTransform
  scope: per_series
  targets: [y]
  perform_inverse_transform: true
  params:
    lambda_range: [0, 2]
    series_overrides:
      SKU_001: {fixed_lambda: 0.3333}
      SKU_002: {fixed_lambda: 0.5}
```

### 6.7 Model-Native Transforms

Some forecast models handle certain transforms internally (e.g., AutoTBATS accepts `box_cox: true` with `bc_lower_bound` and `bc_upper_bound`; AutoARIMA accepts a Box-Cox `lambda` parameter). When `model_native: true` is set on a transform, the system skips external application of that transform and instead merges the transform's parameters into the model callable's `**kwargs`.

The `model_params_mapping` field translates between transform parameter names and the model's expected parameter names. The parameter names should be named identically as the model's expected parameter names.

```yaml
- name: box_cox
  model_native: true
  model_params_mapping:
    box_cox: true
    bc_lower_bound: 0
    bc_upper_bound: 1
```

### 6.8 Built-in Transforms

The system ships with built-in transform classes for common operations. All built-in transforms extend `BaseTransform`. Users can provide custom transforms following the same four-method interface (extending `BaseTransform` is recommended but not enforced at runtime).

**Mathematical transforms (wrap NumPy directly):**

- **NaturalLogTransform**: Applies `np.log` / `np.exp`. Strict — raises an error if the column contains values ≤ 0. The user should chain an `AddConstantTransform` before this transform if the data contains non-positive values.
- **Log1pTransform**: Applies `np.log1p` (i.e., `log(1 + x)`) / `np.expm1`. Numerically stable for small values and naturally handles zeros.
- **PowerTransform**: Applies `column ** power` / `column ** (1/power)`. Supports any user-specified power (e.g., `power: 0.3333` for cube root).
- **AddConstantTransform**: Adds/subtracts a user-specified constant.

**Statistical transforms:**

- **BoxCoxTransform**: Wraps Nixtla's coreforecast (`coreforecast.scalers.boxcox` and `coreforecast.scalers.inv_boxcox`). Supports a `method` parameter (`guerrero` or `loglik`) for lambda selection and a `season_length` parameter required by the Guerrero method. These are simple params passthroughs — the additional parameters in the config are passed to the transform object's `fit_transform` via `**params`, and the system does not require any special handling for them.
- **StandardScaler**: Fits mean and standard deviation from training data. This is a custom implementation (not wrapping sklearn or coreforecast) to keep dependencies minimal and maintain full control over the per-series iteration pattern. The name `StandardScaler` aligns with the established convention in sklearn and other ML packages.

**Domain-specific transforms:**

- **TradingDayNormalization**: Normalizes by trading day count using a shared calendar.

```yaml
# Natural log (strict — errors on non-positive values)
- name: natural_log
  class: tsbricks.blocks.transforms.NaturalLogTransform
  scope: per_series
  targets: [y]
  perform_inverse_transform: true

# Log1p (handles zeros)
- name: log1p
  class: tsbricks.blocks.transforms.Log1pTransform
  scope: per_series
  targets: [y]
  perform_inverse_transform: true

# Box-Cox with Guerrero method
- name: box_cox
  class: tsbricks.blocks.transforms.BoxCoxTransform
  scope: per_series
  targets: [y]
  perform_inverse_transform: true
  params:
    method: guerrero
    season_length: 12
    lower_bound: 0
    upper_bound: 2
```

### 6.9 V1 Future-Proofing Patterns

The following implementation patterns are required in V1 to enable future acceleration (multiprocessing, Polars native support) without API-breaking changes:

1. **Never mutate input DataFrames.** All transform methods must operate on `df.copy()` and return new DataFrames. This ensures correctness under future parallel execution and prevents subtle bugs from shared mutable state.

1. **Store fitted parameters as plain Python types.** `get_fitted_params()` must return `float`, `int`, `str` — not `numpy.float64` or other numpy scalars. This ensures serialization compatibility and avoids type-related surprises in downstream consumers.

1. **No DataFrame index dependency.** All operations must be column-based, never relying on the DataFrame's index. This eliminates a class of bugs when DataFrames are sliced, concatenated, or converted between pandas and Polars (which has no index concept).

1. **Isolate per-series iteration into `_map_per_series`.** `BaseTransform` provides a `_map_per_series(df, target_col, fn)` helper that groups by `unique_id`, applies `fn` to each group, and reassembles the result. All per-series transforms use this single entry point for iteration. This allows future acceleration (e.g., swapping the loop for `joblib.Parallel` or a Polars `group_by.map_groups`) by changing one method.

1. **`BaseTransform` base class.** All built-in transforms extend `BaseTransform`, which defines the four-method interface as abstract methods and provides `_map_per_series`. User-provided transforms should also extend `BaseTransform` but this is not enforced at runtime.

______________________________________________________________________

## 7. Model Interface

### 7.1 Simple Callable Convention

Models are user-provided callables with the following signature:

```python
def my_model(train_df: pd.DataFrame, horizon: int, **kwargs) -> pd.DataFrame:
    """
    Args:
        train_df: Training data in long format (ds, unique_id, y, and
                  optionally exogenous columns).
        horizon: Number of periods to forecast.
        **kwargs: Hyperparameters and other arguments from the config.
                  Includes future_x_df (pd.DataFrame | None) when future
                  exogenous data is available.

    Returns:
        One of the following:
        - DataFrame: Forecast only.
        - Tuple[DataFrame, DataFrame]: Forecast + in-sample fitted values.
        - Tuple[DataFrame, DataFrame | None, object]: Forecast + optional
          fitted values + fitted model object (for serialization).

    Forecast DataFrame schema: ds, unique_id, ypred
    Fitted values DataFrame schema: ds, unique_id, ypred
    """
    ...
```

The system imports the callable dynamically from the import path specified in the config and passes hyperparameters as `**kwargs`. Future exogenous data is passed via `invoke_model`'s explicit `future_x_df` parameter (see section 3.2), which the system then passes through to the model callable as a keyword argument.

### 7.2 Forecast and Fitted Values DataFrame Schema

Both the forecast DataFrame and the fitted values DataFrame use the same schema:

| Column      | Description                                    |
| ----------- | ---------------------------------------------- |
| `ds`        | Timestamp                                      |
| `unique_id` | Series identifier                              |
| `ypred`     | Predicted value (forecast or in-sample fitted) |

The column name `ypred` (no underscore) is the standard name for predictions throughout the system. This applies to both out-of-sample forecasts and in-sample fitted values.

### 7.3 Return Type Detection

The system detects the return type by length:

```python
# Forecast only
return forecast_df

# Forecast + fitted values
return forecast_df, fitted_values_df

# Forecast + fitted values + model object (for serialization)
return forecast_df, fitted_values_df, model_object

# Forecast + model object, no fitted values
return forecast_df, None, model_object
```

The model object (third element) is used by the serialization system when enabled. If the model object is `None` and serialization is enabled, the system logs a warning and skips serialization for that fold.

### 7.4 In-Sample Fitted Values

The system supports three cases for in-sample fitted values:

**Case 1: No fitted values available.** Some models (e.g., pretrained models like Amazon Chronos) do not produce in-sample fitted values. The model callable returns only a forecast DataFrame. The `fitted_values` and `fitted_values_model_scale` fields in `CVResults` and `TestResults` are `None`.

**Case 2: Model exposes fitted values.** Some models (e.g., Nixtla statsforecast, mlforecast) allow extraction of in-sample fitted values as a natural byproduct of the fitting process. In this case, the model callable optionally returns a tuple of `(forecast_df, fitted_values_df)` instead of just `forecast_df`.

The model returns fitted values on the **transformed (model) scale** — these are the fitted values as the model sees them, after transforms have been applied. The system stores these directly as `fitted_values_model_scale`. The system then applies inverse transforms to produce `fitted_values` on the original scale. Both are stored as separate fields in the results dataclasses:

- **`fitted_values`**: Fitted values on the original scale of `y` (after inverse transforms).
- **`fitted_values_model_scale`**: Fitted values on the transformed scale used during modeling (as returned by the model).

The system detects the return type. If the callable returns a DataFrame, there are no fitted values. If it returns a tuple, the system unpacks the forecast and fitted values.

```python
# Case 1 — no fitted values
def chronos_model(train_df, horizon, **kwargs):
    ...
    return forecast_df

# Case 2 — model provides fitted values (on transformed/model scale)
def arima_model(train_df, horizon, **kwargs):
    ...
    fitted_values_df = ...  # extracted from the fitted model, schema: ds, unique_id, ypred
    return forecast_df, fitted_values_df
```

The user can compute residuals from fitted values in whichever convention they prefer (e.g., `y - ŷ` or `ŷ - y`) since the training `y` is available in the train/validation split DataFrames.

**Case 3: System-computed fitted values (out of scope for V1).** Computing in-sample fitted values by re-running the model on the training data is deferred to a future version due to the complexity of varying model semantics across different forecasting libraries.

### 7.5 Model Configuration

```yaml
model:
  callable: my_project.models.my_arima_model
  hyperparameters:
    order: [1, 1, 1]
    seasonal_order: [1, 1, 1, 12]
  model_n_jobs: 4
```

The `model_n_jobs` parameter is passed through to the model callable for models that support internal parallelization (e.g., `n_jobs` in statsforecast). Models that do not support it (e.g., Chronos) simply ignore it.

### 7.6 Model Serialization

Model serialization is opt-in via the config. When enabled, the system attempts to serialize the fitted model on a best-effort basis after each fold. If serialization fails, the system logs a warning and continues.

The user specifies the serialization method:

- **`pickle`**, **`cloudpickle`**, **`joblib`**: The system serializes the model object externally using the specified library.
- **`model_method`**: The system calls a method on the fitted model object itself (e.g., `.save()`). This is the recommended approach for Nixtla packages (statsforecast, mlforecast, neuralforecast), which all provide `.save()` and `.load()` methods on fitted models. The `save_method` field specifies the method name and defaults to `save`.
- **User-provided callable**: An import path to a custom serialization function.

All methods require the model callable to return the fitted model object as the third element of the return tuple (see section 7.1). If the model object is not returned and serialization is enabled, the system logs a warning and skips serialization.

```yaml
# Using Nixtla's built-in .save() method
model:
  callable: my_project.models.my_statsforecast_model
  hyperparameters: {}
  model_n_jobs: 4
  serialization:
    enabled: true
    method: model_method
    save_method: save       # method name on the model object, defaults to "save"

# Using joblib
model:
  callable: my_project.models.my_model
  hyperparameters: {}
  serialization:
    enabled: true
    method: joblib

# Using a custom serialization callable
model:
  callable: my_project.models.my_model
  hyperparameters: {}
  serialization:
    enabled: true
    method: my_project.serializers.custom_save
```

______________________________________________________________________

## 8. Metrics

### 8.1 Two Callable Types

Metrics are user-provided callables in one of two types:

**Simple metrics.** Require only actuals and predictions.

```python
def mae(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    ...
```

**Context-aware metrics.** Additionally receive the training data for computing scale-dependent measures.

```python
def rmsse(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, **kwargs) -> float:
    ...
```

The system determines which arguments to pass based on the `type` field in the config. Extra parameters (e.g., `m` for lag differences, `fallback_scale`) are passed via `**kwargs` from the config.

### 8.2 Tuple Returns for Diagnostics

Metrics may return either a `float` or a `Tuple[float, bool]` where the boolean is an instability flag (e.g., degenerate scale in RMSSE). The system detects the return type and logs instability flags as metadata alongside metric values.

### 8.3 Aggregation Modes

Each metric specifies an aggregation mode in the config:

- **`per_fold_mean`**: The metric is computed per fold, then averaged across folds.
- **`pooled`**: Actuals and predictions (and training data for context-aware metrics) are concatenated across folds before computing the metric once.

### 8.4 Per-Series, Group, and Global Metrics

The system supports per-series metrics (e.g., RMSSE computed for each `unique_id`), group metrics (e.g., WAPE computed across a product category), and global metrics (e.g., WRMSSE computed across all series with aggregation weights). All three can coexist in a single run. The metric callable itself determines whether it operates per-series, per-group, or globally; the system routes data accordingly based on the `scope` field.

A single metric callable can be defined multiple times with different scopes to compute the same metric at different levels of granularity within a single run:

```yaml
metrics:
  definitions:
    - name: wape_per_series
      callable: tsbricks.blocks.metrics.wape
      type: simple
      scope: per_series
      aggregation: per_fold_mean

    - name: wape_by_category
      callable: tsbricks.blocks.metrics.wape
      type: simple
      scope: group
      aggregation: per_fold_mean
      grouping_columns: [product_category]
```

### 8.5 Metric Grouping

Metrics can be computed at intermediate grouping levels (e.g., by product class, country) in addition to per-series and global. Grouping columns can be specified at two levels:

- **Per metric definition**: A `grouping_columns` field on an individual metric definition applies only to that metric. This is required when the metric's `scope` is `group`.
- **Top-level default**: A `grouping_columns` field at the top level of the `metrics` config applies to all metric definitions that have `scope: group` but do not specify their own `grouping_columns`. Per-definition values override the top-level default.

### 8.6 Multiple Metrics Per Run

A single run can specify multiple metrics. All metrics are computed on the same backtest results.

### 8.7 Metrics Configuration

```yaml
metrics:
  definitions:
    - name: rmsse
      callable: tsbricks.blocks.metrics.rmsse
      type: context_aware
      scope: per_series
      aggregation: per_fold_mean
      params:
        m: 1
        return_components: false

    - name: scaled_bias
      callable: tsbricks.blocks.metrics.difference_scaled_bias
      type: context_aware
      scope: per_series
      aggregation: per_fold_mean
      params:
        m: 1
        scale_stat: rms

    - name: wrmsse
      callable: tsbricks.blocks.metrics.wrmsse
      type: context_aware
      scope: global
      aggregation: pooled
      params:
        weights_path: gs://my-bucket/data/weights.parquet

    - name: mae
      callable: tsbricks.blocks.metrics.mae
      type: simple
      scope: per_series
      aggregation: per_fold_mean

    - name: wape_by_category
      callable: tsbricks.blocks.metrics.wape
      type: simple
      scope: group
      aggregation: per_fold_mean
      grouping_columns: [product_category]  # per-definition override

    - name: wape_by_country
      callable: tsbricks.blocks.metrics.wape
      type: simple
      scope: group
      aggregation: per_fold_mean
      # uses top-level default grouping_columns

  grouping_columns: [country]  # default for group-scoped metrics
```

### 8.8 Built-in Metrics

The system ships with built-in implementations of common metrics: MAE, RMSE, MAPE, RMSSE, and difference-scaled bias. Users can provide custom metric callables following the same conventions.

______________________________________________________________________

## 9. Parallelization

### 9.1 Separation of Concerns

Parallelization is split between the model and the evaluation system:

- **Model fitting and prediction**: Parallelization is the model's responsibility. The system passes `model_n_jobs` through to the model callable via `**kwargs`. Models that support internal parallelization (e.g., statsforecast's `n_jobs`) use it; models that do not (e.g., Chronos) ignore it.
- **Forecast evaluation**: The system owns parallelization using Python's `multiprocessing` module.

### 9.2 Evaluation Parallelization Strategies

The user specifies the evaluation parallelization strategy in the config:

- **`across_series`**: Each `unique_id` is evaluated independently in parallel. Primary win for panel data.
- **`across_folds`**: Each cross-validation fold is evaluated in parallel for a given series.
- **`nested`**: Parallelization across both series and folds.

### 9.3 Configuration

```yaml
parallelization:
  parallel_eval_strategy: across_series
  eval_n_workers: 4       # -1 for all available cores
```

### 9.4 Environment Compatibility

The parallelization approach is compatible with Vertex AI pipeline worker nodes, Vertex AI notebook instances, and local laptops with multiple CPU cores.

______________________________________________________________________

## 10. Configuration File

### 10.1 Format

YAML. One model per config file. One run per config file.

### 10.2 Full Config Structure

```yaml
# --- Data ---
data:
  target_col: y
  date_col: ds
  id_col: unique_id
  freq: "MS"               # required — e.g., "MS" (month-start), "D" (daily), "W" (weekly), "h" (hourly)
  exogenous_columns: [price, promotions, temperature]

# --- Cross-Validation ---
cross_validation:
  mode: parametric          # or "explicit"
  horizon: 12
  n_folds: 4               # parametric mode only
  step_size: 3             # parametric mode only
  # forecast_origins:       # explicit mode only
  #   - "2023-01-01"
  #   - "2023-04-01"

# --- Test Fold ---
test:
  enabled: true
  test_origin: "2024-01-01"

# --- Transforms ---
transforms:
  - name: trading_day_normalization
    class: tsbricks.blocks.transforms.TradingDayNormalization
    scope: global
    targets: [y]
    perform_inverse_transform: true
    params:
      calendar: NYSE

  - name: box_cox
    class: tsbricks.blocks.transforms.BoxCoxTransform
    scope: per_series
    targets: [y]
    perform_inverse_transform: true
    model_native: false
    params:
      lambda_range: [0, 2]

# --- Model ---
model:
  callable: my_project.models.auto_arima_forecast
  hyperparameters:
    order: [1, 1, 1]
    seasonal_order: [1, 1, 1, 12]
  model_n_jobs: 4
  serialization:
    enabled: false
    method: model_method
    save_method: save

# --- Metrics ---
metrics:
  definitions:
    - name: rmsse
      callable: tsbricks.blocks.metrics.rmsse
      type: context_aware
      scope: per_series
      aggregation: per_fold_mean
      params:
        m: 1
    - name: wrmsse
      callable: tsbricks.blocks.metrics.wrmsse
      type: context_aware
      scope: global
      aggregation: pooled
      params: {}
    - name: mae
      callable: tsbricks.blocks.metrics.mae
      type: simple
      scope: per_series
      aggregation: per_fold_mean
  grouping_columns: [country]  # default for group-scoped metrics

# --- Parallelization ---
parallelization:
  parallel_eval_strategy: across_series
  eval_n_workers: 4

# --- Artifact Storage ---
artifact_storage:
  uv_lock_path: ./uv.lock
```

**V1 changes from initial design:**

- `data.freq` is required (see section 4.4 for rationale).
- `artifact_storage.output_path` is removed from V1. The system returns structured results as a Python dataclass; the user controls where and how artifacts are persisted.
- The `experiment_tracking` config section is removed from V1. The user controls logging to Vertex AI Experiments directly using the Vertex AI SDK (see section 12).

______________________________________________________________________

## 11. Output Structure

### 11.1 Output Dataclasses

The system returns a `BacktestResults` dataclass composed of separate dataclasses for CV results, test results, and shared metadata. This structural separation ensures test results are isolated from CV results during model selection.

```python
from dataclasses import dataclass, field
import pandas as pd


@dataclass(frozen=True)
class CVResults:
    """Cross-validation results. Used during model selection.
    Dict keys use fold naming convention: fold_0, fold_1, etc. (see section 5.3)."""
    # --- Always present ---
    forecasts_per_fold: dict[str, pd.DataFrame]       # fold_id -> DataFrame(ds, unique_id, ypred)
    metrics: pd.DataFrame
    fold_origins: list[pd.Timestamp]
    train_val_splits_per_fold: dict[str, dict[str, pd.DataFrame]]

    # --- Present depending on model/config ---
    fitted_values: dict[str, pd.DataFrame] | None = None              # original scale
    fitted_values_model_scale: dict[str, pd.DataFrame] | None = None  # transformed scale
    transform_params: dict[str, dict[str, dict]] | None = None
    metric_instability_flags: pd.DataFrame | None = None
    metric_groups: dict[str, pd.DataFrame] | None = None
    fitted_models: dict[str, bytes] | None = None


@dataclass(frozen=True)
class TestResults:
    """Test fold results. Structurally isolated from CV results."""
    forecasts: pd.DataFrame                            # DataFrame(ds, unique_id, ypred)
    metrics: pd.DataFrame
    test_origin: pd.Timestamp
    train_test_split: dict[str, pd.DataFrame]

    # --- Present depending on model/config ---
    fitted_values: pd.DataFrame | None = None              # original scale
    fitted_values_model_scale: pd.DataFrame | None = None  # transformed scale
    transform_params: dict[str, dict] | None = None
    metric_instability_flags: pd.DataFrame | None = None
    metric_groups: pd.DataFrame | None = None
    fitted_model: bytes | None = None


@dataclass(frozen=True)
class BacktestResults:
    """Top-level results containing CV results, test results, and shared metadata."""
    # --- CV results ---
    cv: CVResults

    # --- Shared metadata ---
    horizon: int
    config: dict
    git_hash: str
    uv_lock: str
    run_summary: dict

    # --- Test results (None if test fold disabled) ---
    test: TestResults | None = None

    # --- Escape hatch ---
    extra: dict | None = None
```

Note that `CVResults` uses per-fold structures (e.g., `dict[str, pd.DataFrame]`) because there are multiple CV folds, while `TestResults` uses single DataFrames because there is only one test fold. This makes the types precise to each context.

Both `fitted_values` and `fitted_values_model_scale` share the same DataFrame schema (`ds`, `unique_id`, `ypred`). The difference is the scale: `fitted_values` contains values inverse-transformed back to the original scale of `y`, while `fitted_values_model_scale` contains the model's raw fitted values on the transformed scale. Both are `None` when the model does not return fitted values (see section 7.4).

### 11.2 Metrics DataFrame Schema

The `metrics` DataFrame uses a long format that accommodates per-series, grouped, and global metrics in a single table:

| Column               | Description                                                             |
| -------------------- | ----------------------------------------------------------------------- |
| `metric_name`        | Name of the metric (e.g., `rmsse`, `wrmsse`, `mae`)                     |
| `unique_id`          | Series identifier, or `None` for global/grouped metrics                 |
| `fold`               | Fold identifier (e.g., `fold_0`) or `pooled`                            |
| `aggregation`        | One of `per_series`, `group`, `global`                                  |
| `value`              | Computed metric value                                                   |
| *(grouping columns)* | Optional columns (e.g., `product_class`, `country`) for grouped metrics |

### 11.3 Run Summary

The `run_summary` dict provides a structured overview of run health:

```python
{
    "total_series_attempted": 500,
    "successful_series": 487,
    "failed_series": 13,
    "warned_series": 52,
    "failure_rate": 0.026,
    "warning_rate": 0.104,
    "total_folds": 5,
    "failed_folds": 2,
    "failures": [
        {"unique_id": "SKU_042", "fold": "fold_3", "error": "ValueError: ..."},
        ...
    ],
    "warnings": [
        {"unique_id": "SKU_007", "fold": "fold_1", "warning": "ConvergenceWarning: ..."},
        ...
    ],
}
```

`warned_series` counts unique series with at least one warning, not total warning instances. The detailed `warnings` list captures every instance. Warnings are captured from model fitting using Python's `warnings.catch_warnings()` context manager.

______________________________________________________________________

## 12. Experiment Tracking

### 12.1 Approach

The system prepares structured artifacts in the `BacktestResults` dataclass. The user controls logging to Vertex AI Experiments directly using the Vertex AI SDK. The system does not abstract or wrap the Vertex AI Experiments API in V1.

**V1 note:** The `experiment_tracking` config section has been removed from V1. There is no system-managed experiment tracking configuration. The user is responsible for all Vertex AI Experiments SDK calls. A future version may add an experiment tracking abstraction with pluggable backends (see section 15).

### 12.2 Helper Utilities

The system provides helper utilities that make it convenient to extract artifacts from `BacktestResults` in formats suitable for Vertex AI Experiments logging (e.g., flattened metric dicts, serialized config, DataFrames as parquet bytes).

### 12.3 Artifact Manifest

The following artifacts are available for logging per run:

| Artifact                         | Source                           | Description                                 |
| -------------------------------- | -------------------------------- | ------------------------------------------- |
| YAML config                      | `config`                         | Full configuration for reproducibility      |
| **CV artifacts**                 |                                  |                                             |
| CV train/val splits              | `cv.train_val_splits_per_fold`   | Per-fold training and validation DataFrames |
| CV forecasts                     | `cv.forecasts_per_fold`          | Per-fold ypred aligned to y_true            |
| CV fitted values                 | `cv.fitted_values`               | In-sample fitted values, original scale     |
| CV fitted values (model scale)   | `cv.fitted_values_model_scale`   | In-sample fitted values, transformed scale  |
| CV metric values                 | `cv.metrics`                     | Per-series, grouped, and global metrics     |
| CV instability flags             | `cv.metric_instability_flags`    | Flags for degenerate metric computations    |
| CV grouped metrics               | `cv.metric_groups`               | Metrics by grouping columns                 |
| CV forecast origins              | `cv.fold_origins`                | Dates defining each CV fold                 |
| CV transform params              | `cv.transform_params`            | Fitted/fixed parameters per fold per series |
| CV fitted models                 | `cv.fitted_models`               | Serialized model per fold (when enabled)    |
| **Test artifacts**               |                                  |                                             |
| Test train/test split            | `test.train_test_split`          | Training and test DataFrames                |
| Test forecasts                   | `test.forecasts`                 | ypred aligned to y_true                     |
| Test fitted values               | `test.fitted_values`             | In-sample fitted values, original scale     |
| Test fitted values (model scale) | `test.fitted_values_model_scale` | In-sample fitted values, transformed scale  |
| Test metric values               | `test.metrics`                   | Per-series, grouped, and global metrics     |
| Test instability flags           | `test.metric_instability_flags`  | Flags for degenerate metric computations    |
| Test grouped metrics             | `test.metric_groups`             | Metrics by grouping columns                 |
| Test origin                      | `test.test_origin`               | Forecast origin date for test fold          |
| Test transform params            | `test.transform_params`          | Fitted/fixed parameters per series          |
| Test fitted model                | `test.fitted_model`              | Serialized model (when enabled)             |
| **Shared metadata**              |                                  |                                             |
| Forecast horizon                 | `horizon`                        | Fixed horizon for the run                   |
| uv.lock                          | `uv_lock`                        | Dependency lockfile contents                |
| Git hash                         | `git_hash`                       | Code version identifier                     |
| Run summary                      | `run_summary`                    | Success/failure/warning counts and details  |

______________________________________________________________________

## 13. Error Handling and Validation

### 13.1 Logging

The system uses Python's standard `logging` module. Each module creates a logger via `logging.getLogger(__name__)`. The system emits log records for errors, warnings, and informational events (e.g., skipped series, skipped folds, serialization failures). The caller controls log handler configuration, formatting, and destinations. This follows the standard Python convention for library code and integrates seamlessly with Vertex AI pipeline logging and GCP Cloud Logging.

### 13.2 Validation Implementation

- **Config validation** uses Pydantic models to parse and validate the YAML config, providing clear error messages for missing fields, wrong types, and invalid values.
- **Internal data structures** (e.g., `BacktestResults`) use standard dataclasses since they are constructed by the system with already-validated data.

### 13.3 Upfront Validation

The system validates as much as possible before beginning computation:

- YAML config parses without error.
- Required config sections and fields are present.
- `data.freq` is present and is a valid pandas frequency string.
- Callable import paths for model, metrics, and transforms resolve successfully.
- Required columns (mapped via `target_col`, `date_col`, `id_col`) exist in the input DataFrame.
- Exogenous columns specified in the config exist in the DataFrame.
- Forecast origins (explicit mode) fall within the data's date range. **Each explicit origin is validated against the data's `ds` values** — if an origin doesn't match any `ds` value, it is almost certainly a user error. This cheap upfront check prevents confusing downstream failures.
- Horizon is a positive integer.
- Parametric CV settings produce valid fold origins.
- No contradictory settings (e.g., `model_native: true` with `perform_inverse_transform: true` on the same transform).
- `perform_inverse_transform: false` on a transform targeting `y` raises `NotImplementedError` (V1 constraint — see section 6.5).
- Serialization method is valid if serialization is enabled.
- `test_origin` is required when `test.enabled: true`. Missing `test_origin` with test enabled is a validation error.
- `test_origin + horizon <= data_end`. The test fold's validation window must fit within the available data.
- Test fold origin (if test is enabled) is after the last CV forecast origin, ensuring no overlap between CV validation data and test data.
- If `config_path` and `config` are both provided, or neither is provided, raise an error.
- **Warning (not error):** If `test.enabled: false` but `test_origin` is provided, the system logs a warning that the test origin is being ignored.

### 13.4 Series-Level Failures

If a model callable or transform raises an exception for a specific `unique_id`, the system skips that series, logs the error (including the `unique_id`, fold, and exception details), and continues processing remaining series. Warnings emitted during model fitting are captured and logged but do not cause the series to be skipped.

### 13.5 Fold-Level Failures

If an entire fold fails (e.g., a global transform raises an exception), the system logs the error (including the fold identifier and exception details) and continues with remaining folds.

### 13.6 Run Summary

All errors and warnings are aggregated into the `run_summary` field of `BacktestResults`, providing counts, rates, and detailed failure/warning records for post-run triage.

______________________________________________________________________

## 14. Testing

### 14.1 Framework and Style

All tests use pytest in functional format not a class-based format. Fixtures are defined in `conftest.py`. Minimize setup/teardown methods in favor of fixtures wherever possible.

### 14.2 Scope

V1 includes unit tests for each component covering critical paths only. Tests are focused and limited in number (under 10 per function), tailored to the complexity of each component rather than targeting a coverage percentage.

### 14.3 Components Under Test

- **Config parsing and validation.** Valid configs parse correctly; invalid configs raise clear errors for each validation rule.
- **Cross-validation fold generation.** Both explicit and parametric modes produce correct forecast origin lists. Expanding window splits are correct.
- **Transforms.** `fit_transform`, `transform`, `inverse_transform`, and `get_fitted_params` produce correct results for built-in transforms. Fixed vs. fitted parameter paths work correctly. Per-series and global scope behave correctly.
- **Model callable loading.** Dynamic import from string path resolves correctly. Hyperparameters are passed through.
- **Metrics computation.** Simple and context-aware metrics produce correct values. Tuple returns with instability flags are handled. Pooled and per-fold aggregation produce correct results. Per-series and global scope work correctly.
- **Evaluation parallelization.** Results are equivalent whether run with 1 worker or multiple workers.
- **Error handling.** Failed series are skipped and logged. Failed folds are skipped and logged. Run summary accurately reflects successes, failures, and warnings.
- **Test fold.** Test fold splitting is correct and isolated from CV folds. Test fold uses the same transforms and metrics as CV. Test results are stored in `TestResults` and structurally separate from `CVResults`. Test fold is `None` when disabled.
- **BacktestResults construction.** Required fields must be present. Optional fields default to None. The `extra` dict works as a catch-all.

### 14.4 Fixtures

`conftest.py` provides shared fixtures including: synthetic single-series and panel DataFrames, sample YAML configs, simple model callables, simple and context-aware metric callables, and transform instances.

______________________________________________________________________

## 15. Future Considerations (Out of Scope for V1)

The following capabilities are acknowledged but deferred beyond V1:

- **Sliding window cross-validation.** The expanding window implementation is designed to accommodate this without architectural changes.
- **Integration tests.** End-to-end tests running `run_backtest` with synthetic data and verifying the full pipeline.
- **System-managed data loading.** Reading input data from file paths, GCS URIs, or bucket+prefix configurations specified in the YAML config.
- **Experiment tracking abstraction.** A generic logging interface with pluggable backends for MLflow and Vertex AI Experiments.
- **Model adapter pattern.** A formal adapter interface with `fit`, `predict`, and `get_fitted_values` methods, replacing or extending the simple callable convention.
- **BigQuery data source support.**
- **System-computed fitted values.** Computing in-sample fitted values by re-running the model on the training data, for models that do not natively expose fitted values through their API.
- **Per-group transform scope.** A `per_group` scope for transforms that apply to subsets of series grouped by a column (e.g., currency, product class), enabling different transform parameters per group.
- **Metric evaluation on transformed scale.** Option to compute metrics on the transformed scale (skipping inverse transforms) rather than the original scale of `y`. This is relevant for cases like nominal-to-real currency transforms where evaluation on the transformed scale may be preferred. Deferred due to complexity: it affects the transform pipeline, metric computation (context-aware metrics need `y_train` on the same scale), residual scale consistency, and artifact metadata tracking.
- **Native Polars support.** Full native Polars support throughout the internal pipeline (transforms, metrics, model callables) for performance benefits, replacing the V1 approach of converting Polars to pandas at the entry point.
- **Custom company working day normalization.** A built-in transform that supports working day normalization based on a user-provided company working day calendar, accommodating company-specific holidays and non-standard schedules.
- **Transform parallelization.** Parallelization of per-series transform fitting across `unique_id` values, which is especially important when transforms involve a fitting process to find a parameter (e.g., Box-Cox lambda via Guerrero method). V1 parallelizes model fitting (via model's own `n_jobs`) and evaluation (via the system's `multiprocessing`), but transform fitting is sequential.
- **Native hyperparameter optimization.** Built-in integration with Optuna or similar frameworks, allowing the config to specify hyperparameter search spaces instead of fixed values and having the system manage the optimization loop internally. V1 supports hyperparameter optimization via external orchestration using the dict-based config option.
- **Forward-looking forecast (`run_forecast`).** A second entry point that reuses the same YAML config, transform pipeline, and model invocation primitives to produce a forward-looking forecast on the full dataset. The V1 internal architecture is designed to support this with minimal additional code.
