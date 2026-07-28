# Changelog

All notable changes to this project will be documented in this file.

The format is based on **Keep a Changelog** (https://keepachangelog.com/en/1.1.0/),
and this project adheres to **Semantic Versioning** (https://semver.org/).

## [Unreleased]

### Added

- **`invoke_predict`** — Predict-only counterpart to `invoke_model`. Runs a model's forecast step against an already-fitted model object without refitting, returning the forecast DataFrame (`tsbricks.runner.invoke_predict`). Signature is `invoke_predict(fitted_model, model_config, horizon, future_x_df=None)` which is the same as `invoke_model` except that `train_df` becomes `fitted_model`.
- **`resolve_predict`** — Mirror of `resolve_model` for the predict-only callable. Returns `(predict_fn, predict_params)` so a caller predicting repeatedly can resolve once and call the function directly (`tsbricks.runner.resolve_predict`).
- **`ModelConfig.predict_callable`** — Optional dotted path to a predict-only callable. Not consumed by `run_backtest`, which fits per fold and has no predict-only step.
- **`ModelConfig.predict_params`** — Optional predict-time parameters (e.g. prediction-interval levels). Read by **both** entrypoints, so a project using both declares them once instead of duplicating them across fit and serve.
- **Public `dynamic_import`** — Now exported from `tsbricks.runner`. Resolves any dotted `module.attribute` path.

### Changed

- **BREAKING** — **`ModelConfig.callable` renamed to `fit_predict_callable`** — The field fits *and* forecasts; with `predict_callable` beside it the old name was ambiguous, and it shadowed the Python builtin. No alias or shim: a Pydantic alias would accept the old YAML key while leaving `cfg.model.callable` raising `AttributeError`, breaking Python-constructed and duck-typed configs. Every config must rename the key. `MetricDefinitionConfig.callable` and `ParamResolverConfig.callable` are unchanged.
- **BREAKING** — **`tsbricks.runner._utils` renamed to `tsbricks.runner.utils`** — Hard rename, no compatibility shim. The module contains one function which is now public API. Import `dynamic_import` from `tsbricks.runner` instead.
- **`invoke_model` forwards `predict_params`** — Its callable fits *and* forecasts, so it needs predict-time parameters too. Configs that do not set `predict_params` are unaffected.
- **Overlapping keys emit a `UserWarning` at config-parse time** — When `hyperparameters` and `predict_params` both set a key to *differing* values, `ModelConfig` validation names the overlapping keys and states that `hyperparameters` governs the combined fit-and-forecast call. Overlapping keys with identical values are silent so do not warn. The overlap can be deliberate (fit on 16 cores, serve on 2), so this warns rather than raising.
- **BREAKING** — **A `horizon` key in either params dict now raises `ValueError`** — `horizon` is passed positionally by tsbricks. Almost all such configs already failed, with `TypeError: got multiple values for argument 'horizon'` naming the callable rather than the config at fault. One shape did work and is now rejected: a callable whose horizon parameter is named something else (`def f(train_df, h, **kwargs)`), which let the config's `horizon` fall into `**kwargs`. Remove the key. `future_x_df` is deliberately not guarded: it works today, and a caller-supplied argument overrides it.
- **Model-invocation error messages now name the offending callable** — Exception types are unchanged; only message text differs, and message text is not a stable API.
- **`resolve_model` return values expanded** — It now returns the complete config-derived kwargs which are `{**predict_params, **hyperparameters}`. Previously, `resolve_model` returned `hyperparameters` only. Impacts only callers external to the package using `resolve_model` when inspecting hyperparameters, logging, or config diffing. Configs that do not set `predict_params` are unaffected since the returned dict only differs when `predict_params` set.

### Removed

- **BREAKING** — **`ModelConfig.model_n_jobs` removed** — The field was declared but read nowhere in `src/`; it never had the pass-through effect its documentation claimed. Model parallelism is configured through `hyperparameters` (fit-time) and `predict_params` (predict-time) under the model library's own parameter name. Configs still carrying the key continue to parse, but `cfg.model.model_n_jobs` now raises `AttributeError`.

### Fixed

- **Documentation: `BacktestResults.fitted_models`** — Was documented as containing serialized model bytes; it has never been populated and is always `None`. Model serialization is not implemented in V1. To obtain a fitted model today, call `invoke_model()` directly and take the third element of its return tuple.
- **Documentation: package layout and dependencies** — `PACKAGE_MAINTAINER_SPEC.md` described modules that do not exist (`runner/serialization.py`, `backtesting/helpers.py`), used pre-rename filenames (`config.py`, `cv.py`), omitted four that do exist, listed `cloudpickle`/`joblib` as runner dependencies for an unimplemented serialization layer, and described `evaluation.py` as parallelized. Corrected throughout; unbuilt design is now stated in future tense or deferred to `spec_forecast_backtest_system_v1.md` Appendix A.

## [0.3.1] - 2026-05-08

### Fixed

- **Float-equality assertions in unit tests** — Replaced multiple direct float comparisons in tests to avoid flaky equality checks.

### Changed

- **Commit history** — All commit messages retroactively prefixed with `XFSC-36272: ` for issue tracking. This rewrites all SHAs prior to v0.3.1; clones predating this release should be re-fetched.
- **Untracked `backtest_validation.py`** — Removed from the tracked git index (it was a local notebook driver, not shipped code).

## [0.3.0] - 2026-04-06

### Added

- **`mae` metric** — Mean Absolute Error (`tsbricks.blocks.metrics.mae`).
- **`relative_mae` metric** — Ratio of candidate MAE to benchmark MAE, with support for pre-computed benchmark MAE and `return_components` (`tsbricks.blocks.metrics.relative_mae`).
- **`weighted_signed_bias` metric** — Signed analogue of WAPE for directional bias detection (`tsbricks.blocks.metrics.weighted_signed_bias`).
- **`ax` parameter for `plot_acf` and `plot_pacf`** — Draw onto a user-provided matplotlib Axes for subplot integration.
- **`season_col` parameter for `plot_seasonal`** — Explicit season grouping via a column (e.g. fiscal year), mutually exclusive with `period`. Period is inferred from the largest season group.
- **Partial season warning** — `plot_seasonal` warns when positional grouping (integer `ds`, no `season_col`) produces an uneven last season, suggesting `season_col` as a fix.
- **Date x-tick labels** — `plot_seasonal` uses date-based x-axis labels when `ds` is datetime on both matplotlib and plotly backends; integer `ds` retains positional ticks.
- **Plotly hover labels** — `plot_seasonal` shows original `ds` date on hover when `ds` is datetime, with auto-formatting for any granularity.
- **Null `season_col` validation** — `plot_seasonal` raises `ValueError` when `season_col` contains missing values.
- **Variable forecast horizons** — Per-origin horizon support in cross-validation and test folds, allowing different forecast lengths for each origin via `dict[origin, horizon]` syntax in config.
- **Temporal aggregation** — `aggregate_backtest()` composable function for calendar-based temporal aggregation of backtest results; integrated into `run_backtest()` via `calendar_df` parameter.
- **`AggregatedResults`** — New dataclass on `BacktestResults` for temporally aggregated forecasts and metrics.
- **`EvaluationConfig` schema** — Structured evaluation configuration via `evaluation.native.metrics` path, replacing flat `BacktestConfig.metrics`.
- **Python 3.12 support** — Tested and supported alongside Python 3.11.
- **Dependency version ranges** — Expanded supported ranges for key packages (pandas `>=2.2.2,<3`, numpy `>=2.0,<3`, scipy `>=1.14,<2`, plotly `>=6.0,<7`, pydantic `>=2.5,<3`, statsmodels `>=0.14,<1`, pyarrow `>=17.0,<24`, coreforecast `>=0.0.16,<1`, google-cloud-bigquery `>=3.40.1,<4`).
- **Lower-bounds CI testing** — 2x2 CI matrix (Python 3.11/3.12 x latest/min deps) with `min-overrides.txt` for floor version validation.
- **`exclude-newer` pin** — Pinned timestamp in `pyproject.toml` for reproducible dependency resolution.

### Fixed

- **`_tick_date` partial-season bug** — Representative tick dates are now derived from the longest season, preventing ~1-year x-axis jumps when the first custom season is incomplete.
- **Duplicate timestamp validation** — `calendar_df` in `temporal_agg.py` now raises `ValueError` on duplicate timestamps.
- **Fold-weight origin lookup** — Aggregated fold-weight origin lookup handles skipped folds correctly.

### Changed

- **Renamed `backtesting/aggregations.py`** to `backtesting/metric_agg.py` for clarity.
- **Metrics config path** — Metrics are now configured under `evaluation.native.metrics` instead of `BacktestConfig.metrics`.
- **Documentation** — Added §3.5 (Dependency Policy) and §4.7 (Plots) to `PACKAGE_MAINTAINER_SPEC.md`; added Python 3.12 badge to README.

## [0.2.0] - 2026-03-25

### Added

- **Resilient fold execution** — CV folds that raise exceptions are skipped with errors captured in `run_summary`, instead of aborting the entire backtest.
- **Per-series metric resilience** — Metric evaluation failures for individual series are captured and skipped rather than halting the run.
- **Warning capture** — Warnings emitted during transform, model, and metric stages are intercepted and recorded in `run_summary` with fold/series/stage metadata.
- **`run_summary` on `BacktestResults`** — Always-populated `{"warnings": [...], "errors": [...]}` dict available on every result.
- **`run_summary` attached to exception** — When all CV folds fail, `run_summary` is accessible via the raised `RuntimeError`.
- **Warning utilities** — `format_warnings` and `capture_warnings` helpers in `tsbricks.runner`.

### Changed

- **Documentation** — Added §9 (Warning & Error Handling) to `PACKAGE_MAINTAINER_SPEC.md`; updated backtest spec with actual `run_summary` schema.

## [0.1.0] - 2026-03-22

### Added

- **Backtesting engine** — YAML-driven, configuration-based backtesting via `run_backtest()` with structured `BacktestResults` output.
- **Cross-validation** — Explicit forecast-origin fold generation with support for both datetime and integer `ds` columns; optional held-out test fold.
- **Pydantic config schemas** — Typed, validated configuration for backtests, metrics, transforms, and fold definitions.
- **Metrics** — RMSE, RMSSE, WAPE, and difference-scaled bias with per-series, grouped, and global (pooled) aggregation scopes.
- **Grouped and pooled metric aggregation** — Two-stage aggregated metrics (e.g., global WRMSSE) with config-driven grouping and weighting sources.
- **Parameter resolvers** — Context-aware parameter resolution for evaluation metrics.
- **Transforms** — `BoxCoxTransform` and `WorkdayNormalizeTransform` with fit/apply/inverse lifecycle via `BaseTransform`.
- **Transform pipeline runner** — Chained transform execution, model invocation, and serialization helpers in `tsbricks.runner`.
- **Diagnostics** — Residual ACF, stationarity checks, and summary statistics.
- **ACF/PACF plots** — `plot_acf()` and `plot_pacf()` with Plotly and Matplotlib backends via statsmodels.
- **Seasonal plots** — `plot_seasonal()` with configurable seasonal decomposition, Plotly and Matplotlib backends, and native colormap support.
- **Data I/O** — BigQuery and GCS read/write helpers with Parquet and DataFrame support.
- **Metadata collection** — Git hash and `uv.lock` SHA-256 capture attached to backtest results for reproducibility.
- **Polars interop** — Automatic Polars-to-Pandas conversion at public API boundaries.

<!--
Guidelines:
- Keep entries user-facing: what changed, not how.
- Group changes under Added/Changed/Deprecated/Removed/Fixed/Security.
- When you cut a release:
  1) Move items from [Unreleased] into a new version section
  2) Fill in the release date (YYYY-MM-DD)
  3) Optionally add link references below for GitHub compare links
-->

[0.1.0]: https://github.com/betsyhcamp/tsbricks/releases/tag/v0.1.0
[0.2.0]: https://github.com/betsyhcamp/tsbricks/releases/tag/v0.2.0
[0.3.0]: https://github.com/betsyhcamp/tsbricks/releases/tag/v0.3.0
[unreleased]: https://github.com/betsyhcamp/tsbricks/compare/v0.3.0...HEAD
