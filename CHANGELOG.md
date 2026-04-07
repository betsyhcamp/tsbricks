# Changelog

All notable changes to this project will be documented in this file.

The format is based on **Keep a Changelog** (https://keepachangelog.com/en/1.1.0/),
and this project adheres to **Semantic Versioning** (https://semver.org/).

## [Unreleased]

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
