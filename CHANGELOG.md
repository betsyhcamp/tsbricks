# Changelog

All notable changes to this project will be documented in this file.

The format is based on **Keep a Changelog** (https://keepachangelog.com/en/1.1.0/),
and this project adheres to **Semantic Versioning** (https://semver.org/).

## [Unreleased]

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
[unreleased]: https://github.com/betsyhcamp/tsbricks/compare/v0.1.0...HEAD
