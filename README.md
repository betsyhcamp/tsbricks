# tsbricks

![CI](https://github.com/betsyhcamp/tsbricks/actions/workflows/ci.yml/badge.svg?branch=main)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)

Time series forecasting building blocks and a configuration-driven backtesting system for cross-validated forecast evaluation.

## Overview

`tsbricks` provides reusable forecasting primitives (metrics, transforms, diagnostics, data I/O), shared orchestration (transform pipelines, model invocation, serialization), and a YAML-driven backtesting engine. It is model-agnostic, designed for use within Vertex AI pipelines, notebook instances, and local development.

The package is organized into three namespaces with a strict one-way dependency rule (`backtesting` → `runner` → `blocks`):

- **`tsbricks.blocks`** — Stateless building blocks: metric callables, transform objects, diagnostics, data I/O, utilities.
- **`tsbricks.runner`** — Shared orchestration: transform chain execution, model invocation, serialization. Consumed by backtesting and (future) forecasting.
- **`tsbricks.backtesting`** — Use-case orchestrator: config parsing, fold generation, parallelized evaluation, structured output.

## Installation

Install the latest release from GitHub:

```bash
uv add git+https://github.com/betsyhcamp/tsbricks.git@v0.2.0
```

With optional extras:

```bash
# Matplotlib plotting backend
uv add "tsbricks[matplotlib] @ git+https://github.com/betsyhcamp/tsbricks.git@v0.2.0"

# Polars interop
uv add "tsbricks[polars] @ git+https://github.com/betsyhcamp/tsbricks.git@v0.2.0"

# Multiple extras
uv add "tsbricks[matplotlib,polars] @ git+https://github.com/betsyhcamp/tsbricks.git@v0.2.0"
```

## Quick start

```python
import pandas as pd
from tsbricks.backtesting import run_backtest

df = pd.read_parquet("sales.parquet")
results = run_backtest(config_path="config.yaml", df=df)
```

Individual step functions are also available for custom workflows:

```python
from tsbricks.backtesting import parse_config, generate_folds, evaluate_metrics
from tsbricks.runner import fit_transforms, apply_transforms, inverse_transforms, invoke_model
```

## Requirements

- Python 3.11.11
- [uv](https://github.com/astral-sh/uv)
- [Task](https://taskfile.dev/)

## Getting started

```bash
uv sync
pre-commit install
```

## Development workflow

```bash
task install        # Install dependencies (uv sync)
task lint           # Run linters
task lint-fix       # Run linters with automated fixes
task format         # Auto-format code
task format-check   # Check formatting without modifying
task md-format      # Auto-format Markdown files
task md-check       # Check Markdown formatting without modifying
task test           # Run tests
task check          # Run full CI suite (pre-commit + test + docs + build)
task pre-commit     # Run pre-commit hooks on all files
```

## Notes

- Tool versions are pinned for reproducibility. CI mirrors local commands via the Taskfile.
- Dependencies are managed by `uv`; `uv.lock` should be committed.
- Virtual environments must use the `.venv*` naming convention.
- For VSCode users, copy `.vscode/settings.example.json` to `.vscode/settings.json`.
- See [notes/PACKAGE_MAINTAINER_SPEC.md](notes/PACKAGE_MAINTAINER_SPEC.md) for full architecture details and [notes/spec_forecast_backtest_system_v1.md](notes/spec_forecast_backtest_system_v1.md) for the V1 behavioral specification.
