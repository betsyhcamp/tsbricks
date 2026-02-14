# tsbricks — Package Maintainer Specification

## 1. Overview

`tsbricks` is an internal Python package providing time series forecasting tools for enterprise-scale workflows. It contains reusable forecasting building blocks (metrics, transforms, diagnostics, data I/O, utilities), shared orchestration primitives (transform pipeline execution, model invocation, serialization), and a configuration-driven backtesting system for cross-validated forecast evaluation.

The package is designed for use within Vertex AI pipelines, Vertex AI notebook instances, and local development environments. See `spec_forecast_backtest_system_v1.md` for the full behavioral specification of the backtesting system.

______________________________________________________________________

## 2. Package Architecture

### 2.1 Layout

```
tsbricks/
├── __init__.py
├── blocks/
│   ├── __init__.py
│   ├── metrics/          # Built-in metric callables (MAE, RMSE, RMSSE, MAPE, scaled bias, WAPE, WRMSSE)
│   │   ├── __init__.py
│   │   └── ...
│   ├── transforms/       # Built-in transform classes (BoxCox, ZScore, Log, Log1p, Power, AddConstant, TradingDayNorm)
│   │   ├── __init__.py
│   │   └── ...
│   ├── diagnostics/      # Model fit diagnostics (ACF, PACF)
│   │   ├── __init__.py
│   │   └── ...
│   ├── dataio/           # Data loading and format utilities
│   │   ├── __init__.py
│   │   └── ...
│   └── utils/            # Shared utility functions
│       ├── __init__.py
│       └── ...
├── runner/
│   ├── __init__.py
│   ├── transform_pipeline.py  # Transform chain application and inversion
│   ├── model_invocation.py    # Callable loading, invocation, return tuple unpacking
│   └── serialization.py       # Model serialization (pickle, cloudpickle, joblib, model_method)
├── backtesting/
│   ├── __init__.py       # Exports run_backtest
│   ├── config.py         # Pydantic config models and validation
│   ├── engine.py         # run_backtest orchestration
│   ├── cv.py             # Fold generation, expanding window splits
│   ├── evaluation.py     # Parallelized metric computation
│   ├── results.py        # BacktestResults, CVResults, TestResults dataclasses
│   └── helpers.py        # Artifact extraction utilities for experiment tracking
```

### 2.2 Design Intent

The package is organized into three top-level namespaces that reflect a layered architectural distinction:

**`tsbricks.blocks`** contains stateless building blocks — metric callables, transform objects, diagnostic functions, data I/O utilities, and shared helpers. These components follow well-defined interfaces (the metric callable signature, the four-method transform interface) and have no dependency on the runner's orchestration logic or the backtesting system's configuration model or output structures.

**`tsbricks.runner`** contains shared orchestration primitives — transform chain execution (fit, transform, inverse transform), model callable invocation and return tuple unpacking, and model serialization. These primitives are consumed by `backtesting/` and will be consumed by a future `forecasting/` module. They depend on `blocks/` (e.g., transform objects) but have no dependency on backtest-specific or forecast-specific logic such as fold generation, metric evaluation, or result dataclasses.

**`tsbricks.backtesting`** is a use-case orchestrator that composes primitives from `blocks/` and `runner/` to run cross-validated backtests. It owns configuration parsing, fold generation, parallelized evaluation, error handling, and structured output construction. A future `tsbricks.forecasting` module will follow the same pattern — composing the same `runner/` primitives to produce forward-looking forecasts from the same YAML config.

### 2.3 One-Way Dependency Rule

**`backtesting/` imports from `runner/` and `blocks/`. `runner/` imports from `blocks/`. `blocks/` never imports from `runner/` or `backtesting/`. `runner/` never imports from `backtesting/`.**

The dependency direction is strictly: `backtesting/` → `runner/` → `blocks/`. A future `forecasting/` module will follow the same pattern: `forecasting/` → `runner/` → `blocks/`. No lateral dependencies between `backtesting/` and `forecasting/`.

This is the single most important architectural constraint in the package. It ensures:

- `blocks` modules remain usable independently of the runner and backtesting system. A consumer who only needs pooled RMSE or Box-Cox transforms should never transitively depend on Pydantic config models, multiprocessing orchestration, or model serialization logic.
- `runner` modules are reusable by any use-case orchestrator (backtesting, forecasting) without circular dependencies.
- The package can be mechanically split into separate packages (`tsbricks-blocks`, `tsbricks-runner`, `tsbricks-backtesting`) in the future without redesigning the dependency graph (see section 2.5).
- The interfaces between the namespaces are explicit and testable. The callable contracts (metric signatures, transform four-method interface) defined in the backtesting spec are the API boundary.

**Enforcement.** This rule is enforced in CI via import lint checks. The checks scan source files and fail the build if any violations are found:

```bash
# CI check: blocks must not import from runner or backtesting
if grep -r "from tsbricks.runner\|import tsbricks.runner\|from tsbricks.backtesting\|import tsbricks.backtesting" src/tsbricks/blocks/; then
    echo "FAIL: blocks/ must not import from runner/ or backtesting/"
    exit 1
fi

# CI check: runner must not import from backtesting
if grep -r "from tsbricks.backtesting\|import tsbricks.backtesting" src/tsbricks/runner/; then
    echo "FAIL: runner/ must not import from backtesting/"
    exit 1
fi
```

These checks should be added to the CI pipeline alongside linting and type checking.

### 2.4 Rationale for Single-Package Decision

The backtesting system, runner, and blocks modules are shipped as a single package rather than separate packages. This decision was evaluated at design time and is based on the following factors:

**The built-in implementations are part of the backtesting system's value proposition.** The spec ships built-in metrics (MAE, RMSE, MAPE, RMSSE, scaled bias) and built-in transforms (BoxCox, ZScore, NaturalLog, Log1p, Power, AddConstant, TradingDayNorm). These are designed to work with the backtesting system's callable contracts. Splitting them into a separate package would create user confusion about which package provides what.

**The interfaces have not stabilized.** V1 is the first release. The transform interface, metric callable signatures, and the boundaries between blocks, runner, and backtesting will evolve as features like `per_group` transform scope, sliding window CV, the model adapter pattern, `run_forecast`, and native Optuna integration are added. Co-locating the code allows these interfaces to evolve in lockstep with single-PR refactors and single version bumps.

**Single-team ownership.** The package is currently maintained by a single team. There is no organizational boundary that would benefit from a package boundary. Split packages at organizational boundaries, not conceptual ones.

**Reduced infrastructure overhead.** One package means one Copier template instantiation, one CI/CD pipeline, one version track, and one artifact registry entry. This aligns with the existing `copier-python-mlplatform` template patterns.

**Shared runner primitives support both backtesting and forecasting.** Both `run_backtest` and the future `run_forecast` share the same config model, transform pipeline, and model invocation code via `runner/`. Shipping these as a single package avoids tight cross-package coupling between entry points that share the majority of their implementation.

**The V2+ roadmap is heavily weighted toward backtesting and forecasting.** Sliding window, system-managed data loading, experiment tracking abstraction, model adapter pattern, native Polars support, `run_forecast`, and Optuna integration are all backtesting/runner-side changes. The blocks modules are expected to be more stable. This asymmetry is manageable within a single package with the one-way dependency rule, and does not yet justify the overhead of separate packages.

### 2.5 Conditions for Splitting into Separate Packages

The single-package decision should be revisited if any of the following conditions emerge:

- **Divergent ownership.** A separate team takes responsibility for either the backtesting engine or the blocks forecasting utilities. Package boundaries should follow team boundaries.
- **Divergent consumer bases.** Other teams at the company adopt `tsbricks.blocks` modules (metrics, transforms, diagnostics) for their own evaluation or pipeline code without using the backtesting system, and the backtesting system's dependency footprint (Pydantic, multiprocessing patterns, serialization libraries) causes them friction.
- **Divergent release cadences.** The backtesting system requires frequent releases (e.g., weekly) while blocks modules are stable, and consumers of blocks are forced into unnecessary upgrades.
- **Dependency weight becomes a problem.** The backtesting system accumulates heavy dependencies that are inappropriate for lightweight consumers of blocks utilities.

If splitting is warranted, the one-way dependency rule ensures it is a mechanical operation: extract `tsbricks/blocks/` to `tsbricks-blocks`, extract `tsbricks/runner/` to `tsbricks-runner`, update import paths, and add the appropriate dependency chain (`tsbricks-backtesting` → `tsbricks-runner` → `tsbricks-blocks`). No architectural redesign is required.

______________________________________________________________________

## 3. Dependencies

### 3.1 Blocks Dependencies

Dependencies required by `tsbricks.blocks`:

- `numpy` — numerical operations across metrics, transforms, and diagnostics
- `pandas` — DataFrame operations, Nixtla long format handling
- `coreforecast` — Box-Cox transform (Guerrero method, log-likelihood)

### 3.2 Runner Dependencies

Additional dependencies required by `tsbricks.runner`:

- `cloudpickle` / `joblib` — model serialization (optional, depending on user config)

### 3.3 Backtesting Dependencies

Additional dependencies required by `tsbricks.backtesting`:

- `pydantic` — YAML config parsing and validation
- `pyyaml` — YAML file reading

### 3.4 Optional Dependencies

- `polars` — accepted at the `run_backtest` entry point, converted to pandas internally

______________________________________________________________________

## 4. Public API Surface

### 4.1 Convenience Entry Point

```python
from tsbricks.backtesting import run_backtest
```

A future `run_forecast` entry point will be added in `tsbricks.forecasting`:

```python
# Future (not V1)
from tsbricks.forecasting import run_forecast
```

### 4.2 Composable Step Functions (Backtesting)

```python
from tsbricks.backtesting import parse_config, generate_folds, evaluate_metrics
```

### 4.3 Composable Step Functions (Runner)

```python
from tsbricks.runner import (
    fit_transforms,
    apply_transforms,
    inverse_transforms,
    resolve_model,
    invoke_model,
    serialize_model,
)
```

### 4.4 Built-in Transforms

```python
from tsbricks.blocks.transforms import (
    BoxCoxTransform,
    ZScoreTransform,
    NaturalLogTransform,
    Log1pTransform,
    PowerTransform,
    AddConstantTransform,
    TradingDayNormalization,
)
```

### 4.5 Built-in Metrics

```python
from tsbricks.blocks.metrics import mae, rmse, mape, rmsse, difference_scaled_bias, wape, wrmsse
```

### 4.6 Diagnostics

```python
from tsbricks.blocks.diagnostics import acf, pacf
```

### 4.7 Output Dataclasses

```python
from tsbricks.backtesting.results import BacktestResults, CVResults, TestResults
```

______________________________________________________________________

## 5. Versioning

The package follows semantic versioning. Because `blocks`, `runner`, and `backtesting` are co-versioned in a single package, a breaking change in any namespace constitutes a major version bump. This is acceptable at the current scale; if the asymmetry between blocks stability and backtesting/runner churn becomes a problem, it is a signal to revisit the split decision (see section 2.5).

______________________________________________________________________

## 6. YAML Config Import Paths

The backtesting system's YAML config references callables by import path. There are two categories:

**Built-in callables** use the `tsbricks.blocks` namespace:

```yaml
# Built-in transform
- class: tsbricks.blocks.transforms.BoxCoxTransform

# Built-in metric
- callable: tsbricks.blocks.metrics.rmsse
```

**User-provided callables** use the user's own package namespace:

```yaml
# User-provided model
model:
  callable: my_project.models.auto_arima_forecast

# User-provided custom metric
- callable: my_project.metrics.custom_metric

# User-provided custom serializer
serialization:
  method: my_project.serializers.custom_save
```

The dynamic import mechanism in `tsbricks.runner` resolves any valid Python import path. Built-in and user-provided callables are treated identically at runtime.
