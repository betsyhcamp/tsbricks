"""Pydantic configuration models for the backtesting system."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _warn_non_normalized_dates(dates: list[str], field_name: str) -> None:
    """Emit a warning for date strings not in YYYY-MM-DD format."""
    non_normalized = [d for d in dates if d != pd.Timestamp(d).strftime("%Y-%m-%d")]
    if non_normalized:
        warnings.warn(
            f"{field_name} contains non-normalized date strings: "
            f"{non_normalized}. Consider using YYYY-MM-DD format "
            f"(e.g., '2023-09-01' instead of '2023-9-01').",
            UserWarning,
            stacklevel=2,
        )


class ForecastOriginConfig(BaseModel):
    """Per-origin configuration for variable-horizon backtesting."""

    origin: str | int
    horizon: int = Field(gt=0)


class DataConfig(BaseModel):
    """Data section: column mapping and frequency."""

    target_col: str = "y"
    date_col: str = "ds"
    id_col: str = "unique_id"
    freq: str | int
    exogenous_columns: list[str] | None = None

    @field_validator("freq")
    @classmethod
    def _validate_freq(cls, v: str | int) -> str | int:
        if isinstance(v, int) and v != 1:
            raise ValueError(f"Integer freq must be 1 (for integer ds), got {v}.")
        return v


class CrossValidationConfig(BaseModel):
    """Cross-validation section (explicit mode only in V1).

    Supports two mutually exclusive horizon formats:

    **Uniform** — a single ``horizon`` applies to all origins::

        horizon: 6
        forecast_origins: ["2025-12-01", "2026-12-01"]

    **Variable** — each origin specifies its own ``horizon``::

        forecast_origins:
          - origin: "2025-12-01"
            horizon: 6
          - origin: "2026-12-01"
            horizon: 5
    """

    mode: Literal["explicit"]
    horizon: int | None = Field(default=None, gt=0)
    forecast_origins: list[Any] = Field(min_length=1)

    # Future: parametric mode fields
    n_folds: int | None = None
    step_size: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_origin_objects(
        cls,
        data: Any,
    ) -> Any:
        """Coerce origin dicts to ForecastOriginConfig."""
        if not isinstance(data, dict):
            return data
        origins = data.get("forecast_origins", [])
        if not origins:
            return data
        if isinstance(origins[0], dict) and "origin" in origins[0]:
            data["forecast_origins"] = [
                ForecastOriginConfig(**o) if isinstance(o, dict) else o for o in origins
            ]
        return data

    @model_validator(mode="after")
    def _validate_horizon_format(
        self,
    ) -> CrossValidationConfig:
        """Enforce mutual exclusivity of formats."""
        origins = self.forecast_origins
        is_object_list = len(origins) > 0 and isinstance(
            origins[0], ForecastOriginConfig
        )

        if self.horizon is not None and is_object_list:
            raise ValueError(
                "Cannot specify both a top-level "
                "'horizon' and per-origin horizon objects "
                "in 'forecast_origins'. Use one format "
                "or the other."
            )

        if self.horizon is None and not is_object_list:
            raise ValueError(
                "When 'horizon' is not set, "
                "'forecast_origins' must be a list of "
                "objects each with 'origin' and "
                "'horizon' fields."
            )

        # Validate homogeneous origin types
        raw = self.raw_origins()
        types = {type(o) for o in raw}
        if len(types) > 1:
            raise ValueError(
                "forecast_origins must be all strings "
                "(datetime) or all integers, got mixed "
                f"types: {types}."
            )
        if all(isinstance(o, str) for o in raw):
            _warn_non_normalized_dates(
                [str(o) for o in raw],
                "forecast_origins",
            )

        return self

    def raw_origins(self) -> list[str | int]:
        """Return raw origin values, regardless of format."""
        if self.horizon is not None:
            return list(self.forecast_origins)
        return [o.origin for o in self.forecast_origins]

    def origin_horizon_pairs(
        self,
    ) -> list[tuple[str | int, int]]:
        """Return normalized ``(origin, horizon)`` tuples.

        Works for both uniform and variable horizon formats,
        so downstream code does not need to branch on the
        config shape.
        """
        if self.horizon is not None:
            return [(o, self.horizon) for o in self.forecast_origins]
        return [(o.origin, o.horizon) for o in self.forecast_origins]


class TransformConfig(BaseModel):
    """Single transform definition.

    The ``class_path`` field is aliased as ``class`` so that YAML configs
    can use the natural keyword ``class:`` while Python code uses
    ``class_path=`` (since ``class`` is a reserved word).
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str
    class_path: str = Field(alias="class")
    scope: str = "per_series"
    targets: list[str]
    perform_inverse_transform: bool = True
    params: dict[str, Any] | None = None

    # Out of scope for V1
    model_native: bool | None = None
    series_overrides: dict[str, Any] | None = None

    _GLOBAL_ONLY_TRANSFORMS = {"WorkdayNormalizeTransform"}
    _CALENDAR_SCOPE_TRANSFORMS = {"WorkdayNormalizeTransform"}
    _VALID_CALENDAR_SCOPES = ("global", "per_series")

    @model_validator(mode="after")
    def _check_scope_constraints(self) -> TransformConfig:
        class_name = self.class_path.rsplit(".", 1)[-1]
        if class_name in self._GLOBAL_ONLY_TRANSFORMS and self.scope != "global":
            raise ValueError(
                f"Transform '{self.name}' uses {class_name} which requires "
                f"scope='global', got scope='{self.scope}'."
            )
        if class_name in self._CALENDAR_SCOPE_TRANSFORMS:
            params = self.params or {}
            calendar_scope = params.get("calendar_scope")
            if calendar_scope is None:
                raise ValueError(
                    f"Transform '{self.name}' uses {class_name} which requires "
                    f"'calendar_scope' in params."
                )
            if calendar_scope not in self._VALID_CALENDAR_SCOPES:
                raise ValueError(
                    f"Transform '{self.name}' has invalid calendar_scope="
                    f"'{calendar_scope}'. Must be one of "
                    f"{self._VALID_CALENDAR_SCOPES}."
                )
        return self


class ModelConfig(BaseModel):
    """Model section: callable path and hyperparameters."""

    callable: str
    hyperparameters: dict[str, Any] | None = None
    model_n_jobs: int | None = None

    # Out of scope for V1
    serialization: dict[str, Any] | None = None


class ParamResolverConfig(BaseModel):
    """Configuration for a fold-dependent per-series parameter resolver."""

    callable: str
    params: dict[str, Any] | None = None
    grouping_columns: list[str] | None = None


class MetricDefinitionConfig(BaseModel):
    """Single metric definition within the metrics section."""

    name: str
    callable: str
    type: Literal["simple", "context_aware"]
    scope: Literal["per_series", "group", "global"] = "per_series"
    aggregation: Literal["per_fold_mean", "pooled"] = "per_fold_mean"
    params: dict[str, Any] | None = None
    grouping_columns: list[str] | None = Field(default=None, min_length=1)
    per_series_params: dict[str, dict[str, Any]] | None = None
    param_resolvers: dict[str, ParamResolverConfig] | None = None
    aggregation_callable: str | None = None
    aggregation_params: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check_metric_constraints(self) -> MetricDefinitionConfig:
        # scope: global requires aggregation_callable
        if self.scope == "global" and self.aggregation_callable is None:
            raise ValueError(
                f"Metric '{self.name}' has scope='global' but no "
                f"aggregation_callable. Global scope requires an "
                f"aggregation_callable."
            )

        # Composite grouping keys are not yet supported
        if self.grouping_columns is not None and len(self.grouping_columns) > 1:
            raise ValueError(
                f"Metric '{self.name}' has {len(self.grouping_columns)} "
                f"grouping_columns, but only single-column grouping "
                f"is currently supported."
            )

        # No key overlap between params, per_series_params, and param_resolvers
        keys_params = set(self.params or {})
        keys_psp = set(self.per_series_params or {})
        keys_pr = set(self.param_resolvers or {})
        overlap = (
            (keys_params & keys_psp) | (keys_params & keys_pr) | (keys_psp & keys_pr)
        )
        if overlap:
            raise ValueError(
                f"Metric '{self.name}' has overlapping keys across params, "
                f"per_series_params, and param_resolvers: {overlap}."
            )

        return self


class MetricsConfig(BaseModel):
    """Metrics section: definitions and optional grouping."""

    definitions: list[MetricDefinitionConfig]
    grouping_columns: list[str] | None = Field(default=None, min_length=1)
    grouping_source: str | None = None
    weights_source: str | None = None

    @model_validator(mode="after")
    def _check_group_scope_has_grouping_columns(
        self,
    ) -> MetricsConfig:
        # Composite grouping keys are not yet supported
        if self.grouping_columns is not None and len(self.grouping_columns) > 1:
            raise ValueError(
                f"Top-level metrics.grouping_columns has "
                f"{len(self.grouping_columns)} columns, but only "
                f"single-column grouping is currently supported."
            )

        for defn in self.definitions:
            if defn.scope == "group":
                has_defn_cols = defn.grouping_columns is not None
                has_top_cols = self.grouping_columns is not None
                if not has_defn_cols and not has_top_cols:
                    raise ValueError(
                        f"Metric '{defn.name}' has scope='group' "
                        f"but no grouping_columns on the metric "
                        f"definition or in the top-level "
                        f"metrics config."
                    )
        return self


class TestConfig(BaseModel):
    """Test fold configuration.

    The test fold uses ``cross_validation.horizon``; a separate
    ``test.horizon`` is not supported.
    """

    model_config = ConfigDict(extra="forbid")

    test_origin: str | int

    @model_validator(mode="before")
    @classmethod
    def _reject_horizon(cls, data: Any) -> Any:
        if isinstance(data, dict) and "horizon" in data:
            raise ValueError(
                "test.horizon is not supported; the test fold uses "
                "cross_validation.horizon."
            )
        return data


class BacktestConfig(BaseModel):
    """Top-level backtest configuration."""

    data: DataConfig
    cross_validation: CrossValidationConfig
    transforms: list[TransformConfig] | None = None
    model: ModelConfig
    metrics: MetricsConfig

    # Test fold (optional — presence controls whether test fold runs)
    test: TestConfig | None = None

    # Out of scope for V1
    parallelization: dict[str, Any] | None = None
    artifact_storage: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_test_config(self) -> BacktestConfig:
        if self.test is None:
            return self

        test_origin = self.test.test_origin
        origins = self.cross_validation.raw_origins()

        # Type consistency: test_origin must match forecast_origins type
        origin_types_are_str = all(isinstance(o, str) for o in origins)
        origin_types_are_int = all(isinstance(o, int) for o in origins)

        if origin_types_are_str and not isinstance(test_origin, str):
            raise ValueError(
                "test.test_origin must be a string (datetime) to match "
                "forecast_origins type."
            )
        if origin_types_are_int and not isinstance(test_origin, int):
            raise ValueError(
                "test.test_origin must be an integer to match forecast_origins type."
            )

        # Warn on non-normalized date strings
        if origin_types_are_str and isinstance(test_origin, str):
            _warn_non_normalized_dates([test_origin], "test.test_origin")

        # Ordering: test_origin must be after all forecast_origins
        if origin_types_are_int:
            max_origin = max(int(o) for o in origins)
            if int(test_origin) <= max_origin:
                raise ValueError(
                    f"test.test_origin ({test_origin}) must be strictly after "
                    f"all forecast_origins (max={max_origin})."
                )
        else:
            # Temporal comparison via pd.Timestamp for datetime origins
            max_origin_ts = max(pd.Timestamp(o) for o in origins)
            test_origin_ts = pd.Timestamp(test_origin)
            if test_origin_ts <= max_origin_ts:
                raise ValueError(
                    f"test.test_origin ({test_origin}) must be strictly after "
                    f"all forecast_origins (max={max_origin_ts})."
                )

        return self


def parse_config(
    config_path: str | None = None,
    config: dict | None = None,
) -> BacktestConfig:
    """Parse and validate a YAML file or dict into a BacktestConfig.

    Exactly one of ``config_path`` or ``config`` must be provided.

    Args:
        config_path: Path to a YAML configuration file.
        config: Configuration dict to parse directly.

    Returns:
        Validated ``BacktestConfig`` instance.

    Raises:
        ValueError: If both or neither arguments are provided.
    """
    if config_path is not None and config is not None:
        raise ValueError("Provide exactly one of 'config_path' or 'config', not both.")
    if config_path is None and config is None:
        raise ValueError("Provide exactly one of 'config_path' or 'config'.")

    if config_path is not None:
        raw = yaml.safe_load(Path(config_path).read_text())
        if raw is None:
            raise ValueError(f"YAML file is empty: {config_path}")
    else:
        raw = config

    return BacktestConfig.model_validate(raw)
