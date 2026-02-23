"""Pydantic configuration models for the backtesting system."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    """Data section: column mapping and frequency."""

    target_col: str = "y"
    date_col: str = "ds"
    id_col: str = "unique_id"
    freq: str
    exogenous_columns: list[str] | None = None


class CrossValidationConfig(BaseModel):
    """Cross-validation section (explicit mode only in V1)."""

    mode: Literal["explicit"]
    horizon: int = Field(gt=0)
    forecast_origins: list[str] = Field(min_length=1)

    # Future: parametric mode fields
    n_folds: int | None = None
    step_size: int | None = None


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


class ModelConfig(BaseModel):
    """Model section: callable path and hyperparameters."""

    callable: str
    hyperparameters: dict[str, Any] | None = None
    model_n_jobs: int | None = None

    # Out of scope for V1
    serialization: dict[str, Any] | None = None


class MetricDefinitionConfig(BaseModel):
    """Single metric definition within the metrics section."""

    name: str
    callable: str
    type: Literal["simple", "context_aware"]
    scope: str = "per_series"
    aggregation: str = "per_fold_mean"
    params: dict[str, Any] | None = None


class MetricsConfig(BaseModel):
    """Metrics section: definitions and optional grouping."""

    definitions: list[MetricDefinitionConfig]
    grouping_columns: list[str] | None = None


class BacktestConfig(BaseModel):
    """Top-level backtest configuration."""

    data: DataConfig
    cross_validation: CrossValidationConfig
    transforms: list[TransformConfig] | None = None
    model: ModelConfig
    metrics: MetricsConfig

    # Out of scope for V1
    test: dict[str, Any] | None = None
    parallelization: dict[str, Any] | None = None
    artifact_storage: dict[str, Any] | None = None


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
