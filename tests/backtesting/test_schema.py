"""Tests for backtesting config parsing and Pydantic validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tsbricks.backtesting.schema import BacktestConfig, parse_config


# ---- parse_config argument validation ----


def test_parse_config_both_args_raises() -> None:
    """Providing both config_path and config raises ValueError."""
    with pytest.raises(ValueError, match="exactly one"):
        parse_config(config_path="foo.yaml", config={})


def test_parse_config_no_args_raises() -> None:
    """Providing neither config_path nor config raises ValueError."""
    with pytest.raises(ValueError, match="exactly one"):
        parse_config()


# ---- parse_config from dict: data section ----


def test_parse_valid_dict_data_section(valid_cfg: dict) -> None:
    """Valid dict parses into a BacktestConfig with correct data fields."""
    cfg = parse_config(config=valid_cfg)

    assert isinstance(cfg, BacktestConfig)
    assert cfg.data.freq == "MS"


def test_parse_valid_dict_cv_section(valid_cfg: dict) -> None:
    """Valid dict parses cross_validation section correctly."""
    cfg = parse_config(config=valid_cfg)

    assert cfg.cross_validation.mode == "explicit"
    assert cfg.cross_validation.horizon == 6
    assert len(cfg.cross_validation.forecast_origins) == 2


def test_parse_valid_dict_model_section(valid_cfg: dict) -> None:
    """Valid dict parses model section correctly."""
    cfg = parse_config(config=valid_cfg)

    assert cfg.model.callable == "tsbricks._testing.dummy_models.forecast_only"


def test_parse_valid_dict_metrics_section(valid_cfg: dict) -> None:
    """Valid dict parses metrics section correctly."""
    cfg = parse_config(config=valid_cfg)

    assert len(cfg.metrics.definitions) == 1
    assert cfg.metrics.definitions[0].name == "rmse"


# ---- defaults by config model ----


def test_data_config_defaults(valid_cfg: dict) -> None:
    """DataConfig applies correct defaults for optional column names."""
    cfg = parse_config(config=valid_cfg)

    assert cfg.data.target_col == "y"
    assert cfg.data.date_col == "ds"
    assert cfg.data.id_col == "unique_id"
    assert cfg.data.exogenous_columns is None


def test_model_config_defaults(valid_cfg: dict) -> None:
    """ModelConfig applies correct defaults for optional fields."""
    cfg = parse_config(config=valid_cfg)

    assert cfg.model.hyperparameters == {}
    assert cfg.model.model_n_jobs is None
    assert cfg.model.serialization is None


def test_backtest_config_optional_defaults(valid_cfg: dict) -> None:
    """Top-level optional sections default to None."""
    cfg = parse_config(config=valid_cfg)

    assert cfg.test is None
    assert cfg.parallelization is None
    assert cfg.artifact_storage is None


# ---- transform config parsing ----


def test_parse_config_transform_count(valid_cfg: dict) -> None:
    """Config with one transform has a single-element transforms list."""
    cfg = parse_config(config=valid_cfg)

    assert cfg.transforms is not None
    assert len(cfg.transforms) == 1


def test_transform_class_alias_from_dict(valid_cfg: dict) -> None:
    """The ``class`` key in the dict maps to TransformConfig.class_path."""
    cfg = parse_config(config=valid_cfg)

    assert cfg.transforms is not None
    assert cfg.transforms[0].class_path == "tsbricks.blocks.transforms.BoxCoxTransform"


def test_transform_config_fields(valid_cfg: dict) -> None:
    """Transform config fields are parsed with correct values."""
    cfg = parse_config(config=valid_cfg)
    t = cfg.transforms[0]  # type: ignore[index]

    assert t.name == "box_cox"
    assert t.scope == "per_series"
    assert t.targets == ["y"]
    assert t.perform_inverse_transform is True
    assert t.params == {"method": "guerrero", "season_length": 12}


def test_parse_config_no_transforms(valid_cfg: dict) -> None:
    """Config without transforms section is valid."""
    del valid_cfg["transforms"]
    cfg = parse_config(config=valid_cfg)

    assert cfg.transforms is None


# ---- parse_config from YAML file ----


def test_parse_config_from_yaml(valid_cfg: dict, tmp_path) -> None:
    """A valid YAML file parses into a BacktestConfig."""
    import yaml

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml.dump(valid_cfg))

    cfg = parse_config(config_path=str(yaml_path))

    assert isinstance(cfg, BacktestConfig)
    assert cfg.data.freq == "MS"
    assert cfg.cross_validation.horizon == 6


# ---- class_path via Python name (populate_by_name) ----


def test_transform_config_class_path_by_name(valid_cfg: dict) -> None:
    """TransformConfig accepts class_path= when constructing from Python."""
    transform = valid_cfg["transforms"][0]
    transform["class_path"] = transform.pop("class")

    cfg = parse_config(config=valid_cfg)

    assert cfg.transforms is not None
    assert cfg.transforms[0].class_path == "tsbricks.blocks.transforms.BoxCoxTransform"


# ---- Validation errors for missing required fields ----


def test_missing_data_freq_raises(valid_cfg: dict) -> None:
    """Missing required field data.freq raises ValidationError."""
    del valid_cfg["data"]["freq"]

    with pytest.raises(ValidationError):
        parse_config(config=valid_cfg)


def test_missing_cross_validation_raises(valid_cfg: dict) -> None:
    """Missing cross_validation section raises ValidationError."""
    del valid_cfg["cross_validation"]

    with pytest.raises(ValidationError):
        parse_config(config=valid_cfg)


def test_missing_model_callable_raises(valid_cfg: dict) -> None:
    """Missing model.callable raises ValidationError."""
    del valid_cfg["model"]["callable"]

    with pytest.raises(ValidationError):
        parse_config(config=valid_cfg)


def test_missing_metrics_definitions_raises(valid_cfg: dict) -> None:
    """Missing metrics.definitions raises ValidationError."""
    del valid_cfg["metrics"]["definitions"]

    with pytest.raises(ValidationError):
        parse_config(config=valid_cfg)


def test_invalid_cv_mode_raises(valid_cfg: dict) -> None:
    """Invalid cross_validation.mode value raises ValidationError."""
    valid_cfg["cross_validation"]["mode"] = "rolling"

    with pytest.raises(ValidationError):
        parse_config(config=valid_cfg)


def test_invalid_metric_type_raises(valid_cfg: dict) -> None:
    """Invalid metric type value raises ValidationError."""
    valid_cfg["metrics"]["definitions"][0]["type"] = "unknown"

    with pytest.raises(ValidationError):
        parse_config(config=valid_cfg)


def test_horizon_zero_raises(valid_cfg: dict) -> None:
    """horizon=0 is rejected by the schema."""
    valid_cfg["cross_validation"]["horizon"] = 0

    with pytest.raises(ValidationError):
        parse_config(config=valid_cfg)


def test_horizon_negative_raises(valid_cfg: dict) -> None:
    """Negative horizon is rejected by the schema."""
    valid_cfg["cross_validation"]["horizon"] = -1

    with pytest.raises(ValidationError):
        parse_config(config=valid_cfg)


def test_empty_forecast_origins_raises(valid_cfg: dict) -> None:
    """Empty forecast_origins list is rejected by the schema."""
    valid_cfg["cross_validation"]["forecast_origins"] = []

    with pytest.raises(ValidationError):
        parse_config(config=valid_cfg)


def test_empty_yaml_file_raises(tmp_path) -> None:
    """An empty YAML file raises ValueError with a clear message."""
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text("")

    with pytest.raises(ValueError, match="YAML file is empty"):
        parse_config(config_path=str(yaml_path))


# ---- Out-of-scope fields accepted as dicts ----


def test_out_of_scope_fields_accepted(valid_cfg: dict) -> None:
    """Out-of-scope V1 fields (test, parallelization, artifact_storage) parse as dicts."""
    valid_cfg["test"] = {"enabled": True, "test_origin": "2024-01-01"}
    valid_cfg["parallelization"] = {"parallel_eval_strategy": "across_series"}
    valid_cfg["artifact_storage"] = {"uv_lock_path": "./uv.lock"}

    cfg = parse_config(config=valid_cfg)

    assert cfg.test == {"enabled": True, "test_origin": "2024-01-01"}
    assert cfg.parallelization == {"parallel_eval_strategy": "across_series"}
    assert cfg.artifact_storage == {"uv_lock_path": "./uv.lock"}


# ---- Integer ds config ----


def test_parse_valid_dict_integer_ds(valid_cfg: dict) -> None:
    """Config with freq=1 and integer forecast_origins parses correctly."""
    valid_cfg["data"]["freq"] = 1
    valid_cfg["cross_validation"]["forecast_origins"] = [30, 40]

    cfg = parse_config(config=valid_cfg)

    assert cfg.data.freq == 1
    assert cfg.cross_validation.forecast_origins == [30, 40]


def test_freq_int_not_one_raises(valid_cfg: dict) -> None:
    """Integer freq other than 1 raises ValidationError."""
    valid_cfg["data"]["freq"] = 2

    with pytest.raises(ValidationError, match="Integer freq must be 1"):
        parse_config(config=valid_cfg)


def test_integer_forecast_origins_accepted(valid_cfg: dict) -> None:
    """Integer forecast_origins parse correctly with freq=1."""
    valid_cfg["data"]["freq"] = 1
    valid_cfg["cross_validation"]["forecast_origins"] = [10, 20, 30]

    cfg = parse_config(config=valid_cfg)

    assert cfg.cross_validation.forecast_origins == [10, 20, 30]
