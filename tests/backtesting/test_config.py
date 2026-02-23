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


# ---- parse_config from dict ----


def test_parse_config_from_valid_dict(valid_cfg: dict) -> None:
    """A valid dict parses into a BacktestConfig without error."""
    cfg = parse_config(config=valid_cfg)

    assert isinstance(cfg, BacktestConfig)
    assert cfg.data.freq == "MS"
    assert cfg.cross_validation.mode == "explicit"
    assert cfg.cross_validation.horizon == 6
    assert len(cfg.cross_validation.forecast_origins) == 2
    assert cfg.model.callable == "tsbricks._testing.dummy_models.forecast_only"
    assert len(cfg.metrics.definitions) == 1
    assert cfg.metrics.definitions[0].name == "rmse"


def test_parse_config_defaults(valid_cfg: dict) -> None:
    """Default values are applied for optional fields."""
    cfg = parse_config(config=valid_cfg)

    assert cfg.data.target_col == "y"
    assert cfg.data.date_col == "ds"
    assert cfg.data.id_col == "unique_id"
    assert cfg.data.exogenous_columns is None
    assert cfg.model.hyperparameters == {}
    assert cfg.model.model_n_jobs is None
    assert cfg.model.serialization is None
    assert cfg.test is None
    assert cfg.parallelization is None
    assert cfg.artifact_storage is None


def test_parse_config_transform_fields(valid_cfg: dict) -> None:
    """Transform config fields are parsed correctly from the class alias."""
    cfg = parse_config(config=valid_cfg)

    assert cfg.transforms is not None
    assert len(cfg.transforms) == 1
    t = cfg.transforms[0]
    assert t.name == "box_cox"
    assert t.class_path == "tsbricks.blocks.transforms.BoxCoxTransform"
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


def test_parse_config_yaml_class_alias(tmp_path) -> None:
    """The YAML ``class:`` key maps to TransformConfig.class_path."""
    import yaml

    raw = {
        "data": {"freq": "MS"},
        "cross_validation": {
            "mode": "explicit",
            "horizon": 3,
            "forecast_origins": ["2023-06-01"],
        },
        "transforms": [
            {
                "name": "box_cox",
                "class": "tsbricks.blocks.transforms.BoxCoxTransform",
                "targets": ["y"],
            }
        ],
        "model": {"callable": "some.module.func"},
        "metrics": {
            "definitions": [
                {
                    "name": "rmse",
                    "callable": "tsbricks.blocks.metrics.rmse",
                    "type": "simple",
                }
            ]
        },
    }
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml.dump(raw))

    cfg = parse_config(config_path=str(yaml_path))

    assert cfg.transforms is not None
    assert cfg.transforms[0].class_path == "tsbricks.blocks.transforms.BoxCoxTransform"


# ---- class_path via Python name (populate_by_name) ----


def test_transform_config_class_path_by_name() -> None:
    """TransformConfig accepts class_path= when constructing from Python."""
    cfg = parse_config(
        config={
            "data": {"freq": "D"},
            "cross_validation": {
                "mode": "explicit",
                "horizon": 1,
                "forecast_origins": ["2024-01-01"],
            },
            "transforms": [
                {
                    "name": "test",
                    "class_path": "some.module.Transform",
                    "targets": ["y"],
                }
            ],
            "model": {"callable": "some.module.func"},
            "metrics": {
                "definitions": [
                    {
                        "name": "m",
                        "callable": "some.module.metric",
                        "type": "simple",
                    }
                ]
            },
        }
    )

    assert cfg.transforms is not None
    assert cfg.transforms[0].class_path == "some.module.Transform"


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
