"""Tests for backtesting config parsing and Pydantic validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tsbricks.backtesting.schema import (
    BacktestConfig,
    CrossValidationConfig,
    ForecastOriginConfig,
    MetricDefinitionConfig,
    MetricsConfig,
    ParamResolverConfig,
    parse_config,
)


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
    """Out-of-scope V1 fields (parallelization, artifact_storage) parse as dicts."""
    valid_cfg["parallelization"] = {"parallel_eval_strategy": "across_series"}
    valid_cfg["artifact_storage"] = {"uv_lock_path": "./uv.lock"}

    cfg = parse_config(config=valid_cfg)

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


# ---- Transform scope constraints ----


def test_workday_transform_scope_global_accepted(valid_cfg: dict) -> None:
    """WorkdayNormalizeTransform with scope='global' is valid."""
    valid_cfg["transforms"] = [
        {
            "name": "workday_norm",
            "class": "tsbricks.blocks.transforms.WorkdayNormalizeTransform",
            "scope": "global",
            "targets": ["y"],
            "params": {"calendar_scope": "global"},
        }
    ]
    cfg = parse_config(config=valid_cfg)

    assert cfg.transforms[0].scope == "global"


def test_workday_transform_scope_per_series_raises(valid_cfg: dict) -> None:
    """WorkdayNormalizeTransform with scope='per_series' raises ValidationError."""
    valid_cfg["transforms"] = [
        {
            "name": "workday_norm",
            "class": "tsbricks.blocks.transforms.WorkdayNormalizeTransform",
            "scope": "per_series",
            "targets": ["y"],
            "params": {"calendar_scope": "global"},
        }
    ]
    with pytest.raises(ValidationError, match="requires scope='global'"):
        parse_config(config=valid_cfg)


def test_workday_transform_default_scope_raises(valid_cfg: dict) -> None:
    """WorkdayNormalizeTransform without explicit scope defaults to per_series and raises."""
    valid_cfg["transforms"] = [
        {
            "name": "workday_norm",
            "class": "tsbricks.blocks.transforms.WorkdayNormalizeTransform",
            "targets": ["y"],
            "params": {"calendar_scope": "global"},
        }
    ]
    with pytest.raises(ValidationError, match="requires scope='global'"):
        parse_config(config=valid_cfg)


# ---- calendar_scope validation ----


def test_workday_transform_calendar_scope_global_accepted(valid_cfg: dict) -> None:
    """calendar_scope='global' in params is valid."""
    valid_cfg["transforms"] = [
        {
            "name": "workday_norm",
            "class": "tsbricks.blocks.transforms.WorkdayNormalizeTransform",
            "scope": "global",
            "targets": ["y"],
            "params": {"calendar_scope": "global"},
        }
    ]
    cfg = parse_config(config=valid_cfg)

    assert cfg.transforms[0].params["calendar_scope"] == "global"


def test_workday_transform_calendar_scope_per_series_accepted(valid_cfg: dict) -> None:
    """calendar_scope='per_series' in params is valid."""
    valid_cfg["transforms"] = [
        {
            "name": "workday_norm",
            "class": "tsbricks.blocks.transforms.WorkdayNormalizeTransform",
            "scope": "global",
            "targets": ["y"],
            "params": {"calendar_scope": "per_series"},
        }
    ]
    cfg = parse_config(config=valid_cfg)

    assert cfg.transforms[0].params["calendar_scope"] == "per_series"


def test_workday_transform_missing_calendar_scope_raises(valid_cfg: dict) -> None:
    """WorkdayNormalizeTransform without calendar_scope in params raises."""
    valid_cfg["transforms"] = [
        {
            "name": "workday_norm",
            "class": "tsbricks.blocks.transforms.WorkdayNormalizeTransform",
            "scope": "global",
            "targets": ["y"],
            "params": {},
        }
    ]
    with pytest.raises(ValidationError, match="requires.*calendar_scope"):
        parse_config(config=valid_cfg)


def test_workday_transform_no_params_raises(valid_cfg: dict) -> None:
    """WorkdayNormalizeTransform with no params at all raises."""
    valid_cfg["transforms"] = [
        {
            "name": "workday_norm",
            "class": "tsbricks.blocks.transforms.WorkdayNormalizeTransform",
            "scope": "global",
            "targets": ["y"],
        }
    ]
    with pytest.raises(ValidationError, match="requires.*calendar_scope"):
        parse_config(config=valid_cfg)


def test_workday_transform_invalid_calendar_scope_raises(valid_cfg: dict) -> None:
    """Invalid calendar_scope value raises ValidationError."""
    valid_cfg["transforms"] = [
        {
            "name": "workday_norm",
            "class": "tsbricks.blocks.transforms.WorkdayNormalizeTransform",
            "scope": "global",
            "targets": ["y"],
            "params": {"calendar_scope": "weekly"},
        }
    ]
    with pytest.raises(ValidationError, match="invalid calendar_scope"):
        parse_config(config=valid_cfg)


# ---- Test fold config ----


def test_test_config_datetime_accepted(valid_cfg: dict) -> None:
    """Valid test config with datetime test_origin parses correctly."""
    valid_cfg["test"] = {"test_origin": "2024-01-01"}

    cfg = parse_config(config=valid_cfg)

    assert cfg.test is not None
    assert cfg.test.test_origin == "2024-01-01"


def test_test_config_integer_accepted(valid_cfg: dict) -> None:
    """Valid test config with integer test_origin parses correctly."""
    valid_cfg["data"]["freq"] = 1
    valid_cfg["cross_validation"]["forecast_origins"] = [10, 20]
    valid_cfg["test"] = {"test_origin": 30}

    cfg = parse_config(config=valid_cfg)

    assert cfg.test is not None
    assert cfg.test.test_origin == 30


def test_test_config_absent_is_none(valid_cfg: dict) -> None:
    """Omitting test block results in test=None."""
    cfg = parse_config(config=valid_cfg)

    assert cfg.test is None


def test_test_config_null_is_none(valid_cfg: dict) -> None:
    """Explicit test: null results in test=None."""
    valid_cfg["test"] = None

    cfg = parse_config(config=valid_cfg)

    assert cfg.test is None


def test_test_origin_type_mismatch_str_vs_int(
    valid_cfg: dict,
) -> None:
    """Integer test_origin with string forecast_origins raises."""
    valid_cfg["test"] = {"test_origin": 50}

    with pytest.raises(ValidationError, match="must be a string"):
        parse_config(config=valid_cfg)


def test_test_origin_type_mismatch_int_vs_str(
    valid_cfg: dict,
) -> None:
    """String test_origin with integer forecast_origins raises."""
    valid_cfg["data"]["freq"] = 1
    valid_cfg["cross_validation"]["forecast_origins"] = [10, 20]
    valid_cfg["test"] = {"test_origin": "2024-01-01"}

    with pytest.raises(ValidationError, match="must be an integer"):
        parse_config(config=valid_cfg)


def test_test_origin_not_after_max_origin_raises(
    valid_cfg: dict,
) -> None:
    """test_origin equal to max forecast_origin raises."""
    valid_cfg["test"] = {"test_origin": "2023-07-01"}

    with pytest.raises(ValidationError, match="strictly after"):
        parse_config(config=valid_cfg)


def test_test_origin_before_max_origin_raises(
    valid_cfg: dict,
) -> None:
    """test_origin before max forecast_origin raises."""
    valid_cfg["test"] = {"test_origin": "2023-01-01"}

    with pytest.raises(ValidationError, match="strictly after"):
        parse_config(config=valid_cfg)


def test_test_origin_int_not_after_max_raises(
    valid_cfg: dict,
) -> None:
    """Integer test_origin equal to max forecast_origin raises."""
    valid_cfg["data"]["freq"] = 1
    valid_cfg["cross_validation"]["forecast_origins"] = [10, 20]
    valid_cfg["test"] = {"test_origin": 20}

    with pytest.raises(ValidationError, match="strictly after"):
        parse_config(config=valid_cfg)


def test_test_horizon_rejected(valid_cfg: dict) -> None:
    """Providing horizon in test block raises with helpful message."""
    valid_cfg["test"] = {
        "test_origin": "2024-01-01",
        "horizon": 12,
    }

    with pytest.raises(ValidationError, match="test.horizon is not supported"):
        parse_config(config=valid_cfg)


def test_test_extra_field_rejected(valid_cfg: dict) -> None:
    """Unknown fields in test block are rejected."""
    valid_cfg["test"] = {
        "test_origin": "2024-01-01",
        "unknown_field": True,
    }

    with pytest.raises(ValidationError):
        parse_config(config=valid_cfg)


# ---- Mixed forecast_origins type rejection ----


def test_mixed_forecast_origins_types_raises(
    valid_cfg: dict,
) -> None:
    """Mixed str/int forecast_origins raises ValidationError."""
    valid_cfg["cross_validation"]["forecast_origins"] = [
        "2023-01-01",
        20,
    ]

    with pytest.raises(ValidationError, match="mixed types"):
        parse_config(config=valid_cfg)


# ---- Datetime ordering is temporal, not lexicographic ----


def test_test_origin_temporal_ordering(valid_cfg: dict) -> None:
    """Non-normalized date strings are compared temporally."""
    # "2023-9-01" sorts lexicographically before "2023-07-01"
    # but temporally it's after — should be accepted
    valid_cfg["cross_validation"]["forecast_origins"] = [
        "2023-01-01",
        "2023-07-01",
    ]
    valid_cfg["test"] = {"test_origin": "2023-9-01"}

    cfg = parse_config(config=valid_cfg)

    assert cfg.test is not None


# ---- Non-normalized date warnings ----


def test_non_normalized_forecast_origins_warns(
    valid_cfg: dict,
) -> None:
    """Non-normalized forecast_origins emit a UserWarning."""
    valid_cfg["cross_validation"]["forecast_origins"] = [
        "2023-1-01",
        "2023-07-01",
    ]

    with pytest.warns(UserWarning, match="non-normalized"):
        parse_config(config=valid_cfg)


def test_non_normalized_test_origin_warns(
    valid_cfg: dict,
) -> None:
    """Non-normalized test_origin emits a UserWarning."""
    valid_cfg["test"] = {"test_origin": "2024-1-01"}

    with pytest.warns(UserWarning, match="non-normalized"):
        parse_config(config=valid_cfg)


def test_normalized_dates_no_warning(valid_cfg: dict) -> None:
    """Properly normalized dates do not emit warnings."""
    valid_cfg["test"] = {"test_origin": "2024-01-01"}

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=UserWarning)
        parse_config(config=valid_cfg)


# ---- ParamResolverConfig ----


def test_param_resolver_config_round_trips() -> None:
    """ParamResolverConfig with all fields parses correctly."""
    prc = ParamResolverConfig(
        callable="my.module.resolver",
        params={"season_length": 12},
        grouping_columns=["category"],
    )

    assert prc.callable == "my.module.resolver"
    assert prc.params == {"season_length": 12}
    assert prc.grouping_columns == ["category"]


def test_param_resolver_config_minimal() -> None:
    """ParamResolverConfig with only callable parses correctly."""
    prc = ParamResolverConfig(callable="my.module.resolver")

    assert prc.params is None
    assert prc.grouping_columns is None


# ---- MetricDefinitionConfig fields ----


def test_metric_definition_all_fields() -> None:
    """MetricDefinitionConfig accepts all fields."""
    defn = MetricDefinitionConfig(
        name="wape_global",
        callable="my.metrics.wape",
        type="simple",
        scope="global",
        aggregation="pooled",
        per_series_params={"m": {"A": 12, "B": 6}},
        param_resolvers={"scale": ParamResolverConfig(callable="my.resolvers.scale")},
        aggregation_callable="my.agg.weighted_mean",
        aggregation_params={"normalize": True},
    )

    assert defn.scope == "global"
    assert defn.aggregation == "pooled"
    assert defn.per_series_params == {"m": {"A": 12, "B": 6}}
    assert "scale" in defn.param_resolvers
    assert defn.aggregation_callable == "my.agg.weighted_mean"
    assert defn.aggregation_params == {"normalize": True}


def test_metric_definition_defaults() -> None:
    """MetricDefinitionConfig applies correct defaults for optional fields."""
    defn = MetricDefinitionConfig(
        name="rmse",
        callable="tsbricks.blocks.metrics.rmse",
        type="simple",
    )

    assert defn.scope == "per_series"
    assert defn.aggregation == "per_fold_mean"
    assert defn.grouping_columns is None
    assert defn.per_series_params is None
    assert defn.param_resolvers is None
    assert defn.aggregation_callable is None
    assert defn.aggregation_params is None


# ---- Literal validation ----


def test_invalid_scope_raises() -> None:
    """Invalid scope value raises ValidationError."""
    with pytest.raises(ValidationError):
        MetricDefinitionConfig(
            name="rmse",
            callable="m.rmse",
            type="simple",
            scope="invalid",
        )


def test_invalid_aggregation_raises() -> None:
    """Invalid aggregation value raises ValidationError."""
    with pytest.raises(ValidationError):
        MetricDefinitionConfig(
            name="rmse",
            callable="m.rmse",
            type="simple",
            aggregation="invalid",
        )


# ---- Validator: scope=global requires aggregation_callable ----


def test_scope_global_requires_aggregation_callable() -> None:
    """scope='global' without aggregation_callable raises."""
    with pytest.raises(ValidationError, match="aggregation_callable"):
        MetricDefinitionConfig(
            name="wape",
            callable="m.wape",
            type="simple",
            scope="global",
        )


def test_scope_global_with_aggregation_callable_ok() -> None:
    """scope='global' with aggregation_callable is valid."""
    defn = MetricDefinitionConfig(
        name="wape",
        callable="m.wape",
        type="simple",
        scope="global",
        aggregation_callable="m.agg.weighted_mean",
    )

    assert defn.scope == "global"


# ---- Validator: no key overlap ----


def test_param_key_overlap_params_and_psp_raises() -> None:
    """Overlapping keys in params and per_series_params raises."""
    with pytest.raises(ValidationError, match="overlapping keys"):
        MetricDefinitionConfig(
            name="rmsse",
            callable="m.rmsse",
            type="context_aware",
            params={"m": 1},
            per_series_params={"m": {"A": 12}},
        )


def test_param_key_overlap_params_and_resolvers_raises() -> None:
    """Overlapping keys in params and param_resolvers raises."""
    with pytest.raises(ValidationError, match="overlapping keys"):
        MetricDefinitionConfig(
            name="rmsse",
            callable="m.rmsse",
            type="context_aware",
            params={"scale": 1.0},
            param_resolvers={"scale": ParamResolverConfig(callable="m.resolve_scale")},
        )


def test_param_key_overlap_psp_and_resolvers_raises() -> None:
    """Overlapping keys in per_series_params and resolvers."""
    with pytest.raises(ValidationError, match="overlapping keys"):
        MetricDefinitionConfig(
            name="rmsse",
            callable="m.rmsse",
            type="context_aware",
            per_series_params={"scale": {"A": 1.0}},
            param_resolvers={"scale": ParamResolverConfig(callable="m.resolve_scale")},
        )


def test_no_key_overlap_disjoint_ok() -> None:
    """Disjoint keys across params, psp, resolvers is valid."""
    defn = MetricDefinitionConfig(
        name="rmsse",
        callable="m.rmsse",
        type="context_aware",
        params={"m": 1},
        per_series_params={"fallback": {"A": 1.0}},
        param_resolvers={"scale": ParamResolverConfig(callable="m.resolve_scale")},
    )

    assert defn.params == {"m": 1}


# ---- Validator: scope=group requires grouping_columns ----


def test_scope_group_no_grouping_columns_raises() -> None:
    """scope='group' with no grouping_columns anywhere raises."""
    with pytest.raises(ValidationError, match="grouping_columns"):
        MetricsConfig(
            definitions=[
                MetricDefinitionConfig(
                    name="wape",
                    callable="m.wape",
                    type="simple",
                    scope="group",
                )
            ],
        )


def test_scope_group_with_defn_grouping_columns_ok() -> None:
    """scope='group' with grouping_columns on definition works."""
    cfg = MetricsConfig(
        definitions=[
            MetricDefinitionConfig(
                name="wape",
                callable="m.wape",
                type="simple",
                scope="group",
                grouping_columns=["category"],
            )
        ],
    )

    assert cfg.definitions[0].grouping_columns == ["category"]


def test_scope_group_with_top_level_grouping_columns_ok() -> None:
    """scope='group' with top-level grouping_columns works."""
    cfg = MetricsConfig(
        definitions=[
            MetricDefinitionConfig(
                name="wape",
                callable="m.wape",
                type="simple",
                scope="group",
            )
        ],
        grouping_columns=["category"],
    )

    assert cfg.grouping_columns == ["category"]


# ---- MetricsConfig fields ----


def test_metrics_config_grouping_and_weights_source() -> None:
    """grouping_source and weights_source parse correctly."""
    cfg = MetricsConfig(
        definitions=[
            MetricDefinitionConfig(
                name="rmse",
                callable="m.rmse",
                type="simple",
            )
        ],
        grouping_source="/path/to/grouping.parquet",
        weights_source="/path/to/weights.parquet",
    )

    assert cfg.grouping_source == "/path/to/grouping.parquet"
    assert cfg.weights_source == "/path/to/weights.parquet"


# ---- Full config defaults ----


def test_full_config_defaults(valid_cfg: dict) -> None:
    """Parsed config applies correct defaults for all optional metric fields."""
    cfg = parse_config(config=valid_cfg)

    assert isinstance(cfg, BacktestConfig)
    defn = cfg.metrics.definitions[0]
    assert defn.scope == "per_series"
    assert defn.aggregation == "per_fold_mean"
    assert defn.grouping_columns is None
    assert defn.per_series_params is None
    assert defn.param_resolvers is None
    assert defn.aggregation_callable is None
    assert defn.aggregation_params is None
    assert cfg.metrics.grouping_source is None
    assert cfg.metrics.weights_source is None


# ---- Empty grouping_columns rejected ----


def test_empty_defn_grouping_columns_raises() -> None:
    """Empty grouping_columns on metric definition raises."""
    with pytest.raises(ValidationError):
        MetricDefinitionConfig(
            name="wape",
            callable="m.wape",
            type="simple",
            scope="group",
            grouping_columns=[],
        )


def test_empty_top_level_grouping_columns_raises() -> None:
    """Empty grouping_columns on MetricsConfig raises."""
    with pytest.raises(ValidationError):
        MetricsConfig(
            definitions=[
                MetricDefinitionConfig(
                    name="rmse",
                    callable="m.rmse",
                    type="simple",
                )
            ],
            grouping_columns=[],
        )


def test_multi_defn_grouping_columns_raises() -> None:
    """Multiple grouping_columns on metric definition raises."""
    with pytest.raises(ValidationError, match="single-column grouping"):
        MetricDefinitionConfig(
            name="wape",
            callable="m.wape",
            type="simple",
            scope="group",
            grouping_columns=["category", "region"],
        )


def test_multi_top_level_grouping_columns_raises() -> None:
    """Multiple grouping_columns on MetricsConfig raises."""
    with pytest.raises(ValidationError, match="single-column grouping"):
        MetricsConfig(
            definitions=[
                MetricDefinitionConfig(
                    name="rmse",
                    callable="m.rmse",
                    type="simple",
                )
            ],
            grouping_columns=["category", "region"],
        )


# ---- End-to-end parse_config with metric fields ----


def _cfg_with_all_metric_fields(valid_cfg: dict) -> dict:
    """Inject all metric fields into valid_cfg for parse_config tests."""
    valid_cfg["metrics"] = {
        "definitions": [
            {
                "name": "rmse",
                "callable": "tsbricks.blocks.metrics.rmse",
                "type": "simple",
            },
            {
                "name": "wape_global",
                "callable": "my.metrics.wape",
                "type": "simple",
                "scope": "global",
                "aggregation": "pooled",
                "grouping_columns": ["category"],
                "per_series_params": {
                    "m": {"A": 12, "B": 6},
                },
                "param_resolvers": {
                    "scale": {
                        "callable": "my.resolvers.scale",
                        "params": {"season_length": 12},
                    },
                },
                "aggregation_callable": "my.agg.weighted_mean",
                "aggregation_params": {"normalize": True},
            },
        ],
        "grouping_columns": ["region"],
        "grouping_source": "/data/grouping.parquet",
        "weights_source": "/data/weights.parquet",
    }
    return valid_cfg


def test_parse_config_definition_defaults(
    valid_cfg: dict,
) -> None:
    """Definition with only required fields preserves defaults after parsing."""
    cfg = parse_config(config=_cfg_with_all_metric_fields(valid_cfg))

    d0 = cfg.metrics.definitions[0]
    assert d0.scope == "per_series"
    assert d0.aggregation == "per_fold_mean"
    assert d0.grouping_columns is None
    assert d0.per_series_params is None
    assert d0.param_resolvers is None
    assert d0.aggregation_callable is None
    assert d0.aggregation_params is None


def test_parse_config_definition_all_fields(
    valid_cfg: dict,
) -> None:
    """All metric definition fields parse correctly from raw dict."""
    cfg = parse_config(config=_cfg_with_all_metric_fields(valid_cfg))

    d1 = cfg.metrics.definitions[1]
    assert d1.scope == "global"
    assert d1.aggregation == "pooled"
    assert d1.grouping_columns == ["category"]
    assert d1.per_series_params == {"m": {"A": 12, "B": 6}}
    assert "scale" in d1.param_resolvers
    assert d1.param_resolvers["scale"].callable == "my.resolvers.scale"
    assert d1.param_resolvers["scale"].params == {
        "season_length": 12,
    }
    assert d1.aggregation_callable == "my.agg.weighted_mean"
    assert d1.aggregation_params == {"normalize": True}


def test_parse_config_top_level_metrics_fields(
    valid_cfg: dict,
) -> None:
    """Top-level MetricsConfig fields parse correctly from raw dict."""
    cfg = parse_config(config=_cfg_with_all_metric_fields(valid_cfg))

    assert cfg.metrics.grouping_columns == ["region"]
    assert cfg.metrics.grouping_source == "/data/grouping.parquet"
    assert cfg.metrics.weights_source == "/data/weights.parquet"


# ---- Variable forecast horizon: CrossValidationConfig ----


def _variable_cv_config(
    origins: list[dict],
) -> CrossValidationConfig:
    """Build a variable-horizon CrossValidationConfig."""
    return CrossValidationConfig(
        mode="explicit",
        forecast_origins=origins,
    )


def test_variable_horizon_parses_datetime() -> None:
    """Variable-horizon config with datetime origins parses."""
    cfg = _variable_cv_config(
        [
            {"origin": "2025-06-01", "horizon": 6},
            {"origin": "2025-12-01", "horizon": 3},
        ]
    )

    assert cfg.horizon is None
    assert len(cfg.forecast_origins) == 2
    assert isinstance(cfg.forecast_origins[0], ForecastOriginConfig)


def test_variable_horizon_parses_integer() -> None:
    """Variable-horizon config with integer origins parses."""
    cfg = _variable_cv_config(
        [
            {"origin": 10, "horizon": 5},
            {"origin": 20, "horizon": 3},
        ]
    )

    assert cfg.horizon is None
    assert cfg.forecast_origins[0].origin == 10
    assert cfg.forecast_origins[1].horizon == 3


def test_variable_horizon_missing_horizon_raises() -> None:
    """Origin object without horizon raises ValidationError."""
    with pytest.raises(ValidationError):
        _variable_cv_config(
            [
                {"origin": "2025-06-01", "horizon": 6},
                {"origin": "2025-12-01"},
            ]
        )


def test_variable_horizon_zero_horizon_raises() -> None:
    """Origin object with horizon=0 raises ValidationError."""
    with pytest.raises(ValidationError):
        _variable_cv_config(
            [
                {"origin": "2025-06-01", "horizon": 0},
            ]
        )


def test_variable_horizon_negative_horizon_raises() -> None:
    """Origin object with negative horizon raises."""
    with pytest.raises(ValidationError):
        _variable_cv_config(
            [
                {"origin": "2025-06-01", "horizon": -1},
            ]
        )


def test_both_horizon_and_object_origins_raises() -> None:
    """Top-level horizon + object origins is invalid."""
    with pytest.raises(ValidationError, match="Cannot specify both"):
        CrossValidationConfig(
            mode="explicit",
            horizon=6,
            forecast_origins=[
                {"origin": "2025-06-01", "horizon": 3},
            ],
        )


def test_no_horizon_and_flat_origins_raises() -> None:
    """No top-level horizon + flat origins is invalid."""
    with pytest.raises(ValidationError, match="horizon.*not set"):
        CrossValidationConfig(
            mode="explicit",
            forecast_origins=["2025-06-01", "2025-12-01"],
        )


def test_variable_horizon_mixed_origin_types_raises() -> None:
    """Mixed str/int origins in variable format raises."""
    with pytest.raises(ValidationError, match="mixed types"):
        _variable_cv_config(
            [
                {"origin": "2025-06-01", "horizon": 6},
                {"origin": 10, "horizon": 3},
            ]
        )


def test_variable_horizon_non_normalized_dates_warns() -> None:
    """Non-normalized dates in variable format emit warning."""
    with pytest.warns(UserWarning, match="non-normalized"):
        _variable_cv_config(
            [
                {"origin": "2025-6-01", "horizon": 6},
            ]
        )


# ---- origin_horizon_pairs() ----


def test_origin_horizon_pairs_uniform() -> None:
    """Uniform config returns consistent (origin, horizon) pairs."""
    cfg = CrossValidationConfig(
        mode="explicit",
        horizon=6,
        forecast_origins=["2025-06-01", "2025-12-01"],
    )

    pairs = cfg.origin_horizon_pairs()

    assert pairs == [
        ("2025-06-01", 6),
        ("2025-12-01", 6),
    ]


def test_origin_horizon_pairs_variable() -> None:
    """Variable config returns per-origin (origin, horizon) pairs."""
    cfg = _variable_cv_config(
        [
            {"origin": "2025-06-01", "horizon": 6},
            {"origin": "2025-12-01", "horizon": 3},
        ]
    )

    pairs = cfg.origin_horizon_pairs()

    assert pairs == [
        ("2025-06-01", 6),
        ("2025-12-01", 3),
    ]


def test_origin_horizon_pairs_integer() -> None:
    """origin_horizon_pairs works with integer origins."""
    cfg = CrossValidationConfig(
        mode="explicit",
        horizon=5,
        forecast_origins=[10, 20],
    )

    pairs = cfg.origin_horizon_pairs()

    assert pairs == [(10, 5), (20, 5)]


# ---- raw_origins() ----


def test_raw_origins_uniform() -> None:
    """raw_origins returns flat list for uniform config."""
    cfg = CrossValidationConfig(
        mode="explicit",
        horizon=6,
        forecast_origins=["2025-06-01", "2025-12-01"],
    )

    assert cfg.raw_origins() == [
        "2025-06-01",
        "2025-12-01",
    ]


def test_raw_origins_variable() -> None:
    """raw_origins extracts origin values from objects."""
    cfg = _variable_cv_config(
        [
            {"origin": "2025-06-01", "horizon": 6},
            {"origin": "2025-12-01", "horizon": 3},
        ]
    )

    assert cfg.raw_origins() == [
        "2025-06-01",
        "2025-12-01",
    ]


# ---- Variable horizon via parse_config (full round-trip) ----


def test_parse_config_variable_horizon(
    valid_cfg: dict,
) -> None:
    """Full config with variable horizons parses correctly."""
    del valid_cfg["cross_validation"]["horizon"]
    valid_cfg["cross_validation"]["forecast_origins"] = [
        {"origin": "2023-01-01", "horizon": 6},
        {"origin": "2023-07-01", "horizon": 3},
    ]

    cfg = parse_config(config=valid_cfg)

    assert cfg.cross_validation.horizon is None
    assert len(cfg.cross_validation.forecast_origins) == 2
    pairs = cfg.cross_validation.origin_horizon_pairs()
    assert pairs == [
        ("2023-01-01", 6),
        ("2023-07-01", 3),
    ]


def test_parse_config_variable_horizon_with_test_fold(
    valid_cfg: dict,
) -> None:
    """Variable-horizon config with test fold parses."""
    del valid_cfg["cross_validation"]["horizon"]
    valid_cfg["cross_validation"]["forecast_origins"] = [
        {"origin": "2023-01-01", "horizon": 6},
        {"origin": "2023-07-01", "horizon": 3},
    ]
    valid_cfg["test"] = {"test_origin": "2024-01-01"}

    cfg = parse_config(config=valid_cfg)

    assert cfg.test is not None
    assert cfg.test.test_origin == "2024-01-01"
