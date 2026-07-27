from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from tsbricks.runner.model_invocation import (
    invoke_model,
    resolve_model,
    resolve_predict,
)


# ---- Stand-in configs ----
#
# Deliberately dataclasses rather than the real ModelConfig: runner/ must
# not import from backtesting/ (PACKAGE_MAINTAINER_SPEC §2.3), and these
# tests demonstrate that independence. The dotted paths below are real and
# are resolved through the real dynamic_import -- only the config *type* is
# a stand-in. The real ModelConfig is exercised in tests/backtesting/ and
# in one end-to-end integration test.


@dataclass
class _ModelConfig:
    """Minimal stand-in for ModelConfig used in tests."""

    callable: str
    hyperparameters: dict | None = field(default=None)
    predict_callable: str | None = field(default=None)
    predict_params: dict | None = field(default=None)


@dataclass
class _LegacyModelConfig:
    """Config shaped as it was before this spec added the predict fields.

    Carries no ``predict_callable`` or ``predict_params`` attribute at
    all, which is why both resolvers reach for them with ``getattr``.
    """

    callable: str
    hyperparameters: dict | None = field(default=None)


_PREDICT_ONLY = "tsbricks._testing.dummy_models.predict_only"
_FORECAST_ONLY = "tsbricks._testing.dummy_models.forecast_only"


# ---- resolve_model ----


def test_resolve_model_returns_callable_and_params() -> None:
    """resolve_model returns the imported callable and hyperparameters dict."""
    cfg = _ModelConfig(
        callable="tsbricks._testing.dummy_models.forecast_only",
        hyperparameters={"season_length": 12},
    )
    model_fn, params = resolve_model(cfg)

    assert callable(model_fn)
    assert params == {"season_length": 12}


def test_resolve_model_none_hyperparameters() -> None:
    """None hyperparameters normalises to empty dict."""
    cfg = _ModelConfig(
        callable="tsbricks._testing.dummy_models.forecast_only",
        hyperparameters=None,
    )
    _, params = resolve_model(cfg)

    assert params == {}


def test_resolve_model_merges_predict_params() -> None:
    """predict_params are merged under hyperparameters into one kwargs dict."""
    cfg = _ModelConfig(
        callable=_FORECAST_ONLY,
        hyperparameters={"num_leaves": 31},
        predict_params={"level": [80, 95]},
    )
    _, kwargs = resolve_model(cfg)

    assert kwargs == {"num_leaves": 31, "level": [80, 95]}


def test_resolve_model_hyperparameters_win_on_overlap() -> None:
    """On overlap the hyperparameters value governs, and nothing warns here.

    The overlap warning is a parse-time schema concern (tested in
    tests/backtesting/test_schema.py); the resolver only applies the
    precedence. Warning here would tag a config-authoring fact with a
    fold id and repeat it once per fold inside run_backtest.
    """
    import warnings

    cfg = _ModelConfig(
        callable=_FORECAST_ONLY,
        hyperparameters={"n_jobs": 16},
        predict_params={"n_jobs": 2},
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=UserWarning)
        _, kwargs = resolve_model(cfg)

    assert kwargs["n_jobs"] == 16


def test_resolve_model_without_predict_params_attribute() -> None:
    """A config predating predict_params resolves via the getattr path."""
    cfg = _LegacyModelConfig(
        callable=_FORECAST_ONLY,
        hyperparameters={"num_leaves": 31},
    )

    assert not hasattr(cfg, "predict_params")

    model_fn, kwargs = resolve_model(cfg)

    assert callable(model_fn)
    assert kwargs == {"num_leaves": 31}


@pytest.mark.parametrize("field_name", ["hyperparameters", "predict_params"])
def test_resolve_model_horizon_key_raises(field_name: str) -> None:
    """A horizon key in either params dict raises ValueError naming it."""
    cfg = _ModelConfig(callable=_FORECAST_ONLY, **{field_name: {"horizon": 12}})

    with pytest.raises(ValueError, match=rf"model_config\.{field_name}.*horizon"):
        resolve_model(cfg)


# ---- resolve_predict ----


def test_resolve_predict_returns_callable_and_params() -> None:
    """resolve_predict returns the imported callable and its predict_params."""
    cfg = _ModelConfig(
        callable=_FORECAST_ONLY,
        predict_callable=_PREDICT_ONLY,
        predict_params={"level": [80, 95]},
    )
    predict_fn, params = resolve_predict(cfg)

    assert callable(predict_fn)
    assert predict_fn.__name__ == "predict_only"
    assert params == {"level": [80, 95]}


def test_resolve_predict_none_predict_params() -> None:
    """None predict_params normalises to an empty dict."""
    cfg = _ModelConfig(
        callable=_FORECAST_ONLY,
        predict_callable=_PREDICT_ONLY,
        predict_params=None,
    )
    _, params = resolve_predict(cfg)

    assert params == {}


def test_resolve_predict_none_callable_raises() -> None:
    """predict_callable=None raises ValueError with an actionable remedy."""
    cfg = _ModelConfig(callable=_FORECAST_ONLY, predict_callable=None)

    with pytest.raises(ValueError, match="no 'predict_callable'"):
        resolve_predict(cfg)


def test_resolve_predict_missing_attribute_raises_same_error() -> None:
    """A config with no predict_callable attribute raises the same ValueError.

    This is the test that justifies reading the field with getattr rather
    than direct attribute access: one user mistake ("I did not tell it how
    to predict") produces one exception with one remedy, whether the field
    is absent or None. Direct access would raise AttributeError here and
    ValueError in the test above.
    """
    cfg = _LegacyModelConfig(callable=_FORECAST_ONLY)

    assert not hasattr(cfg, "predict_callable")

    with pytest.raises(ValueError, match="no 'predict_callable'"):
        resolve_predict(cfg)


def test_resolve_predict_horizon_key_raises() -> None:
    """A horizon key in predict_params raises ValueError."""
    cfg = _ModelConfig(
        callable=_FORECAST_ONLY,
        predict_callable=_PREDICT_ONLY,
        predict_params={"horizon": 12},
    )

    with pytest.raises(ValueError, match=r"predict_params.*horizon"):
        resolve_predict(cfg)


def test_resolve_predict_returns_a_copy() -> None:
    """Mutating the returned kwargs dict does not reach the config object."""
    # dict() is a shallow copy: rebinding keys is safe, but mutating a
    # nested value in place (kwargs["level"].append(...)) would reach the
    # config.

    cfg = _ModelConfig(
        callable=_FORECAST_ONLY,
        predict_callable=_PREDICT_ONLY,
        predict_params={"level": [80, 95]},
    )

    _, kwargs = resolve_predict(cfg)
    assert kwargs is not cfg.predict_params, (
        "resolve_predict returned the config's own dict, not a copy"
    )

    kwargs["level"] = [50]
    kwargs["extra"] = True

    assert cfg.predict_params == {"level": [80, 95]}


# ---- invoke_model ----


def test_invoke_dataframe_only(panel_df: pd.DataFrame) -> None:
    """Model returning DataFrame → (forecast, None, None)."""
    cfg = _ModelConfig(callable="tsbricks._testing.dummy_models.forecast_only")
    forecast, fitted, model_obj = invoke_model(panel_df, cfg, horizon=3)

    assert isinstance(forecast, pd.DataFrame)
    assert "ypred" in forecast.columns
    assert len(forecast) == 3 * panel_df["unique_id"].nunique()
    assert fitted is None
    assert model_obj is None


def test_invoke_tuple_of_two(panel_df: pd.DataFrame) -> None:
    """Model returning (forecast, fitted) → (forecast, fitted, None)."""
    cfg = _ModelConfig(callable="tsbricks._testing.dummy_models.forecast_and_fitted")
    forecast, fitted, model_obj = invoke_model(panel_df, cfg, horizon=3)

    assert isinstance(forecast, pd.DataFrame)
    assert isinstance(fitted, pd.DataFrame)
    assert "ypred" in fitted.columns
    assert model_obj is None


def test_invoke_tuple_of_three(panel_df: pd.DataFrame) -> None:
    """Model returning (forecast, fitted, model_object) → all three."""
    cfg = _ModelConfig(
        callable="tsbricks._testing.dummy_models.forecast_fitted_and_model"
    )
    forecast, fitted, model_obj = invoke_model(panel_df, cfg, horizon=3)

    assert isinstance(forecast, pd.DataFrame)
    assert isinstance(fitted, pd.DataFrame)
    assert isinstance(model_obj, dict)
    assert model_obj["name"] == "dummy"


def test_invoke_passes_hyperparameters(panel_df: pd.DataFrame) -> None:
    """Hyperparameters from config are forwarded to the model callable."""
    cfg = _ModelConfig(
        callable="tsbricks._testing.dummy_models.forecast_fitted_and_model",
        hyperparameters={"alpha": 0.5},
    )
    _, _, model_obj = invoke_model(panel_df, cfg, horizon=3)

    assert model_obj["hyperparameters"]["alpha"] == pytest.approx(0.5)


def test_invoke_forwards_future_x_df(panel_df: pd.DataFrame) -> None:
    """future_x_df is forwarded to the model callable as a kwarg."""
    cfg = _ModelConfig(
        callable="tsbricks._testing.dummy_models.forecast_with_exogenous",
    )
    future_x = pd.DataFrame({"ds": [1], "unique_id": ["A"], "price": [9.99]})
    _, _, model_obj = invoke_model(panel_df, cfg, horizon=3, future_x_df=future_x)

    assert model_obj is not None
    pd.testing.assert_frame_equal(model_obj["future_x_df"], future_x)


def test_invoke_invalid_return_type_raises(panel_df: pd.DataFrame) -> None:
    """Model returning an unexpected type raises TypeError."""
    cfg = _ModelConfig(callable="tsbricks._testing.dummy_models.returns_int")

    with pytest.raises(TypeError, match="Model callable must return"):
        invoke_model(panel_df, cfg, horizon=3)


def test_invoke_tuple_of_one_raises(panel_df: pd.DataFrame) -> None:
    """Model returning a 1-tuple raises TypeError."""
    cfg = _ModelConfig(callable="tsbricks._testing.dummy_models.returns_tuple_of_one")

    with pytest.raises(TypeError, match="Model callable must return"):
        invoke_model(panel_df, cfg, horizon=3)


def test_invoke_tuple_of_four_raises(panel_df: pd.DataFrame) -> None:
    """Model returning a 4-tuple raises TypeError."""
    cfg = _ModelConfig(callable="tsbricks._testing.dummy_models.returns_tuple_of_four")

    with pytest.raises(TypeError, match="Model callable must return"):
        invoke_model(panel_df, cfg, horizon=3)
