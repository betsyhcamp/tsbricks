from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from tsbricks.runner.model_invocation import invoke_model, resolve_model


# ---- Stand-in config (replaced by Pydantic model in Phase 5) ----


@dataclass
class _ModelConfig:
    """Minimal stand-in for ModelConfig used in tests."""

    callable: str
    hyperparameters: dict | None = field(default=None)


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

    assert model_obj["hyperparameters"]["alpha"] == 0.5


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
