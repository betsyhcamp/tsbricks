"""Tests for fold-level resilience in run_backtest."""

from __future__ import annotations

import pandas as pd
import pytest

from tsbricks.backtesting import BacktestResults, run_backtest


def _synthetic_monthly_panel() -> pd.DataFrame:
    """Two-series monthly panel with a simple linear trend (24 months)."""
    dates = pd.date_range("2022-01-01", periods=24, freq="MS")
    rows = []
    for uid, base in [("A", 10.0), ("B", 50.0)]:
        for i, ds in enumerate(dates):
            rows.append({"unique_id": uid, "ds": ds, "y": base + float(i)})
    return pd.DataFrame(rows)


def _synthetic_monthly_panel_long() -> pd.DataFrame:
    """Two-series monthly panel (36 months) for test fold tests."""
    dates = pd.date_range("2022-01-01", periods=36, freq="MS")
    rows = []
    for uid, base in [("A", 10.0), ("B", 50.0)]:
        for i, ds in enumerate(dates):
            rows.append({"unique_id": uid, "ds": ds, "y": base + float(i)})
    return pd.DataFrame(rows)


def _config_with_failing_model() -> dict:
    """Config where the model always raises."""
    return {
        "data": {"freq": "MS"},
        "cross_validation": {
            "mode": "explicit",
            "horizon": 6,
            "forecast_origins": ["2023-01-01", "2023-06-01"],
        },
        "model": {
            "callable": ("tsbricks._testing.dummy_models.always_fails"),
        },
        "metrics": {
            "definitions": [
                {
                    "name": "rmse",
                    "callable": "tsbricks.blocks.metrics.rmse",
                    "type": "simple",
                }
            ],
        },
    }


def _config_with_working_model() -> dict:
    """Config where the model works."""
    return {
        "data": {"freq": "MS"},
        "cross_validation": {
            "mode": "explicit",
            "horizon": 6,
            "forecast_origins": ["2023-01-01", "2023-06-01"],
        },
        "model": {
            "callable": ("tsbricks._testing.dummy_models.forecast_only"),
        },
        "metrics": {
            "definitions": [
                {
                    "name": "rmse",
                    "callable": "tsbricks.blocks.metrics.rmse",
                    "type": "simple",
                }
            ],
        },
    }


# ---- CV fold resilience ----


def test_all_folds_fail_raises_runtime_error():
    """When all CV folds fail, raise RuntimeError with run_summary attached."""
    df = _synthetic_monthly_panel()
    cfg = _config_with_failing_model()

    with pytest.raises(RuntimeError, match="All CV folds failed") as exc_info:
        run_backtest(config=cfg, df=df)

    assert hasattr(exc_info.value, "run_summary")
    errors = exc_info.value.run_summary["errors"]
    assert len(errors) == 2
    assert all(e["error_type"] == "ValueError" for e in errors)


def test_single_fold_failure_skips_and_continues(monkeypatch):
    """When one fold fails, the other succeeds."""
    df = _synthetic_monthly_panel()
    cfg = _config_with_working_model()

    call_count = {"n": 0}

    def fail_first_fold(train_df, horizon, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ValueError("First fold fails")
        from tsbricks._testing.dummy_models import forecast_only

        return forecast_only(train_df, horizon, **kwargs)

    import tsbricks._testing.dummy_models as dm

    monkeypatch.setattr(dm, "fail_first_fold", fail_first_fold, raising=False)
    cfg["model"]["callable"] = "tsbricks._testing.dummy_models.fail_first_fold"

    with pytest.warns(UserWarning, match="failed and was skipped"):
        results = run_backtest(config=cfg, df=df)

    # Only one fold succeeded
    assert isinstance(results, BacktestResults)
    assert len(results.cv.forecasts_per_fold) == 1
    assert "fold_1" in results.cv.forecasts_per_fold

    # Error captured in run_summary
    errors = results.run_summary["errors"]
    assert len(errors) == 1
    assert errors[0]["fold"] == "fold_0"
    assert errors[0]["error_type"] == "ValueError"
    assert "First fold fails" in errors[0]["message"]
    assert errors[0]["traceback"] is not None
    assert errors[0]["unique_id"] is None
    assert errors[0]["metric"] is None


def test_all_folds_fail_emits_warnings():
    """Each failed fold emits a UserWarning before RuntimeError."""
    df = _synthetic_monthly_panel()
    cfg = _config_with_failing_model()

    with pytest.warns(UserWarning) as warning_records:
        with pytest.raises(RuntimeError):
            run_backtest(config=cfg, df=df)

    # Two folds, two warnings
    fold_warnings = [
        w for w in warning_records if "failed and was skipped" in str(w.message)
    ]
    assert len(fold_warnings) == 2


# ---- test fold resilience ----


def test_test_fold_failure_preserves_cv_results(monkeypatch):
    """When test fold fails, CV results are still returned."""
    df = _synthetic_monthly_panel_long()
    cfg = _config_with_working_model()
    cfg["test"] = {"test_origin": "2023-07-01"}

    call_count = {"n": 0}

    def fail_on_third_call(train_df, horizon, **kwargs):
        call_count["n"] += 1
        # First two calls are CV folds, third is test fold
        if call_count["n"] == 3:
            raise ValueError("Test fold fails")
        from tsbricks._testing.dummy_models import forecast_only

        return forecast_only(train_df, horizon, **kwargs)

    import tsbricks._testing.dummy_models as dm

    monkeypatch.setattr(dm, "fail_on_third_call", fail_on_third_call, raising=False)
    cfg["model"]["callable"] = "tsbricks._testing.dummy_models.fail_on_third_call"

    with pytest.warns(UserWarning, match="Test fold failed and was skipped"):
        results = run_backtest(config=cfg, df=df)

    # CV results intact
    assert isinstance(results, BacktestResults)
    assert len(results.cv.forecasts_per_fold) == 2

    # Test results are None
    assert results.test is None

    # Error captured in run_summary
    errors = results.run_summary["errors"]
    test_errors = [e for e in errors if e["fold"] == "test"]
    assert len(test_errors) == 1
    assert test_errors[0]["error_type"] == "ValueError"
    assert "Test fold fails" in test_errors[0]["message"]


# ---- warning capture ----


def _config_with_warning_model() -> dict:
    """Config where the model emits a UserWarning."""
    return {
        "data": {"freq": "MS"},
        "cross_validation": {
            "mode": "explicit",
            "horizon": 6,
            "forecast_origins": ["2023-01-01", "2023-06-01"],
        },
        "model": {
            "callable": ("tsbricks._testing.dummy_models.forecast_with_warning"),
        },
        "metrics": {
            "definitions": [
                {
                    "name": "rmse",
                    "callable": "tsbricks.blocks.metrics.rmse",
                    "type": "simple",
                }
            ],
        },
    }


def test_model_warnings_captured_in_run_summary():
    """Warnings emitted by the model callable appear in run_summary."""
    df = _synthetic_monthly_panel()
    cfg = _config_with_warning_model()

    results = run_backtest(config=cfg, df=df)

    model_warnings = [
        w for w in results.run_summary["warnings"] if w["stage"] == "model"
    ]
    assert len(model_warnings) == 2  # one per fold
    assert all("Model convergence issue" in w["message"] for w in model_warnings)
    assert all(w["category"] == "UserWarning" for w in model_warnings)
    assert {w["fold"] for w in model_warnings} == {"fold_0", "fold_1"}


def test_warning_entry_has_all_fields():
    """Each warning entry has the expected schema fields."""
    df = _synthetic_monthly_panel()
    cfg = _config_with_warning_model()

    results = run_backtest(config=cfg, df=df)

    expected_keys = {
        "fold",
        "stage",
        "category",
        "message",
        "filename",
        "lineno",
        "unique_id",
    }
    for w in results.run_summary["warnings"]:
        assert set(w.keys()) == expected_keys


def test_fold_error_stage_is_model_when_model_fails(monkeypatch):
    """Fold-level error records stage='model' when model raises."""
    df = _synthetic_monthly_panel()
    cfg = _config_with_working_model()

    call_count = {"n": 0}

    def fail_first_fold(train_df, horizon, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ValueError("Model blew up")
        from tsbricks._testing.dummy_models import forecast_only

        return forecast_only(train_df, horizon, **kwargs)

    import tsbricks._testing.dummy_models as dm

    monkeypatch.setattr(dm, "fail_first_fold_stage", fail_first_fold, raising=False)
    cfg["model"]["callable"] = "tsbricks._testing.dummy_models.fail_first_fold_stage"

    with pytest.warns(UserWarning):
        results = run_backtest(config=cfg, df=df)

    errors = results.run_summary["errors"]
    assert len(errors) == 1
    assert errors[0]["stage"] == "model"


def test_no_warnings_yields_empty_list():
    """When nothing warns, run_summary['warnings'] is empty."""
    df = _synthetic_monthly_panel()
    cfg = _config_with_working_model()

    results = run_backtest(config=cfg, df=df)

    assert results.run_summary["warnings"] == []
    assert results.run_summary["errors"] == []
