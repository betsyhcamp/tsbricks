"""Integration test: end-to-end backtest with dummy model."""

from __future__ import annotations

import pandas as pd

from tsbricks.backtesting import BacktestResults, run_backtest


def _synthetic_monthly_panel() -> pd.DataFrame:
    """Two-series monthly panel with a simple linear trend (24 months)."""
    dates = pd.date_range("2022-01-01", periods=24, freq="MS")
    rows = []
    for uid, base in [("A", 10.0), ("B", 50.0)]:
        for i, ds in enumerate(dates):
            rows.append({"unique_id": uid, "ds": ds, "y": base + float(i)})
    return pd.DataFrame(rows)


def _minimal_config() -> dict:
    """Config with dummy model, no transforms, RMSE metric, 2 origins."""
    return {
        "data": {"freq": "MS"},
        "cross_validation": {
            "mode": "explicit",
            "horizon": 6,
            "forecast_origins": ["2023-01-01", "2023-06-01"],
        },
        "model": {
            "callable": "tsbricks._testing.dummy_models.forecast_only",
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


def test_minimal_backtest_end_to_end() -> None:
    """Smoke test: synthetic data + dummy model -> BacktestResults."""
    df = _synthetic_monthly_panel()
    cfg = _minimal_config()

    results = run_backtest(config=cfg, df=df)

    # Return type
    assert isinstance(results, BacktestResults)

    # Metrics DataFrame is populated and has expected schema
    metrics = results.cv.metrics
    assert len(metrics) > 0
    assert list(metrics.columns) == [
        "metric_name",
        "unique_id",
        "fold",
        "aggregation",
        "value",
    ]

    # Correct fold count (2 origins = 2 folds)
    assert len(results.cv.forecasts_per_fold) == 2
    assert len(results.cv.train_val_splits_per_fold) == 2

    # Fold origins match configured origins
    assert len(results.cv.fold_origins) == 2
    assert results.cv.fold_origins == [
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-06-01"),
    ]

    # Horizon preserved
    assert results.horizon == 6

    # Config preserved
    assert results.config == cfg

    # No test fold when not configured
    assert results.test is None


def _synthetic_monthly_panel_long() -> pd.DataFrame:
    """Two-series monthly panel with a simple linear trend (36 months)."""
    dates = pd.date_range("2022-01-01", periods=36, freq="MS")
    rows = []
    for uid, base in [("A", 10.0), ("B", 50.0)]:
        for i, ds in enumerate(dates):
            rows.append({"unique_id": uid, "ds": ds, "y": base + float(i)})
    return pd.DataFrame(rows)


def test_backtest_with_test_fold() -> None:
    """End-to-end: backtest with a test fold produces TestResults."""
    df = _synthetic_monthly_panel_long()
    cfg = _minimal_config()
    cfg["test"] = {"test_origin": "2023-07-01"}

    results = run_backtest(config=cfg, df=df)

    assert isinstance(results, BacktestResults)

    # CV results still present and correct
    assert len(results.cv.forecasts_per_fold) == 2

    # Test results populated
    assert results.test is not None
    assert results.test.test_origin == pd.Timestamp("2023-07-01")

    # Test metrics have fold_id="test"
    test_metrics = results.test.metrics
    assert len(test_metrics) > 0
    assert (test_metrics["fold"] == "test").all()
    assert list(test_metrics.columns) == [
        "metric_name",
        "unique_id",
        "fold",
        "aggregation",
        "value",
    ]

    # Test forecasts populated
    assert len(results.test.forecasts) > 0

    # Train/test split stored
    assert set(results.test.train_test_split.keys()) == {"train", "test"}

    # Optional fields are None (core only)
    assert results.test.fitted_values is None
    assert results.test.transform_params is None


# ---- integer ds integration test ----


def _synthetic_integer_panel() -> pd.DataFrame:
    """Two-series panel with integer ds (0-23) and a simple linear trend."""
    rows = []
    for uid, base in [("A", 10.0), ("B", 50.0)]:
        for i in range(24):
            rows.append({"unique_id": uid, "ds": i, "y": base + float(i)})
    return pd.DataFrame(rows)


def _integer_ds_config() -> dict:
    """Config with dummy model, no transforms, RMSE metric, integer ds."""
    return {
        "data": {"freq": 1},
        "cross_validation": {
            "mode": "explicit",
            "horizon": 5,
            "forecast_origins": [10, 15],
        },
        "model": {
            "callable": "tsbricks._testing.dummy_models.forecast_only",
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


def test_minimal_backtest_integer_ds() -> None:
    """Smoke test: integer ds panel + dummy model -> BacktestResults."""
    df = _synthetic_integer_panel()
    cfg = _integer_ds_config()

    results = run_backtest(config=cfg, df=df)

    # Return type
    assert isinstance(results, BacktestResults)

    # Metrics DataFrame is populated and has expected schema
    metrics = results.cv.metrics
    assert len(metrics) > 0
    assert list(metrics.columns) == [
        "metric_name",
        "unique_id",
        "fold",
        "aggregation",
        "value",
    ]

    # Correct fold count (2 origins = 2 folds)
    assert len(results.cv.forecasts_per_fold) == 2
    assert len(results.cv.train_val_splits_per_fold) == 2

    # Fold origins are integers, not timestamps
    assert len(results.cv.fold_origins) == 2
    assert results.cv.fold_origins == [10, 15]

    # Horizon preserved
    assert results.horizon == 5

    # Config preserved
    assert results.config == cfg

    # No test fold when not configured
    assert results.test is None


def test_backtest_integer_ds_with_test_fold() -> None:
    """End-to-end: integer ds backtest with test fold."""
    df = _synthetic_integer_panel()
    cfg = _integer_ds_config()
    cfg["test"] = {"test_origin": 18}

    results = run_backtest(config=cfg, df=df)

    assert isinstance(results, BacktestResults)
    assert results.test is not None
    assert results.test.test_origin == 18

    # Test metrics have fold_id="test"
    assert (results.test.metrics["fold"] == "test").all()

    # CV still works
    assert len(results.cv.forecasts_per_fold) == 2
