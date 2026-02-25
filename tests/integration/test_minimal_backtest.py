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

    # No test fold in V1
    assert results.test is None
