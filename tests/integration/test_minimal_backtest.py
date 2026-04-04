"""Integration test: end-to-end backtest with dummy model."""

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
        "scope",
        "grouping_column_name",
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

    # Horizon dict: origin -> horizon for all folds
    assert results.horizon == {
        pd.Timestamp("2023-01-01"): 6,
        pd.Timestamp("2023-06-01"): 6,
    }

    # Config preserved
    assert results.config == cfg

    # No test fold when not configured
    assert results.test is None

    # Metadata populated
    assert results.git_hash is not None
    assert len(results.git_hash) == 40
    assert all(c in "0123456789abcdef" for c in results.git_hash)

    # uv_lock_info is a dict with expected keys if uv.lock exists, else None
    if results.uv_lock_info is not None:
        assert set(results.uv_lock_info.keys()) == {"path", "sha256"}

    # run_summary always populated with empty lists when no issues
    assert results.run_summary == {"warnings": [], "errors": []}


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
        "scope",
        "grouping_column_name",
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


def test_test_fold_forecast_differs_from_last_cv_fold() -> None:
    """Test fold forecast is derived from the test split, not the last CV fold.

    The dummy model forecasts last_y per series. Since the
    synthetic panel has a linear trend, different training
    cutoffs produce different ypred values. This test verifies
    the test fold forecast uses its own training data.
    """
    df = _synthetic_monthly_panel_long()
    cfg = _minimal_config()
    cfg["test"] = {"test_origin": "2023-07-01"}

    results = run_backtest(config=cfg, df=df)

    assert results.test is not None

    last_cv_fold = list(results.cv.forecasts_per_fold.keys())[-1]
    last_cv_forecast = results.cv.forecasts_per_fold[last_cv_fold]
    test_forecast = results.test.forecasts

    # The test fold trains on more data, so ypred values must
    # differ from the last CV fold for at least one series.
    for uid in ["A", "B"]:
        cv_ypred = last_cv_forecast.loc[
            last_cv_forecast["unique_id"] == uid, "ypred"
        ].iloc[0]
        test_ypred = test_forecast.loc[test_forecast["unique_id"] == uid, "ypred"].iloc[
            0
        ]
        assert cv_ypred != test_ypred, (
            f"Series {uid}: test fold ypred should differ "
            f"from last CV fold (both={cv_ypred})"
        )


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
        "scope",
        "grouping_column_name",
        "aggregation",
        "value",
    ]

    # Correct fold count (2 origins = 2 folds)
    assert len(results.cv.forecasts_per_fold) == 2
    assert len(results.cv.train_val_splits_per_fold) == 2

    # Fold origins are integers, not timestamps
    assert len(results.cv.fold_origins) == 2
    assert results.cv.fold_origins == [10, 15]

    # Horizon dict: origin -> horizon for all folds
    assert results.horizon == {10: 5, 15: 5}

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


def _monthly_weights_df() -> pd.DataFrame:
    """Weights DataFrame for the two-series monthly panel with 2 origins."""
    return pd.DataFrame(
        {
            "unique_id": ["A", "B"] * 2,
            "forecast_origin": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-06-01"),
                pd.Timestamp("2023-06-01"),
            ],
            "raw_weight": [1.0, 2.0, 1.0, 2.0],
        }
    )


# ---- run_backtest accepts grouping_df and weights_df ----


def test_run_backtest_accepts_grouping_df() -> None:
    """run_backtest() accepts a grouping_df DataFrame without error."""
    df = _synthetic_monthly_panel()
    cfg = _minimal_config()
    grouping_df = pd.DataFrame({"unique_id": ["A", "B"], "category": ["cat1", "cat2"]})

    results = run_backtest(config=cfg, df=df, grouping_df=grouping_df)

    assert isinstance(results, BacktestResults)
    assert len(results.cv.forecasts_per_fold) == 2


def test_run_backtest_accepts_weights_df() -> None:
    """run_backtest() accepts a weights_df DataFrame without error."""
    df = _synthetic_monthly_panel()
    cfg = _minimal_config()

    results = run_backtest(config=cfg, df=df, weights_df=_monthly_weights_df())

    assert isinstance(results, BacktestResults)
    assert len(results.cv.forecasts_per_fold) == 2


def test_run_backtest_accepts_grouping_df_path(tmp_path) -> None:
    """run_backtest() accepts a grouping_df file path string."""
    df = _synthetic_monthly_panel()
    cfg = _minimal_config()
    grouping_path = tmp_path / "grouping.parquet"
    grouping = pd.DataFrame({"unique_id": ["A", "B"], "category": ["cat1", "cat2"]})
    grouping.to_parquet(grouping_path)

    results = run_backtest(config=cfg, df=df, grouping_df=str(grouping_path))

    assert isinstance(results, BacktestResults)


# ---- grouping_df validation ----


def _group_scope_config() -> dict:
    """Config with a group-scope metric requiring grouping_df."""
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
                    "name": "rmse_group",
                    "callable": "tsbricks.blocks.metrics.rmse",
                    "type": "simple",
                    "scope": "group",
                    "grouping_columns": ["category"],
                }
            ],
        },
    }


def test_run_backtest_missing_grouping_df_with_group_scope_raises() -> None:
    """Group-scope metric without grouping_df raises ValueError."""
    df = _synthetic_monthly_panel()
    cfg = _group_scope_config()

    with pytest.raises(ValueError, match="grouping_df is required"):
        run_backtest(config=cfg, df=df)


def test_run_backtest_grouping_df_missing_unique_id_raises() -> None:
    """grouping_df without unique_id column raises ValueError."""
    df = _synthetic_monthly_panel()
    cfg = _minimal_config()
    grouping_df = pd.DataFrame({"series": ["A", "B"], "category": ["c1", "c2"]})

    with pytest.raises(ValueError, match="unique_id"):
        run_backtest(config=cfg, df=df, grouping_df=grouping_df)


def test_run_backtest_grouping_df_missing_grouping_column_raises() -> None:
    """grouping_df missing a referenced grouping column raises ValueError."""
    df = _synthetic_monthly_panel()
    cfg = _group_scope_config()
    # Has unique_id but not the 'category' column the metric references
    grouping_df = pd.DataFrame({"unique_id": ["A", "B"], "region": ["east", "west"]})

    with pytest.raises(ValueError, match="missing required grouping columns"):
        run_backtest(config=cfg, df=df, grouping_df=grouping_df)


# ---- weights_df validation ----


def _global_scope_config() -> dict:
    """Config with a global-scope metric requiring weights_df."""
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
                    "name": "wape_global",
                    "callable": "tsbricks.blocks.metrics.rmse",
                    "type": "simple",
                    "scope": "global",
                    "aggregation_callable": "my.agg.weighted_mean",
                }
            ],
        },
    }


def test_run_backtest_accepts_weights_df_path(tmp_path) -> None:
    """run_backtest() accepts a weights_df file path string."""
    df = _synthetic_monthly_panel()
    cfg = _minimal_config()
    weights_path = tmp_path / "weights.parquet"
    _monthly_weights_df().to_parquet(weights_path)

    results = run_backtest(config=cfg, df=df, weights_df=str(weights_path))

    assert isinstance(results, BacktestResults)


def test_run_backtest_missing_weights_df_with_aggregation_callable_raises() -> None:
    """Global metric with aggregation_callable but no weights_df raises."""
    df = _synthetic_monthly_panel()
    cfg = _global_scope_config()

    with pytest.raises(ValueError, match="weights_df is required"):
        run_backtest(config=cfg, df=df)


def test_run_backtest_weights_df_missing_columns_raises() -> None:
    """weights_df missing required columns raises ValueError."""
    df = _synthetic_monthly_panel()
    cfg = _minimal_config()
    # Missing 'raw_weight' column
    weights_df = pd.DataFrame(
        {
            "unique_id": ["A", "B"],
            "forecast_origin": ["2023-01-01", "2023-01-01"],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        run_backtest(config=cfg, df=df, weights_df=weights_df)


def test_run_backtest_weights_df_missing_origin_coverage_raises() -> None:
    """weights_df not covering all forecast origins raises ValueError."""
    df = _synthetic_monthly_panel()
    cfg = _minimal_config()
    # Only covers first origin, missing 2023-06-01
    all_weights = _monthly_weights_df()
    weights_df = all_weights[
        all_weights["forecast_origin"] == pd.Timestamp("2023-01-01")
    ].copy()

    with pytest.raises(ValueError, match="missing rows for forecast origins"):
        run_backtest(config=cfg, df=df, weights_df=weights_df)


# ---- config-driven source loading and fallback ----


def test_run_backtest_loads_grouping_df_from_config_source(tmp_path) -> None:
    """grouping_df loaded from metrics.grouping_source when not passed directly."""
    df = _synthetic_monthly_panel()
    cfg = _minimal_config()
    grouping_path = tmp_path / "grouping.parquet"
    pd.DataFrame({"unique_id": ["A", "B"], "category": ["cat1", "cat2"]}).to_parquet(
        grouping_path
    )
    cfg["metrics"]["grouping_source"] = str(grouping_path)

    results = run_backtest(config=cfg, df=df)

    assert isinstance(results, BacktestResults)
    assert len(results.cv.forecasts_per_fold) == 2


def test_run_backtest_loads_weights_df_from_config_source(tmp_path) -> None:
    """weights_df loaded from metrics.weights_source when not passed directly."""
    df = _synthetic_monthly_panel()
    cfg = _minimal_config()
    weights_path = tmp_path / "weights.parquet"
    _monthly_weights_df().to_parquet(weights_path)
    cfg["metrics"]["weights_source"] = str(weights_path)

    results = run_backtest(config=cfg, df=df)

    assert isinstance(results, BacktestResults)
    assert len(results.cv.forecasts_per_fold) == 2


def test_run_backtest_grouping_columns_fallback_to_top_level() -> None:
    """Metric with no grouping_columns uses top-level metrics.grouping_columns."""
    df = _synthetic_monthly_panel()
    cfg = _minimal_config()
    # Group-scope metric with NO definition-level grouping_columns;
    # top-level metrics.grouping_columns provides the fallback.
    cfg["metrics"]["definitions"].append(
        {
            "name": "rmse_group",
            "callable": "tsbricks.blocks.metrics.rmse",
            "type": "simple",
            "scope": "group",
        }
    )
    cfg["metrics"]["grouping_columns"] = ["category"]
    grouping_df = pd.DataFrame({"unique_id": ["A", "B"], "category": ["cat1", "cat2"]})

    # Validation should pass — grouping_df has the fallback column "category"
    results = run_backtest(config=cfg, df=df, grouping_df=grouping_df)

    assert isinstance(results, BacktestResults)


# ---- variable horizon integration tests ----


def _variable_horizon_config() -> dict:
    """Config with variable per-origin horizons."""
    return {
        "data": {"freq": "MS"},
        "cross_validation": {
            "mode": "explicit",
            "forecast_origins": [
                {"origin": "2023-01-01", "horizon": 3},
                {"origin": "2023-06-01", "horizon": 6},
            ],
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


def test_variable_horizon_end_to_end() -> None:
    """End-to-end backtest with variable per-origin horizons."""
    df = _synthetic_monthly_panel()
    cfg = _variable_horizon_config()

    results = run_backtest(config=cfg, df=df)

    assert isinstance(results, BacktestResults)
    assert len(results.cv.forecasts_per_fold) == 2

    # Horizon dict has per-origin values
    assert results.horizon == {
        pd.Timestamp("2023-01-01"): 3,
        pd.Timestamp("2023-06-01"): 6,
    }

    assert results.test is None


def test_variable_horizon_with_test_fold() -> None:
    """Variable horizons + test fold with explicit test.horizon."""
    df = _synthetic_monthly_panel_long()
    cfg = _variable_horizon_config()
    cfg["test"] = {
        "test_origin": "2023-07-01",
        "horizon": 4,
    }

    results = run_backtest(config=cfg, df=df)

    assert isinstance(results, BacktestResults)
    assert len(results.cv.forecasts_per_fold) == 2

    # Horizon dict includes CV origins + test origin
    assert results.horizon == {
        pd.Timestamp("2023-01-01"): 3,
        pd.Timestamp("2023-06-01"): 6,
        pd.Timestamp("2023-07-01"): 4,
    }

    assert results.test is not None
    assert results.test.test_origin == pd.Timestamp("2023-07-01")


def test_uniform_horizon_with_test_override() -> None:
    """Uniform CV horizon + test fold with different horizon."""
    df = _synthetic_monthly_panel_long()
    cfg = _minimal_config()
    cfg["test"] = {
        "test_origin": "2023-07-01",
        "horizon": 3,
    }

    results = run_backtest(config=cfg, df=df)

    assert isinstance(results, BacktestResults)

    # CV origins get horizon 6, test origin gets 3
    assert results.horizon == {
        pd.Timestamp("2023-01-01"): 6,
        pd.Timestamp("2023-06-01"): 6,
        pd.Timestamp("2023-07-01"): 3,
    }

    assert results.test is not None
