"""Tests for cross-validation fold generation."""

from __future__ import annotations

import pandas as pd
import pytest

from tsbricks.backtesting.cross_validation import generate_folds
from tsbricks.backtesting.schema import CrossValidationConfig


def _cv_config(origins: list[str], horizon: int = 6) -> CrossValidationConfig:
    return CrossValidationConfig(
        mode="explicit", horizon=horizon, forecast_origins=origins
    )


# ---- fold count and structure ----


def test_single_origin_produces_one_fold(monthly_panel, data_config):
    """A single forecast origin produces exactly one fold."""
    folds, test_split = generate_folds(
        monthly_panel, _cv_config(["2023-01-01"]), data_config
    )

    assert test_split is None
    assert len(folds) == 1
    assert "fold_0" in folds


def test_two_origins_produce_two_folds(monthly_panel, data_config):
    """Two forecast origins produce two folds."""
    folds, _ = generate_folds(
        monthly_panel, _cv_config(["2023-01-01", "2023-04-01"]), data_config
    )

    assert len(folds) == 2
    assert list(folds.keys()) == ["fold_0", "fold_1"]


def test_each_fold_has_train_and_val_keys(monthly_panel, data_config):
    """Each fold dict contains exactly 'train' and 'val' keys."""
    folds, _ = generate_folds(monthly_panel, _cv_config(["2023-01-01"]), data_config)

    assert set(folds["fold_0"].keys()) == {"train", "val"}


# ---- split boundaries ----


def test_train_includes_origin_date(monthly_panel, data_config):
    """Training set includes the origin date itself (ds <= origin)."""
    origin = "2023-01-01"
    folds, _ = generate_folds(monthly_panel, _cv_config([origin]), data_config)
    train = folds["fold_0"]["train"]

    assert pd.Timestamp(origin) in train["ds"].values


def test_train_excludes_dates_after_origin(monthly_panel, data_config):
    """Training set contains no dates after the origin."""
    origin = "2023-01-01"
    folds, _ = generate_folds(monthly_panel, _cv_config([origin]), data_config)
    train = folds["fold_0"]["train"]

    assert (train["ds"] <= pd.Timestamp(origin)).all()


def test_val_starts_after_origin(monthly_panel, data_config):
    """Validation set contains no dates at or before the origin."""
    origin = "2023-01-01"
    folds, _ = generate_folds(monthly_panel, _cv_config([origin]), data_config)
    val = folds["fold_0"]["val"]

    assert (val["ds"] > pd.Timestamp(origin)).all()


def test_val_ends_at_origin_plus_horizon(monthly_panel, data_config):
    """Validation set extends exactly horizon periods past the origin."""
    origin = "2023-01-01"
    horizon = 6
    folds, _ = generate_folds(
        monthly_panel, _cv_config([origin], horizon=horizon), data_config
    )
    val = folds["fold_0"]["val"]
    expected_end = pd.Timestamp(origin) + horizon * pd.tseries.frequencies.to_offset(
        "MS"
    )

    assert val["ds"].max() == expected_end


def test_val_row_count_per_series(monthly_panel, data_config):
    """Each series in the val set has exactly horizon rows."""
    origin = "2023-01-01"
    horizon = 6
    folds, _ = generate_folds(
        monthly_panel, _cv_config([origin], horizon=horizon), data_config
    )
    val = folds["fold_0"]["val"]

    for uid in ["A", "B"]:
        assert len(val[val["unique_id"] == uid]) == horizon


# ---- expanding window ----


def test_later_fold_has_more_training_data(monthly_panel, data_config):
    """A later origin produces a larger training set (expanding window)."""
    folds, _ = generate_folds(
        monthly_panel,
        _cv_config(["2022-07-01", "2023-01-01"]),
        data_config,
    )

    train_0 = folds["fold_0"]["train"]
    train_1 = folds["fold_1"]["train"]

    assert len(train_1) > len(train_0)


# ---- chronological ordering ----


def test_folds_ordered_chronologically(monthly_panel, data_config):
    """Folds are ordered by origin date regardless of input order."""
    folds, _ = generate_folds(
        monthly_panel,
        _cv_config(["2023-04-01", "2022-07-01", "2023-01-01"]),
        data_config,
    )

    keys = list(folds.keys())
    assert keys == ["fold_0", "fold_1", "fold_2"]

    # fold_0 should have the earliest (smallest) max train date
    max_dates = [folds[k]["train"]["ds"].max() for k in keys]
    assert max_dates == sorted(max_dates)


# ---- fold naming: zero-padding ----


def test_fold_keys_no_padding_under_10(monthly_panel, data_config):
    """With fewer than 10 folds, keys use single-digit indices."""
    origins = [f"2022-{m:02d}-01" for m in range(1, 4)]
    folds, _ = generate_folds(
        monthly_panel, _cv_config(origins, horizon=1), data_config
    )

    assert list(folds.keys()) == ["fold_0", "fold_1", "fold_2"]


def test_fold_keys_zero_padded_for_11_folds(data_config):
    """With 11 folds (indices 0-10), keys are zero-padded to two digits."""
    dates = pd.date_range("2020-01-01", periods=48, freq="MS")
    df = pd.DataFrame({"unique_id": "A", "ds": dates, "y": 1.0})

    origins = [
        str(d.date()) for d in pd.date_range("2020-06-01", periods=11, freq="MS")
    ]
    folds, _ = generate_folds(df, _cv_config(origins, horizon=1), data_config)

    keys = list(folds.keys())
    assert keys[0] == "fold_00"
    assert keys[10] == "fold_10"


# ---- input validation ----


def test_non_datetime_ds_raises(data_config):
    """String ds column raises ValueError with a clear message."""
    df = pd.DataFrame(
        {"unique_id": ["A", "A"], "ds": ["2023-01-01", "2023-02-01"], "y": [1.0, 2.0]}
    )

    with pytest.raises(ValueError, match="datetime dtype"):
        generate_folds(df, _cv_config(["2023-01-01"]), data_config)
