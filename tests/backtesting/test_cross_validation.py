"""Tests for cross-validation fold generation."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import pandas as pd
import pytest

from tsbricks.backtesting.cross_validation import generate_folds
from tsbricks.backtesting.schema import CrossValidationConfig, TestConfig


def _cv_config(origins: Sequence[str | int], horizon: int = 6) -> CrossValidationConfig:
    return CrossValidationConfig(
        mode="explicit", horizon=horizon, forecast_origins=origins
    )


def _variable_cv_config(
    origin_horizons: list[tuple[str | int, int]],
) -> CrossValidationConfig:
    """Build a variable-horizon CrossValidationConfig."""
    return CrossValidationConfig(
        mode="explicit",
        forecast_origins=[{"origin": o, "horizon": h} for o, h in origin_horizons],
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


def test_string_ds_raises(data_config):
    """String ds column raises ValueError with a clear message."""
    df = pd.DataFrame(
        {"unique_id": ["A", "A"], "ds": ["2023-01-01", "2023-02-01"], "y": [1.0, 2.0]}
    )

    with pytest.raises(ValueError, match="datetime or integer dtype"):
        generate_folds(df, _cv_config(["2023-01-01"]), data_config)


# ---- integer ds ----


def test_integer_ds_single_origin_produces_one_fold(integer_panel, integer_data_config):
    """A single integer origin produces exactly one fold."""
    folds, test_split = generate_folds(
        integer_panel, _cv_config([12], horizon=5), integer_data_config
    )

    assert test_split is None
    assert list(folds.keys()) == ["fold_0"]


def test_integer_ds_train_val_split_boundaries(integer_panel, integer_data_config):
    """Train includes origin, val starts after origin."""
    origin = 12
    folds, _ = generate_folds(
        integer_panel, _cv_config([origin], horizon=5), integer_data_config
    )

    train = folds["fold_0"]["train"]
    val = folds["fold_0"]["val"]

    assert train["ds"].max() == origin
    assert val["ds"].min() == origin + 1


def test_integer_ds_val_ends_at_origin_plus_horizon(integer_panel, integer_data_config):
    """Validation set extends exactly horizon steps past the origin."""
    origin = 10
    horizon = 5
    folds, _ = generate_folds(
        integer_panel, _cv_config([origin], horizon=horizon), integer_data_config
    )
    val = folds["fold_0"]["val"]

    assert val["ds"].max() == origin + horizon


def test_integer_ds_val_row_count_per_series(integer_panel, integer_data_config):
    """Each series in the val set has exactly horizon rows."""
    origin = 10
    horizon = 5
    folds, _ = generate_folds(
        integer_panel, _cv_config([origin], horizon=horizon), integer_data_config
    )
    val = folds["fold_0"]["val"]

    for uid in ["A", "B"]:
        assert len(val[val["unique_id"] == uid]) == horizon


def test_integer_ds_expanding_window(integer_panel, integer_data_config):
    """A later integer origin produces a larger training set."""
    folds, _ = generate_folds(
        integer_panel,
        _cv_config([5, 12], horizon=3),
        integer_data_config,
    )

    train_0 = folds["fold_0"]["train"]
    train_1 = folds["fold_1"]["train"]

    assert len(train_1) > len(train_0)


# ---- dtype / freq cross-validation ----


def test_integer_ds_with_string_freq_raises(integer_panel, data_config):
    """Integer ds column with string freq raises ValueError."""
    with pytest.raises(ValueError, match="Integer ds column requires freq=1"):
        generate_folds(integer_panel, _cv_config([10], horizon=3), data_config)


def test_freq_one_with_datetime_ds_raises(monthly_panel, integer_data_config):
    """freq=1 with datetime ds column raises ValueError."""
    with pytest.raises(ValueError, match="freq=1 requires an integer ds column"):
        generate_folds(monthly_panel, _cv_config([10], horizon=3), integer_data_config)


# ---- test fold: basic structure ----


def _test_config(origin):
    return TestConfig(test_origin=origin)


def test_test_split_returned_when_config_present(monthly_panel, data_config):
    """When test_config is provided, test_split is a dict with train/test keys."""
    _, test_split = generate_folds(
        monthly_panel,
        _cv_config(["2023-01-01"], horizon=3),
        data_config,
        test_config=_test_config("2023-07-01"),
    )

    assert test_split is not None
    assert set(test_split.keys()) == {"train", "test"}


def test_test_split_none_when_config_absent(monthly_panel, data_config):
    """When test_config is None, test_split is None."""
    _, test_split = generate_folds(
        monthly_panel, _cv_config(["2023-01-01"], horizon=3), data_config
    )

    assert test_split is None


# ---- test fold: split boundaries (datetime) ----


def test_test_train_includes_test_origin(monthly_panel, data_config):
    """Test train set includes ds <= test_origin."""
    test_origin = "2023-07-01"
    _, test_split = generate_folds(
        monthly_panel,
        _cv_config(["2023-01-01"], horizon=3),
        data_config,
        test_config=_test_config(test_origin),
    )

    assert test_split is not None
    train = test_split["train"]
    assert pd.Timestamp(test_origin) in train["ds"].values
    assert (train["ds"] <= pd.Timestamp(test_origin)).all()


def test_test_set_starts_after_test_origin(monthly_panel, data_config):
    """Test set contains only dates after test_origin."""
    test_origin = "2023-07-01"
    _, test_split = generate_folds(
        monthly_panel,
        _cv_config(["2023-01-01"], horizon=3),
        data_config,
        test_config=_test_config(test_origin),
    )

    assert test_split is not None
    test = test_split["test"]
    assert (test["ds"] > pd.Timestamp(test_origin)).all()


def test_test_set_ends_at_origin_plus_horizon(monthly_panel, data_config):
    """Test set extends exactly horizon periods past test_origin."""
    test_origin = "2023-07-01"
    horizon = 3
    _, test_split = generate_folds(
        monthly_panel,
        _cv_config(["2023-01-01"], horizon=horizon),
        data_config,
        test_config=_test_config(test_origin),
    )

    assert test_split is not None
    test = test_split["test"]
    expected_end = pd.Timestamp(
        test_origin
    ) + horizon * pd.tseries.frequencies.to_offset("MS")
    assert test["ds"].max() == expected_end


def test_test_set_row_count_per_series(monthly_panel, data_config):
    """Each series in the test set has exactly horizon rows."""
    test_origin = "2023-07-01"
    horizon = 3
    _, test_split = generate_folds(
        monthly_panel,
        _cv_config(["2023-01-01"], horizon=horizon),
        data_config,
        test_config=_test_config(test_origin),
    )

    assert test_split is not None
    test = test_split["test"]
    for uid in ["A", "B"]:
        assert len(test[test["unique_id"] == uid]) == horizon


# ---- test fold: split boundaries (integer) ----


def test_integer_test_split_boundaries(integer_panel, integer_data_config):
    """Integer test split: train includes origin, test starts after."""
    test_origin = 18
    horizon = 3
    _, test_split = generate_folds(
        integer_panel,
        _cv_config([10], horizon=horizon),
        integer_data_config,
        test_config=_test_config(test_origin),
    )

    assert test_split is not None
    train = test_split["train"]
    test = test_split["test"]

    assert train["ds"].max() == test_origin
    assert test["ds"].min() == test_origin + 1
    assert test["ds"].max() == test_origin + horizon


def test_integer_test_set_row_count_per_series(integer_panel, integer_data_config):
    """Each series in the integer test set has exactly horizon rows."""
    test_origin = 18
    horizon = 3
    _, test_split = generate_folds(
        integer_panel,
        _cv_config([10], horizon=horizon),
        integer_data_config,
        test_config=_test_config(test_origin),
    )

    assert test_split is not None
    test = test_split["test"]
    for uid in ["A", "B"]:
        assert len(test[test["unique_id"] == uid]) == horizon


# ---- test fold: runtime validation ----


def test_test_window_exceeds_data_raises(monthly_panel, data_config):
    """Raise ValueError when test_origin + horizon exceeds available data."""
    with pytest.raises(ValueError, match="Test window exceeds available data"):
        generate_folds(
            monthly_panel,
            _cv_config(["2023-01-01"], horizon=6),
            data_config,
            test_config=_test_config("2023-10-01"),
        )


def test_integer_test_window_exceeds_data_raises(integer_panel, integer_data_config):
    """Raise ValueError when integer test_origin + horizon exceeds data."""
    with pytest.raises(ValueError, match="Test window exceeds available data"):
        generate_folds(
            integer_panel,
            _cv_config([10], horizon=10),
            integer_data_config,
            test_config=_test_config(20),
        )


# ---- test fold: overlap warning ----


def test_overlap_warning_emitted(monthly_panel, data_config):
    """Warning emitted when test_origin falls within last CV validation window."""
    with pytest.warns(UserWarning, match="Test fold overlaps with cross-validation"):
        generate_folds(
            monthly_panel,
            _cv_config(["2023-01-01"], horizon=6),
            data_config,
            test_config=_test_config("2023-04-01"),
        )


def test_no_overlap_warning_when_after_cv(monthly_panel, data_config):
    """No warning when test_origin is after the last CV validation window."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=UserWarning)
        generate_folds(
            monthly_panel,
            _cv_config(["2023-01-01"], horizon=3),
            data_config,
            test_config=_test_config("2023-07-01"),
        )


def test_integer_overlap_warning_emitted(integer_panel, integer_data_config):
    """Warning emitted for integer ds when test_origin overlaps last CV window."""
    with pytest.warns(UserWarning, match="Test fold overlaps with cross-validation"):
        generate_folds(
            integer_panel,
            _cv_config([10], horizon=5),
            integer_data_config,
            test_config=_test_config(13),
        )


# ---- test fold: cv folds unaffected ----


def test_cv_folds_unchanged_with_test_config(monthly_panel, data_config):
    """CV folds are identical whether or not test_config is provided."""
    cv_config = _cv_config(["2023-01-01", "2023-04-01"], horizon=3)

    folds_without, _ = generate_folds(monthly_panel, cv_config, data_config)
    folds_with, _ = generate_folds(
        monthly_panel, cv_config, data_config, test_config=_test_config("2023-07-01")
    )

    assert list(folds_without.keys()) == list(folds_with.keys())
    for key in folds_without:
        pd.testing.assert_frame_equal(
            folds_without[key]["train"], folds_with[key]["train"]
        )
        pd.testing.assert_frame_equal(folds_without[key]["val"], folds_with[key]["val"])


# ---- variable horizons: CV fold generation ----


def test_variable_horizon_val_sizes(monthly_panel, data_config):
    """Each fold's val set has the correct per-origin horizon length."""
    cv_config = _variable_cv_config(
        [
            ("2022-07-01", 3),  # 3 months val
            ("2023-01-01", 6),  # 6 months val
        ]
    )
    folds, _ = generate_folds(monthly_panel, cv_config, data_config)

    assert len(folds) == 2
    # fold_0 is earlier origin (2022-07-01) with horizon 3
    for uid in ["A", "B"]:
        val_0 = folds["fold_0"]["val"]
        assert len(val_0[val_0["unique_id"] == uid]) == 3
    # fold_1 is later origin (2023-01-01) with horizon 6
    for uid in ["A", "B"]:
        val_1 = folds["fold_1"]["val"]
        assert len(val_1[val_1["unique_id"] == uid]) == 6


def test_variable_horizon_val_end_dates(monthly_panel, data_config):
    """Each fold's val set ends at origin + horizon periods."""
    cv_config = _variable_cv_config(
        [
            ("2022-07-01", 3),
            ("2023-01-01", 6),
        ]
    )
    folds, _ = generate_folds(monthly_panel, cv_config, data_config)
    ms = pd.tseries.frequencies.to_offset("MS")

    val_0 = folds["fold_0"]["val"]
    assert val_0["ds"].max() == pd.Timestamp("2022-07-01") + 3 * ms

    val_1 = folds["fold_1"]["val"]
    assert val_1["ds"].max() == pd.Timestamp("2023-01-01") + 6 * ms


def test_variable_horizon_integer_ds(integer_panel, integer_data_config):
    """Variable horizons work with integer ds."""
    cv_config = _variable_cv_config(
        [
            (5, 3),
            (12, 5),
        ]
    )
    folds, _ = generate_folds(integer_panel, cv_config, integer_data_config)

    assert len(folds) == 2
    # fold_0 (origin=5, horizon=3): val ds in [6, 7, 8]
    val_0 = folds["fold_0"]["val"]
    assert val_0["ds"].min() == 6
    assert val_0["ds"].max() == 8
    for uid in ["A", "B"]:
        assert len(val_0[val_0["unique_id"] == uid]) == 3
    # fold_1 (origin=12, horizon=5): val ds in [13..17]
    val_1 = folds["fold_1"]["val"]
    assert val_1["ds"].min() == 13
    assert val_1["ds"].max() == 17
    for uid in ["A", "B"]:
        assert len(val_1[val_1["unique_id"] == uid]) == 5


def test_variable_horizon_chronological_order(monthly_panel, data_config):
    """Folds are ordered chronologically regardless of input order."""
    cv_config = _variable_cv_config(
        [
            ("2023-01-01", 6),
            ("2022-07-01", 3),
        ]
    )
    folds, _ = generate_folds(monthly_panel, cv_config, data_config)

    keys = list(folds.keys())
    assert keys == ["fold_0", "fold_1"]
    # fold_0 should have the earlier origin
    assert folds["fold_0"]["train"]["ds"].max() == pd.Timestamp("2022-07-01")
    assert folds["fold_1"]["train"]["ds"].max() == pd.Timestamp("2023-01-01")


# ---- variable horizons: horizon-exceeds-data validation ----


def test_variable_horizon_exceeds_data_raises(
    monthly_panel,
    data_config,
):
    """CV origin with horizon past available data raises ValueError."""
    # monthly_panel ends at 2023-12-01; origin 2023-07-01 + 12 months
    # requires data through 2024-07-01
    cv_config = _variable_cv_config(
        [
            ("2022-07-01", 3),
            ("2023-07-01", 12),
        ]
    )
    with pytest.raises(ValueError, match="CV fold horizon exceeds available data"):
        generate_folds(monthly_panel, cv_config, data_config)


def test_variable_horizon_integer_exceeds_data_raises(
    integer_panel,
    integer_data_config,
):
    """Integer CV origin with horizon past data raises ValueError."""
    # integer_panel ds 0..23; origin 20 + horizon 10 = 30 > 23
    cv_config = _variable_cv_config(
        [
            (5, 3),
            (20, 10),
        ]
    )
    with pytest.raises(ValueError, match="CV fold horizon exceeds available data"):
        generate_folds(integer_panel, cv_config, integer_data_config)


def test_uniform_horizon_exceeds_data_raises(
    monthly_panel,
    data_config,
):
    """Uniform horizon past available data also raises ValueError."""
    # monthly_panel ends at 2023-12-01; origin 2023-07-01 + 12
    cv_config = _cv_config(["2023-07-01"], horizon=12)
    with pytest.raises(ValueError, match="CV fold horizon exceeds available data"):
        generate_folds(monthly_panel, cv_config, data_config)


# ---- test fold: resolved test horizon ----


def test_test_fold_inherits_uniform_horizon(monthly_panel, data_config):
    """Test fold uses cv_config.horizon when test.horizon is None."""
    cv_config = _cv_config(["2023-01-01"], horizon=3)
    tc = TestConfig(test_origin="2023-07-01")

    _, test_split = generate_folds(
        monthly_panel, cv_config, data_config, test_config=tc
    )

    assert test_split is not None
    test_df = test_split["test"]
    ms = pd.tseries.frequencies.to_offset("MS")
    expected_end = pd.Timestamp("2023-07-01") + 3 * ms
    assert test_df["ds"].max() == expected_end
    for uid in ["A", "B"]:
        assert len(test_df[test_df["unique_id"] == uid]) == 3


def test_test_fold_explicit_horizon_override(monthly_panel, data_config):
    """Test fold uses test.horizon when explicitly provided."""
    cv_config = _cv_config(["2023-01-01"], horizon=3)
    tc = TestConfig(test_origin="2023-07-01", horizon=5)

    _, test_split = generate_folds(
        monthly_panel, cv_config, data_config, test_config=tc
    )

    assert test_split is not None
    test_df = test_split["test"]
    ms = pd.tseries.frequencies.to_offset("MS")
    expected_end = pd.Timestamp("2023-07-01") + 5 * ms
    assert test_df["ds"].max() == expected_end
    for uid in ["A", "B"]:
        assert len(test_df[test_df["unique_id"] == uid]) == 5


def test_test_fold_with_variable_cv_horizons(data_config):
    """Test fold works with variable CV horizons + explicit test.horizon."""
    # Need longer panel: 2022-01 to 2024-12
    dates = pd.date_range("2022-01-01", periods=36, freq="MS")
    rows = []
    for uid in ["A", "B"]:
        for ds in dates:
            rows.append({"unique_id": uid, "ds": ds, "y": 1.0})
    df = pd.DataFrame(rows)

    cv_config = _variable_cv_config(
        [
            ("2023-01-01", 6),
            ("2023-07-01", 3),
        ]
    )
    tc = TestConfig(test_origin="2024-01-01", horizon=4)

    folds, test_split = generate_folds(df, cv_config, data_config, test_config=tc)

    # CV folds have per-origin horizons
    assert len(folds) == 2
    for uid in ["A", "B"]:
        val_0 = folds["fold_0"]["val"]
        assert len(val_0[val_0["unique_id"] == uid]) == 6
        val_1 = folds["fold_1"]["val"]
        assert len(val_1[val_1["unique_id"] == uid]) == 3

    # Test fold uses its own horizon of 4
    assert test_split is not None
    test_df = test_split["test"]
    for uid in ["A", "B"]:
        assert len(test_df[test_df["unique_id"] == uid]) == 4


def test_test_fold_explicit_horizon_exceeds_data_raises(
    monthly_panel,
    data_config,
):
    """Test fold with explicit horizon past data raises ValueError."""
    cv_config = _cv_config(["2023-01-01"], horizon=3)
    # monthly_panel ends 2023-12-01; 2023-07-01 + 12 = 2024-07-01
    tc = TestConfig(test_origin="2023-07-01", horizon=12)

    with pytest.raises(ValueError, match="Test window exceeds available data"):
        generate_folds(monthly_panel, cv_config, data_config, test_config=tc)


def test_test_fold_integer_explicit_horizon(
    integer_panel,
    integer_data_config,
):
    """Integer test fold uses explicit test.horizon."""
    cv_config = _cv_config([10], horizon=5)
    tc = TestConfig(test_origin=15, horizon=3)

    _, test_split = generate_folds(
        integer_panel,
        cv_config,
        integer_data_config,
        test_config=tc,
    )

    assert test_split is not None
    test_df = test_split["test"]
    assert test_df["ds"].min() == 16
    assert test_df["ds"].max() == 18
    for uid in ["A", "B"]:
        assert len(test_df[test_df["unique_id"] == uid]) == 3


def test_overlap_warning_uses_last_cv_origin_horizon(
    monthly_panel,
    data_config,
):
    """Overlap warning uses the last CV origin's actual horizon."""
    # Last CV origin 2023-01-01 with horizon=3 -> val_end=2023-04-01
    # test_origin 2023-03-01 < 2023-04-01 -> overlap warning
    cv_config = _variable_cv_config(
        [
            ("2022-07-01", 6),
            ("2023-01-01", 3),
        ]
    )
    tc = TestConfig(test_origin="2023-03-01", horizon=3)

    with pytest.warns(
        UserWarning,
        match="Test fold overlaps with cross-validation",
    ):
        generate_folds(monthly_panel, cv_config, data_config, test_config=tc)
