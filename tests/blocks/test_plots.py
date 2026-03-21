"""Tests for tsbricks.blocks.plots — seasonal validation and computation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tsbricks.blocks.plots import (
    _assign_calendar_seasons,
    _assign_frequency_seasons,
    _assign_positional_seasons,
    _check_data_sufficiency,
    _compute_seasonal_data,
    _is_datetime_time_col,
    _resolve_base_freq,
    _validate_seasonal_inputs,
    _NAMED_PERIODS,
    _SUPPORTED_BASE_FREQS,
)

# =====================================================================
# Shared defaults for _validate_seasonal_inputs
# =====================================================================

_DEFAULTS = dict(
    backend="plotly",
    width=800,
    height=450,
    alpha=0.8,
    palette="magma",
    base_freq=None,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def seasonal_df_datetime():
    """Monthly datetime DataFrame spanning 3 years."""
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    return pd.DataFrame(
        {"time": dates, "value": np.random.default_rng(42).normal(size=36)}
    )


@pytest.fixture
def seasonal_df_integer():
    """Integer-indexed DataFrame with 20 observations."""
    return pd.DataFrame(
        {"time": range(20), "value": np.random.default_rng(42).normal(size=20)}
    )


@pytest.fixture
def seasonal_df_daily():
    """Daily datetime DataFrame spanning 2 years."""
    dates = pd.date_range("2020-01-01", periods=730, freq="D")
    return pd.DataFrame(
        {"time": dates, "value": np.random.default_rng(42).normal(size=730)}
    )


# =====================================================================
# _is_datetime_time_col
# =====================================================================


def test_is_datetime_time_col_datetime64(seasonal_df_datetime):
    """Returns True for datetime64 column."""
    assert _is_datetime_time_col(seasonal_df_datetime, "time") is True


def test_is_datetime_time_col_integer(seasonal_df_integer):
    """Returns False for integer column."""
    assert _is_datetime_time_col(seasonal_df_integer, "time") is False


def test_is_datetime_time_col_object_datetime():
    """Returns True for object-dtype column containing Python datetimes."""
    from datetime import datetime

    df = pd.DataFrame(
        {
            "time": [datetime(2020, 1, 1), datetime(2020, 2, 1), datetime(2020, 3, 1)],
            "value": [1.0, 2.0, 3.0],
        }
    )
    assert _is_datetime_time_col(df, "time") is True


def test_is_datetime_time_col_tz_aware():
    """Returns True for timezone-aware datetime column."""
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-03-01"]
            ).tz_localize("US/Eastern"),
            "value": [1.0, 2.0, 3.0],
        }
    )
    assert _is_datetime_time_col(df, "time") is True


# =====================================================================
# _validate_seasonal_inputs — happy paths
# =====================================================================


def test_validate_seasonal_happy_path_datetime_named_period(seasonal_df_datetime):
    """Passes for datetime time_col with named period."""
    _validate_seasonal_inputs(
        seasonal_df_datetime, "time", "value", "year", **_DEFAULTS
    )


def test_validate_seasonal_happy_path_datetime_all_named_periods(seasonal_df_datetime):
    """Passes for all canonical named periods."""
    for period in _NAMED_PERIODS:
        _validate_seasonal_inputs(
            seasonal_df_datetime, "time", "value", period, **_DEFAULTS
        )


def test_validate_seasonal_happy_path_integer(seasonal_df_integer):
    """Passes for integer time_col with integer period."""
    _validate_seasonal_inputs(seasonal_df_integer, "time", "value", 5, **_DEFAULTS)


def test_validate_seasonal_happy_path_datetime_int_period_with_base_freq(
    seasonal_df_datetime,
):
    """Passes for datetime time_col + integer period + explicit base_freq."""
    _validate_seasonal_inputs(
        seasonal_df_datetime,
        "time",
        "value",
        12,
        backend="plotly",
        width=800,
        height=450,
        alpha=0.8,
        palette="magma",
        base_freq="MS",
    )


def test_validate_seasonal_happy_path_polars():
    """Accepts Polars DataFrame."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"time": list(range(20)), "value": [float(i) for i in range(20)]})
    _validate_seasonal_inputs(df, "time", "value", 5, **_DEFAULTS)


def test_validate_seasonal_happy_path_matplotlib(seasonal_df_datetime):
    """Passes for matplotlib backend."""
    _validate_seasonal_inputs(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        backend="matplotlib",
        width=800,
        height=450,
        alpha=0.8,
        palette="magma",
        base_freq=None,
    )


# =====================================================================
# _validate_seasonal_inputs — DataFrame checks
# =====================================================================


def test_validate_seasonal_rejects_non_dataframe():
    """Raises TypeError for non-DataFrame input."""
    with pytest.raises(TypeError, match="pandas or Polars DataFrame"):
        _validate_seasonal_inputs(
            {"time": [1, 2], "value": [1.0, 2.0]}, "time", "value", 2, **_DEFAULTS
        )


def test_validate_seasonal_empty_df():
    """Raises ValueError for empty DataFrame."""
    df = pd.DataFrame(
        {"time": pd.Series([], dtype="int64"), "value": pd.Series([], dtype="float64")}
    )
    with pytest.raises(ValueError, match="must not be empty"):
        _validate_seasonal_inputs(df, "time", "value", 2, **_DEFAULTS)


# =====================================================================
# _validate_seasonal_inputs — column checks
# =====================================================================


def test_validate_seasonal_missing_time_col(seasonal_df_integer):
    """Raises ValueError when time_col not in DataFrame."""
    with pytest.raises(ValueError, match="time_col 'missing'"):
        _validate_seasonal_inputs(
            seasonal_df_integer, "missing", "value", 5, **_DEFAULTS
        )


def test_validate_seasonal_missing_value_col(seasonal_df_integer):
    """Raises ValueError when value_col not in DataFrame."""
    with pytest.raises(ValueError, match="value_col 'missing'"):
        _validate_seasonal_inputs(
            seasonal_df_integer, "time", "missing", 5, **_DEFAULTS
        )


def test_validate_seasonal_time_col_float_dtype():
    """Raises ValueError when time_col is float."""
    df = pd.DataFrame({"time": [1.0, 2.0, 3.0], "value": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="datetime-like or integer dtype"):
        _validate_seasonal_inputs(df, "time", "value", 2, **_DEFAULTS)


def test_validate_seasonal_value_col_non_numeric():
    """Raises ValueError when value_col is non-numeric."""
    df = pd.DataFrame({"time": [1, 2, 3], "value": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="must be numeric"):
        _validate_seasonal_inputs(df, "time", "value", 2, **_DEFAULTS)


def test_validate_seasonal_time_col_missing_values():
    """Raises ValueError when time_col has NaN."""
    df = pd.DataFrame(
        {
            "time": pd.array([1, pd.NA, 3], dtype="Int64"),
            "value": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="missing value"):
        _validate_seasonal_inputs(df, "time", "value", 2, **_DEFAULTS)


def test_validate_seasonal_duplicate_time():
    """Raises ValueError when time_col has duplicates."""
    df = pd.DataFrame({"time": [1, 2, 2, 3], "value": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="duplicate"):
        _validate_seasonal_inputs(df, "time", "value", 2, **_DEFAULTS)


# =====================================================================
# _validate_seasonal_inputs — period checks
# =====================================================================


def test_validate_seasonal_named_period_with_integer_time_col(seasonal_df_integer):
    """Raises ValueError for named period with integer time_col."""
    with pytest.raises(ValueError, match="Named period.*only allowed for datetime"):
        _validate_seasonal_inputs(
            seasonal_df_integer, "time", "value", "year", **_DEFAULTS
        )


def test_validate_seasonal_invalid_named_period(seasonal_df_datetime):
    """Raises ValueError for unrecognized named period."""
    with pytest.raises(ValueError, match="Invalid named period"):
        _validate_seasonal_inputs(
            seasonal_df_datetime, "time", "value", "decade", **_DEFAULTS
        )


def test_validate_seasonal_integer_period_too_small(seasonal_df_integer):
    """Raises ValueError for integer period < 2."""
    with pytest.raises(ValueError, match="must be >= 2"):
        _validate_seasonal_inputs(seasonal_df_integer, "time", "value", 1, **_DEFAULTS)


def test_validate_seasonal_integer_period_zero(seasonal_df_integer):
    """Raises ValueError for integer period of 0."""
    with pytest.raises(ValueError, match="must be >= 2"):
        _validate_seasonal_inputs(seasonal_df_integer, "time", "value", 0, **_DEFAULTS)


def test_validate_seasonal_period_float_rejected(seasonal_df_integer):
    """Raises TypeError when period is a float."""
    with pytest.raises(TypeError, match="period must be a string or integer"):
        _validate_seasonal_inputs(
            seasonal_df_integer, "time", "value", 5.0, **_DEFAULTS
        )


def test_validate_seasonal_period_bool_rejected(seasonal_df_integer):
    """Raises TypeError when period is a bool."""
    with pytest.raises(TypeError, match="period must be a string or integer"):
        _validate_seasonal_inputs(
            seasonal_df_integer, "time", "value", True, **_DEFAULTS
        )


def test_validate_seasonal_period_none_rejected(seasonal_df_integer):
    """Raises TypeError when period is None."""
    with pytest.raises(TypeError, match="period must be a string or integer"):
        _validate_seasonal_inputs(
            seasonal_df_integer, "time", "value", None, **_DEFAULTS
        )


def test_validate_seasonal_numpy_integer_period(seasonal_df_integer):
    """Accepts numpy integer for period."""
    _validate_seasonal_inputs(
        seasonal_df_integer, "time", "value", np.int64(5), **_DEFAULTS
    )


# =====================================================================
# _validate_seasonal_inputs — base_freq checks
# =====================================================================


def test_validate_seasonal_base_freq_with_integer_time_col(seasonal_df_integer):
    """Raises ValueError when base_freq provided for integer time_col."""
    with pytest.raises(
        ValueError, match="base_freq is only valid when time_col is datetime"
    ):
        _validate_seasonal_inputs(
            seasonal_df_integer,
            "time",
            "value",
            5,
            backend="plotly",
            width=800,
            height=450,
            alpha=0.8,
            palette="magma",
            base_freq="D",
        )


def test_validate_seasonal_base_freq_with_named_period(seasonal_df_datetime):
    """Raises ValueError when base_freq provided with named period."""
    with pytest.raises(
        ValueError, match="base_freq is only valid when period is an integer"
    ):
        _validate_seasonal_inputs(
            seasonal_df_datetime,
            "time",
            "value",
            "year",
            backend="plotly",
            width=800,
            height=450,
            alpha=0.8,
            palette="magma",
            base_freq="MS",
        )


def test_validate_seasonal_unsupported_base_freq(seasonal_df_datetime):
    """Raises ValueError for unsupported base_freq value."""
    with pytest.raises(ValueError, match="Unsupported base_freq"):
        _validate_seasonal_inputs(
            seasonal_df_datetime,
            "time",
            "value",
            12,
            backend="plotly",
            width=800,
            height=450,
            alpha=0.8,
            palette="magma",
            base_freq="min",
        )


def test_validate_seasonal_all_supported_base_freqs(seasonal_df_datetime):
    """All 18 supported base_freq values are accepted."""
    for freq in _SUPPORTED_BASE_FREQS:
        _validate_seasonal_inputs(
            seasonal_df_datetime,
            "time",
            "value",
            12,
            backend="plotly",
            width=800,
            height=450,
            alpha=0.8,
            palette="magma",
            base_freq=freq,
        )


def test_validate_seasonal_base_freq_inferred(seasonal_df_datetime):
    """Passes when base_freq is None and frequency can be inferred."""
    # seasonal_df_datetime has freq="MS" which is in the supported set
    _validate_seasonal_inputs(seasonal_df_datetime, "time", "value", 12, **_DEFAULTS)


def test_validate_seasonal_base_freq_inference_fails():
    """Raises ValueError when inference fails for irregular data."""
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2020-01-01", "2020-01-05", "2020-02-15"]),
            "value": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="Could not infer frequency"):
        _validate_seasonal_inputs(df, "time", "value", 2, **_DEFAULTS)


def test_validate_seasonal_base_freq_inferred_unsupported():
    """Raises ValueError when inferred frequency is outside supported set."""
    # Create minutely data — pd.infer_freq returns "min" or "T" which is not supported
    dates = pd.date_range("2020-01-01", periods=10, freq="min")
    df = pd.DataFrame(
        {"time": dates, "value": np.random.default_rng(42).normal(size=10)}
    )
    with pytest.raises(ValueError, match="not supported"):
        _validate_seasonal_inputs(df, "time", "value", 5, **_DEFAULTS)


# =====================================================================
# _validate_seasonal_inputs — alpha checks
# =====================================================================


def test_validate_seasonal_alpha_zero(seasonal_df_integer):
    """Passes when alpha is 0 (inclusive boundary)."""
    _validate_seasonal_inputs(
        seasonal_df_integer,
        "time",
        "value",
        5,
        backend="plotly",
        width=800,
        height=450,
        alpha=0.0,
        palette="magma",
        base_freq=None,
    )


def test_validate_seasonal_alpha_one(seasonal_df_integer):
    """Passes when alpha is 1 (inclusive boundary)."""
    _validate_seasonal_inputs(
        seasonal_df_integer,
        "time",
        "value",
        5,
        backend="plotly",
        width=800,
        height=450,
        alpha=1.0,
        palette="magma",
        base_freq=None,
    )


def test_validate_seasonal_alpha_negative(seasonal_df_integer):
    """Raises ValueError when alpha is negative."""
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        _validate_seasonal_inputs(
            seasonal_df_integer,
            "time",
            "value",
            5,
            backend="plotly",
            width=800,
            height=450,
            alpha=-0.1,
            palette="magma",
            base_freq=None,
        )


def test_validate_seasonal_alpha_above_one(seasonal_df_integer):
    """Raises ValueError when alpha is above 1."""
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        _validate_seasonal_inputs(
            seasonal_df_integer,
            "time",
            "value",
            5,
            backend="plotly",
            width=800,
            height=450,
            alpha=1.1,
            palette="magma",
            base_freq=None,
        )


# =====================================================================
# _validate_seasonal_inputs — palette checks
# =====================================================================


def test_validate_seasonal_palette_string(seasonal_df_integer):
    """Passes for string palette."""
    _validate_seasonal_inputs(
        seasonal_df_integer,
        "time",
        "value",
        5,
        backend="plotly",
        width=800,
        height=450,
        alpha=0.8,
        palette="viridis",
        base_freq=None,
    )


def test_validate_seasonal_palette_list(seasonal_df_integer):
    """Passes for list palette."""
    _validate_seasonal_inputs(
        seasonal_df_integer,
        "time",
        "value",
        5,
        backend="plotly",
        width=800,
        height=450,
        alpha=0.8,
        palette=["red", "blue", "green"],
        base_freq=None,
    )


def test_validate_seasonal_palette_invalid_type(seasonal_df_integer):
    """Raises TypeError for non-string, non-list palette."""
    with pytest.raises(TypeError, match="palette must be a string or list"):
        _validate_seasonal_inputs(
            seasonal_df_integer,
            "time",
            "value",
            5,
            backend="plotly",
            width=800,
            height=450,
            alpha=0.8,
            palette=42,
            base_freq=None,
        )


# =====================================================================
# _validate_seasonal_inputs — backend / dimension checks
# =====================================================================


def test_validate_seasonal_invalid_backend(seasonal_df_integer):
    """Raises ValueError for unsupported backend."""
    with pytest.raises(ValueError, match="Invalid backend"):
        _validate_seasonal_inputs(
            seasonal_df_integer,
            "time",
            "value",
            5,
            backend="seaborn",
            width=800,
            height=450,
            alpha=0.8,
            palette="magma",
            base_freq=None,
        )


def test_validate_seasonal_width_float_type(seasonal_df_integer):
    """Raises TypeError when width is a float."""
    with pytest.raises(TypeError, match="width must be a positive integer"):
        _validate_seasonal_inputs(
            seasonal_df_integer,
            "time",
            "value",
            5,
            backend="plotly",
            width=800.0,
            height=450,
            alpha=0.8,
            palette="magma",
            base_freq=None,
        )


def test_validate_seasonal_height_nonpositive(seasonal_df_integer):
    """Raises ValueError when height is zero."""
    with pytest.raises(ValueError, match="height must be positive"):
        _validate_seasonal_inputs(
            seasonal_df_integer,
            "time",
            "value",
            5,
            backend="plotly",
            width=800,
            height=0,
            alpha=0.8,
            palette="magma",
            base_freq=None,
        )


# =====================================================================
# _resolve_base_freq
# =====================================================================


def test_resolve_base_freq_explicit():
    """Returns explicit base_freq unchanged."""
    df = pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=12, freq="MS"),
            "value": range(12),
        }
    )
    assert _resolve_base_freq(df, "time", "MS") == "MS"


def test_resolve_base_freq_inferred(seasonal_df_datetime):
    """Infers frequency when base_freq is None."""
    result = _resolve_base_freq(seasonal_df_datetime, "time", None)
    assert result == "MS"


# =====================================================================
# _assign_calendar_seasons — yearly
# =====================================================================


def test_calendar_seasons_yearly():
    """Yearly grouping assigns correct season_ids and positions."""
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    df = pd.DataFrame({"time": dates, "value": range(24)})
    result = _assign_calendar_seasons(df, "time", "year")

    assert list(result["_season_id"].unique()) == ["2020", "2021"]
    # 2020 has 12 months, 2021 has 12 months
    assert list(result.loc[result["_season_id"] == "2020", "_position"]) == list(
        range(1, 13)
    )
    assert list(result.loc[result["_season_id"] == "2021", "_position"]) == list(
        range(1, 13)
    )


def test_calendar_seasons_yearly_alias():
    """'Y' alias works same as 'year'."""
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    df = pd.DataFrame({"time": dates, "value": range(24)})
    result = _assign_calendar_seasons(df, "time", "Y")
    assert list(result["_season_id"].unique()) == ["2020", "2021"]


def test_calendar_seasons_yearly_unequal_lengths():
    """Partial year gets fewer positions."""
    dates = pd.date_range("2020-01-01", periods=15, freq="MS")
    df = pd.DataFrame({"time": dates, "value": range(15)})
    result = _assign_calendar_seasons(df, "time", "year")

    season_2020 = result[result["_season_id"] == "2020"]
    season_2021 = result[result["_season_id"] == "2021"]
    assert len(season_2020) == 12
    assert len(season_2021) == 3
    assert list(season_2021["_position"]) == [1, 2, 3]


# =====================================================================
# _assign_calendar_seasons — quarterly
# =====================================================================


def test_calendar_seasons_quarterly():
    """Quarterly grouping assigns correct season_ids."""
    # 2020 is a leap year: Q1=91 days, need >91 to reach Q2
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({"time": dates, "value": range(100)})
    result = _assign_calendar_seasons(df, "time", "quarter")

    unique_seasons = list(result["_season_id"].unique())
    assert unique_seasons == ["2020-Q1", "2020-Q2"]


def test_calendar_seasons_quarterly_alias():
    """'Q' alias works same as 'quarter'."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({"time": dates, "value": range(100)})
    result = _assign_calendar_seasons(df, "time", "Q")
    assert "2020-Q1" in result["_season_id"].values


# =====================================================================
# _assign_calendar_seasons — monthly
# =====================================================================


def test_calendar_seasons_monthly():
    """Monthly grouping assigns correct season_ids."""
    # 2020: Jan=31, Feb=29 (leap), need >60 to reach March
    dates = pd.date_range("2020-01-01", periods=62, freq="D")
    df = pd.DataFrame({"time": dates, "value": range(62)})
    result = _assign_calendar_seasons(df, "time", "month")

    unique_seasons = list(result["_season_id"].unique())
    assert unique_seasons == ["2020-01", "2020-02", "2020-03"]


def test_calendar_seasons_monthly_alias():
    """'M' alias works same as 'month'."""
    dates = pd.date_range("2020-01-01", periods=62, freq="D")
    df = pd.DataFrame({"time": dates, "value": range(62)})
    result = _assign_calendar_seasons(df, "time", "M")
    assert "2020-01" in result["_season_id"].values


def test_calendar_seasons_monthly_positions():
    """Within-month positions are 1-based observation order."""
    dates = pd.date_range("2020-01-01", periods=31, freq="D")
    df = pd.DataFrame({"time": dates, "value": range(31)})
    result = _assign_calendar_seasons(df, "time", "month")

    jan = result[result["_season_id"] == "2020-01"]
    assert list(jan["_position"]) == list(range(1, 32))


# =====================================================================
# _assign_calendar_seasons — weekly (Sunday start)
# =====================================================================


def test_calendar_seasons_weekly_sunday_start():
    """Weekly grouping uses Sunday as week start."""
    # 2020-01-05 is a Sunday
    dates = pd.date_range("2020-01-05", periods=14, freq="D")
    df = pd.DataFrame({"time": dates, "value": range(14)})
    result = _assign_calendar_seasons(df, "time", "week")

    unique_seasons = list(result["_season_id"].unique())
    # First week starts 2020-01-05 (Sunday),
    # second starts 2020-01-12
    assert unique_seasons == ["2020-01-05", "2020-01-12"]
    week1 = result[result["_season_id"] == "2020-01-05"]
    assert len(week1) == 7
    assert list(week1["_position"]) == list(range(1, 8))


def test_calendar_seasons_weekly_alias():
    """'W' alias works same as 'week'."""
    dates = pd.date_range("2020-01-05", periods=14, freq="D")
    df = pd.DataFrame({"time": dates, "value": range(14)})
    result = _assign_calendar_seasons(df, "time", "W")
    assert "2020-01-05" in result["_season_id"].values


def test_calendar_seasons_weekly_midweek_start():
    """Data starting mid-week creates a partial first week."""
    # 2020-01-08 is a Wednesday; its week starts on Sunday 2020-01-05
    dates = pd.date_range("2020-01-08", periods=10, freq="D")
    df = pd.DataFrame({"time": dates, "value": range(10)})
    result = _assign_calendar_seasons(df, "time", "week")

    first_season = result["_season_id"].iloc[0]
    assert first_season == "2020-01-05"
    # Wed-Sat = 4 days in first week
    week1 = result[result["_season_id"] == "2020-01-05"]
    assert len(week1) == 4


# =====================================================================
# _assign_positional_seasons
# =====================================================================


def test_positional_seasons_basic():
    """Positional grouping with even division."""
    df = pd.DataFrame({"time": range(10), "value": range(10)})
    result = _assign_positional_seasons(df, 5)

    assert list(result["_season_id"].unique()) == ["0", "1"]
    assert list(result["_position"]) == [1, 2, 3, 4, 5] * 2


def test_positional_seasons_partial_last():
    """Partial last season gets fewer positions."""
    df = pd.DataFrame({"time": range(7), "value": range(7)})
    result = _assign_positional_seasons(df, 5)

    season0 = result[result["_season_id"] == "0"]
    season1 = result[result["_season_id"] == "1"]
    assert len(season0) == 5
    assert len(season1) == 2
    assert list(season1["_position"]) == [1, 2]


def test_positional_seasons_labels_are_strings():
    """Season IDs are strings."""
    df = pd.DataFrame({"time": range(10), "value": range(10)})
    result = _assign_positional_seasons(df, 5)
    assert all(isinstance(s, str) for s in result["_season_id"])


# =====================================================================
# _assign_frequency_seasons
# =====================================================================


def test_frequency_seasons_monthly_yearly():
    """Monthly data with period=12 creates yearly seasons."""
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    df = pd.DataFrame({"time": dates, "value": range(24)})
    result = _assign_frequency_seasons(df, "time", 12, "MS")

    assert result["_season_id"].nunique() == 2
    assert list(result["_position"].iloc[:12]) == list(range(1, 13))


def test_frequency_seasons_daily_weekly():
    """Daily data with period=7 creates weekly seasons."""
    dates = pd.date_range("2020-01-01", periods=21, freq="D")
    df = pd.DataFrame({"time": dates, "value": range(21)})
    result = _assign_frequency_seasons(df, "time", 7, "D")

    assert result["_season_id"].nunique() == 3
    assert list(result["_position"]) == list(range(1, 8)) * 3


def test_frequency_seasons_with_missing_observations():
    """Missing observations still get correct positions."""
    # Monthly data with a gap: skip March 2020
    dates = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-04-01", "2020-05-01"])
    df = pd.DataFrame({"time": dates, "value": range(4)})
    result = _assign_frequency_seasons(df, "time", 12, "MS")

    # Positions: Jan=1, Feb=2, (Mar=3 missing), Apr=4, May=5
    assert list(result["_position"]) == [1, 2, 4, 5]


def test_frequency_seasons_labels_are_dates():
    """Season labels use the earliest observation date."""
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    df = pd.DataFrame({"time": dates, "value": range(24)})
    result = _assign_frequency_seasons(df, "time", 12, "MS")

    labels = list(result["_season_id"].unique())
    assert labels[0] == "2020-01-01"
    assert labels[1] == "2021-01-01"


def test_frequency_seasons_weekly_freq():
    """Weekly base_freq with integer period groups correctly."""
    dates = pd.date_range("2020-01-05", periods=8, freq="W-SUN")
    df = pd.DataFrame({"time": dates, "value": range(8)})
    result = _assign_frequency_seasons(df, "time", 4, "W-SUN")

    assert result["_season_id"].nunique() == 2
    assert list(result["_position"].iloc[:4]) == [1, 2, 3, 4]


# =====================================================================
# _check_data_sufficiency
# =====================================================================


def test_sufficiency_named_period_too_few_seasons():
    """Raises ValueError with < 2 seasons (named period)."""
    df = pd.DataFrame({"_season_id": ["2020"] * 5, "_position": range(1, 6)})
    with pytest.raises(ValueError, match="at least 2 seasons"):
        _check_data_sufficiency(df, "year")


def test_sufficiency_named_period_exactly_two_warns():
    """Warns with exactly 2 seasons (named period)."""
    df = pd.DataFrame(
        {
            "_season_id": ["2020"] * 5 + ["2021"] * 5,
            "_position": list(range(1, 6)) * 2,
        }
    )
    with pytest.warns(UserWarning, match="Only 2 distinct"):
        _check_data_sufficiency(df, "year")


def test_sufficiency_named_period_three_no_warning():
    """No warning with >= 3 seasons (named period)."""
    df = pd.DataFrame(
        {
            "_season_id": (["2020"] * 3 + ["2021"] * 3 + ["2022"] * 3),
            "_position": list(range(1, 4)) * 3,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_data_sufficiency(df, "year")


def test_sufficiency_integer_period_too_few_rows():
    """Raises ValueError when len(df) < period."""
    df = pd.DataFrame({"_season_id": ["0"] * 3, "_position": [1, 2, 3]})
    with pytest.raises(ValueError, match="at least 5 observations"):
        _check_data_sufficiency(df, 5)


def test_sufficiency_integer_period_one_full_season_warns():
    """Warns when period <= len(df) < 2 * period."""
    df = pd.DataFrame(
        {
            "_season_id": ["0"] * 5 + ["1"] * 2,
            "_position": [1, 2, 3, 4, 5, 1, 2],
        }
    )
    with pytest.warns(UserWarning, match="at most one full"):
        _check_data_sufficiency(df, 5)


def test_sufficiency_integer_period_two_full_no_warning():
    """No warning when len(df) >= 2 * period."""
    df = pd.DataFrame(
        {
            "_season_id": ["0"] * 5 + ["1"] * 5,
            "_position": list(range(1, 6)) * 2,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_data_sufficiency(df, 5)


# =====================================================================
# _compute_seasonal_data — integration
# =====================================================================


def test_compute_seasonal_data_calendar_path(
    seasonal_df_datetime,
):
    """End-to-end calendar-aligned computation."""
    result = _compute_seasonal_data(seasonal_df_datetime, "time", "value", "year", None)
    assert "_season_id" in result.columns
    assert "_position" in result.columns
    assert result["_season_id"].nunique() == 3
    # Sorted by time_col
    assert result["time"].is_monotonic_increasing


def test_compute_seasonal_data_positional_path(
    seasonal_df_integer,
):
    """End-to-end positional computation."""
    result = _compute_seasonal_data(seasonal_df_integer, "time", "value", 5, None)
    assert "_season_id" in result.columns
    assert result["_season_id"].nunique() == 4
    assert list(result["_position"].iloc[:5]) == list(range(1, 6))


def test_compute_seasonal_data_frequency_path(
    seasonal_df_datetime,
):
    """End-to-end frequency-aligned computation."""
    result = _compute_seasonal_data(seasonal_df_datetime, "time", "value", 12, "MS")
    assert "_season_id" in result.columns
    assert result["_season_id"].nunique() == 3


def test_compute_seasonal_data_frequency_inferred(
    seasonal_df_datetime,
):
    """Frequency-aligned path with inferred base_freq."""
    result = _compute_seasonal_data(seasonal_df_datetime, "time", "value", 12, None)
    assert "_season_id" in result.columns
    assert result["_season_id"].nunique() == 3


def test_compute_seasonal_data_missing_value_col_warns():
    """Warns when value_col has missing values."""
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    values = list(range(24))
    values[5] = np.nan
    values[10] = np.nan
    df = pd.DataFrame({"time": dates, "value": values})

    with pytest.warns(UserWarning, match="2 missing value"):
        _compute_seasonal_data(df, "time", "value", "year", None)


def test_compute_seasonal_data_no_missing_no_warning():
    """No warning when value_col has no missing values."""
    # Use 36 months (3 years) to avoid "only 2 seasons" warning
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    df = pd.DataFrame({"time": dates, "value": range(36)})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _compute_seasonal_data(df, "time", "value", "year", None)


def test_compute_seasonal_data_sorts_by_time_col():
    """Output is sorted by time_col even if input is not."""
    df = pd.DataFrame({"time": [3, 1, 4, 2, 5, 8, 7, 6, 10, 9], "value": range(10)})
    result = _compute_seasonal_data(df, "time", "value", 5, None)
    assert result["time"].is_monotonic_increasing


def test_compute_seasonal_data_sufficiency_error():
    """Raises ValueError when not enough data."""
    df = pd.DataFrame({"time": range(3), "value": range(3)})
    with pytest.raises(ValueError, match="Not enough data"):
        _compute_seasonal_data(df, "time", "value", 5, None)
