"""Tests for tsbricks.blocks.plots — seasonal validation and computation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tsbricks.blocks.plots import (
    _assign_calendar_seasons,
    _assign_custom_seasons,
    _assign_frequency_seasons,
    _assign_positional_seasons,
    _check_data_sufficiency,
    _compute_seasonal_data,
    _is_datetime_time_col,
    _normalize_freq,
    _plot_seasonal_matplotlib,
    _plot_seasonal_plotly,
    _resolve_base_freq,
    _sample_colors,
    _validate_seasonal_inputs,
    plot_seasonal,
    _NAMED_PERIODS,
    _SUPPORTED_BASE_FREQS,
    _VIRIDIS_UPPER,
)

# =====================================================================
# Shared defaults for _validate_seasonal_inputs
# =====================================================================

_DEFAULTS = dict(
    backend="plotly",
    width=800,
    height=450,
    alpha=0.8,
    palette="viridis",
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
        palette="viridis",
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
        palette="viridis",
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
    """Raises ValueError when neither period nor season_col is provided."""
    with pytest.raises(ValueError, match="Either period or season_col"):
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
            palette="viridis",
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
            palette="viridis",
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
            palette="viridis",
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
            palette="viridis",
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
        palette="viridis",
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
        palette="viridis",
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
            palette="viridis",
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
            palette="viridis",
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
            palette="viridis",
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
            palette="viridis",
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
            palette="viridis",
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


# =====================================================================
# _sample_colors — list palette (backend-independent)
# =====================================================================


def test_sample_colors_list_passthrough():
    """Color list is returned directly when long enough."""
    palette = ["red", "blue", "green", "orange", "purple"]
    colors = _sample_colors(palette, 3, "plotly")
    assert colors == ["red", "blue", "green"]


def test_sample_colors_list_exact_length():
    """Color list of exact length passes."""
    palette = ["red", "blue", "green"]
    colors = _sample_colors(palette, 3, "plotly")
    assert colors == palette


def test_sample_colors_list_too_short():
    """Raises ValueError when color list has fewer colors than seasons."""
    with pytest.raises(ValueError, match="2 color.*3 season"):
        _sample_colors(["red", "blue"], 3, "plotly")


def test_sample_colors_list_ignores_backend():
    """List path works with either backend value."""
    palette = ["red", "blue", "green"]
    assert _sample_colors(palette, 2, "plotly") == ["red", "blue"]
    assert _sample_colors(palette, 2, "matplotlib") == ["red", "blue"]


# =====================================================================
# _sample_colors — plotly backend
# =====================================================================


def test_sample_colors_plotly_viridis_count():
    """Returns correct number of rgb() colors for viridis via plotly."""
    colors = _sample_colors("viridis", 5, "plotly")
    assert len(colors) == 5
    assert all(c.startswith("rgb(") for c in colors)


def test_sample_colors_plotly_viridis_excludes_lightest():
    """Viridis samples stop at _VIRIDIS_UPPER (lightest 8% excluded) via plotly."""
    import plotly.colors as pc

    colors = _sample_colors("viridis", 10, "plotly")
    expected_last = pc.sample_colorscale("viridis", [_VIRIDIS_UPPER], colortype="rgb")
    assert colors[-1] == expected_last[0]


def test_sample_colors_plotly_other_cmap_full_range():
    """Non-viridis colormaps sample full [0, 1] range via plotly."""
    import plotly.colors as pc

    colors = _sample_colors("plasma", 10, "plotly")
    expected_first = pc.sample_colorscale("plasma", [0.0], colortype="rgb")
    expected_last = pc.sample_colorscale("plasma", [1.0], colortype="rgb")
    assert colors[0] == expected_first[0]
    assert colors[-1] == expected_last[0]


def test_sample_colors_plotly_single_color():
    """Sampling 1 color returns one element via plotly."""
    colors = _sample_colors("viridis", 1, "plotly")
    assert len(colors) == 1
    assert colors[0].startswith("rgb(")


def test_sample_colors_plotly_unknown_colorscale():
    """Raises ValueError for unrecognized colorscale name via plotly."""
    with pytest.raises(ValueError, match="Unknown colorscale"):
        _sample_colors("not_a_real_colormap", 5, "plotly")


# =====================================================================
# _sample_colors — matplotlib backend
# =====================================================================

mpl = pytest.importorskip("matplotlib")


def test_sample_colors_matplotlib_viridis_count():
    """Returns correct number of hex colors for viridis via matplotlib."""
    colors = _sample_colors("viridis", 5, "matplotlib")
    assert len(colors) == 5
    assert all(c.startswith("#") for c in colors)


def test_sample_colors_matplotlib_viridis_excludes_lightest():
    """Viridis samples stop at _VIRIDIS_UPPER (lightest 8% excluded) via matplotlib."""
    import matplotlib as mpl
    import matplotlib.colors as mcolors

    colors = _sample_colors("viridis", 10, "matplotlib")
    cmap = mpl.colormaps["viridis"]
    last_color = mcolors.to_hex(cmap(_VIRIDIS_UPPER))
    assert colors[-1] == last_color


def test_sample_colors_matplotlib_other_cmap_full_range():
    """Non-viridis colormaps sample the full [0, 1] range via matplotlib."""
    import matplotlib as mpl
    import matplotlib.colors as mcolors

    colors = _sample_colors("plasma", 10, "matplotlib")
    cmap = mpl.colormaps["plasma"]
    assert colors[-1] == mcolors.to_hex(cmap(1.0))
    assert colors[0] == mcolors.to_hex(cmap(0.0))


def test_sample_colors_matplotlib_single_color():
    """Sampling 1 color returns one element via matplotlib."""
    colors = _sample_colors("viridis", 1, "matplotlib")
    assert len(colors) == 1
    assert colors[0].startswith("#")


def test_sample_colors_matplotlib_unknown_colormap():
    """Raises ValueError for unrecognized colormap name via matplotlib."""
    with pytest.raises(ValueError, match="Unknown colormap"):
        _sample_colors("not_a_real_colormap", 5, "matplotlib")


# =====================================================================
# _plot_seasonal_plotly
# =====================================================================


@pytest.fixture
def seasonal_plot_data():
    """Computed seasonal data ready for plotting (3 seasons of 5)."""
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    df = pd.DataFrame(
        {"time": dates, "value": np.random.default_rng(42).normal(size=36)}
    )
    return _compute_seasonal_data(df, "time", "value", "year", None)


def test_plot_seasonal_plotly_returns_figure(seasonal_plot_data):
    """Returns a Plotly Figure object."""
    import plotly.graph_objects as go

    colors = _sample_colors(
        "viridis", seasonal_plot_data["_season_id"].nunique(), "plotly"
    )
    fig = _plot_seasonal_plotly(
        seasonal_plot_data, "time", "value", colors, 0.8, 800, 450
    )
    assert isinstance(fig, go.Figure)


def test_plot_seasonal_plotly_trace_count(seasonal_plot_data):
    """One trace per season."""
    n_seasons = seasonal_plot_data["_season_id"].nunique()
    colors = _sample_colors("viridis", n_seasons, "plotly")
    fig = _plot_seasonal_plotly(
        seasonal_plot_data, "time", "value", colors, 0.8, 800, 450
    )
    assert len(fig.data) == n_seasons


def test_plot_seasonal_plotly_trace_names(seasonal_plot_data):
    """Trace names match season IDs in order."""
    seasons = list(dict.fromkeys(seasonal_plot_data["_season_id"]))
    colors = _sample_colors("viridis", len(seasons), "plotly")
    fig = _plot_seasonal_plotly(
        seasonal_plot_data, "time", "value", colors, 0.8, 800, 450
    )
    trace_names = [t.name for t in fig.data]
    assert trace_names == seasons


def test_plot_seasonal_plotly_axis_labels(seasonal_plot_data):
    """X and Y axis labels follow the spec."""
    colors = _sample_colors(
        "viridis", seasonal_plot_data["_season_id"].nunique(), "plotly"
    )
    fig = _plot_seasonal_plotly(
        seasonal_plot_data, "time", "value", colors, 0.8, 800, 450
    )
    assert fig.layout.xaxis.title.text == "time (time)"
    assert fig.layout.yaxis.title.text == "value"


def test_plot_seasonal_plotly_no_title(seasonal_plot_data):
    """No plot title."""
    colors = _sample_colors(
        "viridis", seasonal_plot_data["_season_id"].nunique(), "plotly"
    )
    fig = _plot_seasonal_plotly(
        seasonal_plot_data, "time", "value", colors, 0.8, 800, 450
    )
    assert fig.layout.title is None or fig.layout.title.text is None


def test_plot_seasonal_plotly_dimensions(seasonal_plot_data):
    """Width and height are set correctly."""
    colors = _sample_colors(
        "viridis", seasonal_plot_data["_season_id"].nunique(), "plotly"
    )
    fig = _plot_seasonal_plotly(
        seasonal_plot_data, "time", "value", colors, 0.8, 600, 300
    )
    assert fig.layout.width == 600
    assert fig.layout.height == 300


def test_plot_seasonal_plotly_legend_visible(seasonal_plot_data):
    """Legend is shown."""
    colors = _sample_colors(
        "viridis", seasonal_plot_data["_season_id"].nunique(), "plotly"
    )
    fig = _plot_seasonal_plotly(
        seasonal_plot_data, "time", "value", colors, 0.8, 800, 450
    )
    assert fig.layout.showlegend is True


def test_plot_seasonal_plotly_mode_lines_markers(seasonal_plot_data):
    """All traces use lines+markers mode."""
    colors = _sample_colors(
        "viridis", seasonal_plot_data["_season_id"].nunique(), "plotly"
    )
    fig = _plot_seasonal_plotly(
        seasonal_plot_data, "time", "value", colors, 0.8, 800, 450
    )
    for trace in fig.data:
        assert trace.mode == "lines+markers"


def test_plot_seasonal_plotly_missing_values_gaps():
    """Missing value_col values create gaps (connectgaps=False)."""
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    values = list(range(24))
    values[5] = np.nan
    df = pd.DataFrame({"time": dates, "value": values})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = _compute_seasonal_data(df, "time", "value", "year", None)

    colors = _sample_colors("viridis", data["_season_id"].nunique(), "plotly")
    fig = _plot_seasonal_plotly(data, "time", "value", colors, 0.8, 800, 450)
    for trace in fig.data:
        assert trace.connectgaps is False


# =====================================================================
# _plot_seasonal_matplotlib
# =====================================================================


def test_plot_seasonal_matplotlib_returns_figure(seasonal_plot_data):
    """Returns a Matplotlib Figure object."""
    import matplotlib.figure

    colors = _sample_colors(
        "viridis", seasonal_plot_data["_season_id"].nunique(), "matplotlib"
    )
    fig = _plot_seasonal_matplotlib(
        seasonal_plot_data, "time", "value", colors, 0.8, 800, 450
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_seasonal_matplotlib_line_count(seasonal_plot_data):
    """One line per season."""
    import matplotlib.pyplot as plt

    n_seasons = seasonal_plot_data["_season_id"].nunique()
    colors = _sample_colors("viridis", n_seasons, "matplotlib")
    fig = _plot_seasonal_matplotlib(
        seasonal_plot_data, "time", "value", colors, 0.8, 800, 450
    )
    ax = fig.axes[0]
    assert len(ax.lines) == n_seasons
    plt.close(fig)


def test_plot_seasonal_matplotlib_axis_labels(seasonal_plot_data):
    """X and Y axis labels follow the spec."""
    import matplotlib.pyplot as plt

    colors = _sample_colors(
        "viridis", seasonal_plot_data["_season_id"].nunique(), "matplotlib"
    )
    fig = _plot_seasonal_matplotlib(
        seasonal_plot_data, "time", "value", colors, 0.8, 800, 450
    )
    ax = fig.axes[0]
    assert ax.get_xlabel() == "time (time)"
    assert ax.get_ylabel() == "value"
    plt.close(fig)


def test_plot_seasonal_matplotlib_legend_labels(seasonal_plot_data):
    """Legend labels match season IDs."""
    import matplotlib.pyplot as plt

    seasons = list(dict.fromkeys(seasonal_plot_data["_season_id"]))
    colors = _sample_colors("viridis", len(seasons), "matplotlib")
    fig = _plot_seasonal_matplotlib(
        seasonal_plot_data, "time", "value", colors, 0.8, 800, 450
    )
    ax = fig.axes[0]
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_texts == seasons
    plt.close(fig)


def test_plot_seasonal_matplotlib_dimensions(seasonal_plot_data):
    """Figure size matches requested pixel dimensions."""
    import matplotlib.pyplot as plt

    colors = _sample_colors(
        "viridis", seasonal_plot_data["_season_id"].nunique(), "matplotlib"
    )
    fig = _plot_seasonal_matplotlib(
        seasonal_plot_data, "time", "value", colors, 0.8, 600, 300
    )
    w, h = fig.get_size_inches()
    assert abs(w - 6.0) < 0.01  # 600 / 100 DPI
    assert abs(h - 3.0) < 0.01  # 300 / 100 DPI
    plt.close(fig)


def test_plot_seasonal_matplotlib_alpha(seasonal_plot_data):
    """Lines have the requested alpha."""
    import matplotlib.pyplot as plt

    colors = _sample_colors(
        "viridis", seasonal_plot_data["_season_id"].nunique(), "matplotlib"
    )
    fig = _plot_seasonal_matplotlib(
        seasonal_plot_data, "time", "value", colors, 0.5, 800, 450
    )
    ax = fig.axes[0]
    for line in ax.lines:
        assert line.get_alpha() == 0.5
    plt.close(fig)


# =====================================================================
# plot_seasonal — shared fixture for suppressing interactive rendering
# =====================================================================


@pytest.fixture
def no_show(monkeypatch):
    """Prevent fig.show() and plt.show() from opening interactive windows."""
    monkeypatch.setattr("plotly.graph_objects.Figure.show", lambda self: None)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)


# =====================================================================
# plot_seasonal — public API integration tests (plotly)
# =====================================================================


def test_plot_seasonal_plotly_return_fig_true(seasonal_df_datetime, no_show):
    """return_fig=True returns a Plotly Figure."""
    import plotly.graph_objects as go

    fig = plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        return_fig=True,
    )
    assert isinstance(fig, go.Figure)


def test_plot_seasonal_plotly_return_fig_false(seasonal_df_datetime, no_show):
    """return_fig=False returns None."""
    result = plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        return_fig=False,
    )
    assert result is None


def test_plot_seasonal_plotly_named_period_year(seasonal_df_datetime, no_show):
    """End-to-end with named period='year'."""
    fig = plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        return_fig=True,
    )
    assert len(fig.data) == 3  # 3 years


def test_plot_seasonal_plotly_named_period_month(seasonal_df_datetime, no_show):
    """End-to-end with named period='month'."""
    fig = plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "month",
        return_fig=True,
    )
    # 36 months → 36 unique month season IDs
    assert len(fig.data) == 36


def test_plot_seasonal_plotly_integer_period_positional(seasonal_df_integer, no_show):
    """End-to-end with integer time_col and integer period."""
    fig = plot_seasonal(
        seasonal_df_integer,
        "time",
        "value",
        5,
        return_fig=True,
    )
    assert len(fig.data) == 4  # 20 obs / 5 = 4 seasons


def test_plot_seasonal_plotly_integer_period_frequency_aligned(
    seasonal_df_datetime,
    no_show,
):
    """End-to-end with datetime time_col and integer period."""
    fig = plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        12,
        base_freq="MS",
        return_fig=True,
    )
    assert len(fig.data) == 3  # 36 months / 12 = 3 seasons


def test_plot_seasonal_plotly_missing_values_warn_and_render(no_show):
    """Missing values in value_col emit a warning and still render."""
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    values = np.arange(36, dtype=float)
    values[5] = np.nan
    df = pd.DataFrame({"time": dates, "value": values})

    with pytest.warns(UserWarning, match="missing value"):
        fig = plot_seasonal(df, "time", "value", "year", return_fig=True)
    assert len(fig.data) == 3


def test_plot_seasonal_plotly_custom_palette_list(seasonal_df_datetime, no_show):
    """Custom color list is applied to traces."""
    colors = ["red", "green", "blue"]
    fig = plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        palette=colors,
        return_fig=True,
    )
    for trace, expected in zip(fig.data, colors):
        assert trace.line.color == expected


def test_plot_seasonal_plotly_custom_alpha(seasonal_df_datetime, no_show):
    """Custom alpha is applied to traces."""
    fig = plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        alpha=0.5,
        return_fig=True,
    )
    for trace in fig.data:
        assert trace.opacity == 0.5


def test_plot_seasonal_plotly_custom_dimensions(seasonal_df_datetime, no_show):
    """Custom width and height are applied."""
    fig = plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        width=600,
        height=300,
        return_fig=True,
    )
    assert fig.layout.width == 600
    assert fig.layout.height == 300


# =====================================================================
# plot_seasonal — public API integration tests (matplotlib)
# =====================================================================


def test_plot_seasonal_matplotlib_return_fig_true(seasonal_df_datetime, no_show):
    """return_fig=True returns a Matplotlib Figure."""
    import matplotlib.figure
    import matplotlib.pyplot as plt

    fig = plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        backend="matplotlib",
        return_fig=True,
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_seasonal_matplotlib_return_fig_false(seasonal_df_datetime, no_show):
    """return_fig=False returns None."""
    import matplotlib.pyplot as plt

    result = plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        backend="matplotlib",
        return_fig=False,
    )
    assert result is None
    plt.close("all")


def test_plot_seasonal_matplotlib_named_period_year(seasonal_df_datetime, no_show):
    """End-to-end with named period='year' on matplotlib."""
    import matplotlib.pyplot as plt

    fig = plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        backend="matplotlib",
        return_fig=True,
    )
    ax = fig.axes[0]
    assert len(ax.lines) == 3
    plt.close(fig)


def test_plot_seasonal_matplotlib_integer_period(seasonal_df_integer, no_show):
    """End-to-end with integer time_col and integer period on matplotlib."""
    import matplotlib.pyplot as plt

    fig = plot_seasonal(
        seasonal_df_integer,
        "time",
        "value",
        5,
        backend="matplotlib",
        return_fig=True,
    )
    ax = fig.axes[0]
    assert len(ax.lines) == 4
    plt.close(fig)


def test_plot_seasonal_matplotlib_custom_alpha(seasonal_df_datetime, no_show):
    """Custom alpha is applied to lines."""
    import matplotlib.pyplot as plt

    fig = plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        backend="matplotlib",
        alpha=0.3,
        return_fig=True,
    )
    ax = fig.axes[0]
    for line in ax.lines:
        assert line.get_alpha() == 0.3
    plt.close(fig)


# =====================================================================
# plot_seasonal — Polars input
# =====================================================================


def test_plot_seasonal_polars_plotly(no_show):
    """Polars DataFrame works end-to-end with plotly backend."""
    pl = pytest.importorskip("polars")
    import plotly.graph_objects as go

    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    df_pl = pl.DataFrame(
        {
            "time": dates.to_list(),
            "value": np.random.default_rng(42).normal(size=36).tolist(),
        }
    )

    fig = plot_seasonal(df_pl, "time", "value", "year", return_fig=True)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3


def test_plot_seasonal_polars_matplotlib(no_show):
    """Polars DataFrame works end-to-end with matplotlib backend."""
    pl = pytest.importorskip("polars")
    import matplotlib.figure
    import matplotlib.pyplot as plt

    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    df_pl = pl.DataFrame(
        {
            "time": dates.to_list(),
            "value": np.random.default_rng(42).normal(size=36).tolist(),
        }
    )

    fig = plot_seasonal(
        df_pl,
        "time",
        "value",
        "year",
        backend="matplotlib",
        return_fig=True,
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


# =====================================================================
# plot_seasonal — error propagation
# =====================================================================


def test_plot_seasonal_empty_dataframe_raises():
    """Empty DataFrame raises ValueError."""
    df = pd.DataFrame(
        {
            "time": pd.Series(dtype="datetime64[ns]"),
            "value": pd.Series(dtype="float64"),
        }
    )
    with pytest.raises(ValueError, match="empty"):
        plot_seasonal(df, "time", "value", "year")


def test_plot_seasonal_insufficient_data_raises():
    """Not enough data for one full season raises ValueError."""
    df = pd.DataFrame({"time": range(3), "value": range(3)})
    with pytest.raises(ValueError, match="Not enough data"):
        plot_seasonal(df, "time", "value", 5)


def test_plot_seasonal_invalid_backend_raises():
    """Invalid backend raises ValueError."""
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    df = pd.DataFrame({"time": dates, "value": range(36)})
    with pytest.raises(ValueError, match="Invalid backend"):
        plot_seasonal(df, "time", "value", "year", backend="seaborn")


def test_plot_seasonal_invalid_period_type_raises():
    """Invalid period type raises TypeError."""
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    df = pd.DataFrame({"time": dates, "value": range(36)})
    with pytest.raises(TypeError, match="period must be"):
        plot_seasonal(df, "time", "value", 3.5)


def test_plot_seasonal_missing_column_raises():
    """Missing column raises ValueError."""
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    df = pd.DataFrame({"time": dates, "value": range(36)})
    with pytest.raises(ValueError, match="not found"):
        plot_seasonal(df, "time", "nonexistent", "year")


# =====================================================================
# return_fig=True suppresses rendering
# =====================================================================


def test_plot_seasonal_return_fig_true_no_show_called(
    seasonal_df_datetime, monkeypatch
):
    """return_fig=True returns figure without calling fig.show()."""
    show_calls = []
    monkeypatch.setattr(
        "plotly.graph_objects.Figure.show",
        lambda self: show_calls.append(1),
    )
    fig = plot_seasonal(seasonal_df_datetime, "time", "value", "year", return_fig=True)
    assert fig is not None
    assert len(show_calls) == 0


def test_plot_seasonal_return_fig_false_calls_show(seasonal_df_datetime, monkeypatch):
    """return_fig=False calls fig.show() exactly once."""
    show_calls = []
    monkeypatch.setattr(
        "plotly.graph_objects.Figure.show",
        lambda self: show_calls.append(1),
    )
    result = plot_seasonal(
        seasonal_df_datetime, "time", "value", "year", return_fig=False
    )
    assert result is None
    assert len(show_calls) == 1


def test_plot_seasonal_matplotlib_return_fig_true_no_show_called(
    seasonal_df_datetime, monkeypatch
):
    """return_fig=True with matplotlib returns figure without plt.show()."""
    show_calls = []
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: show_calls.append(1))
    fig = plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        backend="matplotlib",
        return_fig=True,
    )
    assert fig is not None
    assert len(show_calls) == 0


def test_plot_seasonal_matplotlib_return_fig_false_calls_show(
    seasonal_df_datetime, monkeypatch
):
    """return_fig=False with matplotlib calls plt.show() exactly once."""
    show_calls = []
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: show_calls.append(1))
    result = plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        backend="matplotlib",
        return_fig=False,
    )
    assert result is None
    assert len(show_calls) == 1


# =====================================================================
# _normalize_freq — anchored alias normalization
# =====================================================================


def test_normalize_freq_unanchored_passthrough():
    """Unanchored aliases pass through unchanged."""
    for freq in ("D", "h", "MS", "ME", "QS", "QE", "YS", "YE"):
        assert _normalize_freq(freq) == freq


def test_normalize_freq_anchored_quarterly():
    """Anchored quarterly aliases normalize to unanchored form."""
    assert _normalize_freq("QE-DEC") == "QE"
    assert _normalize_freq("QE-MAR") == "QE"
    assert _normalize_freq("QS-JAN") == "QS"
    assert _normalize_freq("QS-APR") == "QS"


def test_normalize_freq_anchored_yearly():
    """Anchored yearly aliases normalize to unanchored form."""
    assert _normalize_freq("YE-DEC") == "YE"
    assert _normalize_freq("YS-JAN") == "YS"
    assert _normalize_freq("YS-JUL") == "YS"


def test_normalize_freq_weekly_anchored_preserved():
    """Weekly anchored aliases are preserved (already in supported set)."""
    assert _normalize_freq("W-MON") == "W-MON"
    assert _normalize_freq("W-SUN") == "W-SUN"
    assert _normalize_freq("W-FRI") == "W-FRI"


def test_validate_base_freq_accepts_anchored_quarterly_inference():
    """Validation passes for quarterly data where infer_freq returns anchored alias."""
    dates = pd.date_range("2020-01-01", periods=12, freq="QE-DEC")
    df = pd.DataFrame({"time": dates, "value": range(12)})
    # Should not raise — anchored 'QE-DEC' is normalized to 'QE'
    _validate_seasonal_inputs(
        df, "time", "value", 4, "plotly", 800, 450, 0.8, "viridis", None
    )


def test_validate_base_freq_accepts_anchored_yearly_inference():
    """Validation passes for yearly data where infer_freq returns anchored alias."""
    dates = pd.date_range("2015-01-01", periods=10, freq="YS-JAN")
    df = pd.DataFrame({"time": dates, "value": range(10)})
    # Should not raise — anchored 'YS-JAN' is normalized to 'YS'
    _validate_seasonal_inputs(
        df, "time", "value", 5, "plotly", 800, 450, 0.8, "viridis", None
    )


def test_resolve_base_freq_normalizes_anchored(seasonal_df_datetime):
    """_resolve_base_freq returns normalized (unanchored) frequency."""
    dates = pd.date_range("2020-01-01", periods=12, freq="QE-DEC")
    df = pd.DataFrame({"time": dates, "value": range(12)})
    resolved = _resolve_base_freq(df, "time", None)
    assert resolved == "QE"


# =====================================================================
# plot_seasonal — ax parameter tests
# =====================================================================


def test_plot_seasonal_ax_draws_on_provided_axes(
    seasonal_df_datetime,
    no_show,
):
    """When ax is provided, draws on it and returns its parent figure."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    returned = plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        backend="matplotlib",
        ax=ax,
    )
    assert returned is fig
    # 3 years of data → 3 lines
    assert len(ax.lines) == 3
    plt.close(fig)


def test_plot_seasonal_ax_no_tight_layout_call(
    seasonal_df_datetime,
    no_show,
    monkeypatch,
):
    """When ax is provided, tight_layout is not called."""
    import matplotlib.pyplot as plt
    import matplotlib.figure

    tight_layout_called = []
    original = matplotlib.figure.Figure.tight_layout

    def spy(self, *a, **kw):
        tight_layout_called.append(True)
        return original(self, *a, **kw)

    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "tight_layout",
        spy,
    )

    fig, ax = plt.subplots()
    plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        backend="matplotlib",
        ax=ax,
    )
    assert len(tight_layout_called) == 0
    plt.close(fig)


def test_plot_seasonal_ax_does_not_create_new_figure(
    seasonal_df_datetime,
    no_show,
    monkeypatch,
):
    """When ax is provided, plt.subplots is not called internally."""
    import matplotlib.pyplot as plt

    subplots_called = []
    original = plt.subplots

    def spy(*a, **kw):
        subplots_called.append(True)
        return original(*a, **kw)

    monkeypatch.setattr(plt, "subplots", spy)

    fig, ax = plt.subplots()
    subplots_called.clear()  # reset after our own call
    plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        backend="matplotlib",
        ax=ax,
    )
    assert len(subplots_called) == 0
    plt.close(fig)


def test_plot_seasonal_ax_with_plotly_raises(seasonal_df_datetime):
    """ax with backend='plotly' raises ValueError."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="backend='plotly'"):
        plot_seasonal(
            seasonal_df_datetime,
            "time",
            "value",
            "year",
            backend="plotly",
            ax=ax,
        )
    plt.close(fig)


def test_plot_seasonal_ax_invalid_type_raises(seasonal_df_datetime):
    """ax that is not an Axes instance raises TypeError."""
    with pytest.raises(TypeError, match="matplotlib.axes.Axes"):
        plot_seasonal(
            seasonal_df_datetime,
            "time",
            "value",
            "year",
            backend="matplotlib",
            ax="not_an_axes",
        )


def test_plot_seasonal_ax_none_existing_behavior(
    seasonal_df_datetime,
    no_show,
):
    """ax=None preserves existing behavior (return_fig controls return)."""
    import matplotlib.pyplot as plt

    fig = plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        backend="matplotlib",
        return_fig=True,
        ax=None,
    )
    assert fig is not None
    plt.close(fig)

    result = plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        backend="matplotlib",
        return_fig=False,
        ax=None,
    )
    assert result is None
    plt.close("all")


def test_plot_seasonal_ax_in_subplots(seasonal_df_datetime, no_show):
    """ax works correctly in a multi-subplot figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig1 = plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        backend="matplotlib",
        ax=axes[0],
    )
    fig2 = plot_seasonal(
        seasonal_df_datetime,
        "time",
        "value",
        "year",
        backend="matplotlib",
        ax=axes[1],
    )
    # Both return the same parent figure
    assert fig1 is fig
    assert fig2 is fig
    # Both axes have lines drawn
    assert len(axes[0].lines) == 3
    assert len(axes[1].lines) == 3
    plt.close(fig)


# =====================================================================
# season_col — _assign_custom_seasons helper
# =====================================================================


@pytest.fixture
def fiscal_df():
    """Weekly integer DataFrame with a fiscal_year column.

    3 fiscal years, 52 weeks each, except the first which has 51
    (starts at week 2).
    """
    rows = []
    # FY2023: weeks 2–52 (51 rows)
    for w in range(2, 53):
        rows.append({"time": 202300 + w, "value": float(w), "fiscal_year": "FY2023"})
    # FY2024: weeks 1–52 (52 rows)
    for w in range(1, 53):
        rows.append({"time": 202400 + w, "value": float(w), "fiscal_year": "FY2024"})
    # FY2025: weeks 1–52 (52 rows)
    for w in range(1, 53):
        rows.append({"time": 202500 + w, "value": float(w), "fiscal_year": "FY2025"})
    return pd.DataFrame(rows)


# =====================================================================
# _assign_custom_seasons
# =====================================================================


def test_assign_custom_seasons_season_id(fiscal_df):
    result = _assign_custom_seasons(fiscal_df, "fiscal_year")
    assert "_season_id" in result.columns
    assert set(result["_season_id"]) == {"FY2023", "FY2024", "FY2025"}


def test_assign_custom_seasons_position_via_cumcount(fiscal_df):
    result = _assign_custom_seasons(fiscal_df, "fiscal_year")
    # FY2023 has 51 rows → positions 1–51
    fy23 = result[result["_season_id"] == "FY2023"]
    assert fy23["_position"].tolist() == list(range(1, 52))
    # FY2024 has 52 rows → positions 1–52
    fy24 = result[result["_season_id"] == "FY2024"]
    assert fy24["_position"].tolist() == list(range(1, 53))


def test_assign_custom_seasons_preserves_columns(fiscal_df):
    result = _assign_custom_seasons(fiscal_df, "fiscal_year")
    for col in fiscal_df.columns:
        assert col in result.columns


# =====================================================================
# season_col — validation
# =====================================================================


def test_validate_season_col_and_period_raises(fiscal_df):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_seasonal_inputs(
            fiscal_df,
            "time",
            "value",
            period=52,
            season_col="fiscal_year",
            **_DEFAULTS,
        )


def test_validate_neither_season_col_nor_period_raises(seasonal_df_datetime):
    with pytest.raises(ValueError, match="Either period or season_col"):
        _validate_seasonal_inputs(
            seasonal_df_datetime,
            "time",
            "value",
            period=None,
            season_col=None,
            **_DEFAULTS,
        )


def test_validate_missing_season_col_raises(fiscal_df):
    with pytest.raises(ValueError, match="not_a_column"):
        _validate_seasonal_inputs(
            fiscal_df,
            "time",
            "value",
            period=None,
            season_col="not_a_column",
            **_DEFAULTS,
        )


def test_validate_season_col_only_passes(fiscal_df):
    _validate_seasonal_inputs(
        fiscal_df,
        "time",
        "value",
        period=None,
        season_col="fiscal_year",
        **_DEFAULTS,
    )


# =====================================================================
# season_col — _compute_seasonal_data
# =====================================================================


def test_compute_seasonal_data_uses_custom_seasons(fiscal_df):
    result = _compute_seasonal_data(
        fiscal_df,
        "time",
        "value",
        period=None,
        base_freq=None,
        season_col="fiscal_year",
    )
    assert set(result["_season_id"]) == {"FY2023", "FY2024", "FY2025"}


def test_compute_seasonal_data_custom_position_aligns(fiscal_df):
    """Position 1 in each season is the first row of that season."""
    result = _compute_seasonal_data(
        fiscal_df,
        "time",
        "value",
        period=None,
        base_freq=None,
        season_col="fiscal_year",
    )
    for fy in ["FY2023", "FY2024", "FY2025"]:
        season = result[result["_season_id"] == fy]
        assert season["_position"].iloc[0] == 1


def test_compute_seasonal_data_infers_period_from_max_group(fiscal_df):
    """Sufficiency check uses inferred period (52) from largest group."""
    result = _compute_seasonal_data(
        fiscal_df,
        "time",
        "value",
        period=None,
        base_freq=None,
        season_col="fiscal_year",
    )
    assert result["_season_id"].nunique() == 3


# =====================================================================
# season_col — plot_seasonal public API
# =====================================================================


def test_plot_seasonal_season_col_matplotlib(fiscal_df):
    import matplotlib.pyplot as plt

    fig = plot_seasonal(
        fiscal_df,
        "time",
        "value",
        season_col="fiscal_year",
        backend="matplotlib",
        return_fig=True,
    )
    assert fig is not None
    ax = fig.axes[0]
    assert len(ax.lines) == 3
    plt.close(fig)


def test_plot_seasonal_season_col_plotly(fiscal_df):
    fig = plot_seasonal(
        fiscal_df,
        "time",
        "value",
        season_col="fiscal_year",
        backend="plotly",
        return_fig=True,
    )
    assert len(fig.data) == 3


def test_plot_seasonal_season_col_polars(fiscal_df):
    import polars as pl

    pl_df = pl.from_pandas(fiscal_df)
    fig = plot_seasonal(
        pl_df,
        "time",
        "value",
        season_col="fiscal_year",
        backend="matplotlib",
        return_fig=True,
    )
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_seasonal_season_col_partial_first_aligns(fiscal_df):
    """With season_col, a short first season doesn't misalign later ones."""
    import matplotlib.pyplot as plt

    fig = plot_seasonal(
        fiscal_df,
        "time",
        "value",
        season_col="fiscal_year",
        backend="matplotlib",
        return_fig=True,
    )
    ax = fig.axes[0]
    lines = ax.lines
    # FY2023 has 51 points, FY2024 and FY2025 have 52
    x_lengths = [len(line.get_xdata()) for line in lines]
    assert sorted(x_lengths) == [51, 52, 52]
    plt.close(fig)


def test_plot_seasonal_season_col_with_ax(fiscal_df):
    """season_col works with the ax parameter."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    returned = plot_seasonal(
        fiscal_df,
        "time",
        "value",
        season_col="fiscal_year",
        backend="matplotlib",
        ax=ax,
    )
    assert returned is fig
    assert len(ax.lines) == 3
    plt.close(fig)


# =====================================================================
# Partial first season warning (positional grouping)
# =====================================================================


def test_no_partial_warning_on_complete_seasons():
    """No warning when all seasons are complete."""
    df = pd.DataFrame(
        {
            "time": range(104),
            "value": np.random.default_rng(42).normal(size=104),
        }
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _compute_seasonal_data(df, "time", "value", period=52, base_freq=None)
    partial_warnings = [x for x in w if "Season boundaries" in str(x.message)]
    assert len(partial_warnings) == 0


def test_warns_when_last_season_short():
    """Warning fires when rows aren't evenly divisible by period.

    With 155 rows and period=52, the last season has 51 rows.
    """
    df = pd.DataFrame(
        {
            "time": range(155),
            "value": np.random.default_rng(42).normal(size=155),
        }
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _compute_seasonal_data(df, "time", "value", period=52, base_freq=None)
    partial_warnings = [x for x in w if "Season boundaries" in str(x.message)]
    assert len(partial_warnings) == 1
    assert "51 observations" in str(partial_warnings[0].message)
    assert "season_col" in str(partial_warnings[0].message)


def test_no_partial_warning_when_season_col_provided(fiscal_df):
    """No partial season warning when season_col handles grouping."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _compute_seasonal_data(
            fiscal_df,
            "time",
            "value",
            period=None,
            base_freq=None,
            season_col="fiscal_year",
        )
    partial_warnings = [x for x in w if "Season boundaries" in str(x.message)]
    assert len(partial_warnings) == 0
