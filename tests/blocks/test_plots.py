"""Tests for tsbricks.blocks.plots — seasonal input validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsbricks.blocks.plots import (
    _is_datetime_time_col,
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
