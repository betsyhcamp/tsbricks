"""Plotting utilities for time series visualization."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import pandas as pd

from tsbricks.blocks.utils import (
    convert_to_pandas,
    validate_backend,
    validate_column_exists,
    validate_dataframe,
    validate_dimensions,
    validate_no_duplicates,
    validate_no_missing_values,
    validate_not_empty,
    validate_time_col_dtype,
    validate_value_col_dtype,
)

if TYPE_CHECKING:
    import polars as pl

    DataFrameLike: TypeAlias = pd.DataFrame | pl.DataFrame
else:
    DataFrameLike = pd.DataFrame


_NAMED_PERIODS = frozenset({"week", "W", "month", "M", "quarter", "Q", "year", "Y"})

_SUPPORTED_BASE_FREQS = frozenset(
    {
        "W",
        "W-SUN",
        "W-MON",
        "W-TUE",
        "W-WED",
        "W-THU",
        "W-FRI",
        "W-SAT",
        "MS",
        "ME",
        "QS",
        "QE",
        "YS",
        "YE",
        "D",
        "h",
    }
)


def _is_datetime_time_col(df: pd.DataFrame, time_col: str) -> bool:
    """Check if time_col is datetime-like (including object-dtype datetimes)."""
    time_dtype = df[time_col].dtype
    if pd.api.types.is_datetime64_any_dtype(time_dtype):
        return True
    if pd.api.types.is_integer_dtype(time_dtype):
        return False
    # Object column — check via inference
    inferred = pd.api.types.infer_dtype(df[time_col], skipna=True)
    return inferred in ("datetime", "datetime64", "date")


def _validate_seasonal_inputs(
    df: DataFrameLike,
    time_col: str,
    value_col: str,
    period: str | int,
    backend: str,
    width: int,
    height: int,
    alpha: float,
    palette: str | list,
    base_freq: str | None,
) -> None:
    """Validate inputs for plot_seasonal.

    Checks DataFrame type, column existence, dtypes, missing values,
    duplicates, period, base_freq, alpha, palette, backend, and dimensions.
    Converts Polars to pandas internally before column-level checks.
    """
    # --- DataFrame ---
    validate_dataframe(df)
    pdf = convert_to_pandas(df)

    validate_not_empty(pdf)

    # --- Columns ---
    for col_name, label in [(time_col, "time_col"), (value_col, "value_col")]:
        validate_column_exists(pdf, col_name, label)

    validate_time_col_dtype(pdf, time_col)
    validate_value_col_dtype(pdf, value_col)

    # --- Missing time_col values (error) ---
    validate_no_missing_values(pdf, time_col, "time_col")

    # --- Duplicate time_col values (error) ---
    validate_no_duplicates(pdf, time_col, "time_col")

    # --- period ---
    is_datetime = _is_datetime_time_col(pdf, time_col)

    if isinstance(period, str):
        if not is_datetime:
            raise ValueError(
                f"Named period '{period}' is only allowed for datetime-like "
                f"time_col, but time_col '{time_col}' is integer dtype."
            )
        if period not in _NAMED_PERIODS:
            raise ValueError(
                f"Invalid named period '{period}'. "
                f"Must be one of {sorted(_NAMED_PERIODS)}."
            )
    elif isinstance(period, (int, np.integer)) and not isinstance(period, bool):
        if period < 2:
            raise ValueError(f"Integer period must be >= 2, got {period}.")
    else:
        raise TypeError(
            f"period must be a string or integer, got {type(period).__name__}."
        )

    # --- base_freq ---
    _validate_base_freq(pdf, time_col, period, base_freq)

    # --- alpha ---
    if not (0 <= alpha <= 1):
        raise ValueError(f"alpha must be between 0 and 1 inclusive, got {alpha}.")

    # --- palette ---
    if not isinstance(palette, (str, list)):
        raise TypeError(
            f"palette must be a string or list, got {type(palette).__name__}."
        )

    # --- backend / dimensions ---
    validate_backend(backend)
    validate_dimensions(width, height)


def _validate_base_freq(
    df: pd.DataFrame,
    time_col: str,
    period: str | int,
    base_freq: str | None,
) -> None:
    """Validate base_freq parameter.

    base_freq is only valid when time_col is datetime-like and period is
    an integer. When not provided, attempts inference via pd.infer_freq().
    """
    is_datetime = _is_datetime_time_col(df, time_col)
    is_int_period = isinstance(period, (int, np.integer)) and not isinstance(
        period, bool
    )

    # base_freq only valid for datetime time_col + integer period
    if base_freq is not None:
        if not is_datetime:
            raise ValueError(
                "base_freq is only valid when time_col is datetime-like, "
                f"but time_col '{time_col}' is integer dtype."
            )
        if not is_int_period:
            raise ValueError(
                "base_freq is only valid when period is an integer, "
                f"but period='{period}' is a named period."
            )
        if base_freq not in _SUPPORTED_BASE_FREQS:
            raise ValueError(
                f"Unsupported base_freq '{base_freq}'. "
                f"Must be one of {sorted(_SUPPORTED_BASE_FREQS)}."
            )
        return

    # Inference needed only for datetime time_col + integer period
    if not (is_datetime and is_int_period):
        return

    # Attempt inference
    sorted_times = df[time_col].sort_values()
    inferred = pd.infer_freq(sorted_times)

    if inferred is None:
        raise ValueError(
            "Could not infer frequency from time_col. "
            "Please provide base_freq explicitly. "
            f"Supported values: {sorted(_SUPPORTED_BASE_FREQS)}."
        )

    if inferred not in _SUPPORTED_BASE_FREQS:
        raise ValueError(
            f"Inferred frequency '{inferred}' is not supported. "
            "Please provide base_freq explicitly. "
            f"Supported values: {sorted(_SUPPORTED_BASE_FREQS)}."
        )


# =====================================================================
# Mapping from offset aliases to Period frequencies
# =====================================================================

_BASE_FREQ_TO_PERIOD_FREQ = {
    "D": "D",
    "h": "h",
    "W": "W-SUN",
    "W-SUN": "W-SUN",
    "W-MON": "W-MON",
    "W-TUE": "W-TUE",
    "W-WED": "W-WED",
    "W-THU": "W-THU",
    "W-FRI": "W-FRI",
    "W-SAT": "W-SAT",
    "MS": "M",
    "ME": "M",
    "QS": "Q",
    "QE": "Q",
    "YS": "Y",
    "YE": "Y",
}


# =====================================================================
# Season assignment helpers
# =====================================================================


def _resolve_base_freq(
    df: pd.DataFrame,
    time_col: str,
    base_freq: str | None,
) -> str:
    """Return provided base_freq or infer from data.

    Assumes validation has already confirmed that inference will
    succeed and produce a supported value.
    """
    if base_freq is not None:
        return base_freq
    sorted_times = df[time_col].sort_values()
    return pd.infer_freq(sorted_times)


def _assign_calendar_seasons(
    df: pd.DataFrame,
    time_col: str,
    period: str,
) -> pd.DataFrame:
    """Assign _season_id and _position via calendar-aligned grouping.

    Used when time_col is datetime-like and period is a named string.
    """
    pdf = df.copy()
    times = pdf[time_col]

    if period in ("year", "Y"):
        pdf["_season_id"] = times.dt.year.astype(str)
    elif period in ("quarter", "Q"):
        pdf["_season_id"] = (
            times.dt.year.astype(str) + "-Q" + times.dt.quarter.astype(str)
        )
    elif period in ("month", "M"):
        pdf["_season_id"] = times.dt.to_period("M").astype(str)
    elif period in ("week", "W"):
        # Sunday-start weeks: find the Sunday that begins each week
        offset_days = (times.dt.dayofweek + 1) % 7
        week_start = times - pd.to_timedelta(offset_days, unit="D")
        pdf["_season_id"] = week_start.dt.strftime("%Y-%m-%d")

    pdf["_position"] = pdf.groupby("_season_id", sort=False).cumcount() + 1
    return pdf


def _assign_positional_seasons(
    df: pd.DataFrame,
    period: int,
) -> pd.DataFrame:
    """Assign _season_id and _position via positional grouping.

    Used when time_col is integer dtype and period is an integer.
    Every ``period`` consecutive observations form one season.
    """
    pdf = df.copy()
    indices = np.arange(len(pdf))
    pdf["_season_id"] = (indices // period).astype(str)
    pdf["_position"] = (indices % period) + 1
    return pdf


def _assign_frequency_seasons(
    df: pd.DataFrame,
    time_col: str,
    period: int,
    base_freq: str,
) -> pd.DataFrame:
    """Assign _season_id and _position via frequency-aligned grouping.

    Used when time_col is datetime-like and period is an integer.
    ``base_freq`` defines the step size; every ``period`` steps form
    one season.
    """
    pdf = df.copy()
    period_freq = _BASE_FREQ_TO_PERIOD_FREQ[base_freq]
    period_index = pd.PeriodIndex(pdf[time_col], freq=period_freq)
    ordinals = period_index.asi8
    step_indices = ordinals - ordinals[0]

    season_indices = step_indices // period
    pdf["_position"] = (step_indices % period) + 1

    # Label each season by its earliest observation date
    pdf["_season_idx"] = season_indices
    min_dates = pdf.groupby("_season_idx")[time_col].transform("first")
    pdf["_season_id"] = min_dates.dt.strftime("%Y-%m-%d")
    pdf.drop(columns=["_season_idx"], inplace=True)

    return pdf


# =====================================================================
# Data sufficiency
# =====================================================================


def _check_data_sufficiency(
    df: pd.DataFrame,
    period: str | int,
) -> None:
    """Validate that there is enough data for a seasonal plot.

    Raises ValueError if less than one full season. Emits a warning
    if there is exactly one full season.
    """
    if isinstance(period, str):
        # Group-based rule (datetime + named period)
        n_seasons = df["_season_id"].nunique()
        if n_seasons < 2:
            raise ValueError(
                "Not enough data for a seasonal plot. "
                "Need at least 2 seasons, got "
                f"{n_seasons}."
            )
        if n_seasons == 2:
            warnings.warn(
                "Only 2 distinct seasons found. Consider "
                "using more data for a meaningful seasonal "
                "comparison.",
                stacklevel=3,
            )
    else:
        # Count-based rule (integer period)
        n = len(df)
        int_period = int(period)
        if n < int_period:
            raise ValueError(
                "Not enough data for a seasonal plot. "
                f"Need at least {int_period} observations "
                f"(one full season), got {n}."
            )
        if n < 2 * int_period:
            warnings.warn(
                f"Only {n} observations for "
                f"period={int_period}. This gives at most "
                "one full season. Consider using more data "
                "for a meaningful seasonal comparison.",
                stacklevel=3,
            )


# =====================================================================
# Main computation entry point
# =====================================================================


def _compute_seasonal_data(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    period: str | int,
    base_freq: str | None,
) -> pd.DataFrame:
    """Compute season identity and within-season positions.

    Sorts by ``time_col``, assigns ``_season_id`` (str) and
    ``_position`` (1-based int) columns, checks data sufficiency,
    and warns about missing ``value_col`` values.

    Returns:
        A copy of *df* sorted by *time_col* with ``_season_id``
        and ``_position`` columns appended.
    """
    pdf = df.sort_values(time_col).reset_index(drop=True)

    # Warn about missing value_col values
    na_count = pdf[value_col].isna().sum()
    if na_count > 0:
        warnings.warn(
            f"value_col '{value_col}' contains "
            f"{int(na_count)} missing value(s). "
            "These will appear as gaps in the plot.",
            stacklevel=2,
        )

    is_datetime = _is_datetime_time_col(pdf, time_col)

    if isinstance(period, str):
        result = _assign_calendar_seasons(pdf, time_col, period)
    elif is_datetime:
        resolved = _resolve_base_freq(pdf, time_col, base_freq)
        result = _assign_frequency_seasons(pdf, time_col, period, resolved)
    else:
        result = _assign_positional_seasons(pdf, period)

    _check_data_sufficiency(result, period)

    return result
