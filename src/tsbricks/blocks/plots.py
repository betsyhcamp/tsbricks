"""Plotting utilities for time series visualization."""

from __future__ import annotations

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
