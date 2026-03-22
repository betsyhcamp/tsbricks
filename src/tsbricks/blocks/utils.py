"""Shared DataFrame helpers and auxiliary utilities"""

from __future__ import annotations

from typing import Iterable, FrozenSet

import numpy as np

try:
    import pandas as pd
except ImportError:
    # pandas dependency not present in environment
    pd = None
try:
    import polars as pl
except ImportError:
    # polars dependency not present in environment
    pl = None

_REQUIRED: FrozenSet[str] = frozenset({"unique_id", "ds", "y"})
_DPI = 100


def _is_pandas_df(obj) -> bool:
    """Determine if pandas dependency available and if object is pandas dataframe"""
    return pd is not None and isinstance(obj, pd.DataFrame)


def _is_polars_df(obj) -> bool:
    """Determine if polars dependency available and if object is polars dataframe"""
    return pl is not None and isinstance(obj, pl.DataFrame)


def missing_required_columns(df, required: Iterable[str] = _REQUIRED) -> None:
    """Assert that `df` contains the required columns; raise if any are missing.

    This function validates that the provided DataFrame (pandas or polars)
    includes all columns listed in `required`. If the object is not a
    supported DataFrame type, a TypeError is raised. If one or more
    required columns are absent, a `ValueError` is raised enumerating the
    missing names.

    Args:
        df (dataframe): The input table-like expected to be a pandas or polars
            DataFrame.
        required(Iterable[str]): Iterable of column names that must be present.
            Defaults to the canonical set `{"unique_id", "ds", "y"}`.

    Raises:
        TypeError: If `df` is not a pandas or polars DataFrame.
        ValueError: If one or more required columns are missing.
    """

    if _is_pandas_df(df) or _is_polars_df(df):
        columns = list(df.columns)
    else:
        raise TypeError(f"Expects pandas or polars dataframe, got {type(df).__name__}")

    missing = set(required) - set(columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {sorted(missing)}")


def convert_to_pandas(df):
    """Convert a Polars DataFrame to pandas; pass through pandas DataFrames unchanged.

    Args:
        df: A pandas or Polars DataFrame.

    Returns:
        A pandas DataFrame.
    """
    if _is_pandas_df(df):
        return df
    return df.to_pandas()


def pixels_to_figsize(width: int, height: int) -> tuple[float, float]:
    """Convert pixel dimensions to matplotlib figsize (inches) using _DPI."""
    return (width / _DPI, height / _DPI)


def validate_dataframe(df) -> None:
    """Validate that df is a pandas or Polars DataFrame.

    Raises:
        TypeError: If df is not a pandas or Polars DataFrame.
    """
    if not (_is_pandas_df(df) or _is_polars_df(df)):
        raise TypeError(
            f"df must be a pandas or Polars DataFrame, got {type(df).__name__}."
        )


def validate_not_empty(df: pd.DataFrame) -> None:
    """Validate that a pandas DataFrame is not empty.

    Raises:
        ValueError: If the DataFrame has zero rows.
    """
    if len(df) == 0:
        raise ValueError("DataFrame must not be empty.")


def validate_min_rows(df: pd.DataFrame, min_rows: int) -> None:
    """Validate that a pandas DataFrame has at least min_rows rows.

    Raises:
        ValueError: If the DataFrame has fewer than min_rows rows.
    """
    if len(df) < min_rows:
        raise ValueError(
            f"DataFrame must have at least {min_rows} rows, got {len(df)}."
        )


def validate_column_exists(df: pd.DataFrame, col_name: str, label: str) -> None:
    """Validate that a column exists in the DataFrame.

    Args:
        df: pandas DataFrame.
        col_name: Column name to check.
        label: Descriptive label for error messages (e.g. "time_col", "value_col").

    Raises:
        ValueError: If the column is not found.
    """
    if col_name not in df.columns:
        raise ValueError(
            f"{label} '{col_name}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )


def validate_time_col_dtype(df: pd.DataFrame, time_col: str) -> None:
    """Validate that time_col is datetime-like or integer dtype.

    Object columns holding Python datetime objects are accepted via
    pandas value-level inference.

    Raises:
        ValueError: If the column dtype is not datetime-like or integer.
    """
    time_dtype = df[time_col].dtype
    is_datetime = pd.api.types.is_datetime64_any_dtype(time_dtype)
    is_integer = pd.api.types.is_integer_dtype(time_dtype)
    if not (is_datetime or is_integer):
        inferred = pd.api.types.infer_dtype(df[time_col], skipna=True)
        if inferred not in ("datetime", "datetime64", "date"):
            raise ValueError(
                f"time_col '{time_col}' must be datetime-like or integer dtype, "
                f"got {time_dtype}."
            )


def validate_value_col_dtype(df: pd.DataFrame, value_col: str) -> None:
    """Validate that value_col is real numeric (not complex).

    Raises:
        ValueError: If the column is not numeric or is complex.
    """
    value_dtype = df[value_col].dtype
    if not pd.api.types.is_numeric_dtype(value_dtype) or pd.api.types.is_complex_dtype(
        value_dtype
    ):
        raise ValueError(f"value_col '{value_col}' must be numeric, got {value_dtype}.")


def validate_no_missing_values(df: pd.DataFrame, col_name: str, label: str) -> None:
    """Validate that a column has no missing values.

    Args:
        df: pandas DataFrame.
        col_name: Column name to check.
        label: Descriptive label for error messages (e.g. "time_col").

    Raises:
        ValueError: If the column contains missing values.
    """
    na_count = df[col_name].isna().sum()
    if na_count > 0:
        raise ValueError(
            f"{label} '{col_name}' contains {int(na_count)} missing value(s)."
        )


def validate_no_duplicates(df: pd.DataFrame, col_name: str, label: str) -> None:
    """Validate that a column has no duplicate values.

    Args:
        df: pandas DataFrame.
        col_name: Column name to check.
        label: Descriptive label for error messages (e.g. "time_col").

    Raises:
        ValueError: If the column contains duplicates.
    """
    n_dups = df[col_name].duplicated().sum()
    if n_dups > 0:
        raise ValueError(
            f"{label} '{col_name}' contains {int(n_dups)} duplicate value(s)."
        )


def validate_backend(backend: str) -> None:
    """Validate the plotting backend string.

    Raises:
        ValueError: If backend is not 'plotly' or 'matplotlib'.
    """
    if backend not in ("plotly", "matplotlib"):
        raise ValueError(
            f"Invalid backend '{backend}'. Must be 'plotly' or 'matplotlib'."
        )


def validate_dimensions(width: int, height: int) -> None:
    """Validate width and height are positive integers (not bools).

    Raises:
        TypeError: If width or height is not an integer.
        ValueError: If width or height is not positive.
    """
    for param_name, param_val in [("width", width), ("height", height)]:
        if isinstance(param_val, bool) or not isinstance(param_val, (int, np.integer)):
            raise TypeError(
                f"{param_name} must be a positive integer, "
                f"got {type(param_val).__name__}."
            )
        if param_val <= 0:
            raise ValueError(f"{param_name} must be positive, got {param_val}.")
