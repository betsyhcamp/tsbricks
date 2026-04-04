"""Fold generation for cross-validation."""

from __future__ import annotations

import warnings

import pandas as pd

from tsbricks.backtesting.schema import (
    CrossValidationConfig,
    DataConfig,
    TestConfig,
)


def _split_at_origin(
    df: pd.DataFrame,
    origin: pd.Timestamp | int,
    horizon: int,
    is_integer_ds: bool,
    offset: pd.DateOffset | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp | int]:
    """Slice a DataFrame into train/eval sets around a single origin.

    Args:
        df: Panel DataFrame with a ``ds`` column.
        origin: The cutoff point (inclusive in train).
        horizon: Number of periods for the eval window.
        is_integer_ds: Whether ``ds`` is integer-typed.
        offset: Frequency offset for datetime ``ds`` (ignored when integer).

    Returns:
        ``(train, eval_set, end)`` where ``end`` is the upper bound of the
        eval window (``origin + horizon`` for integers, ``origin + horizon *
        offset`` for datetimes).
    """
    if is_integer_ds:
        end = origin + horizon
    else:
        end = origin + horizon * offset
    train = df[df["ds"] <= origin]
    eval_set = df[(df["ds"] > origin) & (df["ds"] <= end)]
    return train, eval_set, end


def generate_folds(
    df: pd.DataFrame,
    cv_config: CrossValidationConfig,
    data_config: DataConfig,
    test_config: TestConfig | None = None,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, pd.DataFrame] | None]:
    """Split a DataFrame into train/val folds based on explicit forecast origins.

    Each fold uses an expanding window: training data includes everything up to
    and including the forecast origin; validation data covers the next
    *horizon* periods after the origin.  The horizon may differ per origin
    when variable-horizon configuration is used.

    When ``test_config`` is provided, a test split is also generated using the
    same windowing logic: train includes ``ds <= test_origin``, test includes
    ``test_origin < ds <= test_origin + horizon periods``.

    Supports both datetime and integer ``ds`` columns.  The mode is inferred
    from the column dtype and cross-validated against ``data_config.freq``.

    Args:
        df: Panel DataFrame with columns ``ds``, ``unique_id``, and ``y``.
            The ``ds`` column must be datetime or integer dtype.
        cv_config: Cross-validation configuration (explicit mode only in V1).
        data_config: Data configuration (used for ``freq``).
        test_config: Optional test fold configuration.  When provided, the
            test split is generated and returned as the second tuple element.

    Returns:
        A tuple ``(cv_folds, test_split)`` where ``cv_folds`` is an ordered
        dict of ``{"fold_X": {"train": df, "val": df}, ...}``.  Folds are
        ordered chronologically by origin value (``fold_0`` has the earliest
        origin).  ``test_split`` is ``{"train": df, "test": df}`` when
        ``test_config`` is provided, otherwise ``None``.
    """
    is_integer_ds = pd.api.types.is_integer_dtype(df["ds"])
    is_datetime_ds = pd.api.types.is_datetime64_any_dtype(df["ds"])

    if is_integer_ds and data_config.freq != 1:
        raise ValueError(
            f"Integer ds column requires freq=1, got freq={data_config.freq!r}."
        )
    if data_config.freq == 1 and not is_integer_ds:
        raise ValueError(
            f"freq=1 requires an integer ds column, but ds has dtype {df['ds'].dtype}."
        )
    if not is_integer_ds and not is_datetime_ds:
        raise ValueError(
            f"The 'ds' column must be datetime or integer dtype, got {df['ds'].dtype}."
        )

    offset: pd.DateOffset | None = None

    # Build sorted (typed_origin, horizon) pairs
    raw_pairs = cv_config.origin_horizon_pairs()
    if is_integer_ds:
        origin_horizon_list = sorted((int(o), h) for o, h in raw_pairs)
    else:
        origin_horizon_list = sorted((pd.Timestamp(o), h) for o, h in raw_pairs)
        offset = pd.tseries.frequencies.to_offset(data_config.freq)

    # Validate: no CV horizon extends past available data
    data_end = df["ds"].max()
    for origin, horizon in origin_horizon_list:
        if is_integer_ds:
            end = origin + horizon
        else:
            end = origin + horizon * offset
        if end > data_end:
            raise ValueError(
                f"CV fold horizon exceeds available data: "
                f"origin={origin} with horizon={horizon} "
                f"requires data through {end}, but data "
                f"ends at {data_end}."
            )

    pad_width = len(str(len(origin_horizon_list) - 1))

    cv_folds: dict[str, dict[str, pd.DataFrame]] = {}
    for i, (origin, horizon) in enumerate(origin_horizon_list):
        train, val, _ = _split_at_origin(df, origin, horizon, is_integer_ds, offset)
        fold_key = f"fold_{i:0{pad_width}d}"
        cv_folds[fold_key] = {"train": train, "val": val}

    # ---- test split ----
    test_split: dict[str, pd.DataFrame] | None = None
    if test_config is not None:
        if is_integer_ds:
            test_origin = int(test_config.test_origin)
        else:
            test_origin = pd.Timestamp(test_config.test_origin)

        train, test, test_end = _split_at_origin(
            df, test_origin, cv_config.horizon, is_integer_ds, offset
        )

        # Runtime validation: test window must fit within available data
        data_end = df["ds"].max()
        if test_end > data_end:
            raise ValueError(
                f"Test window exceeds available data: test_end={test_end} "
                f"but data ends at {data_end}. Need at least "
                f"{cv_config.horizon} periods after test_origin={test_origin}."
            )

        # Overlap warning: test_origin falls within last CV validation window
        last_origin, last_horizon = origin_horizon_list[-1]
        _, _, last_val_end = _split_at_origin(
            df, last_origin, last_horizon, is_integer_ds, offset
        )

        if test_origin < last_val_end:
            warnings.warn(
                f"Test fold overlaps with cross-validation: test_origin "
                f"({test_origin}) falls within the last CV validation window "
                f"(origin={last_origin}, val_end={last_val_end}). The test "
                f"fold should ideally not overlap with any cross-validation "
                f"window.",
                UserWarning,
                stacklevel=2,
            )

        test_split = {"train": train, "test": test}

    return cv_folds, test_split
