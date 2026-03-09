"""Fold generation for cross-validation."""

from __future__ import annotations

import warnings

import pandas as pd

from tsbricks.backtesting.schema import (
    CrossValidationConfig,
    DataConfig,
    TestConfig,
)


def generate_folds(
    df: pd.DataFrame,
    cv_config: CrossValidationConfig,
    data_config: DataConfig,
    test_config: TestConfig | None = None,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, pd.DataFrame] | None]:
    """Split a DataFrame into train/val folds based on explicit forecast origins.

    Each fold uses an expanding window: training data includes everything up to
    and including the forecast origin; validation data covers the next
    ``cv_config.horizon`` periods after the origin.

    When ``test_config`` is provided, a test split is also generated using the
    same windowing logic: train includes ``ds <= test_origin``, test includes
    ``test_origin < ds <= test_origin + horizon periods``.  The test fold uses
    ``cv_config.horizon`` (there is no separate test horizon).

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

    if is_integer_ds:
        origins = sorted(int(o) for o in cv_config.forecast_origins)
    else:
        origins = sorted(pd.Timestamp(o) for o in cv_config.forecast_origins)
        offset = pd.tseries.frequencies.to_offset(data_config.freq)

    pad_width = len(str(len(origins) - 1))

    cv_folds: dict[str, dict[str, pd.DataFrame]] = {}
    for i, origin in enumerate(origins):
        if is_integer_ds:
            val_end = origin + cv_config.horizon
        else:
            val_end = origin + cv_config.horizon * offset

        train = df[df["ds"] <= origin]
        val = df[(df["ds"] > origin) & (df["ds"] <= val_end)]

        fold_key = f"fold_{i:0{pad_width}d}"
        cv_folds[fold_key] = {"train": train, "val": val}

    # ---- test split ----
    test_split: dict[str, pd.DataFrame] | None = None
    if test_config is not None:
        if is_integer_ds:
            test_origin = int(test_config.test_origin)
            test_end = test_origin + cv_config.horizon
        else:
            test_origin = pd.Timestamp(test_config.test_origin)
            test_end = test_origin + cv_config.horizon * offset

        # Runtime validation: test window must fit within available data
        data_end = df["ds"].max()
        if test_end > data_end:
            raise ValueError(
                f"Test window exceeds available data: test_end={test_end} "
                f"but data ends at {data_end}. Need at least "
                f"{cv_config.horizon} periods after test_origin={test_origin}."
            )

        # Overlap warning: test_origin falls within last CV validation window
        last_origin = origins[-1]
        if is_integer_ds:
            last_val_end = last_origin + cv_config.horizon
        else:
            last_val_end = last_origin + cv_config.horizon * offset

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

        train = df[df["ds"] <= test_origin]
        test = df[(df["ds"] > test_origin) & (df["ds"] <= test_end)]
        test_split = {"train": train, "test": test}

    return cv_folds, test_split
