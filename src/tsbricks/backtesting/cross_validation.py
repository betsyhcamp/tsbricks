"""Fold generation for cross-validation."""

from __future__ import annotations

import pandas as pd

from tsbricks.backtesting.schema import CrossValidationConfig, DataConfig


def generate_folds(
    df: pd.DataFrame,
    cv_config: CrossValidationConfig,
    data_config: DataConfig,
    test_config: dict | None = None,
) -> tuple[dict[str, dict[str, pd.DataFrame]], None]:
    """Split a DataFrame into train/val folds based on explicit forecast origins.

    Each fold uses an expanding window: training data includes everything up to
    and including the forecast origin; validation data covers the next
    ``cv_config.horizon`` periods after the origin.

    Supports both datetime and integer ``ds`` columns.  The mode is inferred
    from the column dtype and cross-validated against ``data_config.freq``.

    Args:
        df: Panel DataFrame with columns ``ds``, ``unique_id``, and ``y``.
            The ``ds`` column must be datetime or integer dtype.
        cv_config: Cross-validation configuration (explicit mode only in V1).
        data_config: Data configuration (used for ``freq``).
        test_config: Reserved for future test-fold support. Unused in V1.

    Returns:
        A tuple ``(cv_folds, None)`` where ``cv_folds`` is an ordered dict
        of ``{"fold_X": {"train": df, "val": df}, ...}``.  Folds are ordered
        chronologically by origin value (``fold_0`` has the earliest origin).
        The second element is ``None`` (no test fold in V1).
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

    return cv_folds, None
