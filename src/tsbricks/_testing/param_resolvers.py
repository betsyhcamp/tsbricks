"""Param resolver callables for testing resolver functionality.

These must live in an importable package because ``dynamic_import()``
resolves them by dotted path at runtime.
"""

from __future__ import annotations

import pandas as pd


def constant_resolver(
    y_train: pd.DataFrame,
    grouping_df: pd.DataFrame | None = None,
    value: float = 1.0,
    **kwargs: object,
) -> dict[str, float]:
    """Return a constant value for every unique_id in y_train."""
    unique_ids = y_train["unique_id"].unique()
    return {uid: value for uid in unique_ids}


def grouping_aware_resolver(
    y_train: pd.DataFrame,
    grouping_df: pd.DataFrame | None = None,
    **kwargs: object,
) -> dict[str, float]:
    """Resolver that requires grouping_df; raises if not provided."""
    if grouping_df is None:
        raise ValueError("grouping_aware_resolver requires grouping_df")
    unique_ids = y_train["unique_id"].unique()
    return {uid: 42.0 for uid in unique_ids}
