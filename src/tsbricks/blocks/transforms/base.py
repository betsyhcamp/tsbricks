from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import pandas as pd


class BaseTransform(ABC):
    """Abstract base class for all transforms.

    Defines the four-method interface that all transforms must implement,
    plus a ``_map_per_series`` helper that isolates per-series iteration
    into a single swappable location.
    """

    @abstractmethod
    def fit_transform(
        self, df: pd.DataFrame, target_col: str, **params: object
    ) -> pd.DataFrame:
        """Fit the transform on *df* and return the transformed DataFrame.

        Must always ``df.copy()`` — never mutate the input.
        """

    @abstractmethod
    def transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Apply the already-fitted transform to new data.

        Must always ``df.copy()`` — never mutate the input.
        """

    @abstractmethod
    def inverse_transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Reverse the transform, returning data to the original scale.

        Must always ``df.copy()`` — never mutate the input.
        """

    @abstractmethod
    def get_fitted_params(self) -> dict[str, dict]:
        """Return fitted parameters keyed by ``unique_id``.

        Values must be plain Python types (``float``, ``int``, ``str``),
        **not** numpy scalars.

        Returns:
            ``{unique_id: {param_name: value, ...}, ...}``
        """

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _map_per_series(
        self,
        df: pd.DataFrame,
        target_col: str,
        fn: Callable[[pd.DataFrame, str, str], pd.DataFrame],
    ) -> pd.DataFrame:
        """Group *df* by ``unique_id``, apply *fn* to each group, and
        reassemble the result.

        Args:
            df: Panel DataFrame containing a ``unique_id`` column.
            target_col: Name of the column to transform.
            fn: ``fn(series_df, target_col, unique_id) -> series_df``
                where *series_df* is a single-series slice.

        Returns:
            DataFrame with the same columns as *df*, rows in the same
            order as the original groups.
        """
        parts: list[pd.DataFrame] = []
        for uid, group_df in df.groupby("unique_id", sort=False):
            parts.append(fn(group_df, target_col, str(uid)))
        return pd.concat(parts, ignore_index=True)
