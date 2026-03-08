from __future__ import annotations

from typing import Literal

import pandas as pd

from tsbricks.blocks.transforms.base import BaseTransform

_VALID_SCOPES = ("global", "per_series")
_CalendarScope = Literal["global", "per_series"]

_N_WORKDAYS_COL = "n_workdays"

_GLOBAL_JOIN_KEYS = ["ds"]
_PER_SERIES_JOIN_KEYS = ["ds", "unique_id"]

_MAX_MISSING_KEYS_IN_ERROR = 20


class WorkdayNormalizeTransform(BaseTransform):
    """Normalize a target column by the number of working days per period.

    Divides the target by ``n_workdays`` during the forward transform and
    multiplies by ``n_workdays`` during the inverse transform.  This is
    useful when the raw target (e.g., monthly revenue) is driven by the
    number of business days in each period.

    The transform requires a **calendar DataFrame** that maps each period
    to its working-day count.  The calendar must cover both historical and
    future (forecast-horizon) periods so that ``inverse_transform`` can
    reverse the normalization on predictions.

    Params (passed via ``fit_transform(..., **params)``):
        calendar_df: :class:`~pandas.DataFrame` with a ``ds`` column and a
            ``n_workdays`` column.  When *calendar_scope* is
            ``"per_series"``, the DataFrame must also contain a
            ``unique_id`` column.
        calendar_scope: ``"global"`` when all series share the same
            working-day calendar, or ``"per_series"`` when working days
            vary by series.
    """

    def __init__(self) -> None:
        self._calendar_df: pd.DataFrame | None = None
        self._calendar_scope: _CalendarScope | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_and_store(
        self,
        calendar_df: pd.DataFrame,
        calendar_scope: str,
    ) -> None:
        if calendar_scope not in _VALID_SCOPES:
            raise ValueError(
                f"calendar_scope must be one of {_VALID_SCOPES}, got {calendar_scope!r}"
            )

        required_cols = {"ds", _N_WORKDAYS_COL}
        if calendar_scope == "per_series":
            required_cols.add("unique_id")

        missing = required_cols - set(calendar_df.columns)
        if missing:
            raise ValueError(
                f"calendar_df is missing required columns for "
                f"calendar_scope={calendar_scope!r}: {sorted(missing)}"
            )

        if (calendar_df[_N_WORKDAYS_COL] <= 0).any():
            raise ValueError(
                f"All {_N_WORKDAYS_COL} values in calendar_df must be positive (> 0)."
            )

        join_keys = self._join_keys_for_scope(calendar_scope)
        if calendar_df.duplicated(subset=join_keys).any():
            raise ValueError(
                f"calendar_df contains duplicate entries on "
                f"{join_keys}. Each combination must be unique."
            )

        self._calendar_df = calendar_df.copy()
        self._calendar_scope = calendar_scope  # type: ignore[assignment]

    @staticmethod
    def _join_keys_for_scope(calendar_scope: str) -> list[str]:
        if calendar_scope == "per_series":
            return _PER_SERIES_JOIN_KEYS
        return _GLOBAL_JOIN_KEYS

    def _join_keys(self) -> list[str]:
        assert self._calendar_scope is not None
        return self._join_keys_for_scope(self._calendar_scope)

    def _validate_no_reserved_column(self, df: pd.DataFrame) -> None:
        if _N_WORKDAYS_COL in df.columns:
            raise ValueError(
                f"Input DataFrame already contains column "
                f"'{_N_WORKDAYS_COL}'. This column name is reserved by "
                f"WorkdayNormalizeTransform. Rename it before applying "
                f"this transform."
            )

    def _join_working_days(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._calendar_df is not None
        keys = self._join_keys()
        merged = df.merge(
            self._calendar_df[keys + [_N_WORKDAYS_COL]],
            on=keys,
            how="left",
        )
        nulls = merged[_N_WORKDAYS_COL].isna()
        if nulls.any():
            n_missing = int(nulls.sum())
            missing_rows = df.loc[nulls.values, keys].drop_duplicates()
            missing_keys = [
                tuple(row) if len(keys) > 1 else row[keys[0]]
                for _, row in missing_rows.head(_MAX_MISSING_KEYS_IN_ERROR).iterrows()
            ]
            msg = (
                f"calendar_df does not cover {n_missing} row(s) in the "
                f"data. Missing keys (showing first "
                f"{min(n_missing, _MAX_MISSING_KEYS_IN_ERROR)} of "
                f"{n_missing} total): {missing_keys}"
            )
            raise ValueError(msg)
        return merged

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit_transform(
        self, df: pd.DataFrame, target_col: str, **params: object
    ) -> pd.DataFrame:
        calendar_df = params.get("calendar_df")
        calendar_scope = params.get("calendar_scope")

        if calendar_df is None:
            raise ValueError("calendar_df is required")
        if calendar_scope is None:
            raise ValueError("calendar_scope is required")
        if not isinstance(calendar_df, pd.DataFrame):
            raise TypeError(
                f"calendar_df must be a pandas DataFrame, "
                f"got {type(calendar_df).__name__}"
            )

        self._validate_no_reserved_column(df)
        self._validate_and_store(calendar_df, str(calendar_scope))

        df = df.copy()
        merged = self._join_working_days(df)
        merged[target_col] = merged[target_col] / merged[_N_WORKDAYS_COL]
        return merged.drop(columns=[_N_WORKDAYS_COL]).reset_index(drop=True)

    def transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        self._validate_no_reserved_column(df)
        df = df.copy()
        merged = self._join_working_days(df)
        merged[target_col] = merged[target_col] / merged[_N_WORKDAYS_COL]
        return merged.drop(columns=[_N_WORKDAYS_COL]).reset_index(drop=True)

    def inverse_transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        self._validate_no_reserved_column(df)
        df = df.copy()
        merged = self._join_working_days(df)
        merged[target_col] = merged[target_col] * merged[_N_WORKDAYS_COL]
        return merged.drop(columns=[_N_WORKDAYS_COL]).reset_index(drop=True)

    def get_fitted_params(self) -> dict[str, dict]:
        return {}
