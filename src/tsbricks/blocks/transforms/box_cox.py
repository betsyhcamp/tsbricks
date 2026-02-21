from __future__ import annotations

import numpy as np
import pandas as pd
from coreforecast.scalers import boxcox, boxcox_lambda, inv_boxcox

from tsbricks.blocks.transforms.base import BaseTransform


class BoxCoxTransform(BaseTransform):
    """Box-Cox transform wrapping ``coreforecast``.

    Fits an independent lambda per ``unique_id`` using
    ``coreforecast.scalers.boxcox_lambda``.  The ``method`` (``"loglik"``
    or ``"guerrero"``) and ``season_length`` parameters are forwarded
    directly to ``coreforecast`` via ``**params``.
    """

    def __init__(self) -> None:
        self._fitted_lambdas: dict[str, float] = {}
        self._params: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit_transform(
        self, df: pd.DataFrame, target_col: str, **params: object
    ) -> pd.DataFrame:
        df = df.copy()
        self._params = dict(params)
        self._fitted_lambdas = {}

        def _fit_and_transform(
            series_df: pd.DataFrame, col: str, uid: str
        ) -> pd.DataFrame:
            values = series_df[col].to_numpy(dtype=float)
            lmbda = boxcox_lambda(
                values,
                method=str(self._params.get("method", "loglik")),
                season_length=self._params.get("season_length"),  # type: ignore[arg-type]
            )
            self._fitted_lambdas[uid] = float(lmbda)
            series_df = series_df.copy()
            series_df[col] = boxcox(values, lmbda)
            return series_df

        return self._map_per_series(df, target_col, _fit_and_transform)

    def transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = df.copy()

        def _apply(
            series_df: pd.DataFrame, col: str, uid: str
        ) -> pd.DataFrame:
            lmbda = self._fitted_lambdas[uid]
            values = series_df[col].to_numpy(dtype=float)
            series_df = series_df.copy()
            series_df[col] = boxcox(values, lmbda)
            return series_df

        return self._map_per_series(df, target_col, _apply)

    def inverse_transform(
        self, df: pd.DataFrame, target_col: str
    ) -> pd.DataFrame:
        df = df.copy()

        def _invert(
            series_df: pd.DataFrame, col: str, uid: str
        ) -> pd.DataFrame:
            lmbda = self._fitted_lambdas[uid]
            values = series_df[col].to_numpy(dtype=float)
            series_df = series_df.copy()
            series_df[col] = inv_boxcox(values, lmbda)
            return series_df

        return self._map_per_series(df, target_col, _invert)

    def get_fitted_params(self) -> dict[str, dict]:
        return {
            uid: {"lambda": float(lmbda)}
            for uid, lmbda in self._fitted_lambdas.items()
        }
