from __future__ import annotations

import numpy as np
import pandas as pd

from tsbricks.blocks.transforms import BaseTransform, BoxCoxTransform


# ---- Tests ----


class TestBoxCoxTransform:
    """Tests for BoxCoxTransform."""

    def test_round_trip_fidelity(self, boxcox_panel_df: pd.DataFrame) -> None:
        """fit_transform → inverse_transform ≈ original values."""
        tx = BoxCoxTransform()
        transformed = tx.fit_transform(boxcox_panel_df, "y", method="loglik")
        recovered = tx.inverse_transform(transformed, "y")

        np.testing.assert_allclose(
            recovered["y"].values,
            boxcox_panel_df["y"].values,
            rtol=1e-10,
        )

    def test_per_series_independent_lambdas(
        self, boxcox_panel_df: pd.DataFrame
    ) -> None:
        """Each unique_id gets its own fitted lambda."""
        tx = BoxCoxTransform()
        tx.fit_transform(boxcox_panel_df, "y", method="loglik")

        params = tx.get_fitted_params()
        assert set(params.keys()) == {"A", "B"}
        assert params["A"]["lambda"] != params["B"]["lambda"]

    def test_get_fitted_params_returns_python_floats(
        self, boxcox_panel_df: pd.DataFrame
    ) -> None:
        """Fitted params must be plain Python float, not numpy scalar."""
        tx = BoxCoxTransform()
        tx.fit_transform(boxcox_panel_df, "y", method="loglik")

        for uid, p in tx.get_fitted_params().items():
            assert type(p["lambda"]) is float, (
                f"Expected float for {uid}, got {type(p['lambda'])}"
            )

    def test_input_dataframe_not_mutated(self, boxcox_panel_df: pd.DataFrame) -> None:
        """The original DataFrame must not be changed by any method."""
        original_values = boxcox_panel_df["y"].values.copy()

        tx = BoxCoxTransform()
        transformed = tx.fit_transform(boxcox_panel_df, "y", method="loglik")
        tx.inverse_transform(transformed, "y")

        np.testing.assert_array_equal(boxcox_panel_df["y"].values, original_values)

    def test_transform_applies_stored_lambdas(
        self, boxcox_panel_df: pd.DataFrame
    ) -> None:
        """transform() on new data uses the previously fitted lambdas."""
        tx = BoxCoxTransform()
        transformed_via_fit = tx.fit_transform(boxcox_panel_df, "y", method="loglik")
        transformed_via_apply = tx.transform(boxcox_panel_df, "y")

        np.testing.assert_allclose(
            transformed_via_apply["y"].values,
            transformed_via_fit["y"].values,
            rtol=1e-12,
        )

    def test_guerrero_method(self, boxcox_panel_df: pd.DataFrame) -> None:
        """Guerrero method path with season_length works."""
        tx = BoxCoxTransform()
        transformed = tx.fit_transform(
            boxcox_panel_df, "y", method="guerrero", season_length=4
        )
        recovered = tx.inverse_transform(transformed, "y")

        np.testing.assert_allclose(
            recovered["y"].values,
            boxcox_panel_df["y"].values,
            rtol=1e-10,
        )

    def test_is_base_transform_subclass(self) -> None:
        """BoxCoxTransform must extend BaseTransform."""
        assert issubclass(BoxCoxTransform, BaseTransform)

    def test_columns_preserved(self, boxcox_panel_df: pd.DataFrame) -> None:
        """All columns in the input appear in the output."""
        tx = BoxCoxTransform()
        transformed = tx.fit_transform(boxcox_panel_df, "y", method="loglik")

        assert set(transformed.columns) == set(boxcox_panel_df.columns)
