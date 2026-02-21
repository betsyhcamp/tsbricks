from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from tsbricks.runner.transform_pipeline import (
    apply_transforms,
    fit_transforms,
    inverse_transforms,
)


# ---- Stand-in config (replaced by Pydantic model in Phase 5) ----


@dataclass
class _TransformConfig:
    """Minimal stand-in for TransformConfig used in tests."""

    class_path: str
    perform_inverse_transform: bool = True
    params: dict | None = field(default=None)


# ---- Fixtures ----


@pytest.fixture
def panel_df() -> pd.DataFrame:
    """Synthetic panel with 2 series, strictly positive values."""
    n = 12
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    df_a = pd.DataFrame(
        {
            "unique_id": "A",
            "ds": dates,
            "y": np.arange(10, 10 + n, dtype=float),
        }
    )
    df_b = pd.DataFrame(
        {
            "unique_id": "B",
            "ds": dates,
            "y": np.arange(1, 1 + n, dtype=float) ** 2 * 50,
        }
    )
    return pd.concat([df_a, df_b], ignore_index=True)


# ---- Tests ----


class TestFitTransforms:
    """Tests for fit_transforms."""

    def test_single_boxcox_transform(self, panel_df: pd.DataFrame) -> None:
        """fit_transforms with one BoxCox config returns fitted transform and changed data."""
        cfg = _TransformConfig(
            class_path="tsbricks.blocks.transforms.BoxCoxTransform",
            params={"method": "loglik"},
        )
        fitted, transformed_df = fit_transforms(panel_df, [cfg])

        assert len(fitted) == 1
        assert set(transformed_df.columns) == set(panel_df.columns)
        # Values should have changed (transformed)
        assert not np.allclose(transformed_df["y"].values, panel_df["y"].values)

    def test_empty_transform_list(self, panel_df: pd.DataFrame) -> None:
        """Empty config list returns original DataFrame unchanged."""
        fitted, result_df = fit_transforms(panel_df, [])

        assert len(fitted) == 0
        pd.testing.assert_frame_equal(result_df, panel_df)

    def test_perform_inverse_flag_stored(self, panel_df: pd.DataFrame) -> None:
        """fit_transforms stores _perform_inverse on each transform instance."""
        cfg = _TransformConfig(
            class_path="tsbricks.blocks.transforms.BoxCoxTransform",
            perform_inverse_transform=False,
            params={"method": "loglik"},
        )
        fitted, _ = fit_transforms(panel_df, [cfg])

        assert fitted[0]._perform_inverse is False  # type: ignore[attr-defined]


class TestApplyTransforms:
    """Tests for apply_transforms."""

    def test_apply_matches_fit(self, panel_df: pd.DataFrame) -> None:
        """apply_transforms on the same data produces the same result as fit_transform."""
        cfg = _TransformConfig(
            class_path="tsbricks.blocks.transforms.BoxCoxTransform",
            params={"method": "loglik"},
        )
        fitted, transformed_via_fit = fit_transforms(panel_df, [cfg])
        transformed_via_apply = apply_transforms(panel_df, fitted)

        np.testing.assert_allclose(
            transformed_via_apply["y"].values,
            transformed_via_fit["y"].values,
            rtol=1e-12,
        )

    def test_empty_transform_list(self, panel_df: pd.DataFrame) -> None:
        """Empty fitted list returns DataFrame unchanged."""
        result = apply_transforms(panel_df, [])
        pd.testing.assert_frame_equal(result, panel_df)


class TestInverseTransforms:
    """Tests for inverse_transforms."""

    def test_round_trip(self, panel_df: pd.DataFrame) -> None:
        """fit → apply → inverse ≈ original values."""
        cfg = _TransformConfig(
            class_path="tsbricks.blocks.transforms.BoxCoxTransform",
            params={"method": "loglik"},
        )
        fitted, transformed_df = fit_transforms(panel_df, [cfg])

        # Simulate forecast output: rename y → ypred
        forecast_df = transformed_df.rename(columns={"y": "ypred"})
        recovered = inverse_transforms(forecast_df, fitted)

        np.testing.assert_allclose(
            recovered["ypred"].values,
            panel_df["y"].values,
            rtol=1e-10,
        )

    def test_skips_when_flag_is_false(self, panel_df: pd.DataFrame) -> None:
        """inverse_transforms skips transforms with _perform_inverse=False."""
        cfg = _TransformConfig(
            class_path="tsbricks.blocks.transforms.BoxCoxTransform",
            perform_inverse_transform=False,
            params={"method": "loglik"},
        )
        fitted, transformed_df = fit_transforms(panel_df, [cfg])

        # Simulate forecast output
        forecast_df = transformed_df.rename(columns={"y": "ypred"})
        result = inverse_transforms(forecast_df, fitted)

        # Should NOT have inverse-transformed — values stay transformed
        np.testing.assert_allclose(
            result["ypred"].values,
            transformed_df["y"].values,
            rtol=1e-12,
        )

    def test_empty_transform_list(self, panel_df: pd.DataFrame) -> None:
        """Empty fitted list returns forecast DataFrame unchanged."""
        forecast_df = panel_df.rename(columns={"y": "ypred"})
        result = inverse_transforms(forecast_df, [])
        pd.testing.assert_frame_equal(result, forecast_df)
