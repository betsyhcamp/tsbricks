from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

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


# ---- fit_transforms ----


def test_fit_single_boxcox_transform(panel_df: pd.DataFrame) -> None:
    """fit_transforms with one BoxCox config returns fitted transform and changed data."""
    cfg = _TransformConfig(
        class_path="tsbricks.blocks.transforms.BoxCoxTransform",
        params={"method": "loglik"},
    )
    fitted, transformed_df = fit_transforms(panel_df, [cfg])

    assert len(fitted) == 1
    assert set(transformed_df.columns) == set(panel_df.columns)
    # Values should have changed (transformed)
    assert not np.allclose(transformed_df["y"].to_numpy(), panel_df["y"].to_numpy())


def test_fit_empty_transform_list(panel_df: pd.DataFrame) -> None:
    """Empty config list returns original DataFrame unchanged."""
    fitted, result_df = fit_transforms(panel_df, [])

    assert len(fitted) == 0
    pd.testing.assert_frame_equal(result_df, panel_df)


def test_fit_perform_inverse_flag_stored(panel_df: pd.DataFrame) -> None:
    """fit_transforms stores _perform_inverse on each transform instance."""
    cfg = _TransformConfig(
        class_path="tsbricks.blocks.transforms.BoxCoxTransform",
        perform_inverse_transform=False,
        params={"method": "loglik"},
    )
    fitted, _ = fit_transforms(panel_df, [cfg])

    assert fitted[0]._perform_inverse is False  # type: ignore[attr-defined]


# ---- apply_transforms ----


def test_apply_matches_fit(panel_df: pd.DataFrame) -> None:
    """apply_transforms on the same data produces the same result as fit_transform."""
    cfg = _TransformConfig(
        class_path="tsbricks.blocks.transforms.BoxCoxTransform",
        params={"method": "loglik"},
    )
    fitted, transformed_via_fit = fit_transforms(panel_df, [cfg])
    transformed_via_apply = apply_transforms(panel_df, fitted)

    np.testing.assert_allclose(
        transformed_via_apply["y"].to_numpy(),
        transformed_via_fit["y"].to_numpy(),
        rtol=1e-12,
    )


def test_apply_empty_transform_list(panel_df: pd.DataFrame) -> None:
    """Empty fitted list returns DataFrame unchanged."""
    result = apply_transforms(panel_df, [])
    pd.testing.assert_frame_equal(result, panel_df)


# ---- inverse_transforms ----


def test_inverse_round_trip(panel_df: pd.DataFrame) -> None:
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
        recovered["ypred"].to_numpy(),
        panel_df["y"].to_numpy(),
        rtol=1e-10,
    )


def test_inverse_skips_when_flag_is_false(panel_df: pd.DataFrame) -> None:
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
        result["ypred"].to_numpy(),
        transformed_df["y"].to_numpy(),
        rtol=1e-12,
    )


def test_inverse_empty_transform_list(panel_df: pd.DataFrame) -> None:
    """Empty fitted list returns forecast DataFrame unchanged."""
    forecast_df = panel_df.rename(columns={"y": "ypred"})
    result = inverse_transforms(forecast_df, [])
    pd.testing.assert_frame_equal(result, forecast_df)
