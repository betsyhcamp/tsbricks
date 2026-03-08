from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsbricks.blocks.transforms import BaseTransform, WorkdayNormalizeTransform


# ---- Tests ----


class TestWorkdayNormalizeTransformGlobal:
    """Tests for WorkdayNormalizeTransform with calendar_scope='global'."""

    def test_round_trip_fidelity(
        self, workday_panel_df: pd.DataFrame, global_calendar: pd.DataFrame
    ) -> None:
        """fit_transform → inverse_transform ≈ original values."""
        tx = WorkdayNormalizeTransform()
        transformed = tx.fit_transform(
            workday_panel_df,
            "y",
            calendar_df=global_calendar,
            calendar_scope="global",
        )
        recovered = tx.inverse_transform(transformed, "y")

        np.testing.assert_allclose(
            recovered["y"].values,
            workday_panel_df["y"].values,
            rtol=1e-10,
        )

    def test_forward_transform_divides(
        self, workday_panel_df: pd.DataFrame, global_calendar: pd.DataFrame
    ) -> None:
        """Forward transform divides target by n_workdays."""
        tx = WorkdayNormalizeTransform()
        transformed = tx.fit_transform(
            workday_panel_df,
            "y",
            calendar_df=global_calendar,
            calendar_scope="global",
        )
        # Series A, first month: 1000 / 20 = 50
        series_a = transformed[transformed["unique_id"] == "A"]
        assert series_a.iloc[0]["y"] == pytest.approx(50.0)

    def test_inverse_transform_multiplies(
        self, workday_panel_df: pd.DataFrame, global_calendar: pd.DataFrame
    ) -> None:
        """Inverse transform multiplies target by n_workdays."""
        tx = WorkdayNormalizeTransform()
        transformed = tx.fit_transform(
            workday_panel_df,
            "y",
            calendar_df=global_calendar,
            calendar_scope="global",
        )
        # Create a forecast df with ypred column
        forecast_df = transformed.rename(columns={"y": "ypred"})
        recovered = tx.inverse_transform(forecast_df, "ypred")

        np.testing.assert_allclose(
            recovered["ypred"].values,
            workday_panel_df["y"].values,
            rtol=1e-10,
        )

    def test_transform_matches_fit_transform(
        self, workday_panel_df: pd.DataFrame, global_calendar: pd.DataFrame
    ) -> None:
        """transform() on same data produces same result as fit_transform()."""
        tx = WorkdayNormalizeTransform()
        via_fit = tx.fit_transform(
            workday_panel_df,
            "y",
            calendar_df=global_calendar,
            calendar_scope="global",
        )
        via_transform = tx.transform(workday_panel_df, "y")

        np.testing.assert_allclose(
            via_transform["y"].values,
            via_fit["y"].values,
            rtol=1e-12,
        )

    def test_input_dataframe_not_mutated(
        self, workday_panel_df: pd.DataFrame, global_calendar: pd.DataFrame
    ) -> None:
        """The original DataFrame must not be changed by any method."""
        original_values = workday_panel_df["y"].values.copy()

        tx = WorkdayNormalizeTransform()
        transformed = tx.fit_transform(
            workday_panel_df,
            "y",
            calendar_df=global_calendar,
            calendar_scope="global",
        )
        tx.inverse_transform(transformed, "y")

        np.testing.assert_array_equal(workday_panel_df["y"].values, original_values)

    def test_columns_preserved(
        self, workday_panel_df: pd.DataFrame, global_calendar: pd.DataFrame
    ) -> None:
        """Output has same columns as input — n_workdays is not leaked."""
        tx = WorkdayNormalizeTransform()
        transformed = tx.fit_transform(
            workday_panel_df,
            "y",
            calendar_df=global_calendar,
            calendar_scope="global",
        )
        assert set(transformed.columns) == set(workday_panel_df.columns)

    def test_row_order_preserved(
        self, workday_panel_df: pd.DataFrame, global_calendar: pd.DataFrame
    ) -> None:
        """Output rows are in the same order as input rows."""
        tx = WorkdayNormalizeTransform()
        transformed = tx.fit_transform(
            workday_panel_df,
            "y",
            calendar_df=global_calendar,
            calendar_scope="global",
        )
        pd.testing.assert_series_equal(
            transformed["unique_id"].reset_index(drop=True),
            workday_panel_df["unique_id"].reset_index(drop=True),
        )
        pd.testing.assert_series_equal(
            transformed["ds"].reset_index(drop=True),
            workday_panel_df["ds"].reset_index(drop=True),
        )

    def test_index_is_reset(
        self, workday_panel_df: pd.DataFrame, global_calendar: pd.DataFrame
    ) -> None:
        """Output index is a clean RangeIndex."""
        tx = WorkdayNormalizeTransform()
        transformed = tx.fit_transform(
            workday_panel_df,
            "y",
            calendar_df=global_calendar,
            calendar_scope="global",
        )
        expected_index = pd.RangeIndex(len(transformed))
        pd.testing.assert_index_equal(transformed.index, expected_index)

    def test_is_base_transform_subclass(self) -> None:
        """WorkdayNormalizeTransform must extend BaseTransform."""
        assert issubclass(WorkdayNormalizeTransform, BaseTransform)

    def test_get_fitted_params_returns_empty_dict(
        self, workday_panel_df: pd.DataFrame, global_calendar: pd.DataFrame
    ) -> None:
        """get_fitted_params returns empty dict (no fitted parameters)."""
        tx = WorkdayNormalizeTransform()
        tx.fit_transform(
            workday_panel_df,
            "y",
            calendar_df=global_calendar,
            calendar_scope="global",
        )
        assert tx.get_fitted_params() == {}

    def test_float_workdays_accepted(self, workday_panel_df: pd.DataFrame) -> None:
        """Float n_workdays values are accepted."""
        dates = pd.date_range("2024-01-01", periods=4, freq="MS")
        calendar = pd.DataFrame({"ds": dates, "n_workdays": [20.5, 19.5, 21.5, 22.5]})
        tx = WorkdayNormalizeTransform()
        transformed = tx.fit_transform(
            workday_panel_df, "y", calendar_df=calendar, calendar_scope="global"
        )
        # Series A, first month: 1000 / 20.5
        series_a = transformed[transformed["unique_id"] == "A"]
        assert series_a.iloc[0]["y"] == pytest.approx(1000.0 / 20.5)


class TestWorkdayNormalizeTransformGlobalValidation:
    """Validation error tests for global scope."""

    def test_missing_calendar_df(self, workday_panel_df: pd.DataFrame) -> None:
        tx = WorkdayNormalizeTransform()
        with pytest.raises(ValueError, match="calendar_df is required"):
            tx.fit_transform(workday_panel_df, "y", calendar_scope="global")

    def test_missing_calendar_scope(
        self, workday_panel_df: pd.DataFrame, global_calendar: pd.DataFrame
    ) -> None:
        tx = WorkdayNormalizeTransform()
        with pytest.raises(ValueError, match="calendar_scope is required"):
            tx.fit_transform(workday_panel_df, "y", calendar_df=global_calendar)

    def test_calendar_df_wrong_type(self, workday_panel_df: pd.DataFrame) -> None:
        tx = WorkdayNormalizeTransform()
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            tx.fit_transform(
                workday_panel_df,
                "y",
                calendar_df={"ds": [1], "n_workdays": [20]},
                calendar_scope="global",
            )

    def test_invalid_calendar_scope(
        self, workday_panel_df: pd.DataFrame, global_calendar: pd.DataFrame
    ) -> None:
        tx = WorkdayNormalizeTransform()
        with pytest.raises(ValueError, match="calendar_scope must be one of"):
            tx.fit_transform(
                workday_panel_df,
                "y",
                calendar_df=global_calendar,
                calendar_scope="invalid",
            )

    def test_missing_columns_in_calendar(self, workday_panel_df: pd.DataFrame) -> None:
        bad_cal = pd.DataFrame({"ds": [1, 2], "wrong_col": [20, 21]})
        tx = WorkdayNormalizeTransform()
        with pytest.raises(ValueError, match="missing required columns"):
            tx.fit_transform(
                workday_panel_df,
                "y",
                calendar_df=bad_cal,
                calendar_scope="global",
            )

    def test_non_positive_workdays(self, workday_panel_df: pd.DataFrame) -> None:
        dates = pd.date_range("2024-01-01", periods=4, freq="MS")
        bad_cal = pd.DataFrame({"ds": dates, "n_workdays": [20.0, 0.0, 21.0, 22.0]})
        tx = WorkdayNormalizeTransform()
        with pytest.raises(ValueError, match="must be positive"):
            tx.fit_transform(
                workday_panel_df,
                "y",
                calendar_df=bad_cal,
                calendar_scope="global",
            )

    def test_negative_workdays(self, workday_panel_df: pd.DataFrame) -> None:
        dates = pd.date_range("2024-01-01", periods=4, freq="MS")
        bad_cal = pd.DataFrame({"ds": dates, "n_workdays": [20.0, -1.0, 21.0, 22.0]})
        tx = WorkdayNormalizeTransform()
        with pytest.raises(ValueError, match="must be positive"):
            tx.fit_transform(
                workday_panel_df,
                "y",
                calendar_df=bad_cal,
                calendar_scope="global",
            )

    def test_duplicate_keys_in_calendar(self, workday_panel_df: pd.DataFrame) -> None:
        dates = pd.date_range("2024-01-01", periods=4, freq="MS")
        dup_cal = pd.DataFrame(
            {
                "ds": list(dates) + [dates[0]],
                "n_workdays": [20.0, 19.0, 21.0, 22.0, 18.0],
            }
        )
        tx = WorkdayNormalizeTransform()
        with pytest.raises(ValueError, match="duplicate entries"):
            tx.fit_transform(
                workday_panel_df,
                "y",
                calendar_df=dup_cal,
                calendar_scope="global",
            )

    def test_reserved_column_in_input(self, global_calendar: pd.DataFrame) -> None:
        dates = pd.date_range("2024-01-01", periods=4, freq="MS")
        df_with_reserved = pd.DataFrame(
            {
                "unique_id": "A",
                "ds": dates,
                "y": [100.0, 200.0, 300.0, 400.0],
                "n_workdays": [1, 2, 3, 4],
            }
        )
        tx = WorkdayNormalizeTransform()
        with pytest.raises(ValueError, match="reserved"):
            tx.fit_transform(
                df_with_reserved,
                "y",
                calendar_df=global_calendar,
                calendar_scope="global",
            )

    def test_reserved_column_in_transform(
        self, workday_panel_df: pd.DataFrame, global_calendar: pd.DataFrame
    ) -> None:
        """Reserved column check also applies to transform()."""
        tx = WorkdayNormalizeTransform()
        tx.fit_transform(
            workday_panel_df,
            "y",
            calendar_df=global_calendar,
            calendar_scope="global",
        )
        df_with_reserved = workday_panel_df.copy()
        df_with_reserved["n_workdays"] = 1
        with pytest.raises(ValueError, match="reserved"):
            tx.transform(df_with_reserved, "y")

    def test_reserved_column_in_inverse_transform(
        self, workday_panel_df: pd.DataFrame, global_calendar: pd.DataFrame
    ) -> None:
        """Reserved column check also applies to inverse_transform()."""
        tx = WorkdayNormalizeTransform()
        transformed = tx.fit_transform(
            workday_panel_df,
            "y",
            calendar_df=global_calendar,
            calendar_scope="global",
        )
        transformed_with_reserved = transformed.copy()
        transformed_with_reserved["n_workdays"] = 1
        with pytest.raises(ValueError, match="reserved"):
            tx.inverse_transform(transformed_with_reserved, "y")

    def test_uncovered_rows_raises_with_details(
        self, workday_panel_df: pd.DataFrame
    ) -> None:
        """Missing calendar coverage raises with specific missing keys."""
        # Calendar only covers first 2 of 4 months
        dates = pd.date_range("2024-01-01", periods=2, freq="MS")
        partial_cal = pd.DataFrame({"ds": dates, "n_workdays": [20.0, 19.0]})
        tx = WorkdayNormalizeTransform()
        with pytest.raises(ValueError, match="does not cover") as exc_info:
            tx.fit_transform(
                workday_panel_df,
                "y",
                calendar_df=partial_cal,
                calendar_scope="global",
            )
        error_msg = str(exc_info.value)
        assert "4 row(s)" in error_msg
        assert "2024-03-01" in error_msg or "2024-04-01" in error_msg
