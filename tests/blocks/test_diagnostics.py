"""Tests for tsbricks.blocks.diagnostics"""

from __future__ import annotations

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend for CI/testing
except ImportError:
    pass  # matplotlib tests skipped individually via importorskip

import numpy as np
import pandas as pd
import pytest

from tsbricks.blocks.diagnostics import (
    AcfResult,
    ResidualDiagnostics,
    _center_confint,
    _compute_acf,
    _compute_diagnostics,
    _compute_pacf,
    _convert_to_pandas,
    _plot_acf_pacf_matplotlib,
    _plot_acf_pacf_plotly,
    _prepare_series,
    _validate_inputs,
    _validate_acf_pacf_inputs,
    _plot_matplotlib,
    _plot_plotly,
    plot_acf,
    plot_pacf,
    plot_residual_diagnostics,
)


# =====================================================================
# _compute_diagnostics
# =====================================================================


def test_compute_diagnostics_happy_path(diag_df):
    """Computes sorted residuals, ACF, and KDE for valid input."""
    result = _compute_diagnostics(diag_df, "time", "actual", "fitted", nlags=2)
    assert isinstance(result, ResidualDiagnostics)
    assert np.all(result.timestamps == np.array([1, 2, 3, 4, 5]))
    expected_residuals = np.array([0.5, -0.5, 0.5, 0.5, -0.5])
    assert np.allclose(result.residuals, expected_residuals)
    assert len(result.acf_values) == 3  # nlags=2 -> lags 0,1,2
    assert result.conf_interval > 0
    assert len(result.kde_x) == 200
    assert len(result.kde_y) == 200


def test_compute_diagnostics_default_nlags():
    """Uses guarded default nlags=max(1, min(40, n//4)) when None."""
    df = pd.DataFrame(
        {
            "time": range(8),
            "actual": np.arange(8, dtype=float),
            "fitted": np.zeros(8),
        }
    )
    result = _compute_diagnostics(df, "time", "actual", "fitted", nlags=None)
    # n=8, n//4=2, min(40,2)=2, max(1,2)=2 -> 3 values
    assert len(result.acf_values) == 3


def test_compute_diagnostics_confidence_interval(diag_df):
    """Confidence interval equals 1.96 / sqrt(n)."""
    result = _compute_diagnostics(diag_df, "time", "actual", "fitted", nlags=1)
    expected = 1.96 / np.sqrt(len(diag_df))
    assert result.conf_interval == pytest.approx(expected)


def test_compute_diagnostics_zero_variance_raises():
    """Raises ValueError when residuals have near-zero variance."""
    df = pd.DataFrame(
        {
            "time": range(5),
            "actual": [1.0, 1.0, 1.0, 1.0, 1.0],
            "fitted": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="near-zero variance"):
        _compute_diagnostics(df, "time", "actual", "fitted", nlags=1)


def test_compute_diagnostics_explicit_nlags(diag_df):
    """Honors user-supplied nlags value."""
    result = _compute_diagnostics(diag_df, "time", "actual", "fitted", nlags=3)
    assert len(result.acf_values) == 4  # nlags=3 -> lags 0,1,2,3


# =====================================================================
# _convert_to_pandas
# =====================================================================


def test_convert_to_pandas_noop(diag_df):
    """Returns same object when input is already pandas."""
    result = _convert_to_pandas(diag_df)
    assert result is diag_df


def test_convert_to_pandas_from_polars():
    """Converts polars DataFrame to pandas."""
    pl = pytest.importorskip("polars")
    pl_frame = pl.DataFrame(
        {
            "time": [1, 2, 3],
            "actual": [1.0, 2.0, 3.0],
            "fitted": [0.5, 1.5, 2.5],
        }
    )
    result = _convert_to_pandas(pl_frame)
    assert isinstance(result, pd.DataFrame)


# =====================================================================
# _validate_inputs
# =====================================================================


def test_validate_inputs_valid_passes(diag_df):
    """Passes silently for valid DataFrame and parameters."""
    _validate_inputs(diag_df, "time", "actual", "fitted", "plotly", 800, 600)


def test_validate_inputs_too_few_rows():
    """Raises ValueError when DataFrame has fewer than 2 rows."""
    df = pd.DataFrame({"t": [1], "a": [1.0], "f": [1.0]})
    with pytest.raises(ValueError, match="at least 2 rows"):
        _validate_inputs(df, "t", "a", "f", "plotly", 800, 600)


def test_validate_inputs_missing_columns(diag_df):
    """Raises ValueError when required columns are absent."""
    with pytest.raises(ValueError, match="not found"):
        _validate_inputs(
            diag_df,
            "time",
            "actual",
            "missing_col",
            "plotly",
            800,
            600,
        )


def test_validate_inputs_nan_values():
    """Raises ValueError when NaN values present in columns."""
    df = pd.DataFrame(
        {
            "t": [1, 2],
            "a": [1.0, np.nan],
            "f": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="NaN values found"):
        _validate_inputs(df, "t", "a", "f", "plotly", 800, 600)


def test_validate_inputs_invalid_backend(diag_df):
    """Raises ValueError for unsupported backend string."""
    with pytest.raises(ValueError, match="Invalid backend"):
        _validate_inputs(
            diag_df,
            "time",
            "actual",
            "fitted",
            "seaborn",
            800,
            600,
        )


def test_validate_inputs_nonpositive_width(diag_df):
    """Raises ValueError when width is zero or negative."""
    with pytest.raises(ValueError, match="width must be positive"):
        _validate_inputs(
            diag_df,
            "time",
            "actual",
            "fitted",
            "plotly",
            0,
            600,
        )


def test_validate_inputs_nonpositive_height(diag_df):
    """Raises ValueError when height is zero or negative."""
    with pytest.raises(ValueError, match="height must be positive"):
        _validate_inputs(
            diag_df,
            "time",
            "actual",
            "fitted",
            "plotly",
            800,
            -1,
        )


# =====================================================================
# _plot_matplotlib
# =====================================================================


def test_plot_matplotlib_returns_figure_with_axes(
    sample_diag_data,
):
    """Returns matplotlib Figure with four axes."""
    pytest.importorskip("matplotlib")
    import matplotlib.figure as mpl_fig
    import matplotlib.pyplot as plt

    fig = _plot_matplotlib(sample_diag_data, hist_bins=5, width=800, height=600)
    assert isinstance(fig, mpl_fig.Figure)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_matplotlib_figure_dimensions(sample_diag_data):
    """Figure size matches width/height converted through DPI."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    fig = _plot_matplotlib(sample_diag_data, hist_bins=5, width=1000, height=500)
    w, h = fig.get_size_inches()
    # _DPI = 100, so 1000/100=10.0, 500/100=5.0
    assert w == pytest.approx(10.0)
    assert h == pytest.approx(5.0)
    plt.close(fig)


# =====================================================================
# _plot_plotly
# =====================================================================


def test_plot_plotly_returns_figure(sample_diag_data):
    """Returns plotly Figure with data traces."""
    import plotly.graph_objects as go

    fig = _plot_plotly(sample_diag_data, hist_bins=5, width=800, height=600)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


# =====================================================================
# plot_residual_diagnostics
# =====================================================================


def test_plot_diagnostics_plotly_return_fig(diag_df):
    """Returns plotly Figure when return_fig=True."""
    import plotly.graph_objects as go

    fig = plot_residual_diagnostics(
        diag_df,
        "time",
        "actual",
        "fitted",
        backend="plotly",
        return_fig=True,
    )
    assert isinstance(fig, go.Figure)


def test_plot_diagnostics_matplotlib_return_fig(diag_df):
    """Returns matplotlib Figure when backend='matplotlib'."""
    pytest.importorskip("matplotlib")
    import matplotlib.figure as mpl_fig
    import matplotlib.pyplot as plt

    fig = plot_residual_diagnostics(
        diag_df,
        "time",
        "actual",
        "fitted",
        backend="matplotlib",
        return_fig=True,
    )
    assert isinstance(fig, mpl_fig.Figure)
    plt.close(fig)


def test_plot_diagnostics_polars_input():
    """Accepts polars DataFrame and converts internally."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(
        {
            "time": [1, 2, 3, 4, 5],
            "actual": [1.5, 2.0, 3.0, 4.5, 5.0],
            "fitted": [1.0, 2.5, 2.5, 4.0, 5.5],
        }
    )
    fig = plot_residual_diagnostics(df, "time", "actual", "fitted", return_fig=True)
    assert fig is not None


def test_plot_diagnostics_invalid_input_propagates():
    """Propagates validation errors from _validate_inputs."""
    df = pd.DataFrame({"time": [1], "actual": [1.0], "fitted": [1.0]})
    with pytest.raises(ValueError, match="at least 2 rows"):
        plot_residual_diagnostics(df, "time", "actual", "fitted")


def test_plot_diagnostics_return_fig_false_none(diag_df, mocker):
    """Returns None when return_fig=False."""
    mocker.patch("plotly.graph_objects.Figure.show")
    result = plot_residual_diagnostics(
        diag_df,
        "time",
        "actual",
        "fitted",
        backend="plotly",
        return_fig=False,
    )
    assert result is None


# =====================================================================
# _validate_acf_pacf_inputs
# =====================================================================

_DEFAULTS = dict(backend="plotly", width=800, height=450, alpha=0.05, lags=None)


def test_validate_acf_pacf_happy_path_datetime(acf_df_datetime):
    """Passes silently for valid datetime DataFrame."""
    _validate_acf_pacf_inputs(acf_df_datetime, "time", "value", **_DEFAULTS)


def test_validate_acf_pacf_happy_path_integer(acf_df_integer):
    """Passes silently for valid integer-index DataFrame."""
    _validate_acf_pacf_inputs(acf_df_integer, "time", "value", **_DEFAULTS)


def test_validate_acf_pacf_polars_input():
    """Accepts Polars DataFrame."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"time": [1, 2, 3, 4, 5], "value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    _validate_acf_pacf_inputs(df, "time", "value", **_DEFAULTS)


def test_validate_acf_pacf_rejects_non_dataframe():
    """Raises TypeError for non-DataFrame input."""
    with pytest.raises(TypeError, match="pandas or Polars DataFrame"):
        _validate_acf_pacf_inputs(
            {"time": [1, 2], "value": [1.0, 2.0]},
            "time",
            "value",
            **_DEFAULTS,
        )


def test_validate_acf_pacf_empty_df():
    """Raises ValueError for empty DataFrame."""
    df = pd.DataFrame(
        {"time": pd.Series([], dtype="int64"), "value": pd.Series([], dtype="float64")}
    )
    with pytest.raises(ValueError, match="must not be empty"):
        _validate_acf_pacf_inputs(df, "time", "value", **_DEFAULTS)


def test_validate_acf_pacf_single_row():
    """Raises ValueError for single-row DataFrame."""
    df = pd.DataFrame({"time": [1], "value": [1.0]})
    with pytest.raises(ValueError, match="at least 2 rows"):
        _validate_acf_pacf_inputs(df, "time", "value", **_DEFAULTS)


def test_validate_acf_pacf_missing_time_col(acf_df_integer):
    """Raises ValueError when time_col not in DataFrame."""
    with pytest.raises(ValueError, match="time_col 'missing'"):
        _validate_acf_pacf_inputs(acf_df_integer, "missing", "value", **_DEFAULTS)


def test_validate_acf_pacf_missing_value_col(acf_df_integer):
    """Raises ValueError when value_col not in DataFrame."""
    with pytest.raises(ValueError, match="value_col 'missing'"):
        _validate_acf_pacf_inputs(acf_df_integer, "time", "missing", **_DEFAULTS)


def test_validate_acf_pacf_time_col_float_dtype():
    """Raises ValueError when time_col is float."""
    df = pd.DataFrame({"time": [1.0, 2.0, 3.0], "value": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="datetime-like or integer dtype"):
        _validate_acf_pacf_inputs(df, "time", "value", **_DEFAULTS)


def test_validate_acf_pacf_time_col_string_dtype():
    """Raises ValueError when time_col is string."""
    df = pd.DataFrame({"time": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="datetime-like or integer dtype"):
        _validate_acf_pacf_inputs(df, "time", "value", **_DEFAULTS)


def test_validate_acf_pacf_time_col_object_datetime_accepted():
    """Accepts object-dtype column containing Python datetime objects."""
    from datetime import datetime

    df = pd.DataFrame(
        {
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "value": [1.0, 2.0, 3.0],
        }
    )
    # Should not raise
    _validate_acf_pacf_inputs(df, "time", "value", **_DEFAULTS)


def test_validate_acf_pacf_time_col_tz_aware_accepted():
    """Accepts timezone-aware datetime column."""
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03"]
            ).tz_localize("US/Eastern"),
            "value": [1.0, 2.0, 3.0],
        }
    )
    _validate_acf_pacf_inputs(df, "time", "value", **_DEFAULTS)


def test_validate_acf_pacf_time_col_object_string_rejected():
    """Rejects object-dtype column containing strings (not datetimes)."""
    df = pd.DataFrame(
        {
            "time": pd.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype=object),
            "value": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="datetime-like or integer dtype"):
        _validate_acf_pacf_inputs(df, "time", "value", **_DEFAULTS)


def test_validate_acf_pacf_value_col_non_numeric():
    """Raises ValueError when value_col is non-numeric."""
    df = pd.DataFrame({"time": [1, 2, 3], "value": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="must be numeric"):
        _validate_acf_pacf_inputs(df, "time", "value", **_DEFAULTS)


def test_validate_acf_pacf_value_col_complex_rejected():
    """Raises ValueError when value_col is complex dtype."""
    df = pd.DataFrame({"time": [1, 2, 3], "value": [1 + 2j, 3 + 4j, 5 + 6j]})
    with pytest.raises(ValueError, match="must be numeric"):
        _validate_acf_pacf_inputs(df, "time", "value", **_DEFAULTS)


def test_validate_acf_pacf_missing_time_values():
    """Raises ValueError when time_col has NaN."""
    df = pd.DataFrame(
        {
            "time": pd.array([1, pd.NA, 3], dtype="Int64"),
            "value": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="missing value"):
        _validate_acf_pacf_inputs(df, "time", "value", **_DEFAULTS)


def test_validate_acf_pacf_missing_value_values():
    """Raises ValueError when value_col has NaN."""
    df = pd.DataFrame({"time": [1, 2, 3], "value": [1.0, np.nan, 3.0]})
    with pytest.raises(ValueError, match="missing value"):
        _validate_acf_pacf_inputs(df, "time", "value", **_DEFAULTS)


def test_validate_acf_pacf_duplicate_time():
    """Raises ValueError when time_col has duplicates."""
    df = pd.DataFrame({"time": [1, 2, 2, 3], "value": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="duplicate"):
        _validate_acf_pacf_inputs(df, "time", "value", **_DEFAULTS)


def test_validate_acf_pacf_invalid_backend(acf_df_integer):
    """Raises ValueError for unsupported backend."""
    with pytest.raises(ValueError, match="Invalid backend"):
        _validate_acf_pacf_inputs(
            acf_df_integer,
            "time",
            "value",
            backend="seaborn",
            width=800,
            height=450,
            alpha=0.05,
            lags=None,
        )


def test_validate_acf_pacf_width_float_type(acf_df_integer):
    """Raises TypeError when width is a float."""
    with pytest.raises(TypeError, match="width must be a positive integer"):
        _validate_acf_pacf_inputs(
            acf_df_integer,
            "time",
            "value",
            backend="plotly",
            width=800.0,
            height=450,
            alpha=0.05,
            lags=None,
        )


def test_validate_acf_pacf_height_float_type(acf_df_integer):
    """Raises TypeError when height is a float."""
    with pytest.raises(TypeError, match="height must be a positive integer"):
        _validate_acf_pacf_inputs(
            acf_df_integer,
            "time",
            "value",
            backend="plotly",
            width=800,
            height=450.0,
            alpha=0.05,
            lags=None,
        )


def test_validate_acf_pacf_width_nonpositive(acf_df_integer):
    """Raises ValueError when width is zero."""
    with pytest.raises(ValueError, match="width must be positive"):
        _validate_acf_pacf_inputs(
            acf_df_integer,
            "time",
            "value",
            backend="plotly",
            width=0,
            height=450,
            alpha=0.05,
            lags=None,
        )


def test_validate_acf_pacf_height_negative(acf_df_integer):
    """Raises ValueError when height is negative."""
    with pytest.raises(ValueError, match="height must be positive"):
        _validate_acf_pacf_inputs(
            acf_df_integer,
            "time",
            "value",
            backend="plotly",
            width=800,
            height=-1,
            alpha=0.05,
            lags=None,
        )


def test_validate_acf_pacf_alpha_zero(acf_df_integer):
    """Raises ValueError when alpha is 0."""
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        _validate_acf_pacf_inputs(
            acf_df_integer,
            "time",
            "value",
            backend="plotly",
            width=800,
            height=450,
            alpha=0.0,
            lags=None,
        )


def test_validate_acf_pacf_alpha_one(acf_df_integer):
    """Raises ValueError when alpha is 1."""
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        _validate_acf_pacf_inputs(
            acf_df_integer,
            "time",
            "value",
            backend="plotly",
            width=800,
            height=450,
            alpha=1.0,
            lags=None,
        )


def test_validate_acf_pacf_lags_valid_integer(acf_df_integer):
    """Passes for valid positive integer lags."""
    _validate_acf_pacf_inputs(
        acf_df_integer,
        "time",
        "value",
        backend="plotly",
        width=800,
        height=450,
        alpha=0.05,
        lags=3,
    )


def test_validate_acf_pacf_lags_float_rejected(acf_df_integer):
    """Raises TypeError when lags is a float."""
    with pytest.raises(TypeError, match="lags must be a positive integer"):
        _validate_acf_pacf_inputs(
            acf_df_integer,
            "time",
            "value",
            backend="plotly",
            width=800,
            height=450,
            alpha=0.05,
            lags=3.0,
        )


def test_validate_acf_pacf_lags_bool_rejected(acf_df_integer):
    """Raises TypeError when lags is a bool."""
    with pytest.raises(TypeError, match="lags must be a positive integer"):
        _validate_acf_pacf_inputs(
            acf_df_integer,
            "time",
            "value",
            backend="plotly",
            width=800,
            height=450,
            alpha=0.05,
            lags=True,
        )


def test_validate_acf_pacf_lags_zero_rejected(acf_df_integer):
    """Raises ValueError when lags is 0."""
    with pytest.raises(ValueError, match="lags must be >= 1"):
        _validate_acf_pacf_inputs(
            acf_df_integer,
            "time",
            "value",
            backend="plotly",
            width=800,
            height=450,
            alpha=0.05,
            lags=0,
        )


def test_validate_acf_pacf_lags_negative_rejected(acf_df_integer):
    """Raises ValueError when lags is negative."""
    with pytest.raises(ValueError, match="lags must be >= 1"):
        _validate_acf_pacf_inputs(
            acf_df_integer,
            "time",
            "value",
            backend="plotly",
            width=800,
            height=450,
            alpha=0.05,
            lags=-1,
        )


def test_validate_acf_pacf_lags_numpy_integer(acf_df_integer):
    """Accepts numpy integer for lags."""
    _validate_acf_pacf_inputs(
        acf_df_integer,
        "time",
        "value",
        backend="plotly",
        width=800,
        height=450,
        alpha=0.05,
        lags=np.int64(3),
    )


def test_validate_acf_pacf_width_numpy_integer(acf_df_integer):
    """Accepts numpy integer for width."""
    _validate_acf_pacf_inputs(
        acf_df_integer,
        "time",
        "value",
        backend="plotly",
        width=np.int64(800),
        height=450,
        alpha=0.05,
        lags=None,
    )


def test_validate_acf_pacf_width_bool_rejected(acf_df_integer):
    """Raises TypeError when width is a bool."""
    with pytest.raises(TypeError, match="width must be a positive integer"):
        _validate_acf_pacf_inputs(
            acf_df_integer,
            "time",
            "value",
            backend="plotly",
            width=True,
            height=450,
            alpha=0.05,
            lags=None,
        )


# =====================================================================
# _prepare_series
# =====================================================================


def test_prepare_series_sorts_and_extracts(acf_df_integer):
    """Sorts by time_col and returns value_col as 1-D float array."""
    result = _prepare_series(acf_df_integer, "time", "value")
    # acf_df_integer has time=[5,3,1,4,2], value=[1.0,2.5,1.5,3.0,2.0]
    # sorted by time -> [1,2,3,4,5] -> values [1.5, 2.0, 2.5, 3.0, 1.0]
    expected = np.array([1.5, 2.0, 2.5, 3.0, 1.0])
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == float


def test_prepare_series_polars_input():
    """Converts Polars DataFrame to pandas before extraction."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"t": [3, 1, 2], "v": [30.0, 10.0, 20.0]})
    result = _prepare_series(df, "t", "v")
    np.testing.assert_array_equal(result, [10.0, 20.0, 30.0])


# =====================================================================
# _center_confint
# =====================================================================


def test_center_confint_shifts_bounds():
    """Subtracts correlation values to centre CI at zero."""
    values = np.array([1.0, 0.5, -0.2])
    # statsmodels returns confint centred on the value
    confint = np.array(
        [
            [0.8, 1.2],
            [0.3, 0.7],
            [-0.4, 0.0],
        ]
    )
    ci_lower, ci_upper = _center_confint(values, confint)
    # lower = confint[:,0] - values, upper = confint[:,1] - values
    np.testing.assert_allclose(ci_lower, [-0.2, -0.2, -0.2])
    np.testing.assert_allclose(ci_upper, [0.2, 0.2, 0.2])


def test_center_confint_varying_widths():
    """Works with non-uniform CI widths (e.g. Bartlett formula)."""
    values = np.array([1.0, 0.4])
    confint = np.array(
        [
            [0.7, 1.3],  # width 0.6
            [0.1, 0.7],  # width 0.6, different offset
        ]
    )
    ci_lower, ci_upper = _center_confint(values, confint)
    np.testing.assert_allclose(ci_lower, [-0.3, -0.3])
    np.testing.assert_allclose(ci_upper, [0.3, 0.3])


# =====================================================================
# _compute_acf
# =====================================================================

# Shared kwargs for _compute_acf happy-path calls
_ACF_DEFAULTS: dict = dict(
    alpha=0.05,
    adjusted=False,
    fft=True,
    bartlett_confint=True,
    zero=True,
)


def test_compute_acf_returns_acf_result(acf_df_datetime):
    """Returns an AcfResult with matching array lengths."""
    result = _compute_acf(
        acf_df_datetime,
        "time",
        "value",
        lags=2,
        **_ACF_DEFAULTS,
    )
    assert isinstance(result, AcfResult)
    assert len(result.lags) == 3  # lags 0, 1, 2
    assert len(result.values) == 3
    assert len(result.ci_lower) == 3
    assert len(result.ci_upper) == 3


def test_compute_acf_lag_zero_is_one(acf_df_datetime):
    """ACF at lag 0 is always 1.0."""
    result = _compute_acf(
        acf_df_datetime,
        "time",
        "value",
        lags=2,
        **_ACF_DEFAULTS,
    )
    assert result.values[0] == pytest.approx(1.0)


def test_compute_acf_ci_centered_at_zero(acf_df_datetime):
    """CI bounds are centred at zero, not at the ACF value."""
    result = _compute_acf(
        acf_df_datetime,
        "time",
        "value",
        lags=2,
        **_ACF_DEFAULTS,
    )
    # For each lag the midpoint of (ci_lower, ci_upper) should be ~0
    midpoints = (result.ci_lower + result.ci_upper) / 2
    np.testing.assert_allclose(midpoints, 0.0, atol=1e-10)


def test_compute_acf_zero_false_slices(acf_df_datetime):
    """zero=False removes lag 0 from all arrays."""
    result = _compute_acf(
        acf_df_datetime,
        "time",
        "value",
        lags=2,
        zero=False,
        alpha=0.05,
        adjusted=False,
        fft=True,
        bartlett_confint=True,
    )
    assert result.lags[0] == 1
    assert 0 not in result.lags
    assert len(result.values) == 2  # lags 1, 2 only


def test_compute_acf_zero_true_includes_lag_zero(acf_df_datetime):
    """zero=True keeps lag 0."""
    result = _compute_acf(
        acf_df_datetime,
        "time",
        "value",
        lags=2,
        **_ACF_DEFAULTS,
    )
    assert result.lags[0] == 0


def test_compute_acf_lags_none_uses_statsmodels_default(acf_df_datetime):
    """lags=None delegates default nlags to statsmodels."""
    result = _compute_acf(
        acf_df_datetime,
        "time",
        "value",
        lags=None,
        **_ACF_DEFAULTS,
    )
    # statsmodels default nlags for n=5 is min(10*(log10(5)),4) ~ 4
    # Just verify it returns something reasonable (> 0 lags)
    assert len(result.values) > 1


def test_compute_acf_explicit_lags(acf_df_datetime):
    """Explicit integer lags controls number of lags returned."""
    result = _compute_acf(
        acf_df_datetime,
        "time",
        "value",
        lags=1,
        **_ACF_DEFAULTS,
    )
    assert len(result.values) == 2  # lag 0 + lag 1


def test_compute_acf_sorts_unsorted_input(acf_df_integer):
    """Produces identical results regardless of input row order."""
    result_unsorted = _compute_acf(
        acf_df_integer,
        "time",
        "value",
        lags=2,
        **_ACF_DEFAULTS,
    )
    # Build a pre-sorted copy
    df_sorted = acf_df_integer.sort_values("time").reset_index(drop=True)
    result_sorted = _compute_acf(
        df_sorted,
        "time",
        "value",
        lags=2,
        **_ACF_DEFAULTS,
    )
    np.testing.assert_array_equal(result_unsorted.values, result_sorted.values)


def test_compute_acf_adjusted_passthrough(acf_df_datetime):
    """adjusted=True changes ACF values (uses n-k denominator)."""
    result_default = _compute_acf(
        acf_df_datetime,
        "time",
        "value",
        lags=2,
        alpha=0.05,
        adjusted=False,
        fft=True,
        bartlett_confint=True,
        zero=True,
    )
    result_adjusted = _compute_acf(
        acf_df_datetime,
        "time",
        "value",
        lags=2,
        alpha=0.05,
        adjusted=True,
        fft=True,
        bartlett_confint=True,
        zero=True,
    )
    # Lag 0 is always 1.0 for both, but lag 1+ should differ
    assert not np.allclose(result_default.values[1:], result_adjusted.values[1:])


def test_compute_acf_bartlett_false_gives_flat_ci(acf_df_datetime):
    """bartlett_confint=False produces constant CI width across lags."""
    result = _compute_acf(
        acf_df_datetime,
        "time",
        "value",
        lags=3,
        alpha=0.05,
        adjusted=False,
        fft=True,
        bartlett_confint=False,
        zero=True,
    )
    widths = result.ci_upper - result.ci_lower
    # With bartlett_confint=False the CI width should be the same
    np.testing.assert_allclose(widths, widths[0])


# =====================================================================
# _compute_pacf
# =====================================================================

# Shared kwargs for _compute_pacf happy-path calls
_PACF_DEFAULTS: dict = dict(
    alpha=0.05,
    method=None,
    zero=True,
)


def test_compute_pacf_returns_acf_result(acf_df_datetime):
    """Returns an AcfResult with matching array lengths."""
    result = _compute_pacf(
        acf_df_datetime,
        "time",
        "value",
        lags=2,
        **_PACF_DEFAULTS,
    )
    assert isinstance(result, AcfResult)
    assert len(result.lags) == 3
    assert len(result.values) == 3
    assert len(result.ci_lower) == 3
    assert len(result.ci_upper) == 3


def test_compute_pacf_lag_zero_is_one(acf_df_datetime):
    """PACF at lag 0 is always 1.0."""
    result = _compute_pacf(
        acf_df_datetime,
        "time",
        "value",
        lags=2,
        **_PACF_DEFAULTS,
    )
    assert result.values[0] == pytest.approx(1.0)


def test_compute_pacf_ci_centered_at_zero(acf_df_datetime):
    """CI bounds are centred at zero."""
    result = _compute_pacf(
        acf_df_datetime,
        "time",
        "value",
        lags=2,
        **_PACF_DEFAULTS,
    )
    midpoints = (result.ci_lower + result.ci_upper) / 2
    np.testing.assert_allclose(midpoints, 0.0, atol=1e-10)


def test_compute_pacf_zero_false_slices(acf_df_datetime):
    """zero=False removes lag 0 from all arrays."""
    result = _compute_pacf(
        acf_df_datetime,
        "time",
        "value",
        lags=2,
        alpha=0.05,
        method=None,
        zero=False,
    )
    assert result.lags[0] == 1
    assert 0 not in result.lags
    assert len(result.values) == 2


def test_compute_pacf_lags_none_uses_statsmodels_default(acf_df_datetime):
    """lags=None delegates default nlags to statsmodels."""
    result = _compute_pacf(
        acf_df_datetime,
        "time",
        "value",
        lags=None,
        **_PACF_DEFAULTS,
    )
    assert len(result.values) > 1


def test_compute_pacf_explicit_lags(acf_df_datetime):
    """Explicit integer lags controls number of lags returned."""
    result = _compute_pacf(
        acf_df_datetime,
        "time",
        "value",
        lags=1,
        **_PACF_DEFAULTS,
    )
    assert len(result.values) == 2  # lag 0 + lag 1


def test_compute_pacf_method_none_omits_kwarg(acf_df_datetime, mocker):
    """method=None does not pass method to statsmodels pacf."""
    mock_pacf = mocker.patch(
        "tsbricks.blocks.diagnostics._sm_pacf",
    )
    # Set up return: pacf returns (values, confint) when alpha given
    n = 3  # nlags=2 -> 3 values
    mock_pacf.return_value = (
        np.ones(n),
        np.column_stack([np.zeros(n), 2 * np.ones(n)]),
    )
    _compute_pacf(
        acf_df_datetime,
        "time",
        "value",
        lags=2,
        alpha=0.05,
        method=None,
        zero=True,
    )
    call_kwargs = mock_pacf.call_args[1]
    assert "method" not in call_kwargs


def test_compute_pacf_method_explicit_passes_through(
    acf_df_datetime,
    mocker,
):
    """Explicit method string is forwarded to statsmodels pacf."""
    mock_pacf = mocker.patch(
        "tsbricks.blocks.diagnostics._sm_pacf",
    )
    n = 3
    mock_pacf.return_value = (
        np.ones(n),
        np.column_stack([np.zeros(n), 2 * np.ones(n)]),
    )
    _compute_pacf(
        acf_df_datetime,
        "time",
        "value",
        lags=2,
        alpha=0.05,
        method="ywmle",
        zero=True,
    )
    call_kwargs = mock_pacf.call_args[1]
    assert call_kwargs["method"] == "ywmle"


def test_compute_pacf_sorts_unsorted_input(acf_df_integer):
    """Produces identical results regardless of input row order."""
    result_unsorted = _compute_pacf(
        acf_df_integer,
        "time",
        "value",
        lags=2,
        **_PACF_DEFAULTS,
    )
    df_sorted = acf_df_integer.sort_values("time").reset_index(drop=True)
    result_sorted = _compute_pacf(
        df_sorted,
        "time",
        "value",
        lags=2,
        **_PACF_DEFAULTS,
    )
    np.testing.assert_array_equal(
        result_unsorted.values,
        result_sorted.values,
    )


# =====================================================================
# _plot_acf_pacf_matplotlib
# =====================================================================


def test_plot_acf_pacf_matplotlib_returns_figure(acf_result):
    """Returns a matplotlib Figure with a single Axes."""
    pytest.importorskip("matplotlib")
    import matplotlib.figure as mpl_fig
    import matplotlib.pyplot as plt

    fig = _plot_acf_pacf_matplotlib(acf_result, ylabel="acf", width=800, height=450)
    assert isinstance(fig, mpl_fig.Figure)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_plot_acf_pacf_matplotlib_dimensions(acf_result):
    """Figure size matches width/height converted through DPI."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    fig = _plot_acf_pacf_matplotlib(acf_result, ylabel="acf", width=1000, height=500)
    w, h = fig.get_size_inches()
    assert w == pytest.approx(10.0)
    assert h == pytest.approx(5.0)
    plt.close(fig)


def test_plot_acf_pacf_matplotlib_labels(acf_result):
    """Axes labels are 'lag' and the provided ylabel; no title."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    fig = _plot_acf_pacf_matplotlib(acf_result, ylabel="pacf", width=800, height=450)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "lag"
    assert ax.get_ylabel() == "pacf"
    assert ax.get_title() == ""
    plt.close(fig)


def test_plot_acf_pacf_matplotlib_has_zero_line(acf_result):
    """A horizontal line at y=0 is present."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    fig = _plot_acf_pacf_matplotlib(acf_result, ylabel="acf", width=800, height=450)
    ax = fig.axes[0]
    hlines = [line for line in ax.get_lines() if line.get_ydata()[0] == 0]
    assert len(hlines) >= 1
    plt.close(fig)


def test_plot_acf_pacf_matplotlib_spines_black(acf_result):
    """All spines are black."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    fig = _plot_acf_pacf_matplotlib(acf_result, ylabel="acf", width=800, height=450)
    ax = fig.axes[0]
    for spine in ax.spines.values():
        assert spine.get_edgecolor()[:3] == (0.0, 0.0, 0.0)
    plt.close(fig)


# =====================================================================
# _plot_acf_pacf_plotly
# =====================================================================


def test_plot_acf_pacf_plotly_returns_figure(acf_result):
    """Returns a plotly Figure with data traces."""
    import plotly.graph_objects as go

    fig = _plot_acf_pacf_plotly(acf_result, ylabel="acf", width=800, height=450)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_plot_acf_pacf_plotly_dimensions(acf_result):
    """Layout width/height match the requested pixel dimensions."""
    fig = _plot_acf_pacf_plotly(acf_result, ylabel="acf", width=900, height=500)
    assert fig.layout.width == 900
    assert fig.layout.height == 500


def test_plot_acf_pacf_plotly_labels(acf_result):
    """Axis labels are 'lag' and the provided ylabel; no title."""
    fig = _plot_acf_pacf_plotly(acf_result, ylabel="pacf", width=800, height=450)
    assert fig.layout.xaxis.title.text == "lag"
    assert fig.layout.yaxis.title.text == "pacf"
    assert fig.layout.title is None or fig.layout.title.text is None


def test_plot_acf_pacf_plotly_no_legend(acf_result):
    """Legend is hidden."""
    fig = _plot_acf_pacf_plotly(acf_result, ylabel="acf", width=800, height=450)
    assert fig.layout.showlegend is False


def test_plot_acf_pacf_plotly_has_ci_band(acf_result):
    """First trace is a filled scatter for the confidence band."""
    fig = _plot_acf_pacf_plotly(acf_result, ylabel="acf", width=800, height=450)
    ci_trace = fig.data[0]
    assert ci_trace.fill == "toself"


def test_plot_acf_pacf_plotly_stem_count_matches_lags(acf_result):
    """One stem trace per lag (between CI band trace and marker trace)."""
    fig = _plot_acf_pacf_plotly(acf_result, ylabel="acf", width=800, height=450)
    n_lags = len(acf_result.lags)
    # traces: 1 CI band + n_lags stems + 1 markers = n_lags + 2
    assert len(fig.data) == n_lags + 2


# =====================================================================
# plot_acf  (public API)
# =====================================================================


def test_plot_acf_plotly_return_fig(acf_df_datetime):
    """Returns a plotly Figure when return_fig=True."""
    import plotly.graph_objects as go

    fig = plot_acf(
        acf_df_datetime,
        "time",
        "value",
        return_fig=True,
    )
    assert isinstance(fig, go.Figure)


def test_plot_acf_plotly_return_none(acf_df_datetime, mocker):
    """Returns None when return_fig=False (default)."""
    mocker.patch("plotly.graph_objects.Figure.show")
    result = plot_acf(acf_df_datetime, "time", "value")
    assert result is None


def test_plot_acf_matplotlib_return_fig(acf_df_datetime):
    """Returns a matplotlib Figure when backend='matplotlib'."""
    pytest.importorskip("matplotlib")
    import matplotlib.figure as mpl_fig
    import matplotlib.pyplot as plt

    fig = plot_acf(
        acf_df_datetime,
        "time",
        "value",
        backend="matplotlib",
        return_fig=True,
    )
    assert isinstance(fig, mpl_fig.Figure)
    plt.close(fig)


def test_plot_acf_ylabel_is_acf(acf_df_datetime):
    """Plotly figure y-axis label is 'acf'."""
    fig = plot_acf(acf_df_datetime, "time", "value", return_fig=True)
    assert fig.layout.yaxis.title.text == "acf"


def test_plot_acf_zero_false_excludes_lag_zero(acf_df_datetime):
    """When zero=False, lag 0 is not in the plot data."""
    fig = plot_acf(
        acf_df_datetime,
        "time",
        "value",
        zero=False,
        return_fig=True,
    )
    # The marker trace is the last one; its x values should start at 1
    marker_trace = fig.data[-1]
    assert marker_trace.x[0] == 1


def test_plot_acf_passthrough_params(acf_df_datetime):
    """adjusted, fft, bartlett_confint are accepted without error."""
    fig = plot_acf(
        acf_df_datetime,
        "time",
        "value",
        lags=2,
        adjusted=True,
        fft=False,
        bartlett_confint=False,
        return_fig=True,
    )
    assert fig is not None


def test_plot_acf_validation_propagates(acf_df_datetime):
    """Validation errors from _validate_acf_pacf_inputs propagate."""
    with pytest.raises(ValueError, match="backend"):
        plot_acf(acf_df_datetime, "time", "value", backend="seaborn")


def test_plot_acf_integer_time_col(acf_df_integer):
    """Works with integer time_col."""
    fig = plot_acf(acf_df_integer, "time", "value", return_fig=True)
    assert fig is not None


def test_plot_acf_polars_input():
    """Accepts a Polars DataFrame."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(
        {
            "time": [1, 2, 3, 4, 5],
            "value": [1.0, 2.5, 1.5, 3.0, 2.0],
        }
    )
    fig = plot_acf(df, "time", "value", return_fig=True)
    assert fig is not None


# =====================================================================
# plot_pacf  (public API)
# =====================================================================


def test_plot_pacf_plotly_return_fig(acf_df_datetime):
    """Returns a plotly Figure when return_fig=True."""
    import plotly.graph_objects as go

    fig = plot_pacf(
        acf_df_datetime,
        "time",
        "value",
        return_fig=True,
    )
    assert isinstance(fig, go.Figure)


def test_plot_pacf_plotly_return_none(acf_df_datetime, mocker):
    """Returns None when return_fig=False (default)."""
    mocker.patch("plotly.graph_objects.Figure.show")
    result = plot_pacf(acf_df_datetime, "time", "value")
    assert result is None


def test_plot_pacf_matplotlib_return_fig(acf_df_datetime):
    """Returns a matplotlib Figure when backend='matplotlib'."""
    pytest.importorskip("matplotlib")
    import matplotlib.figure as mpl_fig
    import matplotlib.pyplot as plt

    fig = plot_pacf(
        acf_df_datetime,
        "time",
        "value",
        backend="matplotlib",
        return_fig=True,
    )
    assert isinstance(fig, mpl_fig.Figure)
    plt.close(fig)


def test_plot_pacf_ylabel_is_pacf(acf_df_datetime):
    """Plotly figure y-axis label is 'pacf'."""
    fig = plot_pacf(acf_df_datetime, "time", "value", return_fig=True)
    assert fig.layout.yaxis.title.text == "pacf"


def test_plot_pacf_zero_false_excludes_lag_zero(acf_df_datetime):
    """When zero=False, lag 0 is not in the plot data."""
    fig = plot_pacf(
        acf_df_datetime,
        "time",
        "value",
        zero=False,
        return_fig=True,
    )
    marker_trace = fig.data[-1]
    assert marker_trace.x[0] == 1


def test_plot_pacf_method_none_accepted(acf_df_datetime):
    """method=None is accepted (defers to statsmodels default)."""
    fig = plot_pacf(
        acf_df_datetime,
        "time",
        "value",
        method=None,
        return_fig=True,
    )
    assert fig is not None


def test_plot_pacf_method_explicit_accepted(acf_df_datetime):
    """An explicit method string is accepted."""
    fig = plot_pacf(
        acf_df_datetime,
        "time",
        "value",
        method="ywmle",
        return_fig=True,
    )
    assert fig is not None


def test_plot_pacf_validation_propagates(acf_df_datetime):
    """Validation errors from _validate_acf_pacf_inputs propagate."""
    with pytest.raises(ValueError, match="alpha"):
        plot_pacf(acf_df_datetime, "time", "value", alpha=0.0)


def test_plot_pacf_integer_time_col(acf_df_integer):
    """Works with integer time_col."""
    fig = plot_pacf(acf_df_integer, "time", "value", return_fig=True)
    assert fig is not None


def test_plot_pacf_polars_input():
    """Accepts a Polars DataFrame."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(
        {
            "time": [1, 2, 3, 4, 5],
            "value": [1.0, 2.5, 1.5, 3.0, 2.0],
        }
    )
    fig = plot_pacf(df, "time", "value", return_fig=True)
    assert fig is not None
