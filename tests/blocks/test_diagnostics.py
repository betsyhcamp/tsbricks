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
    ResidualDiagnostics,
    _compute_diagnostics,
    _convert_to_pandas,
    _validate_inputs,
    _validate_acf_pacf_inputs,
    _plot_matplotlib,
    _plot_plotly,
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


def test_validate_acf_pacf_value_col_non_numeric():
    """Raises ValueError when value_col is non-numeric."""
    df = pd.DataFrame({"time": [1, 2, 3], "value": ["a", "b", "c"]})
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
