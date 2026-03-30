"""Plotting utilities for time series visualization."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import pandas as pd

from tsbricks.blocks.utils import (
    _DPI,
    convert_to_pandas,
    pixels_to_figsize,
    validate_ax,
    validate_backend,
    validate_column_exists,
    validate_dataframe,
    validate_dimensions,
    validate_no_duplicates,
    validate_no_missing_values,
    validate_not_empty,
    validate_time_col_dtype,
    validate_value_col_dtype,
)

if TYPE_CHECKING:
    import polars as pl

    DataFrameLike: TypeAlias = pd.DataFrame | pl.DataFrame
else:
    DataFrameLike = pd.DataFrame


_NAMED_PERIODS = frozenset({"week", "W", "month", "M", "quarter", "Q", "year", "Y"})

_SUPPORTED_BASE_FREQS = frozenset(
    {
        "W",
        "W-SUN",
        "W-MON",
        "W-TUE",
        "W-WED",
        "W-THU",
        "W-FRI",
        "W-SAT",
        "MS",
        "ME",
        "QS",
        "QE",
        "YS",
        "YE",
        "D",
        "h",
    }
)


# Anchored aliases that pd.infer_freq() can return (e.g. "QE-DEC", "YS-JAN").
# We strip the anchor suffix to map back to the unanchored base form.
_ANCHORED_PREFIXES = ("QS-", "QE-", "YS-", "YE-", "W-")


def _normalize_freq(freq: str) -> str:
    """Normalize an anchored frequency alias to its unanchored base form.

    For example, ``'QE-DEC'`` becomes ``'QE'`` and ``'YS-JAN'`` becomes
    ``'YS'``.  Weekly anchored forms like ``'W-MON'`` are returned as-is
    because ``_SUPPORTED_BASE_FREQS`` already includes them.
    """
    for prefix in _ANCHORED_PREFIXES:
        if freq.startswith(prefix):
            # Weekly anchored forms are already supported directly
            if prefix == "W-":
                return freq
            return prefix.rstrip("-")
    return freq


def _is_datetime_time_col(df: pd.DataFrame, time_col: str) -> bool:
    """Check if time_col is datetime-like (including object-dtype datetimes)."""
    time_dtype = df[time_col].dtype
    if pd.api.types.is_datetime64_any_dtype(time_dtype):
        return True
    if pd.api.types.is_integer_dtype(time_dtype):
        return False
    # Object column — check via inference
    inferred = pd.api.types.infer_dtype(df[time_col], skipna=True)
    return inferred in ("datetime", "datetime64", "date")


def _validate_seasonal_inputs(
    df: DataFrameLike,
    time_col: str,
    value_col: str,
    period: str | int | None,
    backend: str,
    width: int,
    height: int,
    alpha: float,
    palette: str | list,
    base_freq: str | None,
    ax: object | None = None,
    season_col: str | None = None,
) -> None:
    """Validate inputs for plot_seasonal.

    Checks DataFrame type, column existence, dtypes, missing values,
    duplicates, period, base_freq, alpha, palette, backend, dimensions,
    ax, and season_col.
    Converts Polars to pandas internally before column-level checks.
    """
    # --- DataFrame ---
    validate_dataframe(df)
    pdf = convert_to_pandas(df)

    validate_not_empty(pdf)

    # --- Columns ---
    for col_name, label in [(time_col, "time_col"), (value_col, "value_col")]:
        validate_column_exists(pdf, col_name, label)

    validate_time_col_dtype(pdf, time_col)
    validate_value_col_dtype(pdf, value_col)

    # --- Missing time_col values (error) ---
    validate_no_missing_values(pdf, time_col, "time_col")

    # --- Duplicate time_col values (error) ---
    validate_no_duplicates(pdf, time_col, "time_col")

    # --- season_col / period mutual exclusivity ---
    if season_col is not None and period is not None:
        raise ValueError(
            "season_col and period are mutually exclusive. "
            "Provide one or the other, not both."
        )

    if season_col is None and period is None:
        raise ValueError("Either period or season_col must be provided.")

    # --- season_col ---
    if season_col is not None:
        validate_column_exists(pdf, season_col, "season_col")

    # --- period ---
    if period is not None:
        is_datetime = _is_datetime_time_col(pdf, time_col)

        if isinstance(period, str):
            if not is_datetime:
                raise ValueError(
                    f"Named period '{period}' is only allowed for datetime-like "
                    f"time_col, but time_col '{time_col}' is integer dtype."
                )
            if period not in _NAMED_PERIODS:
                raise ValueError(
                    f"Invalid named period '{period}'. "
                    f"Must be one of {sorted(_NAMED_PERIODS)}."
                )
        elif isinstance(period, (int, np.integer)) and not isinstance(period, bool):
            if period < 2:
                raise ValueError(f"Integer period must be >= 2, got {period}.")
        else:
            raise TypeError(
                f"period must be a string or integer, got {type(period).__name__}."
            )

        # --- base_freq ---
        _validate_base_freq(pdf, time_col, period, base_freq)

    # --- alpha ---
    if not (0 <= alpha <= 1):
        raise ValueError(f"alpha must be between 0 and 1 inclusive, got {alpha}.")

    # --- palette ---
    if not isinstance(palette, (str, list)):
        raise TypeError(
            f"palette must be a string or list, got {type(palette).__name__}."
        )

    # --- backend / dimensions / ax ---
    validate_backend(backend)
    validate_dimensions(width, height)
    validate_ax(ax, backend)


def _validate_base_freq(
    df: pd.DataFrame,
    time_col: str,
    period: str | int,
    base_freq: str | None,
) -> None:
    """Validate base_freq parameter.

    base_freq is only valid when time_col is datetime-like and period is
    an integer. When not provided, attempts inference via pd.infer_freq().
    """
    is_datetime = _is_datetime_time_col(df, time_col)
    is_int_period = isinstance(period, (int, np.integer)) and not isinstance(
        period, bool
    )

    # base_freq only valid for datetime time_col + integer period
    if base_freq is not None:
        if not is_datetime:
            raise ValueError(
                "base_freq is only valid when time_col is datetime-like, "
                f"but time_col '{time_col}' is integer dtype."
            )
        if not is_int_period:
            raise ValueError(
                "base_freq is only valid when period is an integer, "
                f"but period='{period}' is a named period."
            )
        if base_freq not in _SUPPORTED_BASE_FREQS:
            raise ValueError(
                f"Unsupported base_freq '{base_freq}'. "
                f"Must be one of {sorted(_SUPPORTED_BASE_FREQS)}."
            )
        return

    # Inference needed only for datetime time_col + integer period
    if not (is_datetime and is_int_period):
        return

    # Attempt inference
    sorted_times = df[time_col].sort_values()
    inferred = pd.infer_freq(sorted_times)

    if inferred is None:
        raise ValueError(
            "Could not infer frequency from time_col. "
            "Please provide base_freq explicitly. "
            f"Supported values: {sorted(_SUPPORTED_BASE_FREQS)}."
        )

    normalized = _normalize_freq(inferred)
    if normalized not in _SUPPORTED_BASE_FREQS:
        raise ValueError(
            f"Inferred frequency '{inferred}' is not supported. "
            "Please provide base_freq explicitly. "
            f"Supported values: {sorted(_SUPPORTED_BASE_FREQS)}."
        )


# =====================================================================
# Mapping from offset aliases to Period frequencies
# =====================================================================

_BASE_FREQ_TO_PERIOD_FREQ = {
    "D": "D",
    "h": "h",
    "W": "W-SUN",
    "W-SUN": "W-SUN",
    "W-MON": "W-MON",
    "W-TUE": "W-TUE",
    "W-WED": "W-WED",
    "W-THU": "W-THU",
    "W-FRI": "W-FRI",
    "W-SAT": "W-SAT",
    "MS": "M",
    "ME": "M",
    "QS": "Q",
    "QE": "Q",
    "YS": "Y",
    "YE": "Y",
}


# =====================================================================
# Season assignment helpers
# =====================================================================


def _resolve_base_freq(
    df: pd.DataFrame,
    time_col: str,
    base_freq: str | None,
) -> str:
    """Return provided base_freq or infer from data.

    Assumes validation has already confirmed that inference will
    succeed and produce a supported value.
    """
    if base_freq is not None:
        return base_freq
    sorted_times = df[time_col].sort_values()
    return _normalize_freq(pd.infer_freq(sorted_times))


def _assign_calendar_seasons(
    df: pd.DataFrame,
    time_col: str,
    period: str,
) -> pd.DataFrame:
    """Assign _season_id and _position via calendar-aligned grouping.

    Used when time_col is datetime-like and period is a named string.
    """
    pdf = df.copy()
    times = pdf[time_col]

    if period in ("year", "Y"):
        pdf["_season_id"] = times.dt.year.astype(str)
    elif period in ("quarter", "Q"):
        pdf["_season_id"] = (
            times.dt.year.astype(str) + "-Q" + times.dt.quarter.astype(str)
        )
    elif period in ("month", "M"):
        pdf["_season_id"] = times.dt.to_period("M").astype(str)
    elif period in ("week", "W"):
        # Sunday-start weeks: find the Sunday that begins each week
        offset_days = (times.dt.dayofweek + 1) % 7
        week_start = times - pd.to_timedelta(offset_days, unit="D")
        pdf["_season_id"] = week_start.dt.strftime("%Y-%m-%d")

    pdf["_position"] = pdf.groupby("_season_id", sort=False).cumcount() + 1
    return pdf


def _assign_positional_seasons(
    df: pd.DataFrame,
    period: int,
) -> pd.DataFrame:
    """Assign _season_id and _position via positional grouping.

    Used when time_col is integer dtype and period is an integer.
    Every ``period`` consecutive observations form one season.
    """
    pdf = df.copy()
    indices = np.arange(len(pdf))
    pdf["_season_id"] = (indices // period).astype(str)
    pdf["_position"] = (indices % period) + 1
    return pdf


def _assign_custom_seasons(
    df: pd.DataFrame,
    season_col: str,
) -> pd.DataFrame:
    """Assign _season_id and _position via a user-provided season column.

    Used when the caller supplies ``season_col`` to explicitly define
    season boundaries (e.g. fiscal year).  Position is derived from
    row order within each season group via ``cumcount()``, so seasons
    are aligned by position, not by calendar date.  If an early season
    is incomplete (e.g. missing its first observation), the caller
    should pad it with a NaN-valued row so that positions align with
    complete seasons.
    """
    pdf = df.copy()
    if pdf[season_col].isna().any():
        raise ValueError(
            f"season_col '{season_col}' contains missing values. "
            "All rows must have a valid season identifier."
        )
    pdf["_season_id"] = pdf[season_col].astype(str)
    pdf["_position"] = pdf.groupby("_season_id", sort=False).cumcount() + 1
    return pdf


def _assign_frequency_seasons(
    df: pd.DataFrame,
    time_col: str,
    period: int,
    base_freq: str,
) -> pd.DataFrame:
    """Assign _season_id and _position via frequency-aligned grouping.

    Used when time_col is datetime-like and period is an integer.
    ``base_freq`` defines the step size; every ``period`` steps form
    one season.
    """
    pdf = df.copy()
    period_freq = _BASE_FREQ_TO_PERIOD_FREQ[base_freq]
    period_index = pd.PeriodIndex(pdf[time_col], freq=period_freq)
    ordinals = period_index.asi8
    step_indices = ordinals - ordinals[0]

    season_indices = step_indices // period
    pdf["_position"] = (step_indices % period) + 1

    # Label each season by its earliest observation date
    pdf["_season_idx"] = season_indices
    min_dates = pdf.groupby("_season_idx")[time_col].transform("first")
    pdf["_season_id"] = min_dates.dt.strftime("%Y-%m-%d")
    pdf.drop(columns=["_season_idx"], inplace=True)

    return pdf


# =====================================================================
# Data sufficiency
# =====================================================================


def _check_data_sufficiency(
    df: pd.DataFrame,
    period: str | int,
) -> None:
    """Validate that there is enough data for a seasonal plot.

    Raises ValueError if less than one full season. Emits a warning
    if there is exactly one full season.
    """
    if isinstance(period, str):
        # Group-based rule (datetime + named period)
        n_seasons = df["_season_id"].nunique()
        if n_seasons < 2:
            raise ValueError(
                "Not enough data for a seasonal plot. "
                "Need at least 2 seasons, got "
                f"{n_seasons}."
            )
        if n_seasons == 2:
            warnings.warn(
                "Only 2 distinct seasons found. Consider "
                "using more data for a meaningful seasonal "
                "comparison.",
                stacklevel=3,
            )
    else:
        # Count-based rule (integer period)
        n = len(df)
        int_period = int(period)
        if n < int_period:
            raise ValueError(
                "Not enough data for a seasonal plot. "
                f"Need at least {int_period} observations "
                f"(one full season), got {n}."
            )
        if n < 2 * int_period:
            warnings.warn(
                f"Only {n} observations for "
                f"period={int_period}. This gives at most "
                "one full season. Consider using more data "
                "for a meaningful seasonal comparison.",
                stacklevel=3,
            )


# =====================================================================
# Main computation entry point
# =====================================================================


def _compute_seasonal_data(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    period: str | int | None,
    base_freq: str | None,
    season_col: str | None = None,
) -> pd.DataFrame:
    """Compute season identity and within-season positions.

    Sorts by ``time_col``, assigns ``_season_id`` (str) and
    ``_position`` (1-based int) columns, checks data sufficiency,
    and warns about missing ``value_col`` values.

    Returns:
        A copy of *df* sorted by *time_col* with ``_season_id``
        and ``_position`` columns appended.
    """
    pdf = df.sort_values(time_col).reset_index(drop=True)

    # Warn about missing value_col values
    na_count = pdf[value_col].isna().sum()
    if na_count > 0:
        warnings.warn(
            f"value_col '{value_col}' contains "
            f"{int(na_count)} missing value(s). "
            "These will appear as gaps in the plot.",
            stacklevel=2,
        )

    if season_col is not None:
        result = _assign_custom_seasons(pdf, season_col)
        # Infer period from max group size for sufficiency check
        inferred_period = result.groupby("_season_id", sort=False).size().max()
        _check_data_sufficiency(result, int(inferred_period))
    else:
        is_datetime = _is_datetime_time_col(pdf, time_col)

        if isinstance(period, str):
            result = _assign_calendar_seasons(pdf, time_col, period)
        elif is_datetime:
            resolved = _resolve_base_freq(pdf, time_col, base_freq)
            result = _assign_frequency_seasons(pdf, time_col, period, resolved)
        else:
            result = _assign_positional_seasons(pdf, period)
            # Warn if total rows aren't evenly divisible by period.
            # Positional grouping naively chunks rows into groups of
            # ``period``, so uneven division means the last season is
            # short. This *may* indicate misaligned boundaries (e.g.
            # first year of YYYYWW data starting at week 02), but
            # positional grouping can't detect which season is off.
            n_rows = len(result)
            int_period = int(period)
            if n_rows >= int_period and n_rows % int_period != 0:
                last_season_size = n_rows % int_period
                warnings.warn(
                    f"Last season has {last_season_size} observations "
                    f"(expected {int_period}). Season boundaries "
                    "may be misaligned. Consider using season_col "
                    "to define seasons explicitly.",
                    stacklevel=2,
                )

        _check_data_sufficiency(result, period)

    # When time_col is datetime, map each _position to a representative
    # date so renderers can use date-based x-axes.  We pick dates from
    # the longest season so that every position gets a date from a single
    # continuous season (avoids ~1-year jumps when the first season is
    # incomplete, e.g. fiscal year starting mid-cycle).
    if _is_datetime_time_col(result, time_col):
        season_sizes = result.groupby("_season_id", sort=False).size()
        longest_season = season_sizes.idxmax()
        longest = result.loc[result["_season_id"] == longest_season]
        pos_to_date = longest.groupby("_position", sort=False)[time_col].first()
        result["_tick_date"] = result["_position"].map(pos_to_date)

    return result


# =====================================================================
# Palette / color sampling
# =====================================================================

_VIRIDIS_UPPER = 0.92  # exclude lightest 8%

_LINEWIDTH = 1.5
_MARKER_SIZE_PLOTLY = 5
_MARKER_SIZE_MPL = 12  # matplotlib uses area (s), so 12 ≈ ~3.5 px diameter


def _sample_colors(
    palette: str | list,
    n: int,
    backend: str,
) -> list[str]:
    """Return *n* colors from *palette* using the given backend.

    If *palette* is a list, validate length and return it directly.
    If *palette* is a named colormap string, sample *n* evenly-spaced
    colors using the backend's native colormap support.  For viridis,
    the lightest 8% of the range is excluded; all other colormaps use
    the full range.

    Returns rgb() strings for plotly, hex strings for matplotlib.
    """
    if isinstance(palette, list):
        if len(palette) < n:
            raise ValueError(
                f"palette list has {len(palette)} color(s), "
                f"but {n} season(s) require at least {n}."
            )
        return palette[:n]

    upper = _VIRIDIS_UPPER if palette == "viridis" else 1.0
    positions = np.linspace(0.0, upper, n).tolist()

    if backend == "plotly":
        return _sample_colors_plotly(palette, positions)
    return _sample_colors_matplotlib(palette, positions)


def _sample_colors_plotly(palette: str, positions: list[float]) -> list[str]:
    """Sample colors from a Plotly colorscale."""
    import plotly.colors as pc
    from _plotly_utils.exceptions import PlotlyError

    try:
        return pc.sample_colorscale(palette, positions, colortype="rgb")
    except (ValueError, PlotlyError):
        raise ValueError(
            f"Unknown colorscale '{palette}'. "
            "Pass a valid plotly colorscale name or a list of colors."
        ) from None


def _sample_colors_matplotlib(palette: str, positions: list[float]) -> list[str]:
    """Sample colors from a Matplotlib colormap."""
    import matplotlib as mpl
    import matplotlib.colors as mcolors

    try:
        cmap = mpl.colormaps[palette]
    except KeyError:
        raise ValueError(
            f"Unknown colormap '{palette}'. "
            "Pass a valid matplotlib colormap name or a list of colors."
        ) from None

    return [mcolors.to_hex(cmap(p)) for p in positions]


# =====================================================================
# Plotly backend
# =====================================================================


def _plot_seasonal_plotly(
    data: pd.DataFrame,
    time_col: str,
    value_col: str,
    colors: list[str],
    alpha: float,
    width: int,
    height: int,
) -> object:
    """Build a seasonal line plot with Plotly.

    *data* must contain ``_season_id`` and ``_position`` columns.
    *colors* must have one entry per unique season, in chronological order.
    """
    import plotly.graph_objects as go

    seasons = list(dict.fromkeys(data["_season_id"]))  # unique, preserving order
    use_dates = "_tick_date" in data.columns
    x_col = "_tick_date" if use_dates else "_position"

    fig = go.Figure()
    for season, color in zip(seasons, colors):
        mask = data["_season_id"] == season
        trace_kwargs: dict = dict(
            x=data.loc[mask, x_col],
            y=data.loc[mask, value_col],
            mode="lines+markers",
            name=season,
            line=dict(color=color, width=_LINEWIDTH),
            marker=dict(color=color, size=_MARKER_SIZE_PLOTLY),
            opacity=alpha,
            connectgaps=False,
        )
        if use_dates:
            trace_kwargs["customdata"] = data.loc[mask, time_col].values
            trace_kwargs["hovertemplate"] = (
                "%{customdata|%Y-%m-%d}<br>%{y}<extra>%{fullData.name}</extra>"
            )
        fig.add_trace(go.Scatter(**trace_kwargs))

    _grid_color = "rgba(220,220,220,0.25)"
    fig.update_layout(
        width=width,
        height=height,
        xaxis_title=f"time ({time_col})",
        yaxis_title=value_col,
        title=None,
        showlegend=True,
        legend=dict(x=1.01, y=1, xanchor="left", yanchor="top"),
        margin=dict(l=60, r=120, t=30, b=60),
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=True,
            gridcolor=_grid_color,
            showline=True,
            linecolor="black",
            mirror=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=_grid_color,
            showline=True,
            linecolor="black",
            mirror=False,
        ),
    )

    return fig


# =====================================================================
# Matplotlib backend
# =====================================================================


def _plot_seasonal_matplotlib(
    data: pd.DataFrame,
    time_col: str,
    value_col: str,
    colors: list[str],
    alpha: float,
    width: int,
    height: int,
    ax: object | None = None,
) -> object:
    """Build a seasonal line plot with Matplotlib.

    *data* must contain ``_season_id`` and ``_position`` columns.
    *colors* must have one entry per unique season, in chronological order.

    When *ax* is provided, draws on the given axes without creating a
    new figure or calling ``tight_layout``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        figsize = pixels_to_figsize(width, height)
        fig, ax = plt.subplots(figsize=figsize, dpi=_DPI)
        owns_figure = True
    else:
        fig = ax.figure
        owns_figure = False

    seasons = list(dict.fromkeys(data["_season_id"]))
    use_dates = "_tick_date" in data.columns

    for season, color in zip(seasons, colors):
        mask = data["_season_id"] == season
        x_col = "_tick_date" if use_dates else "_position"
        ax.plot(
            data.loc[mask, x_col].values,
            data.loc[mask, value_col].values,
            marker="o",
            markersize=3.5,
            linewidth=_LINEWIDTH,
            color=color,
            alpha=alpha,
            label=season,
        )

    ax.set_xlabel(f"time ({time_col})")
    ax.set_ylabel(value_col)

    if use_dates:
        fig.autofmt_xdate()

    ax.grid(True, color=(0.86, 0.86, 0.86, 0.25), linewidth=0.5)
    ax.set_axisbelow(True)

    # Right-side legend outside plot
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        frameon=False,
    )

    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if owns_figure:
        fig.tight_layout()
    return fig


# =====================================================================
# Public API
# =====================================================================


def plot_seasonal(
    df: DataFrameLike,
    time_col: str,
    value_col: str,
    period: str | int | None = None,
    backend: str = "plotly",
    width: int = 800,
    height: int = 450,
    palette: str | list = "viridis",
    base_freq: str | None = None,
    return_fig: bool = False,
    alpha: float = 0.8,
    ax: object | None = None,
    season_col: str | None = None,
) -> object | None:
    """Plot a seasonal decomposition of a single time series.

    Each line represents one season, with the x-axis showing
    within-season observation position (1, 2, 3, ...).

    Args:
        df: pandas or Polars DataFrame containing the time series.
        time_col: Column name for the time axis (datetime-like or integer).
        value_col: Column name for the values (numeric).
        period: Season length. Named string (``"year"``, ``"quarter"``,
            ``"month"``, ``"week"`` or aliases ``"Y"``, ``"Q"``, ``"M"``,
            ``"W"``) for calendar-aligned grouping, or an integer >= 2.
            Mutually exclusive with *season_col*.
        backend: Plotting backend — ``"plotly"`` or ``"matplotlib"``.
        width: Figure width in pixels. Ignored when *ax* is provided.
        height: Figure height in pixels. Ignored when *ax* is provided.
        palette: Named colormap string or list of colors.
        base_freq: Pandas frequency alias. Required when *time_col* is
            datetime-like and *period* is an integer; inferred if omitted.
        return_fig: If True, return the native figure object after
            rendering. Ignored when *ax* is provided (always returns
            the parent figure).
        alpha: Opacity for lines and markers (0–1).
        ax: Optional matplotlib Axes to draw on. When provided,
            *width*, *height*, and *return_fig* are ignored; the
            function draws on the given axes and returns its parent
            figure. Only valid with ``backend="matplotlib"``.
        season_col: Column name that explicitly defines season
            boundaries (e.g. ``"fiscal_year"``). When provided,
            *period* is inferred from the largest season group.
            Mutually exclusive with *period*.

            Positions are assigned by row order within each season
            (1, 2, 3, ...), so all seasons must start at the same
            logical point.  If the first season is incomplete (e.g.
            a fiscal year starting at week 2), pad it with a row
            whose *value_col* is ``NaN`` so that positions align
            across seasons.  The missing value will appear as a gap
            in the plot.

    Returns:
        The native figure object if *return_fig* is True or *ax* is
        provided, otherwise None.
    """
    _validate_seasonal_inputs(
        df,
        time_col,
        value_col,
        period,
        backend,
        width,
        height,
        alpha,
        palette,
        base_freq,
        ax,
        season_col,
    )

    pdf = convert_to_pandas(df)
    data = _compute_seasonal_data(
        pdf, time_col, value_col, period, base_freq, season_col
    )

    n_seasons = data["_season_id"].nunique()
    colors = _sample_colors(palette, n_seasons, backend)

    if backend == "plotly":
        fig = _plot_seasonal_plotly(
            data, time_col, value_col, colors, alpha, width, height
        )
        if return_fig:
            return fig
        fig.show()
    else:
        fig = _plot_seasonal_matplotlib(
            data,
            time_col,
            value_col,
            colors,
            alpha,
            width,
            height,
            ax,
        )
        if ax is not None:
            return fig
        if return_fig:
            return fig
        import matplotlib.pyplot as plt

        plt.show()
    return None
