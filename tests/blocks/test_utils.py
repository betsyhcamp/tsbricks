import pytest

import tsbricks.blocks.utils as utils

from tsbricks.blocks.utils import (
    _is_pandas_df,
    _is_polars_df,
    missing_required_columns,
    validate_ax,
)


# ---------- _is_pandas_df ----------
def test_is_pandas_df_true_for_df_instance(pd_df):
    df = pd_df()
    assert _is_pandas_df(df) is True


def test_is_pandas_df_false_for_non_df():
    assert _is_pandas_df(["not", "a", "df"]) is False


def test_is_pandas_df_false_when_pandas_unavailable(monkeypatch, pd_df):
    df = pd_df()
    monkeypatch.setattr(utils, "pd", None)
    assert _is_pandas_df(df) is False


# ---------- _is_polars_df ----------
def test_is_polars_df_true_for_df_instance(pl_df):
    df = pl_df()
    assert _is_polars_df(df) is True


def test_is_polars_df_false_for_non_df():
    assert _is_polars_df({"not": "polars"}) is False


def test_is_polars_df_false_when_polars_unavailable(monkeypatch, pl_df):
    df = pl_df()
    monkeypatch.setattr(utils, "pl", None)
    assert _is_polars_df(df) is False


# -------- check pl vs pd dataframes are not interchangable; cross-tests---------------
def test_is_pandas_df_false_for_polars_df(pl_df):
    polars_df = pl_df()
    assert _is_pandas_df(polars_df) is False


def test_is_polars_df_false_for_pandas_df(pd_df):
    pandas_df = pd_df()
    assert _is_polars_df(pandas_df) is False


# -------- check missing required columns: Happy path & failures; pandas --------------
def test_missing_required_passes_pandas(pd_df):
    df = pd_df()
    missing_required_columns(df)  # no raise


def test_missing_required_columns_raises_valueerror_pandas(pd_df):
    df = pd_df(cols=("unique_id", "ds"))  #  "y" missing
    with pytest.raises(ValueError) as e:
        missing_required_columns(df)
    assert "y" in str(e.value)


def test_missing_required_columns_allows_override_required_col(pd_df):
    df = pd_df(cols=("a", "b"))
    with pytest.raises(ValueError):
        missing_required_columns(df, required=("a", "c"))
    missing_required_columns(df, required=("a", "b"))  # Override, no raise


def test_missing_required_columns_raises_typeerror_for_non_df():
    with pytest.raises(TypeError):
        missing_required_columns({"not": "a df"})
    with pytest.raises(TypeError):
        missing_required_columns(123)


# --------- check missing required columns: Happy path & failures; polars -----
def test_missing_required_passes_polars(pl_df):
    df = pl_df()
    missing_required_columns(df)  # no raise


def test_missing_required_columns_raises_valuerror_polars(pl_df):
    df = pl_df(cols=("unique_id", "ds"))  #  "y" missing
    with pytest.raises(ValueError) as e:
        missing_required_columns(df)
    assert "y" in str(e.value)


# ---------- validate_ax ----------
def test_validate_ax_none_passes():
    validate_ax(None, "plotly")  # no raise
    validate_ax(None, "matplotlib")  # no raise


def test_validate_ax_valid_axes_passes():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    validate_ax(ax, "matplotlib")  # no raise
    plt.close(fig)


def test_validate_ax_raises_valueerror_for_plotly_backend():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="backend='plotly'"):
        validate_ax(ax, "plotly")
    plt.close(fig)


def test_validate_ax_raises_typeerror_for_wrong_type():
    with pytest.raises(TypeError, match="matplotlib.axes.Axes instance"):
        validate_ax("not_an_axes", "matplotlib")


def test_validate_ax_raises_typeerror_for_figure():
    import matplotlib.pyplot as plt

    fig, _ = plt.subplots()
    with pytest.raises(TypeError, match="Figure"):
        validate_ax(fig, "matplotlib")
    plt.close(fig)
