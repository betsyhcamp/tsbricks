"""Tests for tsbricks.blocks.dataio"""

from __future__ import annotations

import pytest
import pandas as pd
from jinja2 import Environment, BaseLoader

from tsbricks.blocks.dataio import (
    read_sql,
    replace_params_sql,
    _vars_in_template,
    render_sql_template,
    BigQueryQueryStats,
    query_to_dataframe,
    _check_storage_uri_str,
    _check_gcs_file_stats,
    write_df_to_gcs_parquet,
    write_df_to_gcs_csv,
    write_df_to_gcs,
)


# =====================================================================
# read_sql
# =====================================================================


def test_read_sql_success(tmp_path):
    """Reads SQL file and returns its full contents as a string."""
    p = tmp_path / "query.sql"
    p.write_text("SELECT 1;")
    assert read_sql(p) == "SELECT 1;"


def test_read_sql_missing_file(tmp_path):
    """Raises FileNotFoundError when file does not exist."""
    with pytest.raises(FileNotFoundError, match="SQL file not found"):
        read_sql(tmp_path / "missing.sql")


# =====================================================================
# replace_params_sql
# =====================================================================


def test_replace_params_sql_success():
    """Replaces all placeholders and returns rendered SQL."""
    sql = "SELECT * FROM <<table>> WHERE ds = '<<date>>'"
    result = replace_params_sql(sql, {"table": "sales", "date": "2024-01-01"})
    assert result == "SELECT * FROM sales WHERE ds = '2024-01-01'"


def test_replace_params_sql_missing_placeholder():
    """Raises KeyError when placeholder key not found in SQL."""
    sql = "SELECT * FROM <<table>>"
    with pytest.raises(KeyError, match="not found"):
        replace_params_sql(sql, {"missing_key": "value"})


def test_replace_params_sql_unreplaced_remaining():
    """Raises ValueError when unreplaced placeholders remain."""
    sql = "SELECT * FROM <<table>> WHERE ds = '<<date>>'"
    with pytest.raises(ValueError, match="Placeholders still remain"):
        replace_params_sql(sql, {"table": "sales"})


def test_replace_params_sql_whitespace_tolerance():
    """Replaces placeholders with internal whitespace."""
    sql = "SELECT * FROM << table >> WHERE id = << id >>"
    result = replace_params_sql(sql, {"table": "sales", "id": "42"})
    assert result == "SELECT * FROM sales WHERE id = 42"


# =====================================================================
# _vars_in_template
# =====================================================================


def test_vars_in_template_basic():
    """Extracts top-level Jinja variables from SQL template."""
    env = Environment(loader=BaseLoader())
    sql = "SELECT * FROM {{ schema }}.{{ table }}"
    assert _vars_in_template(env, sql) == {"schema", "table"}


def test_vars_in_template_excludes_builtins():
    """Excludes Jinja builtins like loop from detected variables."""
    env = Environment(loader=BaseLoader())
    sql = "{% for x in items %}{{ loop.index }}{% endfor %}"
    assert _vars_in_template(env, sql) == {"items"}


# =====================================================================
# render_sql_template
# =====================================================================


def test_render_sql_template_success():
    """Renders SQL when params exactly match template variables."""
    sql = "SELECT * FROM {{ schema }}.{{ table }}"
    result = render_sql_template(sql, {"schema": "sales", "table": "events"})
    assert result == "SELECT * FROM sales.events"


def test_render_sql_template_missing_param():
    """Raises ValueError when template variable missing from params."""
    sql = "SELECT * FROM {{ schema }}.{{ table }}"
    with pytest.raises(ValueError, match="missing"):
        render_sql_template(sql, {"schema": "sales"})


def test_render_sql_template_unused_param():
    """Raises ValueError when extra unused parameter supplied."""
    sql = "SELECT * FROM {{ schema }}"
    with pytest.raises(ValueError, match="unused"):
        render_sql_template(sql, {"schema": "sales", "table": "events"})


def test_render_sql_template_missing_and_unused():
    """Raises ValueError listing both missing and unused params."""
    sql = "SELECT * FROM {{ schema }}"
    with pytest.raises(ValueError, match="missing.*unused"):
        render_sql_template(sql, {"table": "events"})


def test_render_sql_template_no_variables():
    """Renders plain SQL unchanged when no variables present."""
    assert render_sql_template("SELECT 1", {}) == "SELECT 1"


# =====================================================================
# BigQueryQueryStats
# =====================================================================


def test_bigquery_stats_to_dict_excludes_none():
    """Excludes None fields from dict when exclude_none=True."""
    stats = BigQueryQueryStats(
        job_id="abc",
        total_rows=None,
        total_bytes_processed=100,
        total_bytes_billed=None,
        cache_hit=None,
        elapsed_seconds=1.0,
        conversion_seconds=0.1,
    )
    d = stats.to_dict()
    assert "total_rows" not in d
    assert d["total_bytes_processed"] == 100


def test_bigquery_stats_to_dict_includes_none():
    """Includes None fields when exclude_none=False."""
    stats = BigQueryQueryStats(
        job_id="abc",
        total_rows=None,
        total_bytes_processed=None,
        total_bytes_billed=None,
        cache_hit=None,
        elapsed_seconds=1.0,
        conversion_seconds=0.1,
    )
    d = stats.to_dict(exclude_none=False)
    assert "total_rows" in d
    assert d["total_rows"] is None


# =====================================================================
# query_to_dataframe
# =====================================================================


def test_query_to_dataframe_pandas(mock_bq_client, mock_row_iterator, sample_pandas_df):
    """Returns pandas DataFrame with all stats fields wired."""
    mock_row_iterator.to_dataframe.return_value = sample_pandas_df

    df, stats = query_to_dataframe(
        "SELECT 1", client=mock_bq_client, dataframe_type="pandas"
    )
    assert isinstance(df, pd.DataFrame)
    assert stats.job_id == "test-job-123"
    assert stats.total_bytes_processed == 1024
    assert stats.total_bytes_billed == 2048
    assert stats.cache_hit is False
    assert stats.total_rows == 100
    assert stats.elapsed_seconds > 0
    assert stats.conversion_seconds >= 0


def test_query_to_dataframe_polars(
    mock_bq_client, mock_row_iterator, sample_arrow_table
):
    """Returns polars DataFrame via Arrow intermediate conversion."""
    pl = pytest.importorskip("polars")
    mock_row_iterator.to_arrow.return_value = sample_arrow_table

    df, stats = query_to_dataframe(
        "SELECT 1",
        client=mock_bq_client,
        dataframe_type="polars",
    )
    assert isinstance(df, pl.DataFrame)
    assert stats.job_id == "test-job-123"


def test_query_to_dataframe_invalid_type(mock_bq_client):
    """Raises ValueError for unsupported dataframe_type."""
    with pytest.raises(ValueError, match="Unsupported dataframe_type"):
        query_to_dataframe(
            "SELECT 1",
            client=mock_bq_client,
            dataframe_type="invalid",
        )


# =====================================================================
# _check_storage_uri_str
# =====================================================================


def test_check_storage_uri_valid():
    """Passes silently for valid gs:// URI."""
    _check_storage_uri_str("gs://bucket/file.parquet")


def test_check_storage_uri_invalid_prefix():
    """Raises ValueError for non-gs:// URI string."""
    with pytest.raises(ValueError, match="storage_uri must be"):
        _check_storage_uri_str("s3://bucket/file.parquet")


def test_check_storage_uri_non_string():
    """Raises ValueError when input is not a string."""
    with pytest.raises(ValueError, match="storage_uri must be"):
        _check_storage_uri_str(None)


# =====================================================================
# _check_gcs_file_stats
# =====================================================================


def test_check_gcs_file_stats_success(mocker):
    """Returns info dict when filesystem.info succeeds."""
    fake_fs = mocker.MagicMock()
    fake_fs.info.return_value = {"size": 1024}
    result = _check_gcs_file_stats("gs://bucket/file", filesystem_obj=fake_fs)
    assert result["size"] == 1024


def test_check_gcs_file_stats_missing_object(mocker):
    """Raises FileNotFoundError when object does not exist."""
    fake_fs = mocker.MagicMock()
    fake_fs.info.side_effect = FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Object not found"):
        _check_gcs_file_stats("gs://bucket/file", filesystem_obj=fake_fs)


# =====================================================================
# write_df_to_gcs_parquet
# =====================================================================


def test_write_parquet_unsupported_type():
    """Raises TypeError when df is not pandas or polars DataFrame."""
    with pytest.raises(TypeError, match="Unsupported df type"):
        write_df_to_gcs_parquet({"a": [1]}, "gs://bucket/file.parquet")


def test_write_parquet_confirm_none_skips_stat(sample_pandas_df, mocker):
    """confirm='none' returns URI dict without stat check."""
    mocker.patch.dict("sys.modules", {"gcsfs": mocker.MagicMock()})
    mocker.patch.object(sample_pandas_df, "to_parquet")
    mock_stats = mocker.patch("tsbricks.blocks.dataio._check_gcs_file_stats")

    result = write_df_to_gcs_parquet(
        sample_pandas_df,
        "gs://bucket/file.parquet",
        confirm="none",
    )
    assert result == {"uri": "gs://bucket/file.parquet"}
    sample_pandas_df.to_parquet.assert_called_once()
    mock_stats.assert_not_called()


# =====================================================================
# write_df_to_gcs_csv
# =====================================================================


def test_write_csv_unsupported_type():
    """Raises TypeError when df is not pandas or polars DataFrame."""
    with pytest.raises(TypeError, match="Unsupported df type"):
        write_df_to_gcs_csv({"a": [1]}, "gs://bucket/file.csv")


def test_write_csv_confirm_none_skips_stat(sample_pandas_df, mocker):
    """confirm='none' returns URI dict without stat check."""
    mocker.patch.dict("sys.modules", {"gcsfs": mocker.MagicMock()})
    mocker.patch.object(sample_pandas_df, "to_csv")
    mock_stats = mocker.patch("tsbricks.blocks.dataio._check_gcs_file_stats")

    result = write_df_to_gcs_csv(
        sample_pandas_df,
        "gs://bucket/file.csv",
        confirm="none",
    )
    assert result == {"uri": "gs://bucket/file.csv"}
    sample_pandas_df.to_csv.assert_called_once()
    mock_stats.assert_not_called()


# =====================================================================
# write_df_to_gcs
# =====================================================================


def test_write_df_to_gcs_parquet_extension_check(sample_pandas_df):
    """Raises ValueError if parquet format but URI has wrong extension."""
    with pytest.raises(ValueError, match="Parquet output requires"):
        write_df_to_gcs(
            sample_pandas_df,
            "gs://bucket/file.csv",
            file_format="parquet",
        )


def test_write_df_to_gcs_csv_extension_check(sample_pandas_df):
    """Raises ValueError if csv format but URI has wrong extension."""
    with pytest.raises(ValueError, match="CSV output requires"):
        write_df_to_gcs(
            sample_pandas_df,
            "gs://bucket/file.parquet",
            file_format="csv",
        )


def test_write_df_to_gcs_invalid_format(sample_pandas_df):
    """Raises ValueError for unsupported file_format."""
    with pytest.raises(ValueError, match="Unsupported file_format"):
        write_df_to_gcs(
            sample_pandas_df,
            "gs://bucket/file.json",
            file_format="json",
        )
