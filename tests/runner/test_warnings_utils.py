"""Tests for tsbricks.runner.warnings_utils."""

from __future__ import annotations

import warnings

import pytest

from tsbricks.runner.warnings_utils import capture_warnings, format_warnings


class TestFormatWarnings:
    """Tests for format_warnings."""

    def test_empty_list_returns_empty(self):
        result = format_warnings([], fold="fold_0", stage="model")
        assert result == []

    def test_single_warning_formatted(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("test message", UserWarning)

        result = format_warnings(caught, fold="fold_0", stage="model", unique_id="A")

        assert len(result) == 1
        entry = result[0]
        assert entry["fold"] == "fold_0"
        assert entry["stage"] == "model"
        assert entry["category"] == "UserWarning"
        assert entry["message"] == "test message"
        assert entry["unique_id"] == "A"
        assert isinstance(entry["filename"], str)
        assert isinstance(entry["lineno"], int)

    def test_multiple_warnings_formatted(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("first", UserWarning)
            warnings.warn("second", RuntimeWarning)

        result = format_warnings(caught, fold="fold_1", stage="transform")

        assert len(result) == 2
        assert result[0]["category"] == "UserWarning"
        assert result[0]["message"] == "first"
        assert result[1]["category"] == "RuntimeWarning"
        assert result[1]["message"] == "second"

    def test_unique_id_defaults_to_none(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("msg", UserWarning)

        result = format_warnings(caught, fold="fold_0", stage="metric")

        assert result[0]["unique_id"] is None

    def test_stage_values_passed_through(self):
        """Stage is passed through as-is — no validation."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("msg", UserWarning)

        for stage in ("transform", "model", "metric"):
            result = format_warnings(caught, fold="fold_0", stage=stage)
            assert result[0]["stage"] == stage


class TestCaptureWarnings:
    """Tests for capture_warnings context manager."""

    def test_captures_warning_into_target(self):
        target: list[dict] = []

        with capture_warnings(target, fold="fold_0", stage="model"):
            warnings.warn("convergence issue", UserWarning)

        assert len(target) == 1
        assert target[0]["message"] == "convergence issue"
        assert target[0]["fold"] == "fold_0"
        assert target[0]["stage"] == "model"

    def test_no_warnings_appends_nothing(self):
        target: list[dict] = []

        with capture_warnings(target, fold="fold_0", stage="model"):
            pass  # no warnings emitted

        assert target == []

    def test_multiple_warnings_captured(self):
        target: list[dict] = []

        with capture_warnings(target, fold="fold_0", stage="transform"):
            warnings.warn("first", UserWarning)
            warnings.warn("second", RuntimeWarning)

        assert len(target) == 2

    def test_appends_to_existing_list(self):
        target: list[dict] = [{"existing": "entry"}]

        with capture_warnings(target, fold="fold_0", stage="model"):
            warnings.warn("new", UserWarning)

        assert len(target) == 2
        assert target[0] == {"existing": "entry"}
        assert target[1]["message"] == "new"

    def test_unique_id_passed_through(self):
        target: list[dict] = []

        with capture_warnings(
            target, fold="fold_0", stage="metric", unique_id="series_A"
        ):
            warnings.warn("msg", UserWarning)

        assert target[0]["unique_id"] == "series_A"

    def test_captures_duplicate_warnings(self):
        """simplefilter('always') should not deduplicate."""
        target: list[dict] = []

        with capture_warnings(target, fold="fold_0", stage="model"):
            warnings.warn("same msg", UserWarning)
            warnings.warn("same msg", UserWarning)

        assert len(target) == 2

    def test_nested_function_warnings_captured(self):
        """Warnings from called functions are captured."""

        def inner_fn():
            warnings.warn("from inner", RuntimeWarning)

        target: list[dict] = []

        with capture_warnings(target, fold="fold_0", stage="model"):
            inner_fn()

        assert len(target) == 1
        assert target[0]["message"] == "from inner"
        assert target[0]["category"] == "RuntimeWarning"

    def test_warnings_preserved_when_wrapped_code_raises(self):
        """Warnings emitted before an exception are still captured."""
        target: list[dict] = []

        with pytest.raises(ValueError, match="boom"):
            with capture_warnings(target, fold="fold_0", stage="model"):
                warnings.warn("before failure", UserWarning)
                raise ValueError("boom")

        assert len(target) == 1
        assert target[0]["message"] == "before failure"
        assert target[0]["fold"] == "fold_0"
        assert target[0]["stage"] == "model"
