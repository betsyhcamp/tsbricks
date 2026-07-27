"""Tests of utils.py functions and classes."""

from __future__ import annotations
import pytest

from tsbricks.runner.utils import dynamic_import


def test_dynamic_import_malformed_path_raises():
    """Dotted path without a dot raises a clear ValueError."""
    with pytest.raises(ValueError, match="Invalid dotted path"):
        dynamic_import("badpath")


def test_dynamic_import_is_public_api() -> None:
    """ "dynamic_import is re-exported from the tsbricks.runner package
    namespace."""
    import tsbricks.runner
    from tsbricks.runner import dynamic_import as exported

    assert exported is dynamic_import
    assert "dynamic_import" in tsbricks.runner.__all__


def test_resolve_predict_is_public_api() -> None:
    """resolve_predict is exported from the tsbricks.runner namespace."""
    import tsbricks.runner
    from tsbricks.runner import resolve_predict as exported
    from tsbricks.runner.model_invocation import resolve_predict

    assert exported is resolve_predict
    assert "resolve_predict" in tsbricks.runner.__all__


def test_invoke_predict_is_public_api() -> None:
    """invoke_predict is exported from the tsbricks.runner namespace."""
    import tsbricks.runner
    from tsbricks.runner import invoke_predict as exported
    from tsbricks.runner.model_invocation import invoke_predict

    assert exported is invoke_predict
    assert "invoke_predict" in tsbricks.runner.__all__
