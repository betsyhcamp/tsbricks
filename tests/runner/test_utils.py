"""Tests of _utils.py functions and classes."""

from __future__ import annotations
import pytest

from tsbricks.runner._utils import dynamic_import


def test_dynamic_import_malformed_path_raises():
    """Dotted path without a dot raises a clear ValueError."""
    with pytest.raises(ValueError, match="Invalid dotted path"):
        dynamic_import("badpath")
