"""Unit tests for tsbricks.blocks.metadata environment metadata collection."""

import hashlib
import subprocess
from unittest.mock import patch

import pytest

from tsbricks.blocks.metadata import (
    MetadataWarning,
    get_git_hash,
    get_uv_lock_info,
)


# ---------- get_git_hash ----------


def test_get_git_hash_returns_full_sha():
    """Happy path: produces a valid 40-character hex commit hash."""
    result = get_git_hash()
    assert result is not None
    assert len(result) == 40
    assert all(c in "0123456789abcdef" for c in result)


def test_get_git_hash_returns_none_when_git_not_installed():
    """Gracefully degrades when git binary is not installed."""
    with patch(
        "tsbricks.blocks.metadata.subprocess.run", side_effect=FileNotFoundError
    ):
        with pytest.warns(MetadataWarning, match="Could not determine git hash"):
            result = get_git_hash()
    assert result is None


def test_get_git_hash_returns_none_when_not_in_repo():
    """Gracefully degrades when the working directory is outside a git repo."""
    with patch(
        "tsbricks.blocks.metadata.subprocess.run",
        side_effect=subprocess.CalledProcessError(128, "git"),
    ):
        with pytest.warns(MetadataWarning, match="Could not determine git hash"):
            result = get_git_hash()
    assert result is None


# ---------- get_uv_lock_info ----------


def test_get_uv_lock_info_with_explicit_path(tmp_path):
    """Happy path: explicit path produces dict with resolved path and sha256."""
    lock_file = tmp_path / "uv.lock"
    lock_file.write_text("some lock content")
    expected_sha = hashlib.sha256(b"some lock content").hexdigest()

    result = get_uv_lock_info(uv_lock_path=lock_file)

    assert result is not None
    assert result["path"] == str(lock_file.resolve())
    assert result["sha256"] == expected_sha


def test_get_uv_lock_info_sha256_matches_independent_computation(tmp_path):
    """Verify sha256 digest is correct against an independently computed hash."""
    content = b"package==1.0.0\nanother-package==2.3.4\n"
    lock_file = tmp_path / "uv.lock"
    lock_file.write_bytes(content)

    result = get_uv_lock_info(uv_lock_path=lock_file)
    assert result["sha256"] == hashlib.sha256(content).hexdigest()


def test_get_uv_lock_info_warns_when_file_not_found(tmp_path):
    """Warns when an explicit path points to a nonexistent file."""
    missing = tmp_path / "uv.lock"
    with pytest.warns(MetadataWarning, match="uv.lock not found"):
        result = get_uv_lock_info(uv_lock_path=missing)
    assert result is None


def test_get_uv_lock_info_warns_when_auto_discover_fails():
    """Gracefully degrades when git is unavailable for auto-discovery."""
    with patch(
        "tsbricks.blocks.metadata.subprocess.run",
        side_effect=FileNotFoundError,
    ):
        with pytest.warns(
            MetadataWarning, match="Could not determine git repository root"
        ):
            result = get_uv_lock_info()
    assert result is None


def test_get_uv_lock_info_auto_discovers_from_git_root(tmp_path):
    """Auto-discovery finds uv.lock at the git repo root without an explicit path."""
    lock_file = tmp_path / "uv.lock"
    lock_file.write_text("auto-discovered content")
    expected_sha = hashlib.sha256(b"auto-discovered content").hexdigest()

    def fake_run(cmd, **kwargs):
        result = subprocess.CompletedProcess(cmd, 0)
        result.stdout = str(tmp_path) + "\n"
        return result

    with patch("tsbricks.blocks.metadata.subprocess.run", side_effect=fake_run):
        result = get_uv_lock_info()

    assert result is not None
    assert result["path"] == str(lock_file.resolve())
    assert result["sha256"] == expected_sha


def test_get_uv_lock_info_auto_discover_warns_when_lock_missing(tmp_path):
    """Warns when auto-discovery finds the git root but uv.lock doesn't exist there."""

    def fake_run(cmd, **kwargs):
        result = subprocess.CompletedProcess(cmd, 0)
        result.stdout = str(tmp_path) + "\n"
        return result

    with patch("tsbricks.blocks.metadata.subprocess.run", side_effect=fake_run):
        with pytest.warns(MetadataWarning, match="uv.lock not found"):
            result = get_uv_lock_info()
    assert result is None
