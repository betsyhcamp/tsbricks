"""Shared fixtures for tsbricks.blocks tests."""

import subprocess

import pytest


@pytest.fixture
def fake_git_toplevel(tmp_path):
    """Mock subprocess.run that returns tmp_path as the git repo root."""

    def _run(cmd, **kwargs):
        result = subprocess.CompletedProcess(cmd, 0)
        result.stdout = str(tmp_path) + "\n"
        return result

    return _run
