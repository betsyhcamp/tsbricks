"""Environment metadata collection for reproducibility and experiment tracking."""

from __future__ import annotations

import hashlib
import subprocess
import warnings
from pathlib import Path


class MetadataWarning(UserWarning):
    """Warning category for metadata collection issues.

    Users can selectively filter these warnings::

        warnings.filterwarnings("ignore", category=MetadataWarning)
    """


def get_git_hash() -> str | None:
    """Return the full 40-character SHA of HEAD.

    Auto-discovers the git repository from the current working directory.

    Returns
    -------
    str | None
        The full commit hash, or ``None`` if git is unavailable or the
        current directory is not inside a git repository.

    Warns
    -----
    MetadataWarning
        If the git hash cannot be determined.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        warnings.warn(
            "Could not determine git hash. "
            "Ensure git is installed and the current directory is inside a git repository.",
            MetadataWarning,
            stacklevel=2,
        )
        return None


def get_uv_lock_info(uv_lock_path: Path | None = None) -> dict | None:
    """Return metadata about the uv.lock file.

    Parameters
    ----------
    uv_lock_path : Path | None
        Explicit path to the uv.lock file. When ``None`` (default),
        auto-discovers ``uv.lock`` at the git repository root.

    Returns
    -------
    dict | None
        ``{"path": "<resolved absolute path>", "sha256": "<hex digest>"}``
        or ``None`` if the file is not found.

    Warns
    -----
    MetadataWarning
        If uv.lock cannot be found (either at the auto-discovered
        location or the explicitly provided path).
    """
    if uv_lock_path is None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=True,
            )
            uv_lock_path = Path(result.stdout.strip()) / "uv.lock"
        except (subprocess.CalledProcessError, FileNotFoundError):
            warnings.warn(
                "Could not determine git repository root to auto-discover uv.lock. "
                "Provide an explicit uv_lock_path or ensure git is installed.",
                MetadataWarning,
                stacklevel=2,
            )
            return None

    resolved = uv_lock_path.resolve()
    if not resolved.is_file():
        warnings.warn(
            f"uv.lock not found at {resolved}. "
            "Dependency metadata will not be recorded.",
            MetadataWarning,
            stacklevel=2,
        )
        return None

    sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return {"path": str(resolved), "sha256": sha256}
