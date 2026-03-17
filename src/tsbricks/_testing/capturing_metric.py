"""Test metric callable that captures kwargs for verification."""

from __future__ import annotations

import numpy as np

# Module-level list to collect kwargs from each call.
captured_calls: list[dict] = []


def capturing_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    **kwargs: object,
) -> float:
    """RMSE that records its kwargs in ``captured_calls``."""
    captured_calls.append(dict(kwargs))
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
