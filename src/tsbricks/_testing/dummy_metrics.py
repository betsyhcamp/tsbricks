"""Dummy metric callables for testing series-level resilience."""

from __future__ import annotations

import numpy as np


def rmse_fails_on_zero_error(
    y_true: np.ndarray, y_pred: np.ndarray, **kwargs: object
) -> float:
    """RMSE that raises when all errors are zero.

    Useful for testing series-level resilience: series with perfect
    predictions will trigger a failure while others succeed.
    """
    errors = y_true - y_pred
    if np.allclose(errors, 0.0):
        raise ValueError("Zero error not allowed")
    return float(np.sqrt(np.mean(errors**2)))
