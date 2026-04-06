"""Aggregation callables for global and group two-stage metric scopes."""

from __future__ import annotations


def unweighted_mean(
    per_series_values: dict[str, float],
    weights: dict[str, float] | None = None,
    **kwargs: object,
) -> float:
    """Simple arithmetic mean of per-series values, ignoring weights."""
    vals = list(per_series_values.values())
    return sum(vals) / len(vals)


def weighted_mean(
    per_series_values: dict[str, float],
    weights: dict[str, float] | None = None,
    **kwargs: object,
) -> float:
    """Weighted mean of per-series values using weights dict."""
    if weights is None:
        raise ValueError("weighted_mean requires weights")
    total_weight = 0.0
    weighted_sum = 0.0
    for uid, val in per_series_values.items():
        w = weights[uid]
        weighted_sum += w * val
        total_weight += w
    return weighted_sum / total_weight


def scaled_mean(
    per_series_values: dict[str, float],
    weights: dict[str, float] | None = None,
    scale: float = 1.0,
    **kwargs: object,
) -> float:
    """Arithmetic mean of per-series values multiplied by scale."""
    vals = list(per_series_values.values())
    return scale * sum(vals) / len(vals)
