from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 8: unskip when run_backtest is wired")
def test_minimal_backtest_end_to_end() -> None:
    """Smoke test: synthetic data + dummy model → BacktestResults with metrics."""
    # Phase 8 will fill in:
    # - Synthetic DataFrame (2 series, 24 months, simple trend)
    # - Minimal config dict with dummy model callable
    # - Call run_backtest(config=cfg, df=df)
    # - Assert results.cv.metrics is not empty, has expected schema
    # - Assert results.cv.forecasts_per_fold has correct fold count
    # - Assert results.cv.fold_origins matches configured origins
