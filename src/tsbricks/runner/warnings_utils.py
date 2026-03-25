"""Utilities for capturing and formatting Python warnings.

These utilities support both the backtesting engine (which captures
warnings automatically) and power users calling composable step
functions directly.  See ``PACKAGE_MAINTAINER_SPEC.md`` §9.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Generator


def format_warnings(
    caught: list[warnings.WarningMessage],
    fold: str,
    stage: str,
    unique_id: str | None = None,
) -> list[dict]:
    """Convert captured ``WarningMessage`` objects to structured dicts.

    Args:
        caught: Warning messages captured by
            ``warnings.catch_warnings(record=True)``.
        fold: Fold identifier (e.g. ``"fold_0"``, ``"test"``).
        stage: Pipeline stage — one of ``"transform"``,
            ``"model"``, ``"metric"``.
        unique_id: Series identifier when attributable;
            ``None`` otherwise.

    Returns:
        List of dicts with keys ``fold``, ``stage``, ``category``,
        ``message``, ``filename``, ``lineno``, ``unique_id``.

    .. note:: See ``PACKAGE_MAINTAINER_SPEC.md`` §9 for the full
       warning capture architecture.
    """
    return [
        {
            "fold": fold,
            "stage": stage,
            "category": w.category.__name__,
            "message": str(w.message),
            "filename": w.filename,
            "lineno": w.lineno,
            "unique_id": unique_id,
        }
        for w in caught
    ]


@contextlib.contextmanager
def capture_warnings(
    target: list[dict],
    fold: str,
    stage: str,
    unique_id: str | None = None,
) -> Generator[None, None, None]:
    """Context manager that captures warnings and appends them to *target*.

    Wraps ``warnings.catch_warnings(record=True)`` with
    ``simplefilter("always")`` so no warnings are deduplicated,
    then formats and appends to *target* on exit.

    Args:
        target: List to append formatted warning dicts to.
        fold: Fold identifier (e.g. ``"fold_0"``, ``"test"``).
        stage: Pipeline stage — one of ``"transform"``,
            ``"model"``, ``"metric"``.
        unique_id: Series identifier when attributable;
            ``None`` otherwise.

    Example::

        my_warnings: list[dict] = []
        with capture_warnings(my_warnings, fold="fold_0", stage="model"):
            invoke_model(model_callable, train_df, ...)
        # my_warnings now contains formatted warning dicts

    .. note:: See ``PACKAGE_MAINTAINER_SPEC.md`` §9 for the full
       warning capture architecture.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield
        finally:
            target.extend(format_warnings(caught, fold, stage, unique_id))
