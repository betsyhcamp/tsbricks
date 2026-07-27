"""Integration test: invoke_predict driven by the production ModelConfig.

Every other test of ``invoke_predict`` lives in ``tests/runner/`` and uses
a dataclass stand-in for the config, mirroring the one-way dependency rule
(``runner/`` must not import from ``backtesting/``) in the test suite. That
independence is worth demonstrating, but it leaves one thing unproven: that
the *production* config type -- a Pydantic ``ModelConfig`` -- actually flows
through resolution and invocation.

This is the file that crosses the namespace boundary on purpose. Nothing
here is mocked: the dotted paths are real, ``dynamic_import`` runs for real,
and the predict callable is really called. The failure mode it exists to
catch is a wiring bug that survives a suite where every test mocked
resolution or injected a fake, so the real path never executed.
"""

from __future__ import annotations

import pandas as pd

from tsbricks.backtesting.schema import ModelConfig
from tsbricks.runner import invoke_predict


def test_invoke_predict_with_real_model_config() -> None:
    """A real ModelConfig resolves and calls, and the forecast comes back.

    The predict double echoes what it received, so the assertions below
    read the arguments back out of the returned frame: the fitted model
    by identity, the horizon, and the predict_params the config declared.
    """
    model_config = ModelConfig(
        callable="tsbricks._testing.dummy_models.forecast_only",
        hyperparameters={"num_leaves": 31},
        predict_callable="tsbricks._testing.dummy_models.predict_only",
        predict_params={"level": [80, 95]},
    )
    fitted_model = {"name": "already-fitted"}

    forecast = invoke_predict(fitted_model, model_config, horizon=6)

    assert isinstance(forecast, pd.DataFrame)
    assert forecast["fitted_model"].iloc[0] is fitted_model
    assert forecast["horizon"].iloc[0] == 6

    # predict_params declared on the production type reach the callable...
    kwargs = forecast["kwargs"].iloc[0]
    assert kwargs["level"] == [80, 95]

    # ...and fit-time hyperparameters do not, even though this config
    # carries both. Sharing one dict would pass num_leaves=31 to a
    # predict function and raise TypeError.
    assert "num_leaves" not in kwargs


# TODO (human): add the failure-path counterpart to the test above --
# test_invoke_predict_real_config_without_predict_callable.
#
# Build a real ModelConfig that sets `callable` but omits predict_callable
# entirely, call invoke_predict, and assert it raises ValueError. Reuse the
# `match=` anchor from tests/runner/ ("no 'predict_callable'").
#
# The thing worth noticing while you write it -- it is why this test is
# worth having at the integration level and not only in tests/runner/:
#
# The guard in resolve_predict is
#     getattr(model_config, "predict_callable", None)
# and it collapses TWO different situations into one error. But a Pydantic
# ModelConfig can only ever be in ONE of them. Because the field is declared
# with a `= None` default, the attribute ALWAYS exists on a real config --
# so the getattr default is unreachable here, and only the `is None` branch
# can fire. The other branch (attribute genuinely absent) is reachable only
# by duck-typed configs, which is what _LegacyModelConfig covers in
# tests/runner/.
#
# So the two tests exercise the same guard through different doors. Neither
# subsumes the other, and a reader who assumes this one is redundant with
# test_resolve_predict_missing_attribute_raises_same_error has it backwards.
