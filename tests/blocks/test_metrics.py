"""Tests for tsbricks.blocks.metrics"""

from __future__ import annotations
import numpy as np
import pytest

from tsbricks.blocks.metrics import (
    _bad_numerator_inputs,
    _bad_denominator_inputs,
    _scale_is_invalid,
    _sanitize_value,
    _SMALL_NUM_BOUND,
    difference_scale,
    rmse,
    rmsse,
    difference_scaled_bias,
)


# =====================================================================
# _bad_numerator_inputs
# =====================================================================


def test_bad_numerator_valid_1d_finite_returns_false(y_true_simple, y_pred_simple):
    """Returns False for valid 1D finite arrays of equal shape."""
    assert _bad_numerator_inputs(y_true_simple, y_pred_simple) is False


def test_bad_numerator_nonfinite_returns_true(y_true_nonfinite, y_pred_nonfinite_pair):
    """Returns True when y_true contains NaN."""
    assert _bad_numerator_inputs(y_true_nonfinite, y_pred_nonfinite_pair) is True


def test_bad_numerator_shape_mismatch_raises(arr):
    """Raises ValueError when array lengths differ."""
    with pytest.raises(ValueError, match="1D arrays of same shape"):
        _bad_numerator_inputs(arr([1, 2]), arr([1, 2, 3]))


def test_bad_numerator_2d_input_raises(y_true_shape_2d, y_pred_shape_2d):
    """Raises ValueError for 2D input arrays."""
    with pytest.raises(ValueError, match="1D arrays of same shape"):
        _bad_numerator_inputs(y_true_shape_2d, y_pred_shape_2d)


# =====================================================================
# _bad_denominator_inputs
# =====================================================================


def test_bad_denominator_valid_returns_false(y_train_unit_diffs):
    """Returns False for valid 1D training array with m=1."""
    assert _bad_denominator_inputs(y_train_unit_diffs, m=1) is False


def test_bad_denominator_m_below_one_raises(y_train_unit_diffs):
    """Raises ValueError if m < 1."""
    with pytest.raises(ValueError, match="m must be >= 1"):
        _bad_denominator_inputs(y_train_unit_diffs, m=0)


def test_bad_denominator_too_short_returns_true(y_train_too_short):
    """Returns True when len(y_train) <= m."""
    assert _bad_denominator_inputs(y_train_too_short, m=1) is True


def test_bad_denominator_2d_train_raises():
    """Raises ValueError for 2D y_train."""
    with pytest.raises(ValueError, match="1D array"):
        _bad_denominator_inputs(np.array([[1.0, 2.0, 3.0]]), m=1)


# =====================================================================
# _scale_is_invalid
# =====================================================================


def test_scale_is_invalid_positive_finite():
    """Returns False for a normal positive scale."""
    assert _scale_is_invalid(1.0) is False


def test_scale_is_invalid_near_zero():
    """Returns True when scale is at or below the guard threshold."""
    assert _scale_is_invalid(_SMALL_NUM_BOUND * 0.5) is True


def test_scale_is_invalid_inf():
    """Returns True for infinite scale."""
    assert _scale_is_invalid(float("inf")) is True


# =====================================================================
# _sanitize_value
# =====================================================================


def test_sanitize_value_finite_passthrough():
    """Returns the float unchanged when finite."""
    assert _sanitize_value(3.14) == pytest.approx(3.14)


def test_sanitize_value_inf_returns_nan():
    """Converts +inf to NaN."""
    assert np.isnan(_sanitize_value(float("inf")))


# =====================================================================
# difference_scale
# =====================================================================


def test_difference_scale_rms_unit_diffs(y_train_unit_diffs):
    """RMS scale is 1.0 for series with unit diffs."""
    scale, unstable = difference_scale(y_train_unit_diffs, m=1, scale_stat="rms")
    assert scale == pytest.approx(1.0)
    assert unstable is False


def test_difference_scale_meanabs_unit_diffs(y_train_unit_diffs):
    """Meanabs scale is 1.0 for series with unit diffs."""
    scale, unstable = difference_scale(y_train_unit_diffs, m=1, scale_stat="meanabs")
    assert scale == pytest.approx(1.0)
    assert unstable is False


def test_difference_scale_rms_double_diffs(y_train_double_diffs):
    """RMS scale is 2.0 for series with double diffs."""
    scale, unstable = difference_scale(y_train_double_diffs, m=1)
    assert scale == pytest.approx(2.0)
    assert unstable is False


def test_difference_scale_constant_series_is_unstable(y_train_constant):
    """Constant series produces zero diffs → unstable."""
    scale, unstable = difference_scale(y_train_constant, m=1)
    assert np.isnan(scale)
    assert unstable is True


def test_difference_scale_near_zero_diffs_is_unstable(y_train_near_zero_diffs):
    """Tiny diffs below the guard threshold → unstable."""
    scale, unstable = difference_scale(y_train_near_zero_diffs, m=1)
    assert unstable is True


def test_difference_scale_invalid_scale_stat_raises(y_train_unit_diffs):
    """Raises ValueError for unrecognized scale_stat."""
    with pytest.raises(ValueError, match="scale_stat must be"):
        difference_scale(y_train_unit_diffs, m=1, scale_stat="bad")


# =====================================================================
# rmse
# =====================================================================


def test_rmse_known_value(y_true_simple, y_pred_simple):
    """RMSE is 1.0 for errors [-1, -1]."""
    np.testing.assert_allclose(rmse(y_true_simple, y_pred_simple), 1.0)


def test_rmse_perfect_prediction(y_true_zero_error, y_pred_zero_error):
    """RMSE is 0.0 when forecast matches actuals exactly."""
    np.testing.assert_allclose(
        rmse(y_true_zero_error, y_pred_zero_error), 0.0, atol=1e-15
    )


def test_rmse_nonfinite_returns_nan(y_true_nonfinite, y_pred_nonfinite_pair):
    """NaN in y_true → returns NaN."""
    assert np.isnan(rmse(y_true_nonfinite, y_pred_nonfinite_pair))


def test_rmse_empty_arrays_returns_nan(arr):
    """Empty input arrays → returns NaN."""
    assert np.isnan(rmse(arr([]), arr([])))


def test_rmse_shape_mismatch_raises(arr):
    """Raises ValueError when array lengths differ."""
    with pytest.raises(ValueError, match="1D arrays of same shape"):
        rmse(arr([1, 2]), arr([1, 2, 3]))


def test_rmse_2d_input_raises(y_true_shape_2d, y_pred_shape_2d):
    """Raises ValueError for 2D input arrays."""
    with pytest.raises(ValueError, match="1D arrays of same shape"):
        rmse(y_true_shape_2d, y_pred_shape_2d)


def test_rmse_kwargs_raises():
    """Raises NotImplementedError when **kwargs are supplied."""
    with pytest.raises(NotImplementedError, match="does not yet support"):
        rmse([1, 2], [1, 2], axis=0)


def test_rmse_accepts_plain_lists():
    """Coerces plain Python lists via np.asarray."""
    np.testing.assert_allclose(rmse([0, 0], [1, 1]), 1.0)


def test_rmse_overflow_returns_nan():
    """Overflow from huge squared errors → sanitized to NaN."""
    big = np.finfo(float).max
    assert np.isnan(rmse([big], [-big]))


# =====================================================================
# rmsse
# =====================================================================


def test_rmsse_perfect_forecast_is_zero(
    y_true_zero_error, y_pred_zero_error, y_train_unit_diffs
):
    """RMSSE is 0.0 when forecast matches actuals exactly."""
    value, unstable = rmsse(
        y_true_zero_error, y_pred_zero_error, y_train=y_train_unit_diffs
    )
    assert value == pytest.approx(0.0)
    assert unstable is False


def test_rmsse_known_value(y_true_simple, y_pred_simple, y_train_unit_diffs):
    """RMSSE is 1.0 for errors=[-1,-1], MSE=1, scale=1."""
    value, unstable = rmsse(y_true_simple, y_pred_simple, y_train=y_train_unit_diffs)
    assert value == pytest.approx(1.0)
    assert unstable is False


def test_rmsse_return_components(y_true_simple, y_pred_simple, y_train_unit_diffs):
    """4-tuple contains correct value, MSE, scale, and flag."""
    value, mse, scale, unstable = rmsse(
        y_true_simple,
        y_pred_simple,
        y_train=y_train_unit_diffs,
        return_components=True,
    )
    assert value == pytest.approx(1.0)
    assert mse == pytest.approx(1.0)
    assert scale == pytest.approx(1.0)
    assert unstable is False


def test_rmsse_both_none_raises(y_true_simple, y_pred_simple):
    """Raises ValueError when both y_train and fallback_scale are None."""
    with pytest.raises(ValueError, match="Both cannot be None"):
        rmsse(y_true_simple, y_pred_simple)


def test_rmsse_fallback_scale(y_true_simple, y_pred_simple, fallback_scale_unit):
    """Uses fallback_scale when y_train not provided."""
    value, unstable = rmsse(
        y_true_simple, y_pred_simple, fallback_scale=fallback_scale_unit
    )
    assert value == pytest.approx(1.0)
    assert unstable is False


def test_rmsse_y_train_takes_precedence(
    y_true_simple, y_pred_simple, y_train_double_diffs
):
    """y_train scale (2.0) used even when fallback_scale (1.0) also given."""
    value, unstable = rmsse(
        y_true_simple,
        y_pred_simple,
        y_train=y_train_double_diffs,
        fallback_scale=1.0,
    )
    # MSE=1, scale=2 → RMSSE = sqrt(1)/2 = 0.5
    assert value == pytest.approx(0.5)
    assert unstable is False


def test_rmsse_nonfinite_numerator_returns_nan(
    y_true_nonfinite, y_pred_nonfinite_pair, y_train_unit_diffs
):
    """NaN in y_true → returns (NaN, False); scale not flagged unstable."""
    value, unstable = rmsse(
        y_true_nonfinite, y_pred_nonfinite_pair, y_train=y_train_unit_diffs
    )
    assert np.isnan(value)
    assert unstable is False


def test_rmsse_unstable_scale(y_true_simple, y_pred_simple, y_train_constant):
    """Constant y_train → unstable scale flagged True."""
    value, unstable = rmsse(y_true_simple, y_pred_simple, y_train=y_train_constant)
    assert np.isnan(value)
    assert unstable is True


# =====================================================================
# difference_scaled_bias
# =====================================================================


def test_dsb_zero_bias(y_true_zero_error, y_pred_zero_error, y_train_unit_diffs):
    """DSB is 0.0 when forecast matches actuals exactly."""
    value, unstable = difference_scaled_bias(
        y_true_zero_error, y_pred_zero_error, y_train=y_train_unit_diffs
    )
    assert value == pytest.approx(0.0)
    assert unstable is False


def test_dsb_negative_bias_underprediction(
    y_true_simple, y_pred_simple, y_train_unit_diffs
):
    """Negative DSB when y_pred < y_true (under-prediction)."""
    # y_pred - y_true = [1-2, 3-4] = [-1, -1] → mean = -1, scale = 1
    value, unstable = difference_scaled_bias(
        y_true_simple, y_pred_simple, y_train=y_train_unit_diffs
    )
    assert value == pytest.approx(-1.0)
    assert unstable is False


def test_dsb_positive_bias_overprediction(
    y_true_simple, y_pred_over, y_train_unit_diffs
):
    """Positive DSB when y_pred > y_true (over-prediction)."""
    # y_pred - y_true = [3-2, 5-4] = [1, 1] → mean = 1, scale = 1
    value, unstable = difference_scaled_bias(
        y_true_simple, y_pred_over, y_train=y_train_unit_diffs
    )
    assert value == pytest.approx(1.0)
    assert unstable is False


def test_dsb_return_components(y_true_simple, y_pred_simple, y_train_unit_diffs):
    """4-tuple contains correct value, mean_error, scale, and flag."""
    value, mean_error, scale, unstable = difference_scaled_bias(
        y_true_simple,
        y_pred_simple,
        y_train=y_train_unit_diffs,
        return_components=True,
    )
    assert value == pytest.approx(-1.0)
    assert mean_error == pytest.approx(-1.0)
    assert scale == pytest.approx(1.0)
    assert unstable is False


def test_dsb_both_none_raises(y_true_simple, y_pred_simple):
    """Raises ValueError when both y_train and fallback_scale are None."""
    with pytest.raises(ValueError, match="Both cannot be None"):
        difference_scaled_bias(y_true_simple, y_pred_simple)


def test_dsb_fallback_scale(y_true_simple, y_pred_simple, fallback_scale_unit):
    """Uses fallback_scale when y_train not provided."""
    value, unstable = difference_scaled_bias(
        y_true_simple, y_pred_simple, fallback_scale=fallback_scale_unit
    )
    assert value == pytest.approx(-1.0)
    assert unstable is False


def test_dsb_meanabs_scale_stat(y_true_simple, y_pred_simple, y_train_double_diffs):
    """DSB uses meanabs scale when requested."""
    # mean_error = -1, meanabs scale of [0,2,4] diffs = 2
    value, unstable = difference_scaled_bias(
        y_true_simple,
        y_pred_simple,
        y_train=y_train_double_diffs,
        scale_stat="meanabs",
    )
    assert value == pytest.approx(-0.5)
    assert unstable is False


def test_dsb_unstable_scale(y_true_simple, y_pred_simple, y_train_constant):
    """Constant y_train → unstable scale flagged True."""
    value, unstable = difference_scaled_bias(
        y_true_simple, y_pred_simple, y_train=y_train_constant
    )
    assert np.isnan(value)
    assert unstable is True
