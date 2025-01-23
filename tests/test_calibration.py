"""Tests for calibration curve computation and visualization."""

import numpy as np
import pytest
from calcurve import CalibrationCurve


def test_clopper_pearson_interval():
    """Test Clopper-Pearson interval computation."""
    from calcurve._calibration import clopper_pearson_interval

    # Test empty bin
    lower, upper = clopper_pearson_interval(0, 0)
    assert lower == 0.0
    assert upper == 1.0

    # Test full bin
    lower, upper = clopper_pearson_interval(10, 10, confidence_level=0.90)
    assert lower > 0.7  # Approximate check
    assert upper == 1.0

    # Test partial bin
    lower, upper = clopper_pearson_interval(5, 10, confidence_level=0.90)
    assert 0.2 < lower < 0.4  # Approximate check
    assert 0.6 < upper < 0.8  # Approximate check


def test_wilson_cc_interval():
    """Test Wilson score interval with continuity correction."""
    from calcurve._calibration import wilson_cc_interval

    # Test empty bin
    lower, upper = wilson_cc_interval(0, 0)
    assert lower == 0.0
    assert upper == 1.0

    # Test full bin
    lower, upper = wilson_cc_interval(10, 10, confidence_level=0.90)
    assert lower > 0.7  # Approximate check
    assert upper == 1.0

    # Test partial bin
    lower, upper = wilson_cc_interval(5, 10, confidence_level=0.90)
    assert 0.2 < lower < 0.4  # Approximate check
    assert 0.6 < upper < 0.8  # Approximate check


def test_calibration_curve_init():
    """Test CalibrationCurve initialization."""
    # Test valid parameters
    cal = CalibrationCurve()
    assert cal.binning_strategy == "quantile"
    assert cal.n_bins == 10
    assert cal.confidence_method == "clopper_pearson"
    assert cal.confidence_level == 0.90

    # Test invalid binning strategy
    with pytest.raises(ValueError, match="binning_strategy must be one of"):
        CalibrationCurve(binning_strategy="invalid")


def test_custom_bin_edges():
    """Test setting custom bin edges."""
    cal = CalibrationCurve(binning_strategy="custom")

    # Test valid bin edges
    valid_edges = np.array([0.0, 0.5, 1.0])
    cal.set_bin_edges(valid_edges)
    assert np.allclose(cal._bin_edges, valid_edges)

    # Test invalid bin edges
    with pytest.raises(ValueError):
        cal.set_bin_edges([0.5, 0.3, 1.0])  # Not increasing
    with pytest.raises(ValueError):
        cal.set_bin_edges([0.1, 0.5, 0.9])  # Not spanning [0, 1]


def test_calibration_curve_perfect():
    """Test calibration curve with perfect predictions."""
    # Create a larger test set for more stable results
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])

    for method in ["clopper_pearson", "wilson_cc", "bootstrap"]:
        cal = CalibrationCurve(
            binning_strategy="uniform", n_bins=2, confidence_method=method
        )
        cal.fit(y_true, y_pred)

        # Check that predictions are well calibrated
        # Allow for some numerical tolerance
        assert np.allclose(cal._prob_true, [0, 1], atol=0.1)
        assert np.allclose(cal._prob_pred, [0.1, 0.9], atol=0.1)

        # Check that confidence intervals are properly ordered
        # and contain reasonable values
        assert np.all(cal._ci_lower >= 0)  # lower bound should be non-negative
        assert np.all(cal._ci_upper <= 1)  # upper bound should be at most 1
        assert np.all(cal._ci_lower <= cal._ci_upper)  # intervals should be ordered

        # Check that confidence intervals are not too wide
        # (this is a heuristic based on the sample size)
        assert np.all(cal._ci_upper - cal._ci_lower <= 0.5)


def test_input_validation():
    """Test input validation."""
    cal = CalibrationCurve()

    # Test different length arrays
    with pytest.raises(ValueError):
        cal.fit(np.array([0, 1]), np.array([0.5]))

    # Test non-binary labels
    with pytest.raises(ValueError):
        cal.fit(np.array([0, 2]), np.array([0.1, 0.9]))

    # Test invalid probabilities
    with pytest.raises(ValueError):
        cal.fit(np.array([0, 1]), np.array([-0.1, 1.1]))


def test_plotting():
    """Test plotting functionality."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.1, 0.9, 0.9])

    cal = CalibrationCurve()

    # Test plotting before fitting
    with pytest.raises(RuntimeError):
        cal.plot()

    # Test plotting after fitting
    cal.fit(y_true, y_pred)
    ax = cal.plot()
    assert ax is not None
