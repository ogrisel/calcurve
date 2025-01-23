"""Tests for calibration curve computation and visualization."""

import matplotlib.pyplot as plt
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


def test_min_samples_per_bins():
    """Test that bins are merged when they have too few samples."""
    # Create data with uneven distribution
    y_pred = np.array([0.1] * 50 + [0.2] * 5 + [0.3] * 5 + [0.9] * 40)
    y_true = np.zeros_like(y_pred)

    # Without min_samples_per_bins
    cal_curve = CalibrationCurve(binning_strategy="uniform", n_bins=10)
    cal_curve.fit(y_true, y_pred)
    original_edges = cal_curve._bin_edges
    assert len(original_edges) == 11  # 10 bins = 11 edges

    # With min_samples_per_bins=10
    cal_curve = CalibrationCurve(
        binning_strategy="uniform", n_bins=10, min_samples_per_bins=10
    )
    cal_curve.fit(y_true, y_pred)
    merged_edges = cal_curve._bin_edges

    # Should have fewer edges due to merging
    assert len(merged_edges) < len(original_edges)

    # Check that all bins now have at least 10 samples
    bin_indices = np.searchsorted(merged_edges, y_pred) - 1
    bin_indices = np.clip(bin_indices, 0, len(merged_edges) - 2)
    bin_counts = np.bincount(bin_indices)
    assert np.all(bin_counts >= 10)

    # Test error handling
    with pytest.raises(ValueError, match="min_samples_per_bins must be at least 1"):
        CalibrationCurve(min_samples_per_bins=0)


def test_bootstrap_edge_cases():
    """Test bootstrap confidence intervals with edge cases."""
    # Case where all samples in a bin belong to same class
    y_pred = np.array([0.1] * 50 + [0.9] * 50)
    y_true = np.zeros_like(y_pred)  # All negative class

    cal = CalibrationCurve(confidence_method="bootstrap", n_bins=2, random_state=42)
    cal.fit(y_true, y_pred)

    # Check that confidence intervals are computed
    assert cal._ci_lower is not None
    assert cal._ci_upper is not None

    # Case with very few samples
    y_pred_small = np.array([0.1, 0.9])
    y_true_small = np.array([0, 1])

    cal = CalibrationCurve(confidence_method="bootstrap", n_bins=2, random_state=42)
    cal.fit(y_true_small, y_pred_small)

    # Check that confidence intervals are computed
    assert cal._ci_lower is not None
    assert cal._ci_upper is not None


def test_merge_bins_edge_cases():
    """Test bin merging with various edge cases."""
    # Case where leftmost bin needs merging
    y_pred = np.array([0.1] * 5 + [0.5] * 50 + [0.9] * 50)
    y_true = np.zeros_like(y_pred)

    cal = CalibrationCurve(
        binning_strategy="uniform", n_bins=3, min_samples_per_bins=10
    )
    cal.fit(y_true, y_pred)

    # Should merge the leftmost bin
    assert len(cal._bin_edges) < 4  # Original had 4 edges for 3 bins

    # Case where rightmost bin needs merging
    y_pred = np.array([0.1] * 50 + [0.5] * 50 + [0.9] * 5)
    cal.fit(y_true, y_pred)

    # Should merge the rightmost bin
    assert len(cal._bin_edges) < 4

    # Case where middle bin needs merging
    y_pred = np.array([0.1] * 50 + [0.5] * 5 + [0.9] * 50)
    cal.fit(y_true, y_pred)

    # Should merge the middle bin
    assert len(cal._bin_edges) < 4


def test_plotting_options():
    """Test plotting with different matplotlib configurations."""
    y_pred = np.array([0.1] * 50 + [0.9] * 50)
    y_true = np.zeros_like(y_pred)

    cal = CalibrationCurve()
    cal.fit(y_true, y_pred)

    # Test plotting without existing axes
    ax = cal.plot()
    assert ax is not None
    plt.close()

    # Test plotting with custom figure size
    fig, ax = plt.subplots(figsize=(10, 8))
    cal.plot(ax=ax)
    plt.close()

    # Test plotting with grid
    fig, ax = plt.subplots()
    ax.grid(True)
    cal.plot(ax=ax)
    plt.close()


def test_input_validation_extended():
    """Test additional input validation cases."""
    # Test invalid confidence method
    with pytest.raises(ValueError, match="confidence_method must be one of"):
        CalibrationCurve(confidence_method="invalid")

    # Test invalid confidence level
    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        CalibrationCurve(confidence_level=1.5)

    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        CalibrationCurve(confidence_level=-0.1)

    # Test invalid n_bootstrap
    with pytest.raises(ValueError, match="n_bootstrap must be positive"):
        CalibrationCurve(confidence_method="bootstrap", n_bootstrap=0)

    # Test invalid n_bins
    with pytest.raises(ValueError, match="n_bins must be positive"):
        CalibrationCurve(n_bins=0)

    # Test invalid bin edges
    cal = CalibrationCurve(binning_strategy="custom")
    with pytest.raises(
        ValueError, match="bin_edges must be strictly increasing and span"
    ):
        cal.set_bin_edges([0, 0.5, 0.3, 1])

    with pytest.raises(
        ValueError, match="bin_edges must be strictly increasing and span"
    ):
        cal.set_bin_edges([-0.1, 0.5, 1])

    with pytest.raises(
        ValueError, match="bin_edges must be strictly increasing and span"
    ):
        cal.set_bin_edges([0, 0.5, 1.1])


def test_bootstrap_bin_splitting():
    """Test bootstrap bin splitting with different binning strategies."""
    # Create data with a large gap
    y_pred = np.array([0.1] * 100 + [0.9] * 100)
    y_true = np.zeros_like(y_pred)
    y_true[100:] = 1  # Perfect separation

    # Test with uniform binning
    cal = CalibrationCurve(
        binning_strategy="uniform",
        confidence_method="bootstrap",
        n_bins=5,
        random_state=42,
    )
    cal.fit(y_true, y_pred)

    # Check that confidence intervals are computed
    assert cal._ci_lower is not None
    assert cal._ci_upper is not None

    # Test with custom bin edges
    cal = CalibrationCurve(
        binning_strategy="custom", confidence_method="bootstrap", random_state=42
    )
    cal.set_bin_edges([0, 0.2, 0.4, 0.6, 0.8, 1])
    cal.fit(y_true, y_pred)

    # Check that confidence intervals are computed
    assert cal._ci_lower is not None
    assert cal._ci_upper is not None

    # Test with quantile binning
    cal = CalibrationCurve(
        binning_strategy="quantile",
        confidence_method="bootstrap",
        n_bins=5,
        random_state=42,
    )
    cal.fit(y_true, y_pred)

    # Check that confidence intervals are computed
    assert cal._ci_lower is not None
    assert cal._ci_upper is not None


def test_plotting_with_confidence():
    """Test plotting with confidence intervals."""
    y_pred = np.array([0.1] * 50 + [0.9] * 50)
    y_true = np.zeros_like(y_pred)
    y_true[50:] = 1  # Perfect separation

    # Test with each confidence method
    for method in ["clopper_pearson", "wilson_cc", "bootstrap"]:
        cal = CalibrationCurve(confidence_method=method)
        cal.fit(y_true, y_pred)
        ax = cal.plot()
        assert ax is not None
        plt.close()


def test_custom_binning_without_edges():
    """Test error when using custom binning without setting edges."""
    y_pred = np.array([0.1] * 50 + [0.9] * 50)
    y_true = np.zeros_like(y_pred)

    cal = CalibrationCurve(binning_strategy="custom")
    with pytest.raises(ValueError, match="bin_edges must be set using set_bin_edges"):
        cal.fit(y_true, y_pred)


def test_edge_case_predictions():
    """Test handling of edge case predictions."""
    # All predictions are the same value
    y_pred = np.array([0.5] * 100)
    y_true = np.random.randint(0, 2, size=100)

    cal = CalibrationCurve(n_bins=10)
    cal.fit(y_true, y_pred)

    # Check that calibration curve is computed
    assert cal._prob_true is not None
    assert cal._prob_pred is not None

    # Test with very few unique values
    y_pred = np.array([0.1] * 50 + [0.9] * 50)
    y_true = np.random.randint(0, 2, size=100)

    cal = CalibrationCurve(n_bins=10)
    cal.fit(y_true, y_pred)

    # Check that calibration curve is computed
    assert cal._prob_true is not None
    assert cal._prob_pred is not None


def test_invalid_confidence_method():
    """Test error when invalid confidence method is used."""
    y_pred = np.array([0.1] * 50 + [0.9] * 50)
    y_true = np.zeros_like(y_pred)

    # Create instance with valid method but then change it
    cal = CalibrationCurve()
    cal.confidence_method = "invalid"

    with pytest.raises(ValueError, match="confidence_method must be one of"):
        cal.fit(y_true, y_pred)


def test_compute_bin_edges():
    """Test _compute_bin_edges directly."""
    y_pred = np.array([0.1] * 50 + [0.9] * 50)

    # Test custom binning without edges
    cal = CalibrationCurve(binning_strategy="custom")
    with pytest.raises(ValueError, match="bin_edges must be set using set_bin_edges"):
        cal._compute_bin_edges(y_pred)


def test_quantile_binning():
    """Test quantile binning strategy with min_samples_per_bins=None."""
    y_pred = np.array([0.1] * 50 + [0.9] * 50)
    y_true = np.zeros_like(y_pred)

    # Test with min_samples_per_bins=None
    cal = CalibrationCurve(
        binning_strategy="quantile",
        n_bins=10,
        min_samples_per_bins=None,
        random_state=42,
    )
    cal.fit(y_true, y_pred)

    # Should have exactly n_bins + 1 edges
    assert len(cal._bin_edges) == 11
    assert cal.n_bins == 10  # Original n_bins unchanged

    # Test with min_samples_per_bins=20
    cal = CalibrationCurve(
        binning_strategy="quantile",
        n_bins=10,
        min_samples_per_bins=20,
        random_state=42,
    )
    cal.fit(y_true, y_pred)

    # Should have fewer bins due to min_samples_per_bins
    assert len(cal._bin_edges) <= 6  # 100 samples / 20 min_samples + 1

    # Edges should span [0, 1] in both cases
    assert cal._bin_edges[0] == 0
    assert cal._bin_edges[-1] == 1
