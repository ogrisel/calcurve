"""Core implementation of calibration curves with confidence intervals."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d


def clopper_pearson_interval(successes, trials, confidence_level=0.90):
    """Compute Clopper-Pearson exact confidence interval for binomial proportion.

    Parameters
    ----------
    successes : int
        Number of successes
    trials : int
        Number of trials
    confidence_level : float, default=0.90
        Confidence level for the interval

    Returns
    -------
    lower, upper : tuple of float
        Lower and upper bounds of the confidence interval
    """
    alpha = 1 - confidence_level
    if trials == 0:
        return 0.0, 1.0

    # Handle edge cases
    if successes == 0:
        lower = 0.0
        upper = 1.0 - (alpha / 2) ** (1.0 / trials)
    elif successes == trials:
        lower = (alpha / 2) ** (1.0 / trials)
        upper = 1.0
    else:
        lower = stats.beta.ppf(alpha / 2, successes, trials - successes + 1)
        upper = stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes)

    return lower, upper


def wilson_cc_interval(successes, trials, confidence_level=0.90):
    """Compute Wilson score interval with continuity correction.

    Parameters
    ----------
    successes : int
        Number of successes
    trials : int
        Number of trials
    confidence_level : float, default=0.90
        Confidence level for the interval

    Returns
    -------
    lower, upper : tuple of float
        Lower and upper bounds of the confidence interval
    """
    if trials == 0:
        return 0.0, 1.0

    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    p = float(successes) / trials

    # Correction factor
    c = 1 / (2 * trials)

    # Wilson score interval with continuity correction
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    spread = z / denominator * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))

    lower = max(0, center - spread - c)
    upper = min(1, center + spread + c)

    return lower, upper


def binom_test_interval(
    successes, trials, confidence_level=0.90, method="clopper_pearson"
):
    """Compute confidence interval using scipy.stats.binomtest.

    Parameters
    ----------
    successes : int
        Number of successes
    trials : int
        Number of trials
    confidence_level : float, default=0.90
        Confidence level for the interval

    Returns
    -------
    lower, upper : tuple of float
        Lower and upper bounds of the confidence interval
    """
    if trials == 0:
        return 0.0, 1.0

    if method == "clopper_pearson":
        scipy_method = "exact"
    elif method == "wilson_cc":
        scipy_method = "wilsoncc"
    else:
        raise ValueError(
            f"Method {method} not recognized, expected one of "
            "['clopper_pearson', 'wilson_cc']"
        )

    result = stats.binomtest(successes, trials, alternative="two-sided")
    lower, upper = result.proportion_ci(
        confidence_level=confidence_level, method=scipy_method
    )
    return lower, upper


class CalibrationCurve:
    """Compute and plot calibration curves with confidence intervals.

    A calibration curve shows the relationship between predicted probabilities
    and the true proportion of positive samples. A perfectly calibrated model
    would have a calibration curve that follows the diagonal y=x line.

    Parameters
    ----------
    binning_strategy : {'quantile', 'uniform', 'custom'}, default='quantile'
        Strategy to bin the predictions
    n_bins : int, default=10
        Number of bins (ignored if binning_strategy='custom')
    min_samples_per_bins : int, default=None
        Minimum number of samples required in each bin. If a bin contains fewer
        samples than this threshold, it will be merged with adjacent bins until
        the threshold is met. This helps ensure reliable calibration estimates
        by avoiding bins with too few samples. Must be at least 1 or None.

        For example, if min_samples_per_bins=20 and a bin contains only 5 samples,
        it will be merged with adjacent bins until the combined bin has at least
        20 samples. This is particularly useful when:
        - Working with imbalanced datasets
        - Using uniform binning with sparse regions
        - Needing robust confidence interval estimates

    confidence_method : str, default='clopper_pearson'
        Method to compute confidence intervals. One of:
        - 'clopper_pearson': exact confidence interval
        - 'wilson_cc': Wilson score with continuity correction
        - 'bootstrap': bootstrap resampling with interpolation
    confidence_level : float, default=0.90
        Confidence level for the intervals
    n_bootstrap : int, default=100
        Number of bootstrap iterations (only used if confidence_method='bootstrap')
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the bootstrap resampling.
        Pass an int for reproducible output across multiple function calls.
    """

    def __init__(
        self,
        binning_strategy="quantile",
        n_bins=10,
        min_samples_per_bins=None,
        confidence_method="clopper_pearson",
        confidence_level=0.90,
        n_bootstrap=100,
        random_state=None,
    ):
        # Validate binning strategy
        valid_strategies = ["quantile", "uniform", "custom"]
        if binning_strategy not in valid_strategies:
            raise ValueError(
                f"binning_strategy must be one of {valid_strategies}, "
                f"got '{binning_strategy}'"
            )

        if min_samples_per_bins is not None and min_samples_per_bins < 1:
            raise ValueError("min_samples_per_bins must be at least 1 or None")

        # Validate confidence method
        valid_methods = ["clopper_pearson", "wilson_cc", "bootstrap"]
        if confidence_method not in valid_methods:
            raise ValueError(
                f"confidence_method must be one of {valid_methods}, "
                f"got '{confidence_method}'"
            )

        # Validate other parameters
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")

        if n_bins < 1:
            raise ValueError("n_bins must be positive")

        if confidence_method == "bootstrap" and n_bootstrap < 1:
            raise ValueError("n_bootstrap must be positive")

        self.binning_strategy = binning_strategy
        self.n_bins = n_bins
        self.min_samples_per_bins = min_samples_per_bins
        self.confidence_method = confidence_method
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

        self._bin_counts = None
        self._bin_edges = None
        self._prob_true = None
        self._prob_pred = None
        self._ci_lower = None
        self._ci_upper = None

    def set_bin_edges(self, bin_edges):
        """Set custom bin edges.

        Parameters
        ----------
        bin_edges : array-like of shape (n_bins + 1,)
            Custom bin edges to use
        """
        bin_edges = np.asarray(bin_edges)
        if not (
            np.all(np.diff(bin_edges) > 0)
            and np.isclose(bin_edges[0], 0)
            and np.isclose(bin_edges[-1], 1)
        ):
            raise ValueError(
                "bin_edges must be strictly increasing and span [0, 1]. "
                f"Got edges from {bin_edges[0]:.3f} to {bin_edges[-1]:.3f} "
                f"with {np.sum(np.diff(bin_edges) <= 0)} non-increasing intervals"
            )
        self._bin_edges = bin_edges
        return self

    def _merge_small_bins(self, bin_edges, y_pred):
        """Merge bins that have fewer than min_samples_per_bins samples.

        The bin with the fewest samples is merged with its neighbor (left or right)
        that has fewer samples. This process is repeated until all bins have at least
        min_samples_per_bins samples.

        Parameters
        ----------
        bin_edges : array-like of shape (n_bins + 1,)
            Bin edges to use for merging

        y_pred : array-like
            Predicted probabilities used to compute bin counts

        Returns
        -------
        bin_edges : array-like of shape (n_bins + 1,)
            Merged bin edges
        """
        if self.min_samples_per_bins is None:
            return bin_edges

        while True:
            # Count samples in each bin
            bin_indices = np.searchsorted(bin_edges, y_pred) - 1
            bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
            bin_counts = np.bincount(bin_indices, minlength=len(bin_edges) - 1)

            # Find smallest bin that's below threshold
            small_bins = np.where(bin_counts < self.min_samples_per_bins)[0]
            if len(small_bins) == 0:
                break

            smallest_bin = small_bins[np.argmin(bin_counts[small_bins])]

            # Decide which neighbor to merge with (prefer the smaller one)
            left_count = bin_counts[smallest_bin - 1] if smallest_bin > 0 else np.inf
            right_count = (
                bin_counts[smallest_bin + 1]
                if smallest_bin < len(bin_counts) - 1
                else np.inf
            )

            # Merge with the smaller neighbor
            if left_count <= right_count and smallest_bin > 0:
                merge_idx = smallest_bin
            else:
                merge_idx = smallest_bin + 1

            # Remove the bin edge to merge bins
            bin_edges = np.delete(bin_edges, merge_idx)

        return bin_edges

    def _compute_bin_edges(self, y_pred):
        """Compute bin edges based on the chosen strategy."""
        if self.binning_strategy == "quantile":
            # For quantile binning, adjust n_bins based on min_samples_per_bins
            if self.min_samples_per_bins is not None:
                n_samples = len(y_pred)
                max_bins = max(1, n_samples // self.min_samples_per_bins)
                n_bins = min(self.n_bins, max_bins)
            else:
                n_bins = self.n_bins

            # Use quantiles to ensure roughly equal number of samples per bin
            edges = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
            # Ensure edges span [0, 1]
            edges[0] = 0
            edges[-1] = 1
        elif self.binning_strategy == "uniform":
            edges = np.linspace(0, 1, self.n_bins + 1)
        else:
            if self._bin_edges is None:
                raise ValueError(
                    "For custom binning strategy, bin_edges must be set using "
                    "set_bin_edges()"
                )
            edges = self._bin_edges

        return self._merge_small_bins(edges, y_pred)

    def _compute_calibration_curve(self, y_true, y_pred):
        """Compute calibration curve for a single set of predictions."""
        bin_edges = self._compute_bin_edges(y_pred)
        n_bins = len(bin_edges) - 1

        bin_indices = np.searchsorted(bin_edges, y_pred) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        bin_sums = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        bin_means = np.bincount(bin_indices, weights=y_pred, minlength=n_bins)

        # Initialize prob_true to 0.5 as least informative prior.
        prob_true = np.ones(n_bins) / 2

        # Initialize prob_pred to edge mid-points.
        prob_pred = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Handle non-empty bins
        mask = bin_counts > 0
        prob_true[mask] = bin_sums[mask] / bin_counts[mask]
        prob_pred[mask] = bin_means[mask] / bin_counts[mask]

        return prob_true, prob_pred, bin_counts, bin_edges

    def _compute_confidence_intervals(
        self, y_true, y_pred, prob_true, prob_pred, bin_counts, bin_edges
    ):
        """Compute confidence intervals using the specified method."""

        # TODO: use scipy.stats.binom_test instead of our own implementation.
        if self.confidence_method in ["clopper_pearson", "wilson_cc"]:
            ci_lower = np.zeros_like(prob_true)
            ci_upper = np.ones_like(prob_true)

            # Compute confidence intervals for each bin
            for i, (successes, trials) in enumerate(
                zip(bin_counts * prob_true, bin_counts)
            ):
                if trials > 0:
                    ci_lower[i], ci_upper[i] = binom_test_interval(
                        int(round(successes)),
                        int(trials),
                        self.confidence_level,
                        method=self.confidence_method,
                    )

            return ci_lower, ci_upper

        # Bootstrap confidence intervals
        rng = np.random.RandomState(self.random_state)
        n_samples = len(y_true)
        bootstrap_curves = []

        # Grid used for interpolation.
        y_pred_grid = np.linspace(bin_edges[0], bin_edges[-1], 1000)

        for _ in range(self.n_bootstrap):
            # Sample with replacement
            indices = rng.randint(0, n_samples, size=n_samples)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            # Compute calibration curve for this bootstrap sample
            prob_true_boot, prob_pred_boot, _, _ = self._compute_calibration_curve(
                y_true_boot, y_pred_boot
            )

            # Interpolated calibration curve on the grid.
            prob_true_boot = interp1d(
                prob_pred_boot,
                prob_true_boot,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )(y_pred_grid)

            bootstrap_curves.append(prob_true_boot)

        # Compute mean curve and confidence intervals
        bootstrap_curves = np.array(bootstrap_curves)
        ci_lower = np.quantile(
            bootstrap_curves, (1 - self.confidence_level) / 2, axis=0
        )
        ci_upper = np.quantile(
            bootstrap_curves, (1 + self.confidence_level) / 2, axis=0
        )

        # Interpolate back to prob_pred locations:
        ci_lower = interp1d(
            y_pred_grid,
            ci_lower,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )(prob_pred).clip(0, 1)
        ci_upper = interp1d(
            y_pred_grid,
            ci_upper,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )(prob_pred).clip(0, 1)
        return ci_lower, ci_upper

    def fit(self, y_true, y_pred):
        """Compute calibration curve and confidence intervals.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels
        y_pred : array-like of shape (n_samples,)
            Predicted probabilities

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")

        if not np.all((y_true == 0) | (y_true == 1)):
            raise ValueError("y_true must contain only 0 and 1")

        if not np.all((y_pred >= 0) & (y_pred <= 1)):
            raise ValueError("y_pred must contain only values between 0 and 1")

        # Validate confidence method
        valid_methods = ["clopper_pearson", "wilson_cc", "bootstrap"]
        if self.confidence_method not in valid_methods:
            raise ValueError(
                f"confidence_method must be one of {valid_methods}, "
                f"got '{self.confidence_method}'"
            )

        # First compute calibration curve to get bin counts
        (
            self._prob_true,
            self._prob_pred,
            self._bin_counts,
            self._bin_edges,
        ) = self._compute_calibration_curve(y_true, y_pred)

        self._ci_lower, self._ci_upper = self._compute_confidence_intervals(
            y_true,
            y_pred,
            self._prob_true,
            self._prob_pred,
            self._bin_counts,
            self._bin_edges,
        )

        return self

    @property
    def bin_counts(self):
        """Get the number of samples in each bin after fitting.

        Returns
        -------
        ndarray of shape (n_bins,)
            Number of samples in each bin. The length may be less than the
            original n_bins if bins were merged due to min_samples_per_bins.

        Raises
        ------
        ValueError
            If the calibration curve has not been fitted yet.
        """
        if not hasattr(self, "_bin_counts") or self._bin_counts is None:
            raise ValueError("Must call fit before accessing bin_counts")
        return self._bin_counts

    def plot(self, ax=None):
        """Plot calibration curve with confidence intervals.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            Axes object to plot on. If None, uses current axes.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object containing the plot.
        """
        if (
            self._prob_true is None
            or self._prob_pred is None
            or self._ci_lower is None
            or self._ci_upper is None
        ):
            raise RuntimeError("Must call fit() before plot()")

        if ax is None:
            ax = plt.gca()

        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        # Plot calibration curve and confidence interval
        ax.plot(self._prob_pred, self._prob_true, ".-", label="Calibration curve")
        ax.fill_between(
            self._prob_pred,
            self._ci_lower,
            self._ci_upper,
            alpha=0.2,
            label=f"{self.confidence_level:.0%} confidence interval",
        )

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration Curve")
        ax.legend(loc="lower right")
        ax.grid(True)

        return ax
