"""Core implementation of calibration curves with confidence intervals."""

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


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


class CalibrationCurve:
    """Calibration curve with confidence intervals for binary classifiers.
    
    Parameters
    ----------
    binning_strategy : {'quantile', 'uniform', 'custom'}, default='quantile'
        Strategy to bin the predictions
    n_bins : int, default=10
        Number of bins (ignored if binning_strategy='custom')
    confidence_method : {'clopper_pearson', 'wilson_cc', 'bootstrap'}, default='clopper_pearson'
        Method to compute confidence intervals
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
            
        self.binning_strategy = binning_strategy
        self.n_bins = n_bins
        self.confidence_method = confidence_method
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        
        self._bin_edges = None
        self._prob_true = None
        self._prob_pred = None
        self._ci_lower = None
        self._ci_upper = None
    
    def _compute_bin_edges(self, y_pred):
        """Compute bin edges based on the chosen strategy."""
        if self.binning_strategy == "quantile":
            return np.quantile(
                y_pred,
                np.linspace(0, 1, self.n_bins + 1),
            )
        elif self.binning_strategy == "uniform":
            return np.linspace(0, 1, self.n_bins + 1)
        else:
            if self._bin_edges is None:
                raise ValueError(
                    "For custom binning strategy, bin_edges must be set using set_bin_edges()"
                )
            return self._bin_edges
    
    def set_bin_edges(self, bin_edges):
        """Set custom bin edges.
        
        Parameters
        ----------
        bin_edges : array-like of shape (n_bins + 1,)
            Custom bin edges to use
        """
        bin_edges = np.asarray(bin_edges)
        if not (np.all(np.diff(bin_edges) > 0) and 
                np.isclose(bin_edges[0], 0) and 
                np.isclose(bin_edges[-1], 1)):
            raise ValueError(
                "bin_edges must be strictly increasing and span [0, 1]. "
                f"Got edges from {bin_edges[0]:.3f} to {bin_edges[-1]:.3f} "
                f"with {np.sum(np.diff(bin_edges) <= 0)} non-increasing intervals"
            )
        self._bin_edges = bin_edges
        return self
    
    def _compute_calibration_curve(self, y_true, y_pred):
        """Compute calibration curve for a single set of predictions."""
        bin_edges = self._compute_bin_edges(y_pred)
        n_bins = len(bin_edges) - 1
        
        bin_indices = np.searchsorted(bin_edges, y_pred) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_sums = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        bin_means = np.bincount(bin_indices, weights=y_pred, minlength=n_bins)
        
        prob_true = np.zeros(n_bins)
        prob_pred = np.zeros(n_bins)
        
        # Handle non-empty bins
        mask = bin_counts > 0
        prob_true[mask] = bin_sums[mask] / bin_counts[mask]
        prob_pred[mask] = bin_means[mask] / bin_counts[mask]
        
        # For empty bins, use the bin center as the predicted probability
        empty_mask = ~mask
        if np.any(empty_mask):
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
            prob_pred[empty_mask] = bin_centers[empty_mask]
            # prob_true for empty bins remains 0
        
        return prob_true, prob_pred, bin_counts
    
    def _compute_confidence_intervals(self, y_true, y_pred):
        """Compute confidence intervals using the specified method."""
        if self.confidence_method in ["clopper_pearson", "wilson_cc"]:
            prob_true, prob_pred, bin_counts = self._compute_calibration_curve(
                y_true, y_pred
            )
            
            interval_func = (
                clopper_pearson_interval if self.confidence_method == "clopper_pearson"
                else wilson_cc_interval
            )
            
            ci_lower = np.zeros_like(prob_true)
            ci_upper = np.zeros_like(prob_true)
            
            for i, (successes, trials) in enumerate(zip(
                (prob_true * bin_counts).astype(int), bin_counts
            )):
                ci_lower[i], ci_upper[i] = interval_func(
                    successes, trials, self.confidence_level
                )
            
            return prob_true, prob_pred, ci_lower, ci_upper
        
        elif self.confidence_method == "bootstrap":
            rng = np.random.RandomState(self.random_state)
            n_samples = len(y_true)
            bootstrap_curves = []
            
            # Store original bin edges
            original_bin_edges = self._compute_bin_edges(y_pred)
            n_bins = len(original_bin_edges) - 1
            
            # Create a fine uniform grid for interpolation
            grid_pred = np.linspace(0, 1, 1000)
            
            for i in range(self.n_bootstrap):
                # Bootstrap resample
                indices = rng.randint(0, n_samples, size=n_samples)
                y_true_boot = y_true[indices]
                y_pred_boot = y_pred[indices]
                
                # Count points in each bin for this bootstrap sample
                bin_indices = np.searchsorted(original_bin_edges, y_pred_boot) - 1
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)
                bin_counts = np.bincount(bin_indices, minlength=n_bins)
                
                # Randomly perturb bin edges by either merging or splitting
                perturbed_edges = original_bin_edges.copy()
                if rng.random() < 0.5 and n_bins > 2:
                    # Compute merge probabilities based on sum of adjacent bin counts
                    merge_probs = np.zeros(n_bins - 1)
                    for j in range(n_bins - 1):
                        merge_probs[j] = bin_counts[j] + bin_counts[j + 1]
                    merge_probs = merge_probs / merge_probs.sum()
                    
                    # Merge two adjacent bins with probability proportional to their counts
                    merge_idx = rng.choice(n_bins - 1, p=merge_probs)
                    perturbed_edges = np.delete(perturbed_edges, merge_idx + 1)
                else:
                    # Split probabilities proportional to bin counts
                    split_probs = bin_counts / bin_counts.sum()
                    
                    # Choose bin to split with probability proportional to its count
                    split_idx = rng.choice(n_bins, p=split_probs)
                    
                    # Split point weighted by density within the bin
                    bin_points = y_pred_boot[bin_indices == split_idx]
                    if len(bin_points) > 0:
                        # Use a random point from the actual data in this bin
                        split_point = rng.choice(bin_points)
                    else:
                        # Fallback to uniform if bin is empty
                        split_point = rng.uniform(
                            perturbed_edges[split_idx],
                            perturbed_edges[split_idx + 1]
                        )
                    perturbed_edges = np.sort(np.insert(perturbed_edges, split_idx + 1, split_point))
                
                # Use perturbed bin edges for this iteration
                self._bin_edges = perturbed_edges
                prob_true_boot, prob_pred_boot, _ = self._compute_calibration_curve(
                    y_true_boot, y_pred_boot
                )
                
                # Interpolate onto the fine grid
                # Use linear interpolation and extend the first/last values for out-of-bounds
                f = interp1d(
                    prob_pred_boot,
                    prob_true_boot,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(prob_true_boot[0], prob_true_boot[-1])
                )
                grid_true = f(grid_pred)
                bootstrap_curves.append(grid_true)
            
            # Reset to original bin edges and compute final curve
            self._bin_edges = original_bin_edges
            prob_true, prob_pred, _ = self._compute_calibration_curve(y_true, y_pred)
            
            # Compute confidence intervals on the grid
            bootstrap_curves = np.array(bootstrap_curves)
            alpha = (1 - self.confidence_level) / 2
            grid_ci_lower = np.percentile(bootstrap_curves, alpha * 100, axis=0)
            grid_ci_upper = np.percentile(bootstrap_curves, (1 - alpha) * 100, axis=0)
            
            # Interpolate confidence intervals back to original prediction points
            f_lower = interp1d(
                grid_pred,
                grid_ci_lower,
                kind='linear',
                bounds_error=False,
                fill_value=(grid_ci_lower[0], grid_ci_lower[-1])
            )
            f_upper = interp1d(
                grid_pred,
                grid_ci_upper,
                kind='linear',
                bounds_error=False,
                fill_value=(grid_ci_upper[0], grid_ci_upper[-1])
            )
            
            ci_lower = f_lower(prob_pred)
            ci_upper = f_upper(prob_pred)
            
            return prob_true, prob_pred, ci_lower, ci_upper
        
        else:
            raise ValueError(
                f"confidence_method must be one of ['clopper_pearson', 'wilson_cc', 'bootstrap'], "
                f"got '{self.confidence_method}'"
            )

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
        
        if not (y_true.ndim == y_pred.ndim == 1 and len(y_true) == len(y_pred)):
            raise ValueError(
                "y_true and y_pred must be 1D arrays of same length. "
                f"Got shapes y_true: {y_true.shape}, y_pred: {y_pred.shape}"
            )
        
        if not np.all((y_true == 0) | (y_true == 1)):
            invalid_values = y_true[(y_true != 0) & (y_true != 1)]
            raise ValueError(
                "y_true must contain only 0 and 1. "
                f"Found invalid values: {np.unique(invalid_values)}"
            )
        
        if not np.all((y_pred >= 0) & (y_pred <= 1)):
            invalid_mask = (y_pred < 0) | (y_pred > 1)
            invalid_values = y_pred[invalid_mask]
            raise ValueError(
                "y_pred must contain probabilities in [0, 1]. "
                f"Found {np.sum(invalid_mask)} values outside range: "
                f"min={np.min(invalid_values):.3f}, max={np.max(invalid_values):.3f}"
            )
        
        (
            self._prob_true,
            self._prob_pred,
            self._ci_lower,
            self._ci_upper,
        ) = self._compute_confidence_intervals(y_true, y_pred)
        
        return self
    
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
            self._prob_true is None or
            self._prob_pred is None or
            self._ci_lower is None or
            self._ci_upper is None
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
            label=f"{self.confidence_level:.0%} confidence interval"
        )
        
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration Curve")
        ax.legend(loc="lower right")
        ax.grid(True)
        
        return ax
