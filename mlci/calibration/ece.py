"""
Calibration analysis for mlci.

Answers the question: "Not just how accurate is my model, but how well-calibrated
are its probability estimates — and is that calibration consistent across seeds?"

What sklearn gives you:
  - A single calibration_curve call (one set of probabilities, one run)
  - No uncertainty on ECE or reliability diagram

What mlci adds:
  - ECE computed across all (seed, split) pairs → distribution of ECE values
  - Reliability diagram with uncertainty bands (showing how calibration varies
    across seeds, not just a single curve)
  - ACE (Adaptive Calibration Error) and MCE (Maximum Calibration Error)
  - Bootstrap CI on ECE, just like bootstrap_ci gives CI on accuracy

Key references
--------------
Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining Well Calibrated
Probabilities Using Bayesian Binning. AAAI.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of
Modern Neural Networks. ICML.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np


# -------------------------------------------------------------------------
# ECE computation
# -------------------------------------------------------------------------

def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """
    Expected Calibration Error (ECE).

    Measures the weighted average gap between confidence and accuracy across
    equal-width (uniform) or equal-frequency (quantile) bins.

    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    Parameters
    ----------
    y_true : array of int, shape (n_samples,)
        True binary labels (0 or 1).
    y_prob : array of float, shape (n_samples,)
        Predicted probabilities for the positive class.
    n_bins : int
        Number of confidence bins. Default 10.
    strategy : str
        "uniform" (equal-width bins) or "quantile" (equal-frequency bins).

    Returns
    -------
    float
        ECE in [0, 1]. Lower is better.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(y_prob, quantiles)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    else:
        raise ValueError(f"strategy must be 'uniform' or 'quantile', got '{strategy}'.")

    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if lo == bin_edges[-2]:  # last bin is inclusive
            mask = (y_prob >= lo) & (y_prob <= hi)
        n_b = mask.sum()
        if n_b == 0:
            continue
        acc_b = float(y_true[mask].mean())
        conf_b = float(y_prob[mask].mean())
        ece += (n_b / n) * abs(acc_b - conf_b)

    return float(ece)


def compute_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Maximum Calibration Error (MCE).

    The worst-case calibration gap across all bins.

    MCE = max_b |acc(B_b) - conf(B_b)|
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    mce = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if lo == bin_edges[-2]:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc_b = float(y_true[mask].mean())
        conf_b = float(y_prob[mask].mean())
        mce = max(mce, abs(acc_b - conf_b))

    return float(mce)


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute data for a reliability (calibration) diagram.

    Parameters
    ----------
    y_true : array of int
    y_prob : array of float
    n_bins : int
    strategy : str  "uniform" or "quantile"

    Returns
    -------
    bin_confidence : np.ndarray, shape (n_bins,)
        Mean predicted confidence in each bin. NaN if bin is empty.
    bin_accuracy : np.ndarray, shape (n_bins,)
        Fraction of positive labels in each bin. NaN if bin is empty.
    bin_count : np.ndarray of int, shape (n_bins,)
        Number of samples in each bin.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(y_prob, quantiles)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0

    bin_conf = np.full(n_bins, np.nan)
    bin_acc = np.full(n_bins, np.nan)
    bin_cnt = np.zeros(n_bins, dtype=int)

    for b, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (y_prob >= lo) & (y_prob < hi)
        if b == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        n_b = mask.sum()
        bin_cnt[b] = n_b
        if n_b == 0:
            continue
        bin_conf[b] = float(y_prob[mask].mean())
        bin_acc[b] = float(y_true[mask].mean())

    return bin_conf, bin_acc, bin_cnt


# -------------------------------------------------------------------------
# CalibrationResults container
# -------------------------------------------------------------------------

@dataclass
class CalibrationResults:
    """
    Calibration analysis across multiple seeds and splits.

    Attributes
    ----------
    ece_scores : np.ndarray, shape (n_seeds, n_splits)
        ECE values for each (seed, split) evaluation.
    mce_scores : np.ndarray, shape (n_seeds, n_splits)
        MCE values for each (seed, split) evaluation.
    bin_confidences : np.ndarray, shape (n_seeds * n_splits, n_bins)
        Per-run mean confidence per bin. NaN where bins are empty.
    bin_accuracies : np.ndarray, shape (n_seeds * n_splits, n_bins)
        Per-run accuracy per bin. NaN where bins are empty.
    bin_counts : np.ndarray, shape (n_seeds * n_splits, n_bins)
    bin_edges : np.ndarray, shape (n_bins + 1,)
    n_bins : int
    model_name : str
    metric : str
        Always "ECE".
    n_seeds : int
    n_splits : int
    strategy : str
    """

    ece_scores: np.ndarray          # (n_seeds, n_splits)
    mce_scores: np.ndarray          # (n_seeds, n_splits)
    bin_confidences: np.ndarray     # (n_seeds * n_splits, n_bins)
    bin_accuracies: np.ndarray      # (n_seeds * n_splits, n_bins)
    bin_counts: np.ndarray          # (n_seeds * n_splits, n_bins)
    bin_edges: np.ndarray           # (n_bins + 1,)
    n_bins: int
    model_name: str = "model"
    metric: str = "ECE"
    n_seeds: int = 1
    n_splits: int = 1
    strategy: str = "uniform"

    # ------------------------------------------------------------------ #
    # Derived statistics

    @property
    def mean_ece(self) -> float:
        return float(np.nanmean(self.ece_scores))

    @property
    def std_ece(self) -> float:
        return float(np.nanstd(self.ece_scores, ddof=1))

    @property
    def mean_mce(self) -> float:
        return float(np.nanmean(self.mce_scores))

    @property
    def ece_ci(self) -> Tuple[float, float]:
        """Bootstrap 95% CI on ECE (resampled at seed level)."""
        seed_means = self.ece_scores.mean(axis=1)  # (n_seeds,)
        rng = np.random.default_rng(42)
        boot = np.array([
            rng.choice(seed_means, size=len(seed_means), replace=True).mean()
            for _ in range(10_000)
        ])
        return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    @property
    def mean_reliability_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mean reliability curve (confidence, accuracy) averaged across all runs.

        Returns
        -------
        mean_conf : np.ndarray, shape (n_bins,)
        mean_acc  : np.ndarray, shape (n_bins,)
        """
        mean_conf = np.nanmean(self.bin_confidences, axis=0)
        mean_acc = np.nanmean(self.bin_accuracies, axis=0)
        return mean_conf, mean_acc

    @property
    def reliability_curve_std(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Std of reliability curve across runs.

        Returns
        -------
        std_conf : np.ndarray
        std_acc  : np.ndarray
        """
        std_conf = np.nanstd(self.bin_confidences, axis=0, ddof=1)
        std_acc = np.nanstd(self.bin_accuracies, axis=0, ddof=1)
        return std_conf, std_acc

    def reliability_curve_ci(
        self, confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Bootstrap CI on the reliability curve, resampled at seed level.

        Returns lower and upper bounds on mean accuracy per bin.

        Returns
        -------
        mean_conf, mean_acc, acc_lower, acc_upper : np.ndarray each (n_bins,)
        """
        n_runs = self.bin_accuracies.shape[0]
        n_seeds_actual = self.n_seeds

        # Reshape to (n_seeds, n_splits, n_bins) then take seed means
        accs = self.bin_accuracies.reshape(n_seeds_actual, self.n_splits, self.n_bins)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            seed_accs = np.nanmean(accs, axis=1)  # (n_seeds, n_bins)

        rng = np.random.default_rng(42)
        alpha_h = (1 - confidence) / 2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            boot_means = np.array([
                np.nanmean(
                    rng.choice(seed_accs, size=n_seeds_actual, replace=True, axis=0),
                    axis=0
                )
                for _ in range(2_000)
            ])  # (2000, n_bins)

        mean_conf, mean_acc = self.mean_reliability_curve
        acc_lower = np.nanpercentile(boot_means, 100 * alpha_h, axis=0)
        acc_upper = np.nanpercentile(boot_means, 100 * (1 - alpha_h), axis=0)

        return mean_conf, mean_acc, acc_lower, acc_upper

    def summary(self) -> str:
        ci_lo, ci_hi = self.ece_ci
        lines = [
            f"{'─' * 55}",
            f"  Calibration Summary: {self.model_name}",
            f"{'─' * 55}",
            f"  Seeds   : {self.n_seeds}   Splits : {self.n_splits}",
            f"  Bins    : {self.n_bins} ({self.strategy})",
            f"{'─' * 55}",
            f"  ECE     : {self.mean_ece:.4f}  (± {self.std_ece:.4f})",
            f"  95% CI  : [{ci_lo:.4f}, {ci_hi:.4f}]",
            f"  MCE     : {self.mean_mce:.4f}",
            f"{'─' * 55}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()