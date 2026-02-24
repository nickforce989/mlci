"""
Bootstrap confidence intervals and summary statistics for ExperimentResults.

The key insight: every ML evaluation metric is a random variable.
Report distributions, not point estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from mlci.core.results import ExperimentResults


@dataclass
class BootstrapCI:
    """
    Result of a bootstrap confidence interval computation.

    Attributes
    ----------
    mean : float
        Point estimate (mean of observed scores).
    lower : float
        Lower bound of the confidence interval.
    upper : float
        Upper bound of the confidence interval.
    confidence : float
        Confidence level (e.g. 0.95).
    n_bootstrap : int
        Number of bootstrap resamples used.
    bootstrap_distribution : np.ndarray
        Full bootstrap distribution (useful for plotting).
    """

    mean: float
    lower: float
    upper: float
    confidence: float = 0.95
    n_bootstrap: int = 10_000
    bootstrap_distribution: Optional[np.ndarray] = None

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def __repr__(self) -> str:
        pct = int(self.confidence * 100)
        return (
            f"{self.mean:.4f} "
            f"[{pct}% CI: {self.lower:.4f}, {self.upper:.4f}] "
            f"(width={self.width:.4f})"
        )


def bootstrap_ci(
    results: ExperimentResults,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    statistic: str = "mean",
    seed: int = 42,
) -> BootstrapCI:
    """
    Compute a bootstrap confidence interval for a summary statistic of
    an ExperimentResults object.

    Parameters
    ----------
    results : ExperimentResults
    confidence : float
        Desired confidence level. Default 0.95.
    n_bootstrap : int
        Number of bootstrap resamples. Default 10_000.
    statistic : str
        Which statistic to bootstrap: "mean" (default), "median", "std",
        "min", "max".
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    BootstrapCI

    Notes
    -----
    We resample at the seed level (rows of results.scores), treating each
    seed as an independent experimental unit. This is the correct choice
    because seeds are independent; splits within a seed share training
    data and are not independent.
    """
    stat_fns = {
        "mean":   np.mean,
        "median": np.median,
        "std":    lambda x: np.std(x, ddof=1),
        "min":    np.min,
        "max":    np.max,
    }
    if statistic not in stat_fns:
        raise ValueError(f"Unknown statistic '{statistic}'. Choose from {list(stat_fns)}")

    stat_fn = stat_fns[statistic]

    # Use per-seed means as the units of resampling
    seed_means = results.seed_means  # shape: (n_seeds,)
    n = len(seed_means)
    observed = float(stat_fn(seed_means))

    rng = np.random.default_rng(seed)
    boot_stats = np.array([
        stat_fn(rng.choice(seed_means, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return BootstrapCI(
        mean=observed,
        lower=lower,
        upper=upper,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        bootstrap_distribution=boot_stats,
    )


def summary(
    results: ExperimentResults,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
) -> str:
    """
    Print a human-readable summary of an ExperimentResults object.

    Includes point estimate, confidence interval, seed variance,
    split variance, and inter-quartile range.
    """
    ci = bootstrap_ci(results, confidence=confidence, n_bootstrap=n_bootstrap)
    seed_std = float(np.std(results.seed_means, ddof=1)) if results.n_seeds > 1 else float("nan")
    split_std = float(np.std(results.split_means, ddof=1)) if results.n_splits > 1 else float("nan")
    q25, q75 = np.percentile(results.flat, [25, 75])

    pct = int(confidence * 100)
    lines = [
        f"{'─'*52}",
        f"  Model   : {results.model_name}",
        f"  Metric  : {results.metric}",
        f"  Seeds   : {results.n_seeds}   Splits: {results.n_splits}",
        f"{'─'*52}",
        f"  Mean    : {ci.mean:.4f}",
        f"  {pct}% CI  : [{ci.lower:.4f}, {ci.upper:.4f}]  (±{ci.width/2:.4f})",
        f"  Std     : {results.std:.4f}",
        f"  IQR     : [{q25:.4f}, {q75:.4f}]",
    ]
    if results.n_seeds > 1:
        lines.append(f"  Seed σ  : {seed_std:.4f}  (variance from random init)")
    if results.n_splits > 1:
        lines.append(f"  Split σ : {split_std:.4f}  (variance from data split)")
    lines.append(f"{'─'*52}")

    output = "\n".join(lines)
    print(output)
    return output
