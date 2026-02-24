"""
Statistical tests for comparing two ML models.

Implements:
1. Paired Wilcoxon signed-rank test (non-parametric, paired by seed)
2. Corrected resampled t-test (Nadeau & Bengio, 2003) — the main contribution
3. Bayesian comparison via Beta posterior (no PyMC required for basic version)

The Nadeau-Bengio corrected t-test is the key method here. It is specifically
designed for comparing ML models evaluated via k-fold cross-validation. It
corrects for the fact that folds share training data, making the observations
non-independent — a fact that the standard paired t-test ignores, leading to
systematically overconfident conclusions.

Reference:
    Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error.
    Machine Learning, 52(3), 239–281.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

from mlci.core.results import ExperimentResults


@dataclass
class ComparisonResult:
    """
    Output of a statistical comparison between two models.

    Attributes
    ----------
    model_a : str
    model_b : str
    method : str
        Name of the statistical test used.
    statistic : float
        Test statistic (t-value, z-value, etc.)
    p_value : float
        Two-sided p-value. For Bayesian method, this is None.
    effect_size : float
        Mean(A) - Mean(B). Positive means A is better (if higher_is_better).
    ci_lower, ci_upper : float
        95% CI on the effect size.
    prob_a_better : float
        Estimated probability that model A is better than model B.
    conclusion : str
        Human-readable conclusion.
    alpha : float
        Significance threshold used.
    """

    model_a: str
    model_b: str
    method: str
    statistic: Optional[float]
    p_value: Optional[float]
    effect_size: float
    ci_lower: float
    ci_upper: float
    prob_a_better: float
    conclusion: str
    alpha: float = 0.05

    def __repr__(self) -> str:
        lines = [
            f"{'─'*60}",
            f"  Comparison : {self.model_a}  vs  {self.model_b}",
            f"  Method     : {self.method}",
            f"  Effect     : {self.effect_size:+.4f}  [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]",
            f"  P(A > B)   : {self.prob_a_better:.3f}",
        ]
        if self.p_value is not None:
            lines.append(f"  p-value    : {self.p_value:.4f}  (α={self.alpha})")
        lines += [
            f"  Conclusion : {self.conclusion}",
            f"{'─'*60}",
        ]
        return "\n".join(lines)


# -----------------------------------------------------------------------
# Corrected Resampled t-test (Nadeau & Bengio, 2003)
# -----------------------------------------------------------------------

def corrected_resampled_ttest(
    results_a: ExperimentResults,
    results_b: ExperimentResults,
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    boot_seed: int = 42,
) -> ComparisonResult:
    """
    Corrected resampled t-test for comparing two models on the same CV folds.

    This is the statistically correct test when both models were evaluated on
    the same k-fold splits. The standard paired t-test is anti-conservative
    here because the folds share training data. Nadeau & Bengio (2003) derived
    a corrected variance estimator that accounts for this.

    Both results objects must have the same shape (same seeds × same splits).

    Parameters
    ----------
    results_a, results_b : ExperimentResults
        Must have identical (n_seeds, n_splits) shapes.
    alpha : float
        Significance threshold. Default 0.05.
    n_bootstrap : int
        Bootstrap resamples for the CI on effect size.
    boot_seed : int
        RNG seed for bootstrap.

    Returns
    -------
    ComparisonResult

    Notes
    -----
    The corrected variance formula is:
        Var_corrected = (1/k + n_test/n_train) * sigma^2
    where k = n_splits, sigma^2 = sample variance of per-fold differences,
    n_test = fold size, n_train = training set size.
    """

    _check_compatible(results_a, results_b)

    n_seeds, n_splits = results_a.scores.shape
    n = len(results_a.flat)  # total observations

    diff = results_a.scores - results_b.scores  # (n_seeds, n_splits)

    # Per-split mean differences (averaged across seeds)
    fold_diffs = diff.mean(axis=0)  # (n_splits,)

    mean_diff = float(fold_diffs.mean())
    sigma2 = float(np.var(fold_diffs, ddof=1))

    # Nadeau-Bengio correction factor
    # Assumes balanced folds: n_test = N / n_splits
    N = len(results_a.flat)  # proxy for dataset size
    n_test = N / n_splits
    n_train = N - n_test
    correction = 1.0 / n_splits + n_test / n_train

    corrected_var = correction * sigma2
    corrected_se = float(np.sqrt(corrected_var / n_seeds))

    if corrected_se == 0:
        t_stat = 0.0
        p_value = 1.0
    else:
        t_stat = mean_diff / corrected_se
        # Degrees of freedom: (n_splits - 1)
        df = n_splits - 1
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=df))

    # Bootstrap CI on mean effect size
    ci_lower, ci_upper = _bootstrap_effect_ci(
        results_a.seed_means, results_b.seed_means,
        n_bootstrap=n_bootstrap, seed=boot_seed,
    )

    prob_a_better = _prob_a_better(results_a, results_b)

    if results_a.higher_is_better:
        a_wins = mean_diff > 0
    else:
        a_wins = mean_diff < 0

    if p_value < alpha:
        winner = results_a.model_name if a_wins else results_b.model_name
        conclusion = (
            f"Statistically significant difference (p={p_value:.4f} < α={alpha}). "
            f"{winner} is better."
        )
    else:
        conclusion = (
            f"No statistically significant difference (p={p_value:.4f} ≥ α={alpha}). "
            f"Cannot conclude either model is better."
        )

    return ComparisonResult(
        model_a=results_a.model_name,
        model_b=results_b.model_name,
        method="Corrected Resampled t-test (Nadeau & Bengio, 2003)",
        statistic=t_stat,
        p_value=p_value,
        effect_size=mean_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        prob_a_better=prob_a_better,
        conclusion=conclusion,
        alpha=alpha,
    )


# -----------------------------------------------------------------------
# Wilcoxon signed-rank test
# -----------------------------------------------------------------------

def wilcoxon_test(
    results_a: ExperimentResults,
    results_b: ExperimentResults,
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    boot_seed: int = 42,
) -> ComparisonResult:
    """
    Paired Wilcoxon signed-rank test, paired by seed.

    Non-parametric. Makes no distributional assumptions.
    Appropriate when the per-seed score differences are not normally distributed.

    Parameters
    ----------
    results_a, results_b : ExperimentResults
    alpha : float
    n_bootstrap, boot_seed : bootstrap CI parameters.

    Returns
    -------
    ComparisonResult
    """

    _check_compatible(results_a, results_b)

    a_means = results_a.seed_means
    b_means = results_b.seed_means
    diffs = a_means - b_means

    if np.all(diffs == 0):
        stat, p_value = 0.0, 1.0
    else:
        stat, p_value = stats.wilcoxon(diffs, alternative="two-sided")

    mean_diff = float(diffs.mean())
    ci_lower, ci_upper = _bootstrap_effect_ci(
        a_means, b_means, n_bootstrap=n_bootstrap, seed=boot_seed
    )
    prob_a_better = _prob_a_better(results_a, results_b)

    if results_a.higher_is_better:
        a_wins = mean_diff > 0
    else:
        a_wins = mean_diff < 0

    if p_value < alpha:
        winner = results_a.model_name if a_wins else results_b.model_name
        conclusion = (
            f"Statistically significant difference (p={p_value:.4f} < α={alpha}). "
            f"{winner} is better."
        )
    else:
        conclusion = (
            f"No statistically significant difference (p={p_value:.4f} ≥ α={alpha}). "
            f"Cannot conclude either model is better."
        )

    return ComparisonResult(
        model_a=results_a.model_name,
        model_b=results_b.model_name,
        method="Wilcoxon Signed-Rank Test (paired by seed)",
        statistic=float(stat),
        p_value=float(p_value),
        effect_size=mean_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        prob_a_better=prob_a_better,
        conclusion=conclusion,
        alpha=alpha,
    )


# -----------------------------------------------------------------------
# Bayesian comparison
# -----------------------------------------------------------------------

def bayesian_comparison(
    results_a: ExperimentResults,
    results_b: ExperimentResults,
    n_samples: int = 50_000,
    seed: int = 42,
) -> ComparisonResult:
    """
    Bayesian comparison of two models using a normal model for the
    per-seed score differences.

    Rather than a binary reject/don't-reject, this returns:
      - P(A > B): posterior probability that model A has higher true performance
      - A credible interval on the mean difference

    No PyMC required. Uses analytical Normal-Normal conjugate update.
    For a bounded metric like accuracy, this is an approximation — but a
    very good one when scores are not too close to 0 or 1.

    Parameters
    ----------
    results_a, results_b : ExperimentResults
    n_samples : int
        Posterior samples to draw.
    seed : int

    Returns
    -------
    ComparisonResult
    """

    _check_compatible(results_a, results_b)

    diffs = results_a.seed_means - results_b.seed_means
    n = len(diffs)
    sample_mean = float(diffs.mean())
    sample_std = float(diffs.std(ddof=1)) if n > 1 else 1e-6

    # Weakly informative prior: N(0, 1)
    # Likelihood: N(mu, sigma^2/n)
    # Posterior: conjugate Normal update
    prior_mean, prior_var = 0.0, 1.0
    likelihood_var = (sample_std ** 2) / n

    post_var = 1.0 / (1.0 / prior_var + 1.0 / likelihood_var)
    post_mean = post_var * (prior_mean / prior_var + sample_mean / likelihood_var)
    post_std = float(np.sqrt(post_var))

    rng = np.random.default_rng(seed)
    posterior_samples = rng.normal(post_mean, post_std, size=n_samples)

    if results_a.higher_is_better:
        prob_a_better = float(np.mean(posterior_samples > 0))
    else:
        prob_a_better = float(np.mean(posterior_samples < 0))

    ci_lower = float(np.percentile(posterior_samples, 2.5))
    ci_upper = float(np.percentile(posterior_samples, 97.5))

    if prob_a_better > 0.95:
        conclusion = (
            f"Strong evidence that {results_a.model_name} is better "
            f"(P(A>B)={prob_a_better:.3f})."
        )
    elif prob_a_better < 0.05:
        conclusion = (
            f"Strong evidence that {results_b.model_name} is better "
            f"(P(A>B)={prob_a_better:.3f})."
        )
    else:
        conclusion = (
            f"Insufficient evidence to prefer either model "
            f"(P(A>B)={prob_a_better:.3f})."
        )

    return ComparisonResult(
        model_a=results_a.model_name,
        model_b=results_b.model_name,
        method="Bayesian Normal-Normal conjugate comparison",
        statistic=None,
        p_value=None,
        effect_size=float(post_mean),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        prob_a_better=prob_a_better,
        conclusion=conclusion,
        alpha=0.05,
    )


# -----------------------------------------------------------------------
# Main dispatch function
# -----------------------------------------------------------------------

def compare(
    results_a: ExperimentResults,
    results_b: ExperimentResults,
    method: str = "corrected_ttest",
    alpha: float = 0.05,
    **kwargs,
) -> ComparisonResult:
    """
    Compare two models statistically.

    Parameters
    ----------
    results_a, results_b : ExperimentResults
    method : str
        One of:
          - "corrected_ttest"  (default) — Nadeau & Bengio corrected t-test
          - "wilcoxon"         — non-parametric paired Wilcoxon test
          - "bayesian"         — Bayesian posterior comparison
          - "all"              — run all three and print all results
    alpha : float
        Significance threshold (ignored for bayesian).

    Returns
    -------
    ComparisonResult (or prints all three if method="all")
    """

    methods = {
        "corrected_ttest": corrected_resampled_ttest,
        "wilcoxon":        wilcoxon_test,
        "bayesian":        bayesian_comparison,
    }

    if method == "all":
        results = {}
        for name, fn in methods.items():
            r = fn(results_a, results_b, **kwargs) if name != "bayesian" else fn(results_a, results_b)
            print(r)
            results[name] = r
        return results

    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(methods)} or 'all'.")

    fn = methods[method]
    if method == "bayesian":
        return fn(results_a, results_b, **kwargs)
    return fn(results_a, results_b, alpha=alpha, **kwargs)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _check_compatible(
    results_a: ExperimentResults,
    results_b: ExperimentResults,
) -> None:
    if results_a.scores.shape != results_b.scores.shape:
        raise ValueError(
            f"Shapes must match for paired comparison. "
            f"Got {results_a.scores.shape} vs {results_b.scores.shape}."
        )


def _bootstrap_effect_ci(
    a_means: np.ndarray,
    b_means: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap CI on mean(A) - mean(B)."""
    n = len(a_means)
    rng = np.random.default_rng(seed)
    boot_diffs = [
        (rng.choice(a_means, n, replace=True) - rng.choice(b_means, n, replace=True)).mean()
        for _ in range(n_bootstrap)
    ]
    alpha = 1.0 - confidence
    return (
        float(np.percentile(boot_diffs, 100 * alpha / 2)),
        float(np.percentile(boot_diffs, 100 * (1 - alpha / 2))),
    )


def _prob_a_better(
    results_a: ExperimentResults,
    results_b: ExperimentResults,
) -> float:
    """Empirical P(A > B) across all (seed, split) pairs."""
    a, b = results_a.flat, results_b.flat
    if results_a.higher_is_better:
        return float(np.mean(a > b))
    return float(np.mean(a < b))
