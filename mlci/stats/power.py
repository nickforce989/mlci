"""
Power analysis and minimum detectable effect (MDE) calculator for mlci.

Answers the key experiment-design question:
  "How many seeds do I need to detect a 0.5% accuracy difference with 80% power?"

Built on the Nadeau-Bengio corrected t-test variance structure, so the power
estimates are consistent with the comparison test mlci uses by default.

Key functions
-------------
mde_n_seeds          : given an effect size, how many seeds are needed?
mde_effect           : given n_seeds, what is the smallest detectable effect?
power_analysis       : full power curve across (effect_size, n_seeds) grid
estimate_sigma       : estimate per-fold score std from ExperimentResults
plot_power_curve     : visualise power curves

Reference
---------
Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error.
Machine Learning, 52(3), 239–281.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
from scipy import stats

# -------------------------------------------------------------------------
# Result containers
# -------------------------------------------------------------------------

@dataclass
class SampleSizeResult:
    """
    Output of mde_n_seeds().

    Attributes
    ----------
    n_seeds : int
        Minimum number of seeds required to achieve the target power.
    effect_size : float
        The effect size (difference in metric) you want to detect.
    power : float
        Achieved statistical power (≥ target_power).
    alpha : float
        Significance level used.
    n_splits : int
        Number of CV folds assumed in the analysis.
    sigma : float
        Assumed per-fold score standard deviation.
    achieved_power : float
        Exact achieved power at the recommended n_seeds.
    """
    n_seeds: int
    effect_size: float
    power: float
    alpha: float
    n_splits: int
    sigma: float
    achieved_power: float

    def __repr__(self) -> str:
        lines = [
            f"{'─' * 60}",
            f"  Minimum Detectable Effect Analysis",
            f"{'─' * 60}",
            f"  Target effect  : {self.effect_size:+.4f}  ({self.effect_size * 100:.2f}% of metric)",
            f"  Target power   : {self.power:.0%}",
            f"  Significance   : α = {self.alpha}",
            f"  CV folds       : {self.n_splits}",
            f"  Assumed σ      : {self.sigma:.5f}",
            f"{'─' * 60}",
            f"  ➜  Minimum seeds needed : {self.n_seeds}",
            f"     Achieved power       : {self.achieved_power:.1%}",
            f"{'─' * 60}",
        ]
        return "\n".join(lines)


@dataclass
class MDEResult:
    """
    Output of mde_effect().

    Attributes
    ----------
    min_detectable_effect : float
        Smallest effect detectable at the requested power level.
    n_seeds : int
        Number of seeds assumed.
    power : float
        Target power.
    alpha : float
        Significance level.
    n_splits : int
        Number of CV folds.
    sigma : float
        Assumed per-fold score standard deviation.
    """
    min_detectable_effect: float
    n_seeds: int
    power: float
    alpha: float
    n_splits: int
    sigma: float

    def __repr__(self) -> str:
        lines = [
            f"{'─' * 60}",
            f"  Minimum Detectable Effect",
            f"{'─' * 60}",
            f"  Seeds          : {self.n_seeds}",
            f"  Target power   : {self.power:.0%}",
            f"  Significance   : α = {self.alpha}",
            f"  CV folds       : {self.n_splits}",
            f"  Assumed σ      : {self.sigma:.5f}",
            f"{'─' * 60}",
            f"  ➜  Min detectable effect : {self.min_detectable_effect:+.4f}",
            f"     ({self.min_detectable_effect * 100:.3f}% of metric)",
            f"{'─' * 60}",
        ]
        return "\n".join(lines)


@dataclass
class PowerCurveResult:
    """
    Output of power_analysis().

    Contains a 2-D grid of power values over (effect_sizes, n_seeds_range).

    Attributes
    ----------
    effects : np.ndarray, shape (n_effects,)
    n_seeds_range : np.ndarray, shape (n_seeds,)
    power_grid : np.ndarray, shape (n_effects, n_seeds)
        power_grid[i, j] = achieved power for effects[i] with n_seeds_range[j] seeds.
    alpha : float
    n_splits : int
    sigma : float
    """
    effects: np.ndarray
    n_seeds_range: np.ndarray
    power_grid: np.ndarray
    alpha: float
    n_splits: int
    sigma: float


# -------------------------------------------------------------------------
# Core power computation (Nadeau-Bengio corrected)
# -------------------------------------------------------------------------

def _nb_correction(n_splits: int, dataset_size: Optional[int] = None) -> float:
    """
    Nadeau-Bengio variance correction factor.

    correction = 1/k + n_test / n_train

    If dataset_size is None, assumes n_test = 1/(k) and n_train = (k-1)/k,
    i.e. the balanced k-fold case with a dataset of size k (relative).
    This gives correction = 1/k + 1/(k-1) which is a conservative upper bound.

    If dataset_size is provided, uses exact fold sizes.
    """
    k = n_splits
    if dataset_size is not None:
        n_test = dataset_size / k
        n_train = dataset_size - n_test
    else:
        # Balanced folds, dataset_size = k (relative)
        n_test = 1.0
        n_train = k - 1.0
    return 1.0 / k + n_test / n_train


def _compute_power(
    effect: float,
    n_seeds: int,
    sigma: float,
    n_splits: int,
    alpha: float,
    dataset_size: Optional[int],
) -> float:
    """
    Compute statistical power for a two-sided corrected t-test.

    Returns achieved power (float in [0, 1]).
    """
    correction = _nb_correction(n_splits, dataset_size)
    se = np.sqrt(correction * sigma**2 / n_seeds)

    if se == 0:
        return 1.0 if effect != 0 else 0.0

    # Degrees of freedom for Nadeau-Bengio = n_splits - 1
    df = n_splits - 1
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=df)

    # Non-centrality parameter
    ncp = abs(effect) / se

    # For very large ncp (clearly powered), short-circuit to avoid scipy overflow
    if ncp > 37:
        return 1.0

    # Power = P(reject H0 | true effect)
    # = P(|t| > t_crit | ncp)
    # Using the non-central t CDF:
    p_upper = stats.nct.sf(t_crit, df=df, nc=ncp)
    p_lower = stats.nct.cdf(-t_crit, df=df, nc=ncp)

    # Guard against NaN from scipy overflow
    power = (0.0 if np.isnan(p_upper) else p_upper) + (0.0 if np.isnan(p_lower) else p_lower)
    return float(np.clip(power, 0.0, 1.0))


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

def mde_n_seeds(
    effect_size: float,
    sigma: float,
    n_splits: int = 5,
    alpha: float = 0.05,
    target_power: float = 0.80,
    dataset_size: Optional[int] = None,
    max_seeds: int = 500,
) -> SampleSizeResult:
    """
    Compute the minimum number of seeds needed to detect a given effect.

    Uses the Nadeau-Bengio corrected t-test, which is the comparison method
    mlci uses by default — so power estimates are directly interpretable.

    Parameters
    ----------
    effect_size : float
        The metric difference you want to reliably detect (e.g. 0.005 for 0.5%).
    sigma : float
        Estimated standard deviation of per-fold scores across seeds.
        Use estimate_sigma() from an existing ExperimentResults, or provide
        a rough estimate (typical values: 0.003–0.010 for accuracy on tabular).
    n_splits : int
        Number of CV folds you plan to use. Default 5.
    alpha : float
        Significance level (Type I error rate). Default 0.05.
    target_power : float
        Desired power (1 - Type II error rate). Default 0.80.
    dataset_size : int or None
        Total dataset size for exact fold-size calculation.
        None uses a conservative balanced approximation.
    max_seeds : int
        Upper bound on seeds to search. If power is still below target at
        max_seeds, a warning is issued and max_seeds is returned.

    Returns
    -------
    SampleSizeResult

    Examples
    --------
    >>> result = mde_n_seeds(effect_size=0.005, sigma=0.005, n_splits=5)
    >>> print(result)

    >>> # Using an estimate from existing experiment data
    >>> sigma_est = estimate_sigma(rf_results)
    >>> result = mde_n_seeds(0.005, sigma_est, n_splits=5, target_power=0.90)
    """
    effect_size = abs(effect_size)
    if effect_size == 0:
        raise ValueError("effect_size must be non-zero.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")
    if not (0 < target_power < 1):
        raise ValueError("target_power must be in (0, 1).")

    for n in range(2, max_seeds + 1):
        p = _compute_power(effect_size, n, sigma, n_splits, alpha, dataset_size)
        if p >= target_power:
            return SampleSizeResult(
                n_seeds=n,
                effect_size=effect_size,
                power=target_power,
                alpha=alpha,
                n_splits=n_splits,
                sigma=sigma,
                achieved_power=p,
            )

    warnings.warn(
        f"Power did not reach {target_power:.0%} within {max_seeds} seeds. "
        f"Consider increasing effect_size or sigma, or using max_seeds={max_seeds * 2}.",
        stacklevel=2,
    )
    achieved = _compute_power(effect_size, max_seeds, sigma, n_splits, alpha, dataset_size)
    return SampleSizeResult(
        n_seeds=max_seeds,
        effect_size=effect_size,
        power=target_power,
        alpha=alpha,
        n_splits=n_splits,
        sigma=sigma,
        achieved_power=achieved,
    )


def mde_effect(
    n_seeds: int,
    sigma: float,
    n_splits: int = 5,
    alpha: float = 0.05,
    target_power: float = 0.80,
    dataset_size: Optional[int] = None,
    precision: float = 1e-6,
) -> MDEResult:
    """
    Compute the minimum detectable effect given a fixed number of seeds.

    This is the inverse of mde_n_seeds: given your experimental budget
    (n_seeds), what is the smallest true difference you can reliably detect?

    Parameters
    ----------
    n_seeds : int
        Number of seeds you plan to run.
    sigma : float
        Estimated per-fold score standard deviation.
    n_splits : int
    alpha : float
    target_power : float
    dataset_size : int or None
    precision : float
        Binary search precision on the effect size. Default 1e-6.

    Returns
    -------
    MDEResult

    Examples
    --------
    >>> result = mde_effect(n_seeds=20, sigma=0.005)
    >>> print(result)
    # "With 20 seeds you can detect effects ≥ 0.003 (0.3%)"
    """
    if n_seeds < 2:
        raise ValueError("n_seeds must be at least 2.")

    # Binary search for the MDE
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        p = _compute_power(mid, n_seeds, sigma, n_splits, alpha, dataset_size)
        if p >= target_power:
            hi = mid
        else:
            lo = mid
        if (hi - lo) < precision:
            break

    return MDEResult(
        min_detectable_effect=hi,
        n_seeds=n_seeds,
        power=target_power,
        alpha=alpha,
        n_splits=n_splits,
        sigma=sigma,
    )


def power_analysis(
    effects: Optional[Sequence[float]] = None,
    n_seeds_range: Optional[Sequence[int]] = None,
    sigma: float = 0.005,
    n_splits: int = 5,
    alpha: float = 0.05,
    dataset_size: Optional[int] = None,
) -> PowerCurveResult:
    """
    Compute a full power grid over (effect_sizes, n_seeds_range).

    Useful for understanding the sensitivity of your design choices — e.g.
    "if I can only afford 15 seeds, which effect sizes am I powered for?"

    Parameters
    ----------
    effects : sequence of float or None
        Effect sizes to evaluate. Default: 0.1% to 2% in 10 steps.
    n_seeds_range : sequence of int or None
        Seed counts to evaluate. Default: 5 to 100 in 10 steps.
    sigma : float
        Estimated per-fold score standard deviation.
    n_splits : int
    alpha : float
    dataset_size : int or None

    Returns
    -------
    PowerCurveResult
        Contains .power_grid (n_effects × n_seeds) and can be passed to
        plot_power_curve() for visualisation.

    Examples
    --------
    >>> curve = power_analysis(sigma=0.005)
    >>> fig = plot_power_curve(curve)
    """
    if effects is None:
        effects = np.linspace(0.001, 0.02, 10)
    if n_seeds_range is None:
        n_seeds_range = list(range(5, 105, 10))

    effects = np.asarray(effects)
    n_seeds_range = np.asarray(n_seeds_range, dtype=int)

    grid = np.zeros((len(effects), len(n_seeds_range)))
    for i, eff in enumerate(effects):
        for j, ns in enumerate(n_seeds_range):
            grid[i, j] = _compute_power(eff, int(ns), sigma, n_splits, alpha, dataset_size)

    return PowerCurveResult(
        effects=effects,
        n_seeds_range=n_seeds_range,
        power_grid=grid,
        alpha=alpha,
        n_splits=n_splits,
        sigma=sigma,
    )


def estimate_sigma(results) -> float:
    """
    Estimate per-fold score standard deviation from an ExperimentResults object.

    This is the key input to the power analysis functions. Running a small
    pilot experiment (5–10 seeds) and calling estimate_sigma() gives you a
    data-driven sigma estimate for proper power analysis.

    The variance is estimated as the pooled standard deviation of per-split
    scores across seeds, which matches what the Nadeau-Bengio test uses.

    Parameters
    ----------
    results : ExperimentResults
        Results from a previous or pilot experiment.

    Returns
    -------
    float
        Estimated sigma (per-fold score std).

    Examples
    --------
    >>> pilot = Experiment(..., n_seeds=5, n_splits=5).run()
    >>> sigma = estimate_sigma(pilot)
    >>> result = mde_n_seeds(0.005, sigma)
    """
    # Per-split means across seeds, then std of those
    fold_means = results.scores.mean(axis=0)  # (n_splits,)
    if len(fold_means) > 1:
        return float(np.std(fold_means, ddof=1))
    # Fallback: overall std
    return float(np.std(results.flat, ddof=1))


def quick_summary(
    sigma: float,
    n_splits: int = 5,
    alpha: float = 0.05,
    effects: Sequence[float] = (0.001, 0.002, 0.005, 0.010, 0.020),
    n_seeds_options: Sequence[int] = (5, 10, 20, 30, 50),
) -> None:
    """
    Print a quick reference table: power for common (effect, n_seeds) pairs.

    Parameters
    ----------
    sigma : float
    n_splits : int
    alpha : float
    effects : sequence of float
    n_seeds_options : sequence of int

    Examples
    --------
    >>> quick_summary(sigma=0.005)
    """
    col_w = 10
    header = f"{'Effect':>10}" + "".join(f"  n={n:<6}" for n in n_seeds_options)
    print(f"\n{'─' * len(header)}")
    print(f"  Power table | σ={sigma:.4f} | k={n_splits} folds | α={alpha}")
    print(f"{'─' * len(header)}")
    print(header)
    print(f"{'─' * len(header)}")
    for eff in effects:
        row = f"  {eff * 100:>6.2f}%  "
        for ns in n_seeds_options:
            p = _compute_power(eff, ns, sigma, n_splits, alpha, None)
            star = "*" if p >= 0.80 else " "
            row += f"  {p:.0%}{star}     "
        print(row)
    print(f"{'─' * len(header)}")
    print("  * = powered at ≥80%\n")


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def plot_power_curve(
    curve: PowerCurveResult,
    target_power: float = 0.80,
    highlight_effects: Optional[Sequence[float]] = None,
    ax=None,
) -> "plt.Figure":
    """
    Plot power curves: power vs n_seeds for each effect size.

    Parameters
    ----------
    curve : PowerCurveResult
        From power_analysis().
    target_power : float
        Horizontal reference line. Default 0.80.
    highlight_effects : list of float or None
        Subset of effect sizes to label prominently.
    ax : matplotlib Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, ax_ = (plt.subplots(figsize=(9, 5)) if ax is None else (ax.figure, ax))

    n_effects = len(curve.effects)
    colours = cm.viridis(np.linspace(0.1, 0.9, n_effects))

    for i, eff in enumerate(curve.effects):
        label = f"{eff * 100:.2f}%"
        lw = 2.5 if (highlight_effects and eff in highlight_effects) else 1.2
        ax_.plot(
            curve.n_seeds_range,
            curve.power_grid[i],
            color=colours[i],
            linewidth=lw,
            label=label,
        )

    ax_.axhline(target_power, color="#E84855", linewidth=1.5, linestyle="--",
                label=f"{target_power:.0%} power")
    ax_.axhline(0.90, color="#F4A261", linewidth=1.0, linestyle=":",
                alpha=0.7, label="90% power")

    ax_.set_xlabel("Number of seeds")
    ax_.set_ylabel("Statistical power")
    ax_.set_title(
        f"Power curves (σ={curve.sigma:.4f}, k={curve.n_splits} folds, α={curve.alpha})"
    )
    ax_.set_ylim(0, 1.05)
    ax_.set_xlim(curve.n_seeds_range[0], curve.n_seeds_range[-1])
    ax_.legend(
        title="Effect size", loc="lower right",
        frameon=True, fontsize=9, ncol=2,
    )
    ax_.grid(True, alpha=0.3)
    ax_.spines["top"].set_visible(False)
    ax_.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def plot_power_heatmap(
    curve: PowerCurveResult,
    target_power: float = 0.80,
) -> "plt.Figure":
    """
    Heatmap of power over (effect_size, n_seeds) grid.

    Cells at or above target_power are highlighted.

    Parameters
    ----------
    curve : PowerCurveResult
    target_power : float

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(10, 5))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(
        curve.power_grid,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        origin="upper",
    )

    # Annotate cells
    for i in range(len(curve.effects)):
        for j in range(len(curve.n_seeds_range)):
            val = curve.power_grid[i, j]
            colour = "black" if 0.35 < val < 0.75 else "white"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=8, color=colour)

    ax.set_xticks(range(len(curve.n_seeds_range)))
    ax.set_xticklabels([str(n) for n in curve.n_seeds_range], fontsize=9)
    ax.set_yticks(range(len(curve.effects)))
    ax.set_yticklabels([f"{e * 100:.2f}%" for e in curve.effects], fontsize=9)
    ax.set_xlabel("Number of seeds")
    ax.set_ylabel("Effect size")
    ax.set_title(
        f"Power heatmap  |  σ={curve.sigma:.4f} | k={curve.n_splits} | α={curve.alpha}\n"
        f"Green ≥ {target_power:.0%} power"
    )

    fig.colorbar(im, ax=ax, label="Power", shrink=0.8)
    fig.tight_layout()
    return fig