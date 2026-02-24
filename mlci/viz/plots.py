"""
Visualisation utilities for mlci.

All functions return matplotlib Figure objects so the caller can save,
display, or further customise them.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from mlci.core.results import ExperimentResults
from mlci.stats.bootstrap import bootstrap_ci
from mlci.stats.anova import decompose_variance


# -----------------------------------------------------------------------
# Style helpers
# -----------------------------------------------------------------------

_PALETTE = ["#2E4057", "#E84855", "#3BB273", "#F4A261", "#8338EC", "#023E8A"]


def _style():
    """Apply a clean, paper-ready matplotlib style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 120,
    })


# -----------------------------------------------------------------------
# 1. Score distribution plot
# -----------------------------------------------------------------------

def plot_score_distribution(
    results: ExperimentResults,
    ax: Optional[plt.Axes] = None,
    color: str = _PALETTE[0],
    show_ci: bool = True,
    confidence: float = 0.95,
) -> plt.Figure:
    """
    Plot the distribution of scores across seeds and splits.

    Shows:
    - KDE of all scores
    - Vertical line at the mean
    - Shaded confidence interval on the mean
    """
    _style()
    fig, ax = (plt.subplots(figsize=(7, 4)) if ax is None else (ax.figure, ax))

    scores_flat = results.flat
    sns.kdeplot(scores_flat, ax=ax, color=color, fill=True, alpha=0.3, linewidth=2)

    mean_val = results.mean
    ax.axvline(mean_val, color=color, linewidth=2, linestyle="--",
               label=f"Mean = {mean_val:.4f}")

    if show_ci:
        ci = bootstrap_ci(results, confidence=confidence)
        ax.axvspan(ci.lower, ci.upper, alpha=0.15, color=color,
                   label=f"{int(confidence*100)}% CI [{ci.lower:.4f}, {ci.upper:.4f}]")

    ax.set_xlabel(results.metric)
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution: {results.model_name}")
    ax.legend(frameon=False)

    # Rug plot
    ax.plot(scores_flat, np.full_like(scores_flat, ax.get_ylim()[0] * 1.05),
            "|", color=color, alpha=0.5, markersize=6)

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------
# 2. Model comparison plot
# -----------------------------------------------------------------------

def plot_comparison(
    results_list: list[ExperimentResults],
    confidence: float = 0.95,
    title: str = "Model Comparison",
) -> plt.Figure:
    """
    Forest plot comparing multiple models.

    Each model gets a point (mean) with a horizontal confidence interval bar.
    Models are sorted by mean score.
    """
    _style()

    results_list = sorted(
        results_list,
        key=lambda r: r.mean,
        reverse=results_list[0].higher_is_better,
    )

    n = len(results_list)
    fig, ax = plt.subplots(figsize=(8, 1.2 * n + 1.5))

    for i, res in enumerate(results_list):
        ci = bootstrap_ci(res, confidence=confidence)
        color = _PALETTE[i % len(_PALETTE)]

        ax.plot(ci.mean, i, "o", color=color, markersize=9, zorder=3)
        ax.hlines(i, ci.lower, ci.upper, color=color, linewidth=3, alpha=0.8)
        ax.vlines(ci.lower, i - 0.15, i + 0.15, color=color, linewidth=2)
        ax.vlines(ci.upper, i - 0.15, i + 0.15, color=color, linewidth=2)

        ax.text(
            ci.upper + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01,
            i,
            f"{ci.mean:.4f} [{ci.lower:.4f}, {ci.upper:.4f}]",
            va="center", fontsize=9, color=color,
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels([r.model_name for r in results_list])
    ax.set_xlabel(results_list[0].metric)
    ax.set_title(f"{title}\n(points = mean, bars = {int(confidence*100)}% CI)")
    ax.invert_yaxis()

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------
# 3. Variance decomposition plot
# -----------------------------------------------------------------------

def plot_variance_decomposition(
    results: ExperimentResults,
) -> plt.Figure:
    """
    Two-panel figure:
    Left:  pie chart of variance components
    Right: heatmap of scores[seed, split]
    """
    _style()

    decomp = decompose_variance(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Pie chart
    ax = axes[0]
    sizes = [decomp.seed_fraction, decomp.split_fraction, decomp.interaction_fraction]
    labels = ["Seed (init)", "Split (data)", "Interaction"]
    colors = [_PALETTE[0], _PALETTE[1], _PALETTE[3]]
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 1e-6]
    if non_zero:
        s, l, c = zip(*non_zero)
        ax.pie(s, labels=l, colors=c, autopct="%1.1f%%",
               startangle=90, pctdistance=0.75)
    ax.set_title(f"Variance Components\n{results.model_name}")

    # Heatmap
    ax = axes[1]
    im = ax.imshow(results.scores, aspect="auto", cmap="RdYlGn",
                   interpolation="nearest")
    ax.set_xlabel("Split index")
    ax.set_ylabel("Seed index")
    ax.set_title(f"Score Heatmap ({results.metric})")
    plt.colorbar(im, ax=ax, label=results.metric)

    # Annotate cells
    for i in range(results.n_seeds):
        for j in range(results.n_splits):
            ax.text(j, i, f"{results.scores[i,j]:.3f}",
                    ha="center", va="center", fontsize=7, color="black")

    fig.suptitle(
        f"σ²_seed={decomp.seed_fraction*100:.1f}%  "
        f"σ²_split={decomp.split_fraction*100:.1f}%",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------
# 4. Learning curve with uncertainty
# -----------------------------------------------------------------------

def plot_learning_curve(
    lc_results,  # LearningCurveResults
    confidence: float = 0.95,
    models_to_compare: Optional[list] = None,
    title: str = "Learning Curve",
) -> plt.Figure:
    """
    Plot a learning curve with confidence bands.

    Parameters
    ----------
    lc_results : LearningCurveResults or list of LearningCurveResults
        If a list, overlays multiple models.
    confidence : float
    title : str
    """
    _style()

    # Normalise to list
    if not isinstance(lc_results, list):
        lc_results = [lc_results]

    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, lc in enumerate(lc_results):
        color = _PALETTE[idx % len(_PALETTE)]
        means, lowers, uppers = [], [], []

        for res in lc.results_per_size:
            ci = bootstrap_ci(res, confidence=confidence)
            means.append(ci.mean)
            lowers.append(ci.lower)
            uppers.append(ci.upper)

        means = np.array(means)
        lowers = np.array(lowers)
        uppers = np.array(uppers)
        xs = lc.train_sizes

        model_name = lc.results_per_size[0].model_name.split(" (frac=")[0]
        ax.plot(xs, means, "o-", color=color, linewidth=2, label=model_name)
        ax.fill_between(xs, lowers, uppers, color=color, alpha=0.2)

    ax.set_xlabel("Training set size")
    ax.set_ylabel(lc_results[0].results_per_size[0].metric)
    ax.set_title(f"{title}\n(bands = {int(confidence*100)}% bootstrap CI)")
    ax.legend(frameon=False)

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------
# 5. Bootstrap distribution
# -----------------------------------------------------------------------

def plot_bootstrap_distribution(
    results: ExperimentResults,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
) -> plt.Figure:
    """
    Plot the bootstrap distribution of the mean score.

    Shows that the 'true' performance estimate is itself uncertain,
    not a single number.
    """
    _style()
    ci = bootstrap_ci(results, confidence=confidence, n_bootstrap=n_bootstrap)
    boot_dist = ci.bootstrap_distribution

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(boot_dist, ax=ax, bins=60, color=_PALETTE[0], alpha=0.7, kde=True)

    ax.axvline(ci.mean, color=_PALETTE[0], linewidth=2, linestyle="--",
               label=f"Mean = {ci.mean:.4f}")
    ax.axvspan(ci.lower, ci.upper, alpha=0.2, color=_PALETTE[1],
               label=f"{int(confidence*100)}% CI [{ci.lower:.4f}, {ci.upper:.4f}]")

    ax.set_xlabel(f"Bootstrap mean ({results.metric})")
    ax.set_ylabel("Count")
    ax.set_title(f"Bootstrap Distribution of Mean Score\n{results.model_name}")
    ax.legend(frameon=False)

    fig.tight_layout()
    return fig
