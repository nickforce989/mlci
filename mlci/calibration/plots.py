"""
Calibration visualisation for mlci.

Functions
---------
plot_reliability_diagram  : reliability diagram with uncertainty bands across seeds
plot_ece_distribution     : distribution (violin/box) of ECE across seeds and splits
plot_calibration_comparison : forest plot comparing ECE of multiple models
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mlci.calibration.ece import CalibrationResults

_PALETTE = ["#2E4057", "#E84855", "#3BB273", "#F4A261", "#8338EC", "#023E8A"]


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 120,
    })


# -------------------------------------------------------------------------
# 1. Reliability diagram with uncertainty
# -------------------------------------------------------------------------

def plot_reliability_diagram(
    results: CalibrationResults,
    confidence: float = 0.95,
    show_histogram: bool = True,
    show_individual_runs: bool = False,
    color: str = _PALETTE[0],
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Reliability (calibration) diagram with bootstrap uncertainty bands.

    Shows:
    - Mean confidence vs mean accuracy per bin (the reliability curve)
    - Bootstrap CI band across seeds on the accuracy per bin
    - Optionally, all individual run curves (faint)
    - Confidence histogram in a secondary panel
    - ECE annotation

    The uncertainty bands are the core contribution over sklearn's
    calibration_curve — they show how stable calibration is across seeds,
    not just where it sits on average.

    Parameters
    ----------
    results : CalibrationResults
    confidence : float
        CI level for uncertainty bands. Default 0.95.
    show_histogram : bool
        Show confidence distribution histogram. Default True.
    show_individual_runs : bool
        Overlay all individual (seed, split) reliability curves. Default False.
    color : str
        Primary colour for the mean curve and CI band.
    ax : matplotlib Axes or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    _style()

    if show_histogram:
        fig, (ax_main, ax_hist) = plt.subplots(
            2, 1, figsize=(7, 8),
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig, ax_main = plt.subplots(figsize=(7, 6))
        ax_hist = None

    # -- Perfect calibration diagonal
    ax_main.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.5, label="Perfect calibration")

    # -- Individual runs (faint)
    if show_individual_runs:
        for i in range(results.bin_confidences.shape[0]):
            bc = results.bin_confidences[i]
            ba = results.bin_accuracies[i]
            valid = ~(np.isnan(bc) | np.isnan(ba))
            if valid.sum() < 2:
                continue
            ax_main.plot(bc[valid], ba[valid], color=color, alpha=0.08, linewidth=0.8)

    # -- Mean curve with CI band
    mean_conf, mean_acc, acc_lo, acc_hi = results.reliability_curve_ci(confidence)
    valid = ~(np.isnan(mean_conf) | np.isnan(mean_acc))

    if valid.sum() >= 2:
        ax_main.plot(
            mean_conf[valid], mean_acc[valid],
            color=color, linewidth=2.5, zorder=5, label=f"Mean reliability",
        )
        ax_main.fill_between(
            mean_conf[valid], acc_lo[valid], acc_hi[valid],
            alpha=0.20, color=color,
            label=f"{int(confidence * 100)}% CI across seeds",
        )
        # Error bars at each bin
        ax_main.errorbar(
            mean_conf[valid], mean_acc[valid],
            yerr=[
                np.clip(mean_acc[valid] - acc_lo[valid], 0, 1),
                np.clip(acc_hi[valid] - mean_acc[valid], 0, 1),
            ],
            fmt="o", color=color, markersize=5, capsize=3, linewidth=1, zorder=6,
        )

    # -- ECE annotation
    ci_lo, ci_hi = results.ece_ci
    ece_text = (
        f"ECE = {results.mean_ece:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]\n"
        f"MCE = {results.mean_mce:.4f}\n"
        f"n={results.n_seeds} seeds × {results.n_splits} splits"
    )
    ax_main.text(
        0.04, 0.95, ece_text,
        transform=ax_main.transAxes,
        va="top", ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="lightgray"),
    )

    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.set_xlabel("Mean predicted confidence")
    ax_main.set_ylabel("Fraction of positives (accuracy)")
    ax_main.set_title(f"Reliability Diagram: {results.model_name}")
    ax_main.legend(frameon=False, fontsize=9, loc="lower right")
    ax_main.set_aspect("equal")

    # -- Confidence histogram
    if ax_hist is not None:
        mean_counts = results.bin_counts.mean(axis=0)
        bin_centers = 0.5 * (results.bin_edges[:-1] + results.bin_edges[1:])
        ax_hist.bar(
            bin_centers,
            mean_counts,
            width=results.bin_edges[1] - results.bin_edges[0],
            color=color,
            alpha=0.6,
            edgecolor="white",
            linewidth=0.5,
        )
        ax_hist.set_xlabel("Predicted confidence")
        ax_hist.set_ylabel("Mean sample count")
        ax_hist.set_xlim(0, 1)
        ax_hist.set_title("Confidence distribution (avg across runs)")
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


# -------------------------------------------------------------------------
# 2. ECE distribution plot
# -------------------------------------------------------------------------

def plot_ece_distribution(
    results: CalibrationResults,
    ax: Optional[plt.Axes] = None,
    color: str = _PALETTE[0],
) -> plt.Figure:
    """
    Distribution of ECE across seeds and splits.

    Shows:
    - Violin plot of ECE values per seed (averaged across splits)
    - Scatter of all individual (seed, split) ECE values
    - Bootstrap CI on the mean ECE

    Parameters
    ----------
    results : CalibrationResults
    ax : matplotlib Axes or None
    color : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    _style()
    fig, ax = (plt.subplots(figsize=(7, 5)) if ax is None else (ax.figure, ax))

    # Per-seed ECE means (averaged across splits)
    seed_eces = results.ece_scores.mean(axis=1)  # (n_seeds,)
    all_eces = results.ece_scores.ravel()

    # Violin of seed means
    parts = ax.violinplot([seed_eces], positions=[0], showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.4)
    parts["cmedians"].set_color(color)
    parts["cmins"].set_color(color)
    parts["cmaxes"].set_color(color)
    parts["cbars"].set_color(color)

    # Individual seed points
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(seed_eces))
    ax.scatter(jitter, seed_eces, color=color, alpha=0.8, s=40, zorder=5,
               label="Seed mean ECE")

    # CI annotation
    ci_lo, ci_hi = results.ece_ci
    ax.annotate(
        f"Mean ECE = {results.mean_ece:.4f}\n95% CI [{ci_lo:.4f}, {ci_hi:.4f}]",
        xy=(0, results.mean_ece),
        xytext=(0.35, results.mean_ece),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="gray"),
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xticks([0])
    ax.set_xticklabels([results.model_name])
    ax.set_ylabel("ECE")
    ax.set_title(f"ECE Distribution: {results.model_name}\n({results.n_seeds} seeds × {results.n_splits} splits)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(-0.5, 0.5)

    fig.tight_layout()
    return fig


# -------------------------------------------------------------------------
# 3. Multi-model calibration comparison
# -------------------------------------------------------------------------

def plot_calibration_comparison(
    results_list: list[CalibrationResults],
    confidence: float = 0.95,
    title: str = "Calibration Comparison",
) -> plt.Figure:
    """
    Forest plot comparing ECE across multiple models.

    Similar to plot_comparison() but for calibration metrics rather than
    accuracy. Shows mean ECE ± CI for each model, sorted by ECE (lower = better).

    Parameters
    ----------
    results_list : list of CalibrationResults
    confidence : float
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    _style()

    n = len(results_list)
    fig, ax = plt.subplots(figsize=(9, max(3, n * 0.7 + 1)))

    # Sort by mean ECE ascending (lower is better)
    sorted_results = sorted(results_list, key=lambda r: r.mean_ece)

    for i, res in enumerate(sorted_results):
        ci_lo, ci_hi = res.ece_ci
        color = _PALETTE[i % len(_PALETTE)]
        ax.barh(
            i, res.mean_ece,
            xerr=[[res.mean_ece - ci_lo], [ci_hi - res.mean_ece]],
            color=color, alpha=0.75, height=0.5,
            error_kw=dict(elinewidth=2, capsize=5, ecolor="black"),
        )
        ax.text(
            ci_hi + 0.001, i,
            f"ECE={res.mean_ece:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]",
            va="center", ha="left", fontsize=9,
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels([r.model_name for r in sorted_results])
    ax.set_xlabel("Expected Calibration Error (lower is better)")
    ax.set_title(f"{title}  |  {confidence:.0%} CI across seeds")
    ax.set_xlim(0, None)
    ax.invert_yaxis()

    fig.tight_layout()
    return fig


# -------------------------------------------------------------------------
# 4. Multi-model reliability overlay
# -------------------------------------------------------------------------

def plot_reliability_overlay(
    results_list: list[CalibrationResults],
    confidence: float = 0.95,
) -> plt.Figure:
    """
    Overlay reliability diagrams for multiple models on a single plot.

    Each model gets its mean reliability curve plus a CI band, allowing
    direct visual comparison of calibration quality across models.

    Parameters
    ----------
    results_list : list of CalibrationResults
    confidence : float

    Returns
    -------
    matplotlib.figure.Figure
    """
    _style()
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.5, label="Perfect calibration")

    for i, res in enumerate(results_list):
        color = _PALETTE[i % len(_PALETTE)]
        mean_conf, mean_acc, acc_lo, acc_hi = res.reliability_curve_ci(confidence)
        valid = ~(np.isnan(mean_conf) | np.isnan(mean_acc))

        if valid.sum() < 2:
            continue

        ax.plot(
            mean_conf[valid], mean_acc[valid],
            color=color, linewidth=2, label=f"{res.model_name} (ECE={res.mean_ece:.4f})",
        )
        ax.fill_between(
            mean_conf[valid], acc_lo[valid], acc_hi[valid],
            alpha=0.12, color=color,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Reliability Diagram Comparison  |  {confidence:.0%} CI across seeds")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_aspect("equal")

    fig.tight_layout()
    return fig