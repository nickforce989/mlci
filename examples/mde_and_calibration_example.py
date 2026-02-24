"""
examples/mde_and_calibration_example.py
========================================

Demonstrates two new mlci features:

  1. Minimum Detectable Effect (MDE) calculator
     ─────────────────────────────────────────────
     "How many seeds do I need to detect a 0.5% accuracy difference with 80% power?"

     Built on the Nadeau-Bengio corrected t-test (the same test mlci uses for
     comparison by default), so power estimates are directly interpretable.

  2. Calibration analysis across seeds
     ────────────────────────────────────
     "Is my model's confidence actually meaningful — and how stable is
      calibration across different random seeds?"

     Computes ECE, MCE, and full reliability diagrams with uncertainty bands
     across seeds, which sklearn's calibration_curve does not provide.

Runtime: ~1 minute
"""

import os
import sys

# Allow running from the repo root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# ── mlci imports ──────────────────────────────────────────────────────────────
from mlci import Experiment
from mlci.stats.bootstrap import summary
from mlci.stats.power import (
    mde_n_seeds,
    mde_effect,
    power_analysis,
    estimate_sigma,
    quick_summary,
    plot_power_curve,
    plot_power_heatmap,
)
from mlci.calibration.experiment import (
    CalibrationExperiment
)

from mlci.calibration.plots import (
    plot_reliability_diagram,
    plot_ece_distribution,
    plot_calibration_comparison,
    plot_reliability_overlay,
)


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

PLOT_DIR = os.path.join(os.path.dirname(__file__), "mde_cal_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

X, y = load_breast_cancer(return_X_y=True)

print("=" * 65)
print("  mlci — MDE Calculator + Calibration Analysis Demo")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — Minimum Detectable Effect Calculator
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n── PART 1: Minimum Detectable Effect (MDE) Calculator ──────────\n")

# ── 1a. Use a quick pilot run to estimate sigma ──────────────────────────────
print("Running a small pilot experiment to estimate score variance (σ)...")
pilot = Experiment(
    model_factory=lambda seed: RandomForestClassifier(random_state=seed),
    X=X, y=y, metric="accuracy",
    n_seeds=5, n_splits=5,
    model_name="RF-Pilot",
).run(verbose=False)

sigma = estimate_sigma(pilot)
print(f"  Estimated σ from pilot: {sigma:.5f}\n")

# ── 1b. How many seeds do I need to detect a 0.5% difference? ───────────────
print("Question: 'How many seeds to detect a 0.5% accuracy difference with 80% power?'")
result = mde_n_seeds(
    effect_size=0.005,   # 0.5%
    sigma=sigma,
    n_splits=5,
    alpha=0.05,
    target_power=0.80,
)
print(result)

# ── 1c. What's the MDE given my actual budget of 20 seeds? ──────────────────
print("\nQuestion: 'With 20 seeds, what is my minimum detectable effect?'")
mde = mde_effect(
    n_seeds=20,
    sigma=sigma,
    n_splits=5,
    alpha=0.05,
    target_power=0.80,
)
print(mde)

# ── 1d. Quick power reference table ─────────────────────────────────────────
print("\nPower reference table for different effect sizes and seed counts:")
quick_summary(
    sigma=sigma,
    n_splits=5,
    alpha=0.05,
    effects=[0.001, 0.002, 0.005, 0.010, 0.020],
    n_seeds_options=[5, 10, 20, 30, 50],
)

# ── 1e. Power curves and heatmap ─────────────────────────────────────────────
print("Generating power visualisations...")

curve = power_analysis(
    effects=np.linspace(0.001, 0.015, 8),
    n_seeds_range=list(range(5, 60, 5)),
    sigma=sigma,
    n_splits=5,
    alpha=0.05,
)

fig_curve = plot_power_curve(
    curve,
    target_power=0.80,
    highlight_effects=[0.005, 0.010],
)
path_curve = os.path.join(PLOT_DIR, "power_curves.png")
fig_curve.savefig(path_curve, dpi=150, bbox_inches="tight")
plt.close(fig_curve)
print(f"  Saved: {path_curve}")

fig_heat = plot_power_heatmap(curve, target_power=0.80)
path_heat = os.path.join(PLOT_DIR, "power_heatmap.png")
fig_heat.savefig(path_heat, dpi=150, bbox_inches="tight")
plt.close(fig_heat)
print(f"  Saved: {path_heat}")

# ── 1f. Verify: run 20 seeds and compare — does the test behave as expected? ─
print("\nVerification: running 20-seed experiment to check test behaviour...")
from mlci.stats.tests import compare

rf20 = Experiment(
    model_factory=lambda seed: RandomForestClassifier(random_state=seed),
    X=X, y=y, metric="accuracy",
    n_seeds=20, n_splits=5,
    model_name="RandomForest",
).run(verbose=False)

gb20 = Experiment(
    model_factory=lambda seed: GradientBoostingClassifier(random_state=seed),
    X=X, y=y, metric="accuracy",
    n_seeds=20, n_splits=5,
    model_name="GradientBoosting",
).run(verbose=False)

effect_seen = abs(rf20.mean - gb20.mean)
print(f"  Observed effect: {effect_seen * 100:.3f}%  (MDE was {mde.min_detectable_effect * 100:.3f}%)")
print(f"  {'Within MDE — expect non-significant result' if effect_seen < mde.min_detectable_effect else 'Above MDE — expect significant result'}")
result_cmp = compare(rf20, gb20, method="corrected_ttest")
print(result_cmp)


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Calibration Analysis
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n── PART 2: Calibration Analysis Across Seeds ────────────────────\n")
print("Running calibration experiments for 3 models (this is the interesting part)...")
print("sklearn's calibration_curve gives a single run. mlci gives you uncertainty.\n")

N_SEEDS = 15
N_SPLITS = 5

# ── Model factories ───────────────────────────────────────────────────────────
models = {
    "RandomForest": lambda seed: RandomForestClassifier(
        n_estimators=100, random_state=seed
    ),
    "GradientBoosting": lambda seed: GradientBoostingClassifier(
        n_estimators=100, random_state=seed
    ),
    "LogisticRegression": lambda seed: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
    ]),
}

cal_results = {}
for name, factory in models.items():
    print(f"  Calibration: {name}")
    cal = CalibrationExperiment(
        model_factory=factory,
        X=X, y=y,
        n_seeds=N_SEEDS,
        n_splits=N_SPLITS,
        n_bins=10,
        strategy="uniform",
        model_name=name,
    )
    cal_results[name] = cal.run(verbose=False)
    print(cal_results[name].summary())
    print()

# ── Per-model reliability diagrams with uncertainty bands ─────────────────────
print("Generating reliability diagrams...")
for name, res in cal_results.items():
    fig = plot_reliability_diagram(
        res,
        confidence=0.95,
        show_histogram=True,
        show_individual_runs=True,  # faint overlay of all runs
    )
    path = os.path.join(PLOT_DIR, f"reliability_{name.lower()}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# ── ECE distribution plot ──────────────────────────────────────────────────────
_ECE_COLORS = ["#2E4057", "#E84855", "#3BB273"]
print("\nGenerating ECE distribution plots...")
for idx, (name, res) in enumerate(cal_results.items()):
    fig = plot_ece_distribution(res, color=_ECE_COLORS[idx % 3])
    path = os.path.join(PLOT_DIR, f"ece_dist_{name.lower()}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# ── Multi-model comparison ────────────────────────────────────────────────────
print("\nGenerating multi-model calibration comparison...")

fig_cmp = plot_calibration_comparison(
    list(cal_results.values()),
    confidence=0.95,
    title="Model Calibration Comparison (Breast Cancer)",
)
path_cmp = os.path.join(PLOT_DIR, "calibration_comparison.png")
fig_cmp.savefig(path_cmp, dpi=150, bbox_inches="tight")
plt.close(fig_cmp)
print(f"  Saved: {path_cmp}")

fig_overlay = plot_reliability_overlay(
    list(cal_results.values()),
    confidence=0.95,
)
path_overlay = os.path.join(PLOT_DIR, "reliability_overlay.png")
fig_overlay.savefig(path_overlay, dpi=150, bbox_inches="tight")
plt.close(fig_overlay)
print(f"  Saved: {path_overlay}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n── Summary ──────────────────────────────────────────────────────\n")
print("PART 1 — MDE Calculator:")
print(f"  • With σ={sigma:.4f} and 5 folds, you need {result.n_seeds} seeds to")
print(f"    detect a 0.5% effect at 80% power.")
print(f"  • With 20 seeds, your MDE is {mde.min_detectable_effect * 100:.3f}%.")
print()
print("PART 2 — Calibration Analysis:")
for name, res in cal_results.items():
    ci_lo, ci_hi = res.ece_ci
    print(
        f"  • {name:<22} ECE = {res.mean_ece:.4f}  "
        f"[{ci_lo:.4f}, {ci_hi:.4f}]  MCE = {res.mean_mce:.4f}"
    )
print()
print(f"All plots saved to: {PLOT_DIR}/")
print()
print("Key insight: sklearn.calibration_curve() gives you a single reliability")
print("curve. mlci gives you the uncertainty bands on that curve across seeds,")
print("so you know whether your model's calibration is genuinely stable or just")
print("lucky on a particular random split.")