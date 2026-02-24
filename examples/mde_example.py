"""
examples/mde_example.py
========================

Demonstrates the Minimum Detectable Effect (MDE) calculator in mlci.

  "How many seeds do I need to detect a 0.5% accuracy difference with 80% power?"

Built on the Nadeau-Bengio corrected t-test (the same test mlci uses for
comparison by default), so power estimates are directly interpretable.

Runtime: ~30 seconds
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
from sklearn.datasets import load_breast_cancer

# ── mlci imports ──────────────────────────────────────────────────────────────
from mlci import Experiment
from mlci.stats.power import (
    mde_n_seeds,
    mde_effect,
    power_analysis,
    estimate_sigma,
    quick_summary,
    plot_power_curve,
    plot_power_heatmap,
)


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

PLOT_DIR = os.path.join(os.path.dirname(__file__), "mde_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

X, y = load_breast_cancer(return_X_y=True)

print("=" * 65)
print("  mlci — Minimum Detectable Effect (MDE) Calculator Demo")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 1a. Use a quick pilot run to estimate sigma
# ─────────────────────────────────────────────────────────────────────────────

print("\nRunning a small pilot experiment to estimate score variance (σ)...")
pilot = Experiment(
    model_factory=lambda seed: RandomForestClassifier(random_state=seed),
    X=X, y=y, metric="accuracy",
    n_seeds=5, n_splits=5,
    model_name="RF-Pilot",
).run(verbose=False)

sigma = estimate_sigma(pilot)
print(f"  Estimated σ from pilot: {sigma:.5f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1b. How many seeds do I need to detect a 0.5% difference?
# ─────────────────────────────────────────────────────────────────────────────

print("Question: 'How many seeds to detect a 0.5% accuracy difference with 80% power?'")
result = mde_n_seeds(
    effect_size=0.005,   # 0.5%
    sigma=sigma,
    n_splits=5,
    alpha=0.05,
    target_power=0.80,
)
print(result)


# ─────────────────────────────────────────────────────────────────────────────
# 1c. What's the MDE given my actual budget of 20 seeds?
# ─────────────────────────────────────────────────────────────────────────────

print("\nQuestion: 'With 20 seeds, what is my minimum detectable effect?'")
mde = mde_effect(
    n_seeds=20,
    sigma=sigma,
    n_splits=5,
    alpha=0.05,
    target_power=0.80,
)
print(mde)


# ─────────────────────────────────────────────────────────────────────────────
# 1d. Quick power reference table
# ─────────────────────────────────────────────────────────────────────────────

print("\nPower reference table for different effect sizes and seed counts:")
quick_summary(
    sigma=sigma,
    n_splits=5,
    alpha=0.05,
    effects=[0.001, 0.002, 0.005, 0.010, 0.020],
    n_seeds_options=[5, 10, 20, 30, 50],
)


# ─────────────────────────────────────────────────────────────────────────────
# 1e. Power curves and heatmap
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# 1f. Verify: run 20 seeds and compare — does the test behave as expected?
# ─────────────────────────────────────────────────────────────────────────────

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
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n── Summary ──────────────────────────────────────────────────────\n")
print("MDE Calculator:")
print(f"  • With σ={sigma:.4f} and 5 folds, you need {result.n_seeds} seeds to")
print(f"    detect a 0.5% effect at 80% power.")
print(f"  • With 20 seeds, your MDE is {mde.min_detectable_effect * 100:.3f}%.")
print()
print(f"All plots saved to: {PLOT_DIR}/")