"""
examples/calibration_example.py
=================================

Demonstrates calibration analysis across seeds in mlci.

  "Is my model's confidence actually meaningful — and how stable is
   calibration across different random seeds?"

Computes ECE, MCE, and full reliability diagrams with uncertainty bands
across seeds, which sklearn's calibration_curve does not provide.

Runtime: ~30 seconds
"""

import os
import sys

# Allow running from the repo root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# ── mlci imports ──────────────────────────────────────────────────────────────
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

PLOT_DIR = os.path.join(os.path.dirname(__file__), "calibration_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

X, y = load_breast_cancer(return_X_y=True)

print("=" * 65)
print("  mlci — Calibration Analysis Across Seeds Demo")
print("=" * 65)

print("\nRunning calibration experiments for 3 models (this is the interesting part)...")
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


# ─────────────────────────────────────────────────────────────────────────────
# Run calibration experiments
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Per-model reliability diagrams with uncertainty bands
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# ECE distribution plots
# ─────────────────────────────────────────────────────────────────────────────

_ECE_COLORS = ["#2E4057", "#E84855", "#3BB273"]
print("\nGenerating ECE distribution plots...")
for idx, (name, res) in enumerate(cal_results.items()):
    fig = plot_ece_distribution(res, color=_ECE_COLORS[idx % 3])
    path = os.path.join(PLOT_DIR, f"ece_dist_{name.lower()}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Multi-model comparison
# ─────────────────────────────────────────────────────────────────────────────

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
print("Calibration Analysis:")
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