"""
mlci full demo — covers all three user types and all major features.

Produces:
  - Console output: summaries, comparisons, variance decompositions
  - Saved plots:    score_distributions.png
                    model_comparison.png
                    variance_decomposition.png
                    learning_curve.png
                    bootstrap_distribution.png

Run with:
    python examples/full_demo.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, works everywhere
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from mlci import Experiment
from mlci.core.results import ExperimentResults
from mlci.stats.bootstrap import bootstrap_ci, summary
from mlci.stats.tests import compare, corrected_resampled_ttest
from mlci.stats.anova import decompose_variance
from mlci.sensitivity.learning_curve import learning_curve
from mlci.viz.plots import (
    plot_score_distribution,
    plot_comparison,
    plot_variance_decomposition,
    plot_learning_curve,
    plot_bootstrap_distribution,
)

PLOT_DIR = "examples/plots"
import os
os.makedirs(PLOT_DIR, exist_ok=True)

def banner(title):
    print("\n" + "═" * 62)
    print(f"  {title}")
    print("═" * 62)

def section(title):
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


# ═══════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════

banner("mlci — Full Feature Demo")
X, y = load_breast_cancer(return_X_y=True)
print(f"\n  Dataset : Breast Cancer Wisconsin")
print(f"  Samples : {len(y)}   Features: {X.shape[1]}   Classes: 2")
print(f"  Task    : Binary classification (malignant vs benign)")


# ═══════════════════════════════════════════════════════════════
# PART 1 — Run experiments (Type 1 user: comparing models)
# ═══════════════════════════════════════════════════════════════

banner("PART 1 — Running Experiments (20 seeds × 5 folds each)")

models = {
    "RandomForest":      lambda seed: RandomForestClassifier(n_estimators=50, random_state=seed),
    "GradientBoosting":  lambda seed: GradientBoostingClassifier(n_estimators=50, random_state=seed),
    "LogisticRegression":lambda seed: LogisticRegression(random_state=seed, max_iter=5000, C=0.1),
    "SVM":               lambda seed: SVC(random_state=seed, probability=False),
}

all_results = {}
for name, factory in models.items():
    print(f"\n  → {name}")
    exp = Experiment(
        model_factory=factory,
        X=X, y=y,
        metric="accuracy",
        n_seeds=20,
        n_splits=5,
        model_name=name,
    )
    all_results[name] = exp.run(verbose=True)

rf  = all_results["RandomForest"]
gb  = all_results["GradientBoosting"]
lr  = all_results["LogisticRegression"]
svm = all_results["SVM"]


# ═══════════════════════════════════════════════════════════════
# PART 2 — Bootstrap summaries (proper uncertainty)
# ═══════════════════════════════════════════════════════════════

banner("PART 2 — Bootstrap Summaries with Confidence Intervals")

for res in all_results.values():
    print()
    summary(res, confidence=0.95)


# ═══════════════════════════════════════════════════════════════
# PART 3 — Variance decomposition
# ═══════════════════════════════════════════════════════════════

banner("PART 3 — Variance Decomposition (seed vs split)")

for res in all_results.values():
    print()
    print(decompose_variance(res))


# ═══════════════════════════════════════════════════════════════
# PART 4 — Statistical comparisons
# ═══════════════════════════════════════════════════════════════

banner("PART 4 — Statistical Model Comparisons")

section("RF vs GB — close competitors (all three methods)")
compare(rf, gb, method="all")

section("RF vs LR — clear winner expected")
compare(rf, lr, method="corrected_ttest")

section("GB vs SVM")
compare(gb, svm, method="corrected_ttest")

section("Bayesian: RF vs LR (P(A > B) instead of p-value)")
compare(rf, lr, method="bayesian")


# ═══════════════════════════════════════════════════════════════
# PART 5 — Type 2 user: bring-your-own scores
# ═══════════════════════════════════════════════════════════════

banner("PART 5 — Bring-Your-Own Scores (no Experiment needed)")

print("\n  Simulating scores from an external training run...")
rng = np.random.RandomState(99)
external_scores = 0.961 + rng.normal(0, 0.015, size=(8, 5))
external_scores = np.clip(external_scores, 0, 1)

external_results = ExperimentResults(
    scores=external_scores,
    metric="accuracy",
    model_name="ExternalModel (e.g. PyTorch CNN)",
    higher_is_better=True,
)

print()
summary(external_results)

section("Comparing external model against RandomForest")
# Shapes differ (8 seeds vs 20) so we use the smaller set
rf_small = ExperimentResults(
    scores=rf.scores[:8, :],
    metric=rf.metric,
    model_name=rf.model_name,
)
compare(rf_small, external_results, method="corrected_ttest")


# ═══════════════════════════════════════════════════════════════
# PART 6 — Custom metric
# ═══════════════════════════════════════════════════════════════

banner("PART 6 — Custom Metric (F1 macro)")

section("Running RandomForest with F1 macro")
rf_f1 = Experiment(
    model_factory=lambda seed: RandomForestClassifier(n_estimators=50, random_state=seed),
    X=X, y=y,
    metric="f1",
    n_seeds=10,
    n_splits=5,
    model_name="RandomForest (F1)",
    higher_is_better=True,
).run(verbose=False)

print()
summary(rf_f1)

section("Running LogReg with F1 macro")
lr_f1 = Experiment(
    model_factory=lambda seed: LogisticRegression(random_state=seed, max_iter=5000, C=0.1),
    X=X, y=y,
    metric="f1",
    n_seeds=10,
    n_splits=5,
    model_name="LogReg (F1)",
    higher_is_better=True,
).run(verbose=False)

print()
summary(lr_f1)
compare(rf_f1, lr_f1, method="corrected_ttest")


# ═══════════════════════════════════════════════════════════════
# PART 7 — Learning curves
# ═══════════════════════════════════════════════════════════════

banner("PART 7 — Learning Curves with Uncertainty Bands")

print("\n  Computing learning curves for RF and LR...")
lc_rf = learning_curve(
    model_factory=lambda seed: RandomForestClassifier(n_estimators=50, random_state=seed),
    X=X, y=y,
    train_fractions=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    n_seeds=10, n_splits=5,
    model_name="RandomForest",
)
lc_lr = learning_curve(
    model_factory=lambda seed: LogisticRegression(random_state=seed, max_iter=5000, C=0.1),
    X=X, y=y,
    train_fractions=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    n_seeds=10, n_splits=5,
    model_name="LogisticRegression",
)

print("\n  Learning curve (RF) — mean ± CI at each training fraction:")
for frac, res in zip(lc_rf.train_fractions, lc_rf.results_per_size):
    ci = bootstrap_ci(res)
    print(f"    {frac*100:.0f}% training data  →  {ci.mean:.4f}  [{ci.lower:.4f}, {ci.upper:.4f}]")


# ═══════════════════════════════════════════════════════════════
# PART 8 — Plots
# ═══════════════════════════════════════════════════════════════

banner("PART 8 — Generating Plots")

# 8a. Score distributions for all models
print("\n  Saving: score_distributions.png")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (name, res) in zip(axes.flat, all_results.items()):
    plot_score_distribution(res, ax=ax, show_ci=True)
plt.suptitle("Score Distributions (20 seeds × 5 folds)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/score_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# 8b. Forest plot comparison
print("  Saving: model_comparison.png")
fig = plot_comparison(list(all_results.values()), confidence=0.95,
                      title="Model Comparison — Breast Cancer")
fig.savefig(f"{PLOT_DIR}/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# 8c. Variance decompositions
print("  Saving: variance_decomposition.png")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax_pair, res in zip([(axes[0], axes[1])], [rf]):
    big_fig = plot_variance_decomposition(res)
big_fig.savefig(f"{PLOT_DIR}/variance_decomposition.png", dpi=150, bbox_inches="tight")
plt.close("all")

# 8d. Learning curve (both models overlaid)
print("  Saving: learning_curve.png")
fig = plot_learning_curve([lc_rf, lc_lr],
                          title="Learning Curve — RF vs Logistic Regression")
fig.savefig(f"{PLOT_DIR}/learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# 8e. Bootstrap distribution
print("  Saving: bootstrap_distribution.png")
fig = plot_bootstrap_distribution(rf, confidence=0.95, n_bootstrap=20_000)
fig.savefig(f"{PLOT_DIR}/bootstrap_distribution.png", dpi=150, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════
# PART 9 — Save / load roundtrip
# ═══════════════════════════════════════════════════════════════

banner("PART 9 — Save & Load Results (JSON roundtrip)")

save_path = "examples/rf_results.json"
rf.save(save_path)
print(f"\n  Saved RF results → {save_path}")

loaded = ExperimentResults.load(save_path)
print(f"  Loaded back      → {loaded}")

ci_orig   = bootstrap_ci(rf)
ci_loaded = bootstrap_ci(loaded)
print(f"\n  Original CI : [{ci_orig.lower:.4f}, {ci_orig.upper:.4f}]")
print(f"  Loaded CI   : [{ci_loaded.lower:.4f}, {ci_loaded.upper:.4f}]")
print(f"  Match       : {abs(ci_orig.mean - ci_loaded.mean) < 1e-10}")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

banner("FINAL SUMMARY")

print("\n  Model performance (mean accuracy, 95% CI):\n")
for name, res in sorted(all_results.items(), key=lambda x: -x[1].mean):
    ci = bootstrap_ci(res)
    bar = "█" * int(ci.mean * 40)
    print(f"  {name:<22} {ci.mean:.4f}  [{ci.lower:.4f}, {ci.upper:.4f}]  {bar}")

print("\n  Key comparison results:\n")
pairs = [("RandomForest", "GradientBoosting"), ("RandomForest", "LogisticRegression"),
         ("GradientBoosting", "SVM")]
for a_name, b_name in pairs:
    r = corrected_resampled_ttest(all_results[a_name], all_results[b_name])
    sig = "✓ significant" if r.p_value < 0.05 else "✗ not significant"
    print(f"  {a_name} vs {b_name}")
    print(f"    effect={r.effect_size:+.4f}  p={r.p_value:.4f}  → {sig}")
    print()

print(f"\n  Plots saved to: {PLOT_DIR}/")
print(f"    • score_distributions.png")
print(f"    • model_comparison.png")
print(f"    • variance_decomposition.png")
print(f"    • learning_curve.png")
print(f"    • bootstrap_distribution.png")
print()
