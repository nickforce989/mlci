"""
mlci — sklearn integration example
=====================================

Demonstrates mlci.integrations.sklearn:
  - PipelineFactory  : build model+scaler pipelines from config dicts
  - wrap_cross_val_scores : bring your own scores into mlci
  - ModelGrid        : run a grid of models in one call

Also shows how all three feed naturally into mlci's stats and viz.

Run with:
    python examples/sklearn_example.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlci.integrations.sklearn import PipelineFactory, wrap_cross_val_scores, ModelGrid
from mlci import Experiment
from mlci.stats.bootstrap import bootstrap_ci, summary
from mlci.stats.tests import compare
from mlci.stats.anova import decompose_variance
from mlci.sensitivity.learning_curve import learning_curve
from mlci.viz.plots import (
    plot_comparison,
    plot_score_distribution,
    plot_learning_curve,
    plot_variance_decomposition,
)

PLOT_DIR = "examples/sklearn_plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def banner(title):
    print("\n" + "═" * 64)
    print(f"  {title}")
    print("═" * 64)

def section(title):
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}")


# ═══════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════

banner("mlci — sklearn Integration Example")

X, y = load_breast_cancer(return_X_y=True)
print(f"\n  Dataset : Breast Cancer Wisconsin")
print(f"  Samples : {len(y)}   Features: {X.shape[1]}   Classes: 2")


# ═══════════════════════════════════════════════════════════════
# PART 1 — PipelineFactory
# ═══════════════════════════════════════════════════════════════
#
# PipelineFactory builds a scaler + model pipeline from a config dict.
# The seed is injected automatically into random_state at call time.
# Useful when you want to define experiments declaratively.

banner("PART 1 — PipelineFactory")

section("Defining models as config dicts instead of lambdas")

# These three are equivalent ways to define the same experiment:

# --- Standard lambda approach (what you've seen before) ---
factory_lambda = lambda seed: Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(random_state=seed, max_iter=2000, C=0.1)),
])

# --- PipelineFactory approach (cleaner, declarative) ---
factory_pf = PipelineFactory(
    LogisticRegression,
    params={"max_iter": 2000, "C": 0.1},
    scaler="standard",
)

# --- PipelineFactory with different scalers ---
factory_robust = PipelineFactory(
    LogisticRegression,
    params={"max_iter": 2000, "C": 0.1},
    scaler="robust",      # better for data with outliers
)
factory_minmax = PipelineFactory(
    LogisticRegression,
    params={"max_iter": 2000, "C": 0.1},
    scaler="minmax",
)
factory_none = PipelineFactory(
    RandomForestClassifier,
    params={"n_estimators": 100},
    scaler="none",        # tree models don't need scaling
)

print("\n  Factories defined:")
print(f"    {factory_pf}")
print(f"    {factory_robust}")
print(f"    {factory_none}")

# Verify a factory produces a working model
model = factory_pf(seed=42)
model.fit(X, y)
print(f"\n  Quick check — factory_pf(seed=42) trains OK: {model.score(X, y):.4f}")

section("Running PipelineFactory models through Experiment")

results_standard = Experiment(
    model_factory=factory_pf,
    X=X, y=y,
    metric="accuracy",
    n_seeds=15,
    n_splits=5,
    model_name="LogReg (StandardScaler)",
).run(verbose=True)

results_robust = Experiment(
    model_factory=factory_robust,
    X=X, y=y,
    metric="accuracy",
    n_seeds=15,
    n_splits=5,
    model_name="LogReg (RobustScaler)",
).run(verbose=True)

results_minmax = Experiment(
    model_factory=factory_minmax,
    X=X, y=y,
    metric="accuracy",
    n_seeds=15,
    n_splits=5,
    model_name="LogReg (MinMaxScaler)",
).run(verbose=True)

section("Does the choice of scaler matter? Statistical comparison")

print()
summary(results_standard)
print()
summary(results_robust)
print()
summary(results_minmax)

print()
compare(results_standard, results_robust,  method="corrected_ttest")
compare(results_standard, results_minmax,  method="corrected_ttest")


# ═══════════════════════════════════════════════════════════════
# PART 2 — wrap_cross_val_scores
# ═══════════════════════════════════════════════════════════════
#
# You already have CV scores from your own code — maybe sklearn's
# cross_val_score, maybe a custom loop, maybe a collaborator sent
# you a CSV of results. wrap_cross_val_scores brings them into mlci
# without rerunning anything.

banner("PART 2 — wrap_cross_val_scores (bring your own scores)")

section("Scenario: scores from sklearn cross_val_score across multiple seeds")

print("\n  Running cross_val_score manually across 8 seeds...")
external_scores = []
for seed in range(8):
    scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=seed),
        X, y, cv=5, scoring="accuracy",
    )
    external_scores.append(scores)
    print(f"    seed {seed}: {scores.round(4)}  mean={scores.mean():.4f}")

external_scores = np.array(external_scores)   # shape (8, 5)

# Wrap into mlci
rf_external = wrap_cross_val_scores(
    external_scores,
    metric="accuracy",
    model_name="RF (external cross_val_score)",
)

print()
summary(rf_external)

section("Scenario: scores from an external source (e.g. a collaborator's results)")

# Simulating results sent from another researcher's machine
# (different hardware, different random state implementation, etc.)
# You just have the numbers — no way to rerun.
collab_scores = np.array([
    [0.956, 0.947, 0.965, 0.938, 0.951],
    [0.961, 0.952, 0.943, 0.957, 0.948],
    [0.948, 0.963, 0.955, 0.944, 0.960],
    [0.952, 0.941, 0.958, 0.963, 0.947],
    [0.965, 0.949, 0.952, 0.956, 0.942],
])

collab_results = wrap_cross_val_scores(
    collab_scores,
    metric="accuracy",
    model_name="Collaborator's Model",
)

print()
summary(collab_results)

section("Comparing your RF against collaborator's model")

# The paired comparison requires the same number of seeds.
# Trim the larger one to match.
n_common = min(rf_external.n_seeds, collab_results.n_seeds)
rf_trimmed = wrap_cross_val_scores(
    rf_external.scores[:n_common],
    metric="accuracy",
    model_name=rf_external.model_name,
)
compare(rf_trimmed, collab_results, method="all")

section("Scenario: 1D scores (single run, multiple folds only)")

# Some people only run one seed — wrap_cross_val_scores handles 1D too
single_run_scores = cross_val_score(
    GradientBoostingClassifier(n_estimators=100, random_state=0),
    X, y, cv=10, scoring="accuracy",
)
print(f"\n  Single run, 10-fold CV scores: {single_run_scores.round(4)}")

gb_single = wrap_cross_val_scores(
    single_run_scores,
    metric="accuracy",
    model_name="GradientBoosting (1 seed, 10 folds)",
)
print()
summary(gb_single)
print("\n  Note: with only 1 seed the CI is wider — more seeds = narrower CI.")


# ═══════════════════════════════════════════════════════════════
# PART 3 — ModelGrid
# ═══════════════════════════════════════════════════════════════
#
# Run many models in one call. Returns {name: ExperimentResults}
# ready for comparison and plotting.

banner("PART 3 — ModelGrid (run many models at once)")

section("Large model grid — 8 classifiers on breast cancer")

grid = ModelGrid(
    models={
        "LogisticRegression": PipelineFactory(LogisticRegression,
                                params={"max_iter": 2000, "C": 1.0}, scaler="standard"),
        "RidgeClassifier":    PipelineFactory(RidgeClassifier,
                                scaler="standard"),
        "SVM (RBF)":          PipelineFactory(SVC,
                                params={"kernel": "rbf", "C": 1.0}, scaler="standard"),
        "KNN":                PipelineFactory(KNeighborsClassifier,
                                params={"n_neighbors": 5}, scaler="standard"),
        "GaussianNB":         PipelineFactory(GaussianNB, scaler="standard"),
        "RandomForest":       PipelineFactory(RandomForestClassifier,
                                params={"n_estimators": 100}, scaler="none"),
        "ExtraTrees":         PipelineFactory(ExtraTreesClassifier,
                                params={"n_estimators": 100}, scaler="none"),
        "GradientBoosting":   PipelineFactory(GradientBoostingClassifier,
                                params={"n_estimators": 100}, scaler="none"),
    },
    X=X, y=y,
    metric="accuracy",
    n_seeds=15,
    n_splits=5,
)

all_results = grid.run(verbose=True)

section("Bootstrap summaries for all models")
for name, res in all_results.items():
    print()
    summary(res)

section("Ranking table")
print(f"\n  {'Model':<26} {'Mean':>7} {'CI Lower':>9} {'CI Upper':>9} {'Width':>8}")
print(f"  {'─'*26} {'─'*7} {'─'*9} {'─'*9} {'─'*8}")
for name, res in sorted(all_results.items(), key=lambda x: -x[1].mean):
    ci = bootstrap_ci(res)
    print(f"  {name:<26} {ci.mean:>7.4f} {ci.lower:>9.4f} {ci.upper:>9.4f} {ci.upper-ci.lower:>8.4f}")

section("Key pairwise comparisons (best vs rest)")
best_name = max(all_results, key=lambda n: all_results[n].mean)
best_res  = all_results[best_name]
print(f"\n  Best model: {best_name} (mean={best_res.mean:.4f})\n")
for name, res in all_results.items():
    if name == best_name:
        continue
    r = compare(best_res, res, method="corrected_ttest")


# ═══════════════════════════════════════════════════════════════
# PART 4 — Multi-dataset ModelGrid
# ═══════════════════════════════════════════════════════════════
#
# Run the same grid on multiple datasets. Classic benchmark setup.

banner("PART 4 — Same ModelGrid Across Multiple Datasets")

datasets = {
    "BreastCancer": load_breast_cancer(return_X_y=True),
    "Wine":         load_wine(return_X_y=True),
    "Digits":       load_digits(return_X_y=True),
}

small_grid_factories = {
    "LogisticRegression": PipelineFactory(LogisticRegression,
                            params={"max_iter": 2000}, scaler="standard"),
    "RandomForest":       PipelineFactory(RandomForestClassifier,
                            params={"n_estimators": 50}, scaler="none"),
    "SVM":                PipelineFactory(SVC,
                            params={"kernel": "rbf"}, scaler="standard"),
}

multi_results = {}   # {(dataset_name, model_name): ExperimentResults}

for ds_name, (Xd, yd) in datasets.items():
    print(f"\n  Dataset: {ds_name} ({len(yd)} samples)")
    g = ModelGrid(
        models=small_grid_factories,
        X=Xd, y=yd,
        metric="accuracy",
        n_seeds=10,
        n_splits=5,
    )
    for model_name, res in g.run(verbose=True).items():
        multi_results[(ds_name, model_name)] = res

section("Cross-dataset ranking table")
print(f"\n  {'Model':<22}", end="")
for ds_name in datasets:
    print(f"  {ds_name:>14}", end="")
print()
print(f"  {'─'*22}", end="")
for _ in datasets:
    print(f"  {'─'*14}", end="")
print()

for model_name in small_grid_factories:
    print(f"  {model_name:<22}", end="")
    for ds_name in datasets:
        ci = bootstrap_ci(multi_results[(ds_name, model_name)])
        print(f"  {ci.mean:.4f} ±{ci.upper-ci.mean:.4f}", end="")
    print()


# ═══════════════════════════════════════════════════════════════
# PART 5 — Variance decomposition across scalers
# ═══════════════════════════════════════════════════════════════

banner("PART 5 — Variance Decomposition")

print("\n  Comparing variance sources across scaler choices:\n")
for res in [results_standard, results_robust, results_minmax]:
    print()
    print(decompose_variance(res))


# ═══════════════════════════════════════════════════════════════
# PART 6 — Learning curves using PipelineFactory
# ═══════════════════════════════════════════════════════════════

banner("PART 6 — Learning Curves")

lc_lr = learning_curve(
    model_factory=PipelineFactory(LogisticRegression,
                    params={"max_iter": 2000, "C": 1.0}, scaler="standard"),
    X=X, y=y,
    train_fractions=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    n_seeds=8, n_splits=5,
    model_name="LogisticRegression",
)
lc_rf = learning_curve(
    model_factory=PipelineFactory(RandomForestClassifier,
                    params={"n_estimators": 100}, scaler="none"),
    X=X, y=y,
    train_fractions=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    n_seeds=8, n_splits=5,
    model_name="RandomForest",
)
lc_svm = learning_curve(
    model_factory=PipelineFactory(SVC, params={"kernel": "rbf"}, scaler="standard"),
    X=X, y=y,
    train_fractions=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    n_seeds=8, n_splits=5,
    model_name="SVM",
)

print(f"\n  {'Fraction':>10}  {'LogReg':>8}  {'RF':>8}  {'SVM':>8}")
print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}")
for frac, rlr, rrf, rsvm in zip(
    lc_lr.train_fractions,
    lc_lr.results_per_size,
    lc_rf.results_per_size,
    lc_svm.results_per_size,
):
    print(
        f"  {frac*100:>9.0f}%  "
        f"{bootstrap_ci(rlr).mean:>8.4f}  "
        f"{bootstrap_ci(rrf).mean:>8.4f}  "
        f"{bootstrap_ci(rsvm).mean:>8.4f}"
    )


# ═══════════════════════════════════════════════════════════════
# PART 7 — Plots
# ═══════════════════════════════════════════════════════════════

banner("PART 7 — Generating Plots")

# Forest plot: all 8 models from ModelGrid
print(f"\n  Saving: sklearn_model_comparison.png")
fig = plot_comparison(
    list(all_results.values()),
    title="8 Classifiers — Breast Cancer (accuracy, 95% CI)",
)
fig.savefig(f"{PLOT_DIR}/sklearn_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# Scaler comparison
print(f"  Saving: sklearn_scaler_comparison.png")
fig = plot_comparison(
    [results_standard, results_robust, results_minmax],
    title="Scaler Choice — LogisticRegression (does it matter?)",
)
fig.savefig(f"{PLOT_DIR}/sklearn_scaler_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# Score distributions for top 3 models
print(f"  Saving: sklearn_score_distributions.png")
top3 = sorted(all_results.items(), key=lambda x: -x[1].mean)[:3]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Top 3 Models — Score Distributions", fontsize=12)
colors = ["#2E4057", "#E84855", "#3BB273"]
for ax, (name, res), color in zip(axes, top3, colors):
    import seaborn as sns
    ci = bootstrap_ci(res)
    sns.kdeplot(res.flat, ax=ax, color=color, fill=True, alpha=0.3, linewidth=2)
    ax.axvline(ci.mean, color=color, linewidth=2, linestyle="--")
    ax.axvspan(ci.lower, ci.upper, alpha=0.15, color=color)
    ax.set_title(f"{name}\n{ci.mean:.4f} [{ci.lower:.4f}, {ci.upper:.4f}]", fontsize=9)
    ax.set_xlabel("Accuracy")
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/sklearn_score_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# Learning curve: 3 models
print(f"  Saving: sklearn_learning_curve.png")
fig = plot_learning_curve(
    [lc_lr, lc_rf, lc_svm],
    title="Learning Curves: LogReg vs RF vs SVM (bands = 95% CI)",
)
fig.savefig(f"{PLOT_DIR}/sklearn_learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# Cross-dataset heatmap
print(f"  Saving: sklearn_cross_dataset_heatmap.png")
model_names_small = list(small_grid_factories.keys())
ds_names = list(datasets.keys())
heat = np.array([
    [bootstrap_ci(multi_results[(ds, m)]).mean for ds in ds_names]
    for m in model_names_small
])
fig, ax = plt.subplots(figsize=(8, 4))
import seaborn as sns
sns.heatmap(
    heat, annot=True, fmt=".4f",
    xticklabels=ds_names, yticklabels=model_names_small,
    cmap="YlOrRd", ax=ax, linewidths=0.5,
    vmin=heat.min() - 0.01, vmax=heat.max() + 0.01,
)
ax.set_title("Mean Accuracy Across Datasets (15 seeds × 5 folds)", fontsize=11)
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/sklearn_cross_dataset_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# Variance decomposition for best model
print(f"  Saving: sklearn_variance_decomposition.png")
fig = plot_variance_decomposition(best_res)
fig.savefig(f"{PLOT_DIR}/sklearn_variance_decomposition.png", dpi=150, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

banner("FINAL SUMMARY")

print("\n  ── All 8 models on Breast Cancer ──\n")
for name, res in sorted(all_results.items(), key=lambda x: -x[1].mean):
    ci = bootstrap_ci(res)
    bar = "█" * int(ci.mean * 40)
    print(f"  {name:<26} {ci.mean:.4f}  [{ci.lower:.4f}, {ci.upper:.4f}]  {bar}")

print("\n  ── Does scaler choice matter for LogReg? ──\n")
for res in [results_standard, results_robust, results_minmax]:
    ci = bootstrap_ci(res)
    print(f"  {res.model_name:<30} {ci.mean:.4f}  [{ci.lower:.4f}, {ci.upper:.4f}]")

print(f"\n  Plots saved to: {PLOT_DIR}/")
for f in sorted(os.listdir(PLOT_DIR)):
    if f.endswith(".png"):
        print(f"    • {f}")
print()
