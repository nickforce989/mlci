"""
mlci worked example — the core value proposition in one script.

This demonstrates the main claim of the library:

  "Model A beats Model B by 0.3% on accuracy"
  — is this meaningful, or just noise?

With standard practice (single run, no CI): you can't tell.
With mlci: you can.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import mlci
from mlci import Experiment
from mlci.stats.bootstrap import summary
from mlci.stats.tests import compare
from mlci.stats.anova import decompose_variance

print("=" * 60)
print("  mlci: Statistically Rigorous ML Benchmarking")
print("=" * 60)

# -----------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------

X, y = load_breast_cancer(return_X_y=True)
print(f"\nDataset: Breast Cancer ({len(y)} samples, binary classification)\n")

# -----------------------------------------------------------------------
# Standard practice: single run, point estimate
# -----------------------------------------------------------------------

print("─" * 60)
print("STANDARD PRACTICE: single run, no uncertainty estimate")
print("─" * 60)

from sklearn.model_selection import cross_val_score

rf_single = cross_val_score(
    RandomForestClassifier(n_estimators=50, random_state=0),
    X, y, cv=5, scoring="accuracy"
).mean()

gb_single = cross_val_score(
    GradientBoostingClassifier(random_state=0),
    X, y, cv=5, scoring="accuracy"
).mean()

print(f"  RandomForest     : {rf_single:.4f}")
print(f"  GradientBoosting : {gb_single:.4f}")
diff = abs(rf_single - gb_single)
print(f"  Difference       : {diff:.4f}")
print(f"  Conclusion from standard practice: ", end="")
if diff > 0.001:
    winner = "RandomForest" if rf_single > gb_single else "GradientBoosting"
    print(f"{winner} is better. (But is this real?)")
else:
    print("They're equal. (But are they?)")

# -----------------------------------------------------------------------
# mlci: proper uncertainty quantification
# -----------------------------------------------------------------------

print("\n" + "─" * 60)
print("mlci: repeated evaluation with statistical rigour")
print("─" * 60)

rf_exp = Experiment(
    model_factory=lambda seed: RandomForestClassifier(n_estimators=50, random_state=seed),
    X=X, y=y,
    metric="accuracy",
    n_seeds=20,
    n_splits=5,
    model_name="RandomForest",
)

gb_exp = Experiment(
    model_factory=lambda seed: GradientBoostingClassifier(random_state=seed),
    X=X, y=y,
    metric="accuracy",
    n_seeds=20,
    n_splits=5,
    model_name="GradientBoosting",
)

lr_exp = Experiment(
    model_factory=lambda seed: LogisticRegression(random_state=seed, max_iter=2000),
    X=X, y=y,
    metric="accuracy",
    n_seeds=20,
    n_splits=5,
    model_name="LogisticRegression",
)

print("\nRunning RandomForest...")
rf_results = rf_exp.run(verbose=True)
print("\nRunning GradientBoosting...")
gb_results = gb_exp.run(verbose=True)
print("\nRunning LogisticRegression...")
lr_results = lr_exp.run(verbose=True)

# -----------------------------------------------------------------------
# Summary with CIs
# -----------------------------------------------------------------------

print("\n" + "─" * 60)
print("SUMMARY WITH UNCERTAINTY")
print("─" * 60)
for res in [rf_results, gb_results, lr_results]:
    print()
    summary(res)

# -----------------------------------------------------------------------
# Statistical comparisons
# -----------------------------------------------------------------------

print("\n" + "─" * 60)
print("STATISTICAL COMPARISON: RandomForest vs GradientBoosting")
print("─" * 60)
print()
compare(rf_results, gb_results, method="all")

print("\n" + "─" * 60)
print("STATISTICAL COMPARISON: RandomForest vs LogisticRegression")
print("─" * 60)
print()
compare(rf_results, lr_results, method="corrected_ttest")

# -----------------------------------------------------------------------
# Variance decomposition
# -----------------------------------------------------------------------

print("\n" + "─" * 60)
print("WHERE DOES THE VARIANCE COME FROM?")
print("─" * 60)
for res in [rf_results, gb_results, lr_results]:
    print()
    print(decompose_variance(res))

# -----------------------------------------------------------------------
# The key message
# -----------------------------------------------------------------------

from mlci.stats.bootstrap import bootstrap_ci
from mlci.stats.tests import corrected_resampled_ttest

rf_ci = bootstrap_ci(rf_results)
gb_ci = bootstrap_ci(gb_results)
comparison = corrected_resampled_ttest(rf_results, gb_results)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n  Standard practice reported:")
print(f"    RF={rf_single:.4f}  GB={gb_single:.4f}  Δ={abs(rf_single-gb_single):.4f}")
print(f"\n  mlci reports:")
print(f"    RF = {rf_ci.mean:.4f}  95% CI [{rf_ci.lower:.4f}, {rf_ci.upper:.4f}]")
print(f"    GB = {gb_ci.mean:.4f}  95% CI [{gb_ci.lower:.4f}, {gb_ci.upper:.4f}]")
print(f"\n  Corrected t-test p-value : {comparison.p_value:.4f}")
print(f"  Effect size              : {comparison.effect_size:+.4f}")
print(f"  Conclusion: {comparison.conclusion}")
print()
