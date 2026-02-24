"""
examples/bayesian_example.py
-----------------------------
Demonstrates the PyMC-based full Bayesian hierarchical comparison.

Requires:
    pip install mlci[bayesian]

Runtime: ~2–4 minutes (MCMC sampling).
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer

from mlci import Experiment
from mlci.stats.bootstrap import summary
from mlci.stats.bayesian_hierarchical import bayesian_hierarchical_compare

# ── 1. Run experiments ─────────────────────────────────────────────────────

X, y = load_breast_cancer(return_X_y=True)

rf = Experiment(
    model_factory=lambda seed: RandomForestClassifier(random_state=seed),
    X=X, y=y, metric="accuracy", n_seeds=20, n_splits=5,
    model_name="RandomForest",
)
gb = Experiment(
    model_factory=lambda seed: GradientBoostingClassifier(random_state=seed),
    X=X, y=y, metric="accuracy", n_seeds=20, n_splits=5,
    model_name="GradientBoosting",
)

print("Running RandomForest …")
rf_results = rf.run()
print("Running GradientBoosting …")
gb_results = gb.run()

# ── 2. Bootstrap summaries ─────────────────────────────────────────────────

print("\n── Bootstrap Summary ──")
summary(rf_results)
summary(gb_results)

# ── 3. Full Bayesian hierarchical comparison ────────────────────────────────

print("\nRunning PyMC hierarchical model …")
result = bayesian_hierarchical_compare(
    rf_results,
    gb_results,
    rope=0.005,          # ±0.5 percentage points = practically irrelevant
    hdi_prob=0.94,
    draws=2000,
    tune=1000,
    chains=4,
    target_accept=0.9,
    random_seed=42,
    progressbar=True,
    plot=True,
    plot_path="examples/plots/",
)
print(result)

# ── 4. Interpret the result ────────────────────────────────────────────────

print("Posterior mean difference:", f"{result.mu_mean:+.4f}")
print("P(RandomForest > GradientBoosting):", f"{result.prob_a_better:.3f}")
print(f"Fraction of posterior in ROPE {result.rope}:", f"{result.rope_fraction:.1%}")

# ── 5. Access the full ArviZ InferenceData for further analysis ────────────

idata = result.idata
print("\nArviZ summary for key parameters:")
import arviz as az
print(az.summary(idata, var_names=["mu_global", "sigma_within", "sigma_between"]))
