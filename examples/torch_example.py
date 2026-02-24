"""
mlci — PyTorch integration example
====================================

Demonstrates mlci.integrations.torch.TorchTrainer:
  - Define any nn.Module architecture
  - Wrap it in TorchTrainer (sklearn-compatible)
  - Pass it directly to mlci.Experiment — works identically to RandomForest
  - Get confidence intervals, comparisons, variance decomposition
  - Caching: completed runs saved to disk, skipped on re-run

Requirements:
    pip install torch
    pip install -e .   (from inside your mlci/ folder)

Run with:
    python examples/torch_example.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Check torch is available ────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")
except ImportError:
    print("PyTorch not found. Install it with:")
    print("  pip install torch")
    print("See https://pytorch.org/get-started/locally/ for platform-specific instructions.")
    exit(1)

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mlci import Experiment
from mlci.integrations.torch import TorchTrainer, seed_torch
from mlci.integrations.sklearn import PipelineFactory
from mlci.stats.bootstrap import bootstrap_ci, summary
from mlci.stats.tests import compare
from mlci.stats.anova import decompose_variance
from mlci.sensitivity.learning_curve import learning_curve
from mlci.viz.plots import plot_comparison, plot_learning_curve, plot_score_distribution

PLOT_DIR = "examples/torch_plots"
CACHE_DIR = "examples/torch_cache"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def banner(title):
    print("\n" + "═" * 62)
    print(f"  {title}")
    print("═" * 62)

def section(title):
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


# ═══════════════════════════════════════════════════════════════
# STEP 1 — Define your neural network architectures
# ═══════════════════════════════════════════════════════════════
#
# Any nn.Module works. The only rule: output layer must produce
# raw logits (no softmax/sigmoid — TorchTrainer applies them internally).

class SmallMLP(nn.Module):
    """Two hidden layers: 30 → 64 → 32 → 1"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class DeepMLP(nn.Module):
    """Four hidden layers with batch normalisation: 30 → 128 → 64 → 32 → 16 → 1"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


class WideMLP(nn.Module):
    """One wide hidden layer: 30 → 256 → 1"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════
# STEP 2 — Load data
# ═══════════════════════════════════════════════════════════════

banner("mlci — PyTorch Integration Example")

X, y = load_breast_cancer(return_X_y=True)
print(f"\n  Dataset  : Breast Cancer Wisconsin")
print(f"  Samples  : {len(y)}   Features: {X.shape[1]}   Classes: 2")
print(f"  Cache dir: {CACHE_DIR}  (completed runs saved here)")
print(f"\n  Note: TorchTrainer scales features internally via StandardScaler.")
print(f"  You do NOT need to scale X yourself.")


# ═══════════════════════════════════════════════════════════════
# PART 1 — Single verbose training run
# ═══════════════════════════════════════════════════════════════

banner("PART 1 — Single Training Run (verbose)")

section("SmallMLP: 30 → 64 → 32 → 1")

# Scale manually for this standalone demo
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

trainer = TorchTrainer(
    model_fn=SmallMLP,
    lr=1e-3,
    n_epochs=100,
    batch_size=32,
    patience=15,
    weight_decay=1e-4,
    random_state=42,
    verbose=True,
)
trainer.fit(X_scaled, y)

print(f"\n  Training accuracy : {trainer.score(X_scaled, y):.4f}")
print(f"  Epochs run        : {len(trainer.train_losses_)}")
print(f"  Best val loss     : {min(trainer.val_losses_):.4f}")


# ═══════════════════════════════════════════════════════════════
# PART 2 — Plugging TorchTrainer into mlci.Experiment
# ═══════════════════════════════════════════════════════════════
#
# This is the key step. model_factory is called with seed as the
# only argument — TorchTrainer takes it as random_state, which
# seeds torch + numpy + random + CUDA all at once.
#
# The Pipeline wraps TorchTrainer with a StandardScaler so
# features are scaled correctly on each train/test split.

banner("PART 2 — Running mlci Experiments (10 seeds × 5 folds)")

def make_torch_model(model_fn, lr=1e-3, n_epochs=100, seed=0):
    """Factory: returns a sklearn Pipeline(scaler + TorchTrainer)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("net", TorchTrainer(
            model_fn=model_fn,
            lr=lr,
            n_epochs=n_epochs,
            batch_size=32,
            patience=15,
            weight_decay=1e-4,
            random_state=seed,
            cache_dir=CACHE_DIR,   # ← caching: re-runs skip completed fits
            verbose=False,
        )),
    ])

print("\n  → SmallMLP")
exp_small = Experiment(
    model_factory=lambda seed: make_torch_model(SmallMLP, seed=seed),
    X=X, y=y,
    metric="accuracy",
    n_seeds=10,
    n_splits=5,
    model_name="SmallMLP",
)
res_small = exp_small.run(verbose=True)

print("\n  → DeepMLP (BatchNorm + Dropout)")
exp_deep = Experiment(
    model_factory=lambda seed: make_torch_model(DeepMLP, n_epochs=120, seed=seed),
    X=X, y=y,
    metric="accuracy",
    n_seeds=10,
    n_splits=5,
    model_name="DeepMLP",
)
res_deep = exp_deep.run(verbose=True)

print("\n  → WideMLP")
exp_wide = Experiment(
    model_factory=lambda seed: make_torch_model(WideMLP, seed=seed),
    X=X, y=y,
    metric="accuracy",
    n_seeds=10,
    n_splits=5,
    model_name="WideMLP",
)
res_wide = exp_wide.run(verbose=True)

# Classical baselines for comparison
print("\n  → RandomForest (baseline)")
exp_rf = Experiment(
    model_factory=lambda seed: RandomForestClassifier(n_estimators=100, random_state=seed),
    X=X, y=y,
    metric="accuracy",
    n_seeds=10,
    n_splits=5,
    model_name="RandomForest",
)
res_rf = exp_rf.run(verbose=True)

print("\n  → LogisticRegression (baseline)")
exp_lr = Experiment(
    model_factory=lambda seed: Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(random_state=seed, max_iter=2000)),
    ]),
    X=X, y=y,
    metric="accuracy",
    n_seeds=10,
    n_splits=5,
    model_name="LogisticRegression",
)
res_lr = exp_lr.run(verbose=True)


# ═══════════════════════════════════════════════════════════════
# PART 3 — Bootstrap summaries
# ═══════════════════════════════════════════════════════════════

banner("PART 3 — Bootstrap Summaries with Confidence Intervals")

all_results = {
    "SmallMLP":         res_small,
    "DeepMLP":          res_deep,
    "WideMLP":          res_wide,
    "RandomForest":     res_rf,
    "LogisticRegression": res_lr,
}

for name, res in all_results.items():
    print()
    summary(res, confidence=0.95)


# ═══════════════════════════════════════════════════════════════
# PART 4 — Statistical comparisons
# ═══════════════════════════════════════════════════════════════

banner("PART 4 — Statistical Comparisons")

section("Best PyTorch model vs RandomForest (all three methods)")
best_torch = max([res_small, res_deep, res_wide], key=lambda r: r.mean)
print(f"\n  Best PyTorch model: {best_torch.model_name}  (mean={best_torch.mean:.4f})")
compare(best_torch, res_rf, method="all")

section("Best PyTorch model vs LogisticRegression")
compare(best_torch, res_lr, method="corrected_ttest")

section("SmallMLP vs DeepMLP — is extra depth worth it?")
compare(res_small, res_deep, method="corrected_ttest")

section("Bayesian: SmallMLP vs RandomForest")
compare(res_small, res_rf, method="bayesian")


# ═══════════════════════════════════════════════════════════════
# PART 5 — Variance decomposition
# ═══════════════════════════════════════════════════════════════

banner("PART 5 — Variance Decomposition")

print("\n  Key question: for neural networks, is variance dominated by")
print("  random weight initialisation (seed) or by data split?\n")

for name, res in all_results.items():
    print()
    print(decompose_variance(res))


# ═══════════════════════════════════════════════════════════════
# PART 6 — Caching demonstration
# ═══════════════════════════════════════════════════════════════

banner("PART 6 — Caching Demonstration")

print("\n  Re-running SmallMLP experiment — cached runs will be skipped.")
print("  This should complete near-instantly.\n")

import time
t0 = time.time()
res_small_rerun = Experiment(
    model_factory=lambda seed: make_torch_model(SmallMLP, seed=seed),
    X=X, y=y,
    metric="accuracy",
    n_seeds=10,
    n_splits=5,
    model_name="SmallMLP",
).run(verbose=True)
elapsed = time.time() - t0

print(f"\n  Time to re-run with cache: {elapsed:.1f}s")
print(f"  Scores match original    : "
      f"{np.allclose(res_small.scores, res_small_rerun.scores)}")

n_cached = len(os.listdir(CACHE_DIR))
print(f"  Cache files on disk      : {n_cached}")


# ═══════════════════════════════════════════════════════════════
# PART 7 — Learning curves
# ═══════════════════════════════════════════════════════════════

banner("PART 7 — Learning Curves with Uncertainty Bands")

print("\n  Computing learning curves for SmallMLP vs RandomForest...")

lc_torch = learning_curve(
    model_factory=lambda seed: make_torch_model(SmallMLP, seed=seed),
    X=X, y=y,
    train_fractions=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    n_seeds=5, n_splits=5,
    model_name="SmallMLP",
)
lc_rf = learning_curve(
    model_factory=lambda seed: RandomForestClassifier(n_estimators=100, random_state=seed),
    X=X, y=y,
    train_fractions=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    n_seeds=5, n_splits=5,
    model_name="RandomForest",
)

print(f"\n  {'Fraction':>10}  {'MLP mean':>10}  {'MLP CI':>20}  "
      f"{'RF mean':>10}  {'RF CI':>20}")
print(f"  {'─'*10}  {'─'*10}  {'─'*20}  {'─'*10}  {'─'*20}")
for frac, rm, rr in zip(lc_torch.train_fractions,
                         lc_torch.results_per_size,
                         lc_rf.results_per_size):
    cm = bootstrap_ci(rm)
    cr = bootstrap_ci(rr)
    print(
        f"  {frac*100:>9.0f}%  {cm.mean:>10.4f}  "
        f"[{cm.lower:.4f}, {cm.upper:.4f}]  "
        f"{cr.mean:>10.4f}  [{cr.lower:.4f}, {cr.upper:.4f}]"
    )


# ═══════════════════════════════════════════════════════════════
# PART 8 — Plots
# ═══════════════════════════════════════════════════════════════

banner("PART 8 — Generating Plots")

# Forest plot
print(f"\n  Saving: torch_model_comparison.png")
fig = plot_comparison(list(all_results.values()),
                      title="PyTorch MLPs vs Classical Models (accuracy, 95% CI)")
fig.savefig(f"{PLOT_DIR}/torch_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# Score distribution for best torch model
print(f"  Saving: torch_score_distribution.png")
fig = plot_score_distribution(best_torch, show_ci=True)
fig.savefig(f"{PLOT_DIR}/torch_score_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# Learning curve
print(f"  Saving: torch_learning_curve.png")
fig = plot_learning_curve([lc_torch, lc_rf],
                          title="Learning Curve: SmallMLP vs RandomForest")
fig.savefig(f"{PLOT_DIR}/torch_learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# Training loss curves for one seed
print(f"  Saving: torch_training_curves.png")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("PyTorch Training Curves (seed=42)", fontsize=12)
for ax, (name, model_fn) in zip(axes, [
    ("SmallMLP", SmallMLP),
    ("DeepMLP",  DeepMLP),
    ("WideMLP",  WideMLP),
]):
    t = TorchTrainer(model_fn=model_fn, lr=1e-3, n_epochs=100,
                     patience=15, random_state=42, verbose=False)
    t.fit(X_scaled, y)
    ep = range(len(t.train_losses_))
    ax.plot(ep, t.train_losses_, label="Train", linewidth=2)
    ax.plot(ep, t.val_losses_,   label="Val",   linewidth=2, linestyle="--")
    ax.axvline(np.argmin(t.val_losses_), color="green", linewidth=1.5,
               linestyle=":", label=f"Best ({np.argmin(t.val_losses_)})")
    ax.set_title(name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/torch_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

banner("FINAL SUMMARY")

print("\n  All models (sorted by mean accuracy):\n")
for name, res in sorted(all_results.items(), key=lambda x: -x[1].mean):
    ci = bootstrap_ci(res)
    bar = "█" * int(ci.mean * 40)
    tag = " ← PyTorch" if "MLP" in name else ""
    print(f"  {name:<22} {ci.mean:.4f}  [{ci.lower:.4f}, {ci.upper:.4f}]  {bar}{tag}")

print("\n  Plots saved to:", PLOT_DIR)
for f in sorted(os.listdir(PLOT_DIR)):
    if f.endswith(".png"):
        print(f"    • {f}")

print(f"\n  Cache files: {len(os.listdir(CACHE_DIR))} runs saved to {CACHE_DIR}")
print(f"  Re-running this script will skip all cached fits.\n")