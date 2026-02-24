"""
mlci neural network demo — from scratch MLP vs classical models.

Implements a fully connected neural network (MLP) from scratch using
only NumPy — no PyTorch, no TensorFlow. The network has:
  - Configurable depth and width
  - ReLU activations (hidden layers) + Sigmoid output
  - Mini-batch SGD with momentum
  - He weight initialisation
  - L2 regularisation
  - Early stopping

All wrapped in a sklearn-compatible interface so mlci works on it
identically to RandomForest or any other model.

Produces:
  Console: summaries, comparisons, variance decompositions,
           per-epoch training curves, architecture comparison table
  Plots:   nn_vs_classical_comparison.png
           nn_training_curves.png
           nn_score_distributions.png
           nn_architecture_comparison.png
           nn_learning_curve.png
           nn_variance_decomposition.png
           nn_bootstrap_distribution.png

Run with:
    python examples/nn_demo.py
"""

import warnings
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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

PLOT_DIR = "examples/nn_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

_PALETTE = ["#2E4057", "#E84855", "#3BB273", "#F4A261", "#8338EC", "#023E8A"]


# ═══════════════════════════════════════════════════════════════
# NEURAL NETWORK — FROM SCRATCH
# ═══════════════════════════════════════════════════════════════

class MLP(BaseEstimator, ClassifierMixin):
    """
    Multi-Layer Perceptron for binary classification.
    Implemented from scratch using NumPy only.

    Architecture: Input → [Dense → ReLU] × n_hidden_layers → Dense → Sigmoid
    Training:     Mini-batch SGD with momentum + L2 regularisation
    Init:         He initialisation (correct for ReLU)

    Parameters
    ----------
    hidden_layers : tuple of int
        Number of neurons per hidden layer.
        (64,) = one hidden layer with 64 units
        (128, 64) = two hidden layers
    learning_rate : float
    momentum : float
        SGD momentum coefficient.
    l2 : float
        L2 regularisation strength.
    batch_size : int
    n_epochs : int
    patience : int
        Early stopping — stop if val loss doesn't improve for this many epochs.
    random_state : int
    verbose : bool
    """

    def __init__(
        self,
        hidden_layers=(64, 32),
        learning_rate=0.01,
        momentum=0.9,
        l2=1e-4,
        batch_size=32,
        n_epochs=200,
        patience=20,
        random_state=0,
        verbose=False,
    ):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2 = l2
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Activations
    # ------------------------------------------------------------------

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _relu_deriv(x):
        return (x > 0).astype(float)

    @staticmethod
    def _sigmoid(x):
        # Numerically stable sigmoid
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    @staticmethod
    def _bce_loss(y_true, y_pred):
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, layer_sizes):
        """He initialisation — optimal for ReLU activations."""
        rng = np.random.RandomState(self.random_state)
        self.weights_ = []
        self.biases_ = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            std = np.sqrt(2.0 / fan_in)  # He init
            W = rng.randn(layer_sizes[i], layer_sizes[i + 1]) * std
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights_.append(W)
            self.biases_.append(b)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, X):
        """
        Returns:
            activations : list of layer outputs (including input)
            pre_activations : list of pre-activation values (z = Wx + b)
        """
        activations = [X]
        pre_activations = []

        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            z = activations[-1] @ W + b
            pre_activations.append(z)
            if i < len(self.weights_) - 1:
                a = self._relu(z)
            else:
                a = self._sigmoid(z)  # output layer
            activations.append(a)

        return activations, pre_activations

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def _backward(self, activations, pre_activations, y):
        """
        Backpropagation. Returns gradients for all weights and biases.
        """
        n = len(y)
        grads_W = [None] * len(self.weights_)
        grads_b = [None] * len(self.biases_)

        # Output layer gradient: d(BCE)/d(sigmoid) simplifies cleanly
        delta = activations[-1] - y.reshape(-1, 1)   # (n, 1)

        for i in reversed(range(len(self.weights_))):
            grads_W[i] = (activations[i].T @ delta) / n
            grads_b[i] = delta.mean(axis=0, keepdims=True)

            # L2 regularisation gradient
            grads_W[i] += self.l2 * self.weights_[i]

            if i > 0:
                delta = (delta @ self.weights_[i].T) * self._relu_deriv(pre_activations[i - 1])

        return grads_W, grads_b

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_features = X.shape[1]
        layer_sizes = [n_features] + list(self.hidden_layers) + [1]
        self._init_weights(layer_sizes)

        # Momentum buffers
        vel_W = [np.zeros_like(W) for W in self.weights_]
        vel_b = [np.zeros_like(b) for b in self.biases_]

        rng = np.random.RandomState(self.random_state)

        # Validation split for early stopping (10%)
        n = len(y)
        val_size = max(1, int(0.1 * n))
        idx = rng.permutation(n)
        val_idx, train_idx = idx[:val_size], idx[val_size:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]

        best_val_loss = np.inf
        best_weights = None
        best_biases = None
        no_improve = 0

        self.train_losses_ = []
        self.val_losses_ = []

        for epoch in range(self.n_epochs):
            # Shuffle training data
            perm = rng.permutation(len(X_train))
            X_shuf, y_shuf = X_train[perm], y_train[perm]

            # Mini-batch updates
            for start in range(0, len(X_shuf), self.batch_size):
                Xb = X_shuf[start:start + self.batch_size]
                yb = y_shuf[start:start + self.batch_size]

                acts, pre_acts = self._forward(Xb)
                grads_W, grads_b = self._backward(acts, pre_acts, yb)

                for i in range(len(self.weights_)):
                    vel_W[i] = self.momentum * vel_W[i] - self.learning_rate * grads_W[i]
                    vel_b[i] = self.momentum * vel_b[i] - self.learning_rate * grads_b[i]
                    self.weights_[i] += vel_W[i]
                    self.biases_[i]  += vel_b[i]

            # Track losses
            _, _ = self._forward(X_train)
            train_pred = self._forward(X_train)[0][-1].ravel()
            val_pred   = self._forward(X_val)[0][-1].ravel()
            train_loss = self._bce_loss(y_train, train_pred)
            val_loss   = self._bce_loss(y_val,   val_pred)

            self.train_losses_.append(train_loss)
            self.val_losses_.append(val_loss)

            if self.verbose and epoch % 20 == 0:
                val_acc = np.mean((val_pred > 0.5) == y_val)
                print(f"    epoch {epoch:4d}  train_loss={train_loss:.4f}  "
                      f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

            # Early stopping
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_weights = [W.copy() for W in self.weights_]
                best_biases  = [b.copy() for b in self.biases_]
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    if self.verbose:
                        print(f"    Early stopping at epoch {epoch}")
                    break

        # Restore best weights
        self.weights_ = best_weights
        self.biases_  = best_biases
        self.classes_ = np.array([0, 1])
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        acts, _ = self._forward(X)
        p = acts[-1].ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ═══════════════════════════════════════════════════════════════
# SKLEARN PIPELINE FACTORY (handles scaling internally)
# ═══════════════════════════════════════════════════════════════

def make_mlp(hidden_layers=(64, 32), lr=0.01, seed=0):
    """Wrap MLP in a pipeline that scales features first."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    MLP(
            hidden_layers=hidden_layers,
            learning_rate=lr,
            random_state=seed,
            verbose=False,
        ))
    ])


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def banner(title):
    print("\n" + "═" * 64)
    print(f"  {title}")
    print("═" * 64)

def section(title):
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}")


# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════

banner("mlci — Neural Network Demo")

X, y = load_breast_cancer(return_X_y=True)
print(f"\n  Dataset  : Breast Cancer Wisconsin")
print(f"  Samples  : {len(y)}   Features: {X.shape[1]}   Classes: 2")
print(f"  Positive : {y.sum()} ({y.mean()*100:.1f}%)   Negative: {(1-y).sum()}")
print(f"\n  Neural network: MLP from scratch (NumPy only)")
print(f"  Architecture  : Input(30) → ReLU layers → Sigmoid output")
print(f"  Training      : Mini-batch SGD + momentum + L2 + early stopping")
print(f"  Scaling       : StandardScaler (required for neural nets)")


# ═══════════════════════════════════════════════════════════════
# PART 1 — Single verbose training run to show what's happening
# ═══════════════════════════════════════════════════════════════

banner("PART 1 — Single Training Run (verbose, to show learning)")

section("Architecture: 30 → 64 → 32 → 1")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

net = MLP(hidden_layers=(64, 32), learning_rate=0.01, n_epochs=300,
          patience=30, random_state=42, verbose=True)
net.fit(X_scaled, y)
train_acc = net.score(X_scaled, y)
print(f"\n  Final training accuracy : {train_acc:.4f}")
print(f"  Total epochs run        : {len(net.train_losses_)}")
print(f"  Best validation loss    : {min(net.val_losses_):.4f}")
print(f"  (Early stopping kicks in when val loss stops improving)")


# ═══════════════════════════════════════════════════════════════
# PART 2 — Architecture comparison
# ═══════════════════════════════════════════════════════════════

banner("PART 2 — Comparing Neural Network Architectures (10 seeds × 5 folds)")

architectures = {
    "MLP-Tiny   [30→16→1]":        lambda seed: make_mlp((16,),       lr=0.01, seed=seed),
    "MLP-Small  [30→64→32→1]":     lambda seed: make_mlp((64, 32),    lr=0.01, seed=seed),
    "MLP-Medium [30→128→64→32→1]": lambda seed: make_mlp((128,64,32), lr=0.01, seed=seed),
    "MLP-Wide   [30→256→1]":       lambda seed: make_mlp((256,),      lr=0.005, seed=seed),
}

arch_results = {}
for name, factory in architectures.items():
    print(f"\n  → {name}")
    exp = Experiment(
        model_factory=factory,
        X=X, y=y,
        metric="accuracy",
        n_seeds=10,
        n_splits=5,
        model_name=name.split("[")[0].strip(),
    )
    arch_results[name] = exp.run(verbose=True)

section("Architecture Comparison Table")
print(f"\n  {'Architecture':<38} {'Mean':>7} {'CI Lower':>9} {'CI Upper':>9} {'Seed σ':>8} {'Split σ':>8}")
print(f"  {'─'*38} {'─'*7} {'─'*9} {'─'*9} {'─'*8} {'─'*8}")
for name, res in sorted(arch_results.items(), key=lambda x: -x[1].mean):
    ci = bootstrap_ci(res)
    d  = decompose_variance(res)
    print(
        f"  {name:<38} {ci.mean:>7.4f} {ci.lower:>9.4f} {ci.upper:>9.4f} "
        f"{res.seed_means.std():>8.4f} {res.split_means.std():>8.4f}"
    )


# ═══════════════════════════════════════════════════════════════
# PART 3 — NN vs classical models
# ═══════════════════════════════════════════════════════════════

banner("PART 3 — Neural Network vs Classical Models (20 seeds × 5 folds)")

best_arch = (64, 32)

classical_models = {
    "MLP [64→32]":       lambda seed: make_mlp(best_arch, seed=seed),
    "RandomForest":      lambda seed: RandomForestClassifier(n_estimators=50, random_state=seed),
    "GradientBoosting":  lambda seed: GradientBoostingClassifier(n_estimators=50, random_state=seed),
    "LogisticRegression":lambda seed: Pipeline([
                             ("scaler", StandardScaler()),
                             ("lr", LogisticRegression(random_state=seed, max_iter=2000, C=1.0))
                         ]),
}

all_results = {}
for name, factory in classical_models.items():
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

mlp_res = all_results["MLP [64→32]"]
rf_res  = all_results["RandomForest"]
gb_res  = all_results["GradientBoosting"]
lr_res  = all_results["LogisticRegression"]


# ═══════════════════════════════════════════════════════════════
# PART 4 — Bootstrap summaries
# ═══════════════════════════════════════════════════════════════

banner("PART 4 — Bootstrap Summaries with Confidence Intervals")

for name, res in all_results.items():
    print()
    summary(res, confidence=0.95)


# ═══════════════════════════════════════════════════════════════
# PART 5 — Statistical comparisons
# ═══════════════════════════════════════════════════════════════

banner("PART 5 — Statistical Comparisons")

section("MLP vs RandomForest — all three methods")
compare(mlp_res, rf_res, method="all")

section("MLP vs GradientBoosting")
compare(mlp_res, gb_res, method="corrected_ttest")

section("MLP vs LogisticRegression")
compare(mlp_res, lr_res, method="corrected_ttest")

section("Bayesian: MLP vs RandomForest")
compare(mlp_res, rf_res, method="bayesian")


# ═══════════════════════════════════════════════════════════════
# PART 6 — Variance decomposition
# ═══════════════════════════════════════════════════════════════

banner("PART 6 — Variance Decomposition")

print("\n  Key question: is MLP variance dominated by random init (seed)")
print("  or by which data ended up in the test set (split)?\n")

for name, res in all_results.items():
    print()
    print(decompose_variance(res))


# ═══════════════════════════════════════════════════════════════
# PART 7 — Learning curves
# ═══════════════════════════════════════════════════════════════

banner("PART 7 — Learning Curves with Uncertainty Bands")

print("\n  How much data does each model need to reach peak performance?")
print("  Computing curves for MLP and RandomForest...\n")

lc_mlp = learning_curve(
    model_factory=lambda seed: make_mlp((64, 32), seed=seed),
    X=X, y=y,
    train_fractions=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    n_seeds=8, n_splits=5,
    model_name="MLP",
)
lc_rf = learning_curve(
    model_factory=lambda seed: RandomForestClassifier(n_estimators=50, random_state=seed),
    X=X, y=y,
    train_fractions=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    n_seeds=8, n_splits=5,
    model_name="RandomForest",
)

print(f"\n  {'Fraction':>10}  {'MLP mean':>10}  {'MLP CI':>20}  {'RF mean':>10}  {'RF CI':>20}")
print(f"  {'─'*10}  {'─'*10}  {'─'*20}  {'─'*10}  {'─'*20}")
for frac, res_mlp, res_rf in zip(
    lc_mlp.train_fractions,
    lc_mlp.results_per_size,
    lc_rf.results_per_size,
):
    ci_mlp = bootstrap_ci(res_mlp)
    ci_rf  = bootstrap_ci(res_rf)
    print(
        f"  {frac*100:>9.0f}%  {ci_mlp.mean:>10.4f}  "
        f"[{ci_mlp.lower:.4f}, {ci_mlp.upper:.4f}]  "
        f"{ci_rf.mean:>10.4f}  "
        f"[{ci_rf.lower:.4f}, {ci_rf.upper:.4f}]"
    )


# ═══════════════════════════════════════════════════════════════
# PART 8 — Plots
# ═══════════════════════════════════════════════════════════════

banner("PART 8 — Generating Plots")


# -- 8a. Training curves (loss over epochs for one run) --
print(f"\n  Saving: nn_training_curves.png")

# Retrain with different architectures, capture loss curves
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Neural Network Training Curves\n(train loss vs val loss per architecture)", fontsize=13)

arch_configs = [
    ("Tiny  [30→16→1]",        (16,),       0.01),
    ("Small [30→64→32→1]",     (64, 32),    0.01),
    ("Medium [30→128→64→32→1]",(128,64,32), 0.01),
    ("Wide  [30→256→1]",       (256,),      0.005),
]

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

for ax, (name, layers, lr) in zip(axes.flat, arch_configs):
    net = MLP(hidden_layers=layers, learning_rate=lr, n_epochs=300,
              patience=30, random_state=42)
    net.fit(X_sc, y)
    epochs = range(len(net.train_losses_))
    ax.plot(epochs, net.train_losses_, color=_PALETTE[0], linewidth=2, label="Train loss")
    ax.plot(epochs, net.val_losses_,   color=_PALETTE[1], linewidth=2,
            linestyle="--", label="Val loss")
    ax.axvline(np.argmin(net.val_losses_), color=_PALETTE[2], linewidth=1.5,
               linestyle=":", alpha=0.8, label=f"Best epoch ({np.argmin(net.val_losses_)})")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    final_acc = net.score(X_sc, y)
    ax.text(0.98, 0.95, f"Train acc: {final_acc:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=_PALETTE[0])

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/nn_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()


# -- 8b. Score distributions for all models --
print(f"  Saving: nn_score_distributions.png")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Score Distributions (20 seeds × 5 folds)\nKDE of raw accuracy scores", fontsize=13)

for ax, (name, res), color in zip(axes.flat, all_results.items(), _PALETTE):
    scores = res.flat
    ci = bootstrap_ci(res)
    sns.kdeplot(scores, ax=ax, color=color, fill=True, alpha=0.3, linewidth=2)
    ax.axvline(ci.mean, color=color, linewidth=2, linestyle="--")
    ax.axvspan(ci.lower, ci.upper, alpha=0.15, color=color)
    ax.plot(scores, np.full_like(scores, ax.get_ylim()[0]), "|",
            color=color, alpha=0.4, markersize=5)
    ax.set_title(f"{name}\nmean={ci.mean:.4f}  95% CI [{ci.lower:.4f}, {ci.upper:.4f}]",
                 fontsize=10)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Density")
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/nn_score_distributions.png", dpi=150, bbox_inches="tight")
plt.close()


# -- 8c. Forest plot: NN vs classical --
print(f"  Saving: nn_vs_classical_comparison.png")
fig = plot_comparison(list(all_results.values()), confidence=0.95,
                      title="MLP vs Classical Models — Breast Cancer (accuracy)")
plt.savefig(f"{PLOT_DIR}/nn_vs_classical_comparison.png", dpi=150, bbox_inches="tight")
plt.close()


# -- 8d. Architecture comparison bar chart with CI --
print(f"  Saving: nn_architecture_comparison.png")

fig, ax = plt.subplots(figsize=(11, 5))
names, means, lowers, uppers = [], [], [], []
for name, res in sorted(arch_results.items(), key=lambda x: x[1].mean):
    ci = bootstrap_ci(res)
    short = name.split("[")[0].strip()
    names.append(short)
    means.append(ci.mean)
    lowers.append(ci.mean - ci.lower)
    uppers.append(ci.upper - ci.mean)

y_pos = np.arange(len(names))
bars = ax.barh(y_pos, means, xerr=[lowers, uppers],
               color=_PALETTE[:len(names)], alpha=0.8,
               error_kw={"elinewidth": 2, "capsize": 5, "capthick": 2, "ecolor": "black"})

for bar, mean in zip(bars, means):
    ax.text(mean + 0.001, bar.get_y() + bar.get_height()/2,
            f"{mean:.4f}", va="center", fontsize=10)

ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.set_xlabel("Mean Accuracy (95% CI)")
ax.set_title("Neural Network Architecture Comparison\n(10 seeds × 5 folds, error bars = 95% bootstrap CI)")
ax.set_xlim(0.88, 0.985)
ax.spines[["top", "right"]].set_visible(False)
ax.axvline(max(means), color="gray", linewidth=1, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/nn_architecture_comparison.png", dpi=150, bbox_inches="tight")
plt.close()


# -- 8e. Learning curves (MLP vs RF) --
print(f"  Saving: nn_learning_curve.png")
fig = plot_learning_curve([lc_mlp, lc_rf],
                          title="Learning Curve: MLP vs RandomForest\n(bands = 95% bootstrap CI)")
plt.savefig(f"{PLOT_DIR}/nn_learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()


# -- 8f. Variance decomposition for MLP --
print(f"  Saving: nn_variance_decomposition.png")
fig = plot_variance_decomposition(mlp_res)
plt.savefig(f"{PLOT_DIR}/nn_variance_decomposition.png", dpi=150, bbox_inches="tight")
plt.close()


# -- 8g. Bootstrap distribution for MLP --
print(f"  Saving: nn_bootstrap_distribution.png")
fig = plot_bootstrap_distribution(mlp_res, confidence=0.95, n_bootstrap=20_000)
plt.savefig(f"{PLOT_DIR}/nn_bootstrap_distribution.png", dpi=150, bbox_inches="tight")
plt.close()


# -- 8h. Big summary panel --
print(f"  Saving: nn_summary_panel.png")

fig = plt.figure(figsize=(16, 12))
fig.suptitle("mlci — Neural Network Evaluation Summary", fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1: Forest plot of all models
ax1 = fig.add_subplot(gs[0, :2])
sorted_results = sorted(all_results.items(), key=lambda x: x[1].mean, reverse=True)
for i, (name, res) in enumerate(sorted_results):
    ci = bootstrap_ci(res)
    color = _PALETTE[i]
    ax1.plot(ci.mean, i, "o", color=color, markersize=10, zorder=3)
    ax1.hlines(i, ci.lower, ci.upper, color=color, linewidth=4, alpha=0.8)
    ax1.text(ci.upper + 0.0005, i, f"{ci.mean:.4f}", va="center", fontsize=9, color=color)
ax1.set_yticks(range(len(sorted_results)))
ax1.set_yticklabels([n for n, _ in sorted_results], fontsize=9)
ax1.set_xlabel("Accuracy")
ax1.set_title("Model Comparison (95% CI)")
ax1.invert_yaxis()
ax1.spines[["top", "right"]].set_visible(False)

# Panel 2: MLP score distribution
ax2 = fig.add_subplot(gs[0, 2])
ci_mlp = bootstrap_ci(mlp_res)
sns.kdeplot(mlp_res.flat, ax=ax2, color=_PALETTE[0], fill=True, alpha=0.3)
ax2.axvline(ci_mlp.mean, color=_PALETTE[0], linewidth=2, linestyle="--")
ax2.axvspan(ci_mlp.lower, ci_mlp.upper, alpha=0.2, color=_PALETTE[0])
ax2.set_title("MLP Score Distribution")
ax2.set_xlabel("Accuracy")
ax2.spines[["top", "right"]].set_visible(False)

# Panel 3: Training loss curve (best arch)
ax3 = fig.add_subplot(gs[1, :2])
net_demo = MLP(hidden_layers=(64,32), learning_rate=0.01, n_epochs=300,
               patience=30, random_state=0)
net_demo.fit(X_sc, y)
ep = range(len(net_demo.train_losses_))
ax3.plot(ep, net_demo.train_losses_, color=_PALETTE[0], linewidth=2, label="Train")
ax3.plot(ep, net_demo.val_losses_,   color=_PALETTE[1], linewidth=2,
         linestyle="--", label="Validation")
ax3.axvline(np.argmin(net_demo.val_losses_), color=_PALETTE[2],
            linewidth=1.5, linestyle=":", label=f"Best epoch")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("BCE Loss")
ax3.set_title("MLP [64→32] Training Curve")
ax3.legend(frameon=False, fontsize=9)
ax3.spines[["top", "right"]].set_visible(False)

# Panel 4: Variance decomposition pie
ax4 = fig.add_subplot(gs[1, 2])
d = decompose_variance(mlp_res)
sizes = [d.seed_fraction, d.split_fraction, d.interaction_fraction]
labels = ["Seed", "Split", "Interaction"]
non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, _PALETTE) if s > 1e-4]
if non_zero:
    s, l, c = zip(*non_zero)
    ax4.pie(s, labels=l, colors=c, autopct="%1.1f%%", startangle=90)
ax4.set_title("MLP Variance Sources")

# Panel 5: Learning curve
ax5 = fig.add_subplot(gs[2, :])
for lc, color, name in [(lc_mlp, _PALETTE[0], "MLP"), (lc_rf, _PALETTE[1], "RandomForest")]:
    means_lc, lows, highs, sizes_lc = [], [], [], []
    for res in lc.results_per_size:
        ci = bootstrap_ci(res)
        means_lc.append(ci.mean)
        lows.append(ci.lower)
        highs.append(ci.upper)
        sizes_lc.append(len(res.flat))
    xs = lc.train_sizes
    ax5.plot(xs, means_lc, "o-", color=color, linewidth=2.5, label=name, markersize=7)
    ax5.fill_between(xs, lows, highs, color=color, alpha=0.18)
ax5.set_xlabel("Training set size")
ax5.set_ylabel("Accuracy")
ax5.set_title("Learning Curve: MLP vs RandomForest (bands = 95% CI)")
ax5.legend(frameon=False)
ax5.spines[["top", "right"]].set_visible(False)

plt.savefig(f"{PLOT_DIR}/nn_summary_panel.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saving: nn_summary_panel.png")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

banner("FINAL SUMMARY")

print("\n  ── Architecture comparison ──\n")
for name, res in sorted(arch_results.items(), key=lambda x: -x[1].mean):
    ci = bootstrap_ci(res)
    bar = "█" * int(ci.mean * 40)
    short = name.split("[")[0].strip()
    print(f"  {short:<26} {ci.mean:.4f}  [{ci.lower:.4f}, {ci.upper:.4f}]  {bar}")

print("\n  ── MLP vs classical models ──\n")
for name, res in sorted(all_results.items(), key=lambda x: -x[1].mean):
    ci = bootstrap_ci(res)
    bar = "█" * int(ci.mean * 40)
    print(f"  {name:<26} {ci.mean:.4f}  [{ci.lower:.4f}, {ci.upper:.4f}]  {bar}")

print("\n  ── Statistical significance (corrected t-test) ──\n")
pairs = [
    ("MLP [64→32]", "RandomForest"),
    ("MLP [64→32]", "GradientBoosting"),
    ("MLP [64→32]", "LogisticRegression"),
]
for a_name, b_name in pairs:
    r = corrected_resampled_ttest(all_results[a_name], all_results[b_name])
    sig = "✓ significant" if r.p_value < 0.05 else "✗ not significant"
    direction = "MLP better" if r.effect_size > 0 else "Baseline better"
    print(f"  MLP vs {b_name:<22} effect={r.effect_size:+.4f}  "
          f"p={r.p_value:.4f}  {sig}  ({direction})")

print(f"\n  Plots saved to: {PLOT_DIR}/")
for fname in sorted(os.listdir(PLOT_DIR)):
    if fname.endswith(".png"):
        print(f"    • {fname}")
print()
