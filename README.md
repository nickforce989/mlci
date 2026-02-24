# MLci: Machine Learning confidence interval

**Statistically rigorous ML model evaluation and benchmarking.**

Every number you report from an ML experiment is a random variable. This library forces you to treat it that way.

---

## Installation

```bash
pip install mlci
```

For PyTorch integration:
```bash
pip install mlci torch
```

For Bayesian comparison with full PyMC backend:
```bash
pip install mlci[bayesian]
```

### Requirements

- Python 3.9+
- numpy >= 1.23
- scipy >= 1.9
- scikit-learn >= 1.1
- matplotlib >= 3.5
- seaborn >= 0.12
- pandas >= 1.5
- joblib >= 1.2
- tqdm >= 4.64

All dependencies are installed automatically with `pip install mlci`. To install manually:

```bash
pip install numpy scipy scikit-learn matplotlib seaborn pandas joblib tqdm
```

To run the examples, no additional dependencies are needed — they use datasets that ship with scikit-learn. The PyTorch example additionally requires `pip install torch`.

---

## The Problem

Standard ML practice:
```
RandomForest     accuracy: 0.9631
GradientBoosting accuracy: 0.9631
Difference: 0.0000 — they're equal
```

But these numbers came from a single run with a single random seed. Run it again and you get different numbers. The variance of accuracy across seeds on a typical benchmark is 0.2–0.5%. A 0.3% difference between two models is often indistinguishable from noise — but almost no paper will tell you that.

**mlci** makes it trivial to do this correctly.

---

## Quickstart

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from mlci import Experiment
from mlci.stats.bootstrap import summary
from mlci.stats.tests import compare

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

rf_results = rf.run()
gb_results = gb.run()
```

**Summary with uncertainty:**
```
────────────────────────────────────────────────────
  Model   : RandomForest
  Metric  : accuracy
  Seeds   : 20   Splits: 5
────────────────────────────────────────────────────
  Mean    : 0.9593
  95% CI  : [0.9577, 0.9608]  (±0.0015)
  Seed σ  : 0.0037  (variance from random init)
  Split σ : 0.0061  (variance from data split)
────────────────────────────────────────────────────
```

**Statistical comparison:**
```python
compare(rf_results, gb_results, method="corrected_ttest")
```
```
────────────────────────────────────────────────────────────
  Comparison : RandomForest  vs  GradientBoosting
  Method     : Corrected Resampled t-test (Nadeau & Bengio, 2003)
  Effect     : +0.0004  [-0.0023, +0.0029]
  P(A > B)   : 0.430
  p-value    : 0.6305  (α=0.05)
  Conclusion : No statistically significant difference (p=0.6305 >= α=0.05).
               Cannot conclude either model is better.
────────────────────────────────────────────────────────────
```

The standard single-run reported both models at identical accuracy. mlci correctly identifies the difference is noise.

---

## Running the Examples

Five ready-to-run example scripts are included in the `examples/` folder. All use datasets that ship with scikit-learn — no downloads needed.

---

### `worked_example.py` — start here

The simplest and most focused example. Demonstrates the single most important claim of the library: that the standard practice of reporting a single accuracy number is misleading, and that two models which look identical under standard evaluation can be properly assessed with mlci.

Runs three models first the standard way (one number), then with mlci (20 seeds × 5 folds, confidence intervals, corrected t-test). Ends with a side-by-side summary showing exactly what information was lost.

```bash
python examples/worked_example.py
# Runtime: ~1 minute
```

**Run this first** — it's the clearest demonstration of why the library exists.

---

### `full_example.py` — full feature walkthrough

Covers every major feature across nine parts: four classical models, bootstrap summaries, variance decomposition, all three comparison methods, bring-your-own scores, custom metrics, learning curves, five plots, and JSON save/load.

```bash
python examples/full_example.py
# Runtime: ~3 minutes
```

**Plots produced** (saved to `examples/plots/`):
- `score_distributions.png` — KDE of raw scores per model
- `model_comparison.png` — forest plot with CI bars
- `variance_decomposition.png` — pie chart + score heatmap
- `learning_curve.png` — RF vs LogReg with uncertainty bands
- `bootstrap_distribution.png` — bootstrap distribution of the mean

---

### `neural_network_example.py` — neural network from scratch

Implements a fully-connected MLP from scratch using NumPy only — no PyTorch, no TensorFlow. He initialisation, ReLU activations, mini-batch SGD with momentum, L2 regularisation, early stopping. Wrapped in a sklearn-compatible interface so mlci works on it identically to any other model.

```bash
python examples/neural_network_example.py
# Runtime: ~2 minutes
```

**Plots produced** (saved to `examples/nn_plots/`):
- `nn_training_curves.png` — loss curves (train vs val) for each architecture
- `nn_score_distributions.png` — KDE of raw scores for all models
- `nn_vs_classical_comparison.png` — forest plot: MLP vs classical baselines
- `nn_architecture_comparison.png` — bar chart comparing network sizes with CI
- `nn_learning_curve.png` — MLP vs RandomForest learning curves with uncertainty bands
- `nn_variance_decomposition.png` — variance sources for the MLP
- `nn_bootstrap_distribution.png` — bootstrap distribution of MLP mean accuracy
- `nn_summary_panel.png` — single-figure summary combining five panels

**Key finding:** on a small tabular dataset (569 samples), LogisticRegression outperforms the MLP (0.979 vs 0.973, p=0.0001). Neural networks need large datasets to outperform well-tuned linear models. mlci makes this conclusion rigorous rather than anecdotal.

---

### `sklearn_example.py` — sklearn integration showcase

Demonstrates `mlci.integrations.sklearn` across four parts: defining models with `PipelineFactory` instead of lambdas, wrapping external scores with `wrap_cross_val_scores`, running 8 classifiers at once with `ModelGrid`, and running the same grid across three datasets to produce a cross-dataset comparison heatmap.

```bash
python examples/sklearn_example.py
# Runtime: ~2 minutes
```

**Plots produced** (saved to `examples/sklearn_plots/`):
- `sklearn_model_comparison.png` — 8 classifiers ranked by accuracy with CI bars
- `sklearn_scaler_comparison.png` — does scaler choice matter? (yes, dramatically)
- `sklearn_score_distributions.png` — KDE for the top 3 models
- `sklearn_learning_curve.png` — LogReg vs RF vs SVM with uncertainty bands
- `sklearn_cross_dataset_heatmap.png` — mean accuracy grid: 3 models × 3 datasets
- `sklearn_variance_decomposition.png` — variance sources for the best model

**Key finding:** MinMaxScaler drops LogisticRegression accuracy from 0.977 to 0.937 — a large, statistically significant difference that would be invisible from a single run.

---

### `torch_example.py` — PyTorch integration

Plugs real PyTorch neural networks into mlci via `TorchTrainer` — a sklearn-compatible wrapper that handles seeding (torch + numpy + random + CUDA), early stopping, and **disk caching** (completed runs are saved and skipped on re-run). Compares three architectures (Small, Deep, Wide MLP) against classical baselines with full statistical rigour.

```bash
pip install torch   # required
python examples/torch_example.py
# Runtime: ~5 minutes first run; near-instant on re-run due to caching
```

**Plots produced** (saved to `examples/torch_plots/`):
- `torch_model_comparison.png` — PyTorch MLPs vs classical baselines
- `torch_score_distribution.png` — score distribution for the best architecture
- `torch_learning_curve.png` — SmallMLP vs RandomForest with uncertainty bands
- `torch_training_curves.png` — loss curves for all three architectures

**Key feature — caching:** each (seed, split) training run is hashed by architecture + config + data and saved to disk. Kill the process halfway through 50 runs, rerun the script, and it continues from where it left off. Essential when training takes minutes per run.

---

## Core Features

### 1. Bootstrap Confidence Intervals

```python
from mlci.stats.bootstrap import bootstrap_ci, summary

ci = bootstrap_ci(results, confidence=0.95)
print(ci)
# 0.9593 [95% CI: 0.9577, 0.9608] (width=0.0031)
```

Resamples at the seed level — the correct choice, since seeds are independent but folds within a seed share training data and are not.

---

### 2. Three Statistical Comparison Methods

```python
from mlci.stats.tests import compare

# Method 1 (default): Corrected Resampled t-test (Nadeau & Bengio, 2003)
compare(a, b, method="corrected_ttest")

# Method 2: Wilcoxon signed-rank test (non-parametric, no distributional assumptions)
compare(a, b, method="wilcoxon")

# Method 3: Bayesian — reports P(A > B) rather than a binary p-value
compare(a, b, method="bayesian")

# Run all three at once
compare(a, b, method="all")
```

#### Why the Nadeau-Bengio corrected t-test matters

The standard paired t-test applied to k-fold CV scores is anti-conservative: it systematically overstates significance because the folds share training data, making the observations non-independent. Nadeau & Bengio (2003) derived a corrected variance estimator that accounts for this. It is well-known in the statistical ML literature but almost never used in practice because no mainstream library implements it cleanly.

**Reference:** Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239-281.

---

### 3. Variance Decomposition

Where does your score variance come from — random initialisation, or the data split?

```python
from mlci.stats.anova import decompose_variance

print(decompose_variance(results))
```
```
────────────────────────────────────────────────────────
  Variance Decomposition: RandomForest (accuracy)
────────────────────────────────────────────────────────
  Total variance  : 0.000435
  Seed variance   : 0.000010  (2.3%)   <- random init
  Split variance  : 0.000016  (3.6%)   <- which data in test set
  Interaction     : 0.000409  (94.1%)  <- seed×split interaction
────────────────────────────────────────────────────────
```

Uses a two-way balanced ANOVA decomposition. High split variance means your evaluation is sensitive to which portion of your data ends up in the test set — a signal that you may need more folds or a larger dataset.

---

### 4. Sensitivity Analysis

#### Learning curves with uncertainty bands

```python
from mlci.sensitivity.learning_curve import learning_curve
from mlci.viz.plots import plot_learning_curve

lc = learning_curve(
    model_factory=lambda seed: RandomForestClassifier(random_state=seed),
    X=X, y=y,
    train_fractions=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    n_seeds=10, n_splits=5,
)
fig = plot_learning_curve(lc)
```

Each point on the curve has a confidence band rather than being a single line — so you can see when two models that look different are actually statistically indistinguishable at smaller data scales.

#### Hyperparameter sensitivity

How much does each hyperparameter actually move the needle?

```python
from mlci.sensitivity.hyperparameter import hyperparameter_sensitivity

results = hyperparameter_sensitivity(
    model_factory=lambda seed, n_estimators, max_depth:
        RandomForestClassifier(n_estimators=n_estimators,
                               max_depth=max_depth,
                               random_state=seed),
    param_grid={
        "n_estimators": [10, 50, 100, 200],
        "max_depth":    [None, 5, 10],
    },
    X=X, y=y, n_seeds=10, n_splits=5,
)
results.summary_table()
# Most sensitive parameter: n_estimators  σ = 0.0089
# Second most sensitive:    max_depth     σ = 0.0031
```

Each combination in the grid is evaluated as a full mlci experiment (n_seeds × n_splits), giving you confidence intervals on every point and a sensitivity ranking showing which parameters to prioritise when tuning.

#### Dataset sensitivity

Which samples in your dataset are genuinely hard to classify?

```python
from mlci.sensitivity.dataset import dataset_sensitivity

results = dataset_sensitivity(
    model_factory=lambda seed: RandomForestClassifier(random_state=seed),
    X=X, y=y, n_seeds=10, n_splits=5,
)
print(results)
# Easy  (rate=0.0): 412 samples  — always correct
# Medium (0<r≤0.5): 134 samples  — sometimes wrong
# Hard  (rate>0.5):  23 samples  — usually wrong

hard_X = X[results.hard_indices]   # the genuinely ambiguous samples
```

Tracks per-sample misclassification rates across all (seed, split) pairs. Samples that are frequently wrong regardless of seed are genuinely ambiguous — useful for dataset curation and identifying potential labelling errors before publishing a benchmark.

---

### 5. sklearn Integrations

#### PipelineFactory

Build model + scaler pipelines from a config dict, with automatic seed injection:

```python
from mlci.integrations.sklearn import PipelineFactory

factory = PipelineFactory(
    LogisticRegression,
    params={"max_iter": 2000, "C": 1.0},
    scaler="standard",   # or "robust", "minmax", "none"
)

exp = Experiment(model_factory=factory, X=X, y=y, ...)
```

#### wrap_cross_val_scores

Already have scores? Bring them in without rerunning:

```python
from mlci.integrations.sklearn import wrap_cross_val_scores
import numpy as np

scores = np.vstack([
    cross_val_score(RandomForestClassifier(random_state=s), X, y, cv=5)
    for s in range(10)
])

results = wrap_cross_val_scores(scores, metric="accuracy", model_name="RF")
summary(results)   # full CI, variance decomposition, comparisons
```

Works with any source: sklearn, PyTorch training loops, a collaborator's array of numbers — anything that produces scores.

#### ModelGrid

Run many models in one call:

```python
from mlci.integrations.sklearn import ModelGrid

grid = ModelGrid(
    models={
        "LogisticRegression": PipelineFactory(LogisticRegression, scaler="standard"),
        "RandomForest":       PipelineFactory(RandomForestClassifier, scaler="none"),
        "SVM":                PipelineFactory(SVC, scaler="standard"),
    },
    X=X, y=y, metric="accuracy", n_seeds=15, n_splits=5,
)
all_results = grid.run()
# {model_name: ExperimentResults} — pass directly to plot_comparison(), compare(), etc.
```

---

### 6. PyTorch Integration

Wrap any `nn.Module` in a sklearn-compatible interface with caching:

```python
import torch.nn as nn
from mlci.integrations.torch import TorchTrainer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x)

exp = Experiment(
    model_factory=lambda seed: Pipeline([
        ("scaler", StandardScaler()),
        ("net", TorchTrainer(
            model_fn=MyNet,
            lr=1e-3,
            n_epochs=100,
            patience=15,
            random_state=seed,
            cache_dir="./mlci_cache",   # resume interrupted runs
        )),
    ]),
    X=X, y=y, metric="accuracy", n_seeds=10, n_splits=5,
    model_name="MyNet",
)
results = exp.run()
summary(results)
```

`TorchTrainer` handles seeding (torch + numpy + random + CUDA), early stopping with validation-based checkpoint restore, and disk caching keyed by architecture + config + data fingerprint. Different architectures always get different cache entries.

---

### 7. Multi-Dataset Benchmarking

Run a model grid across multiple datasets with Benjamini-Hochberg multiple testing correction:

```python
from mlci.benchmarks.runner import run_benchmark, print_benchmark_report
from mlci.benchmarks.datasets import load_all_classification

results = run_benchmark(
    models={
        "RF":  lambda seed: RandomForestClassifier(random_state=seed),
        "LR":  PipelineFactory(LogisticRegression, scaler="standard"),
        "SVM": PipelineFactory(SVC, scaler="standard"),
    },
    datasets=load_all_classification(),
    n_seeds=10, n_splits=5,
)
print_benchmark_report(results)
```

Without multiple testing correction, comparing 3 models on 4 datasets means 12 pairwise tests — at α=0.05 you would expect spurious significant results by chance. Benjamini-Hochberg controls the false discovery rate across all comparisons and is built in automatically.

---

### 8. Visualisation

```python
from mlci.viz.plots import (
    plot_score_distribution,      # KDE + CI on the mean
    plot_comparison,              # Forest plot for multiple models
    plot_variance_decomposition,  # Pie chart + score heatmap
    plot_bootstrap_distribution,  # Bootstrap distribution of the mean
    plot_learning_curve,          # Learning curve with uncertainty bands
)
```

---

## Supported Metrics

Built-in: `"accuracy"`, `"f1"`, `"f1_binary"`, `"roc_auc"`, `"rmse"`, `"mae"`, `"r2"`, `"mse"`

Or pass any callable `(y_true, y_pred) -> float`:

```python
Experiment(
    model_factory=...,
    metric=lambda yt, yp: my_custom_metric(yt, yp),
)
```

---

## Saving and Loading Results

```python
results.save("rf_results.json")
loaded = ExperimentResults.load("rf_results.json")
```

Results are stored as plain JSON — no pickle, no version lock-in. Share results with collaborators and they can run the full statistical analysis without retraining anything.

---

## Design Principles

**Seeds are the unit of resampling.** When we bootstrap, we resample at the seed level, not the individual (seed, split) level. Seeds are independent; folds within a seed share training data and are not independent.

**Report distributions, not point estimates.** The default output of every function is either a distribution or a statistic with a confidence interval.

**Correct by default.** The default comparison method is the Nadeau-Bengio corrected t-test, not the standard paired t-test, because the standard test is wrong for this use case.

**No hidden state.** `ExperimentResults` is a plain data container. You can inspect, serialise, and reconstruct it without going through any library-specific machinery.

**Framework-agnostic.** The library works with any model that implements `fit(X, y)` and `predict(X)`. sklearn, PyTorch, XGBoost, your own custom model — all treated identically.

---

## Author
Niccolò Forzano --- [github.com/nickforce989](https://github.com/nickforce989) · [nic.forz@gmail.com](mailto:nic.forz@gmail.com)

---

## Citation

If you use mlci in published work, please cite it:

```bibtex
@software{forzano2026mlci,
  author    = {Forzano, Niccolo},
  title     = {mlci: Statistically Rigorous ML Model Evaluation and Benchmarking},
  year      = {2026},
  url       = {https://github.com/nickforce989/mlci},
  note      = {Python package}
}
```

Or in plain text:

> Forzano, N. (2026). *mlci: Statistically Rigorous ML Model Evaluation and Benchmarking*. https://github.com/nickforce989/mlci

Please also cite the underlying statistical method the library is built on:

```bibtex
@article{nadeau2003inference,
  author  = {Nadeau, Claude and Bengio, Yoshua},
  title   = {Inference for the Generalization Error},
  journal = {Machine Learning},
  year    = {2003},
  volume  = {52},
  number  = {3},
  pages   = {239--281}
}
```

---

## Roadmap

- [x] Bootstrap confidence intervals
- [x] Corrected resampled t-test (Nadeau & Bengio, 2003)
- [x] Wilcoxon and Bayesian comparison methods
- [x] Variance decomposition (two-way ANOVA)
- [x] Learning curves with uncertainty bands
- [x] Hyperparameter sensitivity analysis
- [x] Dataset sensitivity (per-sample misclassification rates)
- [x] sklearn integrations (PipelineFactory, ModelGrid, wrap_cross_val_scores)
- [x] PyTorch integration with caching and early stopping
- [x] Multi-dataset benchmarking with Benjamini-Hochberg correction
- [ ] Full PyTorch/HuggingFace training loop integration with Slurm support
- [ ] PyMC-based full Bayesian hierarchical comparison
- [ ] HTML report generation
- [ ] Integration with MLflow / Weights & Biases
- [ ] Support for multi-class and multi-label metrics

---

## License

MIT