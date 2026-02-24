"""
Sensitivity analysis: learning curves with uncertainty bands.

Standard learning curve plots show a single line. This module shows the
actual distribution of performance at each training set size, making it
visible when two models that look different at full data are
indistinguishable at smaller scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from mlci.core.experiment import _seed_everything, _run_single
from mlci.core.results import ExperimentResults
from mlci.stats.bootstrap import bootstrap_ci


@dataclass
class LearningCurveResults:
    """
    Results of a learning curve experiment.

    Attributes
    ----------
    train_sizes : np.ndarray
        Array of training set sizes (absolute counts).
    train_fractions : np.ndarray
        Corresponding fractions of the full training set.
    results_per_size : list[ExperimentResults]
        One ExperimentResults per training size.
    """

    train_sizes: np.ndarray
    train_fractions: np.ndarray
    results_per_size: list


def learning_curve(
    model_factory: Callable,
    X, y,
    metric: Union[str, Callable] = "accuracy",
    train_fractions: Sequence[float] = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    n_seeds: int = 10,
    n_splits: int = 5,
    n_jobs: int = 1,
    model_name: str = "model",
    higher_is_better: bool = True,
) -> LearningCurveResults:
    """
    Compute a learning curve with uncertainty bands.

    For each training fraction, runs a full Experiment (n_seeds × n_splits)
    and collects ExperimentResults. The result can be plotted with
    mlci.viz.plot_learning_curve().

    Parameters
    ----------
    model_factory : callable
        Function (seed) -> unfitted model.
    X, y : array-like
    metric : str or callable
    train_fractions : sequence of float
        Fractions of training data to use. Must be in (0, 1].
    n_seeds, n_splits, n_jobs, model_name, higher_is_better
        Passed to Experiment.

    Returns
    -------
    LearningCurveResults
    """
    from mlci.core.experiment import Experiment

    X = np.asarray(X)
    y = np.asarray(y)

    fractions = np.asarray(sorted(train_fractions))
    results_per_size = []
    sizes = []

    for frac in tqdm(fractions, desc="learning curve"):
        if frac == 1.0:
            X_sub, y_sub = X, y
        else:
            n_sub = max(int(len(y) * frac), 10)
            # Subsample deterministically from the full training set
            rng = np.random.RandomState(0)
            idx = rng.choice(len(y), size=n_sub, replace=False)
            X_sub, y_sub = X[idx], y[idx]

        exp = Experiment(
            model_factory=model_factory,
            X=X_sub, y=y_sub,
            metric=metric,
            n_seeds=n_seeds,
            n_splits=n_splits,
            n_jobs=n_jobs,
            model_name=f"{model_name} (frac={frac:.2f})",
            higher_is_better=higher_is_better,
        )
        res = exp.run(verbose=False)
        results_per_size.append(res)
        sizes.append(len(y_sub))

    return LearningCurveResults(
        train_sizes=np.array(sizes),
        train_fractions=fractions,
        results_per_size=results_per_size,
    )
