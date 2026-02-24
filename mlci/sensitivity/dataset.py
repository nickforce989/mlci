"""
mlci.sensitivity.dataset
========================

Dataset sensitivity analysis.

Answers: which samples in your dataset are genuinely hard to classify?
Which drive most of the variance across different seeds and splits?

Method: run repeated evaluation and track, for each test sample,
how often it gets misclassified across all (seed, split) pairs where
it appeared in the test set. Samples with a high misclassification
rate are the hard cases — the model is genuinely uncertain about them
regardless of how it was trained.

This goes beyond just reporting accuracy: it shows *where* the model
fails, not just *how often*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm

from mlci.core.experiment import _seed_everything


@dataclass
class DatasetSensitivityResults:
    """
    Output of a dataset sensitivity analysis.

    Attributes
    ----------
    misclassification_rate : np.ndarray, shape (n_samples,)
        For each sample, the fraction of times it was misclassified
        across all (seed, split) pairs where it appeared in the test set.
        Values range from 0 (always correct) to 1 (always wrong).
    appearances : np.ndarray, shape (n_samples,)
        Number of times each sample appeared in a test set.
    hard_indices : np.ndarray
        Indices of the hardest samples (misclassification_rate > 0.5),
        sorted by rate descending.
    easy_indices : np.ndarray
        Indices of the easiest samples (misclassification_rate == 0).
    n_seeds : int
    n_splits : int
    model_name : str
    """

    misclassification_rate: np.ndarray
    appearances: np.ndarray
    hard_indices: np.ndarray
    easy_indices: np.ndarray
    n_seeds: int
    n_splits: int
    model_name: str = "model"

    def __repr__(self) -> str:
        n = len(self.misclassification_rate)
        n_hard = len(self.hard_indices)
        n_easy = len(self.easy_indices)
        n_medium = n - n_hard - n_easy
        lines = [
            f"{'─'*56}",
            f"  Dataset Sensitivity: {self.model_name}",
            f"  ({self.n_seeds} seeds × {self.n_splits} splits)",
            f"{'─'*56}",
            f"  Total samples   : {n}",
            f"  Easy  (rate=0)  : {n_easy}  ({n_easy/n*100:.1f}%)  — always correct",
            f"  Medium (0<r≤0.5): {n_medium}  ({n_medium/n*100:.1f}%)  — sometimes wrong",
            f"  Hard  (rate>0.5): {n_hard}  ({n_hard/n*100:.1f}%)  — usually wrong",
            f"{'─'*56}",
            f"  Mean misclassification rate : {self.misclassification_rate.mean():.3f}",
            f"  Hardest sample rate         : {self.misclassification_rate.max():.3f}",
            f"{'─'*56}",
        ]
        return "\n".join(lines)


def dataset_sensitivity(
    model_factory: Callable,
    X,
    y,
    metric: Union[str, Callable] = "accuracy",
    n_seeds: int = 10,
    n_splits: int = 5,
    stratified: bool = True,
    model_name: str = "model",
) -> DatasetSensitivityResults:
    """
    Identify which samples in your dataset are hardest to classify.

    Runs the model repeatedly across seeds and splits, tracking per-sample
    misclassification rates. Samples that are frequently misclassified
    regardless of the random seed are genuinely ambiguous or mislabelled.

    Parameters
    ----------
    model_factory : callable
        (seed) -> unfitted sklearn-compatible model.
    X, y : array-like
    n_seeds : int
    n_splits : int
    stratified : bool
    model_name : str

    Returns
    -------
    DatasetSensitivityResults

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from mlci.sensitivity.dataset import dataset_sensitivity
    >>>
    >>> results = dataset_sensitivity(
    ...     model_factory=lambda seed: RandomForestClassifier(random_state=seed),
    ...     X=X, y=y,
    ...     n_seeds=10, n_splits=5,
    ...     model_name="RandomForest",
    ... )
    >>> print(results)
    >>> # Inspect hardest samples
    >>> hard_X = X[results.hard_indices]
    >>> hard_y = y[results.hard_indices]
    """

    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = len(y)

    # Accumulators: for each sample, count appearances and errors
    appearances = np.zeros(n_samples, dtype=int)
    errors      = np.zeros(n_samples, dtype=int)

    cv_cls = StratifiedKFold if stratified else KFold

    total = n_seeds * n_splits
    with tqdm(total=total, desc="dataset sensitivity") as pbar:
        for seed in range(n_seeds):
            _seed_everything(seed)
            cv = cv_cls(n_splits=n_splits, shuffle=True, random_state=seed)

            for train_idx, test_idx in cv.split(X, y):
                model = model_factory(seed)
                model.fit(X[train_idx], y[train_idx])
                y_pred = model.predict(X[test_idx])

                appearances[test_idx] += 1
                errors[test_idx] += (y_pred != y[test_idx]).astype(int)
                pbar.update(1)

    # Misclassification rate — avoid division by zero for unvisited samples
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(appearances > 0, errors / appearances, np.nan)

    hard_mask = rate > 0.5
    easy_mask = rate == 0.0

    hard_indices = np.where(hard_mask)[0][np.argsort(-rate[hard_mask])]
    easy_indices = np.where(easy_mask)[0]

    return DatasetSensitivityResults(
        misclassification_rate=rate,
        appearances=appearances,
        hard_indices=hard_indices,
        easy_indices=easy_indices,
        n_seeds=n_seeds,
        n_splits=n_splits,
        model_name=model_name,
    )
