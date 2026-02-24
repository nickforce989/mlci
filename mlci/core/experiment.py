"""
Experiment runner: execute a model evaluation multiple times across seeds and
data splits, collecting raw scores into an ExperimentResults object.

Supports:
  - sklearn-compatible models (fit/predict interface)
  - Custom callable evaluators
  - Parallel execution via joblib
  - Deterministic seeding (numpy + random + torch if available)
"""

from __future__ import annotations

import random
import time
import warnings
from typing import Callable, Optional, Sequence, Union

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm

from mlci.core.results import ExperimentResults


def _seed_everything(seed: int) -> None:
    """Set all relevant random seeds deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _run_single(
    model_factory: Callable,
    X, y,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    metric_fn: Callable,
    seed: int,
) -> float:
    """Run one (seed, split) combination. Returns a scalar score."""
    _seed_everything(seed)

    model = model_factory(seed=seed)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return float(metric_fn(y_test, y_pred))


class Experiment:
    """
    Repeated evaluation of an ML model across multiple seeds and data splits.

    Parameters
    ----------
    model_factory : callable
        A function that takes a `seed` keyword argument and returns a
        freshly initialised, unfitted model with a sklearn-compatible
        fit/predict interface.
    X, y : array-like
        Full dataset. Splitting and subsetting is handled internally.
    metric : callable or str
        Evaluation function (y_true, y_pred) -> float, or a string name
        from the built-in registry ("accuracy", "f1", "rmse", "mae", "r2").
    n_seeds : int
        Number of random seeds to run. Default 10.
    n_splits : int
        Number of cross-validation folds. Use 1 for a fixed 80/20 split.
    stratified : bool
        Use stratified k-fold (for classification). Default True.
    n_jobs : int
        Parallelism. -1 = use all cores. Default 1.
    model_name : str
        Name for the model in outputs.
    higher_is_better : bool
        Whether larger metric values are better.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_iris
    >>> from mlci import Experiment
    >>>
    >>> X, y = load_iris(return_X_y=True)
    >>>
    >>> exp = Experiment(
    ...     model_factory=lambda seed: RandomForestClassifier(random_state=seed),
    ...     X=X, y=y,
    ...     metric="accuracy",
    ...     n_seeds=10,
    ...     n_splits=5,
    ...     model_name="RandomForest",
    ... )
    >>> results = exp.run()
    >>> print(results)
    """

    _METRIC_REGISTRY: dict[str, Callable] = {}

    def __init__(
        self,
        model_factory: Callable,
        X,
        y,
        metric: Union[str, Callable] = "accuracy",
        n_seeds: int = 10,
        n_splits: int = 5,
        stratified: bool = True,
        n_jobs: int = 1,
        model_name: str = "model",
        higher_is_better: bool = True,
        metadata: Optional[dict] = None,
    ):
        self.model_factory = model_factory
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.metric_fn = self._resolve_metric(metric)
        self.metric_name = metric if isinstance(metric, str) else metric.__name__
        self.n_seeds = n_seeds
        self.n_splits = n_splits
        self.stratified = stratified
        self.n_jobs = n_jobs
        self.model_name = model_name
        self.higher_is_better = higher_is_better
        self.metadata = metadata or {}

    # ------------------------------------------------------------------
    # Metric resolution
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_metric(cls, metric: Union[str, Callable]) -> Callable:
        if callable(metric):
            return metric

        from sklearn import metrics as skm

        registry = {
            "accuracy":  skm.accuracy_score,
            "f1":        lambda yt, yp: skm.f1_score(yt, yp, average="macro", zero_division=0),
            "f1_binary": lambda yt, yp: skm.f1_score(yt, yp, average="binary", zero_division=0),
            "roc_auc":   skm.roc_auc_score,
            "rmse":      lambda yt, yp: float(np.sqrt(skm.mean_squared_error(yt, yp))),
            "mae":       skm.mean_absolute_error,
            "r2":        skm.r2_score,
            "mse":       skm.mean_squared_error,
        }

        if metric not in registry:
            raise ValueError(
                f"Unknown metric '{metric}'. "
                f"Available: {list(registry.keys())}. "
                "Or pass a callable (y_true, y_pred) -> float."
            )
        return registry[metric]

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def _get_splits(self, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return list of (train_idx, test_idx) tuples for one seed."""
        if self.n_splits == 1:
            # Single 80/20 split
            rng = np.random.RandomState(seed)
            n = len(self.y)
            idx = rng.permutation(n)
            split = int(0.8 * n)
            return [(idx[:split], idx[split:])]

        cv_cls = StratifiedKFold if self.stratified else KFold
        cv = cv_cls(n_splits=self.n_splits, shuffle=True, random_state=seed)
        return list(cv.split(self.X, self.y))

    # ------------------------------------------------------------------
    # Main run method
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> ExperimentResults:
        """
        Execute all (seed × split) combinations and return an ExperimentResults.

        Returns
        -------
        ExperimentResults
            Shape (n_seeds, n_splits).
        """
        seeds = list(range(self.n_seeds))
        all_splits = [self._get_splits(seed) for seed in seeds]

        # Build flat list of tasks
        tasks = [
            (seed, split_idx, train_idx, test_idx)
            for seed, splits in zip(seeds, all_splits)
            for split_idx, (train_idx, test_idx) in enumerate(splits)
        ]

        if verbose:
            print(
                f"Running {self.model_name} | "
                f"{self.n_seeds} seeds × {self.n_splits} splits = "
                f"{len(tasks)} evaluations"
            )

        t0 = time.time()

        def _task(seed, split_idx, train_idx, test_idx):
            return (seed, split_idx, _run_single(
                self.model_factory, self.X, self.y,
                train_idx, test_idx, self.metric_fn, seed,
            ))

        if self.n_jobs == 1:
            raw = [
                _task(s, si, tr, te)
                for s, si, tr, te in tqdm(tasks, disable=not verbose, desc="eval")
            ]
        else:
            raw = Parallel(n_jobs=self.n_jobs)(
                delayed(_task)(s, si, tr, te) for s, si, tr, te in tasks
            )

        # Assemble into (n_seeds, n_splits) array
        scores = np.full((self.n_seeds, self.n_splits), np.nan)
        for seed, split_idx, score in raw:
            scores[seed, split_idx] = score

        if verbose:
            elapsed = time.time() - t0
            print(f"Done in {elapsed:.1f}s — mean {self.metric_name}: {scores.mean():.4f}")

        return ExperimentResults(
            scores=scores,
            metric=self.metric_name,
            model_name=self.model_name,
            higher_is_better=self.higher_is_better,
            metadata={
                **self.metadata,
                "n_seeds": self.n_seeds,
                "n_splits": self.n_splits,
            },
        )
