"""
mlci.integrations.sklearn
=========================

Utilities for working with sklearn-compatible models in mlci.

Provides:
  - PipelineFactory       : build model+scaler pipelines from config dicts
  - wrap_cross_val_scores : turn existing cross_val_score arrays into
                            ExperimentResults without rerunning anything
  - ModelGrid             : run a grid of models and return all results
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from mlci.core.results import ExperimentResults


# -----------------------------------------------------------------------
# Scaler registry
# -----------------------------------------------------------------------

_SCALERS = {
    "standard": StandardScaler,
    "minmax":   MinMaxScaler,
    "robust":   RobustScaler,
    "none":     None,
}


# -----------------------------------------------------------------------
# PipelineFactory
# -----------------------------------------------------------------------

class PipelineFactory:
    """
    Build a sklearn Pipeline from a config dict.

    Useful when you want to define experiments declaratively (e.g. from a
    YAML file) rather than writing lambda functions by hand.

    Parameters
    ----------
    estimator : sklearn estimator class (not instance)
    params : dict
        Keyword arguments passed to the estimator constructor.
        The special key `"random_state"` is automatically set to the
        seed passed at call time, overriding any value in params.
    scaler : str or None
        One of "standard", "minmax", "robust", "none". Default "standard".

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> factory = PipelineFactory(
    ...     RandomForestClassifier,
    ...     params={"n_estimators": 100, "max_depth": 5},
    ...     scaler="standard",
    ... )
    >>> model = factory(seed=42)   # returns a fresh, unfitted Pipeline
    """

    def __init__(
        self,
        estimator,
        params: Optional[Dict[str, Any]] = None,
        scaler: str = "none",
    ):
        self.estimator_cls = estimator
        self.params = params or {}
        self.scaler = scaler

        if scaler not in _SCALERS:
            raise ValueError(
                f"Unknown scaler '{scaler}'. Choose from {list(_SCALERS)}."
            )

    def __call__(self, seed: int = 0) -> Union[Pipeline, BaseEstimator]:
        p = dict(self.params)
        # Inject seed if the estimator accepts random_state
        try:
            import inspect
            sig = inspect.signature(self.estimator_cls.__init__)
            if "random_state" in sig.parameters:
                p["random_state"] = seed
        except Exception:
            pass

        estimator = self.estimator_cls(**p)

        scaler_cls = _SCALERS[self.scaler]
        if scaler_cls is None:
            return estimator

        return Pipeline([
            ("scaler", scaler_cls()),
            ("estimator", estimator),
        ])

    def __repr__(self) -> str:
        return (
            f"PipelineFactory({self.estimator_cls.__name__}, "
            f"scaler={self.scaler!r}, params={self.params})"
        )


# -----------------------------------------------------------------------
# wrap_cross_val_scores
# -----------------------------------------------------------------------

def wrap_cross_val_scores(
    scores: Union[np.ndarray, List[float], List[List[float]]],
    metric: str = "accuracy",
    model_name: str = "model",
    higher_is_better: bool = True,
) -> ExperimentResults:
    """
    Wrap existing cross-validation scores into an ExperimentResults object
    without rerunning any experiments.

    Use this when you already have scores from sklearn's cross_val_score,
    cross_validate, or your own training loop, and want to use mlci's
    statistical tools on them.

    Parameters
    ----------
    scores : array-like
        Either:
          - 1-D (n_splits,)        : one run, multiple folds
          - 2-D (n_seeds, n_splits): multiple runs, multiple folds
        Treated as per-seed per-split accuracy values.
    metric : str
    model_name : str
    higher_is_better : bool

    Returns
    -------
    ExperimentResults

    Examples
    --------
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from mlci.integrations.sklearn import wrap_cross_val_scores
    >>>
    >>> # Run cross-val yourself
    >>> scores_seed0 = cross_val_score(
    ...     RandomForestClassifier(random_state=0), X, y, cv=5
    ... )
    >>> scores_seed1 = cross_val_score(
    ...     RandomForestClassifier(random_state=1), X, y, cv=5
    ... )
    >>> scores_seed2 = cross_val_score(
    ...     RandomForestClassifier(random_state=2), X, y, cv=5
    ... )
    >>>
    >>> # Stack and wrap
    >>> import numpy as np
    >>> results = wrap_cross_val_scores(
    ...     np.vstack([scores_seed0, scores_seed1, scores_seed2]),
    ...     metric="accuracy",
    ...     model_name="RandomForest",
    ... )
    >>> from mlci.stats.bootstrap import summary
    >>> summary(results)
    """
    arr = np.asarray(scores, dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]   # (1, n_splits)

    return ExperimentResults(
        scores=arr,
        metric=metric,
        model_name=model_name,
        higher_is_better=higher_is_better,
    )


# -----------------------------------------------------------------------
# ModelGrid
# -----------------------------------------------------------------------

class ModelGrid:
    """
    Run a grid of models on the same dataset and return all results.

    Provides a clean way to compare many models in one call, returning
    a dict of {model_name: ExperimentResults} that can be passed directly
    to mlci's comparison and visualisation functions.

    Parameters
    ----------
    models : dict
        {model_name: factory_callable}
        Each value must be a callable (seed) -> unfitted sklearn model.
    X, y : array-like
    metric : str or callable
    n_seeds : int
    n_splits : int
    n_jobs : int
    higher_is_better : bool

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> from mlci.integrations.sklearn import ModelGrid, PipelineFactory
    >>>
    >>> grid = ModelGrid(
    ...     models={
    ...         "RandomForest":  lambda seed: RandomForestClassifier(random_state=seed),
    ...         "GradientBoost": lambda seed: GradientBoostingClassifier(random_state=seed),
    ...         "LogisticReg":   PipelineFactory(LogisticRegression, scaler="standard"),
    ...     },
    ...     X=X, y=y,
    ...     metric="accuracy",
    ...     n_seeds=10,
    ...     n_splits=5,
    ... )
    >>> all_results = grid.run()
    >>> # all_results is a dict: {name: ExperimentResults}
    >>>
    >>> # Use directly with mlci viz
    >>> from mlci.viz.plots import plot_comparison
    >>> fig = plot_comparison(list(all_results.values()))
    """

    def __init__(
        self,
        models: Dict[str, Callable],
        X,
        y,
        metric: Union[str, Callable] = "accuracy",
        n_seeds: int = 10,
        n_splits: int = 5,
        n_jobs: int = 1,
        higher_is_better: bool = True,
    ):
        self.models = models
        self.X = X
        self.y = y
        self.metric = metric
        self.n_seeds = n_seeds
        self.n_splits = n_splits
        self.n_jobs = n_jobs
        self.higher_is_better = higher_is_better

    def run(self, verbose: bool = True) -> Dict[str, ExperimentResults]:
        from mlci.core.experiment import Experiment

        results = {}
        for name, factory in self.models.items():
            if verbose:
                print(f"\n→ {name}")
            exp = Experiment(
                model_factory=factory,
                X=self.X,
                y=self.y,
                metric=self.metric,
                n_seeds=self.n_seeds,
                n_splits=self.n_splits,
                n_jobs=self.n_jobs,
                model_name=name,
                higher_is_better=self.higher_is_better,
            )
            results[name] = exp.run(verbose=verbose)
        return results
