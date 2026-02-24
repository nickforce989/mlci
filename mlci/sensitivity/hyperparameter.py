"""
mlci.sensitivity.hyperparameter
================================

Hyperparameter sensitivity analysis.

Answers the question: how sensitive is my model's performance to
hyperparameter choice, and which hyperparameters matter most?

The key insight: variance in scores comes from three sources —
  1. Random seed (initialisation noise)
  2. Data split (which samples are in the test set)
  3. Hyperparameter choice

This module isolates source 3 by running a grid of hyperparameter
combinations and decomposing how much each one moves the needle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from mlci.core.results import ExperimentResults
from mlci.stats.bootstrap import bootstrap_ci


@dataclass
class HparamSensitivityResults:
    """
    Output of a hyperparameter sensitivity analysis.

    Attributes
    ----------
    param_names : list of str
        Names of the hyperparameters that were varied.
    param_grid : list of dict
        Each entry is one hyperparameter combination.
    results : list of ExperimentResults
        One ExperimentResults per combination in param_grid.
    best_params : dict
        The combination with the highest mean score.
    best_result : ExperimentResults
        The ExperimentResults for best_params.
    sensitivity : dict
        {param_name: float} — estimated sensitivity (std of means when
        this param is varied, other params held at their best values).
    """

    param_names: List[str]
    param_grid: List[Dict[str, Any]]
    results: List[ExperimentResults]
    best_params: Dict[str, Any]
    best_result: ExperimentResults
    sensitivity: Dict[str, float]

    def summary_table(self) -> str:
        """Print a table of all combinations sorted by mean score."""
        lines = [
            f"{'─'*70}",
            f"  Hyperparameter Sensitivity — {self.results[0].metric}",
            f"{'─'*70}",
            f"  {'Params':<40} {'Mean':>7} {'CI Lower':>9} {'CI Upper':>9}",
            f"  {'─'*40} {'─'*7} {'─'*9} {'─'*9}",
        ]

        sorted_pairs = sorted(
            zip(self.param_grid, self.results),
            key=lambda x: x[1].mean,
            reverse=self.results[0].higher_is_better,
        )

        for params, res in sorted_pairs:
            ci = bootstrap_ci(res)
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            lines.append(
                f"  {param_str:<40} {ci.mean:>7.4f} {ci.lower:>9.4f} {ci.upper:>9.4f}"
            )

        lines.append(f"{'─'*70}")
        lines.append(f"\n  Most sensitive parameter: "
                     f"{max(self.sensitivity, key=self.sensitivity.get)}")
        lines.append(f"  Sensitivity (std of means):")
        for k, v in sorted(self.sensitivity.items(), key=lambda x: -x[1]):
            lines.append(f"    {k:<30} σ = {v:.4f}")
        lines.append(f"{'─'*70}")

        output = "\n".join(lines)
        print(output)
        return output


def hyperparameter_sensitivity(
    model_factory: Callable,
    param_grid: Dict[str, List[Any]],
    X,
    y,
    metric: Union[str, Callable] = "accuracy",
    n_seeds: int = 10,
    n_splits: int = 5,
    model_name: str = "model",
    higher_is_better: bool = True,
    n_jobs: int = 1,
) -> HparamSensitivityResults:
    """
    Run a grid of hyperparameter combinations and measure sensitivity.

    For each combination in param_grid, runs a full mlci Experiment
    (n_seeds × n_splits) and collects ExperimentResults. Then computes
    per-parameter sensitivity as the standard deviation of mean scores
    when that parameter is varied (other params held fixed at best values).

    Parameters
    ----------
    model_factory : callable
        A function (seed, **hparams) -> unfitted model.
        Must accept `seed` as first argument and the hyperparameter
        names as keyword arguments.
    param_grid : dict
        {param_name: [value1, value2, ...]}
        All combinations are evaluated (full grid search).
    X, y : array-like
    metric : str or callable
    n_seeds, n_splits, n_jobs : passed to Experiment
    model_name : str
    higher_is_better : bool

    Returns
    -------
    HparamSensitivityResults

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from mlci.sensitivity.hyperparameter import hyperparameter_sensitivity
    >>>
    >>> results = hyperparameter_sensitivity(
    ...     model_factory=lambda seed, n_estimators, max_depth:
    ...         RandomForestClassifier(
    ...             n_estimators=n_estimators,
    ...             max_depth=max_depth,
    ...             random_state=seed,
    ...         ),
    ...     param_grid={
    ...         "n_estimators": [10, 50, 100, 200],
    ...         "max_depth":    [None, 5, 10],
    ...     },
    ...     X=X, y=y,
    ...     metric="accuracy",
    ...     n_seeds=5,
    ...     n_splits=5,
    ... )
    >>> results.summary_table()
    """
    from mlci.core.experiment import Experiment

    X = np.asarray(X)
    y = np.asarray(y)

    # Build all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = [dict(zip(keys, combo)) for combo in product(*values)]

    print(
        f"Hyperparameter sensitivity: {len(combinations)} combinations × "
        f"{n_seeds} seeds × {n_splits} splits = "
        f"{len(combinations) * n_seeds * n_splits} evaluations"
    )

    all_results = []
    for combo in tqdm(combinations, desc="hparam grid"):
        param_str = ", ".join(f"{k}={v}" for k, v in combo.items())
        exp = Experiment(
            model_factory=lambda seed, c=combo: model_factory(seed, **c),
            X=X, y=y,
            metric=metric,
            n_seeds=n_seeds,
            n_splits=n_splits,
            n_jobs=n_jobs,
            model_name=f"{model_name}({param_str})",
            higher_is_better=higher_is_better,
        )
        all_results.append(exp.run(verbose=False))

    # Best combination
    means = [r.mean for r in all_results]
    best_idx = int(np.argmax(means) if higher_is_better else np.argmin(means))
    best_params = combinations[best_idx]
    best_result = all_results[best_idx]

    # Per-parameter sensitivity
    # For each param, fix all others at best value and measure std of means
    sensitivity = {}
    for key in keys:
        param_means = []
        for val in param_grid[key]:
            # Find all combinations where this param = val,
            # others = best values
            matching = [
                (combo, res)
                for combo, res in zip(combinations, all_results)
                if combo[key] == val
                and all(
                    combo[k] == best_params[k]
                    for k in keys if k != key
                )
            ]
            if matching:
                param_means.append(np.mean([r.mean for _, r in matching]))

        sensitivity[key] = float(np.std(param_means)) if len(param_means) > 1 else 0.0

    return HparamSensitivityResults(
        param_names=keys,
        param_grid=combinations,
        results=all_results,
        best_params=best_params,
        best_result=best_result,
        sensitivity=sensitivity,
    )
