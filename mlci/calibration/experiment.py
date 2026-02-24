"""
CalibrationExperiment: runs a classification model across seeds and splits,
collects predicted probabilities, and returns a CalibrationResults object.

This is the calibration-aware twin of Experiment. The key difference: instead
of only tracking a scalar metric, it saves the full (y_true, y_prob) pairs for
every (seed, split) run, enabling reliable diagram uncertainty quantification.

Usage
-----
>>> from mlci.calibration.experiment import CalibrationExperiment
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.datasets import load_breast_cancer

>>> X, y = load_breast_cancer(return_X_y=True)
>>> cal = CalibrationExperiment(
...     model_factory=lambda seed: RandomForestClassifier(random_state=seed),
...     X=X, y=y,
...     n_seeds=20, n_splits=5,
...     model_name="RandomForest",
... )
>>> cal_results = cal.run()
>>> print(cal_results)
>>> fig = plot_reliability_diagram(cal_results)
"""

from __future__ import annotations

import random
import time
import warnings
from typing import Callable, Optional, Union

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm

from mlci.calibration.ece import (
    CalibrationResults,
    compute_ece,
    compute_mce,
    reliability_curve,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _run_single_calibration(
    model_factory: Callable,
    X, y,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    n_bins: int,
    strategy: str,
    class_index: int,
):
    """Run one (seed, split). Returns (ece, mce, bin_conf, bin_acc, bin_cnt)."""
    _seed_everything(seed)

    model = model_factory(seed=seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)

    # Get probability estimates
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        # Fallback: sigmoid-transform decision function
        df = model.decision_function(X_test)
        proba = 1.0 / (1.0 + np.exp(-df))
        proba = np.column_stack([1 - proba, proba])
    else:
        raise AttributeError(
            f"Model {type(model).__name__} has neither predict_proba nor "
            "decision_function. Calibration analysis requires probability outputs."
        )

    if proba.ndim == 1:
        y_prob = proba
    else:
        n_classes = proba.shape[1]
        if n_classes == 2:
            # Binary: use positive class probability
            y_prob = proba[:, 1]
        else:
            # Multiclass: use the specified class index
            y_prob = proba[:, class_index]
            # For multiclass ECE, convert y_test to binary (one-vs-rest)
            y_test = (y_test == class_index).astype(int)

    ece = compute_ece(y_test, y_prob, n_bins=n_bins, strategy=strategy)
    mce = compute_mce(y_test, y_prob, n_bins=n_bins)
    bin_conf, bin_acc, bin_cnt = reliability_curve(
        y_test, y_prob, n_bins=n_bins, strategy=strategy
    )

    return ece, mce, bin_conf, bin_acc, bin_cnt


class CalibrationExperiment:
    """
    Repeated calibration analysis of a classifier across seeds and data splits.

    Unlike Experiment (which tracks a scalar metric), CalibrationExperiment
    collects predicted probabilities and computes:
      - ECE (Expected Calibration Error) for every (seed, split)
      - MCE (Maximum Calibration Error) for every (seed, split)
      - Reliability diagram data (confidence vs accuracy per bin) for every run

    The result is a CalibrationResults object that gives you:
      - Distribution of ECE across seeds → CI on calibration quality
      - Reliability diagram with uncertainty bands → how stable is calibration?

    Parameters
    ----------
    model_factory : callable
        A function `model_factory(seed=int)` returning an unfitted classifier
        with a `predict_proba` method (or `decision_function` as fallback).
    X, y : array-like
        Classification dataset. For multiclass problems, set `class_index`.
    n_seeds : int
        Number of random seeds. Default 10.
    n_splits : int
        Number of CV folds. Default 5.
    n_bins : int
        Number of calibration bins. Default 10.
    strategy : str
        "uniform" (equal-width bins) or "quantile" (equal-frequency bins).
    class_index : int
        For multiclass: which class probability to calibrate on (one-vs-rest).
        For binary: ignored (always uses positive class). Default 1.
    stratified : bool
        Use stratified k-fold. Default True.
    model_name : str

    Examples
    --------
    >>> cal = CalibrationExperiment(
    ...     model_factory=lambda seed: RandomForestClassifier(random_state=seed),
    ...     X=X, y=y, n_seeds=20, n_splits=5,
    ... )
    >>> results = cal.run()
    >>> print(results)
    >>> fig = plot_reliability_diagram(results)
    >>> fig.savefig("calibration.png")
    """

    def __init__(
        self,
        model_factory: Callable,
        X,
        y,
        n_seeds: int = 10,
        n_splits: int = 5,
        n_bins: int = 10,
        strategy: str = "uniform",
        class_index: int = 1,
        stratified: bool = True,
        model_name: str = "model",
    ):
        self.model_factory = model_factory
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.n_seeds = n_seeds
        self.n_splits = n_splits
        self.n_bins = n_bins
        self.strategy = strategy
        self.class_index = class_index
        self.stratified = stratified
        self.model_name = model_name

    def _get_splits(self, seed: int):
        if self.n_splits == 1:
            rng = np.random.RandomState(seed)
            n = len(self.y)
            idx = rng.permutation(n)
            split = int(0.8 * n)
            return [(idx[:split], idx[split:])]

        cv_cls = StratifiedKFold if self.stratified else KFold
        cv = cv_cls(n_splits=self.n_splits, shuffle=True, random_state=seed)
        return list(cv.split(self.X, self.y))

    def run(self, verbose: bool = True) -> CalibrationResults:
        """
        Run calibration analysis across all seeds and splits.

        Returns
        -------
        CalibrationResults
        """
        seeds = list(range(self.n_seeds))
        all_splits = [self._get_splits(seed) for seed in seeds]

        tasks = [
            (seed, split_idx, train_idx, test_idx)
            for seed, splits in zip(seeds, all_splits)
            for split_idx, (train_idx, test_idx) in enumerate(splits)
        ]

        n_runs = len(tasks)

        if verbose:
            print(
                f"Calibration analysis: {self.model_name} | "
                f"{self.n_seeds} seeds × {self.n_splits} splits = {n_runs} runs | "
                f"{self.n_bins} bins ({self.strategy})"
            )

        t0 = time.time()

        ece_arr = np.full((self.n_seeds, self.n_splits), np.nan)
        mce_arr = np.full((self.n_seeds, self.n_splits), np.nan)
        bin_confs = np.full((n_runs, self.n_bins), np.nan)
        bin_accs = np.full((n_runs, self.n_bins), np.nan)
        bin_cnts = np.zeros((n_runs, self.n_bins), dtype=int)

        run_idx = 0
        for seed, split_idx, train_idx, test_idx in tqdm(
            tasks, disable=not verbose, desc="calibration"
        ):
            ece, mce, bc, ba, bct = _run_single_calibration(
                self.model_factory, self.X, self.y,
                train_idx, test_idx, seed,
                self.n_bins, self.strategy, self.class_index,
            )
            ece_arr[seed, split_idx] = ece
            mce_arr[seed, split_idx] = mce
            bin_confs[run_idx] = bc
            bin_accs[run_idx] = ba
            bin_cnts[run_idx] = bct
            run_idx += 1

        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)

        if verbose:
            elapsed = time.time() - t0
            print(
                f"Done in {elapsed:.1f}s — "
                f"mean ECE: {np.nanmean(ece_arr):.4f} "
                f"(± {np.nanstd(ece_arr, ddof=1):.4f})"
            )

        return CalibrationResults(
            ece_scores=ece_arr,
            mce_scores=mce_arr,
            bin_confidences=bin_confs,
            bin_accuracies=bin_accs,
            bin_counts=bin_cnts,
            bin_edges=bin_edges,
            n_bins=self.n_bins,
            model_name=self.model_name,
            n_seeds=self.n_seeds,
            n_splits=self.n_splits,
            strategy=self.strategy,
        )