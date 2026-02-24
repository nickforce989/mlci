"""
mlci.benchmarks.runner
======================

Multi-dataset benchmarking with multiple testing correction.

The key problem: if you compare two models on 5 datasets and use
α=0.05 for each test, you expect one spurious significant result
by chance alone (5 × 0.05 = 0.25 expected false positives).

This module:
  1. Runs a model grid across multiple datasets
  2. Applies Benjamini-Hochberg correction for multiple comparisons
  3. Reports which wins are robust vs which are noise
  4. Produces a cross-dataset ranking with uncertainty
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from mlci.benchmarks.datasets import BenchmarkDataset
from mlci.core.results import ExperimentResults
from mlci.stats.bootstrap import bootstrap_ci
from mlci.stats.tests import corrected_resampled_ttest


@dataclass
class BenchmarkResult:
    """
    Results for one model on one dataset.

    Attributes
    ----------
    model_name : str
    dataset_name : str
    result : ExperimentResults
    mean : float
    ci_lower, ci_upper : float
    """

    model_name: str
    dataset_name: str
    result: ExperimentResults
    mean: float
    ci_lower: float
    ci_upper: float


@dataclass
class MultiBenchmarkResults:
    """
    Full results from running a model grid across multiple datasets.

    Attributes
    ----------
    model_names : list of str
    dataset_names : list of str
    results : dict
        {(model_name, dataset_name): ExperimentResults}
    comparisons : dict
        {dataset_name: {(model_a, model_b): ComparisonResult}}
    rankings : dict
        {dataset_name: list of (model_name, mean_score)} sorted best-first
    corrected_wins : dict
        {dataset_name: model_name or None}
        The model that wins on each dataset after multiple testing correction.
        None if no significant winner after correction.
    """

    model_names: List[str]
    dataset_names: List[str]
    results: Dict[Tuple[str, str], ExperimentResults]
    comparisons: Dict
    rankings: Dict
    corrected_wins: Dict


def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Benjamini-Hochberg procedure for controlling the false discovery rate.

    More powerful than Bonferroni for multiple comparisons — controls
    the expected proportion of false positives rather than the probability
    of any false positive.

    Parameters
    ----------
    p_values : list of float
    alpha : float
        FDR threshold. Default 0.05.

    Returns
    -------
    list of bool — True where the null hypothesis is rejected after correction.
    """
    n = len(p_values)
    if n == 0:
        return []

    # Rank p-values
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    reject = [False] * n

    for rank, (orig_idx, p) in enumerate(indexed, start=1):
        if p <= (rank / n) * alpha:
            reject[orig_idx] = True
        else:
            # Once a p-value fails, all larger ones fail too (BH is sequential)
            break

    # Enforce monotonicity: if rank k rejects, all ranks < k reject
    if any(reject):
        max_reject_rank = max(
            rank for rank, (_, _) in enumerate(indexed, 1)
            if reject[indexed[rank - 1][0]]
        )
        for rank, (orig_idx, _) in enumerate(indexed, 1):
            if rank <= max_reject_rank:
                reject[orig_idx] = True

    return reject


def run_benchmark(
    models: Dict[str, Callable],
    datasets: List[BenchmarkDataset],
    n_seeds: int = 10,
    n_splits: int = 5,
    n_jobs: int = 1,
    alpha: float = 0.05,
    reference_model: Optional[str] = None,
    verbose: bool = True,
) -> MultiBenchmarkResults:
    """
    Run a model grid across multiple datasets with multiple testing correction.

    Parameters
    ----------
    models : dict
        {model_name: factory_callable}
        Each callable takes (seed) and returns an unfitted sklearn model.
    datasets : list of BenchmarkDataset
    n_seeds, n_splits, n_jobs : passed to each Experiment
    alpha : float
        Significance threshold after BH correction. Default 0.05.
    reference_model : str or None
        If set, all pairwise comparisons are made against this model only.
        If None, all pairwise combinations are tested.
    verbose : bool

    Returns
    -------
    MultiBenchmarkResults
    """
    from mlci.core.experiment import Experiment

    model_names = list(models.keys())
    dataset_names = [d.name for d in datasets]

    all_results: Dict[Tuple[str, str], ExperimentResults] = {}

    # ── Run all experiments ──────────────────────────────────────
    total = len(models) * len(datasets)
    if verbose:
        print(f"Running {len(models)} models × {len(datasets)} datasets = {total} experiment sets")
        print(f"Each set: {n_seeds} seeds × {n_splits} splits\n")

    for dataset in datasets:
        if verbose:
            print(f"  Dataset: {dataset.name} ({dataset.n_samples} samples)")

        for model_name, factory in models.items():
            if verbose:
                print(f"    → {model_name}")

            exp = Experiment(
                model_factory=factory,
                X=dataset.X,
                y=dataset.y,
                metric=dataset.default_metric,
                n_seeds=n_seeds,
                n_splits=n_splits,
                n_jobs=n_jobs,
                model_name=model_name,
                higher_is_better=(dataset.task == "classification"
                                   or dataset.default_metric in ("accuracy", "f1", "roc_auc", "r2")),
            )
            all_results[(model_name, dataset.name)] = exp.run(verbose=False)

    # ── Rankings per dataset ─────────────────────────────────────
    rankings: Dict[str, List] = {}
    for dataset in datasets:
        res_list = [(m, all_results[(m, dataset.name)]) for m in model_names]
        higher = all_results[(model_names[0], dataset.name)].higher_is_better
        sorted_res = sorted(res_list, key=lambda x: x[1].mean, reverse=higher)
        rankings[dataset.name] = [(m, r.mean) for m, r in sorted_res]

    # ── Pairwise comparisons with BH correction ──────────────────
    comparisons: Dict = {}
    corrected_wins: Dict = {}

    for dataset in datasets:
        comparisons[dataset.name] = {}

        if reference_model is not None:
            pairs = [
                (reference_model, m)
                for m in model_names if m != reference_model
            ]
        else:
            pairs = [
                (model_names[i], model_names[j])
                for i in range(len(model_names))
                for j in range(i + 1, len(model_names))
            ]

        p_values = []
        pair_results = []
        for a_name, b_name in pairs:
            res_a = all_results[(a_name, dataset.name)]
            res_b = all_results[(b_name, dataset.name)]
            comp = corrected_resampled_ttest(res_a, res_b)
            comparisons[dataset.name][(a_name, b_name)] = comp
            p_values.append(comp.p_value)
            pair_results.append((a_name, b_name, comp))

        # Apply BH correction
        if p_values:
            rejected = benjamini_hochberg(p_values, alpha=alpha)
            sig_pairs = [
                (a, b, comp)
                for (a, b, comp), rej in zip(pair_results, rejected)
                if rej
            ]

            if sig_pairs:
                # Find the best model among those involved in significant comparisons
                higher = all_results[(model_names[0], dataset.name)].higher_is_better
                best = max(
                    model_names,
                    key=lambda m: all_results[(m, dataset.name)].mean
                ) if higher else min(
                    model_names,
                    key=lambda m: all_results[(m, dataset.name)].mean
                )
                corrected_wins[dataset.name] = best
            else:
                corrected_wins[dataset.name] = None

    return MultiBenchmarkResults(
        model_names=model_names,
        dataset_names=dataset_names,
        results=all_results,
        comparisons=comparisons,
        rankings=rankings,
        corrected_wins=corrected_wins,
    )


def print_benchmark_report(results: MultiBenchmarkResults, alpha: float = 0.05) -> None:
    """
    Print a full benchmark report to console.

    Shows per-dataset rankings, pairwise comparison p-values,
    multiple testing correction results, and an overall win count.
    """

    print("\n" + "═" * 70)
    print("  MULTI-DATASET BENCHMARK REPORT")
    print("═" * 70)

    # Per-dataset rankings
    for dataset_name in results.dataset_names:
        print(f"\n  ── {dataset_name} ──")
        for rank, (model_name, mean) in enumerate(results.rankings[dataset_name], 1):
            ci = bootstrap_ci(results.results[(model_name, dataset_name)])
            winner = " ← best" if rank == 1 else ""
            print(f"    {rank}. {model_name:<26} {ci.mean:.4f}  [{ci.lower:.4f}, {ci.upper:.4f}]{winner}")

        winner = results.corrected_wins.get(dataset_name)
        if winner:
            print(f"    Significant winner (BH corrected): {winner}")
        else:
            print(f"    No significant winner after multiple testing correction")

    # Win counts across datasets
    print(f"\n  ── Overall win count (after BH correction) ──")
    win_counts = {m: 0 for m in results.model_names}
    for dataset_name in results.dataset_names:
        winner = results.corrected_wins.get(dataset_name)
        if winner:
            win_counts[winner] += 1

    for model_name, wins in sorted(win_counts.items(), key=lambda x: -x[1]):
        bar = "█" * wins
        print(f"    {model_name:<26} {wins} wins  {bar}")

    print("\n" + "═" * 70)
