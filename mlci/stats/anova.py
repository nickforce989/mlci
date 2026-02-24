"""
Variance decomposition: how much of your score variance comes from
random initialisation (seed) vs which data ended up in the test set (split)?

These are fundamentally different sources of uncertainty:
  - Seed variance: aleatoric — the same dataset, different model trajectories
  - Split variance: epistemic — how sensitive is your evaluation to the
    specific subset of data you happened to test on?

Uses a one-way ANOVA decomposition on the (n_seeds × n_splits) score matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlci.core.results import ExperimentResults


@dataclass
class VarianceDecomposition:
    """
    Output of a variance decomposition on ExperimentResults.

    Attributes
    ----------
    total_variance : float
    seed_variance : float
        Variance attributable to random seed choice.
    split_variance : float
        Variance attributable to data split choice.
    interaction_variance : float
        Residual (seed × split interaction).
    seed_fraction : float
        Fraction of total variance from seeds.
    split_fraction : float
        Fraction of total variance from splits.
    interaction_fraction : float
    """

    total_variance: float
    seed_variance: float
    split_variance: float
    interaction_variance: float
    seed_fraction: float
    split_fraction: float
    interaction_fraction: float
    model_name: str = ""
    metric: str = ""

    def __repr__(self) -> str:
        lines = [
            f"{'─'*56}",
            f"  Variance Decomposition: {self.model_name} ({self.metric})",
            f"{'─'*56}",
            f"  Total variance  : {self.total_variance:.6f}",
            f"  Seed variance   : {self.seed_variance:.6f}  ({self.seed_fraction*100:.1f}%)",
            f"  Split variance  : {self.split_variance:.6f}  ({self.split_fraction*100:.1f}%)",
            f"  Interaction     : {self.interaction_variance:.6f}  ({self.interaction_fraction*100:.1f}%)",
            f"{'─'*56}",
            f"  Interpretation:",
        ]

        if self.seed_fraction > 0.5:
            lines.append(
                "  High seed variance: results are sensitive to random init."
            )
            lines.append(
                "  Run more seeds to get a stable estimate.")
        if self.split_fraction > 0.5:
            lines.append(
                "  High split variance: results are sensitive to which"
            )
            lines.append(
                "  data ends up in the test set. Consider more folds or")
            lines.append(
                "  a larger test set.")
        if self.seed_fraction < 0.2 and self.split_fraction < 0.2:
            lines.append(
                "  Most variance is in the seed×split interaction — the"
            )
            lines.append(
                "  effect of different seeds changes across splits.")

        lines.append(f"{'─'*56}")
        return "\n".join(lines)


def decompose_variance(results: ExperimentResults) -> VarianceDecomposition:
    """
    Decompose the variance of results.scores into seed and split components.

    Uses a balanced two-way ANOVA without replication:
      score[i,j] = mu + alpha_i (seed effect) + beta_j (split effect) + eps_ij

    Parameters
    ----------
    results : ExperimentResults
        Must have n_seeds > 1 and n_splits > 1 for a meaningful decomposition.

    Returns
    -------
    VarianceDecomposition
    """

    scores = results.scores  # (n_seeds, n_splits)
    n_seeds, n_splits = scores.shape

    if n_seeds < 2:
        raise ValueError(
            "Need at least 2 seeds for variance decomposition. "
            f"Got n_seeds={n_seeds}."
        )
    if n_splits < 2:
        raise ValueError(
            "Need at least 2 splits for variance decomposition. "
            f"Got n_splits={n_splits}."
        )

    grand_mean = scores.mean()
    seed_means = scores.mean(axis=1)   # (n_seeds,)
    split_means = scores.mean(axis=0)  # (n_splits,)

    # Sum of squares
    SS_seed = n_splits * np.sum((seed_means - grand_mean) ** 2)
    SS_split = n_seeds * np.sum((split_means - grand_mean) ** 2)

    # Total SS
    SS_total = np.sum((scores - grand_mean) ** 2)

    # Interaction / residual
    SS_interaction = SS_total - SS_seed - SS_split

    # Convert to variance components (mean squares)
    df_seed = n_seeds - 1
    df_split = n_splits - 1
    df_interaction = df_seed * df_split
    df_total = n_seeds * n_splits - 1

    MS_seed = SS_seed / df_seed
    MS_split = SS_split / df_split
    MS_interaction = SS_interaction / df_interaction if df_interaction > 0 else 0.0

    # Variance components (estimate of sigma^2 contributions)
    # Under balanced design:
    # E[MS_seed]        = sigma^2_eps + n_splits * sigma^2_seed
    # E[MS_split]       = sigma^2_eps + n_seeds  * sigma^2_split
    # E[MS_interaction] = sigma^2_eps
    sigma2_eps = max(MS_interaction, 0.0)
    sigma2_seed = max((MS_seed - sigma2_eps) / n_splits, 0.0)
    sigma2_split = max((MS_split - sigma2_eps) / n_seeds, 0.0)

    total_var = sigma2_seed + sigma2_split + sigma2_eps
    if total_var == 0:
        seed_frac = split_frac = interaction_frac = 0.0
    else:
        seed_frac = sigma2_seed / total_var
        split_frac = sigma2_split / total_var
        interaction_frac = sigma2_eps / total_var

    return VarianceDecomposition(
        total_variance=total_var,
        seed_variance=sigma2_seed,
        split_variance=sigma2_split,
        interaction_variance=sigma2_eps,
        seed_fraction=seed_frac,
        split_fraction=split_frac,
        interaction_fraction=interaction_frac,
        model_name=results.model_name,
        metric=results.metric,
    )
