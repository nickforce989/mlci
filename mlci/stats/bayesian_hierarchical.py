"""
mlci/stats/bayesian_hierarchical.py
-------------------------------------
Full Bayesian hierarchical comparison of two ML models using PyMC.

Why this is better than the existing analytic Bayesian method
-------------------------------------------------------------
The simple ``method="bayesian"`` comparison in :func:`mlci.stats.tests.compare`
works by fitting a Normal distribution to the bootstrap distribution of the
mean difference and computing P(A > B) analytically.  This is fast but ignores
the *structure* of the score matrix: the (n_seeds × n_splits) observations are
not all independent — folds within a seed share training data.

This module builds a proper two-level hierarchical model:

.. code-block:: text

    d[s, k]  ~  Normal(mu_s,      sigma_within)   # observed difference
    mu_s     ~  Normal(mu_global, sigma_between)   # per-seed random effect
    mu_global ~ Normal(0, 0.10)                    # global mean difference
    sigma_within ~ HalfNormal(0.05)                # within-seed (fold) noise
    sigma_between ~ HalfNormal(0.05)               # between-seed noise

Here d[s, k] = score_A[s, k] - score_B[s, k] is the per-(seed, split)
score difference.  The random effect mu_s absorbs correlations among the k
folds that share the same random seed, so the global inference on mu_global
is correctly calibrated.

Outputs
-------
* Posterior mean and 94 % HDI for mu_global (the expected performance gap).
* P(A > B) = fraction of posterior samples where mu_global > 0.
* ROPE (Region Of Practical Equivalence) analysis: what fraction of the
  posterior falls inside [-rope, +rope]?
* An optional trace plot and posterior plot via ArviZ.

Requirements
------------
    pip install mlci[bayesian]
    # which installs: pymc>=5.0, arviz>=0.16
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy imports — only needed at call time
# ---------------------------------------------------------------------------

def _require_pymc() -> tuple:
    """Import PyMC and ArviZ, raising a clear error if not installed."""
    try:
        import pymc as pm          # noqa: PLC0415
        import arviz as az         # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "PyMC and ArviZ are required for full Bayesian hierarchical "
            "comparison.\n\n"
            "Install them with:\n\n"
            "    pip install mlci[bayesian]\n\n"
            "or directly:\n\n"
            "    pip install pymc>=5.0 arviz>=0.16\n"
        ) from exc
    return pm, az


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BayesianHierarchicalResult:
    """
    Container for the output of :func:`bayesian_hierarchical_compare`.

    Attributes
    ----------
    model_a : str
        Name of model A.
    model_b : str
        Name of model B.
    metric : str
        Metric name (informational only).
    mu_mean : float
        Posterior mean of the global mean difference (A − B).
    mu_hdi_low : float
        Lower bound of the 94 % HDI for mu_global.
    mu_hdi_high : float
        Upper bound of the 94 % HDI for mu_global.
    prob_a_better : float
        P(A > B) = P(mu_global > 0), estimated from posterior samples.
    rope_fraction : float
        Fraction of the posterior for mu_global that falls inside the ROPE.
    rope : tuple[float, float]
        The ROPE interval used, e.g. (-0.005, +0.005).
    idata : arviz.InferenceData
        Full ArviZ inference data object for further analysis.
    """
    model_a: str
    model_b: str
    metric: str
    mu_mean: float
    mu_hdi_low: float
    mu_hdi_high: float
    prob_a_better: float
    rope_fraction: float
    rope: tuple[float, float]
    idata: object  # arviz.InferenceData

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        rope_lo, rope_hi = self.rope
        decision = self._decision_string()
        return (
            f"\n{'─' * 68}\n"
            f"  Bayesian Hierarchical Comparison (PyMC)\n"
            f"{'─' * 68}\n"
            f"  A  :  {self.model_a}\n"
            f"  B  :  {self.model_b}\n"
            f"  Metric    : {self.metric}\n"
            f"{'─' * 68}\n"
            f"  μ(A−B)    : {self.mu_mean:+.4f}  "
            f"[94% HDI: {self.mu_hdi_low:+.4f}, {self.mu_hdi_high:+.4f}]\n"
            f"  P(A > B)  : {self.prob_a_better:.3f}\n"
            f"  ROPE      : [{rope_lo:+.4f}, {rope_hi:+.4f}]\n"
            f"  In ROPE   : {self.rope_fraction:.1%} of posterior\n"
            f"{'─' * 68}\n"
            f"  Decision  : {decision}\n"
            f"{'─' * 68}\n"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BayesianHierarchicalResult("
            f"model_a={self.model_a!r}, model_b={self.model_b!r}, "
            f"mu_mean={self.mu_mean:.4f}, "
            f"prob_a_better={self.prob_a_better:.3f})"
        )

    def _decision_string(self) -> str:
        rope_lo, rope_hi = self.rope
        hdi_in_rope = (self.mu_hdi_low >= rope_lo) and (self.mu_hdi_high <= rope_hi)
        hdi_above_rope = self.mu_hdi_low > rope_hi
        hdi_below_rope = self.mu_hdi_high < rope_lo

        if hdi_in_rope:
            return (
                f"The 94% HDI lies entirely within the ROPE → "
                f"models are practically equivalent."
            )
        if hdi_above_rope:
            return (
                f"{self.model_a} is practically better than {self.model_b} "
                f"(HDI entirely above ROPE, P(A>B)={self.prob_a_better:.3f})."
            )
        if hdi_below_rope:
            return (
                f"{self.model_b} is practically better than {self.model_a} "
                f"(HDI entirely below ROPE, P(A>B)={self.prob_a_better:.3f})."
            )
        return (
            f"Inconclusive — HDI overlaps the ROPE. "
            f"Collect more data or relax the ROPE threshold."
        )


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def bayesian_hierarchical_compare(
    results_a,
    results_b,
    *,
    rope: float | tuple[float, float] = 0.005,
    hdi_prob: float = 0.94,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 42,
    progressbar: bool = True,
    plot: bool = False,
    plot_path: Optional[str] = None,
) -> BayesianHierarchicalResult:
    """
    Full Bayesian hierarchical comparison of two models using PyMC.

    Parameters
    ----------
    results_a, results_b : ExperimentResults
        Results from :class:`mlci.Experiment`.  Each must expose a
        ``.scores`` attribute of shape ``(n_seeds, n_splits)`` and a
        ``.model_name`` / ``.metric`` string attribute.
    rope : float or (float, float)
        Region Of Practical Equivalence.  Differences smaller than this
        are treated as practically irrelevant.  If a single float ``r`` is
        given, the ROPE is ``(-r, +r)``.  A good default for accuracy is
        0.005 (half a percentage point).
    hdi_prob : float
        Probability mass of the Highest Density Interval to report.
        Default 0.94 (94 %) following McElreath / Kruschke conventions.
    draws : int
        Number of MCMC samples per chain after tuning.
    tune : int
        Number of burn-in / tuning steps per chain.
    chains : int
        Number of independent MCMC chains.
    target_accept : float
        Target acceptance rate for the NUTS sampler.  Increase towards 0.99
        if you see divergences.
    random_seed : int
        Seed for reproducibility.
    progressbar : bool
        Whether to display the PyMC sampling progress bar.
    plot : bool
        If True, produce and show/save posterior and trace plots via ArviZ.
    plot_path : str or None
        Directory to save plots.  ``None`` means show interactively.

    Returns
    -------
    BayesianHierarchicalResult

    Examples
    --------
    >>> from mlci.stats.bayesian_hierarchical import bayesian_hierarchical_compare
    >>> result = bayesian_hierarchical_compare(rf_results, gb_results, rope=0.005)
    >>> print(result)

    Passing the result to ``compare()`` (method dispatch)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    You can also call this via the unified interface::

        from mlci.stats.tests import compare
        compare(rf_results, gb_results, method="bayesian_hierarchical")
    """
    pm, az = _require_pymc()

    # ------------------------------------------------------------------ #
    # 1.  Validate inputs and extract score matrices
    # ------------------------------------------------------------------ #
    scores_a: np.ndarray = np.asarray(results_a.scores, dtype=float)
    scores_b: np.ndarray = np.asarray(results_b.scores, dtype=float)

    if scores_a.shape != scores_b.shape:
        raise ValueError(
            f"Score matrices must have the same shape, got "
            f"{scores_a.shape} vs {scores_b.shape}.  Run both experiments "
            f"with identical n_seeds and n_splits."
        )

    n_seeds, n_splits = scores_a.shape
    # Per-(seed, split) difference: shape (n_seeds, n_splits)
    diff = scores_a - scores_b

    model_a_name: str = getattr(results_a, "model_name", "Model A")
    model_b_name: str = getattr(results_b, "model_name", "Model B")
    metric_name: str  = getattr(results_a, "metric", "metric")

    # Normalise ROPE argument
    if isinstance(rope, (int, float)):
        rope_lo, rope_hi = -float(rope), float(rope)
    else:
        rope_lo, rope_hi = float(rope[0]), float(rope[1])

    # ------------------------------------------------------------------ #
    # 2.  Build the hierarchical PyMC model
    # ------------------------------------------------------------------ #
    #
    #   diff[s, k] ~ Normal(mu_s,      sigma_within)
    #   mu_s       ~ Normal(mu_global, sigma_between)
    #   mu_global  ~ Normal(0, 0.10)
    #   sigma_within  ~ HalfNormal(0.05)
    #   sigma_between ~ HalfNormal(0.05)
    #
    # Using non-centred parameterisation for better sampling geometry.
    # ------------------------------------------------------------------ #

    with pm.Model() as model:  # noqa: F841  (used by context)
        # Hyper-priors
        mu_global     = pm.Normal("mu_global",     mu=0.0,  sigma=0.10)
        sigma_between = pm.HalfNormal("sigma_between", sigma=0.05)
        sigma_within  = pm.HalfNormal("sigma_within",  sigma=0.05)

        # Non-centred per-seed random effects
        seed_offset = pm.Normal("seed_offset", mu=0.0, sigma=1.0,
                                shape=n_seeds)
        mu_s = pm.Deterministic("mu_s",
                                mu_global + sigma_between * seed_offset)

        # Likelihood — broadcast mu_s over splits
        # mu_s shape: (n_seeds,)  →  (n_seeds, 1)  broadcasts with diff
        pm.Normal(
            "obs",
            mu=mu_s[:, None],
            sigma=sigma_within,
            observed=diff,
        )

        # ----------------------------------------------------------------
        # 3.  Sample
        # ----------------------------------------------------------------
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            return_inferencedata=True,
        )

    # ------------------------------------------------------------------ #
    # 4.  Summarise posterior
    # ------------------------------------------------------------------ #
    posterior_mu: np.ndarray = (
        idata.posterior["mu_global"].values.flatten()  # shape (chains*draws,)
    )

    mu_mean   = float(np.mean(posterior_mu))
    hdi_vals  = az.hdi(idata, var_names=["mu_global"], hdi_prob=hdi_prob)
    mu_hdi    = hdi_vals["mu_global"].values  # [low, high]
    mu_hdi_low, mu_hdi_high = float(mu_hdi[0]), float(mu_hdi[1])

    prob_a_better = float(np.mean(posterior_mu > 0))
    rope_fraction = float(
        np.mean((posterior_mu >= rope_lo) & (posterior_mu <= rope_hi))
    )

    result = BayesianHierarchicalResult(
        model_a       = model_a_name,
        model_b       = model_b_name,
        metric        = metric_name,
        mu_mean       = mu_mean,
        mu_hdi_low    = mu_hdi_low,
        mu_hdi_high   = mu_hdi_high,
        prob_a_better = prob_a_better,
        rope_fraction = rope_fraction,
        rope          = (rope_lo, rope_hi),
        idata         = idata,
    )

    # ------------------------------------------------------------------ #
    # 5.  Optional plots
    # ------------------------------------------------------------------ #
    if plot:
        _make_plots(az, idata, result, plot_path)

    return result


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def _make_plots(az, idata, result: BayesianHierarchicalResult,
                plot_path: Optional[str]) -> None:
    """Produce ArviZ posterior and trace plots."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rope_lo, rope_hi = result.rope

    # --- Posterior plot with ROPE shading ---
    fig, ax = plt.subplots(figsize=(8, 4))
    az.plot_posterior(
        idata,
        var_names=["mu_global"],
        hdi_prob=0.94,
        ref_val=0,
        rope=[rope_lo, rope_hi],
        ax=ax,
    )
    ax.set_title(
        f"Posterior of μ(A−B)   ({result.model_a} vs {result.model_b})\n"
        f"P(A>B) = {result.prob_a_better:.3f}  |  "
        f"In ROPE = {result.rope_fraction:.1%}"
    )

    if plot_path:
        import os  # noqa: PLC0415
        os.makedirs(plot_path, exist_ok=True)
        fig.savefig(
            os.path.join(plot_path, "bayesian_posterior.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

    # --- Trace plot ---
    az.plot_trace(idata, var_names=["mu_global", "sigma_between", "sigma_within"])
    if plot_path:
        import os  # noqa: PLC0415
        plt.savefig(
            os.path.join(plot_path, "bayesian_trace.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()
    else:
        plt.tight_layout()
        plt.show()