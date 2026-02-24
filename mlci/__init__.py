"""
mlci: Statistically rigorous ML model evaluation and benchmarking.

Every number you report from an ML experiment is a random variable.
This library forces you to treat it that way.
"""

from mlci.core.experiment import Experiment
from mlci.core.results import ExperimentResults
from mlci.stats.bootstrap import bootstrap_ci, summary
from mlci.stats.tests import compare
from mlci.stats.anova import decompose_variance
from mlci.integrations.sklearn import PipelineFactory, wrap_cross_val_scores, ModelGrid
from mlci.benchmarks.datasets import (
    load_breast_cancer_bench, load_iris_bench, load_digits_bench,
    load_wine_bench, load_diabetes_bench, load_california_bench,
    load_all_classification, load_all_regression,
)
from mlci.benchmarks.runner import run_benchmark, print_benchmark_report
from mlci.sensitivity.hyperparameter import hyperparameter_sensitivity
from mlci.sensitivity.dataset import dataset_sensitivity
from mlci.sensitivity.learning_curve import learning_curve

# Power analysis / MDE calculator
from mlci.stats.power import (
    mde_n_seeds,
    mde_effect,
    power_analysis,
    estimate_sigma,
    quick_summary as power_quick_summary,
    plot_power_curve,
    plot_power_heatmap,
    SampleSizeResult,
    MDEResult,
    PowerCurveResult,
)

# Calibration analysis
from mlci.calibration.experiment import (
    CalibrationExperiment,
)

from mlci.calibration.plots import (
    plot_reliability_diagram,
    plot_ece_distribution,
    plot_calibration_comparison,
    plot_reliability_overlay,
)

from mlci.calibration.ece import (
    CalibrationResults,
    compute_ece,
    compute_mce,
    reliability_curve,
)

__version__ = "0.1.0"
__all__ = [
    # Core
    "Experiment", "ExperimentResults",
    # Stats
    "bootstrap_ci", "summary", "compare", "decompose_variance",
    # Integrations
    "PipelineFactory", "wrap_cross_val_scores", "ModelGrid",
    # Benchmarks
    "run_benchmark", "print_benchmark_report",
    "load_breast_cancer_bench", "load_iris_bench", "load_digits_bench",
    "load_wine_bench", "load_diabetes_bench", "load_california_bench",
    "load_all_classification", "load_all_regression",
    # Sensitivity
    "hyperparameter_sensitivity", "dataset_sensitivity", "learning_curve",
    # Power analysis / MDE
    "mde_n_seeds", "mde_effect", "power_analysis", "estimate_sigma",
    "power_quick_summary", "plot_power_curve", "plot_power_heatmap",
    "SampleSizeResult", "MDEResult", "PowerCurveResult",
    # Calibration
    "CalibrationExperiment", "CalibrationResults",
    "compute_ece", "compute_mce", "reliability_curve",
    "plot_reliability_diagram", "plot_ece_distribution",
    "plot_calibration_comparison", "plot_reliability_overlay",
]