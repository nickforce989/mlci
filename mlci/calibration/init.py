"""
mlci.calibration — confidence calibration analysis across seeds.

Public API
----------
CalibrationExperiment   : run a classifier across seeds, collect probabilities
CalibrationResults      : results container with ECE, MCE, reliability curves
compute_ece             : compute ECE for a single (y_true, y_prob) pair
compute_mce             : compute MCE for a single (y_true, y_prob) pair
reliability_curve       : reliability diagram data for a single run
plot_reliability_diagram : reliability diagram with uncertainty bands
plot_ece_distribution   : ECE distribution across seeds
plot_calibration_comparison : forest plot of ECE across models
plot_reliability_overlay    : overlay reliability curves for multiple models
"""

from mlci.calibration.ece import (
    CalibrationResults,
    compute_ece,
    compute_mce,
    reliability_curve,
)
from mlci.calibration.experiment import CalibrationExperiment
from mlci.calibration.plots import (
    plot_reliability_diagram,
    plot_ece_distribution,
    plot_calibration_comparison,
    plot_reliability_overlay,
)

__all__ = [
    "CalibrationExperiment",
    "CalibrationResults",
    "compute_ece",
    "compute_mce",
    "reliability_curve",
    "plot_reliability_diagram",
    "plot_ece_distribution",
    "plot_calibration_comparison",
    "plot_reliability_overlay",
]