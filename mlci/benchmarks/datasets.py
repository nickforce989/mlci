"""
mlci.benchmarks.datasets
========================

Standard benchmark datasets for reproducible ML evaluation.

All datasets are loaded from sklearn (no downloads needed) and returned
as (X, y, metadata) tuples with a consistent interface.

Available datasets
------------------
Classification:
  load_breast_cancer_bench()   — binary, 569 samples, 30 features
  load_iris_bench()            — multiclass, 150 samples, 4 features
  load_digits_bench()          — multiclass, 1797 samples, 64 features
  load_wine_bench()            — multiclass, 178 samples, 13 features

Regression:
  load_diabetes_bench()        — 442 samples, 10 features
  load_california_bench()      — 20640 samples, 8 features

All:
  load_all_classification()    — returns list of all classification datasets
  load_all_regression()        — returns list of all regression datasets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class BenchmarkDataset:
    """
    A standardised benchmark dataset.

    Attributes
    ----------
    name : str
    X : np.ndarray
    y : np.ndarray
    task : str — "classification" or "regression"
    n_samples : int
    n_features : int
    n_classes : int or None (None for regression)
    default_metric : str — recommended metric for this dataset
    description : str
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    task: str
    n_samples: int
    n_features: int
    n_classes: int | None
    default_metric: str
    description: str

    def __repr__(self) -> str:
        class_str = f", {self.n_classes} classes" if self.n_classes else ""
        return (
            f"BenchmarkDataset('{self.name}', "
            f"{self.n_samples} samples, {self.n_features} features{class_str}, "
            f"metric='{self.default_metric}')"
        )


# -----------------------------------------------------------------------
# Classification datasets
# -----------------------------------------------------------------------

def load_breast_cancer_bench() -> BenchmarkDataset:
    """Binary classification. 569 samples, 30 features."""
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    return BenchmarkDataset(
        name="BreastCancer",
        X=data.data.astype(float),
        y=data.target,
        task="classification",
        n_samples=len(data.target),
        n_features=data.data.shape[1],
        n_classes=2,
        default_metric="accuracy",
        description="Binary classification of malignant vs benign tumours. "
                     "30 numeric features derived from cell nucleus images.",
    )


def load_iris_bench() -> BenchmarkDataset:
    """Multiclass classification. 150 samples, 4 features, 3 classes."""
    from sklearn.datasets import load_iris
    data = load_iris()
    return BenchmarkDataset(
        name="Iris",
        X=data.data.astype(float),
        y=data.target,
        task="classification",
        n_samples=len(data.target),
        n_features=data.data.shape[1],
        n_classes=3,
        default_metric="accuracy",
        description="Classic multiclass classification of iris flower species. "
                     "4 numeric features. Small but useful for quick tests.",
    )


def load_digits_bench() -> BenchmarkDataset:
    """Multiclass classification. 1797 samples, 64 features, 10 classes."""
    from sklearn.datasets import load_digits
    data = load_digits()
    return BenchmarkDataset(
        name="Digits",
        X=data.data.astype(float),
        y=data.target,
        task="classification",
        n_samples=len(data.target),
        n_features=data.data.shape[1],
        n_classes=10,
        default_metric="accuracy",
        description="Handwritten digit recognition (0-9). 8×8 pixel images "
                     "flattened to 64 features. Larger than Iris, multiclass.",
    )


def load_wine_bench() -> BenchmarkDataset:
    """Multiclass classification. 178 samples, 13 features, 3 classes."""
    from sklearn.datasets import load_wine
    data = load_wine()
    return BenchmarkDataset(
        name="Wine",
        X=data.data.astype(float),
        y=data.target,
        task="classification",
        n_samples=len(data.target),
        n_features=data.data.shape[1],
        n_classes=3,
        default_metric="accuracy",
        description="Wine cultivar classification from chemical analysis. "
                     "13 numeric features, 3 classes.",
    )


# -----------------------------------------------------------------------
# Regression datasets
# -----------------------------------------------------------------------

def load_diabetes_bench() -> BenchmarkDataset:
    """Regression. 442 samples, 10 features."""
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    return BenchmarkDataset(
        name="Diabetes",
        X=data.data.astype(float),
        y=data.target.astype(float),
        task="regression",
        n_samples=len(data.target),
        n_features=data.data.shape[1],
        n_classes=None,
        default_metric="rmse",
        description="Diabetes disease progression prediction. "
                     "10 numeric features, continuous target.",
    )


def load_california_bench() -> BenchmarkDataset:
    """Regression. 20640 samples, 8 features. Requires internet on first download."""
    try:
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
    except Exception as e:
        raise RuntimeError(
            "CaliforniaHousing requires a one-time download. "
            "Make sure you have internet access and run:\n"
            "  from sklearn.datasets import fetch_california_housing\n"
            "  fetch_california_housing()  # downloads and caches\n"
            f"Original error: {e}"
        ) from e
    return BenchmarkDataset(
        name="CaliforniaHousing",
        X=data.data.astype(float),
        y=data.target.astype(float),
        task="regression",
        n_samples=len(data.target),
        n_features=data.data.shape[1],
        n_classes=None,
        default_metric="rmse",
        description="California housing price prediction. "
                     "8 numeric features, continuous target (median house value). "
                     "Largest dataset in the standard suite.",
    )


# -----------------------------------------------------------------------
# Collection loaders
# -----------------------------------------------------------------------

def load_all_classification() -> List[BenchmarkDataset]:
    """Return all classification benchmark datasets."""
    return [
        load_breast_cancer_bench(),
        load_iris_bench(),
        load_digits_bench(),
        load_wine_bench(),
    ]


def load_all_regression() -> List[BenchmarkDataset]:
    """Return all regression benchmark datasets."""
    return [
        load_diabetes_bench(),
        load_california_bench(),
    ]
