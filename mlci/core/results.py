"""
ExperimentResults: the central data structure of mlci.

Holds a (n_seeds x n_splits) array of scalar scores, plus metadata.
Everything downstream — bootstrap, hypothesis tests, plots — works on this.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ExperimentResults:
    """
    Container for the output of a repeated evaluation experiment.

    Attributes
    ----------
    scores : np.ndarray, shape (n_seeds, n_splits)
        Raw metric values. Each row is one random seed, each column is one
        data split (or a single column if no cross-validation was used).
    metric : str
        Name of the metric (e.g. "accuracy", "f1", "mse").
    model_name : str
        Human-readable name for the model, used in plots and reports.
    higher_is_better : bool
        Whether larger values of the metric are better.
    metadata : dict
        Arbitrary key-value pairs (hyperparameters, dataset name, etc.).
    """

    scores: np.ndarray          # shape: (n_seeds, n_splits)
    metric: str = "accuracy"
    model_name: str = "model"
    higher_is_better: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.scores = np.asarray(self.scores, dtype=float)
        if self.scores.ndim == 1:
            # Treat a 1-D array as (n_seeds, 1)
            self.scores = self.scores[:, np.newaxis]
        if self.scores.ndim != 2:
            raise ValueError(
                f"scores must be 1-D or 2-D, got shape {self.scores.shape}"
            )

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def n_seeds(self) -> int:
        return self.scores.shape[0]

    @property
    def n_splits(self) -> int:
        return self.scores.shape[1]

    @property
    def flat(self) -> np.ndarray:
        """All scores as a 1-D array (n_seeds * n_splits,)."""
        return self.scores.ravel()

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores))

    @property
    def std(self) -> float:
        return float(np.std(self.scores, ddof=1))

    @property
    def seed_means(self) -> np.ndarray:
        """Mean across splits for each seed. Shape: (n_seeds,)."""
        return self.scores.mean(axis=1)

    @property
    def split_means(self) -> np.ndarray:
        """Mean across seeds for each split. Shape: (n_splits,)."""
        return self.scores.mean(axis=0)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save results to disk (JSON for portability)."""
        path = Path(path)
        data = {
            "scores": self.scores.tolist(),
            "metric": self.metric,
            "model_name": self.model_name,
            "higher_is_better": self.higher_is_better,
            "metadata": self.metadata,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentResults":
        """Load results from a JSON file saved with .save()."""
        path = Path(path)
        data = json.loads(path.read_text())
        return cls(
            scores=np.array(data["scores"]),
            metric=data["metric"],
            model_name=data["model_name"],
            higher_is_better=data["higher_is_better"],
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return (
            f"ExperimentResults(model='{self.model_name}', "
            f"metric='{self.metric}', "
            f"shape=({self.n_seeds} seeds × {self.n_splits} splits), "
            f"mean={self.mean:.4f}, std={self.std:.4f})"
        )
