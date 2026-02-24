"""
mlci.integrations.torch
=======================

PyTorch integration for mlci.

Wraps a PyTorch training loop in a sklearn-compatible interface
(fit / predict) so it works seamlessly with mlci's Experiment runner.

Key features:
  - Correct seeding: torch + numpy + random + CUDA
  - Caching: completed (seed, split) runs are saved to disk and skipped
    on re-run — essential when training takes hours
  - Early stopping with checkpoint restore
  - Automatic device selection (CUDA if available)
  - Works with any nn.Module architecture

Requirements:
    pip install torch

Usage
-----
>>> import torch
>>> import torch.nn as nn
>>> from mlci.integrations.torch import TorchTrainer
>>> from mlci import Experiment
>>>
>>> class MLP(nn.Module):
...     def __init__(self):
...         super().__init__()
...         self.net = nn.Sequential(
...             nn.Linear(30, 64), nn.ReLU(),
...             nn.Linear(64, 32), nn.ReLU(),
...             nn.Linear(32, 1),
...         )
...     def forward(self, x):
...         return self.net(x)
>>>
>>> exp = Experiment(
...     model_factory=lambda seed: TorchTrainer(
...         model_fn=MLP,
...         lr=1e-3,
...         n_epochs=50,
...         batch_size=32,
...         random_state=seed,
...         cache_dir="./mlci_cache",
...     ),
...     X=X, y=y,
...     metric="accuracy",
...     n_seeds=10,
...     n_splits=5,
...     model_name="MLP",
... )
>>> results = exp.run()
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import warnings
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def _check_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for mlci.integrations.torch.\n"
            "Install it with: pip install torch\n"
            "See https://pytorch.org/get-started/locally/ for platform-specific instructions."
        )


def seed_torch(seed: int) -> None:
    """
    Set all relevant random seeds for fully reproducible PyTorch training.

    Covers: Python random, NumPy, PyTorch CPU, PyTorch CUDA (all GPUs).
    Also sets torch.backends.cudnn.deterministic = True.
    """
    torch = _check_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TorchTrainer(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible wrapper around a PyTorch training loop.

    Implements fit() and predict() so it works as a drop-in inside
    mlci's Experiment runner. Handles seeding, caching, early stopping,
    and device management automatically.

    Parameters
    ----------
    model_fn : callable
        A zero-argument function that returns a fresh nn.Module instance.
        Called once per fit() to ensure clean initialisation.
        Example: lambda: MyNet(input_dim=30, hidden=64)
    lr : float
        Learning rate for Adam optimiser. Default 1e-3.
    n_epochs : int
        Maximum number of training epochs. Default 100.
    batch_size : int
        Mini-batch size. Default 32.
    patience : int
        Early stopping patience (epochs without val loss improvement).
        Default 10. Set to None to disable early stopping.
    val_fraction : float
        Fraction of training data to hold out for early stopping.
        Default 0.1. Only used if patience is not None.
    weight_decay : float
        L2 regularisation (Adam weight_decay). Default 1e-4.
    device : str or None
        "cuda", "cpu", or None (auto-detect). Default None.
    random_state : int
        Random seed. Default 0.
    cache_dir : str or None
        If set, completed runs are cached to disk. On re-run, cached
        results are loaded instead of retraining. Default None.
    verbose : bool
        Print loss at each epoch. Default False.

    Notes
    -----
    The cache key is a hash of (model architecture, training config,
    data split indices, seed). Two runs with identical config and data
    will reuse the cached score.

    This class assumes binary or multi-class classification. The output
    layer must produce logits (no softmax/sigmoid — those are applied
    internally during training and prediction).
    """

    def __init__(
        self,
        model_fn: Callable,
        lr: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 32,
        patience: Optional[int] = 10,
        val_fraction: float = 0.1,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
        random_state: int = 0,
        cache_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        self.model_fn = model_fn
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.weight_decay = weight_decay
        self.device = device
        self.random_state = random_state
        self.cache_dir = cache_dir
        self.verbose = verbose

    def _get_device(self):
        torch = _check_torch()
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _arch_fingerprint(self) -> str:
        """Hash of model architecture (layer names + shapes)."""
        try:
            tmp = self.model_fn()
            shape_str = str([(n, tuple(p.shape)) for n, p in tmp.named_parameters()])
            return hashlib.md5(shape_str.encode()).hexdigest()[:8]
        except Exception:
            return hashlib.md5(str(self.model_fn).encode()).hexdigest()[:8]

    def _cache_key(self, X_train: np.ndarray, y_train: np.ndarray) -> str:
        """Hash of model architecture + training config + data fingerprint."""
        config = {
            "arch": self._arch_fingerprint(),
            "lr": self.lr,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "weight_decay": self.weight_decay,
            "random_state": self.random_state,
            "n_train": len(y_train),
            "data_hash": hashlib.md5(
                np.ascontiguousarray(X_train[:5]).tobytes()
            ).hexdigest(),
        }
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]

    def _load_cache(self, key: str) -> Optional[np.ndarray]:
        if self.cache_dir is None:
            return None
        path = Path(self.cache_dir) / f"{key}_weights.npz"
        if path.exists():
            if self.verbose:
                print(f"    [cache hit] Loading weights from {path}")
            return np.load(str(path), allow_pickle=True)
        return None

    def _save_cache(self, key: str, state_dict: dict) -> None:
        if self.cache_dir is None:
            return
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        path = Path(self.cache_dir) / f"{key}_weights.npz"
        np.savez(str(path), **{k: v.cpu().numpy() for k, v in state_dict.items()})

    def fit(self, X, y):
        torch = _check_torch()
        import torch.nn as nn

        seed_torch(self.random_state)
        device = self._get_device()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Validation split for early stopping
        n = len(y)
        rng = np.random.RandomState(self.random_state)
        if self.patience is not None:
            val_size = max(1, int(self.val_fraction * n))
            idx = rng.permutation(n)
            val_idx, train_idx = idx[:val_size], idx[val_size:]
            X_val, y_val = X[val_idx], y[val_idx]
            X_tr,  y_tr  = X[train_idx], y[train_idx]
        else:
            X_tr, y_tr = X, y
            X_val, y_val = None, None

        # Build model
        self.model_ = self.model_fn().to(device)

        # Check cache
        cache_key = self._cache_key(X_tr, y_tr)
        cached = self._load_cache(cache_key)
        if cached is not None:
            state = {k: torch.tensor(cached[k]) for k in cached.files}
            self.model_.load_state_dict(state)
            return self

        # Loss and optimiser
        if self.n_classes_ == 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        optimiser = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Convert to tensors
        Xt = torch.tensor(X_tr, dtype=torch.float32).to(device)
        if self.n_classes_ == 2:
            yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(device)
        else:
            yt = torch.tensor(y_tr, dtype=torch.long).to(device)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        self.train_losses_ = []
        self.val_losses_ = []

        self.model_.train()
        for epoch in range(self.n_epochs):
            # Shuffle
            perm = torch.randperm(len(Xt))
            Xt_s, yt_s = Xt[perm], yt[perm]

            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(Xt_s), self.batch_size):
                Xb = Xt_s[start:start + self.batch_size]
                yb = yt_s[start:start + self.batch_size]

                # BatchNorm requires at least 2 samples per batch
                if len(Xb) < 2:
                    continue

                optimiser.zero_grad()
                out = self.model_(Xb)
                if self.n_classes_ == 2:
                    loss = criterion(out, yb)
                else:
                    loss = criterion(out, yb)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
                n_batches += 1

            train_loss = epoch_loss / n_batches
            self.train_losses_.append(train_loss)

            # Validation
            if X_val is not None and self.patience is not None:
                self.model_.eval()
                with torch.no_grad():
                    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
                    if self.n_classes_ == 2:
                        yv = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
                    else:
                        yv = torch.tensor(y_val, dtype=torch.long).to(device)
                    val_out = self.model_(Xv)
                    val_loss = criterion(val_out, yv).item()
                self.val_losses_.append(val_loss)
                self.model_.train()

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        if self.verbose:
                            print(f"    Early stopping at epoch {epoch}")
                        break

                if self.verbose and epoch % 10 == 0:
                    print(f"    epoch {epoch:4d}  train={train_loss:.4f}  val={val_loss:.4f}")
            else:
                if self.verbose and epoch % 10 == 0:
                    print(f"    epoch {epoch:4d}  train={train_loss:.4f}")

        # Restore best weights
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        # Save to cache
        self._save_cache(cache_key, self.model_.state_dict())

        return self

    def predict_proba(self, X) -> np.ndarray:
        torch = _check_torch()
        self.model_.eval()
        device = self._get_device()
        X = torch.tensor(np.asarray(X, dtype=np.float32)).to(device)

        with torch.no_grad():
            out = self.model_(X)
            if self.n_classes_ == 2:
                import torch.nn.functional as F
                p = torch.sigmoid(out).cpu().numpy().ravel()
                return np.column_stack([1 - p, p])
            else:
                import torch.nn.functional as F
                p = F.softmax(out, dim=1).cpu().numpy()
                return p

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
