"""Epoch-based ERP/P300 spatial filtering and classification baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _epochs(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 1:
        raise ValueError("X must have shape (trials, channels, samples)")
    if not np.isfinite(values).all():
        raise ValueError("X must contain only finite values")
    return values


@dataclass
class XDawnLDA:
    """xDAWN spatial filters followed by a regularized LDA classifier.

    This epoch-based form estimates the target ERP from labeled training
    epochs. It expects already segmented stimulus epochs; event timing and
    target/non-target construction belong to the caller's data pipeline.
    """

    n_components: int = 4
    positive_label: Any = 1
    ridge: float = 1e-6
    filters_: np.ndarray | None = None
    classifier_: Any = None
    classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XDawnLDA":
        epochs = _epochs(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(epochs):
            raise ValueError("y must contain one label per epoch")
        if len(np.unique(labels)) < 2:
            raise ValueError("xDAWN+LDA requires at least two target classes")
        if self.positive_label not in labels:
            raise ValueError("positive_label is absent from training labels")
        if self.n_components < 1 or self.n_components > epochs.shape[1]:
            raise ValueError("n_components must be between 1 and the channel count")
        try:
            from scipy.linalg import eigh
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        except ImportError as exc:  # pragma: no cover - optional dependencies
            raise ImportError("install scipy and scikit-learn to use xDAWN+LDA") from exc

        target = epochs[labels == self.positive_label]
        evoked = target.mean(axis=0)
        signal_cov = evoked @ evoked.T
        total_cov = np.einsum("nct,ndt->cd", epochs, epochs, optimize=True)
        total_cov /= max(epochs.shape[0] * epochs.shape[-1], 1)
        scale = max(float(np.trace(total_cov)) / total_cov.shape[0], 1.0)
        total_cov += self.ridge * scale * np.eye(total_cov.shape[0])
        values, vectors = eigh(signal_cov, total_cov, check_finite=False)
        order = np.argsort(values)[::-1][: self.n_components]
        self.filters_ = vectors[:, order].T
        features = self._transform(epochs)
        self.classifier_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.classifier_.fit(features, labels)
        self.classes_ = self.classifier_.classes_
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.filters_ is None:
            raise RuntimeError("call fit before transform")
        values = _epochs(X)
        if values.shape[1] != self.filters_.shape[1]:
            raise ValueError("channel count differs from fitted xDAWN filters")
        return np.einsum("kc,nct->nkt", self.filters_, values, optimize=True).reshape(len(values), -1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classifier_ is None:
            raise RuntimeError("call fit before predict")
        return self.classifier_.predict(self._transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classifier_ is None:
            raise RuntimeError("call fit before predict_proba")
        return self.classifier_.predict_proba(self._transform(X))
