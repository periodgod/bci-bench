"""Classical benchmark models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .features import filterbank_spatial_logvar_features, select_top_features
from .signal import MI_FILTERBANK_BANDS


class StandardizedDiagonalLDA:
    """Shared diagonal-covariance LDA with no native linear algebra dependency."""

    def __init__(self, variance_floor: float = 1e-5):
        self.variance_floor = float(variance_floor)
        self.classes_: np.ndarray | None = None
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.variance_: np.ndarray | None = None
        self.log_priors_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "StandardizedDiagonalLDA":
        features = np.asarray(x, dtype=np.float32)
        labels = np.asarray(y, dtype=np.int64)
        classes = np.unique(labels)
        if classes.size < 2:
            raise ValueError("At least two classes are required")
        self.classes_ = classes
        self.center_ = np.mean(features, axis=0)
        self.scale_ = np.maximum(np.std(features, axis=0), 1e-6)
        z = (features - self.center_) / self.scale_
        means = []
        counts = []
        pooled = np.zeros(z.shape[1], dtype=np.float32)
        dof = 0
        for label in classes:
            subset = z[labels == label]
            means.append(np.mean(subset, axis=0))
            counts.append(len(subset))
            if len(subset) > 1:
                pooled += np.var(subset, axis=0, ddof=1) * float(len(subset) - 1)
                dof += len(subset) - 1
        self.means_ = np.stack(means, axis=0).astype(np.float32)
        self.variance_ = np.maximum(pooled / max(dof, 1), self.variance_floor).astype(np.float32)
        priors = np.asarray(counts, dtype=np.float32) / float(np.sum(counts))
        self.log_priors_ = np.log(np.maximum(priors, 1e-12)).astype(np.float32)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.center_ is None or self.scale_ is None:
            raise RuntimeError("Model is not fitted")
        assert self.means_ is not None
        assert self.variance_ is not None
        assert self.log_priors_ is not None
        z = (np.asarray(x, dtype=np.float32) - self.center_) / self.scale_
        scores = []
        for mean, log_prior in zip(self.means_, self.log_priors_):
            sq = (z - mean.reshape((1, -1))) ** 2
            scores.append(-0.5 * np.sum(sq / self.variance_.reshape((1, -1)), axis=1) + float(log_prior))
        return np.stack(scores, axis=1).astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.decision_function(x)
        assert self.classes_ is not None
        return self.classes_[np.argmax(scores, axis=1)].astype(np.int64)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = self.decision_function(x)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp = np.exp(scores)
        return (exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)).astype(np.float32)


@dataclass
class FilterBankLDAModel:
    """Stable FBCSP-style baseline: filter-bank spatial log-variance + diagonal LDA."""

    bands: Sequence[tuple[float, float]] = MI_FILTERBANK_BANDS
    top_k: int | None = 48
    include_channel_power: bool = True
    classifier: StandardizedDiagonalLDA = field(default_factory=StandardizedDiagonalLDA)
    selected_indices_: np.ndarray | None = None
    feature_names_: list[str] | None = None
    fs_: int | None = None
    channel_names_: list[str] | None = None

    def fit(
        self,
        eeg: np.ndarray,
        y: np.ndarray,
        fs: int,
        channel_names: Sequence[str],
    ) -> "FilterBankLDAModel":
        features, names = filterbank_spatial_logvar_features(
            eeg,
            fs,
            channel_names,
            bands=self.bands,
            include_channel_power=self.include_channel_power,
        )
        selected, indices = select_top_features(features, y, self.top_k)
        self.classifier.fit(selected, y)
        self.selected_indices_ = indices
        self.feature_names_ = [names[int(idx)] for idx in indices]
        self.fs_ = int(fs)
        self.channel_names_ = list(channel_names)
        return self

    def transform(self, eeg: np.ndarray) -> np.ndarray:
        if self.selected_indices_ is None or self.fs_ is None or self.channel_names_ is None:
            raise RuntimeError("Model is not fitted")
        features, _ = filterbank_spatial_logvar_features(
            eeg,
            self.fs_,
            self.channel_names_,
            bands=self.bands,
            include_channel_power=self.include_channel_power,
        )
        return features[:, self.selected_indices_]

    def predict(self, eeg: np.ndarray) -> np.ndarray:
        return self.classifier.predict(self.transform(eeg))

    def predict_proba(self, eeg: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self.transform(eeg))
