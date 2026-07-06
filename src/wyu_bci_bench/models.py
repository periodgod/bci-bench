"""Classical benchmark models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .features import filterbank_spatial_logvar_features, select_top_features
from .signal import MI_FILTERBANK_BANDS


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(x, dtype=np.float32)
    center = np.mean(features, axis=0)
    scale = np.maximum(np.std(features, axis=0), 1e-6)
    return ((features - center) / scale).astype(np.float32), center.astype(np.float32), scale.astype(np.float32)


def _standardize_apply(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float32) - center) / np.maximum(scale, 1e-6)).astype(np.float32)


class NearestCentroidClassifier:
    """Standardized nearest-class-centroid classifier."""

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NearestCentroidClassifier":
        z, center, scale = _standardize_fit(x)
        labels = np.asarray(y, dtype=np.int64)
        classes = np.unique(labels)
        self.classes_ = classes
        self.center_ = center
        self.scale_ = scale
        self.centroids_ = np.stack([np.mean(z[labels == label], axis=0) for label in classes], axis=0).astype(np.float32)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.center_ is None or self.scale_ is None or self.centroids_ is None:
            raise RuntimeError("Model is not fitted")
        z = _standardize_apply(x, self.center_, self.scale_)
        distances = np.sum((z[:, None, :] - self.centroids_[None, :, :]) ** 2, axis=2)
        return self.classes_[np.argmin(distances, axis=1)].astype(np.int64)


class DiagonalGaussianNB:
    """Gaussian naive Bayes with per-class diagonal variance."""

    def __init__(self, variance_floor: float = 1e-3) -> None:
        self.variance_floor = float(variance_floor)
        self.classes_: np.ndarray | None = None
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.variances_: np.ndarray | None = None
        self.log_priors_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DiagonalGaussianNB":
        z, center, scale = _standardize_fit(x)
        labels = np.asarray(y, dtype=np.int64)
        classes = np.unique(labels)
        means = []
        variances = []
        priors = []
        for label in classes:
            subset = z[labels == label]
            means.append(np.mean(subset, axis=0))
            variances.append(np.maximum(np.var(subset, axis=0), self.variance_floor))
            priors.append(float(len(subset)) / max(float(len(labels)), 1.0))
        self.classes_ = classes
        self.center_ = center
        self.scale_ = scale
        self.means_ = np.stack(means, axis=0).astype(np.float32)
        self.variances_ = np.stack(variances, axis=0).astype(np.float32)
        self.log_priors_ = np.log(np.maximum(np.asarray(priors, dtype=np.float32), 1e-12))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.center_ is None or self.scale_ is None:
            raise RuntimeError("Model is not fitted")
        assert self.means_ is not None
        assert self.variances_ is not None
        assert self.log_priors_ is not None
        z = _standardize_apply(x, self.center_, self.scale_)
        scores = []
        for mean, variance, log_prior in zip(self.means_, self.variances_, self.log_priors_):
            scores.append(-0.5 * np.sum(np.log(variance) + (z - mean) ** 2 / variance, axis=1) + float(log_prior))
        stacked = np.stack(scores, axis=1)
        return self.classes_[np.argmax(stacked, axis=1)].astype(np.int64)


class KNearestNeighborsClassifier:
    """Small-data KNN classifier with deterministic tie breaking."""

    def __init__(self, k: int = 3) -> None:
        self.k = int(k)
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.train_x_: np.ndarray | None = None
        self.train_y_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNearestNeighborsClassifier":
        z, center, scale = _standardize_fit(x)
        self.center_ = center
        self.scale_ = scale
        self.train_x_ = z
        self.train_y_ = np.asarray(y, dtype=np.int64)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None or self.train_x_ is None or self.train_y_ is None:
            raise RuntimeError("Model is not fitted")
        z = _standardize_apply(x, self.center_, self.scale_)
        preds = []
        classes = sorted(np.unique(self.train_y_).tolist())
        for row in z:
            distances = np.sum((self.train_x_ - row.reshape((1, -1))) ** 2, axis=1)
            nearest = np.argsort(distances)[: max(1, min(self.k, len(distances)))]
            labels, counts = np.unique(self.train_y_[nearest], return_counts=True)
            best_count = np.max(counts)
            tied = [int(label) for label, count in zip(labels, counts) if int(count) == int(best_count)]
            if len(tied) == 1:
                preds.append(tied[0])
                continue
            tied_distance = {
                label: float(np.mean([distances[idx] for idx in nearest if int(self.train_y_[idx]) == label]))
                for label in tied
            }
            preds.append(min(tied, key=lambda label: (tied_distance[label], classes.index(label))))
        return np.asarray(preds, dtype=np.int64)


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
