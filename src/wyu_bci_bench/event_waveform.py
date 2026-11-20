"""Low-frequency event-related EEG features for cross-subject experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def low_frequency_waveforms(
    epochs: np.ndarray,
    *,
    fs: float = 120.0,
    cutoff: float = 6.0,
    downsample: int = 12,
    baseline_samples: int = 12,
    prefix_seconds: float | None = None,
) -> np.ndarray:
    """CAR, zero-phase low-pass, baseline-correct and decimate windowed EEG.

    Input and output use (trials, channels, samples). When ``prefix_seconds``
    is set, the input is truncated before filtering so future samples cannot
    affect a feedback-free prefix through the zero-phase filter.
    """
    x = np.asarray(epochs, dtype=np.float64)
    if x.ndim != 3 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("epochs must be a finite (trials, channels, samples) array")
    if fs <= 0 or cutoff <= 0 or cutoff >= fs / 2 or downsample < 1:
        raise ValueError("invalid sampling or filter parameters")
    if baseline_samples < 1:
        raise ValueError("baseline_samples must be positive")
    if prefix_seconds is not None:
        stop = int(round(prefix_seconds * fs))
        if stop < baseline_samples or stop > x.shape[-1]:
            raise ValueError("prefix_seconds selects too few or too many samples")
        x = x[..., :stop]
    x = x - x.mean(axis=1, keepdims=True)
    try:
        from scipy.signal import butter, sosfiltfilt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("install scipy to use low-frequency waveform features") from exc
    sos = butter(4, cutoff, fs=fs, output="sos")
    filtered = sosfiltfilt(sos, x, axis=-1)
    filtered -= filtered[..., :baseline_samples].mean(axis=-1, keepdims=True)
    return filtered[..., 6::downsample].astype(np.float32)


def _validate(X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 3 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError("X must be finite (trials, channels, samples)")
    if y is not None and (np.asarray(y).ndim != 1 or len(y) != len(x)):
        raise ValueError("y must contain one label per trial")
    return x


def _oas_covariance(samples: np.ndarray) -> np.ndarray:
    from sklearn.covariance import oas

    cov, _ = oas(np.asarray(samples, dtype=np.float64), assume_centered=False)
    return cov


def _class_filters(X: np.ndarray, y: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.linalg import eigh

    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("at least two classes are required")
    channels = X.shape[1]
    if not 1 <= n_components <= channels:
        raise ValueError("n_components must be between 1 and the channel count")
    templates = np.stack([X[y == cls].mean(axis=0) for cls in classes])
    residuals = np.concatenate(
        [(X[y == cls] - templates[i]).transpose(0, 2, 1).reshape(-1, channels)
         for i, cls in enumerate(classes)],
        axis=0,
    )
    noise_cov = _oas_covariance(residuals)
    noise_cov += max(float(np.trace(noise_cov)) / channels, 1.0) * 1e-8 * np.eye(channels)
    filters = []
    for template in templates:
        signal_cov = template @ template.T / max(template.shape[-1], 1)
        values, vectors = eigh(signal_cov, noise_cov, check_finite=False)
        filters.append(vectors[:, np.argsort(values)[::-1][:n_components]].T)
    return templates, np.concatenate(filters, axis=0)


def _oas_epoch_covariances(X: np.ndarray) -> np.ndarray:
    from sklearn.covariance import oas

    n, channels, _ = X.shape
    covs = np.empty((n, channels, channels), dtype=np.float64)
    for i, trial in enumerate(X):
        covs[i], _ = oas(trial.T, assume_centered=False)
    return covs


def _tangent_features(covs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    from .riemann import riemannian_mean, spd_log, spd_power

    ref = riemannian_mean(covs) if reference is None else reference
    invsqrt = spd_power(ref, -0.5)
    upper = np.triu_indices(covs.shape[1])
    scale = np.where(upper[0] == upper[1], 1.0, np.sqrt(2.0))
    out = []
    for cov in covs:
        whitened = invsqrt @ cov @ invsqrt
        out.append(spd_log(whitened)[upper] * scale)
    return np.asarray(out, dtype=np.float32)


@dataclass
class XDawnTangentLogistic:
    """Class-wise xDAWN filters, OAS covariances, tangent space and Logistic."""

    n_components_per_class: int = 4
    C: float = 0.003
    filters_: np.ndarray | None = None
    classes_: np.ndarray | None = None
    reference_: np.ndarray | None = None
    scaler_: object | None = None
    classifier_: object | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XDawnTangentLogistic":
        x = _validate(X, y)
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)
        _, self.filters_ = _class_filters(x, labels, self.n_components_per_class)
        projected = np.einsum("kc,nct->nkt", self.filters_, x, optimize=True)
        covs = _oas_epoch_covariances(projected)
        from .riemann import riemannian_mean

        self.reference_ = riemannian_mean(covs)
        features = _tangent_features(covs, self.reference_)
        self._fit_classifier(features, labels)
        return self

    def _fit_classifier(self, features: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler_ = StandardScaler().fit(features)
        self.classifier_ = LogisticRegression(
            C=self.C, class_weight="balanced", max_iter=3000, random_state=33
        ).fit(self.scaler_.transform(features), y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.filters_ is None or self.reference_ is None:
            raise RuntimeError("call fit before transform")
        x = _validate(X)
        if x.shape[1] != self.filters_.shape[1]:
            raise ValueError("channel count differs from fitted filters")
        projected = np.einsum("kc,nct->nkt", self.filters_, x, optimize=True)
        return _tangent_features(_oas_epoch_covariances(projected), self.reference_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classifier_ is None or self.scaler_ is None:
            raise RuntimeError("call fit before predict_proba")
        return self.classifier_.predict_proba(self.scaler_.transform(self.transform(X)))


@dataclass
class ClassTemplateProjection:
    """Class-specific low-rank template correlations as a compact feature."""

    n_components_per_class: int = 4
    absolute_correlation: bool = False
    templates_: np.ndarray | None = None
    components_: np.ndarray | None = None
    classes_: np.ndarray | None = None
    scaler_: object | None = None
    classifier_: object | None = None

    def _features(self, X: np.ndarray) -> np.ndarray:
        if self.templates_ is None or self.components_ is None:
            raise RuntimeError("call fit before transform")
        rows = []
        for trial in X:
            feats = []
            for template, basis in zip(self.templates_, self.components_):
                a = basis @ trial
                b = basis @ template
                a = a - a.mean(axis=1, keepdims=True)
                b = b - b.mean(axis=1, keepdims=True)
                denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
                corr = np.sum(a * b, axis=1) / np.maximum(denom, 1e-12)
                feats.extend(np.abs(corr) if self.absolute_correlation else corr)
            rows.append(feats)
        return np.asarray(rows, dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassTemplateProjection":
        x = _validate(X, y)
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)
        if len(self.classes_) < 2 or not 1 <= self.n_components_per_class <= x.shape[1]:
            raise ValueError("invalid class or component count")
        templates, components = [], []
        for cls in self.classes_:
            template = x[labels == cls].mean(axis=0)
            u, _, _ = np.linalg.svd(template, full_matrices=False)
            templates.append(template)
            components.append(u[:, :self.n_components_per_class].T)
        self.templates_ = np.stack(templates)
        self.components_ = np.stack(components)
        features = self._features(x)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler_ = StandardScaler().fit(features)
        self.classifier_ = LogisticRegression(
            C=0.003, class_weight="balanced", max_iter=3000, random_state=33
        ).fit(self.scaler_.transform(features), labels)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.classifier_ is None:
            raise RuntimeError("call fit before transform")
        x = _validate(X)
        if x.shape[1:] != self.templates_.shape[1:]:
            raise ValueError("channel or time dimensions differ from fitted templates")
        return self._features(x)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classifier_ is None or self.scaler_ is None:
            raise RuntimeError("call fit before predict_proba")
        return self.classifier_.predict_proba(self.scaler_.transform(self.transform(X)))
