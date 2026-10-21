"""Reference-signal and trial-template baselines for SSVEP decoding.

Epochs use ``(trials, channels, samples)``. This module contains classical
CCA, filter-bank CCA and task-related component analysis (TRCA) baselines;
it does not perform event detection or infer stimulus frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _validate_epochs(X: np.ndarray) -> np.ndarray:
    epochs = np.asarray(X, dtype=np.float64)
    if epochs.ndim != 3 or min(epochs.shape) < 1:
        raise ValueError("X must have shape (trials, channels, samples) with non-empty axes")
    if not np.isfinite(epochs).all():
        raise ValueError("X must contain only finite values")
    return epochs


def reference_signals(
    sampling_rate: float,
    samples: int,
    frequencies: Sequence[float],
    harmonics: int = 3,
    phases: Sequence[float] | None = None,
) -> np.ndarray:
    """Build sin/cos reference waves with shape (classes, 2*harmonics, samples)."""
    frequencies = np.asarray(frequencies, dtype=float)
    if sampling_rate <= 0 or samples < 2 or harmonics < 1:
        raise ValueError("sampling_rate must be positive, samples >= 2, and harmonics >= 1")
    if frequencies.ndim != 1 or len(frequencies) == 0 or np.any(frequencies <= 0):
        raise ValueError("frequencies must be a non-empty sequence of positive values")
    if np.any(frequencies * harmonics >= sampling_rate / 2):
        raise ValueError("all requested harmonics must be below the Nyquist frequency")
    phase_values = np.zeros(len(frequencies)) if phases is None else np.asarray(phases, dtype=float)
    if phase_values.shape != frequencies.shape:
        raise ValueError("phases must contain one value per frequency")
    time = np.arange(samples, dtype=float) / float(sampling_rate)
    signals = []
    for frequency, phase in zip(frequencies, phase_values):
        waves = []
        for harmonic in range(1, harmonics + 1):
            angle = 2 * np.pi * harmonic * frequency * time + phase
            waves.extend((np.sin(angle), np.cos(angle)))
        signals.append(waves)
    return np.asarray(signals)


def _cca_score(signal: np.ndarray, reference: np.ndarray, ridge: float = 1e-8) -> float:
    """Largest regularized canonical correlation between channels and references."""
    x = signal - signal.mean(axis=1, keepdims=True)
    y = reference - reference.mean(axis=1, keepdims=True)
    scale_x = max(float(np.trace(x @ x.T)) / max(x.shape[0], 1), 1.0)
    scale_y = max(float(np.trace(y @ y.T)) / max(y.shape[0], 1), 1.0)
    cov_x = x @ x.T / x.shape[1] + ridge * scale_x * np.eye(x.shape[0])
    cov_y = y @ y.T / y.shape[1] + ridge * scale_y * np.eye(y.shape[0])
    cross = x @ y.T / x.shape[1]
    eig_x, vec_x = np.linalg.eigh(cov_x)
    eig_y, vec_y = np.linalg.eigh(cov_y)
    inv_x = (vec_x / np.sqrt(np.maximum(eig_x, ridge * scale_x))) @ vec_x.T
    inv_y = (vec_y / np.sqrt(np.maximum(eig_y, ridge * scale_y))) @ vec_y.T
    correlations = np.linalg.svd(inv_x @ cross @ inv_y, compute_uv=False)
    return float(np.clip(correlations[0], 0.0, 1.0))


@dataclass
class SSVEPCCA:
    """Classify trials by their canonical correlation with stimulus references."""

    frequencies: Sequence[float]
    sampling_rate: float
    harmonics: int = 3
    phases: Sequence[float] | None = None
    ridge: float = 1e-8

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        epochs = _validate_epochs(X)
        refs = reference_signals(self.sampling_rate, epochs.shape[-1], self.frequencies,
                                 self.harmonics, self.phases)
        return np.asarray([[_cca_score(epoch, ref, self.ridge) for ref in refs]
                           for epoch in epochs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.decision_function(X).argmax(axis=1)


def _filter_epochs(X: np.ndarray, sampling_rate: float, bands: Sequence[tuple[float, float]]) -> list[np.ndarray]:
    if not bands:
        raise ValueError("at least one filter band is required")
    try:
        from scipy.signal import butter, sosfiltfilt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("install scipy to use filter-bank SSVEP methods") from exc
    filtered = []
    for low, high in bands:
        if low <= 0 or high <= low or high >= sampling_rate / 2:
            raise ValueError("each filter band must satisfy 0 < low < high < Nyquist")
        sos = butter(4, (low, high), btype="bandpass", fs=sampling_rate, output="sos")
        try:
            filtered.append(sosfiltfilt(sos, X, axis=-1))
        except ValueError as exc:
            raise ValueError("epochs are too short for zero-phase filter padding") from exc
    return filtered


@dataclass
class FilterBankCCA(SSVEPCCA):
    """Filter-bank CCA with weighted squared correlations (FBCCA-style)."""

    bands: Sequence[tuple[float, float]] | None = None
    weight_exponent: float = 1.25
    weight_offset: float = 0.25

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        epochs = _validate_epochs(X)
        bands = self.bands
        if bands is None:
            upper = min(90.0, self.sampling_rate / 2 - 1.0)
            bands = tuple((low, upper) for low in (6.0, 14.0, 22.0, 30.0, 38.0) if low < upper)
        filtered = _filter_epochs(epochs, self.sampling_rate, bands)
        refs = reference_signals(self.sampling_rate, epochs.shape[-1], self.frequencies,
                                 self.harmonics, self.phases)
        scores = np.zeros((len(epochs), len(refs)), dtype=float)
        for band_index, band_epochs in enumerate(filtered, start=1):
            weight = band_index ** (-self.weight_exponent) + self.weight_offset
            for trial_index, epoch in enumerate(band_epochs):
                scores[trial_index] += weight * np.asarray(
                    [_cca_score(epoch, ref, self.ridge) ** 2 for ref in refs]
                )
        return scores


@dataclass
class TRCA:
    """Task-related component analysis for labeled, repeated SSVEP trials."""

    ridge: float = 1e-6
    filters_: list[np.ndarray] | None = None
    templates_: list[np.ndarray] | None = None
    classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TRCA":
        epochs = _validate_epochs(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(epochs):
            raise ValueError("y must contain one class label per trial")
        classes = np.unique(labels)
        if len(classes) < 2:
            raise ValueError("TRCA requires at least two classes")
        filters, templates = [], []
        for label in classes:
            trials = epochs[labels == label]
            if len(trials) < 2:
                raise ValueError(f"class {label!r} needs at least two training trials")
            trials = trials - trials.mean(axis=-1, keepdims=True)
            cross = np.zeros((epochs.shape[1], epochs.shape[1]))
            total = np.zeros_like(cross)
            for trial in trials:
                total += trial @ trial.T
            for i in range(len(trials)):
                for j in range(i + 1, len(trials)):
                    cross += trials[i] @ trials[j].T + trials[j] @ trials[i].T
            scale = max(float(np.trace(total)) / total.shape[0], 1.0)
            total += self.ridge * scale * np.eye(total.shape[0])
            try:
                from scipy.linalg import eigh
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("install scipy to use TRCA") from exc
            values, vectors = eigh(cross, total, check_finite=False)
            filters.append(vectors[:, np.argmax(values)])
            templates.append(trials.mean(axis=0))
        self.filters_ = filters
        self.templates_ = templates
        self.classes_ = classes
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        epochs = _validate_epochs(X)
        if self.filters_ is None or self.templates_ is None or self.classes_ is None:
            raise RuntimeError("call fit before decision_function")
        if epochs.shape[1:] != self.templates_[0].shape:
            raise ValueError("test epochs must match the fitted channel and sample dimensions")
        scores = np.zeros((len(epochs), len(self.filters_)))
        for row, epoch in enumerate(epochs):
            epoch = epoch - epoch.mean(axis=-1, keepdims=True)
            for col, (spatial, template) in enumerate(zip(self.filters_, self.templates_)):
                projected = spatial @ epoch
                target = spatial @ template
                if np.std(projected) > 0 and np.std(target) > 0:
                    scores[row, col] = np.corrcoef(projected, target)[0, 1]
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("call fit before predict")
        return self.classes_[self.decision_function(X).argmax(axis=1)]


def _as_mne_epochs(X: np.ndarray, sampling_rate: float, labels: np.ndarray):
    """Build the MNE Epochs contract required by MOABB's maintained SSVEP estimators."""
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("install the optional dependencies with pip install -e '.[ssvep]'") from exc
    names = [f"EEG{index + 1:03d}" for index in range(X.shape[1])]
    info = mne.create_info(names, sfreq=float(sampling_rate), ch_types="eeg", verbose=False)
    codes, encoded = np.unique(labels, return_inverse=True)
    events = np.column_stack((np.arange(len(X)) * X.shape[-1], np.zeros(len(X), dtype=int), encoded + 1))
    event_id = {f"class-{index}": index + 1 for index in range(len(codes))}
    return mne.EpochsArray(X, info, events=events, event_id=event_id, verbose=False)


@dataclass
class EnsembleTRCA:
    """Ensemble TRCA (eTRCA) using MOABB's maintained SSVEP implementation.

    ``frequencies`` maps each sorted class label to its stimulus frequency.
    That distinction matters: labels are not generally the physical flicker
    frequencies used by the filter bank.
    """

    frequencies: Sequence[float]
    sampling_rate: float
    n_fbands: int = 3
    estimator_: object | None = None
    classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleTRCA":
        epochs = _validate_epochs(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(epochs):
            raise ValueError("y must contain one class label per trial")
        classes = np.unique(labels)
        frequencies = np.asarray(self.frequencies, dtype=float)
        if len(classes) < 2 or frequencies.shape != classes.shape or np.any(frequencies <= 0):
            raise ValueError("frequencies must map every sorted class to a positive stimulus frequency")
        if not 1 <= self.n_fbands <= len(classes):
            raise ValueError("n_fbands must be between one and the number of classes")
        from moabb.pipelines.classification import SSVEP_TRCA

        frequency_labels = frequencies[np.searchsorted(classes, labels)]
        model = SSVEP_TRCA(n_fbands=self.n_fbands, is_ensemble=True)
        model.fit(_as_mne_epochs(epochs, self.sampling_rate, labels), frequency_labels)
        self.estimator_ = model
        self.classes_ = classes
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        epochs = _validate_epochs(X)
        if self.estimator_ is None or self.classes_ is None:
            raise RuntimeError("call fit before predict")
        if len(self.classes_) != len(self.frequencies):
            raise RuntimeError("fitted class/frequency mapping is inconsistent")
        prediction_events = np.zeros(len(epochs), dtype=np.int64)
        predicted_frequency = self.estimator_.predict(
            _as_mne_epochs(epochs, self.sampling_rate, prediction_events)
        )
        frequencies = np.asarray(self.frequencies, dtype=float)
        indices = np.argmin(np.abs(np.asarray(predicted_frequency)[:, None] - frequencies[None, :]), axis=1)
        return self.classes_[indices]


@dataclass
class TDCA:
    """Task-discriminant component analysis (TDCA), backed by MOABB.

    Temporal delay embedding, shared Fisher discriminant filters, and the
    filter-bank classifier are delegated to MOABB. The frequency sequence is
    matched to sorted class labels and never inferred from test labels.
    """

    frequencies: Sequence[float]
    sampling_rate: float
    n_fbands: int = 3
    n_components: int = 1
    n_delay: int = 3
    estimator_: object | None = None
    classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TDCA":
        epochs = _validate_epochs(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(epochs):
            raise ValueError("y must contain one class label per trial")
        classes = np.unique(labels)
        frequencies = np.asarray(self.frequencies, dtype=float)
        if len(classes) < 2 or frequencies.shape != classes.shape or np.any(frequencies <= 0):
            raise ValueError("frequencies must map every sorted class to a positive stimulus frequency")
        if not 1 <= self.n_fbands <= len(classes):
            raise ValueError("n_fbands must be between one and the number of classes")
        if not 1 <= self.n_delay < epochs.shape[-1] or self.n_components < 1:
            raise ValueError("n_delay must be below the epoch length and n_components must be positive")
        from moabb.pipelines.classification import SSVEP_TDCA

        frequency_labels = frequencies[np.searchsorted(classes, labels)]
        model = SSVEP_TDCA(n_fbands=self.n_fbands, n_components=self.n_components,
                           n_delay=self.n_delay, is_ensemble=True)
        model.fit(_as_mne_epochs(epochs, self.sampling_rate, labels), frequency_labels)
        self.estimator_ = model
        self.classes_ = classes
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        epochs = _validate_epochs(X)
        if self.estimator_ is None or self.classes_ is None:
            raise RuntimeError("call fit before predict")
        prediction_events = np.zeros(len(epochs), dtype=np.int64)
        predicted_frequency = self.estimator_.predict(
            _as_mne_epochs(epochs, self.sampling_rate, prediction_events)
        )
        frequencies = np.asarray(self.frequencies, dtype=float)
        indices = np.argmin(np.abs(np.asarray(predicted_frequency)[:, None] - frequencies[None, :]), axis=1)
        return self.classes_[indices]
