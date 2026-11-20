"""Common Spatial Pattern (CSP) and filter-bank CSP classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class FilterBankCSP:
    """Learn one CSP projection per fixed band and classify log-power features.

    All CSP filters and the SVM are fitted from the supplied training epochs.
    The implementation uses MNE's maintained CSP solver rather than a local
    eigenvalue solver. Inputs follow ``(trials, channels, samples)``.
    """

    bands: Sequence[tuple[float, float]] = ((8.0, 12.0), (12.0, 16.0), (16.0, 24.0), (24.0, 30.0))
    n_components: int = 4
    csp_reg: float | str | None = "ledoit_wolf"
    classifier: str = "svm"
    csp_filters_: list[object] = field(default_factory=list, init=False)
    classifier_: object | None = field(default=None, init=False)
    classes_: np.ndarray | None = field(default=None, init=False)
    sampling_rate_: float | None = field(default=None, init=False)
    channel_count_: int | None = field(default=None, init=False)

    @staticmethod
    def _validate(X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 3 or min(values.shape) < 1:
            raise ValueError("X must have shape (trials, channels, samples)")
        if not np.isfinite(values).all():
            raise ValueError("X must contain only finite values")
        return values

    def _filter(self, X: np.ndarray) -> list[np.ndarray]:
        from scipy.signal import butter, sosfiltfilt

        if self.sampling_rate_ is None:
            raise RuntimeError("model has no sampling-rate contract")
        nyquist = self.sampling_rate_ / 2
        filtered = []
        for low, high in self.bands:
            if low <= 0 or high <= low or high >= nyquist:
                raise ValueError("bands must satisfy 0 < low < high < Nyquist")
            sos = butter(4, (low, high), btype="bandpass", fs=self.sampling_rate_, output="sos")
            try:
                filtered.append(sosfiltfilt(sos, X, axis=-1))
            except ValueError as exc:
                raise ValueError("epochs are too short for zero-phase filter padding") from exc
        if not filtered:
            raise ValueError("at least one frequency band is required")
        return filtered

    def _features(self, X: np.ndarray, fit: bool = False, y: np.ndarray | None = None) -> np.ndarray:
        from mne.decoding import CSP

        features = []
        if fit:
            self.csp_filters_ = []
        band_epochs = self._filter(X)
        for index, epochs in enumerate(band_epochs):
            if fit:
                csp = CSP(n_components=self.n_components, reg=self.csp_reg, log=True,
                          transform_into="average_power", norm_trace=False)
                csp.fit(epochs, y)
                self.csp_filters_.append(csp)
            features.append(self.csp_filters_[index].transform(epochs))
        return np.concatenate(features, axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray, sampling_rate: float) -> "FilterBankCSP":
        values = self._validate(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(values):
            raise ValueError("y must contain one label per trial")
        classes = np.unique(labels)
        if len(classes) < 2:
            raise ValueError("CSP requires at least two classes")
        if self.n_components < 1 or self.n_components > values.shape[1]:
            raise ValueError("n_components must be between one and the channel count")
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        self.sampling_rate_ = float(sampling_rate)
        self.channel_count_ = values.shape[1]
        self.classes_ = classes
        features = self._features(values, fit=True, y=labels)
        try:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("install scikit-learn to use FilterBankCSP") from exc
        if self.classifier == "svm":
            readout = SVC(kernel="linear", decision_function_shape="ovr")
        elif self.classifier == "lda":
            readout = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        else:
            raise ValueError("classifier must be 'svm' or 'lda'")
        self.classifier_ = make_pipeline(StandardScaler(), readout)
        self.classifier_.fit(features, labels)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        values = self._validate(X)
        if self.classifier_ is None or self.classes_ is None or self.channel_count_ is None:
            raise RuntimeError("call fit before decision_function")
        if values.shape[1] != self.channel_count_:
            raise ValueError("test channel count must match the fitted model")
        features = self._features(values)
        raw = np.asarray(self.classifier_.decision_function(features))
        if len(self.classes_) == 2 and raw.ndim == 1:
            raw = np.column_stack([-raw, raw])
        return raw

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classifier_ is None or self.channel_count_ is None:
            raise RuntimeError("call fit before predict")
        values = self._validate(X)
        if values.shape[1] != self.channel_count_:
            raise ValueError("test channel count must match the fitted model")
        return self.classifier_.predict(self._features(values))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Native classifier probabilities, columns ordered as ``classes_``.

        The default uncalibrated SVM deliberately does not support this method.
        """
        if self.classifier_ is None or self.channel_count_ is None:
            raise RuntimeError("call fit before predict_proba")
        if not hasattr(self.classifier_, "predict_proba"):
            raise ValueError("classifier has no calibrated probability interface; use classifier='lda'")
        values = self._validate(X)
        if values.shape[1] != self.channel_count_:
            raise ValueError("test channel count must match the fitted model")
        return self.classifier_.predict_proba(self._features(values))
