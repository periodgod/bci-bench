"""Sklearn machine-learning models on EEG-aware feature matrices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .features import filterbank_spatial_logvar_features, log_bandpower_features, select_top_features
from .riemann import covariance_matrices, riemannian_mean, tangent_feature_names, tangent_space


SKLEARN_BANDS = ((4.0, 8.0), (8.0, 13.0), (13.0, 20.0), (20.0, 30.0), (30.0, 40.0))
SKLEARN_TANGENT_BANDS = ((8.0, 30.0), (4.0, 40.0))


def _make_classifier(name: str, seed: int):
    try:
        from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC, LinearSVC
        from sklearn.tree import DecisionTreeClassifier
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("scikit-learn is required for --model ml-suite") from exc

    if name == "linear_svm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(C=0.2, class_weight="balanced", dual="auto", max_iter=6000, random_state=seed),
        )
    if name == "rbf_svm":
        return make_pipeline(
            StandardScaler(),
            SVC(C=1.0, gamma="scale", class_weight="balanced", random_state=seed),
        )
    if name == "logreg_l2":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.35,
                class_weight="balanced",
                max_iter=4000,
                solver="lbfgs",
                random_state=seed,
            ),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=240,
            max_depth=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=320,
            max_depth=6,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    if name == "hist_gradient":
        return HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=120,
            max_leaf_nodes=12,
            l2_regularization=1.0,
            random_state=seed,
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
        )
    if name == "ada_boost":
        return AdaBoostClassifier(
            n_estimators=120,
            learning_rate=0.35,
            random_state=seed,
        )
    if name == "gaussian_nb":
        return make_pipeline(
            StandardScaler(),
            GaussianNB(var_smoothing=1e-1),
        )
    raise ValueError(f"Unknown sklearn classifier: {name}")


@dataclass
class EEGFeatureSklearnModel:
    """Sklearn classifier on split-local EEG features and feature selection."""

    classifier_name: str
    feature_mode: str = "full"  # fixed, tangent, full
    top_k: int | None = 128
    seed: int = 42
    shrinkage: float = 0.1
    classifier_: object | None = None
    selected_indices_: np.ndarray | None = None
    feature_names_: list[str] | None = None
    fs_: int | None = None
    channel_names_: list[str] | None = None
    references_: list[np.ndarray] = field(default_factory=list)

    def _fixed_features(self, eeg: np.ndarray, fs: int, channel_names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
        bandpower, bandpower_names = log_bandpower_features(
            eeg,
            fs,
            bands=SKLEARN_BANDS,
            car=True,
            channel_names=channel_names,
        )
        spatial, spatial_names = filterbank_spatial_logvar_features(
            eeg,
            fs,
            channel_names,
            bands=SKLEARN_BANDS,
            include_channel_power=True,
        )
        return np.concatenate([bandpower, spatial], axis=1).astype(np.float32), bandpower_names + spatial_names

    def _tangent_features(self, eeg: np.ndarray, fs: int, fit: bool) -> tuple[np.ndarray, list[str]]:
        if self.channel_names_ is None:
            raise RuntimeError("channel_names_ must be set before tangent features")
        parts: list[np.ndarray] = []
        names: list[str] = []
        if fit:
            self.references_ = []
        for band_idx, band in enumerate(SKLEARN_TANGENT_BANDS):
            covs = covariance_matrices(eeg, fs, band=band, shrinkage=self.shrinkage)
            if fit:
                reference = riemannian_mean(covs)
                self.references_.append(reference)
            else:
                reference = self.references_[band_idx]
            parts.append(tangent_space(covs, reference))
            names.extend(tangent_feature_names(self.channel_names_, prefix=f"{band[0]:g}-{band[1]:g}Hz:ts:"))
        return np.concatenate(parts, axis=1).astype(np.float32), names

    def _features(self, eeg: np.ndarray, fs: int, fit: bool) -> tuple[np.ndarray, list[str]]:
        if self.channel_names_ is None:
            raise RuntimeError("Model channel_names_ are not initialized")
        mode = str(self.feature_mode).lower()
        if mode == "fixed":
            return self._fixed_features(eeg, fs, self.channel_names_)
        if mode == "tangent":
            return self._tangent_features(eeg, fs, fit=fit)
        if mode != "full":
            raise ValueError(f"Unknown feature_mode: {self.feature_mode}")
        fixed, fixed_names = self._fixed_features(eeg, fs, self.channel_names_)
        tangent, tangent_names = self._tangent_features(eeg, fs, fit=fit)
        return np.concatenate([fixed, tangent], axis=1).astype(np.float32), fixed_names + tangent_names

    def fit(
        self,
        eeg: np.ndarray,
        y: np.ndarray,
        fs: int,
        channel_names: Sequence[str],
    ) -> "EEGFeatureSklearnModel":
        labels = np.asarray(y, dtype=np.int64)
        self.fs_ = int(fs)
        self.channel_names_ = list(channel_names)
        features, names = self._features(eeg, self.fs_, fit=True)
        selected, indices = select_top_features(features, labels, self.top_k)
        self.classifier_ = _make_classifier(self.classifier_name, self.seed)
        self.classifier_.fit(selected, labels)
        self.selected_indices_ = indices
        self.feature_names_ = [names[int(index)] for index in indices]
        return self

    def transform(self, eeg: np.ndarray) -> np.ndarray:
        if self.fs_ is None or self.selected_indices_ is None:
            raise RuntimeError("Model is not fitted")
        features, _ = self._features(eeg, self.fs_, fit=False)
        return features[:, self.selected_indices_]

    def predict(self, eeg: np.ndarray) -> np.ndarray:
        if self.classifier_ is None:
            raise RuntimeError("Model is not fitted")
        return np.asarray(self.classifier_.predict(self.transform(eeg)), dtype=np.int64)
