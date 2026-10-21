"""基于对称正定协方差矩阵的 Riemannian EEG 分类器和切空间特征。

协方差矩阵位于正定矩阵流形上；这里用特征分解定义矩阵幂、对数和指数，再构造
黎曼均值、距离与切空间。协方差收缩改善小样本数值稳定性；参考均值只可用训练折估计。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import linalg as spla

from .features import select_top_features
from .models import StandardizedDiagonalLDA
from .signal import MI_FILTERBANK_BANDS, bandpass_fft, common_average_reference, ensure_3d_epochs


def _sym(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (arr + np.swapaxes(arr, -1, -2))


def _mm(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.einsum("ij,jk->ik", np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64), optimize=False)


def _eigh_spd(matrix: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """对称化后特征分解，将过小特征值截到 eps 以保持数值正定。"""
    values, vectors = spla.eigh(_sym(matrix), check_finite=False)
    return np.maximum(values, eps), vectors


def spd_power(matrix: np.ndarray, power: float, eps: float = 1e-8) -> np.ndarray:
    """通过特征值幂计算 SPD 矩阵实数次幂。"""
    values, vectors = _eigh_spd(matrix, eps=eps)
    return _sym(_mm(vectors * (values**power).reshape((1, -1)), vectors.T))


def spd_log(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """通过特征值对数计算 SPD 矩阵对数。"""
    values, vectors = _eigh_spd(matrix, eps=eps)
    return _sym(_mm(vectors * np.log(values).reshape((1, -1)), vectors.T))


def spd_exp(matrix: np.ndarray) -> np.ndarray:
    """通过特征值指数计算对称矩阵指数。"""
    values, vectors = spla.eigh(_sym(matrix), check_finite=False)
    return _sym(_mm(vectors * np.exp(values).reshape((1, -1)), vectors.T))


def covariance_matrices(
    eeg: np.ndarray,
    fs: float,
    band: tuple[float, float] | None = None,
    shrinkage: float = 0.1,
    trace_normalize: bool = True,
    car: bool = True,
) -> np.ndarray:
    """为每个试次估计 SPD 协方差，可先 CAR/带通并向单位阵轻度收缩。"""
    x = common_average_reference(eeg) if car else ensure_3d_epochs(eeg)
    if band is not None:
        x = bandpass_fft(x, fs, band[0], band[1])
    x = x - np.mean(x, axis=-1, keepdims=True)
    trials, channels, samples = x.shape
    covs = np.empty((trials, channels, channels), dtype=np.float64)
    eye = np.eye(channels, dtype=np.float64)
    shrink = float(np.clip(shrinkage, 0.0, 1.0))
    for idx in range(trials):
        sample = x[idx].astype(np.float64)
        cov = np.einsum("ct,dt->cd", sample, sample, optimize=False) / max(samples - 1, 1)
        trace = float(np.trace(cov))
        target = (trace / max(channels, 1)) * eye
        cov = (1.0 - shrink) * cov + shrink * target
        if trace_normalize:
            cov = cov / max(float(np.trace(cov)), 1e-12)
        covs[idx] = _sym(cov + 1e-8 * eye)
    return covs


def riemannian_distance(first: np.ndarray, second: np.ndarray, eps: float = 1e-8) -> float:
    """计算仿射不变黎曼距离，即白化后协方差特征值对数的二范数。"""
    invsqrt = spd_power(first, -0.5, eps=eps)
    whitened = _mm(_mm(invsqrt, _sym(second)), invsqrt)
    values, _ = _eigh_spd(whitened, eps=eps)
    return float(np.linalg.norm(np.log(values)))


def riemannian_mean(
    covariances: np.ndarray,
    max_iter: int = 30,
    tolerance: float = 1e-6,
    eps: float = 1e-8,
) -> np.ndarray:
    """迭代计算 SPD 矩阵黎曼均值，直到更新量小于容差或达到迭代上限。"""
    covs = np.asarray(covariances, dtype=np.float64)
    if covs.ndim != 3 or covs.shape[0] == 0:
        raise ValueError("covariances must have shape (trials, channels, channels)")
    mean = _sym(np.mean(covs, axis=0))
    mean = mean + eps * np.eye(mean.shape[0], dtype=np.float64)
    for _ in range(max_iter):
        sqrt_mean = spd_power(mean, 0.5, eps=eps)
        invsqrt_mean = spd_power(mean, -0.5, eps=eps)
        tangent = np.mean([spd_log(_mm(_mm(invsqrt_mean, cov), invsqrt_mean), eps=eps) for cov in covs], axis=0)
        step = float(np.linalg.norm(tangent, ord="fro"))
        mean = _sym(_mm(_mm(sqrt_mean, spd_exp(tangent)), sqrt_mean))
        if step < tolerance:
            break
    return _sym(mean)


def tangent_space(covariances: np.ndarray, reference: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """将协方差白化到参考点、取矩阵对数并向量化上三角。"""
    covs = np.asarray(covariances, dtype=np.float64)
    channels = int(covs.shape[1])
    invsqrt = spd_power(reference, -0.5, eps=eps)
    upper = np.triu_indices(channels)
    scale = np.ones(len(upper[0]), dtype=np.float64)
    scale[upper[0] != upper[1]] = np.sqrt(2.0)
    features = np.empty((covs.shape[0], len(scale)), dtype=np.float32)
    for idx, cov in enumerate(covs):
        logm = spd_log(_mm(_mm(invsqrt, cov), invsqrt), eps=eps)
        features[idx] = (logm[upper] * scale).astype(np.float32)
    return features


def tangent_feature_names(channel_names: Sequence[str], prefix: str = "") -> list[str]:
    names = list(channel_names)
    result = []
    for row in range(len(names)):
        for col in range(row, len(names)):
            result.append(f"{prefix}{names[row]}~{names[col]}")
    return result


@dataclass
class RiemannianMDMClassifier:
    """MDM：将试次分给协方差黎曼均值最近的类别。"""

    band: tuple[float, float] = (8.0, 30.0)
    shrinkage: float = 0.1
    classes_: np.ndarray | None = None
    means_: list[np.ndarray] = field(default_factory=list)
    fs_: int | None = None

    def fit(self, eeg: np.ndarray, y: np.ndarray, fs: int) -> "RiemannianMDMClassifier":
        labels = np.asarray(y, dtype=np.int64)
        covs = covariance_matrices(eeg, fs, band=self.band, shrinkage=self.shrinkage)
        classes = np.unique(labels)
        self.classes_ = classes
        self.means_ = [riemannian_mean(covs[labels == label]) for label in classes]
        self.fs_ = int(fs)
        return self

    def predict(self, eeg: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.fs_ is None or not self.means_:
            raise RuntimeError("Model is not fitted")
        covs = covariance_matrices(eeg, self.fs_, band=self.band, shrinkage=self.shrinkage)
        distances = np.asarray(
            [[riemannian_distance(cov, mean) for mean in self.means_] for cov in covs],
            dtype=np.float64,
        )
        return self.classes_[np.argmin(distances, axis=1)].astype(np.int64)


@dataclass
class FilterBankMDMClassifier:
    """滤波器组 MDM：对多个运动想象子频带的平方黎曼距离求和后分类。"""

    bands: Sequence[tuple[float, float]] = MI_FILTERBANK_BANDS
    shrinkage: float = 0.1
    classes_: np.ndarray | None = None
    means_by_band_: list[list[np.ndarray]] = field(default_factory=list)
    fs_: int | None = None

    def fit(self, eeg: np.ndarray, y: np.ndarray, fs: int) -> "FilterBankMDMClassifier":
        labels = np.asarray(y, dtype=np.int64)
        classes = np.unique(labels)
        self.classes_ = classes
        self.means_by_band_ = []
        for band in self.bands:
            covs = covariance_matrices(eeg, fs, band=band, shrinkage=self.shrinkage)
            self.means_by_band_.append([riemannian_mean(covs[labels == label]) for label in classes])
        self.fs_ = int(fs)
        return self

    def predict(self, eeg: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.fs_ is None or not self.means_by_band_:
            raise RuntimeError("Model is not fitted")
        total: np.ndarray | None = None
        for band, means in zip(self.bands, self.means_by_band_):
            covs = covariance_matrices(eeg, self.fs_, band=band, shrinkage=self.shrinkage)
            distances = np.asarray(
                [[riemannian_distance(cov, mean) ** 2 for mean in means] for cov in covs],
                dtype=np.float64,
            )
            total = distances if total is None else total + distances
        assert total is not None
        return self.classes_[np.argmin(total, axis=1)].astype(np.int64)


@dataclass
class TangentSpaceLDAModel:
    """单频带协方差切空间嵌入后接对角 LDA；参考点由训练数据估计。"""

    band: tuple[float, float] = (8.0, 30.0)
    shrinkage: float = 0.1
    classifier: StandardizedDiagonalLDA = field(default_factory=StandardizedDiagonalLDA)
    reference_: np.ndarray | None = None
    fs_: int | None = None

    def fit(self, eeg: np.ndarray, y: np.ndarray, fs: int) -> "TangentSpaceLDAModel":
        covs = covariance_matrices(eeg, fs, band=self.band, shrinkage=self.shrinkage)
        self.reference_ = riemannian_mean(covs)
        self.classifier.fit(tangent_space(covs, self.reference_), y)
        self.fs_ = int(fs)
        return self

    def transform(self, eeg: np.ndarray) -> np.ndarray:
        if self.reference_ is None or self.fs_ is None:
            raise RuntimeError("Model is not fitted")
        covs = covariance_matrices(eeg, self.fs_, band=self.band, shrinkage=self.shrinkage)
        return tangent_space(covs, self.reference_)

    def predict(self, eeg: np.ndarray) -> np.ndarray:
        return self.classifier.predict(self.transform(eeg))


@dataclass
class FilterBankTangentSpaceLDAModel:
    """Filter-bank tangent-space covariance features with supervised feature selection."""

    bands: Sequence[tuple[float, float]] = MI_FILTERBANK_BANDS
    shrinkage: float = 0.1
    top_k: int | None = 64
    classifier: StandardizedDiagonalLDA = field(default_factory=StandardizedDiagonalLDA)
    references_: list[np.ndarray] = field(default_factory=list)
    selected_indices_: np.ndarray | None = None
    feature_names_: list[str] | None = None
    fs_: int | None = None
    channel_names_: list[str] | None = None

    def _features(self, eeg: np.ndarray, fs: int, fit: bool = False) -> tuple[np.ndarray, list[str]]:
        parts: list[np.ndarray] = []
        names: list[str] = []
        if fit:
            self.references_ = []
        if self.channel_names_ is None:
            raise RuntimeError("channel_names_ must be set before feature extraction")
        for band_idx, band in enumerate(self.bands):
            covs = covariance_matrices(eeg, fs, band=band, shrinkage=self.shrinkage)
            if fit:
                reference = riemannian_mean(covs)
                self.references_.append(reference)
            else:
                reference = self.references_[band_idx]
            prefix = f"{band[0]:g}-{band[1]:g}Hz:"
            parts.append(tangent_space(covs, reference))
            names.extend(tangent_feature_names(self.channel_names_, prefix=prefix))
        return np.concatenate(parts, axis=1).astype(np.float32), names

    def fit(
        self,
        eeg: np.ndarray,
        y: np.ndarray,
        fs: int,
        channel_names: Sequence[str],
    ) -> "FilterBankTangentSpaceLDAModel":
        self.fs_ = int(fs)
        self.channel_names_ = list(channel_names)
        features, names = self._features(eeg, self.fs_, fit=True)
        selected, indices = select_top_features(features, y, self.top_k)
        self.classifier.fit(selected, y)
        self.selected_indices_ = indices
        self.feature_names_ = [names[int(idx)] for idx in indices]
        return self

    def transform(self, eeg: np.ndarray) -> np.ndarray:
        if self.fs_ is None or self.selected_indices_ is None or self.channel_names_ is None or not self.references_:
            raise RuntimeError("Model is not fitted")
        features, _ = self._features(eeg, self.fs_, fit=False)
        return features[:, self.selected_indices_]

    def predict(self, eeg: np.ndarray) -> np.ndarray:
        return self.classifier.predict(self.transform(eeg))
