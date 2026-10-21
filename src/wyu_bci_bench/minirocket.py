"""MiniRocket-style deterministic convolutional features for EEG epochs.

This is a lightweight, dependency-free MiniRocket-inspired transform. It keeps
the core idea useful for small EEG datasets: many fixed zero-sum convolutional
kernels, train-fold-only bias quantiles, PPV features, and a simple linear
classifier. It is not a drop-in reproduction of the original sktime MiniRocket
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .features import select_top_features
from .models import StandardizedDiagonalLDA
from .signal import bandpass_fft, common_average_reference, ensure_3d_epochs


@dataclass(frozen=True)
class MiniRocketKernel:
    """固定卷积核：参与卷积的通道索引、时间膨胀率和核权重。"""
    channels: np.ndarray
    dilation: int
    weights: np.ndarray


def _make_dilations(n_times: int, kernel_size: int, count: int) -> np.ndarray:
    """按几何间隔选取覆盖短/长时间尺度的 dilation 值。"""
    max_dilation = max(1, int((int(n_times) - 1) // max(kernel_size - 1, 1)))
    if max_dilation <= 1:
        return np.asarray([1], dtype=np.int64)
    raw = np.unique(np.round(np.geomspace(1, max_dilation, num=max(4, min(count, 32)))).astype(np.int64))
    return raw[raw >= 1]


def _make_kernels(
    n_channels: int,
    n_times: int,
    n_kernels: int,
    seed: int,
    kernel_size: int = 9,
    max_channels_per_kernel: int = 4,
) -> list[MiniRocketKernel]:
    """用给定种子生成稀疏通道、零和权重的确定性卷积核集合。"""
    rng = np.random.default_rng(seed)
    dilations = _make_dilations(n_times, kernel_size, n_kernels)
    kernels: list[MiniRocketKernel] = []
    for idx in range(int(n_kernels)):
        dilation = int(dilations[idx % len(dilations)])
        channel_count = int(rng.integers(1, min(max_channels_per_kernel, n_channels) + 1))
        channels = np.sort(rng.choice(n_channels, size=channel_count, replace=False)).astype(np.int64)
        weights = -np.ones(kernel_size, dtype=np.float32)
        positive = rng.choice(kernel_size, size=3, replace=False)
        weights[positive] = 2.0
        weights = weights / np.sqrt(float(kernel_size * channel_count))
        kernels.append(MiniRocketKernel(channels=channels, dilation=dilation, weights=weights.astype(np.float32)))
    return kernels


def _kernel_response(x: np.ndarray, kernel: MiniRocketKernel) -> np.ndarray:
    """逐通道、逐 tap 计算有效卷积响应，不做边界 padding。"""
    epochs = ensure_3d_epochs(x)
    kernel_size = int(kernel.weights.size)
    dilation = int(kernel.dilation)
    out_len = epochs.shape[2] - (kernel_size - 1) * dilation
    if out_len <= 0:
        raise ValueError(f"Kernel dilation={dilation} is too large for n_times={epochs.shape[2]}")
    response = np.zeros((epochs.shape[0], out_len), dtype=np.float32)
    for channel in kernel.channels:
        channel_data = epochs[:, int(channel), :]
        for tap, weight in enumerate(kernel.weights):
            start = tap * dilation
            response += float(weight) * channel_data[:, start : start + out_len]
    return response


@dataclass
class MiniRocketLDAModel:
    """轻量 MiniRocket 风格 EEG 特征加对角 LDA。

    训练时确定滤波/标准化参数、随机固定卷积核的响应分位数 bias，以及标签相关的
    特征选择；应用到验证/测试数据时复用这些训练折状态。它借鉴 MiniRocket 思路，
    不是 sktime 官方实现的逐项复刻。
    """

    band: tuple[float, float] = (8.0, 30.0)
    n_kernels: int = 128
    top_k: int | None = 128
    seed: int = 42
    kernel_size: int = 9
    max_channels_per_kernel: int = 4
    classifier: StandardizedDiagonalLDA = field(default_factory=StandardizedDiagonalLDA)
    kernels_: list[MiniRocketKernel] | None = None
    biases_: np.ndarray | None = None
    selected_indices_: np.ndarray | None = None
    feature_names_: list[str] | None = None
    fs_: int | None = None
    center_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def _prepare_fit(self, eeg: np.ndarray, fs: int) -> np.ndarray:
        """对训练 EEG 做 CAR、频带滤波并按训练数据逐通道标准化。"""
        x = common_average_reference(eeg)
        x = bandpass_fft(x, fs, float(self.band[0]), float(self.band[1]))
        self.center_ = np.mean(x, axis=(0, 2), keepdims=True).astype(np.float32)
        self.scale_ = np.maximum(np.std(x, axis=(0, 2), keepdims=True), 1e-6).astype(np.float32)
        return ((x - self.center_) / self.scale_).astype(np.float32)

    def _prepare_apply(self, eeg: np.ndarray, fs: int) -> np.ndarray:
        """对新 EEG 复用训练得到的中心与尺度，不重新估计标准化统计量。"""
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Model is not fitted")
        x = common_average_reference(eeg)
        x = bandpass_fft(x, fs, float(self.band[0]), float(self.band[1]))
        return ((x - self.center_) / self.scale_).astype(np.float32)

    def _fit_biases(self, x: np.ndarray) -> np.ndarray:
        """仅用训练折卷积响应的分位数确定每个核的阈值。"""
        if self.kernels_ is None:
            raise RuntimeError("Kernels are not initialized")
        biases = []
        for idx, kernel in enumerate(self.kernels_):
            response = _kernel_response(x, kernel).reshape(-1)
            quantile = 0.1 + 0.8 * ((idx * 0.6180339887498949) % 1.0)
            biases.append(float(np.quantile(response, quantile)))
        return np.asarray(biases, dtype=np.float32)

    def _features_from_prepared(self, x: np.ndarray) -> np.ndarray:
        """每个核提取 PPV（正响应比例）和最大中心化响应两列特征。"""
        if self.kernels_ is None or self.biases_ is None:
            raise RuntimeError("Model is not fitted")
        parts: list[np.ndarray] = []
        for kernel, bias in zip(self.kernels_, self.biases_):
            response = _kernel_response(x, kernel)
            centered = response - float(bias)
            ppv = np.mean(centered > 0.0, axis=1)
            max_response = np.max(centered, axis=1)
            parts.append(ppv.astype(np.float32))
            parts.append(max_response.astype(np.float32))
        return np.stack(parts, axis=1).astype(np.float32)

    def fit(
        self,
        eeg: np.ndarray,
        y: np.ndarray,
        fs: int,
        channel_names: Sequence[str] | None = None,
    ) -> "MiniRocketLDAModel":
        """拟合卷积变换状态、训练折特征选择器和 LDA 分类器。"""
        del channel_names
        x = self._prepare_fit(eeg, fs)
        self.fs_ = int(fs)
        self.kernels_ = _make_kernels(
            n_channels=x.shape[1],
            n_times=x.shape[2],
            n_kernels=int(self.n_kernels),
            seed=int(self.seed),
            kernel_size=int(self.kernel_size),
            max_channels_per_kernel=int(self.max_channels_per_kernel),
        )
        self.biases_ = self._fit_biases(x)
        features = self._features_from_prepared(x)
        selected, indices = select_top_features(features, y, self.top_k)
        self.classifier.fit(selected, y)
        self.selected_indices_ = indices
        names = []
        for idx in range(int(self.n_kernels)):
            names.append(f"k{idx:03d}:ppv")
            names.append(f"k{idx:03d}:max")
        self.feature_names_ = [names[int(index)] for index in indices]
        return self

    def transform(self, eeg: np.ndarray) -> np.ndarray:
        """按训练合同转换 EEG，并只返回已选特征列。"""
        if self.fs_ is None or self.selected_indices_ is None:
            raise RuntimeError("Model is not fitted")
        x = self._prepare_apply(eeg, self.fs_)
        return self._features_from_prepared(x)[:, self.selected_indices_]

    def predict(self, eeg: np.ndarray) -> np.ndarray:
        """返回输入 EEG 的整数类别预测。"""
        return self.classifier.predict(self.transform(eeg))

    def predict_proba(self, eeg: np.ndarray) -> np.ndarray:
        """返回输入 EEG 的类别概率。"""
        return self.classifier.predict_proba(self.transform(eeg))
