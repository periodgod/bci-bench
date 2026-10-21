"""用于运动想象基准的空间、频带和方差特征。

默认输入为 ``(N,C,T)``。这些函数只产生窗口级特征，不负责拟合分类器；凡是
依赖标签的选择（如 Fisher 排序）必须在每一个训练折内调用，不能先对全数据选特征。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .signal import MI_FILTERBANK_BANDS, bandpass_fft, common_average_reference, ensure_3d_epochs


PAIR_CHANNELS = (
    ("FC3", "FC4"),
    ("C3", "C4"),
    ("CP3", "CP4"),
    ("C1", "C2"),
    ("FC1", "FC2"),
    ("CP1", "CP2"),
)


def _channel_lookup(channel_names: Sequence[str]) -> dict[str, int]:
    """生成与 ``signal.normalize_channel_key`` 等价的本模块通道索引表。"""
    return {"".join(ch for ch in str(name).upper() if ch.isalnum()): idx for idx, name in enumerate(channel_names)}


def fixed_spatial_filters(channel_names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """构造单位通道滤波器，以及左右电极对的差分/平均滤波器。

    返回矩阵形状为 ``(空间分量数, 通道数)``。左右差分可突出侧化信息，左右平均
    表示对称活动；缺少任一电极的电极对会被跳过。
    """
    channels = len(channel_names)
    filters: list[np.ndarray] = []
    names: list[str] = []
    for idx, name in enumerate(channel_names):
        vec = np.zeros(channels, dtype=np.float32)
        vec[idx] = 1.0
        filters.append(vec)
        names.append(str(name))
    lookup = _channel_lookup(channel_names)
    for left, right in PAIR_CHANNELS:
        left_idx = lookup.get(left.upper())
        right_idx = lookup.get(right.upper())
        if left_idx is None or right_idx is None:
            continue
        contrast = np.zeros(channels, dtype=np.float32)
        contrast[left_idx] = 0.70710677
        contrast[right_idx] = -0.70710677
        filters.append(contrast)
        names.append(f"{left}-{right}")
        average = np.zeros(channels, dtype=np.float32)
        average[left_idx] = 0.70710677
        average[right_idx] = 0.70710677
        filters.append(average)
        names.append(f"{left}+{right}")
    return np.stack(filters, axis=0).astype(np.float32), names


def project_spatial_filters(eeg: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """把每个空间滤波器应用到每窗信号，输出 ``(N,滤波器数,T)``。"""
    epochs = ensure_3d_epochs(eeg)
    spatial = np.asarray(filters, dtype=np.float32)
    if spatial.ndim != 2 or spatial.shape[1] != epochs.shape[1]:
        raise ValueError("filters must have shape (components, channels)")
    return np.sum(epochs[:, None, :, :] * spatial[None, :, :, None], axis=2).astype(np.float32)


def spatial_log_variance(eeg: np.ndarray, filters: np.ndarray, normalize: bool = True) -> np.ndarray:
    """对投影信号算对数方差；可选择将同一窗口各空间分量方差归一化。"""
    projected = project_spatial_filters(eeg, filters)
    variance = np.var(projected, axis=-1)
    if normalize:
        variance = variance / np.maximum(np.sum(variance, axis=1, keepdims=True), 1e-12)
    return np.log(np.maximum(variance, 1e-12)).astype(np.float32)


def log_bandpower_features(
    eeg: np.ndarray,
    fs: float,
    bands: Sequence[tuple[float, float]] = MI_FILTERBANK_BANDS,
    car: bool = True,
    channel_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """逐频带 FFT 滤波后计算各通道对数方差，返回特征矩阵与列名。"""
    x = common_average_reference(eeg) if car else ensure_3d_epochs(eeg)
    names_for_channels = list(channel_names) if channel_names is not None else [str(idx) for idx in range(x.shape[1])]
    if len(names_for_channels) != x.shape[1]:
        names_for_channels = [str(idx) for idx in range(x.shape[1])]
    parts: list[np.ndarray] = []
    names: list[str] = []
    for low, high in bands:
        filtered = bandpass_fft(x, fs, low, high)
        variance = np.var(filtered, axis=-1)
        parts.append(np.log(np.maximum(variance, 1e-12)).astype(np.float32))
        names.extend([f"{low:g}-{high:g}Hz:{name}" for name in names_for_channels])
    return np.concatenate(parts, axis=1).astype(np.float32), names


def filterbank_spatial_logvar_features(
    eeg: np.ndarray,
    fs: float,
    channel_names: Sequence[str],
    bands: Sequence[tuple[float, float]] = MI_FILTERBANK_BANDS,
    include_channel_power: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """逐频带提取固定空间滤波器的归一化对数方差，可附加通道功率特征。"""
    x = common_average_reference(eeg)
    filters, filter_names = fixed_spatial_filters(channel_names)
    parts: list[np.ndarray] = []
    names: list[str] = []
    for low, high in bands:
        filtered = bandpass_fft(x, fs, low, high)
        spatial = spatial_log_variance(filtered, filters, normalize=True)
        parts.append(spatial)
        names.extend([f"{low:g}-{high:g}Hz:{name}" for name in filter_names])
        if include_channel_power:
            power = np.log(np.maximum(np.var(filtered, axis=-1), 1e-12)).astype(np.float32)
            parts.append(power)
            names.extend([f"{low:g}-{high:g}Hz:power:{name}" for name in channel_names])
    return np.concatenate(parts, axis=1).astype(np.float32), names


def fisher_scores(features: np.ndarray, y: np.ndarray) -> np.ndarray:
    """按类间变化相对类内变化给每个特征打分，分数越高通常越有区分力。"""
    x = np.asarray(features, dtype=np.float32)
    labels = np.asarray(y, dtype=np.int64)
    classes = np.unique(labels)
    if classes.size != 2:
        overall = np.var(x, axis=0)
        between = np.zeros(x.shape[1], dtype=np.float32)
        for label in classes:
            subset = x[labels == label]
            between += float(len(subset)) * (np.mean(subset, axis=0) - np.mean(x, axis=0)) ** 2
        return between / np.maximum(overall, 1e-6)
    first = x[labels == classes[0]]
    second = x[labels == classes[1]]
    diff = np.mean(second, axis=0) - np.mean(first, axis=0)
    within = np.var(first, axis=0) + np.var(second, axis=0)
    return (diff * diff / np.maximum(within, 1e-6)).astype(np.float32)


def select_top_features(features: np.ndarray, y: np.ndarray, top_k: int | None) -> tuple[np.ndarray, np.ndarray]:
    """按 Fisher 分数选前 k 列，并返回排序后原始列索引。

    ``top_k`` 无效或不小于总列数时原样返回。由于使用了标签 y，必须只在训练折
    计算索引，再将相同索引应用到该折验证/测试数据。
    """
    if top_k is None or top_k <= 0 or top_k >= features.shape[1]:
        idx = np.arange(features.shape[1], dtype=np.int64)
    else:
        scores = fisher_scores(features, y)
        idx = np.sort(np.argsort(scores)[-int(top_k) :]).astype(np.int64)
    return np.asarray(features)[:, idx], idx
