"""面向试次优先 EEG 数组的基础信号处理工具。

多数函数输入/输出形状为 ``(N,C,T)``：试次或窗口、通道、采样点。这里的函数
只处理数值数组，不会自动推断微伏/伏单位；滤波频率使用 Hz，采样率也必须由调用方
明确传入。处理先后顺序和参数属于实验协议的一部分，应由上层流程记录。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Sequence

import numpy as np


BANDS = OrderedDict(
    [
        ("delta", (1.0, 4.0)),
        ("theta", (4.0, 8.0)),
        ("mu", (8.0, 12.0)),
        ("low_beta", (12.0, 20.0)),
        ("high_beta", (20.0, 30.0)),
        ("gamma_low", (30.0, 45.0)),
    ]
)

MI_FILTERBANK_BANDS = (
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 30.0),
)

MOTOR_CHANNELS = (
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
)

MOTOR11_CHANNELS = (
    "FC3",
    "FCz",
    "FC4",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "CP3",
    "CPz",
    "CP4",
)

CENTRAL3_CHANNELS = ("C3", "Cz", "C4")


def normalize_channel_key(name: str) -> str:
    """规范通道名用于匹配：转大写并移除空格、连字符等非字母数字字符。"""
    return "".join(ch for ch in str(name).upper() if ch.isalnum())


def select_channel_indices(
    channel_names: Sequence[str],
    requested: Iterable[str],
    min_channels: int = 1,
) -> tuple[np.ndarray, list[str]]:
    """按 requested 的顺序从原通道中查找索引，并返回实际原始通道名。

    缺失通道会被跳过；若找到数量少于 ``min_channels`` 则报错。顺序按 requested
    保留，因此它同时定义了下游数组的通道顺序。
    """
    lookup = {normalize_channel_key(name): idx for idx, name in enumerate(channel_names)}
    selected: list[int] = []
    names: list[str] = []
    for name in requested:
        key = normalize_channel_key(name)
        if key in lookup:
            selected.append(lookup[key])
            names.append(str(channel_names[lookup[key]]))
    if len(selected) < min_channels:
        raise ValueError(f"Only found {len(selected)} requested channels; need at least {min_channels}")
    return np.asarray(selected, dtype=np.int64), names


def ensure_3d_epochs(eeg: np.ndarray, name: str = "eeg") -> np.ndarray:
    """把单个 ``(C,T)`` 试次提升为 ``(1,C,T)``，并拒绝其他维度。"""
    arr = np.asarray(eeg, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (trials, channels, samples)")
    return arr


def common_average_reference(eeg: np.ndarray) -> np.ndarray:
    """逐试次、逐采样点减去所有通道均值（CAR），不跨试次估计参考。"""
    x = ensure_3d_epochs(eeg)
    return (x - np.mean(x, axis=1, keepdims=True)).astype(np.float32)


def standardize_epochs(eeg: np.ndarray) -> np.ndarray:
    """每个试次的每个通道沿时间独立做 z-score，标准差下限防止除零。"""
    x = ensure_3d_epochs(eeg)
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return ((x - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def bandpass_fft(eeg: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    """用 FFT 硬频率掩码保留 [low, high] Hz，输入输出形状一致。

    这是对有限长试次的频域截断，边界可能产生振铃，不等价于具有过渡带的 IIR/FIR
    滤波器；不要在未核对协议时把它替换成另一种滤波器。
    """
    x = ensure_3d_epochs(eeg)
    fs = float(fs)
    low = max(0.0, float(low))
    high = min(float(high), fs / 2.0)
    if high <= low or x.shape[-1] < 2:
        return x.astype(np.float32, copy=True)
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / fs)
    spectrum = np.fft.rfft(x, axis=-1)
    mask = (freqs >= low) & (freqs <= high)
    spectrum *= mask.reshape((1, 1, -1))
    return np.fft.irfft(spectrum, n=x.shape[-1], axis=-1).astype(np.float32)


def compute_psd(eeg: np.ndarray, fs: float, n_fft: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """返回 Hann 窗去均值后的单边频率轴和功率谱密度数组。"""
    x = ensure_3d_epochs(eeg)
    samples = int(x.shape[-1])
    n_fft = int(n_fft or samples)
    if n_fft < samples:
        n_fft = samples
    window = np.hanning(samples).astype(np.float32)
    centered = x - np.mean(x, axis=-1, keepdims=True)
    tapered = centered * window.reshape((1, 1, -1))
    spectrum = np.fft.rfft(tapered, n=n_fft, axis=-1)
    denom = max(float(fs) * float(np.sum(window * window)), 1e-12)
    psd = (np.abs(spectrum) ** 2 / denom).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))
    return freqs.astype(np.float32), psd


def bandpower_from_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: dict[str, tuple[float, float]] | Sequence[tuple[float, float]] = BANDS,
) -> tuple[list[str], np.ndarray]:
    """对指定频带内 PSD 按频率积分，返回频带名和功率特征。

    输出最后一维对应 bands 顺序；没有频点落入的窄频带记为零。区间端点当前
    均包含在掩码中，频带相邻时端点频点可能同时进入两带。
    """
    freq_arr = np.asarray(freqs, dtype=np.float32)
    power = np.asarray(psd, dtype=np.float32)
    if isinstance(bands, dict):
        names = list(bands.keys())
        ranges = list(bands.values())
    else:
        ranges = list(bands)
        names = [f"{low:g}-{high:g}Hz" for low, high in ranges]
    values = []
    for low, high in ranges:
        mask = (freq_arr >= float(low)) & (freq_arr <= float(high))
        if not np.any(mask):
            values.append(np.zeros(power.shape[:-1], dtype=np.float32))
        else:
            integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            values.append(integrate(power[..., mask], freq_arr[mask], axis=-1).astype(np.float32))
    return names, np.stack(values, axis=-1).astype(np.float32)


def crop_to_min_samples(arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
    """把一组数组沿最后一维截到共同最短长度，不进行补零或重采样。"""
    min_samples = min(int(np.asarray(item).shape[-1]) for item in arrays)
    return [np.asarray(item)[..., :min_samples] for item in arrays]
