"""慢波/直流窗口特征：每通道 ``n_segments`` 段子窗均值 + 窗内线性斜率。

常见管线会对每个窗口去均值并从 ≥0.5–1 Hz 起滤波，这部分低于 1 Hz 的偏移与漂移信息
因此被丢掉。这里保留输入的数值精度：float32 输入的子窗均值按 float32 累加。
"""

from __future__ import annotations

import numpy as np


def slow_dc_features(eeg: np.ndarray, n_segments: int = 8) -> np.ndarray:
    """返回 ``(N, C*n_segments + C)``：先按通道展开子窗均值，再接每通道斜率。"""
    x = np.asarray(eeg)
    n, c, t = x.shape
    if t % n_segments:
        raise ValueError("window length must be divisible by n_segments")
    means = x.reshape(n, c, n_segments, t // n_segments).mean(-1)
    tt = np.arange(t) - (t - 1) / 2
    slope = (x * tt).sum(-1) / (tt * tt).sum()
    return np.concatenate([means.reshape(n, -1), slope], axis=1).astype(np.float32)
