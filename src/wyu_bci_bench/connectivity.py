"""频带内相位滞后连接与 log-Euclidean 滤波器组协方差向量。

虚部相干（imaginary coherency, Nolte 2004）只保留有相位滞后的耦合，零延迟的
体积传导和共同参考对它的贡献为零，因此对参考电极选择不敏感。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .signal import bandpass_fft, common_average_reference


def imaginary_coherency(eeg: np.ndarray, fs: float, band: tuple[float, float],
                        segment: slice | None = None) -> np.ndarray:
    """每个窗口在频带内跨频率平均的虚部相干，返回上三角（不含对角）``(N, C*(C-1)/2)``。"""
    x = np.asarray(eeg, dtype=np.float64)
    if segment is not None:
        x = x[..., segment]
    x = x - x.mean(-1, keepdims=True)
    t = x.shape[-1]
    win = np.hanning(t)
    spec = np.fft.rfft(x * win, axis=-1)
    freqs = np.fft.rfftfreq(t, 1.0 / fs)
    sel = (freqs >= band[0]) & (freqs < band[1])
    if sel.sum() < 2:
        raise ValueError("band must cover at least two frequency bins")
    s = spec[..., sel]
    cross = np.einsum("ncf,ndf->ncd", s, np.conj(s))
    power = np.real(np.einsum("ncc->nc", cross))
    denom = np.sqrt(np.clip(power[:, :, None] * power[:, None, :], 1e-30, None))
    icoh = np.imag(cross) / denom
    iu = np.triu_indices(x.shape[1], 1)
    return icoh[:, iu[0], iu[1]].astype(np.float32)


def logeuclid_filterbank(eeg: np.ndarray, fs: float, bands: Sequence[tuple[float, float]],
                         segments: Sequence[slice], car: bool = True, shrinkage: float = 0.1) -> np.ndarray:
    """频带×时间段协方差的 log-Euclidean 向量（上三角，非对角乘 √2），按 band 外层、segment 内层拼接。

    整窗先（可选）CAR 并带通，再切段、去均值、估计协方差并向迹/通道数×单位阵收缩。
    """
    x = common_average_reference(np.asarray(eeg, dtype=np.float64)) if car else np.asarray(eeg, dtype=np.float64)
    c = x.shape[1]
    iu = np.triu_indices(c)
    w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    blocks = []
    for band in bands:
        xb = bandpass_fft(x, fs, band[0], band[1])
        for sl in segments:
            seg = xb[..., sl]
            seg = seg - seg.mean(-1, keepdims=True)
            cov = np.einsum("nct,ndt->ncd", seg, seg) / seg.shape[-1]
            tr = np.trace(cov, axis1=1, axis2=2) / c
            cov = (1 - shrinkage) * cov + shrinkage * tr[:, None, None] * np.eye(c)
            vals, vecs = np.linalg.eigh(cov)
            logm = np.einsum("nij,nj,nkj->nik", vecs, np.log(np.clip(vals, 1e-10, None)), vecs)
            blocks.append((logm[:, iu[0], iu[1]] * w).astype(np.float32))
    return np.concatenate(blocks, axis=1)
