"""Fixed time-frequency "math prior" features (log/relative band power + Hjorth).

Ported from ``proteus_update1/differentiable_feature_module_loso.py``. With the
default arguments the output is the 1394-D vector (2 half-windows x 41 channels
x 17 statistics) used by M0/M1/M2 and the 2026-10-19 math-prior probe.
Requires PyTorch.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - torch is optional for the numpy-only toolkit
    torch = None
    nn = None

PRIOR_BANDS = ((1., 4.), (4., 8.), (8., 13.), (13., 20.), (20., 30.), (30., 45.), (45., 60.))


def _require_torch() -> None:
    if torch is None:
        raise ImportError("wyu_bci_bench.priors needs PyTorch: pip install -e .[ml]")


def prior_feature_dim(n_channels: int = 41, n_segments: int = 2, n_bands: int = len(PRIOR_BANDS)) -> int:
    """按拼接合同计算特征长度：每通道每段包含频带功率、相对功率和 3 个 Hjorth 值。"""
    return n_segments * n_channels * (2 * n_bands + 3)


if nn is not None:

    class MathPriorFeatures(nn.Module):
        """Window-local CAR, then per-segment log power, relative power and log Hjorth.

        ``learn_edges=True`` lets each band edge move by at most 1 Hz (the
        differentiable-feature experiment); the prior itself uses fixed edges.
        """

        def __init__(self, learn_edges: bool = False, fs: float = 120.0, segment_length: int = 240,
                     segment_starts: tuple[int, ...] = (0, 240), bands=PRIOR_BANDS):
            super().__init__()
            self.fs = float(fs)
            self.segment_length = int(segment_length)
            self.segment_starts = tuple(segment_starts)
            self.center_edges = torch.tensor(bands, dtype=torch.float32)
            self.edge_delta = nn.Parameter(torch.zeros_like(self.center_edges), requires_grad=learn_edges)
            self.register_buffer("hann", torch.hann_window(self.segment_length, periodic=True))
            self.register_buffer("freqs", torch.fft.rfftfreq(self.segment_length, d=1 / self.fs))

        def band_edges(self):
            """返回有效频带边界；可学习时每条边最多相对中心移动 1 Hz。"""
            return (self.center_edges.to(self.edge_delta.device) + torch.tanh(self.edge_delta) * 1.0).clamp(0.5, 70.0)

        def forward(self, x):
            """输入 ``(N,C,T)`` 微伏 EEG，输出每窗一行的拼接数学先验特征。"""
            x = x - x.mean(dim=1, keepdim=True)
            edges = self.band_edges()
            hann = self.hann.to(dtype=x.dtype)
            freqs = self.freqs.to(dtype=x.dtype)
            L = self.segment_length
            outputs = []
            for start in self.segment_starts:
                segment = x[..., start:start + L]
                centered = segment - segment.mean(dim=-1, keepdim=True)
                spectrum = torch.fft.rfft(centered * hann, dim=-1)
                psd = spectrum.abs().square() / (self.fs * hann.square().sum()).clamp_min(1e-12)
                one_sided = torch.ones_like(freqs)
                one_sided[1:-1] = 2.
                psd = psd * one_sided
                low = torch.sigmoid((freqs[None, :] - edges[:, 0, None]) / 0.35)
                high = torch.sigmoid((edges[:, 1, None] - freqs[None, :]) / 0.35)
                power = torch.einsum("bcf,kf->bck", psd, low * high) * (self.fs / L)
                log_power = torch.log(power.clamp_min(1e-10))
                relative = power / power.sum(dim=-1, keepdim=True).clamp_min(1e-10)
                first = torch.diff(segment, dim=-1)
                second = torch.diff(segment, n=2, dim=-1)
                activity = centered.var(dim=-1, unbiased=False)
                mobility = (first.var(dim=-1, unbiased=False) / activity.clamp_min(1e-12)).sqrt()
                complexity = (second.var(dim=-1, unbiased=False) /
                              first.var(dim=-1, unbiased=False).clamp_min(1e-12)).sqrt()
                hjorth = torch.stack((activity.clamp_min(1e-12).log(), mobility.clamp_min(1e-12).log(),
                                      complexity.clamp_min(1e-12).log()), dim=-1)
                outputs.append(torch.cat((log_power.flatten(1), relative.flatten(1), hjorth.flatten(1)), dim=1))
            return torch.cat(outputs, dim=1)

        def edge_penalty(self):
            """频带边界偏移的平方惩罚，供训练目标限制边界漂移。"""
            return self.edge_delta.square().mean()


def math_prior_features(X: np.ndarray, device: str = "cpu", batch: int = 128, **kwargs) -> np.ndarray:
    """Numpy in, numpy out. ``X`` in microvolts, (N, C, T).

    默认 CPU 是为了和报告口径保持数值一致；CUDA float32 FFT 会产生细微数值差异，
    在既有 PROTEUS LOSO 探针中分数从 CPU 0.5113 变为 CUDA 0.5109。切换设备时
    应记录设备，不能把这种变化误认为随机种子差异。
    """
    _require_torch()
    module = MathPriorFeatures(False, **kwargs).to(device).eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(X), batch):
            chunk = torch.as_tensor(np.asarray(X[start:start + batch], dtype=np.float32), device=device)
            chunks.append(module(chunk).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)
