"""Session-causal calibration: unlabeled state that is reset at every session.

Two corrections share one rule: window ``t`` of a stream may only use the
unlabeled windows ``0..t-1`` of the same stream, and the state starts empty
when the stream (subject x session) begins. This matches a Codabench solver
whose ``reset_state()`` is called before every session and whose ``predict``
sees windows in recording order. No labels are used.

- ``causal_center``: subtract a shrunk running mean of the past features,
  ``z_t - n/(n+alpha) * mean(z_0..z_{t-1})``. Applied identically to training
  streams and test streams so the classifier is fitted on the same quantity it
  is evaluated on.
- ``causal_class_balance``: divide the current class probabilities by a shrunk
  running mean of the past predicted probabilities and renormalise. PROTEUS
  presents the three commands in near-equal numbers per session; the prior
  only uses that design fact, not the labels.

``SessionCausalState`` is the streaming form used inside a submission; the
batch functions are the reference used in LOSO experiments and must agree
with it (see tests).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def stream_keys(*groups: Iterable) -> np.ndarray:
    """把 subject、session 等分组列拼成一个流标识；同一标识内按行顺序视为时间顺序。"""
    cols = [np.asarray(g).astype(str) for g in groups]
    if not cols:
        raise ValueError("need at least one grouping column")
    out = cols[0].astype(object)
    for c in cols[1:]:
        out = out + "|" + c
    return out.astype(str)


def _stream_indices(keys: np.ndarray):
    keys = np.asarray(keys)
    for k in dict.fromkeys(keys.tolist()):
        yield np.flatnonzero(keys == k)


def causal_center(Z: np.ndarray, keys: np.ndarray, alpha: float = 32.0, init: np.ndarray | None = None) -> np.ndarray:
    """逐流因果中心化。行顺序必须是各流内的记录顺序。

    第一个窗口不变；之后减去收缩后的历史均值，收缩权重 ``n/(n+alpha)``。
    给定 ``init``（例如训练折的总体均值）时，参考点从 ``init`` 起步并向历史
    均值收缩：``ref_t = init + n/(n+alpha) * (mean(z_0..z_{t-1}) - init)``，
    第一个窗口减去 ``init``。``init`` 只能来自训练数据。
    """
    Z = np.asarray(Z, dtype=np.float64)
    out = Z.copy()
    for ix in _stream_indices(keys):
        cs = np.cumsum(Z[ix], axis=0)
        n = np.arange(len(ix), dtype=np.float64)[:, None]
        mu = np.zeros_like(Z[ix])
        mu[1:] = cs[:-1] / n[1:]
        if init is None:
            out[ix] = Z[ix] - (n / (n + alpha)) * mu
        else:
            base = np.asarray(init, dtype=np.float64)[None, :]
            out[ix] = Z[ix] - (base + (n / (n + alpha)) * (mu - base))
    return out


def causal_euclidean_align(
    X: np.ndarray,
    keys: np.ndarray,
    alpha: float = 32.0,
    init_cov: np.ndarray | None = None,
    ridge: float = 1e-6,
) -> np.ndarray:
    """逐流因果欧氏对齐（EA）：窗口 ``t`` 用同流 ``0..t-1`` 的平均协方差白化。

    参考协方差 ``R_t = C0 + n/(n+alpha) * (mean(C_0..C_{t-1}) - C0)``。``C0`` 为
    ``init_cov``（应取训练折的平均协方差）；缺省时取单位阵乘以首窗平均方差。
    输出 ``R_t^{-1/2} x_t``。不使用标签，也不使用当前或未来窗口的统计量。
    """
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("X must have shape (trials, channels, samples)")
    centered = values - values.mean(axis=-1, keepdims=True)
    covs = np.einsum("nct,ndt->ncd", centered, centered, optimize=True) / values.shape[-1]
    channels = values.shape[1]
    out = np.empty_like(values)
    for ix in _stream_indices(keys):
        if init_cov is None:
            base = np.eye(channels) * float(np.trace(covs[ix[0]])) / channels
        else:
            base = np.asarray(init_cov, dtype=np.float64)
        running = np.zeros_like(base)
        for n, i in enumerate(ix):
            ref = base if n == 0 else base + (n / (n + alpha)) * (running / n - base)
            vals, vecs = np.linalg.eigh(0.5 * (ref + ref.T))
            vals = np.clip(vals, ridge * max(float(np.mean(vals)), 1e-12), None)
            out[i] = (vecs * vals ** -0.5) @ vecs.T @ values[i]
            running += covs[i]
    return out


def causal_class_balance(P: np.ndarray, keys: np.ndarray, alpha: float = 32.0) -> np.ndarray:
    """逐流因果类别均衡：用历史预测均值估计当前流的预测偏置并除去。"""
    P = np.asarray(P, dtype=np.float64)
    k = P.shape[1]
    out = P.copy()
    for ix in _stream_indices(keys):
        cs = np.cumsum(P[ix], axis=0)
        n = np.arange(len(ix), dtype=np.float64)[:, None]
        m = np.full_like(P[ix], 1.0 / k)
        m[1:] = cs[:-1] / n[1:]
        w = n / (n + alpha)
        prior = (1.0 - w) / k + w * m
        q = P[ix] / np.clip(prior, 1e-12, None)
        out[ix] = q / q.sum(axis=1, keepdims=True)
    return out


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=-1, keepdims=True)


def _fixed_point_bias(history: np.ndarray, target: np.ndarray, iterations: int, tolerance: float) -> np.ndarray:
    """Solve a logit offset whose mean adjusted historical probabilities match target."""
    eps = 1e-12
    log_history = np.log(np.clip(history, eps, 1.0))
    bias = np.zeros(history.shape[1], dtype=np.float64)
    log_target = np.log(target)
    for _ in range(iterations):
        adjusted = _softmax_rows(log_history + bias)
        mean = np.clip(adjusted.mean(axis=0), eps, 1.0)
        error = log_target - np.log(mean)
        error -= error.mean()
        if np.max(np.abs(error)) <= tolerance:
            break
        bias += 0.5 * error
    return bias


def causal_fixed_point_prior_match(
    P: np.ndarray,
    keys: np.ndarray,
    alpha: float = 32.0,
    burn_in: int = 8,
    iterations: int = 100,
    tolerance: float = 1e-7,
    history_window: int | None = None,
) -> np.ndarray:
    """Causal logit-prior matching by fixed point, shrunk toward no correction.

    For each stream, fit a class-logit offset on prior *raw* predictions so
    their adjusted mean matches a uniform prior. Apply the offset to the
    current window with n/(n+alpha) shrinkage; the current window enters the
    history only after it has been corrected. No labels or future windows are
    used. Before ``burn_in`` prior windows, probabilities pass through.
    """
    P = np.asarray(P, dtype=np.float64)
    keys = np.asarray(keys)
    if P.ndim != 2 or len(P) != len(keys) or P.shape[1] < 2:
        raise ValueError("P must be (n_windows, n_classes) and align with keys")
    if not np.isfinite(P).all() or np.any(P < 0) or np.any(P.sum(axis=1) <= 0):
        raise ValueError("P must contain finite non-negative rows with positive mass")
    if alpha <= 0 or burn_in < 0 or iterations < 1 or tolerance <= 0 or (history_window is not None and history_window < 1):
        raise ValueError("alpha, iterations and tolerance must be positive; burn_in cannot be negative")
    P = P / P.sum(axis=1, keepdims=True)
    out = P.copy()
    target = np.full(P.shape[1], 1.0 / P.shape[1], dtype=np.float64)
    for ix in _stream_indices(keys):
        for pos in range(len(ix)):
            if pos < burn_in:
                continue
            history = P[ix[max(0, pos-history_window) if history_window is not None else 0:pos]]
            bias = _fixed_point_bias(history, target, iterations, tolerance)
            shrink = pos / (pos + alpha)
            out[ix[pos]] = _softmax_rows(np.log(np.clip(P[ix[pos]], 1e-12, 1.0)) + shrink * bias)
    return out


class SessionCausalState:
    """Streaming version for inference: call ``reset()`` at each session start.

    ``center(z)`` and ``balance(p)`` take one window (or a batch in recording
    order) and return the corrected values; each window is corrected with the
    state *before* it is added, so a window never sees itself.
    """

    def __init__(self, alpha_center: float = 32.0, alpha_balance: float = 32.0):
        self.alpha_center = float(alpha_center)
        self.alpha_balance = float(alpha_balance)
        self.reset()

    def reset(self) -> None:
        """清空当前流的历史特征和历史预测。"""
        self._z_sum = None
        self._p_sum = None
        self._nz = 0
        self._np = 0

    def center(self, z: np.ndarray) -> np.ndarray:
        """逐行中心化并把原始特征加入历史。"""
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        out = np.empty_like(z)
        for i, row in enumerate(z):
            if self._nz:
                w = self._nz / (self._nz + self.alpha_center)
                out[i] = row - w * self._z_sum / self._nz
            else:
                out[i] = row
                self._z_sum = np.zeros_like(row)
            self._z_sum += row
            self._nz += 1
        return out

    def balance(self, p: np.ndarray) -> np.ndarray:
        """逐行类别均衡并把未校正的预测加入历史。"""
        p = np.atleast_2d(np.asarray(p, dtype=np.float64))
        k = p.shape[1]
        out = np.empty_like(p)
        for i, row in enumerate(p):
            if self._np:
                w = self._np / (self._np + self.alpha_balance)
                prior = (1.0 - w) / k + w * self._p_sum / self._np
            else:
                prior = np.full(k, 1.0 / k)
                self._p_sum = np.zeros(k)
            q = row / np.clip(prior, 1e-12, None)
            out[i] = q / q.sum()
            self._p_sum += row
            self._np += 1
        return out


class FixedPointPriorMatchState:
    """Streaming fixed-point prior match; reset before each session."""

    def __init__(self, n_classes: int = 3, alpha: float = 32.0, burn_in: int = 8,
                 iterations: int = 100, tolerance: float = 1e-7, history_window: int | None = None):
        if n_classes < 2:
            raise ValueError("n_classes must be at least two")
        if alpha <= 0 or burn_in < 0 or iterations < 1 or tolerance <= 0 or (history_window is not None and history_window < 1):
            raise ValueError("invalid fixed-point prior-matching parameters")
        self.n_classes = int(n_classes)
        self.alpha = float(alpha)
        self.burn_in = int(burn_in)
        self.iterations = int(iterations)
        self.tolerance = float(tolerance)
        self.history_window = history_window
        self.reset()

    def reset(self) -> None:
        self._history: list[np.ndarray] = []
        self._count = 0

    def update(self, probability: np.ndarray) -> np.ndarray:
        row = np.asarray(probability, dtype=np.float64).reshape(-1)
        if row.shape != (self.n_classes,) or not np.isfinite(row).all() or np.any(row < 0) or row.sum() <= 0:
            raise ValueError("probability must be a finite non-negative vector with positive mass")
        row = row / row.sum()
        n = self._count
        if n < self.burn_in:
            corrected = row.copy()
        else:
            history = np.asarray(self._history)
            target = np.full(self.n_classes, 1.0 / self.n_classes)
            bias = _fixed_point_bias(history, target, self.iterations, self.tolerance)
            corrected = _softmax_rows(np.log(np.clip(row, 1e-12, 1.0)) + n / (n + self.alpha) * bias)
        self._history.append(row.copy())
        self._count += 1
        if self.history_window is not None:
            self._history = self._history[-self.history_window:]
        return corrected
