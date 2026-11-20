"""会话内无标签方法：把一个会话当作一组窗口，而不是彼此独立的样本。

规则同 ``session_causal``：窗口 ``t`` 只能用同一流（participant×session）里
``0..t-1`` 的无标签窗口，每个流开始时状态为空。``causal=False`` 的版本会用到
同一会话的未来窗口，只能作上界诊断，不符合 Track 2 的部署合同。
"""

from __future__ import annotations

import numpy as np


def _streams(keys):
    keys = np.asarray(keys)
    for k in dict.fromkeys(keys.tolist()):
        yield np.flatnonzero(keys == k)


def _unit_rows(Z):
    Z = np.asarray(Z, dtype=np.float64)
    return Z / np.clip(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12, None)


def causal_zscore(Z: np.ndarray, keys: np.ndarray, alpha: float = 32.0,
                  init_mean: np.ndarray | None = None, init_var: np.ndarray | None = None,
                  block: int = 2048) -> np.ndarray:
    """逐流因果 z 分数：均值与方差都从初值出发，按 ``n/(n+alpha)`` 收缩到历史估计。

    ``init_mean``/``init_var`` 应来自训练折，缺省为 0 和 1。
    """
    Z = np.asarray(Z, dtype=np.float32)
    d = Z.shape[1]
    m0 = np.zeros(d) if init_mean is None else np.asarray(init_mean, dtype=np.float64)
    v0 = np.ones(d) if init_var is None else np.asarray(init_var, dtype=np.float64)
    out = np.empty_like(Z)
    for ix in _streams(keys):
        n = np.arange(len(ix), dtype=np.float64)[:, None]
        w = n / (n + alpha)
        for s in range(0, d, block):
            x = Z[ix, s:s + block].astype(np.float64)
            cs = np.cumsum(x, 0) - x
            cq = np.cumsum(x * x, 0) - x * x
            mu = np.where(n > 0, cs / np.maximum(n, 1), 0.0)
            var = np.where(n > 1, cq / np.maximum(n, 1) - mu * mu, 0.0)
            mean = m0[s:s + block] + w * (mu - m0[s:s + block])
            v = v0[s:s + block] + w * (np.clip(var, 0, None) - v0[s:s + block])
            out[ix, s:s + block] = ((x - mean) / np.sqrt(np.clip(v, 1e-12, None))).astype(np.float32)
    return out


def neighbor_smooth(P: np.ndarray, Z: np.ndarray, keys: np.ndarray, k: int = 10, lam: float = 0.5,
                    kappa: float = 10.0, causal: bool = True) -> np.ndarray:
    """会话内近邻平滑：``(1-λ_t) p_t + λ_t * mean(p_j, j∈kNN_t)``，余弦距离。

    因果版本只在本流已见窗口里找近邻，``λ_t = λ * n/(n+kappa)``。
    """
    P = np.asarray(P, dtype=np.float64)
    U = _unit_rows(Z)
    out = P.copy()
    for ix in _streams(keys):
        S = U[ix] @ U[ix].T
        for t in range(len(ix)):
            pool = np.arange(t) if causal else np.delete(np.arange(len(ix)), t)
            if len(pool) == 0:
                continue
            kk = min(k, len(pool))
            nb = pool[np.argpartition(-S[t, pool], kk - 1)[:kk]]
            lt = lam * len(pool) / (len(pool) + kappa) if causal else lam
            out[ix[t]] = (1 - lt) * P[ix[t]] + lt * P[ix[nb]].mean(0)
    return out


def causal_adaptive_lda(P: np.ndarray, Z: np.ndarray, keys: np.ndarray, class_means: np.ndarray,
                        precision: np.ndarray, tau: float = 20.0, beta: float = 1.0, burn: int = 10) -> np.ndarray:
    """会话自适应 LDA：以输入概率作软标签，因果地估计本会话类均值并修正预测。

    ``class_means`` (K,d) 与 ``precision`` (d,d) 来自训练折。会话类均值
    ``m_c = (tau*mu_c + Σ_{j<t} r_jc z_j) / (tau + Σ_{j<t} r_jc)``；输出
    ``softmax(log p_t + beta * LDA_logits(z_t; m))``，在前 ``burn`` 窗不修正。
    """
    P = np.asarray(P, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    mu = np.asarray(class_means, dtype=np.float64)
    W = np.asarray(precision, dtype=np.float64)
    out = P.copy()
    for ix in _streams(keys):
        sz = np.zeros_like(mu)
        sr = np.zeros(mu.shape[0])
        for t, i in enumerate(ix):
            if t >= burn:
                m = (tau * mu + sz) / (tau + sr)[:, None]
                logits = m @ W @ Z[i] - 0.5 * np.einsum("kd,de,ke->k", m, W, m)
                adj = (logits - logits.max()) * beta
                lp = np.log(np.clip(P[i], 1e-12, None)) + adj
                lp -= lp.max()
                out[i] = np.exp(lp) / np.exp(lp).sum()
            sz += P[i][:, None] * Z[i][None, :]
            sr += P[i]
    return out


def knn_vote_proba(train_Z: np.ndarray, train_y: np.ndarray, test_Z: np.ndarray, k: int = 25,
                   n_classes: int = 3, block: int = 512) -> np.ndarray:
    """记忆检索：余弦距离 k 近邻投票，加一平滑后归一化为概率。"""
    A = _unit_rows(train_Z)
    B = _unit_rows(test_Z)
    y = np.asarray(train_y)
    out = np.empty((len(B), n_classes))
    for s in range(0, len(B), block):
        S = B[s:s + block] @ A.T
        nb = np.argpartition(-S, k - 1, axis=1)[:, :k]
        counts = np.stack([(y[nb] == c).sum(1) for c in range(n_classes)], 1) + 1.0
        out[s:s + block] = counts / counts.sum(1, keepdims=True)
    return out
