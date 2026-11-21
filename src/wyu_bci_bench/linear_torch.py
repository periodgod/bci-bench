"""L2 多类 softmax 回归（PyTorch 全批 L-BFGS），用于高维特征的快速线性读出。

目标函数与 sklearn ``LogisticRegression(C=C)``（multinomial、L2、带截距）相同：
``C * sum_i CE_i + 0.5 * ||W||^2``，截距不受罚。先按训练集均值/标准差标准化。
返回的参数都是 numpy 数组，可直接写进提交文件，不需要 pickle。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SoftmaxRegression:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: np.ndarray

    def logits(self, features: np.ndarray) -> np.ndarray:
        z = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        return z @ self.weight.T + self.bias

    def predict_proba(self, features: np.ndarray, *, offset_logits: np.ndarray | None = None) -> np.ndarray:
        logits = self.logits(features)
        if offset_logits is not None:
            logits += _offset(offset_logits, logits.shape)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)


def _offset(logits, shape):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != shape or not np.isfinite(logits).all():
        raise ValueError("offset_logits must be finite and match (samples, classes)")
    return logits


def fit_softmax_regression(features: np.ndarray, y: np.ndarray, C: float = 0.01, n_classes: int | None = None,
                           device: str = "cpu", max_iter: int = 500, tol: float = 1e-7, *,
                           feature_multiplier: np.ndarray | None = None,
                           offset_logits: np.ndarray | None = None) -> SoftmaxRegression:
    """拟合 L2 softmax，乘数在标准化后施加，offset 为冻结的样本级分数。"""
    import torch
    from torch.nn import functional as F

    X = np.asarray(features, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    k = int(n_classes or labels.max() + 1)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-12] = 1.0
    if feature_multiplier is not None:
        multiplier = np.asarray(feature_multiplier, dtype=np.float64)
        if multiplier.shape != scale.shape or not np.isfinite(multiplier).all() or (multiplier <= 0).any():
            raise ValueError("feature_multiplier must contain one finite positive value per feature")
        scale = scale / multiplier
    Z = torch.as_tensor((X - mean) / scale, dtype=torch.float64, device=device)
    t = torch.as_tensor(labels, device=device)
    W = torch.zeros((k, X.shape[1]), dtype=torch.float64, device=device, requires_grad=True)
    b = torch.zeros(k, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, tolerance_grad=tol, tolerance_change=tol * 1e-2,
                            history_size=20, line_search_fn="strong_wolfe")
    inv = 1.0 / (C * len(labels))
    offset = torch.as_tensor(_offset(offset_logits, (len(X), k)), device=device) if offset_logits is not None else 0.

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Z @ W.T + b + offset, t, reduction="mean") + 0.5 * inv * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return SoftmaxRegression(mean=mean, scale=scale, weight=W.detach().cpu().numpy(), bias=b.detach().cpu().numpy())
