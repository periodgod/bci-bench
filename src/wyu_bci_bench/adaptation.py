"""Unsupervised domain-alignment transforms for EEG features and epochs.

Target-domain statistics are estimated without labels. In a benchmark, this
is transductive/test-time adaptation if computed from evaluation-domain inputs;
the evaluation protocol must explicitly allow access to those inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import numpy as np


def _symmetric_power(matrix: np.ndarray, power: float, ridge: float) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    scale = max(float(np.trace(matrix)) / max(len(matrix), 1), 1.0)
    values = np.maximum(values, ridge * scale)
    return (vectors * values**power) @ vectors.T


def _epoch_covariance(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 1:
        raise ValueError("X must have shape (trials, channels, samples)")
    if not np.isfinite(values).all():
        raise ValueError("X must contain only finite values")
    centered = values - values.mean(axis=-1, keepdims=True)
    covariances = centered @ np.swapaxes(centered, 1, 2) / values.shape[-1]
    return covariances.mean(axis=0)


def euclidean_align(X: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Whiten a domain using its unlabeled mean trial covariance (EA).

    The input should contain only the domain used to estimate the alignment
    matrix. Do not concatenate training and held-out domains before calling.
    """
    values = np.asarray(X, dtype=np.float64)
    covariance = _epoch_covariance(values)
    whitening = _symmetric_power(covariance, -0.5, ridge)
    return np.einsum("cd,ndt->nct", whitening, values, optimize=True)


def coral_align(source: np.ndarray, target: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Map source feature mean/covariance to an unlabeled target feature set.

    Returns transformed source features; target features remain unchanged.
    Fitting this transform on evaluation inputs is transductive test-time
    adaptation and must be allowed by the declared protocol.
    """
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if src.ndim != 2 or tgt.ndim != 2 or src.shape[1] != tgt.shape[1]:
        raise ValueError("source and target must be 2-D feature arrays with equal feature counts")
    if len(src) < 2 or len(tgt) < 2:
        raise ValueError("source and target each need at least two samples")
    if not np.isfinite(src).all() or not np.isfinite(tgt).all():
        raise ValueError("source and target must contain only finite values")
    source_mean = src.mean(axis=0)
    target_mean = tgt.mean(axis=0)
    source_centered = src - source_mean
    target_centered = tgt - target_mean
    source_cov = source_centered.T @ source_centered / (len(src) - 1)
    target_cov = target_centered.T @ target_centered / (len(tgt) - 1)
    mapping = _symmetric_power(source_cov, -0.5, ridge) @ _symmetric_power(target_cov, 0.5, ridge)
    return source_centered @ mapping + target_mean


@dataclass
class RiemannianAlignment:
    """Whiten each named domain around its Riemannian mean covariance.

    This uses PyRiemann's geometric mean for SPD covariance matrices. Like EA,
    fitting a transform on held-out target epochs is transductive test-time
    adaptation and must be declared by the evaluation protocol.
    """

    ridge: float = 1e-6
    transforms_: dict[Hashable, np.ndarray] | None = None

    def fit(self, X: np.ndarray, domains: np.ndarray) -> "RiemannianAlignment":
        values = np.asarray(X, dtype=np.float64)
        keys = np.asarray(domains)
        if values.ndim != 3 or len(values) != len(keys) or not np.isfinite(values).all():
            raise ValueError("X must be finite (trials, channels, samples) with one domain key per trial")
        if self.ridge <= 0:
            raise ValueError("ridge must be positive")
        try:
            from pyriemann.geometry.mean import gmean
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("install the optional dependency with pip install -e '.[domain-adaptation]'") from exc
        transforms = {}
        for key in np.unique(keys):
            subset = values[keys == key]
            centered = subset - subset.mean(axis=-1, keepdims=True)
            covs = centered @ np.swapaxes(centered, 1, 2) / values.shape[-1]
            scale = max(float(np.mean(np.trace(covs, axis1=1, axis2=2))) / values.shape[1], 1.0)
            covs += np.eye(values.shape[1])[None, :, :] * (self.ridge * scale)
            mean = gmean(covs, metric="riemann")
            transforms[key.item() if hasattr(key, "item") else key] = _symmetric_power(mean, -0.5, self.ridge)
        self.transforms_ = transforms
        return self

    def transform(self, X: np.ndarray, domains: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        keys = np.asarray(domains)
        if values.ndim != 3 or len(values) != len(keys):
            raise ValueError("X must be 3-D and domains must contain one key per trial")
        if self.transforms_ is None:
            raise RuntimeError("call fit before transform")
        out = np.empty_like(values)
        for key in np.unique(keys):
            normalized = key.item() if hasattr(key, "item") else key
            if normalized not in self.transforms_:
                raise ValueError(f"unknown domain {normalized!r}; fit its unlabeled alignment first")
            mask = keys == key
            out[mask] = np.einsum("cd,ndt->nct", self.transforms_[normalized], values[mask], optimize=True)
        return out

    def fit_transform(self, X: np.ndarray, domains: np.ndarray) -> np.ndarray:
        return self.fit(X, domains).transform(X, domains)


def maximum_mean_discrepancy(source, target, bandwidths=(0.5, 1.0, 2.0)):
    """Biased multi-kernel RBF MMD² between source and target feature batches."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch is required for MMD") from exc
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source and target must be 2-D feature tensors with equal feature width")
    if len(source) == 0 or len(target) == 0:
        raise ValueError("source and target batches must be non-empty")
    scales = torch.as_tensor(bandwidths, dtype=source.dtype, device=source.device)
    if scales.ndim != 1 or scales.numel() == 0 or torch.any(scales <= 0):
        raise ValueError("bandwidths must be a non-empty sequence of positive values")
    xx = torch.cdist(source, source).square()
    yy = torch.cdist(target, target).square()
    xy = torch.cdist(source, target).square()
    k_xx = torch.exp(-xx[..., None] / (2 * scales.square())).mean(dim=-1)
    k_yy = torch.exp(-yy[..., None] / (2 * scales.square())).mean(dim=-1)
    k_xy = torch.exp(-xy[..., None] / (2 * scales.square())).mean(dim=-1)
    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()


try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = nn = F = None


if nn is not None:
    class _ReverseGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, features, strength):
            ctx.strength = strength
            return features.view_as(features)

        @staticmethod
        def backward(ctx, gradient):
            return -ctx.strength * gradient, None


    class DomainAdversarialHead(nn.Module):
        """Predict domain IDs while reversing its gradient into the encoder."""

        def __init__(self, features: int, domains: int, hidden: int = 128, strength: float = 1.0):
            super().__init__()
            if min(features, domains, hidden) < 1 or strength < 0:
                raise ValueError("features, domains and hidden must be positive; strength cannot be negative")
            self.strength = float(strength)
            self.classifier = nn.Sequential(nn.Linear(features, hidden), nn.GELU(), nn.Linear(hidden, domains))

        def forward(self, features):
            return self.classifier(_ReverseGradient.apply(features, self.strength))


    def domain_adversarial_loss(task_logits, task_labels, domain_logits, domain_labels, weight: float = 1.0):
        """Task cross-entropy plus domain cross-entropy through a reversed gradient path."""
        if weight < 0:
            raise ValueError("weight must be non-negative")
        return F.cross_entropy(task_logits, task_labels) + weight * F.cross_entropy(domain_logits, domain_labels)
else:  # pragma: no cover
    class DomainAdversarialHead:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DANN")

    def domain_adversarial_loss(*args, **kwargs):
        raise ImportError("PyTorch is required for DANN")


def adapt_batch_norm(model, target_batches, *, reset_running_stats: bool = True, max_batches: int | None = None) -> int:
    """Update BatchNorm running statistics from unlabeled target-domain batches.

    The model's non-BatchNorm layers stay in evaluation mode, so dropout and
    other training-only behavior are disabled during calibration. The caller
    is responsible for passing only the target inputs allowed by its protocol.
    """
    if torch is None:
        raise ImportError("PyTorch is required for AdaBN")
    batch_norms = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)
                   and module.track_running_stats]
    if not batch_norms:
        raise ValueError("model has no tracked BatchNorm layers")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    states = [(module, module.training) for module in model.modules()]
    running_state = [
        (bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone())
        for bn in batch_norms
    ]
    model.eval()
    for bn in batch_norms:
        if reset_running_stats:
            bn.reset_running_stats()
        bn.train()
    seen = 0
    try:
        with torch.no_grad():
            for batch in target_batches:
                if max_batches is not None and seen >= max_batches:
                    break
                inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
                model(inputs)
                seen += 1
    finally:
        for module, (mean, variance, counter) in zip(batch_norms, running_state):
            if seen == 0:
                module.running_mean.copy_(mean)
                module.running_var.copy_(variance)
                module.num_batches_tracked.copy_(counter)
        for module, training in states:
            module.training = training
    if seen == 0:
        raise ValueError("target_batches yielded no batches")
    return seen


@dataclass
class EuclideanAlignment:
    """Fit one unlabeled EA transform per explicitly named domain."""

    ridge: float = 1e-6
    transforms_: dict[Hashable, np.ndarray] | None = None

    def fit(self, X: np.ndarray, domains: np.ndarray) -> "EuclideanAlignment":
        values = np.asarray(X, dtype=np.float64)
        keys = np.asarray(domains)
        if values.ndim != 3 or len(values) != len(keys):
            raise ValueError("X must be 3-D and domains must contain one domain key per trial")
        transforms: dict[Hashable, np.ndarray] = {}
        for key in np.unique(keys):
            mask = keys == key
            covariance = _epoch_covariance(values[mask])
            transforms[key.item() if hasattr(key, "item") else key] = _symmetric_power(covariance, -0.5, self.ridge)
        self.transforms_ = transforms
        return self

    def transform(self, X: np.ndarray, domains: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        keys = np.asarray(domains)
        if values.ndim != 3 or len(values) != len(keys):
            raise ValueError("X must be 3-D and domains must contain one domain key per trial")
        if self.transforms_ is None:
            raise RuntimeError("call fit before transform")
        out = np.empty_like(values)
        for key in np.unique(keys):
            normalized_key = key.item() if hasattr(key, "item") else key
            if normalized_key not in self.transforms_:
                raise ValueError(f"unknown domain {normalized_key!r}; fit its unlabeled alignment first")
            mask = keys == key
            out[mask] = np.einsum("cd,ndt->nct", self.transforms_[normalized_key], values[mask], optimize=True)
        return out

    def fit_transform(self, X: np.ndarray, domains: np.ndarray) -> np.ndarray:
        return self.fit(X, domains).transform(X, domains)
