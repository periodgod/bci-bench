"""Reusable representation and adaptation methods from PROTEUS experiments."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.linalg import eigh
from scipy.special import ndtri
from scipy.signal import periodogram


def full_session_center(features: np.ndarray, keys: Iterable) -> np.ndarray:
    """Subtract each stream's full-session feature mean (uses future windows)."""
    x = np.asarray(features, dtype=np.float32)
    k = np.asarray(keys)
    if x.ndim != 2 or k.shape != (len(x),):
        raise ValueError("features must be 2-D and keys must align with rows")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")
    out = x.copy()
    for key in dict.fromkeys(k.tolist()):
        ix = np.flatnonzero(k == key)
        out[ix] -= out[ix].mean(axis=0, keepdims=True)
    return out


def causal_scale(features: np.ndarray, keys: Iterable, source_variance: np.ndarray,
                 beta: float = 32.0) -> np.ndarray:
    """Center by past mean and scale by a beta-shrunk past/source variance."""
    x = np.asarray(features, dtype=np.float64)
    k = np.asarray(keys)
    prior = np.asarray(source_variance, dtype=np.float64)
    if x.ndim != 2 or k.shape != (len(x),) or prior.shape != (x.shape[1],):
        raise ValueError("features, keys, and source_variance must align")
    if not np.isfinite(x).all() or not np.isfinite(prior).all() or np.any(prior <= 0) or beta < 0:
        raise ValueError("features and positive source variances must be finite; beta non-negative")
    out = np.empty_like(x)
    for key in dict.fromkeys(k.tolist()):
        ix = np.flatnonzero(k == key)
        total = np.zeros(x.shape[1])
        square = np.zeros(x.shape[1])
        for j, row in enumerate(x[ix]):
            n = j
            mean = total / n if n else 0.0
            var = (square - total * total / n) / max(n - 1, 1) if n > 1 else prior
            var = (n * np.maximum(var, 0.0) + beta * prior) / (n + beta) if n + beta else prior
            out[ix[j]] = (row - mean) / np.sqrt(np.maximum(var, 1e-12)) * np.sqrt(prior)
            total += row
            square += row * row
    return out.astype(np.float32)


def causal_rank_normalize(features: np.ndarray, keys: Iterable, source_sorted: np.ndarray,
                          beta: float = 32.0) -> np.ndarray:
    """Map past empirical ranks to Gaussian scores, shrinking to source CDF."""
    x = np.asarray(features, dtype=np.float64)
    k = np.asarray(keys)
    ref = np.asarray(source_sorted, dtype=np.float64)
    if x.ndim != 2 or k.shape != (len(x),) or ref.ndim != 2 or ref.shape[1] != x.shape[1]:
        raise ValueError("features, keys, and source_sorted dimensions must align")
    if not len(ref) or not np.isfinite(x).all() or not np.isfinite(ref).all() or beta < 0:
        raise ValueError("rank inputs must be finite and beta non-negative")
    ref = np.sort(ref, axis=0)
    out = np.empty_like(x)
    for key in dict.fromkeys(k.tolist()):
        ix = np.flatnonzero(k == key)
        history = []
        for row_index in ix:
            row = x[row_index]
            source_cdf = np.array([(np.searchsorted(ref[:, d], row[d], side="right") + 0.5) / (len(ref) + 1)
                                   for d in range(x.shape[1])])
            if history:
                hist = np.asarray(history)
                past_cdf = np.array([(np.count_nonzero(hist[:, d] <= row[d]) + 0.5) / (len(hist) + 1)
                                     for d in range(x.shape[1])])
            else:
                past_cdf = source_cdf
            n = len(history)
            q = (n * past_cdf + beta * source_cdf) / (n + beta) if n + beta else source_cdf
            out[row_index] = ndtri(np.clip(q, 1e-4, 1 - 1e-4))
            history.append(row.copy())
    return out.astype(np.float32)


def exponential_center(features: np.ndarray, keys: Iterable, half_life: float = 64.0) -> np.ndarray:
    """Subtract a per-stream exponentially weighted mean of prior rows."""
    x = np.asarray(features, dtype=np.float64)
    k = np.asarray(keys)
    if x.ndim != 2 or k.shape != (len(x),) or not np.isfinite(x).all() or half_life <= 0:
        raise ValueError("expected finite 2-D features, aligned keys, and positive half_life")
    decay = np.exp(np.log(0.5) / half_life)
    out = np.empty_like(x)
    for key in dict.fromkeys(k.tolist()):
        ix = np.flatnonzero(k == key)
        weighted_sum, weight = np.zeros(x.shape[1]), 0.0
        for i in ix:
            mean = weighted_sum / weight if weight else x[i]
            out[i] = x[i] - mean if weight else x[i]
            weighted_sum = decay * weighted_sum + (1 - decay) * x[i]
            weight = decay * weight + (1 - decay)
    return out.astype(np.float32)


def fit_wccn(train_z: np.ndarray, train_y: np.ndarray, shrinkage: float = 0.5) -> np.ndarray:
    """Return a symmetric within-class whitening transform shrunk to identity."""
    z, y = np.asarray(train_z, dtype=np.float64), np.asarray(train_y)
    if z.ndim != 2 or y.shape != (len(z),) or not np.isfinite(z).all() or not 0 <= shrinkage <= 1:
        raise ValueError("expected finite row-aligned features and shrinkage in [0,1]")
    residual = np.concatenate([z[y == c] - z[y == c].mean(0) for c in np.unique(y)])
    cov = residual.T @ residual / max(len(residual), 1)
    cov = (1 - shrinkage) * cov + shrinkage * np.eye(cov.shape[0])
    values, vectors = np.linalg.eigh(cov)
    return (vectors * (1 / np.sqrt(np.maximum(values, 1e-8)))) @ vectors.T


def periodic_spectral_features(raw: np.ndarray, fs: float = 120.0) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-half periodic residual log power and 1/f offset/exponent."""
    from .priors import PRIOR_BANDS

    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] < 1 or x.shape[2] % 2 or not np.isfinite(x).all() or fs <= 0:
        raise ValueError("raw must be finite (N,C,T) EEG with even T and positive fs")
    n, channels, time = x.shape
    half = time // 2
    periodic, aperiodic = [], []
    window = np.hanning(half)
    for start in (0, half):
        segment = x[:, :, start:start + half]
        segment = segment - segment.mean(axis=-1, keepdims=True)
        freqs, psd = periodogram(segment, fs=fs, window=window, detrend=False,
                                 scaling="density", axis=-1)
        mask = (freqs >= 2.0) & (freqs <= 40.0)
        logf = np.log(freqs[mask])
        logp = np.log(np.maximum(psd[..., mask], 1e-20))
        design = np.stack((np.ones_like(logf), logf), axis=1)
        coef = np.linalg.lstsq(design, logp.reshape(-1, len(logf)).T, rcond=None)[0]
        fitted = (design @ coef).T.reshape(n, channels, -1)
        residual = logp - fitted
        band_values = []
        for lo, hi in PRIOR_BANDS:
            band = (freqs[mask] >= max(lo, 2.0)) & (freqs[mask] < min(hi, 40.00001))
            if np.any(band):
                power = np.mean(np.exp(np.clip(residual[..., band], -30, 30)), axis=-1)
            else:
                power = np.zeros((n, channels), dtype=np.float64)
            band_values.append(np.log(np.maximum(power, 1e-12)))
        periodic.append(np.stack(band_values, axis=-1).astype(np.float32))
        aperiodic.append(coef.T.reshape(n, channels, 2).astype(np.float32))
    hjorth = []
    for start in (0, half):
        seg = x[:, :, start:start + half]
        centered = seg - seg.mean(-1, keepdims=True)
        first, second = np.diff(seg, axis=-1), np.diff(seg, n=2, axis=-1)
        activity = np.var(centered, axis=-1)
        mobility = np.sqrt(np.var(first, axis=-1) / np.maximum(activity, 1e-12))
        complexity = np.sqrt(np.var(second, axis=-1) / np.maximum(np.var(first, axis=-1), 1e-12))
        hjorth.append(np.stack((np.log(np.maximum(activity, 1e-12)),
                                np.log(np.maximum(mobility, 1e-12)),
                                np.log(np.maximum(complexity, 1e-12))), axis=-1).astype(np.float32))
    periodic_features = np.concatenate([np.concatenate((p.reshape(n, -1), h.reshape(n, -1)), axis=1)
                                        for p, h in zip(periodic, hjorth)], axis=1)
    aperiodic_features = np.concatenate([a.reshape(n, -1) for a in aperiodic], axis=1)
    return periodic_features, aperiodic_features


def cell_class_sample_weights(labels: np.ndarray, cell_ids: Iterable) -> np.ndarray:
    """Weights giving each observed cell×class group equal total loss mass."""
    y, cells = np.asarray(labels), np.asarray(cell_ids).astype(str)
    if y.ndim != 1 or cells.shape != y.shape or not len(y):
        raise ValueError("labels and cell_ids must be non-empty aligned vectors")
    pairs = list(zip(cells.tolist(), y.tolist()))
    counts = {pair: pairs.count(pair) for pair in set(pairs)}
    weights = np.asarray([1.0 / counts[pair] for pair in pairs], dtype=np.float64)
    return (weights / weights.mean()).astype(np.float32)


def cell_class_balanced_cross_entropy(logits, targets, sample_weights):
    """Weighted per-window CE for cell×class-balanced training."""
    import torch
    from torch.nn import functional as F

    weights = torch.as_tensor(sample_weights, dtype=logits.dtype, device=logits.device)
    if logits.ndim != 2 or targets.shape != (len(logits),) or weights.shape != targets.shape:
        raise ValueError("logits, targets, and sample_weights must align")
    if not torch.isfinite(logits).all() or not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("logits and non-negative sample weights must be finite")
    per_item = F.cross_entropy(logits, targets, reduction="none")
    return (per_item * weights).sum() / weights.sum().clamp_min(1e-12)


class GroupDRO:
    """Exponentiated-gradient GroupDRO state over fixed training group IDs."""

    def __init__(self, n_groups: int, eta: float = 0.01):
        if n_groups < 1 or eta <= 0:
            raise ValueError("n_groups and eta must be positive")
        self.eta = float(eta)
        self.log_weights = np.full(n_groups, -np.log(n_groups), dtype=np.float64)

    @property
    def weights(self) -> np.ndarray:
        log_q = self.log_weights - self.log_weights.max()
        q = np.exp(log_q)
        return q / q.sum()

    def update(self, group_losses: np.ndarray) -> np.ndarray:
        losses = np.asarray(group_losses, dtype=np.float64)
        if losses.shape != self.log_weights.shape or not np.isfinite(losses).all():
            raise ValueError("group_losses must be finite and match n_groups")
        self.log_weights += self.eta * losses
        self.log_weights -= self.log_weights.max()
        return self.weights


def group_dro_cross_entropy(logits, targets, group_ids, state: GroupDRO):
    """Compute batch GroupDRO CE and update group weights from detached losses."""
    import torch
    from torch.nn import functional as F

    if logits.ndim != 2 or targets.shape != (len(logits),) or group_ids.shape != targets.shape:
        raise ValueError("logits, targets, and group_ids must align")
    per_item = F.cross_entropy(logits, targets, reduction="none")
    losses = torch.zeros(len(state.log_weights), dtype=logits.dtype, device=logits.device)
    present = torch.zeros(len(state.log_weights), dtype=torch.bool, device=logits.device)
    for group in torch.unique(group_ids):
        gi = int(group.item())
        mask = group_ids == group
        losses[gi] = per_item[mask].mean()
        present[gi] = True
    state.update(losses.detach().cpu().numpy())
    q = torch.as_tensor(state.weights, dtype=logits.dtype, device=logits.device)
    q = q * present
    q = q / q.sum().clamp_min(1e-12)
    return (q * losses).sum()


def nuisance_covariance(train_z: np.ndarray, train_keys: Iterable,
                        block_size: int = 32) -> tuple[np.ndarray, int]:
    """Estimate nuisance covariance from ordered within-stream block means."""
    z = np.asarray(train_z, dtype=np.float64)
    keys = np.asarray(train_keys)
    if z.ndim != 2 or keys.shape != (len(z),):
        raise ValueError("train_z must be 2-D and train_keys must align with rows")
    if not np.isfinite(z).all() or block_size < 1:
        raise ValueError("train_z must be finite and block_size positive")
    means = []
    for key in dict.fromkeys(keys.tolist()):
        ix = np.flatnonzero(keys == key)
        for start in range(0, len(ix), block_size):
            means.append(z[ix[start:start + block_size]].mean(0))
    if len(means) < 2:
        raise ValueError("nuisance covariance needs at least two training blocks")
    means = np.asarray(means, dtype=np.float64)
    means -= means.mean(0, keepdims=True)
    return means.T @ means / (len(means) - 1), len(means)


def class_scatter(train_z: np.ndarray, train_y: np.ndarray,
                  n_classes: int = 3) -> np.ndarray:
    """Between-class scatter with equal weight for each configured class."""
    z, y = np.asarray(train_z, dtype=np.float64), np.asarray(train_y)
    if z.ndim != 2 or y.shape != (len(z),) or not len(z):
        raise ValueError("train_z and train_y must be non-empty and row-aligned")
    if n_classes < 2 or not np.isfinite(z).all() or not np.isin(y, np.arange(n_classes)).all():
        raise ValueError("expected finite features and labels in [0, n_classes)")
    if any(not np.any(y == cls) for cls in range(n_classes)):
        raise ValueError("every configured class must occur in train_y")
    overall = z.mean(0)
    means = np.stack([z[y == cls].mean(0) for cls in range(n_classes)])
    delta = means - overall
    return delta.T @ delta / n_classes


def fit_nuisance_subspace(train_z: np.ndarray, train_y: np.ndarray, train_keys: Iterable,
                          n_components: int, block_size: int = 32,
                          class_ridge: float = 1e-3) -> tuple[np.ndarray, np.ndarray, int]:
    """Return descending generalized eigenvectors for nuisance/class scatter."""
    z = np.asarray(train_z, dtype=np.float64)
    y = np.asarray(train_y)
    if not 1 <= n_components <= z.shape[1] or class_ridge <= 0:
        raise ValueError("n_components must fit feature width and class_ridge must be positive")
    nuisance, n_blocks = nuisance_covariance(z, train_keys, block_size)
    category = class_scatter(z, y)
    values, vectors = eigh(nuisance, category + class_ridge * np.eye(category.shape[0]),
                           check_finite=True)
    order = np.argsort(values)[::-1]
    return values[order[:n_components]], vectors[:, order[:n_components]], n_blocks


def remove_nuisance_subspace(z: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Apply the experiment's right-side nuisance-subspace removal operator."""
    x, v = np.asarray(z), np.asarray(vectors)
    if x.ndim != 2 or v.ndim != 2 or x.shape[1] != v.shape[0]:
        raise ValueError("z and vectors must be 2-D with matching feature dimensions")
    if not np.isfinite(x).all() or not np.isfinite(v).all():
        raise ValueError("projection inputs must be finite")
    return x - (x @ v) @ v.T


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def source_prototypes(embeddings: np.ndarray, labels: np.ndarray,
                      n_classes: int = 3) -> np.ndarray:
    """Compute normalized source prototypes in class-index order."""
    emb, y = np.asarray(embeddings, dtype=np.float32), np.asarray(labels)
    if emb.ndim != 2 or y.shape != (len(emb),) or not np.isfinite(emb).all():
        raise ValueError("embeddings and labels must be finite, 2-D, and row-aligned")
    if n_classes < 2 or not np.isin(y, np.arange(n_classes)).all():
        raise ValueError("labels must be in [0, n_classes)")
    if any(not np.any(y == cls) for cls in range(n_classes)):
        raise ValueError("every configured class must occur in labels")
    protos = np.stack([emb[y == cls].mean(0) for cls in range(n_classes)])
    return _normalize_rows(protos).astype(np.float32)


def sinkhorn_assignments(similarity: np.ndarray, epsilon: float = 0.05,
                         iterations: int = 20) -> np.ndarray:
    """Balanced row/column assignment probabilities for sample-class scores."""
    sim = np.asarray(similarity, dtype=np.float64)
    if sim.ndim != 2 or not sim.shape[0] or sim.shape[1] < 2:
        raise ValueError("similarity must have shape (n_samples, n_classes)")
    if not np.isfinite(sim).all() or epsilon <= 0 or iterations < 1:
        raise ValueError("similarity must be finite; epsilon and iterations must be positive")
    log_q = sim / epsilon
    log_q -= log_q.max()
    q = np.exp(log_q)
    q /= max(q.sum(), 1e-300)
    n, k = q.shape
    row, col = np.full(n, 1.0 / n), np.full(k, 1.0 / k)
    for _ in range(iterations):
        q *= (row / np.maximum(q.sum(axis=1), 1e-300))[:, None]
        q *= (col / np.maximum(q.sum(axis=0), 1e-300))[None, :]
    return q.astype(np.float32)


def causal_prototype_prob(embeddings: np.ndarray, source_proto: np.ndarray,
                          source_prob: np.ndarray, *, memory: int = 128,
                          burn_in: int = 32, shrink_kappa: float = 32.0,
                          sinkhorn_epsilon: float = 0.05, sinkhorn_iterations: int = 20,
                          temperature: float = 0.1,
                          source_probability_weight: float = 0.5) -> np.ndarray:
    """Causally adapt source prototypes using only earlier unlabeled embeddings."""
    emb = np.asarray(embeddings, dtype=np.float32)
    proto = np.asarray(source_proto, dtype=np.float32)
    prob = np.asarray(source_prob, dtype=np.float32)
    n_classes = proto.shape[0] if proto.ndim == 2 else 0
    if emb.ndim != 2 or proto.ndim != 2 or emb.shape[1] != proto.shape[1] \
            or prob.shape != (len(emb), n_classes):
        raise ValueError("expected aligned embeddings, source prototypes, and probabilities")
    if not n_classes or not np.isfinite(emb).all() or not np.isfinite(proto).all() \
            or not np.isfinite(prob).all():
        raise ValueError("prototype adaptation inputs must be finite and non-empty")
    if memory < 1 or burn_in < 0 or shrink_kappa < 0 or temperature <= 0 \
            or not 0 <= source_probability_weight <= 1:
        raise ValueError("invalid prototype adaptation parameters")
    history: list[np.ndarray] = []
    out = np.empty_like(prob, dtype=np.float32)
    for i, z in enumerate(emb):
        n = min(len(history), memory)
        adapted = proto
        if n >= burn_in and n > 0:
            hist = np.asarray(history[-memory:], dtype=np.float32)
            q = sinkhorn_assignments(hist @ proto.T, sinkhorn_epsilon, sinkhorn_iterations)
            target = _normalize_rows(q.T @ hist / np.maximum(q.sum(0)[:, None], 1e-8))
            shrink = n / (n + shrink_kappa) if n + shrink_kappa else 1.0
            target = _normalize_rows((1 - shrink) * proto + shrink * target)
            adapted = _normalize_rows((1 - shrink) * proto + shrink * target)
        sim = (z[None, :] @ adapted.T).reshape(-1) / temperature
        sim -= sim.max()
        pp = np.exp(sim).astype(np.float32)
        pp /= pp.sum()
        out[i] = source_probability_weight * prob[i] + (1 - source_probability_weight) * pp
        history.append(z)
    return out


def causal_gaussian_prototype_prob(embeddings: np.ndarray, source_mean: np.ndarray,
                                   source_variance: np.ndarray, source_prob: np.ndarray,
                                   *, labels: np.ndarray | None = None, memory: int = 128,
                                   shrink_kappa: float = 32.0, temperature: float = 1.0,
                                   source_probability_weight: float = 0.5,
                                   sinkhorn_epsilon: float = 0.05,
                                   sinkhorn_iterations: int = 20) -> np.ndarray:
    """Causal diagonal-Gaussian prototypes; optional labels define an oracle bound."""
    z = np.asarray(embeddings, dtype=np.float64)
    means = np.asarray(source_mean, dtype=np.float64)
    var = np.asarray(source_variance, dtype=np.float64)
    prob = np.asarray(source_prob, dtype=np.float64)
    k = means.shape[0] if means.ndim == 2 else 0
    y = None if labels is None else np.asarray(labels)
    if z.ndim != 2 or means.ndim != 2 or z.shape[1] != means.shape[1] \
            or var.shape != means.shape or prob.shape != (len(z), k) \
            or (y is not None and y.shape != (len(z),)):
        raise ValueError("Gaussian prototype inputs must be row-aligned with matching dimensions")
    if not k or not np.isfinite(z).all() or not np.isfinite(means).all() \
            or not np.isfinite(var).all() or np.any(var <= 0) or not np.isfinite(prob).all():
        raise ValueError("Gaussian prototype inputs must be finite with positive variance")
    if y is not None and not np.isin(y, np.arange(k)).all():
        raise ValueError("oracle labels must be in [0, n_classes)")
    if memory < 1 or shrink_kappa < 0 or temperature <= 0 \
            or not 0 <= source_probability_weight <= 1:
        raise ValueError("invalid Gaussian prototype parameters")
    history, out = [], np.empty_like(prob, dtype=np.float32)
    for i, row in enumerate(z):
        prior_means = means.copy()
        if history:
            hist = np.asarray(history)
            if y is not None:
                weights = np.eye(k)[y[:i]]
            else:
                logits = -0.5 * (((hist[:, None, :] - means[None, :, :]) ** 2) / var[None, :, :]).sum(-1)
                weights = sinkhorn_assignments(logits / temperature, sinkhorn_epsilon, sinkhorn_iterations)
                weights *= len(hist)
            counts = weights.sum(0)
            hist_means = weights.T @ hist / np.maximum(counts[:, None], 1e-8)
            shrink = counts / (counts + shrink_kappa)
            prior_means = (1 - shrink[:, None]) * means + shrink[:, None] * hist_means
        logits = -0.5 * (((row[None, :] - prior_means) ** 2) / var).sum(-1)
        logits -= logits.max()
        gp = np.exp(logits / temperature)
        gp /= gp.sum()
        out[i] = source_probability_weight * prob[i] + (1 - source_probability_weight) * gp
        history.append(row.copy())
    return out
