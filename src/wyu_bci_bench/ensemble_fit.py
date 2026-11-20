"""Low-dimensional ensemble fitting on caller-supplied, isolated inner OOF."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .calibration import _validated_probabilities


def fit_simplex_weights(probabilities, labels, *, sample_weight=None, shrinkage=.1):
    """Fit nonnegative sum-one weights using weighted NLL and uniform shrinkage.

    Input is (samples, members, classes). Cross-fitting is the caller's duty;
    this function must never receive an outer test label.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if p.ndim != 3 or not len(p) or p.shape[1] < 1 or shrinkage < 0:
        raise ValueError("expected nonempty (samples, members, classes), shrinkage >= 0")
    for j in range(p.shape[1]):
        _validated_probabilities(p[:, j], y)
    sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if sw.shape != y.shape or not np.isfinite(sw).all() or (sw < 0).any() or sw.sum() <= 0:
        raise ValueError("invalid sample weights")
    sw = sw / sw.sum()
    truth = p[np.arange(len(y)), :, y]
    prior = np.full(p.shape[1], 1 / p.shape[1])

    def objective(w):
        mixed = np.maximum(truth @ w, 1e-12)
        loss = -sw @ np.log(mixed) + shrinkage * np.square(w-prior).sum()
        grad = -truth.T @ (sw / mixed) + 2 * shrinkage * (w-prior)
        return float(loss), grad

    result = minimize(objective, prior, jac=True, method="SLSQP",
                      bounds=[(0, 1)] * len(prior),
                      constraints={"type": "eq", "fun": lambda w: w.sum()-1,
                                   "jac": lambda w: np.ones_like(w)},
                      options={"ftol": 1e-12, "maxiter": 500})
    if not result.success:
        raise RuntimeError(f"simplex fit failed: {result.message}")
    weights = np.clip(result.x, 0, 1)
    return weights / weights.sum()
