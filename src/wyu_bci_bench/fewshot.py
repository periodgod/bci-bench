"""Few-shot linear decoder adaptation with source-derived Gaussian priors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True)
class GaussianDecoderPrior:
    """Gaussian prior represented by low-rank directions and isotropic residual."""

    mean: np.ndarray
    directions: np.ndarray
    variances: np.ndarray
    residual_variance: float

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=np.float64)
        directions = np.asarray(self.directions, dtype=np.float64)
        variances = np.asarray(self.variances, dtype=np.float64)
        if mean.ndim != 1 or directions.ndim != 2 or directions.shape[0] != len(mean):
            raise ValueError("prior mean/directions have incompatible shapes")
        if directions.shape[1] != len(variances) or np.any(variances < 0):
            raise ValueError("prior direction variances have incompatible shapes")
        if not np.isfinite(mean).all() or not np.isfinite(directions).all() or not np.isfinite(variances).all():
            raise ValueError("prior parameters must be finite")
        if not np.isfinite(self.residual_variance) or self.residual_variance <= 0:
            raise ValueError("residual_variance must be finite and positive")


def balanced_kshot_indices(
    labels: np.ndarray,
    runs: np.ndarray,
    k: int,
    *,
    repeats: int = 200,
    seed: int = 20261101,
) -> np.ndarray:
    """Sample total-k labels evenly over two classes and two calibration runs.

    For k >= 4, each run x class stratum contributes k/4 examples. For k=2,
    one example per class is selected and its run is randomized. If k contains
    all available examples, every repetition is the same deterministic set.
    """
    labels = np.asarray(labels)
    runs = np.asarray(runs).astype(str)
    if labels.ndim != 1 or runs.shape != labels.shape or repeats < 1:
        raise ValueError("labels/runs must be aligned vectors and repeats positive")
    classes = np.unique(labels)
    run_names = np.unique(runs)
    if len(classes) != 2 or len(run_names) != 2:
        raise ValueError("balanced sampling requires two classes and two runs")
    if k < 2 or k % 2 or (k > 2 and k % 4):
        raise ValueError("k must be 2 or a positive multiple of four")
    strata = {(run, cls): np.flatnonzero((runs == run) & (labels == cls))
              for run in run_names for cls in classes}
    if any(len(ix) == 0 for ix in strata.values()) or k > len(labels):
        raise ValueError("each run/class stratum must be nonempty and k cannot exceed available data")
    rng = np.random.default_rng(seed)
    out = np.empty((repeats, k), dtype=np.int64)
    for rep in range(repeats):
        chosen = []
        if k == 2:
            for cls in classes:
                run = run_names[int(rng.integers(0, len(run_names)))]
                chosen.append(int(rng.choice(strata[(run, cls)])))
        else:
            per_stratum = k // 4
            for run in run_names:
                for cls in classes:
                    ix = strata[(run, cls)]
                    if per_stratum > len(ix):
                        raise ValueError("k requests more than a run/class stratum contains")
                    chosen.extend(rng.choice(ix, size=per_stratum, replace=False).tolist())
        out[rep] = rng.permutation(np.asarray(chosen, dtype=np.int64))
    return out


def estimate_decoder_prior(
    source_weights: np.ndarray,
    base_weight: np.ndarray,
    rank: int | None,
    *,
    variance_floor: float = 1e-12,
) -> GaussianDecoderPrior:
    """Estimate a low-rank-plus-isotropic prior from source decoder deviations.

    ``rank=None`` retains all empirical directions, up to ``n_sources - 1``.
    The prior mean remains exactly ``base_weight``; only the covariance is
    estimated from source-specific deviations.
    """
    weights = np.asarray(source_weights, dtype=np.float64)
    base = np.asarray(base_weight, dtype=np.float64)
    if weights.ndim != 2 or base.ndim != 1 or weights.shape[1] != len(base):
        raise ValueError("source_weights and base_weight have incompatible shapes")
    if len(weights) < 2 or not np.isfinite(weights).all() or not np.isfinite(base).all():
        raise ValueError("at least two finite source decoder weights are required")
    if variance_floor <= 0 or not np.isfinite(variance_floor):
        raise ValueError("variance_floor must be finite and positive")
    delta = weights - base
    centered = delta - delta.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular ** 2 / max(1, len(weights) - 1)
    max_rank = min(len(weights) - 1, len(base), len(eigenvalues))
    kept = max_rank if rank is None else int(rank)
    if kept < 0 or kept > max_rank:
        raise ValueError(f"rank must be between 0 and {max_rank}")
    directions = vt[:kept].T.copy()
    variances = eigenvalues[:kept].copy()
    total_variance = float(eigenvalues.sum())
    residual_dimension = max(1, len(base) - kept)
    residual = max((total_variance - float(variances.sum())) / residual_dimension, variance_floor)
    return GaussianDecoderPrior(base.copy(), directions, variances, residual)


def isotropic_decoder_prior(
    source_weights: np.ndarray,
    base_weight: np.ndarray,
    variance_multiplier: float = 1.0,
    *,
    variance_floor: float = 1e-12,
) -> GaussianDecoderPrior:
    """Build an isotropic prior using the mean source coefficient variance."""
    weights = np.asarray(source_weights, dtype=np.float64)
    base = np.asarray(base_weight, dtype=np.float64)
    if weights.ndim != 2 or base.ndim != 1 or weights.shape[1] != len(base) or len(weights) < 2:
        raise ValueError("source_weights and base_weight have incompatible shapes")
    if variance_multiplier <= 0 or not np.isfinite(variance_multiplier):
        raise ValueError("variance_multiplier must be finite and positive")
    delta = weights - base
    variance = float(np.var(delta, axis=0, ddof=1).mean()) * variance_multiplier
    return GaussianDecoderPrior(base.copy(), np.empty((len(base), 0)), np.empty(0),
                                max(variance, variance_floor))


def _prior_penalty(theta: np.ndarray, prior: GaussianDecoderPrior) -> tuple[float, np.ndarray]:
    delta = theta - prior.mean
    if prior.directions.shape[1]:
        coordinates = prior.directions.T @ delta
        variances = prior.variances + prior.residual_variance
        residual = delta - prior.directions @ coordinates
        value = np.sum(coordinates ** 2 / variances) + np.sum(residual ** 2) / prior.residual_variance
        gradient = prior.directions @ (coordinates / variances) + residual / prior.residual_variance
    else:
        value = np.sum(delta ** 2) / prior.residual_variance
        gradient = delta / prior.residual_variance
    return float(value), gradient


def _prior_covariance_product(values: np.ndarray, prior: GaussianDecoderPrior) -> np.ndarray:
    """Multiply vectors by the covariance without materializing a dense matrix."""
    values = np.asarray(values, dtype=np.float64)
    if prior.directions.shape[1]:
        coordinates = prior.directions.T @ values
        return (prior.directions @ ((prior.variances + prior.residual_variance)[:, None] * coordinates)
                + prior.residual_variance * (values - prior.directions @ coordinates))
    return prior.residual_variance * values


def fit_map_logistic(
    X: np.ndarray,
    labels: np.ndarray,
    prior: GaussianDecoderPrior,
    *,
    max_iter: int = 200,
    tolerance: float = 1e-9,
    initial_parameters: np.ndarray | None = None,
) -> np.ndarray:
    """Fit binary Logistic MAP parameters; the final coordinate is intercept.

    Empty calibration data returns the prior mean exactly, making the k=0
    behavior explicit. Labels may be hard 0/1 values or soft probabilities.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] + 1 != len(prior.mean) or y.shape != (len(X),):
        raise ValueError("X, labels and prior dimensions do not match")
    if not np.isfinite(X).all() or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("features must be finite and labels must be in [0,1]")
    if max_iter < 1 or tolerance <= 0:
        raise ValueError("max_iter and tolerance must be positive")
    if not len(y):
        return prior.mean.copy()
    design = np.column_stack([X, np.ones(len(X))])
    covariance_design = _prior_covariance_product(design.T, prior)
    kernel = design @ covariance_design
    kernel = (kernel + kernel.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    cutoff = np.finfo(np.float64).eps * max(kernel.shape) * max(float(eigenvalues[-1]), 1.0)
    keep = eigenvalues > cutoff
    if not np.any(keep):
        return prior.mean.copy()
    basis = eigenvectors[:, keep]
    roots = np.sqrt(eigenvalues[keep])
    whitened_design = basis * roots[None, :]
    parameter_map = covariance_design @ (basis / roots[None, :])
    initial = prior.mean if initial_parameters is None else np.asarray(initial_parameters, dtype=np.float64)
    if initial.shape != prior.mean.shape or not np.isfinite(initial).all():
        raise ValueError("initial_parameters must be a finite vector matching the prior")
    if initial_parameters is None:
        gamma0 = np.zeros(len(roots), dtype=np.float64)
    else:
        gamma0 = np.linalg.lstsq(whitened_design, design @ (initial - prior.mean), rcond=None)[0]

    def objective(gamma):
        logits = design @ prior.mean + whitened_design @ gamma
        likelihood = np.logaddexp(0.0, logits) - y * logits
        gradient = whitened_design.T @ (expit(logits) - y) + gamma
        return float(likelihood.sum() + 0.5 * gamma @ gamma), gradient

    result = minimize(objective, gamma0, jac=True, method="L-BFGS-B",
                      options={"maxiter": int(max_iter), "ftol": min(float(tolerance), 1e-12),
                               "gtol": min(float(tolerance), 1e-9)})
    if not np.isfinite(result.x).all():
        raise FloatingPointError("MAP Logistic optimizer returned non-finite parameters")
    if not result.success and np.linalg.norm(result.jac, ord=np.inf) > 1e-4:
        raise RuntimeError(f"MAP Logistic did not converge: {result.message}")
    return prior.mean + parameter_map @ result.x


def predict_map_logistic(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Return binary class probabilities for MAP Logistic parameters."""
    X = np.asarray(X, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if X.ndim != 2 or theta.shape != (X.shape[1] + 1,):
        raise ValueError("X and theta have incompatible shapes")
    p1 = expit(X @ theta[:-1] + theta[-1])
    return np.column_stack([1.0 - p1, p1])


def balance_unlabeled_probabilities(probabilities: np.ndarray, iterations: int = 50) -> np.ndarray:
    """Rescale binary probabilities to a uniform marginal without labels."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("probabilities must be a finite non-negative (n, 2) array")
    if len(p) == 0 or iterations < 1 or np.any(p.sum(axis=1) <= 0):
        raise ValueError("probability rows must have mass and iterations must be positive")
    q = p / p.sum(axis=1, keepdims=True)
    target = np.full(2, 0.5)
    for _ in range(iterations):
        mean = np.maximum(q.mean(axis=0), 1e-12)
        q *= target / mean
        q /= q.sum(axis=1, keepdims=True)
    return q


def semi_supervised_em_logistic(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    prior: GaussianDecoderPrior,
    *,
    iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Run batch soft-label EM with fixed labeled rows and a uniform class prior."""
    X_labeled = np.asarray(X_labeled, dtype=np.float64)
    y_labeled = np.asarray(y_labeled, dtype=np.float64)
    X_unlabeled = np.asarray(X_unlabeled, dtype=np.float64)
    if X_labeled.ndim != 2 or X_unlabeled.ndim != 2 or X_labeled.shape[1] != X_unlabeled.shape[1]:
        raise ValueError("labeled and unlabeled feature shapes are incompatible")
    if y_labeled.shape != (len(X_labeled),) or iterations < 1:
        raise ValueError("labels must align with labeled rows and iterations must be positive")
    theta = fit_map_logistic(X_labeled, y_labeled, prior)
    pseudo = np.empty((len(X_unlabeled), 2), dtype=np.float64)
    for _ in range(iterations):
        pseudo = balance_unlabeled_probabilities(predict_map_logistic(X_unlabeled, theta))
        X = np.concatenate([X_labeled, X_unlabeled])
        soft_y = np.concatenate([y_labeled, pseudo[:, 1]])
        theta = fit_map_logistic(X, soft_y, prior, initial_parameters=theta)
    return theta, pseudo
