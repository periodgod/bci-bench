"""Window-local Welch band power / relative power / Hjorth feature variants (numpy + scipy).

Ported from ``proteus_update1/temporal_features.py`` (``extract_variant``),
which many PROTEUS submissions and the 8-subject phase-split experiments use.
Unlike ``priors.math_prior_features`` (Hann FFT per half, torch), this uses
Welch PSD with 7 bands up to 70 Hz and supports clipping / CAR / segment
variants such as ``"pre_post_car"``. No statistics are learned.
"""

from __future__ import annotations

import numpy as np

WELCH_BANDS = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (45, 70))


def per_channel_features(X: np.ndarray, fs: float = 120., nperseg: int = 240, noverlap: int = 120,
                         bands=WELCH_BANDS):
    from scipy.integrate import trapezoid
    from scipy.signal import welch

    freqs, psd = welch(X, fs=fs, axis=-1, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    powers = np.stack([trapezoid(psd[..., (freqs >= lo) & (freqs < hi)], freqs[(freqs >= lo) & (freqs < hi)], axis=-1)
                       for lo, hi in bands], axis=-1)
    log_power = np.log(np.maximum(powers, 1e-10))
    relative = powers / np.maximum(powers.sum(axis=-1, keepdims=True), 1e-10)
    centered = X - X.mean(axis=-1, keepdims=True)
    first, second = np.diff(X, axis=-1), np.diff(X, n=2, axis=-1)
    activity = np.var(centered, axis=-1)
    mobility = np.sqrt(np.var(first, axis=-1) / np.maximum(activity, 1e-12))
    complexity = np.sqrt(np.var(second, axis=-1) / np.maximum(np.var(first, axis=-1), 1e-12))
    hjorth = np.stack([np.log(np.maximum(activity, 1e-12)), np.log(np.maximum(mobility, 1e-12)),
                       np.log(np.maximum(complexity, 1e-12))], axis=-1)
    return log_power, relative, hjorth


def segment_features(X: np.ndarray, style: str = "standard", **kwargs) -> np.ndarray:
    log_power, relative, hjorth = per_channel_features(X, **kwargs)
    if style == "scale_invariant":
        log_power = log_power - np.median(log_power, axis=(1, 2), keepdims=True)
        hjorth[:, :, 0] -= np.median(hjorth[:, :, 0], axis=1, keepdims=True)
    elif style == "no_high":
        log_power = log_power[..., :6]
        relative = relative[..., :6]
        relative = relative / np.maximum(relative.sum(axis=-1, keepdims=True), 1e-10)
    elif style != "standard":
        raise ValueError(style)
    return np.concatenate([p.reshape(len(X), -1) for p in (log_power, relative, hjorth)], axis=1).astype(np.float32)


def extract_variant(X: np.ndarray, variant: str, expected_shape: tuple[int, int] | None = (41, 480)) -> np.ndarray:
    """``variant`` = segment prefix (whole/pre/post/pre_post) + optional style + ``_clipped`` / ``_car``.

    e.g. ``"pre_post_car"`` (early/late half + CAR, 8-subject 0.5562 line), ``"whole"`` (0.4289 baseline).
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or (expected_shape is not None and X.shape[1:] != tuple(expected_shape)):
        raise ValueError(f"expected input (N, {expected_shape}), got {X.shape}")
    if variant.endswith("_clipped"):
        center = np.median(X, axis=-1, keepdims=True)
        mad = np.median(np.abs(X - center), axis=-1, keepdims=True)
        limit = 6 * 1.4826 * np.maximum(mad, 0.5)
        X = np.clip(X, center - limit, center + limit)
    if variant.endswith("_car"):
        X = X - X.mean(axis=1, keepdims=True)
    style = "scale_invariant" if "scale_invariant" in variant else "no_high" if "no_high" in variant else "standard"
    half = X.shape[-1] // 2
    if variant.startswith("pre_post"):
        return np.concatenate([segment_features(X[:, :, :half], style), segment_features(X[:, :, half:], style)], axis=1)
    if variant.startswith("pre"):
        return segment_features(X[:, :, :half], style)
    if variant.startswith("post"):
        return segment_features(X[:, :, half:], style)
    if variant.startswith("whole"):
        return segment_features(X, style)
    raise ValueError(variant)
