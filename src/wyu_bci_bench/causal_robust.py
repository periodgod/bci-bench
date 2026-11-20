"""Robust causal running standardization variants for prequential streams."""
import numpy as np


class RobustZScoreState:
    """Past-only moments; clipping affects stored observations, never current output.

    Without clipping, cumulative mode matches session_transductive.causal_zscore.
    Exponential mode discounts sufficient statistics with decay 1-1/history;
    their total mass approaches history. Both modes shrink mean and variance
    toward the same training prior with mass/(mass+alpha).
    """
    def __init__(self, init_mean, init_var, *, mode="clip", alpha=32., history=64, clip=5.):
        self.m0 = np.asarray(init_mean, dtype=float).copy()
        self.v0 = np.asarray(init_var, dtype=float).copy()
        if self.m0.ndim != 1 or self.v0.shape != self.m0.shape or not np.isfinite(self.m0).all() or not np.isfinite(self.v0).all() or (self.v0 <= 0).any():
            raise ValueError("initial moments must be finite aligned vectors with positive variance")
        if mode not in {"clip", "ewma"} or alpha <= 0 or history < 1 or clip <= 0:
            raise ValueError("invalid state parameters")
        self.mode, self.alpha, self.history, self.clip = mode, alpha, history, clip
        self.reset()

    def reset(self):
        self.mass, self.n = 0., 0
        self.total, self.squares = np.zeros_like(self.m0), np.zeros_like(self.v0)

    def step(self, observation):
        x = np.asarray(observation, dtype=float)
        if x.shape != self.m0.shape or not np.isfinite(x).all():
            raise ValueError("observation must be a finite vector matching the prior")
        hist_mean = self.total / max(self.mass, 1e-12)
        hist_var = np.maximum(self.squares / max(self.mass, 1e-12) - hist_mean**2, 0) if self.n > 1 else np.zeros_like(self.v0)
        w = self.mass / (self.mass + self.alpha)
        mean = self.m0 + w * (hist_mean - self.m0)
        var = self.v0 + w * (hist_var - self.v0)
        std = np.sqrt(np.maximum(var, 1e-12))
        out = (x - mean) / std
        stored = mean + np.clip(out, -self.clip, self.clip) * std if self.mode == "clip" else x
        decay = 1. if self.mode == "clip" else 1. - 1. / self.history
        self.total = decay * self.total + stored
        self.squares = decay * self.squares + stored**2
        self.mass = decay * self.mass + 1
        self.n += 1
        return out.astype(np.float32)


def robust_causal_zscore(values, keys, *, mode="clip", alpha=32., history=64,
                         init_mean=None, init_var=None, clip=5.):
    x = np.asarray(values, dtype=np.float64)
    keys = np.asarray(keys)
    if x.ndim != 2 or len(x) != len(keys) or not np.isfinite(x).all():
        raise ValueError("values must be a finite 2-D matrix aligned with keys")
    d = x.shape[1]
    prior_mean = np.zeros(d) if init_mean is None else np.asarray(init_mean, dtype=float)
    prior_var = np.ones(d) if init_var is None else np.asarray(init_var, dtype=float)
    out = np.empty_like(x)
    for key in dict.fromkeys(keys.tolist()):
        ix = np.flatnonzero(keys == key)
        state = RobustZScoreState(prior_mean, prior_var, mode=mode, alpha=alpha, history=history, clip=clip)
        for row in ix:
            out[row] = state.step(x[row])
    return out.astype(np.float32)
