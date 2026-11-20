"""SessionFM: session-conditioned EEG foundation model and causal state.

The module implements the architecture in SessionFM_会话感知脑电基础模型架构设计_20261027.md:
4 s / 120 Hz windows, 0.5 s tokens, temporal and relative-spectrum branches,
coordinate encoding, zero-initialized session FiLM, criss-cross attention,
masked reconstruction heads, and prior-anchored downstream logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..priors import PRIOR_BANDS


def session_spectral_features(x: torch.Tensor, fs: float = 120.0) -> torch.Tensor:
    """Return four 1 s x 10-d log-band/Hjorth features per channel."""
    if x.ndim != 3 or x.shape[-1] != 480:
        raise ValueError("SessionFM expects windows shaped (batch, channels, 480)")
    # Input is already window-CAR and scaled by the preprocessing contract.
    segments = x.reshape(*x.shape[:2], 4, 120)
    centered = segments - segments.mean(dim=-1, keepdim=True)
    hann = torch.hann_window(120, periodic=True, device=x.device, dtype=x.dtype)
    spectrum = torch.fft.rfft(centered * hann, dim=-1)
    psd = spectrum.abs().square() / (fs * hann.square().sum()).clamp_min(1e-12)
    freqs = torch.fft.rfftfreq(120, d=1 / fs, device=x.device)
    one_sided = torch.ones_like(freqs)
    one_sided[1:-1] = 2
    psd = psd * one_sided
    powers = []
    for low, high in PRIOR_BANDS:
        band = (freqs >= low) & (freqs < high)
        powers.append((psd[..., band].sum(-1) * fs / 120).clamp_min(1e-10).log())
    first, second = torch.diff(segments, dim=-1), torch.diff(segments, n=2, dim=-1)
    activity = centered.var(dim=-1, unbiased=False).clamp_min(1e-12)
    mobility = (first.var(dim=-1, unbiased=False) / activity).sqrt().clamp_min(1e-12)
    complexity = (second.var(dim=-1, unbiased=False) /
                  first.var(dim=-1, unbiased=False).clamp_min(1e-12)).sqrt().clamp_min(1e-12)
    hjorth = torch.stack((activity.log(), mobility.log(), complexity.log()), dim=-1)
    return torch.cat((torch.stack(powers, dim=-1), hjorth), dim=-1)


def sessionfm_pretraining_spectral_target(
    raw: torch.Tensor, relative: torch.Tensor, variant: str
) -> torch.Tensor:
    """Select the shared L_spec/L_global target for a pretraining variant."""
    if variant == "absolute_spectrum":
        return raw
    if variant in {"main", "fixed_n_zero", "no_film", "wave_only"}:
        return relative
    raise ValueError(f"unknown SessionFM pretraining variant: {variant}")


def coordinate_fourier(coords: torch.Tensor, n_frequencies: int = 4) -> torch.Tensor:
    """Fourier encode (x,y,z) electrode coordinates and normalized token time."""
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape (batch, channels, 3)")
    b, c, _ = coords.shape
    t = torch.linspace(0, 1, 8, device=coords.device, dtype=coords.dtype)
    # Electrode coordinates are stored in meters; centimeters make a 3 cm
    # neighborhood visible to the lowest spatial Fourier frequency.
    xyz_t = torch.cat(((coords[:, :, None, :] * 100).expand(b, c, 8, 3),
                       t[None, None, :, None].expand(b, c, 8, 1)), dim=-1)
    frequencies = (2.0 ** torch.arange(n_frequencies, device=coords.device, dtype=coords.dtype)) * torch.pi
    phase = xyz_t[..., None] * frequencies
    return torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)


class CrissCrossBlock(nn.Module):
    """One shared 8-head QKV projection split across spatial/time axes."""

    def __init__(self, dim: int = 128, heads_per_axis: int = 4, hidden: int = 256,
                 drop_path: float = 0.1):
        super().__init__()
        self.dim = dim
        self.heads_per_axis = heads_per_axis
        self.head_dim = dim // (heads_per_axis * 2)
        if dim % (heads_per_axis * 2):
            raise ValueError("dim must be divisible by twice the heads per axis")
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.geglu = nn.Linear(dim, 2 * hidden, bias=False)
        self.geglu_out = nn.Linear(hidden, dim, bias=False)
        self.drop_path = float(drop_path)

    def _attention(self, qkv: torch.Tensor, *, spatial: bool) -> torch.Tensor:
        b, c, t, _, _ = qkv.shape
        h = self.heads_per_axis
        q, k, v = qkv.unbind(dim=3)
        q, k, v = (part.reshape(b, c, t, h * 2, self.head_dim) for part in (q, k, v))
        if spatial:
            q, k, v = (p[..., :h, :].permute(0, 2, 3, 1, 4).reshape(b * t, h, c, self.head_dim)
                       for p in (q, k, v))
        else:
            q, k, v = (p[..., h:, :].permute(0, 1, 3, 2, 4).reshape(b * c, h, t, self.head_dim)
                       for p in (q, k, v))
        z = F.scaled_dot_product_attention(q, k, v)
        if spatial:
            return z.reshape(b, t, h, c, self.head_dim).permute(0, 3, 1, 2, 4).reshape(b, c, t, h * self.head_dim)
        return z.reshape(b, c, h, t, self.head_dim).permute(0, 1, 3, 2, 4).reshape(b, c, t, h * self.head_dim)

    def _drop(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path == 0:
            return x
        keep = 1 - self.drop_path
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

    def forward(self, x: torch.Tensor, film: tuple[torch.Tensor, torch.Tensor] | None = None):
        b, c, t, d = x.shape
        h = self.norm1(x)
        if film is not None:
            gamma, beta = film
            h = h * (1 + gamma[:, :, None, :]) + beta[:, :, None, :]
        qkv = self.qkv(h).reshape(b, c, t, 3, d)
        spatial, temporal = self._attention(qkv, spatial=True), self._attention(qkv, spatial=False)
        x = x + self._drop(self.proj(torch.cat((spatial, temporal), dim=-1)))
        a, gate = self.geglu(self.norm2(x)).chunk(2, dim=-1)
        x = x + self._drop(self.geglu_out(a * F.gelu(gate)))
        return x


class SessionBaselineState:
    """Per-session causal spectral baseline; current window is added after use."""

    def __init__(self, alpha: float = 32.0):
        self.alpha = float(alpha)
        self.reset()

    def reset(self) -> None:
        self.spectral_sum: torch.Tensor | None = None
        self.count = 0

    def baseline(self, prior_mean: torch.Tensor, *, n_channels: int) -> torch.Tensor:
        prior_mean = prior_mean.reshape(n_channels, 10)
        if self.count == 0:
            return prior_mean
        historical = self.spectral_sum / self.count
        weight = self.count / (self.count + self.alpha)
        return weight * historical + (1 - weight) * prior_mean

    def update(self, features: torch.Tensor) -> None:
        value = features.detach().reshape(-1, 4, 10).mean(dim=1)
        self.spectral_sum = value if self.spectral_sum is None else self.spectral_sum + value
        self.count += 1


class SessionFM(nn.Module):
    """~1.5M-parameter SessionFM-S with batch and causal single-window APIs."""

    def __init__(self, n_classes: int = 3, dim: int = 128, depth: int = 6,
                 heads_per_axis: int = 4, hidden: int = 341, alpha: float = 32.0,
                 drop_path: float = 0.1, use_sbc_film: bool = True):
        super().__init__()
        self.dim, self.n_classes, self.alpha = dim, n_classes, float(alpha)
        self.use_sbc_film = bool(use_sbc_film)
        self.temporal = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3), nn.GELU(), nn.AdaptiveAvgPool1d(2),
        )
        self.spectral = nn.Linear(10, 64)
        self.embed_norm = nn.LayerNorm(128)
        self.position_fourier = nn.Linear(32, dim, bias=False)
        self.position_mlp = nn.Sequential(nn.Linear(4, 128), nn.GELU(), nn.Linear(128, dim))
        self.position_norm = nn.LayerNorm(dim)
        self.global_query = nn.Parameter(torch.zeros(dim))
        self.global_head = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 10))
        self.film = nn.ModuleList([nn.Sequential(nn.Linear(21, 64), nn.GELU(), nn.Linear(64, 2 * dim))
                                   for _ in range(depth)])
        for module in self.film:
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)
        self.blocks = nn.ModuleList([CrissCrossBlock(dim, heads_per_axis, hidden, drop_path)
                                     for _ in range(depth)])
        self.mask_token = nn.Parameter(torch.zeros(128))
        self.wave_head = nn.Linear(dim, 60)
        self.spec_head = nn.Linear(dim, 10)
        self.pool_query = nn.Parameter(torch.randn(dim) * 0.02)
        self.residual_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128), nn.GELU(),
                                           nn.Linear(128, n_classes))
        self.gate = nn.Parameter(torch.tensor(-2.5))
        self.register_buffer("default_coords", torch.zeros(1, 64, 3))
        self.register_buffer("prior_mean", torch.zeros(64, 10))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _prepare(self, x: torch.Tensor, coords: torch.Tensor | None,
                 baseline: torch.Tensor | None, n_seen: torch.Tensor | int,
                 mask: torch.Tensor | None):
        if x.ndim != 3 or x.shape[-1] != 480:
            raise ValueError("SessionFM expects (batch, channels, 480) inputs")
        b, c, _ = x.shape
        if coords is None:
            if c > self.default_coords.shape[1]:
                raise ValueError("provide coordinates for more than 64 channels")
            coords = self.default_coords[:, :c].expand(b, -1, -1)
        elif coords.ndim == 2:
            coords = coords[None].expand(b, -1, -1)
        raw_spec = session_spectral_features(x)
        if baseline is None:
            if self.prior_mean.shape[0] >= c:
                baseline = self.prior_mean[:c][None].expand(b, -1, -1)
            else:
                baseline = x.new_zeros((b, c, 10))
        if baseline.ndim == 2:
            baseline = baseline[None].expand(b, -1, -1)
        if mask is not None:
            if mask.shape != (b, c, 8):
                raise ValueError("mask must have shape (batch, channels, 8)")
            x = x.reshape(b, c, 8, 60).masked_fill(mask[..., None], 0).reshape(b, c, 480)
        temporal = self.temporal(x.reshape(b * c * 8, 1, 60)).flatten(1).reshape(b, c, 8, 64)
        relative_spec = raw_spec - baseline[:, :, None, :]
        spec_tokens = relative_spec.repeat_interleave(2, dim=2)
        if mask is not None:
            segment_mask = mask.reshape(b, c, 4, 2).all(dim=-1)
            spec_tokens = spec_tokens.masked_fill(segment_mask.repeat_interleave(2, dim=2)[..., None], 0)
        spectral = self.spectral(spec_tokens)
        if self.training:
            coords = coords + torch.randn_like(coords) * 0.0025
        t = torch.linspace(0, 1, 8, device=coords.device, dtype=coords.dtype)
        xyz_t = torch.cat((coords[:, :, None, :].expand(b, c, 8, 3),
                           t[None, None, :, None].expand(b, c, 8, 1)), dim=-1)
        fourier = self.position_fourier(coordinate_fourier(coords))
        nonlinear = self.position_mlp(xyz_t)
        pe = self.position_norm(fourier + nonlinear)
        tokens = self.embed_norm(torch.cat((temporal, spectral), dim=-1)) + pe
        if mask is not None:
            tokens = torch.where(mask[..., None], self.mask_token + pe, tokens)
        n = torch.as_tensor(n_seen, device=x.device, dtype=x.dtype)
        if n.ndim == 0:
            n = n.expand(b)
        elif n.ndim > 1:
            n = n.reshape(b)
        g = baseline.mean(dim=1, keepdim=True).expand(-1, c, -1)
        context = torch.cat((baseline, g, n.clamp_min(0).log1p()[:, None, None].expand(-1, c, -1)), dim=-1)
        return tokens, raw_spec, relative_spec, context, pe

    def forward(self, x: torch.Tensor, *, coords: torch.Tensor | None = None,
                baseline: torch.Tensor | None = None, n_seen: torch.Tensor | int = 0,
                mask: torch.Tensor | None = None, prior_logits: torch.Tensor | None = None):
        tokens, raw_spec, relative_spec, context, pe = self._prepare(x, coords, baseline, n_seen, mask)
        layer_outputs = []
        for film, block in zip(self.film, self.blocks):
            if self.use_sbc_film:
                gamma, beta = film(context).chunk(2, dim=-1)
                tokens = block(tokens, (gamma, beta))
            else:
                tokens = block(tokens)
            layer_outputs.append(tokens)
        layer_tokens = torch.stack(layer_outputs, dim=1)
        global_scores = torch.einsum("blctd,d->blct", layer_tokens, self.global_query) / self.dim ** 0.5
        global_weights = torch.softmax(global_scores, dim=-1)
        per_channel = torch.einsum("blct,blctd->blcd", global_weights, layer_tokens)
        global_token = per_channel.mean(dim=2).mean(dim=1)
        pooled_weights = torch.softmax(
            (torch.einsum("bctd,d->bct", tokens, self.pool_query) / self.dim ** 0.5).flatten(1), dim=1
        ).reshape(tokens.shape[:3])
        pooled = torch.einsum("bct,bctd->bd", pooled_weights, tokens)
        residual_logits = self.residual_head(pooled)
        logits = residual_logits if prior_logits is None else prior_logits + self.gate.sigmoid() * residual_logits
        wave_reconstruction = self.wave_head(tokens)
        spec_reconstruction = self.spec_head(tokens.reshape(tokens.shape[0], tokens.shape[1], 4, 2, self.dim).mean(3))
        segment_position = pe.reshape(tokens.shape[0], tokens.shape[1], 4, 2, self.dim).mean(dim=3)
        global_query = global_token[:, None, None, :].expand(-1, tokens.shape[1], 4, -1)
        global_reconstruction = self.global_head(torch.cat((global_query, segment_position), dim=-1))
        return {
            "logits": logits,
            "embedding": pooled,
            "tokens": tokens,
            "wave_reconstruction": wave_reconstruction,
            "spec_reconstruction": spec_reconstruction,
            "global_reconstruction": global_reconstruction,
            "global_token": global_token,
            "position_encoding": pe,
            "raw_spectral_features": raw_spec,
            "relative_spectral_features": relative_spec,
            "mask": mask,
        }

    def forward_stream(self, x: torch.Tensor, state: SessionBaselineState, *,
                       coords: torch.Tensor, prior_logits: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Predict one window from past-only session state, then update state."""
        if x.ndim == 2:
            x = x.unsqueeze(0)
        c = x.shape[1]
        baseline = state.baseline(self.prior_mean[:c], n_channels=c).to(device=x.device, dtype=x.dtype)
        result = self(x, coords=coords, baseline=baseline, n_seen=state.count, prior_logits=prior_logits)
        state.update(result["raw_spectral_features"])
        return result


__all__ = ["SessionFM", "SessionBaselineState", "CrissCrossBlock", "session_spectral_features",
           "sessionfm_pretraining_spectral_target", "coordinate_fourier"]
