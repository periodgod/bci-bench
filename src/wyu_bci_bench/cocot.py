"""CoCoT-EEG architecture and fold-local supervised fine-tuning recipe.

The model follows the paper's convolution-first, L2-patch, Fourier-time-position
and Transformer design. The paper does not publish an implementation link or a
complete kernel list, so kernel sizes and a few head details are explicit bench
configuration rather than claims of bitwise reproduction.
"""

from __future__ import annotations

import copy
import math
import os
import random
from typing import Any, Mapping

import numpy as np


def _torch_modules() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for CoCoT-EEG") from exc
    return torch, nn, F, (DataLoader, TensorDataset)


def robust_normalize(eeg: np.ndarray) -> np.ndarray:
    """Normalize each epoch/channel using the paper's median and MAD rule."""
    x = np.asarray(eeg, dtype=np.float32)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError("eeg must be a finite (trials, channels, samples) array")
    median = np.median(x, axis=-1, keepdims=True)
    mad = np.median(np.abs(x - median), axis=-1, keepdims=True)
    return ((x - median) / (mad + 1e-6)).astype(np.float32)


def resample_epochs(eeg: np.ndarray, sampling_rate: int, target_rate: int = 100) -> np.ndarray:
    """Polyphase-resample epochs to the paper's 100 Hz input rate."""
    x = np.asarray(eeg, dtype=np.float32)
    if x.ndim != 3 or sampling_rate < 1 or target_rate < 1:
        raise ValueError("eeg must be 3-D and sampling rates must be positive")
    if sampling_rate == target_rate:
        return x.copy()
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install the ml extra to resample CoCoT-EEG inputs") from exc
    divisor = math.gcd(int(sampling_rate), int(target_rate))
    return resample_poly(x, target_rate // divisor, sampling_rate // divisor, axis=-1).astype(np.float32)


def _model_classes():
    torch, nn, F, _ = _torch_modules()

    class DropPath(nn.Module):
        def __init__(self, probability: float):
            super().__init__()
            self.probability = float(probability)

        def forward(self, x):
            if not self.training or self.probability == 0:
                return x
            keep = 1.0 - self.probability
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
            return x * mask

    class MultiScaleConvTokenizer(nn.Module):
        def __init__(self, model_dim: int, patch_size: int, kernel_sizes: tuple[int, ...]):
            super().__init__()
            if not kernel_sizes or any(k < 1 or k % 2 == 0 for k in kernel_sizes):
                raise ValueError("kernel_sizes must be a non-empty sequence of positive odd integers")
            base, remainder = divmod(model_dim, len(kernel_sizes))
            widths = [base + (i < remainder) for i in range(len(kernel_sizes))]
            if min(widths) < 1:
                raise ValueError("model_dim must be at least the number of kernel branches")
            self.patch_size = int(patch_size)
            self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
            self.branches = nn.ModuleList()
            for kernel, width in zip(self.kernel_sizes, widths):
                groups = min(8, width)
                while width % groups:
                    groups -= 1
                self.branches.append(nn.Sequential(
                    nn.Conv1d(1, width, kernel, padding=kernel // 2, bias=False),
                    nn.GroupNorm(groups, width),
                    nn.GELU(),
                ))
            self.norm = nn.LayerNorm(model_dim)

        def forward(self, x):
            batch, channels, samples = x.shape
            usable = samples - samples % self.patch_size
            if usable < self.patch_size:
                raise ValueError("input must contain at least one full patch")
            per_channel = x.reshape(batch * channels, 1, samples)
            features = torch.cat([branch(per_channel) for branch in self.branches], dim=1)
            features = features[..., :usable].reshape(batch, channels, features.shape[1], -1)
            features = features.unflatten(-1, (usable // self.patch_size, self.patch_size))
            tokens = features.square().sum(dim=-1).clamp_min(1e-12).sqrt().transpose(2, 3)
            return self.norm(tokens)

    class TransformerBlock(nn.Module):
        def __init__(self, model_dim: int, ffn_dim: int, heads: int, drop_path: float):
            super().__init__()
            self.norm1 = nn.LayerNorm(model_dim)
            self.attention = nn.MultiheadAttention(model_dim, heads, batch_first=True)
            self.norm2 = nn.LayerNorm(model_dim)
            self.gate_value = nn.Linear(model_dim, 2 * ffn_dim)
            self.ffn_out = nn.Linear(ffn_dim, model_dim)
            self.gamma1 = nn.Parameter(torch.full((model_dim,), 1e-4))
            self.gamma2 = nn.Parameter(torch.full((model_dim,), 1e-4))
            self.drop_path = DropPath(drop_path)

        def forward(self, x):
            normalized = self.norm1(x)
            attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
            x = x + self.drop_path(self.gamma1 * attended)
            value, gate = self.gate_value(self.norm2(x)).chunk(2, dim=-1)
            hidden = F.silu(gate) * value
            return x + self.drop_path(self.gamma2 * self.ffn_out(hidden))

    class CoCoTEncoder(nn.Module):
        def __init__(self, model_dim: int, ffn_dim: int, layers: int, heads: int,
                     patch_size: int, kernel_sizes: tuple[int, ...], drop_path_rate: float):
            super().__init__()
            if model_dim % heads:
                raise ValueError("model_dim must be divisible by attention heads")
            if layers < 1 or patch_size < 1 or not 0 <= drop_path_rate < 1:
                raise ValueError("layers, patch_size and drop_path_rate are out of range")
            self.model_dim = int(model_dim)
            self.patch_size = int(patch_size)
            self.tokenizer = MultiScaleConvTokenizer(model_dim, patch_size, kernel_sizes)
            rates = torch.linspace(0, drop_path_rate, layers).tolist()
            self.blocks = nn.ModuleList([
                TransformerBlock(model_dim, ffn_dim, heads, rates[i]) for i in range(layers)
            ])
            self.norm = nn.LayerNorm(model_dim)

        def forward(self, x):
            tokens = self.tokenizer(x)
            batch, channels, steps, width = tokens.shape
            half = (width + 1) // 2
            timescales = torch.linspace(1.0, 10000.0, half, device=x.device, dtype=x.dtype)
            phase = torch.arange(steps, device=x.device, dtype=x.dtype)[:, None] / timescales[None, :]
            position = torch.empty((steps, width), device=x.device, dtype=x.dtype)
            position[:, 0::2] = torch.sin(phase[:, :position[:, 0::2].shape[1]])
            position[:, 1::2] = torch.cos(phase[:, :position[:, 1::2].shape[1]])
            tokens = tokens + position[None, None, :, :]
            tokens = tokens.reshape(batch, channels * steps, width)
            for block in self.blocks:
                tokens = block(tokens)
            return self.norm(tokens).reshape(batch, channels, steps, width)

    class ResidualGLUBlock(nn.Module):
        def __init__(self, hidden_dim: int, residual_scale: float, dropout: float):
            super().__init__()
            self.norm = nn.LayerNorm(hidden_dim)
            self.projection = nn.Linear(hidden_dim, hidden_dim * 2)
            self.dropout = nn.Dropout(dropout)
            self.residual_scale = float(residual_scale)

        def forward(self, x):
            value, gate = self.projection(self.norm(x)).chunk(2, dim=-1)
            return x + self.residual_scale * self.dropout(value * torch.sigmoid(gate))

    class CoCoTEEG(nn.Module):
        def __init__(self, channels: int, samples: int, classes: int, *,
                     model_dim: int = 512, ffn_dim: int = 2048, layers: int = 12,
                     heads: int = 8, patch_size: int = 50,
                     kernel_sizes: tuple[int, ...] = (3, 5, 9, 17, 33, 65, 97, 121),
                     drop_path_rate: float = 0.1, head_dim: int = 1024,
                     head_layers: int = 2, dropout: float = 0.1,
                     residual_scale: float = 0.1):
            super().__init__()
            if min(channels, samples, classes, head_dim, head_layers) < 1:
                raise ValueError("channels, samples, classes and head dimensions must be positive")
            if samples < patch_size:
                raise ValueError("samples must be at least one patch long")
            self.channels = int(channels)
            self.samples = int(samples)
            self.encoder = CoCoTEncoder(model_dim, ffn_dim, layers, heads, patch_size,
                                        tuple(kernel_sizes), drop_path_rate)
            tokens = channels * (samples // patch_size)
            self.head_projection = nn.Sequential(
                nn.Linear(tokens * model_dim, head_dim), nn.GELU(), nn.Dropout(dropout)
            )
            self.head_blocks = nn.Sequential(*[
                ResidualGLUBlock(head_dim, residual_scale, dropout) for _ in range(head_layers)
            ])
            self.head_norm = nn.LayerNorm(head_dim)
            self.classifier = nn.Linear(head_dim, classes)

        def forward(self, x):
            if x.ndim != 3 or x.shape[1:] != (self.channels, self.samples):
                raise ValueError(f"expected (batch, {self.channels}, {self.samples}), got {tuple(x.shape)}")
            features = self.encoder(x).flatten(1)
            projected = self.head_projection(features)
            hidden = self.head_blocks(projected) + projected
            return self.classifier(self.head_norm(hidden))

        def load_encoder_state_dict(self, state_dict: Mapping[str, Any], *, strict: bool = True):
            """Load an encoder-only checkpoint, excluding its pretraining projector."""
            return self.encoder.load_state_dict(state_dict, strict=strict)

    return CoCoTEEG


def make_cocot_model(channels: int, samples: int, classes: int, **kwargs: Any):
    """Create a CoCoT-EEG classifier for one fixed epoch shape."""
    return _model_classes()(channels, samples, classes, **kwargs)


def _mixup_batch(torch: Any, x: Any, y: Any, alpha: float):
    if alpha <= 0:
        raise ValueError("mixup_alpha must be positive")
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(len(x), device=x.device)
    return lam * x + (1.0 - lam) * x[permutation], y, y[permutation], lam


def fit_predict_cocot(
    train_eeg: np.ndarray,
    train_y: np.ndarray,
    test_eeg: np.ndarray,
    classes: int,
    *,
    sampling_rate: int,
    seed: int = 42,
    epochs: int = 40,
    patience: int = 8,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 5e-2,
    mixup_alpha: float | None = None,
    encoder_state_dict: Mapping[str, Any] | None = None,
    device: str | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Fine-tune on an outer training fold and return hard trial predictions.

    The default optimizer and head dimensions follow Appendix B. Validation is
    split by trial inside the outer training fold; early stopping is a bench
    protocol choice because the paper does not specify a stopping rule.
    """
    torch, nn, F, loaders = _torch_modules()
    DataLoader, TensorDataset = loaders
    from .deep import validation_split

    if sampling_rate < 1:
        raise ValueError("sampling_rate must be positive")
    train_x = robust_normalize(resample_epochs(train_eeg, sampling_rate))
    test_x = robust_normalize(resample_epochs(test_eeg, sampling_rate))
    y = np.asarray(train_y, dtype=np.int64)
    if train_x.shape[1:] != test_x.shape[1:] or len(train_x) != len(y):
        raise ValueError("train/test epoch shapes and training labels must agree")
    if len(train_x) < 2 or len(test_x) == 0 or classes < 2 or not np.isin(y, np.arange(classes)).all():
        raise ValueError("need at least two training trials, one test trial and valid class labels")
    if epochs < 1 or patience < 1 or batch_size < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("training hyperparameters are out of range")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fit_idx, val_idx = validation_split(y, seed=seed)
    if not len(val_idx):
        raise ValueError("the training fold is too small for an internal validation split")

    options = dict(model_kwargs or {})
    model = make_cocot_model(train_x.shape[1], train_x.shape[2], classes, **options).to(run_device)
    if encoder_state_dict is not None:
        model.load_encoder_state_dict(encoder_state_dict, strict=True)
    train_tensor = torch.from_numpy(train_x[fit_idx])
    train_labels = torch.from_numpy(y[fit_idx])
    val_tensor = torch.from_numpy(train_x[val_idx]).to(run_device)
    val_labels = torch.from_numpy(y[val_idx]).to(run_device)
    test_tensor = torch.from_numpy(test_x).to(run_device)
    loader = DataLoader(TensorDataset(train_tensor, train_labels), batch_size=min(batch_size, len(fit_idx)),
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_loss, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(run_device), by.to(run_device)
            optimizer.zero_grad(set_to_none=True)
            if mixup_alpha is None:
                loss = F.cross_entropy(model(bx), by)
            else:
                mixed, ya, yb, lam = _mixup_batch(torch, bx, by, mixup_alpha)
                logits = model(mixed)
                loss = lam * F.cross_entropy(logits, ya) + (1.0 - lam) * F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_loss = float(F.cross_entropy(model(val_tensor), val_labels).item())
        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        predictions = model(test_tensor).argmax(dim=1).cpu().numpy().astype(np.int64)
    metadata: dict[str, float | str] = {
        "model": "cocot_eeg",
        "best_epoch": float(best_epoch),
        "best_validation_loss": float(best_loss),
        "parameter_count": float(sum(p.numel() for p in model.parameters())),
        "device": str(run_device),
        "mixup_alpha": float(mixup_alpha or 0.0),
        "pretrained_encoder": float(encoder_state_dict is not None),
        "sampling_rate": 100.0,
    }
    return predictions, metadata
