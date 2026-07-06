"""Lightweight capsule-network voter for EEG epochs."""

from __future__ import annotations

import copy
import os
from typing import Any

import numpy as np

from .deep import aggregate_trial_predictions, make_trial_windows, set_seed, validation_split


def _torch() -> Any:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for capsule benchmarks") from exc
    return torch, nn, DataLoader, TensorDataset


def _squash(x: Any, dim: int = -1, eps: float = 1e-8) -> Any:
    squared_norm = (x * x).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * x / (squared_norm.sqrt() + eps)


def _make_capsule_class(nn: Any, torch: Any) -> type:
    class DigitCaps(nn.Module):
        def __init__(self, input_caps: int, input_dim: int, classes: int, output_dim: int = 12, routing: int = 3):
            super().__init__()
            self.classes = int(classes)
            self.routing = int(routing)
            self.weight = nn.Parameter(0.02 * torch.randn(input_caps, classes, output_dim, input_dim))

        def forward(self, x):
            # x: batch x input_caps x input_dim
            u_hat = torch.einsum("bid,icod->bico", x, self.weight)
            logits = torch.zeros(x.size(0), x.size(1), self.classes, 1, device=x.device, dtype=x.dtype)
            for step in range(self.routing):
                coeff = torch.softmax(logits, dim=2)
                s = (coeff * u_hat).sum(dim=1)
                v = _squash(s, dim=-1)
                if step < self.routing - 1:
                    logits = logits + (u_hat * v[:, None, :, :]).sum(dim=-1, keepdim=True)
            return v

    class EEGCapsuleNet(nn.Module):
        def __init__(self, channels: int, fs: int, classes: int, dropout: float = 0.35):
            super().__init__()
            temporal_kernel = max(16, int(round(float(fs) * 0.25)))
            primary_caps = 16
            primary_dim = 8
            pooled_steps = 8
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 32, (channels, 1), bias=False),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(dropout),
                nn.Conv2d(32, primary_caps * primary_dim, (1, 9), padding=(0, 4), bias=False),
                nn.BatchNorm2d(primary_caps * primary_dim),
                nn.ELU(),
                nn.AdaptiveAvgPool2d((1, pooled_steps)),
            )
            self.primary_caps = primary_caps
            self.primary_dim = primary_dim
            self.pooled_steps = pooled_steps
            self.digit_caps = DigitCaps(primary_caps * pooled_steps, primary_dim, classes)

        def forward(self, x):
            feat = self.features(x)
            batch = feat.size(0)
            feat = feat.view(batch, self.primary_caps, self.primary_dim, self.pooled_steps)
            feat = feat.permute(0, 1, 3, 2).contiguous().view(batch, self.primary_caps * self.pooled_steps, self.primary_dim)
            primary = _squash(feat, dim=-1)
            capsules = self.digit_caps(primary)
            return torch.sqrt(torch.sum(capsules * capsules, dim=-1) + 1e-8)

    return EEGCapsuleNet


def fit_predict_capsule_net(
    train_eeg: np.ndarray,
    train_y: np.ndarray,
    test_eeg: np.ndarray,
    fs: int,
    classes: int,
    seed: int = 42,
    max_epochs: int = 20,
    patience: int = 5,
) -> tuple[np.ndarray, dict[str, float]]:
    torch, nn, DataLoader, TensorDataset = _torch()
    set_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y = np.asarray(train_y, dtype=np.int64)
    train_idx, val_idx = validation_split(y, seed=seed)
    train_windows, train_labels, _ = make_trial_windows(train_eeg[train_idx], y[train_idx], fs, window_seconds=2.0, step_seconds=0.75)
    val_windows, val_labels, _ = make_trial_windows(train_eeg[val_idx], y[val_idx], fs, window_seconds=2.0, step_seconds=0.75)
    test_windows, _, test_trial_ids = make_trial_windows(
        test_eeg,
        np.zeros(len(test_eeg), dtype=np.int64),
        fs,
        window_seconds=2.0,
        step_seconds=0.75,
    )

    train_tensor = torch.from_numpy(train_windows[:, None, :, :])
    train_label_tensor = torch.from_numpy(train_labels)
    val_tensor = torch.from_numpy(val_windows[:, None, :, :]).to(device)
    val_label_tensor = torch.from_numpy(val_labels).to(device)
    test_tensor = torch.from_numpy(test_windows[:, None, :, :]).to(device)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_tensor, train_label_tensor),
        batch_size=min(32, len(train_tensor)),
        shuffle=True,
        generator=generator,
    )
    CapsuleNet = _make_capsule_class(nn, torch)
    model = CapsuleNet(train_windows.shape[1], fs, classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            scale = 0.9 + 0.2 * torch.rand((len(batch_x), 1, 1, 1), device=device)
            noise = 0.01 * torch.randn_like(batch_x)
            logits = model(batch_x * scale + noise)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_tensor), val_label_tensor).item())
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(test_tensor), dim=1).cpu().numpy()
    predictions = aggregate_trial_predictions(probabilities, test_trial_ids, len(test_eeg))
    metadata = {
        "best_epoch": float(best_epoch),
        "best_validation_loss": float(best_loss),
        "parameter_count": float(sum(parameter.numel() for parameter in model.parameters())),
        "device_is_cuda": float(device.type == "cuda"),
    }
    return predictions, metadata
