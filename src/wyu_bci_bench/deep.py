"""可选 PyTorch EEGNet 试次级基线和窗口训练工具。

先在每个 trial 内标准化，再切成有重叠的训练窗口；预测阶段把同一 trial 的窗口
概率平均回 trial 级预测。validation_split 是当前训练折内部划分，不是外层测试集。
"""

from __future__ import annotations

import copy
import os
import random
from typing import Any

import numpy as np


def _torch() -> Any:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("PyTorch is required for EEGNet benchmarks") from exc
    return torch, nn, DataLoader, TensorDataset


def standardize_trials(eeg: np.ndarray) -> np.ndarray:
    """每个试次、每个通道独立沿时间做标准化。"""
    x = np.asarray(eeg, dtype=np.float32)
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return ((x - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def make_trial_windows(
    eeg: np.ndarray,
    labels: np.ndarray,
    fs: int,
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 ``(trial,channel,time)`` 切成有重叠窗口，并返回窗口标签和源试次编号。"""
    x = standardize_trials(eeg)
    y = np.asarray(labels, dtype=np.int64)
    window = min(x.shape[-1], max(32, int(round(float(window_seconds) * int(fs)))))
    step = max(1, int(round(float(step_seconds) * int(fs))))
    starts = list(range(0, x.shape[-1] - window + 1, step)) or [0]
    last = x.shape[-1] - window
    if starts[-1] != last:
        starts.append(last)
    windows: list[np.ndarray] = []
    window_labels: list[int] = []
    trial_ids: list[int] = []
    for trial_idx in range(len(x)):
        for start in starts:
            windows.append(x[trial_idx, :, start : start + window])
            window_labels.append(int(y[trial_idx]))
            trial_ids.append(trial_idx)
    return (
        np.stack(windows).astype(np.float32),
        np.asarray(window_labels, dtype=np.int64),
        np.asarray(trial_ids, dtype=np.int64),
    )


def set_seed(seed: int) -> None:
    torch, _, _, _ = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_eegnet_class(nn: Any) -> type:
    class EEGNet(nn.Module):
        def __init__(self, channels: int, fs: int, classes: int, dropout: float = 0.35):
            super().__init__()
            temporal_kernel = max(16, int(round(float(fs) * 0.5)))
            separable_kernel = max(8, int(round(float(fs) * 0.1)))
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(dropout),
                nn.Conv2d(
                    16,
                    16,
                    (1, separable_kernel),
                    padding=(0, separable_kernel // 2),
                    groups=16,
                    bias=False,
                ),
                nn.Conv2d(16, 16, (1, 1), bias=False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(dropout),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(16, int(classes))

        def forward(self, x):
            return self.classifier(self.features(x).flatten(1))

    return EEGNet


def _make_shallow_class(nn: Any) -> type:
    class ShallowConvNet(nn.Module):
        """Schirrmeister 2017 浅层卷积网：时间卷积→空间卷积→平方→平均池化→log→线性。

        核长按采样率换算（250 Hz 下 25/75/15 点 → ``fs`` 下等时长），纯 PyTorch 实现，
        不依赖 Braindecode，便于写进提交文件。
        """

        def __init__(self, channels: int, fs: float, classes: int, filters: int = 40, dropout: float = 0.5):
            super().__init__()
            k = max(5, int(round(float(fs) * 0.1)))
            pool = max(5, int(round(float(fs) * 0.3)))
            stride = max(2, int(round(float(fs) * 0.06)))
            self.temporal = nn.Conv2d(1, filters, (1, k), bias=True)
            self.spatial = nn.Conv2d(filters, filters, (channels, 1), bias=False)
            self.bn = nn.BatchNorm2d(filters)
            self.pool = nn.AvgPool2d((1, pool), stride=(1, stride))
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.LazyLinear(int(classes))

        def forward(self, x):
            import torch

            if x.ndim == 3:
                x = x.unsqueeze(1)
            z = self.bn(self.spatial(self.temporal(x)))
            z = torch.log(torch.clamp(self.pool(z * z), min=1e-6))
            return self.classifier(self.dropout(z).flatten(1))

    return ShallowConvNet


def make_shallow_convnet(channels: int, fs: float, classes: int, samples: int, **kwargs: Any):
    """构造纯 PyTorch ShallowConvNet，并用一次哑前向确定线性层维度。输入 ``(B,C,T)``。"""
    torch, nn, _, _ = _torch()
    model = _make_shallow_class(nn)(channels, fs, classes, **kwargs)
    with torch.no_grad():
        model.eval()
        model(torch.zeros(2, channels, samples))
    return model


def make_eegnet(channels: int, fs: float, classes: int, **kwargs: Any):
    """构造本模块的纯 PyTorch EEGNet。输入 ``(B,1,C,T)``。"""
    _, nn, _, _ = _torch()
    return _make_eegnet_class(nn)(channels, int(round(float(fs))), classes, **kwargs)


def validation_split(y: np.ndarray, seed: int = 42, fraction: float = 0.22) -> tuple[np.ndarray, np.ndarray]:
    """在当前训练折内部按类别抽出验证试次，不接触外层测试行。"""
    labels = np.asarray(y, dtype=np.int64)
    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        val_count = 1 if len(idx) >= 3 else 0
        val_count = max(val_count, int(round(len(idx) * float(fraction)))) if len(idx) >= 4 else val_count
        val.extend(int(v) for v in idx[:val_count])
        train.extend(int(v) for v in idx[val_count:])
    if not val:
        return np.asarray(train, dtype=np.int64), np.asarray(train, dtype=np.int64)
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(val), dtype=np.int64)


def aggregate_trial_probabilities(probabilities: np.ndarray, trial_ids: np.ndarray, trial_count: int) -> np.ndarray:
    """平均每个源试次的窗口概率，保留完整类别分布供后续融合。"""
    values = np.asarray(probabilities, dtype=np.float64)
    ids = np.asarray(trial_ids, dtype=np.int64)
    if values.ndim != 2 or ids.shape != (len(values),) or trial_count < 1:
        raise ValueError("probabilities must be 2D, trial_ids must match its rows, and trial_count must be positive")
    result = np.empty((trial_count, values.shape[1]), dtype=np.float64)
    for trial_idx in range(trial_count):
        mask = ids == trial_idx
        if not mask.any():
            raise ValueError(f"trial {trial_idx} has no window probabilities")
        result[trial_idx] = values[mask].mean(axis=0)
    return result


def aggregate_trial_predictions(probabilities: np.ndarray, trial_ids: np.ndarray, trial_count: int) -> np.ndarray:
    """平均每个源试次所有窗口的概率，再映射为 trial 级硬标签。"""
    return aggregate_trial_probabilities(probabilities, trial_ids, trial_count).argmax(axis=1).astype(np.int64)


def fit_predict_eegnet(
    train_eeg: np.ndarray,
    train_y: np.ndarray,
    test_eeg: np.ndarray,
    fs: int,
    classes: int,
    seed: int = 42,
    max_epochs: int = 40,
    patience: int = 8,
) -> tuple[np.ndarray, dict[str, float]]:
    """训练 EEGNet，依据内部验证损失早停，并汇总测试窗口到试次级预测。"""
    torch, nn, DataLoader, TensorDataset = _torch()
    set_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y = np.asarray(train_y, dtype=np.int64)
    train_idx, val_idx = validation_split(y, seed=seed)
    train_windows, train_labels, _ = make_trial_windows(train_eeg[train_idx], y[train_idx], fs)
    val_windows, val_labels, _ = make_trial_windows(train_eeg[val_idx], y[val_idx], fs)
    test_windows, _, test_trial_ids = make_trial_windows(test_eeg, np.zeros(len(test_eeg), dtype=np.int64), fs)

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
    EEGNet = _make_eegnet_class(nn)
    model = EEGNet(train_windows.shape[1], fs, classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
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
            noise = 0.015 * torch.randn_like(batch_x)
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
