"""Additional common EEG deep-learning baselines backed by Braindecode.

Inputs use the bench's epoch convention ``(trials, channels, samples)``. Each
outer training fold is split into training/validation trials before overlapping
windows are made, so windows from one trial cannot leak across those subsets.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import numpy as np

from .deep import aggregate_trial_predictions, make_trial_windows, set_seed, validation_split


MODEL_CLASSES = {
    "shallow_fbcsp": "ShallowFBCSPNet",
    "deep4": "Deep4Net",
    "eegconformer": "EEGConformer",
    "fbcnet": "FBCNet",
    "atcnet": "ATCNet",
    "tcn": "TCN",
    "eegtcnet": "EEGTCNet",
    "eeginception_mi": "EEGInceptionMI",
    "eegnex": "EEGNeX",
    "eegitnet": "EEGITNet",
}

CUSTOM_MODELS = {"eeg_resnet", "inception_time"}


def _make_custom_temporal_model(name: str, channels: int, classes: int):
    """Build compact 1-D residual or InceptionTime-style EEG classifiers."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride=stride, padding=3, bias=False)
            self.norm1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, 7, padding=3, bias=False)
            self.norm2 = nn.BatchNorm1d(out_channels)
            self.skip = (nn.Identity() if in_channels == out_channels and stride == 1 else
                         nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                                       nn.BatchNorm1d(out_channels)))

        def forward(self, x):
            y = F.gelu(self.norm1(self.conv1(x)))
            y = self.norm2(self.conv2(y))
            return F.gelu(y + self.skip(x))

    class EEGResNet(nn.Module):
        """Residual temporal convolutions with a trial-level global readout."""
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(channels, 32, 15, stride=2, padding=7, bias=False),
                                      nn.BatchNorm1d(32), nn.GELU())
            self.blocks = nn.Sequential(ResidualBlock(32, 32), ResidualBlock(32, 64, 2),
                                        ResidualBlock(64, 64), ResidualBlock(64, 128, 2))
            self.head = nn.Linear(128, classes)

        def forward(self, x):
            return self.head(self.blocks(self.stem(x)).mean(dim=-1))

    class InceptionBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int = 24):
            super().__init__()
            bottleneck = max(16, in_channels // 2)
            self.bottleneck = nn.Conv1d(in_channels, bottleneck, 1, bias=False)
            self.paths = nn.ModuleList([
                nn.Conv1d(bottleneck, out_channels, kernel, padding=kernel // 2, bias=False)
                for kernel in (9, 19, 39)
            ])
            self.pool_path = nn.Conv1d(in_channels, out_channels, 1, bias=False)
            self.norm = nn.BatchNorm1d(out_channels * 4)

        def forward(self, x):
            y = self.bottleneck(x)
            pooled = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
            return F.gelu(self.norm(torch.cat([path(y) for path in self.paths] + [self.pool_path(pooled)], dim=1)))

    class InceptionTime(nn.Module):
        """Parallel temporal kernels capture short-to-long waveform motifs."""
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([InceptionBlock(channels), InceptionBlock(96), InceptionBlock(96)])
            self.skips = nn.ModuleList([nn.Conv1d(channels, 96, 1), nn.Identity(), nn.Identity()])
            self.head = nn.Linear(96, classes)

        def forward(self, x):
            for index, block in enumerate(self.blocks):
                x = block(x) + self.skips[index](x)
                if index < 2:
                    x = F.gelu(x)
            return self.head(x.mean(dim=-1))

    return EEGResNet() if name == "eeg_resnet" else InceptionTime()


def _build_model(name: str, model_class: Any, channels: int, samples: int,
                 classes: int, sampling_rate: float) -> Any:
    if name == "tcn":
        import torch.nn as nn

        class MeanTemporalReadout(nn.Module):
            """将 Braindecode TCN 的逐时间点分数平均为每 trial 一组类别分数。"""

            def __init__(self):
                super().__init__()
                self.network = model_class(n_chans=channels, n_outputs=classes)

            def forward(self, x):
                scores = self.network(x)
                return scores.mean(dim=-1) if scores.ndim == 3 else scores

        return MeanTemporalReadout()
    return model_class(n_chans=channels, n_outputs=classes, n_times=samples, sfreq=sampling_rate)


def available_models() -> tuple[str, ...]:
    """Names of supported Braindecode and local temporal baselines."""
    return tuple(MODEL_CLASSES) + tuple(sorted(CUSTOM_MODELS))


def make_braindecode_model(
    name: str,
    *,
    channels: int,
    samples: int,
    classes: int,
    sampling_rate: float,
) -> Any:
    """Create one supported Braindecode model for the supplied epoch shape.

    The function loads Braindecode lazily, so the base bench installation does
    not require it. Channel order and preprocessing remain the caller's contract.
    """
    if min(channels, samples, classes) < 1 or sampling_rate <= 0:
        raise ValueError("channels, samples, classes and sampling_rate must be positive")
    if name in CUSTOM_MODELS:
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("install the optional dependency with pip install -e '.[ml]'") from exc
        return _make_custom_temporal_model(name, channels, classes)
    try:
        from braindecode import models
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("install the optional dependency with pip install -e '.[braindecode]'") from exc
    try:
        class_name = MODEL_CLASSES[name]
    except KeyError as exc:
        raise ValueError(f"unknown model {name!r}; choose one of {available_models()}") from exc
    model_class = getattr(models, class_name, None)
    if model_class is None:  # pragma: no cover - incompatible optional dependency version
        raise RuntimeError(f"Braindecode does not provide {class_name}; install braindecode>=1.8,<2")
    return _build_model(name, model_class, channels, samples, classes, sampling_rate)


def fit_predict_braindecode(
    name: str,
    train_eeg: np.ndarray,
    train_y: np.ndarray,
    test_eeg: np.ndarray,
    sampling_rate: int,
    classes: int,
    *,
    seed: int = 42,
    max_epochs: int = 40,
    patience: int = 8,
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
    device: str | None = None,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Train one Braindecode baseline and return trial-level test predictions.

    ``train_eeg`` and ``test_eeg`` are ``(trials, channels, samples)`` arrays.
    Validation is split by trial within the supplied training fold; test labels
    are never accepted. Test-window probabilities are averaged per trial.
    """
    train_x = np.asarray(train_eeg, dtype=np.float32)
    test_x = np.asarray(test_eeg, dtype=np.float32)
    labels = np.asarray(train_y, dtype=np.int64)
    if train_x.ndim != 3 or test_x.ndim != 3:
        raise ValueError("train_eeg and test_eeg must have shape (trials, channels, samples)")
    if train_x.shape[1:] != test_x.shape[1:] or len(train_x) != len(labels):
        raise ValueError("training labels and train/test EEG shapes must agree")
    if len(train_x) < 2 or len(test_x) == 0:
        raise ValueError("at least two training trials and one test trial are required")
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch is required to train Braindecode models") from exc

    set_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    fit_idx, val_idx = validation_split(labels, seed=seed)
    train_windows, train_labels, _ = make_trial_windows(
        train_x[fit_idx], labels[fit_idx], sampling_rate, window_seconds, step_seconds
    )
    val_windows, val_labels, _ = make_trial_windows(
        train_x[val_idx], labels[val_idx], sampling_rate, window_seconds, step_seconds
    )
    test_windows, _, test_trial_ids = make_trial_windows(
        test_x, np.zeros(len(test_x), dtype=np.int64), sampling_rate, window_seconds, step_seconds
    )

    train_tensor = torch.from_numpy(train_windows)
    train_target = torch.from_numpy(train_labels)
    val_tensor = torch.from_numpy(val_windows).to(run_device)
    val_target = torch.from_numpy(val_labels).to(run_device)
    test_tensor = torch.from_numpy(test_windows).to(run_device)
    loader = DataLoader(
        TensorDataset(train_tensor, train_target), batch_size=min(32, len(train_tensor)),
        shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    model = make_braindecode_model(
        name, channels=train_x.shape[1], samples=train_windows.shape[-1],
        classes=classes, sampling_rate=sampling_rate,
    ).to(run_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(run_device)
            batch_y = batch_y.to(run_device)
            scale = 0.9 + 0.2 * torch.rand((len(batch_x), 1, 1), device=run_device)
            logits = model(batch_x * scale + 0.015 * torch.randn_like(batch_x))
            if logits.ndim != 2 or logits.shape[1] != classes:
                raise RuntimeError(f"{name} must return (batch, {classes}) class scores; got {tuple(logits.shape)}")
            loss = criterion(logits, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_tensor), val_target).item())
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(test_tensor), dim=1).cpu().numpy()
    predictions = aggregate_trial_predictions(probabilities, test_trial_ids, len(test_x))
    metadata: dict[str, float | str] = {
        "model": name,
        "best_epoch": float(best_epoch),
        "best_validation_loss": float(best_loss),
        "parameter_count": float(sum(parameter.numel() for parameter in model.parameters())),
        "device": str(run_device),
    }
    return predictions, metadata
