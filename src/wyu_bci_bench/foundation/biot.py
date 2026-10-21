"""Frozen BIOT features under the official 200 Hz montage contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from . import require_verified


BIOT_16_CHANNELS = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
)
BIOT_18_CHANNELS = BIOT_16_CHANNELS + ("C3-A2", "C4-A1")
TARGET_FS = 200
DEFAULT_SAMPLES = 2000


def prepare_input(
    X: np.ndarray,
    fs: float,
    channel_names: tuple[str, ...] | list[str],
    model_channels: tuple[str, ...] | list[str],
    *,
    sample_length: float = 10.0,
) -> np.ndarray:
    """Resample, order, and normalize epochs to the official BIOT input contract.

    ``X`` is ``(epochs, channels, samples)``. Channel identities must be the
    exact bipolar montage expected by the checkpoint. The official TUAB loader
    normalizes each epoch/channel by its own 95th absolute-amplitude percentile.
    """
    import scipy.signal

    require_verified("biot")
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or not np.isfinite(X).all():
        raise ValueError("X must be a finite (epochs, channels, samples) array")
    if fs <= 0 or sample_length <= 0:
        raise ValueError("fs and sample_length must be positive")
    channel_names = tuple(channel_names)
    model_channels = tuple(model_channels)
    if model_channels not in (BIOT_16_CHANNELS, BIOT_18_CHANNELS):
        raise ValueError("model_channels must be the official ordered BIOT 16- or 18-channel montage")
    if len(channel_names) != X.shape[1] or len(set(channel_names)) != len(channel_names):
        raise ValueError("channel_names must uniquely identify every channel in X")
    missing = [name for name in model_channels if name not in channel_names]
    if missing:
        raise ValueError(f"input is missing checkpoint montage channels: {missing}")
    if not np.isclose(X.shape[-1] / fs, sample_length, rtol=0, atol=0.5 / fs):
        raise ValueError(f"input duration {X.shape[-1] / fs:g}s does not match sample_length={sample_length:g}s")

    indices = [channel_names.index(name) for name in model_channels]
    ordered = X[:, indices, :]
    target_samples = int(round(sample_length * TARGET_FS))
    if fs != TARGET_FS or ordered.shape[-1] != target_samples:
        ordered = scipy.signal.resample(ordered, target_samples, axis=-1).astype(np.float32)
    scale = np.quantile(np.abs(ordered), 0.95, axis=-1, keepdims=True, method="linear")
    return ordered / (scale + 1e-8)


def load_encoder(
    repo: str | Path,
    weights: str | Path,
    *,
    n_channels: int,
    n_fft: int = 200,
    hop_length: int = 100,
):
    """Load the official BIOT encoder and a matching official checkpoint."""
    import torch

    require_verified("biot")
    if n_channels not in (16, 18):
        raise ValueError("BIOT checkpoints in the official repository use 16 or 18 channels")
    if n_fft != 200 or hop_length != 100:
        raise ValueError("official pretrained BIOT feature contract requires n_fft=200 and hop_length=100")
    repo = Path(repo)
    model_file = repo / "model" / "biot.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"official BIOT model file not found: {model_file}")
    spec = importlib.util.spec_from_file_location("wyu_bci_bench_official_biot", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official BIOT implementation from {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder = module.BIOTEncoder(n_channels=n_channels, n_fft=n_fft, hop_length=hop_length)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def extract_features(
    X: np.ndarray,
    fs: float,
    channel_names: tuple[str, ...] | list[str],
    model_channels: tuple[str, ...] | list[str],
    *,
    repo: str | Path,
    weights: str | Path,
    sample_length: float = 10.0,
    device: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract 256-D frozen BIOT embeddings; no task head or label is used."""
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    prepared = prepare_input(X, fs, channel_names, model_channels, sample_length=sample_length)
    encoder = load_encoder(repo, weights, n_channels=len(tuple(model_channels)))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(prepared), batch_size):
            tensor = torch.as_tensor(prepared[start:start + batch_size], dtype=torch.float32, device=device)
            embedding = encoder(tensor)
            if embedding.ndim != 2 or embedding.shape[1] != 256:
                raise RuntimeError(f"unexpected BIOT embedding shape: {tuple(embedding.shape)}")
            outputs.append(embedding.cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)
