"""EEGPT encoder features for named subsets of its pretrained channel montage.

The adapter owns channel ordering, epoch-duration checks, resampling and the
official encoder call. Dataset-specific filtering and amplitude scaling stay
explicit inputs because the authors' pretraining mixture does not define one
universal numeric scale for every source.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from . import require_verified


EEGPT_CHANNELS = (
    "FP1", "FPZ", "FP2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POZ", "PO4", "PO8", "O1", "OZ", "O2",
)
TARGET_FS = 256
SAMPLE_LENGTH = 4.0
TARGET_SAMPLES = 1024
PATCH_SIZE = 64


def _canonical_channels(channel_names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(str(name).upper().strip().rstrip(".") for name in channel_names)


def prepare_input(
    X: np.ndarray,
    fs: float,
    channel_names: tuple[str, ...] | list[str],
    *,
    amplitude_scale: float,
) -> np.ndarray:
    """Reorder, resample and explicitly scale EEGPT windows.

    ``X`` must contain 4-second epochs shaped ``(epochs, channels, samples)``.
    The caller chooses ``amplitude_scale`` from the dataset's documented
    preprocessing; the adapter deliberately has no default scale.
    """
    import scipy.signal

    require_verified("eegpt")
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[0] == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a non-empty finite (epochs, channels, samples) array")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and positive")
    if not np.isfinite(amplitude_scale) or amplitude_scale <= 0:
        raise ValueError("amplitude_scale must be finite and positive")
    if not np.isclose(X.shape[-1] / fs, SAMPLE_LENGTH, rtol=0, atol=0.5 / fs):
        raise ValueError(f"input duration {X.shape[-1] / fs:g}s does not match the official 4-second window")

    channels = _canonical_channels(channel_names)
    if len(channels) != X.shape[1] or len(set(channels)) != len(channels):
        raise ValueError("channel_names must uniquely identify every input channel")
    unknown = [name for name in channels if name not in EEGPT_CHANNELS]
    if unknown:
        raise ValueError(f"unknown pretrained EEGPT montage channels: {unknown}")
    selected = [name for name in EEGPT_CHANNELS if name in channels]
    ordered = X[:, [channels.index(name) for name in selected], :]
    if fs != TARGET_FS or ordered.shape[-1] != TARGET_SAMPLES:
        ordered = scipy.signal.resample(ordered, TARGET_SAMPLES, axis=-1).astype(np.float32)
    return (ordered * np.float32(amplitude_scale)).astype(np.float32, copy=False)


def load_encoder(repo: str | Path, weights: str | Path | None, *, channel_names=EEGPT_CHANNELS, seed=0):
    """Strictly load the official EEGPT encoder from a compatible safetensors file."""
    import torch
    from safetensors import safe_open
    from torch import nn
    from functools import partial

    require_verified("eegpt")
    repo = Path(repo)
    channels = _canonical_channels(channel_names)
    if not channels or len(set(channels)) != len(channels) or set(channels)-set(EEGPT_CHANNELS):
        raise ValueError("channel_names must be a unique named pretrained montage subset")
    model_file = repo / "downstream" / "Modules" / "models" / "EEGPT_mcae_finetune.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"official EEGPT model file not found: {model_file}")
    if weights is not None and not Path(weights).is_file():
        raise FileNotFoundError(f"EEGPT safetensors checkpoint not found: {weights}")

    spec = importlib.util.spec_from_file_location("wyu_bci_bench_official_eegpt", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official EEGPT implementation from {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(seed)
    encoder = module.EEGTransformer(
        img_size=[len(channels), TARGET_SAMPLES],
        patch_size=PATCH_SIZE,
        patch_stride=PATCH_SIZE,
        embed_dim=512,
        embed_num=4,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    if weights is not None:
        with safe_open(str(weights), framework="pt", device="cpu") as checkpoint:
            state = {
                key[len("target_encoder."):]: checkpoint.get_tensor(key)
                for key in checkpoint.keys()
                if key.startswith("target_encoder.")
            }
        if not state:
            raise ValueError("checkpoint contains no target_encoder.* tensors")
        encoder.load_state_dict(state, strict=True)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def extract_features(
    X: np.ndarray,
    fs: float,
    channel_names: tuple[str, ...] | list[str],
    *,
    repo: str | Path,
    weights: str | Path,
    amplitude_scale: float,
    device: str | None = None,
    batch_size: int = 16,
) -> np.ndarray:
    """Return 512-D mean-pooled frozen features; labels are never used."""
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    prepared = prepare_input(X, fs, channel_names, amplitude_scale=amplitude_scale)
    channels = [name for name in EEGPT_CHANNELS if name in _canonical_channels(channel_names)]
    encoder = load_encoder(repo, weights, channel_names=channels)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    chan_ids = encoder.prepare_chan_ids(channels).to(device)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(prepared), batch_size):
            tensor = torch.as_tensor(prepared[start:start + batch_size], dtype=torch.float32, device=device)
            tokens = encoder(tensor, chan_ids)
            if tuple(tokens.shape[1:]) != (16, 4, 512):
                raise RuntimeError(f"unexpected EEGPT token shape: {tuple(tokens.shape)}")
            outputs.append(tokens.mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)
