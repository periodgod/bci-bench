"""CBraMod frozen-feature extraction under the verified 4 s / 200 Hz / uV-over-100 contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from . import require_verified

PATCH = 200
TARGET_FS = 200


def prepare_input(X_uV: np.ndarray, fs: float) -> np.ndarray:
    """按已核验合同把 bench 窗口变成 CBraMod 输入。

    顺序是沿通道做 CAR、微伏除以 100、重采样至 200 Hz，再拆成整秒 patch；
    输出形状 ``(N,C,P,200)``。例如 4 秒输入得到 ``P=4``。重采样只改变采样网格，
    不会恢复原始 120 Hz 数据在 60 Hz 奈奎斯特频率以上的信号成分。
    """
    import scipy.signal

    require_verified("cbramod")
    X = np.asarray(X_uV, dtype=np.float32)
    X = X - X.mean(axis=1, keepdims=True)
    X = X / 100.0
    n_target = int(round(X.shape[-1] * TARGET_FS / fs))
    if n_target % PATCH:
        raise ValueError(f"{X.shape[-1]} samples @ {fs} Hz -> {n_target} @ 200 Hz is not a whole number of 1 s patches")
    X = scipy.signal.resample(X, n_target, axis=-1).astype(np.float32)
    return X.reshape(X.shape[0], X.shape[1], n_target // PATCH, PATCH)


def load_cbramod(repo: str | Path, weights: str | Path | None, seed: int | None = None):
    """从指定源码目录构造模型并可选加载权重。

    ``weights=None`` 且显式给 seed 时生成同结构随机初始化对照；seed 仅控制可复现
    初始化，不表示预训练权重。加载使用 strict=True，权重键不完全匹配会报错。
    """
    import torch

    repo = str(Path(repo))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from models.cbramod import CBraMod  # type: ignore

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    model = CBraMod()
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.proj_out = torch.nn.Identity()
    return model.eval()


def extract_features(X_uV: np.ndarray, fs: float, repo: str | Path, weights: str | Path | None,
                     seed: int | None = None, device: str | None = None, batch: int = 32) -> np.ndarray:
    """提取每窗 200 维嵌入；模型 token 在通道和时间 patch 维上取平均。

    传入已训练权重得到预训练表征；传 ``weights=None, seed=s`` 得随机初始化对照。
    后续下游分类器必须在各训练折中拟合，且应将随机对照与预训练表征使用相同协议。
    """
    import torch

    if weights is None and seed is None:
        raise ValueError("random-initialisation control needs an explicit seed")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cbramod(repo, weights, seed).to(device)
    out = []
    with torch.inference_mode():
        for start in range(0, len(X_uV), batch):
            x = torch.as_tensor(prepare_input(X_uV[start:start + batch], fs), device=device)
            tokens = model(x)
            if tuple(tokens.shape[1:]) != tuple(x.shape[1:]):
                raise RuntimeError(f"unexpected CBraMod output {tuple(tokens.shape)}")
            out.append(tokens.mean(dim=(1, 2)).cpu().numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float32)


def extract_tokens(X_uV: np.ndarray, fs: float, repo: str | Path, weights: str | Path | None,
                   device: str | None = None, batch: int = 16, *, seed: int | None = None,
                   before_last: bool = False) -> np.ndarray:
    """Return full (N,C,4,200) tokens or deterministic frozen-prefix activations.

    ``before_last`` stops immediately before the last encoder block, allowing
    exact last-block-only fine tuning with the frozen prefix in evaluation mode.
    No training-data statistics are fitted during extraction.
    """
    import torch
    if weights is None and seed is None:
        raise ValueError("random-initialisation control needs an explicit seed")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cbramod(repo, weights, seed).to(device)
    out = []
    with torch.inference_mode():
        for start in range(0, len(X_uV), batch):
            x = torch.as_tensor(prepare_input(X_uV[start:start+batch], fs), device=device)
            if before_last:
                tokens = model.patch_embedding(x)
                for layer in model.encoder.layers[:-1]:
                    tokens = layer(tokens)
            else:
                tokens = model(x)
            if tuple(tokens.shape[1:]) != tuple(x.shape[1:]) or tokens.shape[2] != 4 or tokens.shape[3] != 200:
                raise RuntimeError(f"unexpected CBraMod output {tuple(tokens.shape)}")
            out.append(tokens.cpu().numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float32)


def pool_tokens(tokens: np.ndarray, mode: str, regions=None) -> np.ndarray:
    """Pool full tokens globally, by time, or by supplied channel regions/halves."""
    t = np.asarray(tokens)
    if t.ndim != 4 or t.shape[2] != 4:
        raise ValueError("expected N,C,4,D tokens")
    if mode == "global":
        return t.mean(axis=(1, 2))
    if mode == "time":
        return t.mean(axis=1).reshape(len(t), -1)
    if mode != "region_phase" or regions is None:
        raise ValueError("mode must be global, time or region_phase with channel groups")
    groups = [np.asarray(g, dtype=int) for g in regions]
    flat = np.concatenate(groups)
    if any(len(g) == 0 for g in groups) or sorted(flat.tolist()) != list(range(t.shape[1])):
        raise ValueError("regions must partition channels exactly once")
    return np.concatenate([t[:, group, half].mean(axis=(1, 2)) for group in groups
                           for half in (slice(0, 2), slice(2, 4))], axis=1)


def make_last_block_classifier(repo, weights, *, seed=0, classes=3):
    """Trainable last encoder block plus time-preserving linear head.

    Input is a cached evaluation-mode frozen-prefix activation; this removes
    redundant prefix evaluation without changing the computation being trained.
    """
    import torch
    from torch import nn
    model = load_cbramod(repo, weights, seed)
    class LastBlockReadout(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block
            self.head = nn.Linear(800, classes)

        def forward(self, tokens):
            return self.head(self.block(tokens).mean(dim=1).flatten(1))
    return LastBlockReadout(model.encoder.layers[-1])
