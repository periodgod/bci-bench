"""带有参与者、会话和设备接口分组信息的 EEG 窗口数据集。

The original ``EEGDataset`` targets within-subject K-fold on robot recordings.
Cross-subject and cross-session protocols (PROTEUS LOSO, NeuralBench sealed
sessions, OmniEEG subject-isolated splits) need every window to keep its group
keys, so they use ``GroupedEEGDataset`` instead.

新生读数据时，先记住主数组 ``X`` 的轴顺序是 ``(窗口数, 通道数, 采样点数)``，
而不是常见机器学习表格里的 ``(样本数, 特征数)``。一行 ``y[i]`` 与
``subjects[i]``、``sessions[i]``、``interfaces[i]`` 描述的是同一个窗口。
分组键不能丢：它们决定 LOSO 切分是否真正隔离参与者，也决定官方风格指标
如何给每个实验单元加权。
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

PROTEUS_CLASSES = ["motor_imagery", "mental_arithmetic", "word_generation"]
PROTEUS_FS = 120
PROTEUS_SHAPE = (41, 480)


@dataclass(frozen=True)
class GroupedEEGDataset:
    """一份窗口级 EEG 数据及其分组元数据。

    ``X`` 通常是微伏（uV），形状为 ``(N, C, T)``；``y`` 是长度 ``N`` 的整数
    类别编号。其余分组数组也必须长度为 ``N``，保证第 i 行的信号、标签和来源
    元信息保持对齐。``fs`` 是每秒采样点数（Hz），不是窗口长度。
    """

    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    interfaces: np.ndarray
    fs: float
    channel_names: list[str]
    class_names: list[str]
    unit: str = "uV"
    source: str = ""
    source_sha256: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
    recording_order: np.ndarray | None = None

    def __post_init__(self) -> None:
        """在对象创建时尽早检查数组行数与通道名，避免错误延迟到训练阶段。"""
        n = len(self.y)
        for name in ("X", "subjects", "sessions", "interfaces"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} rows, expected {n}")
        if self.recording_order is not None and len(self.recording_order) != n:
            raise ValueError("recording_order length does not match y")
        if self.X.ndim != 3:
            raise ValueError(f"X must be 3-D (windows, channels, samples), got {self.X.shape}")
        if self.X.shape[1] != len(self.channel_names):
            raise ValueError("channel_names length does not match X")

    @property
    def meta(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """按指标接口要求的固定次序返回分组键：参与者、会话、接口。"""
        return self.subjects, self.sessions, self.interfaces

    @property
    def people(self) -> list[str]:
        return sorted(set(self.subjects.tolist()))

    def subset(self, mask: np.ndarray) -> "GroupedEEGDataset":
        """按布尔掩码取子集，并同步切片信号、标签和全部分组列。

        只切 ``X`` 而忘记切分组列会导致静默错配；此方法集中处理这些同步操作。
        ``mask`` 的长度应与窗口数相同。
        """
        mask = np.asarray(mask)
        return GroupedEEGDataset(
            X=self.X[mask], y=self.y[mask], subjects=self.subjects[mask], sessions=self.sessions[mask],
            interfaces=self.interfaces[mask], fs=self.fs, channel_names=list(self.channel_names),
            class_names=list(self.class_names), unit=self.unit, source=self.source,
            source_sha256=self.source_sha256, metadata=dict(self.metadata),
            recording_order=None if self.recording_order is None else self.recording_order[mask],
        )

    def summary(self) -> dict[str, object]:
        """返回适合写入报告/日志的轻量数据概况，不包含原始 EEG 波形。"""
        return {
            "shape": list(self.X.shape),
            "fs": self.fs,
            "unit": self.unit,
            "classes": list(self.class_names),
            "class_counts": {str(k): int(v) for k, v in sorted(Counter(self.y.tolist()).items())},
            "subjects": self.people,
            "sessions": sorted(set(self.sessions.tolist())),
            "interfaces": sorted(set(self.interfaces.tolist())),
            "cells": len(set(zip(*(m.tolist() for m in self.meta)))),
            "source": self.source,
            "source_sha256": self.source_sha256,
        }


def file_sha256(path: str | Path) -> str:
    """分块计算文件 SHA-256，用于确认缓存来源未被意外替换。

    每次只读 1 MiB，因而大文件也不必整体载入内存。哈希用于追溯身份，
    不代表数据内容正确或实验协议有效。
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_grouped_npz(
    path: str | Path,
    *,
    fs: float,
    class_names: Sequence[str],
    x_key: str = "X",
    y_key: str = "y",
    subject_key: str = "subjects",
    session_key: str = "sessions",
    interface_key: str | None = "interfaces",
    channel_key: str | None = "channels",
    unit: str = "uV",
    with_hash: bool = True,
) -> GroupedEEGDataset:
    """从 NPZ 窗口缓存读取数组并组装成统一数据对象。

    ``*_key`` 参数允许不同数据集用不同字段名保存相同概念。文件未提供会话时
    暂记作 ``ses-1``；未提供接口时记作 ``all``，这只是缺省占位，不应被解释为
    数据真的只有一个实验会话或设备。``allow_pickle=False`` 避免加载任意对象。
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        X = data[x_key].astype(np.float32)
        y = data[y_key].astype(np.int64)
        subjects = data[subject_key].astype(str)
        sessions = data[session_key].astype(str) if session_key in data else np.full(len(y), "ses-1")
        if interface_key and interface_key in data:
            interfaces = data[interface_key].astype(str)
        else:
            interfaces = np.full(len(y), "all")
        if channel_key and channel_key in data:
            channels = [str(c) for c in data[channel_key].tolist()]
        else:
            channels = [f"ch{i}" for i in range(X.shape[1])]
        recording_order = data["recording_order"].copy() if "recording_order" in data else None
    return GroupedEEGDataset(
        X=X, y=y, subjects=subjects, sessions=sessions, interfaces=interfaces, fs=float(fs),
        channel_names=channels, class_names=list(class_names), unit=unit, source=str(path),
        source_sha256=file_sha256(path) if with_hash else "", recording_order=recording_order,
    )


def load_proteus_windows(path: str | Path, *, with_hash: bool = True) -> GroupedEEGDataset:
    """PROTEUS three-command cache: (N, 41, 480) microvolts at 120 Hz, cue + 1 s to cue + 5 s.

    新生提示：120 Hz 下 480 个点对应 4 秒信号。窗口从提示 cue 后 1 秒开始，
    所以这里读入的不是从 cue 起始直接截取的前 4 秒。这个函数只读取和验证缓存，
    不会重新执行滤波、重采样或窗口截取。

    The cache is produced by ``challenge2026_track02/proteus_update1/train_proteus_update1.py``
    (notch -> filter -> resample to 120 Hz -> volts * 1e6 -> 480 samples after cue + 1 s).
    """
    dataset = load_grouped_npz(path, fs=PROTEUS_FS, class_names=PROTEUS_CLASSES, with_hash=with_hash)
    if dataset.X.shape[1:] != PROTEUS_SHAPE:
        raise ValueError(f"PROTEUS cache must be (N, 41, 480), got {dataset.X.shape}")
    if not np.isfinite(dataset.X).all():
        raise ValueError("PROTEUS cache contains non-finite samples")
    if set(np.unique(dataset.y).tolist()) - {0, 1, 2}:
        raise ValueError("PROTEUS labels must be 0/1/2")
    return dataset
