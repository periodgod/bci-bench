"""读取、核对并合并本地单段 EEG 试次文件。

文件约定为 NPZ；加载器会识别若干常见键名，但不会猜测信号单位或重采样。
``Recording.eeg`` 与 ``EEGDataset.eeg`` 的轴顺序均为 ``(试次数, 通道数, 采样点)``。
跨文件合并前会检查采样率和通道选择是否一致，防止把不兼容信号拼成一个矩阵。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .signal import (
    CENTRAL3_CHANNELS,
    MOTOR11_CHANNELS,
    MOTOR_CHANNELS,
    crop_to_min_samples,
    ensure_3d_epochs,
    select_channel_indices,
)


BINARY_MI_ALIASES = {
    "left": "left_hand_mi",
    "left_hand": "left_hand_mi",
    "left_hand_mi": "left_hand_mi",
    "left mi": "left_hand_mi",
    "right": "right_hand_mi",
    "right_hand": "right_hand_mi",
    "right_hand_mi": "right_hand_mi",
    "right mi": "right_hand_mi",
}

MOVEMENT_ALIASES = {
    "forward": "forward",
    "backward": "backward",
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down",
    "rest": "rest",
}


@dataclass(frozen=True)
class Recording:
    """一个源文件读取后的原始试次集合；标签仍保留文件中的文本形式。"""
    path: Path
    eeg: np.ndarray
    labels: np.ndarray
    fs: int
    channel_names: list[str]
    aux: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.eeg.shape)


@dataclass(frozen=True)
class EEGDataset:
    """多个 Recording 清理并拼接后的训练数据。

    ``labels`` 是规范化后的字符串，``y`` 是与其一一对应的整数类别编码；
    ``label_names[y[i]]`` 可还原类别名称。通道顺序、采样率和时间长度对合并后
    的所有试次保持一致。
    """
    eeg: np.ndarray
    labels: np.ndarray
    y: np.ndarray
    fs: int
    channel_names: list[str]
    label_names: list[str]
    paths: list[Path]
    source_trial_counts: list[int]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.eeg.shape)

    def summary(self) -> dict[str, object]:
        """生成不含波形本身的数据清点信息，便于写实验日志。"""
        payload = {
            "shape": list(self.shape),
            "sampling_rate_hz": int(self.fs),
            "channels": list(self.channel_names),
            "label_names": list(self.label_names),
            "label_counts": dict(Counter(str(label) for label in self.labels)),
            "source_files": [str(path) for path in self.paths],
            "source_trial_counts": [int(v) for v in self.source_trial_counts],
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


def _decode_item(value: object) -> str:
    """将 NPZ 中的 bytes、NumPy 标量和普通对象统一转换为文本。"""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _first_existing_key(files: Sequence[str], candidates: Sequence[str]) -> str | None:
    """按候选优先级做大小写不敏感的键名匹配。"""
    lower = {name.lower(): name for name in files}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _scalar(data: np.lib.npyio.NpzFile, key: str | None, default: object = None) -> object:
    """只接受单值数组作为元数据标量，避免把多值字段静默压成一个值。"""
    if key is None:
        return default
    value = np.asarray(data[key])
    if value.shape == ():
        return value.item()
    if value.size == 1:
        return value.reshape(-1)[0].item()
    return default


def _as_string_array(values: np.ndarray, length: int | None = None, default: str = "unknown") -> np.ndarray:
    """把标签展平成文本数组，并可检查/广播到指定试次数。"""
    arr = np.asarray(values, dtype=object).reshape(-1)
    if arr.size == 0 and length is not None:
        arr = np.asarray([default] * length, dtype=object)
    if length is not None:
        if arr.size == 1 and length > 1:
            arr = np.asarray([arr[0]] * length, dtype=object)
        if arr.size != length:
            raise ValueError(f"Label count {arr.size} does not match trial count {length}")
    return np.asarray([_decode_item(item) for item in arr], dtype=object)


def _channel_names(data: np.lib.npyio.NpzFile, key: str | None, channels: int) -> list[str]:
    """读取通道名；字段缺失或数量不匹配时使用稳定的 Ch1、Ch2... 占位名。"""
    if key is None:
        return [f"Ch{idx + 1}" for idx in range(channels)]
    names = [_decode_item(item) for item in np.asarray(data[key], dtype=object).reshape(-1)]
    if len(names) != channels:
        return [f"Ch{idx + 1}" for idx in range(channels)]
    return names


def load_npz_recording(path: str | Path) -> Recording:
    """从 NPZ 读取一份记录，识别 EEG、标签、采样率、通道名和辅助信号字段。

    该函数以试次数为第一维标准化二维/三维输入，但不会改滤波、单位、基准或
    采样率。输入文件可信时才能打开 pickle 对象，因为兼容来源可能含对象数组。
    """
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        files = list(data.files)
        eeg_key = _first_existing_key(files, ("eeg", "x", "epochs", "trials", "data"))
        if eeg_key is None:
            raise ValueError(f"{path} does not contain an EEG array key")
        eeg = ensure_3d_epochs(np.asarray(data[eeg_key], dtype=np.float32), eeg_key)
        label_key = _first_existing_key(files, ("y", "labels", "label", "targets", "trial_labels"))
        if label_key is None:
            labels = np.asarray(["unknown"] * eeg.shape[0], dtype=object)
        else:
            labels = _as_string_array(np.asarray(data[label_key], dtype=object), length=eeg.shape[0])
        fs_key = _first_existing_key(files, ("fs", "sfreq", "sampling_rate", "sampling_rate_hz"))
        fs = int(round(float(_scalar(data, fs_key, 250))))
        channel_key = _first_existing_key(
            files,
            ("channel_names", "ch_names", "channels", "eeg_channel_names"),
        )
        channels = _channel_names(data, channel_key, eeg.shape[1])
        aux_key = _first_existing_key(files, ("aux", "emg", "auxiliary"))
        aux = None
        if aux_key is not None:
            aux = ensure_3d_epochs(np.asarray(data[aux_key], dtype=np.float32), aux_key)
        metadata = {
            "keys": files,
            "eeg_key": eeg_key,
            "label_key": label_key,
            "fs_key": fs_key,
            "channel_key": channel_key,
            "aux_key": aux_key,
        }
    return Recording(path=path, eeg=eeg, labels=labels, fs=fs, channel_names=channels, aux=aux, metadata=metadata)


def scan_recordings(paths: str | Path | Iterable[str | Path], pattern: str = "*.npz") -> list[Path]:
    """扫描一个或多个文件/目录，递归发现符合 pattern 的 NPZ，并按路径排序。"""
    if isinstance(paths, (str, Path)):
        roots: list[Path] = [Path(paths)]
    else:
        roots = [Path(item) for item in paths]
    found: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".npz":
            found.append(root)
        elif root.is_dir():
            found.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted(set(found))


def summarize_recordings(paths: str | Path | Iterable[str | Path]) -> list[dict[str, object]]:
    """逐文件尝试读取摘要；单个文件错误被记录下来，不中断整批盘点。"""
    summaries = []
    for path in scan_recordings(paths):
        try:
            recording = load_npz_recording(path)
            summaries.append(
                {
                    "path": str(path),
                    "shape": list(recording.shape),
                    "sampling_rate_hz": int(recording.fs),
                    "channels": len(recording.channel_names),
                    "label_counts": dict(Counter(str(label) for label in recording.labels)),
                }
            )
        except Exception as exc:
            summaries.append({"path": str(path), "error": str(exc)})
    return summaries


def normalize_label(label: object, mode: str = "all") -> str | None:
    """把常见标签别名映射为规范名称；返回 None 表示当前任务应丢弃该试次。"""
    text = _decode_item(label).strip()
    key = text.lower()
    if mode == "binary_mi":
        return BINARY_MI_ALIASES.get(key)
    if mode == "six_movement":
        value = MOVEMENT_ALIASES.get(key)
        return value if value != "rest" else None
    explicit_binary = {
        "left_hand",
        "left_hand_mi",
        "left mi",
        "right_hand",
        "right_hand_mi",
        "right mi",
    }
    if key in explicit_binary:
        return BINARY_MI_ALIASES[key]
    if key in MOVEMENT_ALIASES:
        return MOVEMENT_ALIASES[key]
    return text or None


def label_order(labels: Sequence[str], mode: str = "all") -> list[str]:
    """生成确定性的类别顺序；二分类/动作分类优先遵循项目约定顺序。"""
    if mode == "binary_mi":
        order = ["left_hand_mi", "right_hand_mi"]
    elif mode == "six_movement":
        order = ["forward", "backward", "left", "right", "up", "down"]
    else:
        order = []
    present = set(str(item) for item in labels)
    ordered = [item for item in order if item in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def encode_labels(labels: Sequence[object], mode: str = "all") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """规范化并编码标签，返回保留文本、整数 y 和整数索引到名称的列表。

    被过滤的标签不会出现在前三个返回值中；调用方必须用相同 keep 掩码过滤 EEG，
    否则信号与 y 将错位。
    """
    normalized = [normalize_label(label, mode) for label in labels]
    keep = np.asarray([item is not None for item in normalized], dtype=bool)
    kept = np.asarray([item for item in normalized if item is not None], dtype=object)
    names = label_order([str(item) for item in kept], mode=mode)
    lookup = {name: idx for idx, name in enumerate(names)}
    y = np.asarray([lookup[str(item)] for item in kept], dtype=np.int64)
    return kept, y, names


def _requested_channels(channel_mode: str, requested_channels: Sequence[str] | None) -> Sequence[str] | None:
    """把便捷通道模式转换成明确的通道名单；显式名单优先于预设模式。"""
    if requested_channels:
        return list(requested_channels)
    if channel_mode == "all":
        return None
    if channel_mode == "motor":
        return MOTOR_CHANNELS
    if channel_mode == "motor11":
        return MOTOR11_CHANNELS
    if channel_mode == "central3":
        return CENTRAL3_CHANNELS
    raise ValueError(f"Unknown channel_mode: {channel_mode}")


def merge_recordings(
    recordings: Sequence[Recording],
    label_mode: str = "all",
    channel_mode: str = "motor11",
    requested_channels: Sequence[str] | None = None,
) -> EEGDataset:
    """选择一致通道、裁成共同长度、拼接源文件并过滤/编码标签。

    这里只沿每段末尾裁剪到最短采样点数，不做时间重采样；采样率不同会直接报错。
    这避免插值算法悄悄改变实验输入，但也意味着调用者需事先统一采样率。
    ``source_trial_counts`` 记录各源文件原有试次数，不是过滤后的类别计数。
    """
    if not recordings:
        raise ValueError("No recordings to merge")
    fs = int(recordings[0].fs)
    requested = _requested_channels(channel_mode, requested_channels)
    selected_arrays: list[np.ndarray] = []
    selected_labels: list[np.ndarray] = []
    source_counts: list[int] = []
    selected_names: list[str] | None = None
    paths: list[Path] = []
    for recording in recordings:
        if int(recording.fs) != fs:
            raise ValueError(f"Sampling rate mismatch: {recording.path} has {recording.fs}, expected {fs}")
        if requested is None:
            idx = np.arange(len(recording.channel_names), dtype=np.int64)
            names = list(recording.channel_names)
        else:
            idx, names = select_channel_indices(recording.channel_names, requested, min_channels=3)
        if selected_names is None:
            selected_names = list(names)
        elif list(names) != selected_names:
            raise ValueError(f"Channel selection mismatch in {recording.path}")
        selected_arrays.append(recording.eeg[:, idx, :])
        selected_labels.append(recording.labels)
        source_counts.append(int(recording.eeg.shape[0]))
        paths.append(recording.path)
    selected_arrays = crop_to_min_samples(selected_arrays)
    eeg = np.concatenate(selected_arrays, axis=0).astype(np.float32)
    labels = np.concatenate(selected_labels, axis=0)
    kept_labels, y, names = encode_labels(labels, mode=label_mode)
    keep = np.asarray([normalize_label(label, label_mode) is not None for label in labels], dtype=bool)
    if len(names) < 2:
        raise ValueError(f"Need at least two labels after filtering; got {names}")
    return EEGDataset(
        eeg=eeg[keep],
        labels=kept_labels,
        y=y,
        fs=fs,
        channel_names=selected_names or [],
        label_names=names,
        paths=paths,
        source_trial_counts=source_counts,
    )


def load_dataset(
    paths: str | Path | Iterable[str | Path],
    label_mode: str = "all",
    channel_mode: str = "motor11",
    requested_channels: Sequence[str] | None = None,
) -> EEGDataset:
    """扫描目录、加载每个 NPZ，然后调用 ``merge_recordings`` 完成统一处理。"""
    files = scan_recordings(paths)
    recordings = [load_npz_recording(path) for path in files]
    return merge_recordings(
        recordings,
        label_mode=label_mode,
        channel_mode=channel_mode,
        requested_channels=requested_channels,
    )
