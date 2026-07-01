"""Loading and merging local EEG recordings."""

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
    eeg: np.ndarray
    labels: np.ndarray
    y: np.ndarray
    fs: int
    channel_names: list[str]
    label_names: list[str]
    paths: list[Path]
    source_trial_counts: list[int]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.eeg.shape)

    def summary(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "sampling_rate_hz": int(self.fs),
            "channels": list(self.channel_names),
            "label_names": list(self.label_names),
            "label_counts": dict(Counter(str(label) for label in self.labels)),
            "source_files": [str(path) for path in self.paths],
            "source_trial_counts": [int(v) for v in self.source_trial_counts],
        }


def _decode_item(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _first_existing_key(files: Sequence[str], candidates: Sequence[str]) -> str | None:
    lower = {name.lower(): name for name in files}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _scalar(data: np.lib.npyio.NpzFile, key: str | None, default: object = None) -> object:
    if key is None:
        return default
    value = np.asarray(data[key])
    if value.shape == ():
        return value.item()
    if value.size == 1:
        return value.reshape(-1)[0].item()
    return default


def _as_string_array(values: np.ndarray, length: int | None = None, default: str = "unknown") -> np.ndarray:
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
    if key is None:
        return [f"Ch{idx + 1}" for idx in range(channels)]
    names = [_decode_item(item) for item in np.asarray(data[key], dtype=object).reshape(-1)]
    if len(names) != channels:
        return [f"Ch{idx + 1}" for idx in range(channels)]
    return names


def load_npz_recording(path: str | Path) -> Recording:
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
    normalized = [normalize_label(label, mode) for label in labels]
    keep = np.asarray([item is not None for item in normalized], dtype=bool)
    kept = np.asarray([item for item in normalized if item is not None], dtype=object)
    names = label_order([str(item) for item in kept], mode=mode)
    lookup = {name: idx for idx, name in enumerate(names)}
    y = np.asarray([lookup[str(item)] for item in kept], dtype=np.int64)
    return kept, y, names


def _requested_channels(channel_mode: str, requested_channels: Sequence[str] | None) -> Sequence[str] | None:
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
    files = scan_recordings(paths)
    recordings = [load_npz_recording(path) for path in files]
    return merge_recordings(
        recordings,
        label_mode=label_mode,
        channel_mode=channel_mode,
        requested_channels=requested_channels,
    )
