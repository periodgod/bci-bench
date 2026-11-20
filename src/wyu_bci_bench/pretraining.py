"""Unlabelled, sharded EEG corpora and channel-coordinate resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .grouped import file_sha256


def read_unlabelled_edf(path: str | Path):
    """Load an EDF as continuous EEG and discard annotations before callers see it."""
    import mne

    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    ecg_channels = {name: "ecg" for name in raw.ch_names if "ecg" in name.casefold()}
    if ecg_channels:
        raw.set_channel_types(ecg_channels, verbose="ERROR")
    raw.set_annotations(mne.Annotations([], [], []))
    return raw


@dataclass(frozen=True)
class PretrainingShard:
    """One memory-mapped, single-dataset signal shard with ordered group keys."""

    name: str
    dataset: str
    X: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    recording_order: np.ndarray
    channel_names: tuple[str, ...]
    fs: float
    unit: str
    source_path: Path
    source_sha256: str

    def __post_init__(self) -> None:
        n = self.X.shape[0]
        if self.X.ndim != 3:
            raise ValueError("shard signals must have shape (windows, channels, samples)")
        if self.X.shape[1] != len(self.channel_names):
            raise ValueError("channel_names length does not match signal channels")
        for field in (self.subjects, self.sessions, self.recording_order):
            if len(field) != n:
                raise ValueError("shard index arrays must align with signal windows")
        if self.recording_order.ndim != 1 or not np.issubdtype(self.recording_order.dtype, np.number):
            raise ValueError("recording_order must be a numeric one-dimensional array")

    def ordered_session_indices(self) -> dict[tuple[str, str], np.ndarray]:
        """Map subject/session to stable row indices ordered by recording time."""
        keys = sorted(set(zip(self.subjects.astype(str).tolist(), self.sessions.astype(str).tolist())))
        result = {}
        for subject, session in keys:
            rows = np.flatnonzero((self.subjects.astype(str) == subject) & (self.sessions.astype(str) == session))
            result[(subject, session)] = rows[np.argsort(self.recording_order[rows], kind="stable")]
        return result


@dataclass(frozen=True)
class PretrainingCorpus:
    """Validated set of independent memory-mapped corpus shards."""

    shards: tuple[PretrainingShard, ...]
    manifest_path: Path
    manifest_sha256: str

    @property
    def n_windows(self) -> int:
        return sum(shard.X.shape[0] for shard in self.shards)

    def close(self) -> None:
        """Release all memory-mapped signal files owned by this corpus."""
        for shard in self.shards:
            mmap = getattr(shard.X, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self) -> "PretrainingCorpus":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def load_pretraining_corpus(manifest_path: str | Path, *, verify_hashes: bool = True) -> PretrainingCorpus:
    """Load a JSON manifest of float16/float32 ``.npy`` shards and label-free indices.

    Each manifest shard provides ``name``, ``dataset``, ``signals``, ``index``,
    ``channels``, ``fs``, ``unit``, and optional SHA-256 fields. Index NPZ files
    contain only ``subjects``, ``sessions`` and numeric ``recording_order``.
    Paths are resolved relative to the manifest; no EEG labels are loaded.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = []
    names = set()
    for item in manifest.get("shards", []):
        name = str(item["name"])
        if name in names:
            raise ValueError(f"duplicate shard name: {name}")
        names.add(name)
        signals_path = (manifest_path.parent / item["signals"]).resolve()
        index_path = (manifest_path.parent / item["index"]).resolve()
        if verify_hashes:
            expected = item.get("sha256", {})
            for key, path in (("signals", signals_path), ("index", index_path)):
                if expected.get(key) and file_sha256(path) != expected[key]:
                    raise ValueError(f"SHA-256 mismatch for shard {name} {key}")
        X = np.load(signals_path, mmap_mode="r", allow_pickle=False)
        with np.load(index_path, allow_pickle=False) as index:
            missing = {"subjects", "sessions", "recording_order"} - set(index.files)
            if missing:
                raise KeyError(f"shard {name} index lacks {sorted(missing)}")
            if {"y", "labels", "events"} & set(index.files):
                raise ValueError(f"shard {name} index must not contain task labels or events")
            subjects = index["subjects"].astype(str)
            sessions = index["sessions"].astype(str)
            order = index["recording_order"].copy()
        if X.dtype not in (np.dtype("float16"), np.dtype("float32")):
            raise ValueError(f"shard {name} signals must be float16 or float32, got {X.dtype}")
        shards.append(PretrainingShard(
            name=name, dataset=str(item["dataset"]), X=X, subjects=subjects, sessions=sessions,
            recording_order=order, channel_names=tuple(map(str, item["channels"])),
            fs=float(item["fs"]), unit=str(item["unit"]), source_path=signals_path,
            source_sha256=file_sha256(signals_path) if verify_hashes else "",
        ))
    if not shards:
        raise ValueError("pretraining manifest contains no shards")
    return PretrainingCorpus(tuple(shards), manifest_path, file_sha256(manifest_path))


def resolve_channel_coordinates(channel_names: list[str] | tuple[str, ...], *,
                                montage_name: str = "standard_1005") -> tuple[np.ndarray, np.ndarray]:
    """Return MNE montage XYZ coordinates in metres and a mask for known channels.

    Unknown channels receive NaN coordinates and a false mask; callers must drop
    those channels and record them in the corpus manifest before model input.
    """
    try:
        import mne
    except ImportError as exc:  # pragma: no cover
        raise ImportError("channel coordinates require MNE") from exc
    positions = mne.channels.make_standard_montage(montage_name).get_positions()["ch_pos"]
    coords = np.full((len(channel_names), 3), np.nan, dtype=np.float32)
    known = np.zeros(len(channel_names), dtype=bool)
    canonical = {str(name).casefold(): value for name, value in positions.items()}
    aliases = {"t3": "T7", "t4": "T8", "t5": "P7", "t6": "P8"}
    for i, name in enumerate(channel_names):
        normalized = str(name).strip()
        if normalized.casefold().startswith("eeg "):
            normalized = normalized[4:].strip()
        normalized = aliases.get(normalized.casefold(), normalized)
        point = canonical.get(normalized.casefold())
        if point is not None:
            coords[i] = np.asarray(point, dtype=np.float32)
            known[i] = True
    return coords, known


def preprocess_unlabelled_recording(raw, *, target_fs: float = 120.0,
                                    window_samples: int = 480, step_samples: int = 240,
                                    amplitude_limit_uv: float = 500.0,
                                    flat_std_limit_uv: float = 0.5) -> tuple[np.ndarray, dict]:
    """Filter a continuous MNE recording and return valid, label-free EEG windows.

    Output shape is (windows, known channels, samples), float16, in 100-uV units
    clipped to [-5, 5]. Unknown-montage channels and rejected windows are
    summarized in metadata. The input Raw is copied and remains unchanged.
    """
    import mne

    recording = raw.copy().pick("eeg")
    original_names = list(recording.ch_names)
    coordinates, known = resolve_channel_coordinates(original_names)
    if not known.any():
        return np.empty((0, 0, window_samples), dtype=np.float16), {
            "channel_names": [], "coordinates_m": [], "dropped_channels": original_names,
            "n_windows_total": 0, "n_rejected_amplitude": 0, "n_rejected_flat": 0,
        }
    dropped = [original_names[i] for i in np.flatnonzero(~known)]
    if not known.all():
        recording.pick([original_names[i] for i in np.flatnonzero(known)])
        coordinates = coordinates[known]
    fs = float(recording.info["sfreq"])
    notch = [frequency for frequency in (50.0, 60.0) if frequency < fs / 2 - 1]
    if notch:
        recording.notch_filter(freqs=notch, verbose="ERROR")
    recording.filter(l_freq=0.1, h_freq=min(75.0, fs / 2 - 1), verbose="ERROR")
    recording.resample(target_fs, npad="auto", verbose="ERROR")
    signal = recording.get_data().astype(np.float32) * 1e6
    count = max(0, 1 + (signal.shape[1] - window_samples) // step_samples)
    kept = []
    rejected_amplitude = 0
    rejected_flat = 0
    for start in range(count):
        window = signal[:, start * step_samples:start * step_samples + window_samples].copy()
        window -= window.mean(axis=0, keepdims=True)
        flat = bool(window.std(axis=1).min() < flat_std_limit_uv)
        amplitude = bool(np.max(np.abs(window)) > amplitude_limit_uv)
        rejected_flat += int(flat)
        rejected_amplitude += int(amplitude and not flat)
        if not flat and not amplitude:
            kept.append(np.clip(window / 100.0, -5.0, 5.0).astype(np.float16))
    windows = np.stack(kept) if kept else np.empty((0, signal.shape[0], window_samples), dtype=np.float16)
    return windows, {
        "channel_names": list(recording.ch_names), "coordinates_m": coordinates.tolist(),
        "dropped_channels": dropped, "n_windows_total": count,
        "n_rejected_amplitude": rejected_amplitude, "n_rejected_flat": rejected_flat,
    }
