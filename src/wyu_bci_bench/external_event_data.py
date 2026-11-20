"""Label-aware event and fixed-window extraction from continuous EEG recordings."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def read_edf_with_events(path: str | Path):
    """Read an EDF while retaining its annotations for audited event mapping."""
    import mne

    return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")


def extract_windows_from_raw(raw, starts_seconds: Sequence[float], *, channel_names: Sequence[str],
                             duration_seconds: float, target_fs: float = 120.0,
                             car: bool = True) -> np.ndarray:
    """Extract ordered windows as microvolts, resampling once and optionally CARing each sample.

    Starts are relative to the beginning of the raw recording. Out-of-bounds
    windows and missing channels raise instead of being silently dropped.
    """
    starts = np.asarray(starts_seconds, dtype=np.float64)
    names = list(channel_names)
    if starts.ndim != 1 or not np.isfinite(starts).all() or np.any(starts < 0):
        raise ValueError("starts_seconds must be a finite non-negative vector")
    if not names or len(set(names)) != len(names) or duration_seconds <= 0 or target_fs <= 0:
        raise ValueError("channels must be unique and duration/target_fs must be positive")
    missing = [name for name in names if name not in raw.ch_names]
    if missing:
        raise ValueError(f"recording is missing channels: {missing}")
    samples = int(round(duration_seconds * target_fs))
    if samples < 2:
        raise ValueError("duration_seconds is too short at target_fs")
    recording = raw.copy().pick(names)
    if not np.isclose(recording.info["sfreq"], target_fs):
        recording.resample(target_fs, npad="auto", verbose="ERROR")
    signal = recording.get_data().astype(np.float64) * 1e6
    starts_samples = np.rint(starts * target_fs).astype(np.int64)
    if np.any(starts_samples + samples > signal.shape[1]):
        raise ValueError("a requested window extends beyond the recording")
    windows = np.stack([signal[:, start:start + samples] for start in starts_samples])
    if car:
        windows = windows - windows.mean(axis=1, keepdims=True)
    return windows.astype(np.float32)


def extract_annotated_windows(raw, event_to_label: dict[str, int], *, channel_names: Sequence[str],
                              offset_seconds: float, duration_seconds: float,
                              target_fs: float = 120.0, car: bool = True):
    """Extract ordered windows after matching annotation descriptions to labels.

    The offset is relative to the onset of each matched annotation. Unknown
    annotations are ignored; matched events must have unique, increasing onsets.
    Returns ``(windows, labels, event_onsets)`` in recording order.
    """
    if not event_to_label or offset_seconds < 0 or not np.isfinite(offset_seconds):
        raise ValueError("event_to_label must be nonempty and offset_seconds non-negative")
    descriptions = np.asarray([str(value) for value in raw.annotations.description], dtype=object)
    onsets = np.asarray(raw.annotations.onset, dtype=np.float64)
    event_map = {str(code): int(label) for code, label in event_to_label.items()}
    selected = np.flatnonzero(np.isin(descriptions, list(event_map)))
    if len(selected) == 0:
        raise ValueError("recording contains no annotations matching event_to_label")
    selected = selected[np.argsort(onsets[selected], kind="stable")]
    selected_onsets = onsets[selected]
    if not np.isfinite(selected_onsets).all() or np.any(np.diff(selected_onsets) <= 0):
        raise ValueError("matched event onsets must be finite and strictly increasing")
    labels = np.asarray([event_map[descriptions[i]] for i in selected], dtype=np.int64)
    windows = extract_windows_from_raw(
        raw, selected_onsets + float(offset_seconds), channel_names=channel_names,
        duration_seconds=duration_seconds, target_fs=target_fs, car=car,
    )
    return windows, labels, selected_onsets


def extract_sliding_windows(raw, *, channel_names: Sequence[str], duration_seconds: float,
                            step_seconds: float, target_fs: float = 120.0,
                            car: bool = True) -> np.ndarray:
    """Extract complete fixed-step windows from the start of a continuous recording."""
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    n_samples = int(round(duration_seconds * target_fs))
    step_samples = int(round(step_seconds * target_fs))
    n_recording = int(round(raw.n_times * target_fs / raw.info["sfreq"]))
    if n_samples < 2 or step_samples < 1:
        raise ValueError("window and step are too short at target_fs")
    count = max(0, 1 + (n_recording - n_samples) // step_samples)
    starts = np.arange(count, dtype=np.float64) * (step_samples / target_fs)
    return extract_windows_from_raw(raw, starts, channel_names=channel_names,
                                    duration_seconds=duration_seconds, target_fs=target_fs, car=car)
