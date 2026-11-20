"""Window-cache adapters for independent external EEG datasets.

These functions do not download data or infer task/cue mappings. Input caches
must already contain audited windows and integer labels in the caller-declared
class order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .grouped import GroupedEEGDataset, file_sha256


def _load_external_windows(path: str | Path, *, fs: float, class_names: Sequence[str],
                           expected_classes: int, unit: str = "uV") -> GroupedEEGDataset:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"X", "y", "subjects", "sessions", "interfaces", "channels", "recording_order"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"external window cache lacks required audited fields: {sorted(missing)}")
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        subjects = data["subjects"].astype(str)
        sessions = data["sessions"].astype(str)
        interfaces = data["interfaces"].astype(str)
        channels = [str(channel) for channel in data["channels"].tolist()]
        recording_order = data["recording_order"].copy()
    if len(class_names) != expected_classes or set(np.unique(y).tolist()) - set(range(expected_classes)):
        raise ValueError(f"expected labels 0..{expected_classes - 1} and {expected_classes} explicit class names")
    return GroupedEEGDataset(
        X=X, y=y, subjects=subjects, sessions=sessions, interfaces=interfaces, fs=float(fs),
        channel_names=channels, class_names=list(class_names), unit=unit, source=str(path),
        source_sha256=file_sha256(path), recording_order=recording_order,
    )


def load_lee2019_mi_windows(path: str | Path, *, fs: float = 120.0,
                            class_names: Sequence[str]) -> GroupedEEGDataset:
    """Load an audited Lee2019_MI window cache with explicit trial order.

    Cue timing, event-to-class mapping and preprocessing are deliberately the
    cache producer's responsibility and must be recorded before K9.
    """
    return _load_external_windows(path, fs=fs, class_names=class_names, expected_classes=2)


def load_bnci2015_004_windows(path: str | Path, *, fs: float = 120.0,
                              class_names: Sequence[str], unit: str = "uV") -> GroupedEEGDataset:
    """Load audited three-class BNCI2015-004 windows without assuming a channel map.

    The dataset's five MOABB events are not mapped to the thesis's three classes
    here. Callers must first verify and encode the chosen three classes, and the
    exact channel list must be stored in the cache.
    """
    return _load_external_windows(path, fs=fs, class_names=class_names,
                                  expected_classes=3, unit=unit)
