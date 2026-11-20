"""BNCI2015-004 loading and data-contract audit helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ..grouped import GroupedEEGDataset, file_sha256

BNCI2015_004_CHANNELS = [
    "AFz", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4", "T3", "C3",
    "Cz", "C4", "T4", "CP3", "CPz", "CP4", "P7", "P5", "P3", "P1", "Pz",
    "P2", "P4", "P6", "P8", "PO3", "PO4", "O1", "O2",
]
BNCI2015_004_EVENT_IDS = {"math": 1, "letter": 2, "rotation": 3, "count": 4, "baseline": 5}
BNCI2015_004_TASKS = {
    "math": "mental_subtraction",
    "letter": "word_association",
    "rotation": "spatial_navigation",
    "count": "mental_counting",
    "baseline": "baseline",
}


def audit_bnci2015_004(*, subjects: Iterable[int] | None = None,
                       data_path: str | Path | None = None,
                       download: bool = False) -> dict[str, object]:
    """Return MOABB metadata and optionally fetch data for a file audit.

    MOABB download location follows its MNE dataset configuration; callers
    should set that location before requesting download=True.
    """
    try:
        import moabb
        from moabb.datasets import BNCI2015_004
    except ImportError as exc:  # pragma: no cover
        raise ImportError("BNCI audit requires moabb 1.7.x") from exc
    dataset = BNCI2015_004()
    requested = list(subjects) if subjects is not None else list(dataset.subject_list)
    root = Path(data_path) if data_path is not None else None
    data = dataset.get_data(subjects=requested) if download else {}
    files = []
    if root is not None and root.exists():
        files = [{"path": str(p), "sha256": file_sha256(p), "bytes": p.stat().st_size}
                 for p in root.rglob("*") if p.is_file()]
    sessions = sorted({str(session) for subject in data.values() for session in subject})
    return {
        "dataset": "BNCI2015-004",
        "moabb_version": getattr(moabb, "__version__", "unknown"),
        "subjects": requested,
        "n_subjects": len(requested),
        "sessions": sessions,
        "events": dict(BNCI2015_004_EVENT_IDS),
        "tasks": dict(BNCI2015_004_TASKS),
        "channels": list(BNCI2015_004_CHANNELS),
        "n_channels_contract": len(BNCI2015_004_CHANNELS),
        "sampling_rate_hz": 256,
        "window_contract": None,
        "window_contract_status": "cue-relative window must be verified against the original trial annotations",
        "source_files": files,
    }


def make_grouped_dataset(X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
                         sessions: np.ndarray, interfaces: np.ndarray,
                         *, source: str, channel_names: list[str], class_names: list[str],
                         fs: float = 256.0) -> GroupedEEGDataset:
    """Wrap audited BNCI windows; callers must provide the verified class and channel maps."""
    channels = list(channel_names)
    if np.asarray(X).shape[1] != len(channels):
        raise ValueError(f"expected {len(channels)} channels, got {np.asarray(X).shape[1]}")
    return GroupedEEGDataset(
        X=np.asarray(X, dtype=np.float32), y=np.asarray(y, dtype=np.int64),
        subjects=np.asarray(subjects).astype(str), sessions=np.asarray(sessions).astype(str),
        interfaces=np.asarray(interfaces).astype(str), fs=float(fs), channel_names=channels,
        class_names=list(class_names), source=source,
    )
