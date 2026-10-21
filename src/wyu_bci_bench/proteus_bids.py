"""Build the PROTEUS window cache from the NEMAR nm000290 BIDS download.

Ported from ``proteus_update1/train_proteus_update1.py:extract_windows``. Per
EDF run: EEG picks (41 channels, reordered to the first run's order), notch at
50/60 Hz and harmonics, 0.1-75 Hz band-pass, resample to 120 Hz, volts -> uV,
then 480 samples starting 1 s after each cue. Needs ``mne`` and ``pandas``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np

from .grouped import PROTEUS_CLASSES, PROTEUS_FS, GroupedEEGDataset

CLASS_ID = {"mi": 0, "sub": 1, "word": 2}
INTERFACE = {"AcquisitionBH": "BrainHero", "OnlineRawBH": "BrainHero",
             "AcquisitionGraz": "Graz", "OnlineRawGraz": "Graz"}
NOTCH = [50, 60, 100, 120, 150, 180, 200, 240]
RUN_RE = re.compile(r"(sub-P\d+)_ses-(\d+)_task-(.+)_run-(\d+)_eeg\.edf$")


def iter_runs(root: str | Path, subjects: Iterable[str] | None = None):
    """按文件名合同枚举支持的 EEG EDF，并解析参与者、会话和接口。"""
    keep = None if subjects is None else set(subjects)
    for edf in sorted(Path(root).rglob("*_eeg.edf")):
        m = RUN_RE.search(edf.name)
        if not m or m.group(3) not in INTERFACE:
            continue
        if keep is not None and m.group(1) not in keep:
            continue
        yield edf, m.group(1), f"ses-{m.group(2)}", INTERFACE[m.group(3)]


def extract_windows(root: str | Path, subjects: Iterable[str] | None = None, offset_s: float = 1.0,
                    n_samples: int = 480, fs: float = PROTEUS_FS, verbose: bool = False) -> GroupedEEGDataset:
    """从 BIDS EDF/events.tsv 重建 PROTEUS 定长窗口缓存。

    每个有效事件按事件 onset 加 ``offset_s`` 截取 ``n_samples`` 点；超出记录边界的
    不完整窗口跳过。所有 run 的 EEG 通道必须能对齐到首个 run 的顺序。预处理步骤
    （notch、带通、重采样、伏转微伏）是缓存合同的一部分，不能只改窗口参数就声称
    与现有缓存等价。
    """
    import mne
    import pandas as pd

    xs, ys, subs, sess, itfs = [], [], [], [], []
    channels = None
    for edf, subject, session, interface in iter_runs(root, subjects):
        events = pd.read_csv(edf.with_name(edf.name.replace("_eeg.edf", "_events.tsv")), sep="\t")
        events = events[events["trial_type"].isin(CLASS_ID)]
        if events.empty:
            continue
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
        raw.pick("eeg", verbose="ERROR")
        if channels is None:
            channels = list(raw.ch_names)
            if len(channels) != 41:
                raise ValueError(f"expected 41 EEG channels, found {len(channels)}")
        elif raw.ch_names != channels:
            if set(raw.ch_names) != set(channels):
                raise ValueError(f"EEG channel mismatch in {edf}")
            raw.reorder_channels(channels)
        raw.notch_filter(NOTCH, picks="all", phase="zero", filter_length="auto", n_jobs=1, verbose="ERROR")
        raw.filter(0.1, 75.0, picks="all", skip_by_annotation=("edge", "bad_acq_skip"), n_jobs=1, verbose="ERROR")
        raw.resample(fs, n_jobs=1, verbose="ERROR")
        signal = raw.get_data().astype(np.float32) * 1e6
        for event in events.itertuples(index=False):
            start = int(round((float(event.onset) + offset_s) * fs))
            window = signal[:, start:start + n_samples]
            if window.shape != (len(channels), n_samples):
                continue
            xs.append(window)
            ys.append(CLASS_ID[event.trial_type])
            subs.append(subject)
            sess.append(session)
            itfs.append(interface)
        if verbose:
            print(f"{edf.name}: {len(xs)} windows total", flush=True)
    if not xs:
        raise RuntimeError(f"no windows extracted under {root}")
    return GroupedEEGDataset(X=np.stack(xs), y=np.asarray(ys, np.int64), subjects=np.asarray(subs),
                             sessions=np.asarray(sess), interfaces=np.asarray(itfs), fs=fs,
                             channel_names=channels, class_names=PROTEUS_CLASSES, source=str(root),
                             metadata={"offset_s": offset_s, "n_samples": n_samples})


def save_cache(dataset: GroupedEEGDataset, path: str | Path) -> None:
    """写出与现有 ``proteus_windows_v1.npz`` 一致的字段名，供统一加载器读取。"""
    np.savez_compressed(path, X=dataset.X, y=dataset.y, subjects=dataset.subjects, sessions=dataset.sessions,
                        interfaces=dataset.interfaces, channels=np.asarray(dataset.channel_names))
