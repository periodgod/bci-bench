"""本地连续 raw_stream 脑机接口记录的读取、质控、预处理和 epoch 切片。

This module is for recordings that store a continuous ``raw_stream`` plus JSON
trial timing records, such as the arm_bci_robot MI/ME sessions. Each file is
treated as a separate acquisition segment: filtering and epoching never cross
recording boundaries.
"""

# 与 data.py 中已保存成单试次数组的 NPZ 不同，本模块从连续采样流和试次时间记录
# 重建 epoch。每个文件独立滤波和切片，禁止跨采集文件边界滤波。

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .data import EEGDataset, scan_recordings


DEFAULT_EEG8_CLASS_NAMES = {
    1: "左手想象",
    2: "左手执行",
    3: "右手想象",
    4: "右手执行",
    5: "双手想象",
    6: "双手执行",
    7: "双脚想象",
    8: "双脚执行",
}

MI_ME_STATE_LABELS = ["想象", "执行"]
MI_ME_BODY_LABELS = ["左手", "右手", "双手", "双脚"]

MI_ME_LABEL_MODES = {
    "all",
    "mi_me8",
    "mi_me_state",
    "mi_me_body",
    "mi_me_body4",
    "mi_me_imagery_body",
    "mi_me_imagery_body4",
    "mi_me_execution_body",
    "mi_me_execution_body4",
}


@dataclass(frozen=True)
class ContinuousPreprocessConfig:
    """连续记录预处理合同，字段会写入 EEGDataset.metadata 供追溯。

    baseline_mode 决定基线区间，task_offset_s 决定事件后截取起点，bad_channel_mode
    决定通道质控策略。更改这些参数会改变训练输入，应作为实验配置记录。
    """
    eeg_channels: int = 32
    bandpass_low: float = 4.0
    bandpass_high: float = 40.0
    filter_order: int = 6
    line_freq: float = 50.0
    notch_q: float = 30.0
    apply_filter: bool = True
    task_offset_s: float = 0.5
    baseline_s: float = 0.5
    baseline_mode: str = "pre_cue"  # pre_cue, epoch_start, none
    min_epoch_s: float = 0.5
    reject_z: float = 8.0
    bad_channel_mode: str = "auto"  # off, auto, strict, manual
    bad_channels: tuple[str, ...] = ()
    require_task_times: bool = True
    align_timeline_to_raw_end: bool = True


@dataclass(frozen=True)
class RawSegment:
    """单个原始采集文件中的连续信号、采样率和试次时间记录。"""
    path: Path
    raw: np.ndarray
    fs: int
    channel_names: list[str]
    trial_records: list[dict[str, Any]]
    task_names: dict[int, str] = field(default_factory=dict)


def _decode_item(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _decode_scalar(value: object, default: object = "") -> object:
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return arr.item()
        if arr.size == 1:
            return arr.reshape(-1)[0].item()
    except Exception:
        pass
    return default


def _json_field(data: np.lib.npyio.NpzFile, key: str, default: Any) -> Any:
    if key not in data:
        return default
    raw = _decode_scalar(data[key], "")
    if raw in ("", None):
        return default
    try:
        return json.loads(str(raw))
    except Exception:
        return default


def _channel_names(data: np.lib.npyio.NpzFile, channels: int) -> list[str]:
    for key in ("raw_channel_names", "channel_names", "ch_names", "channels"):
        if key in data:
            names = [_decode_item(item) for item in np.asarray(data[key], dtype=object).reshape(-1)]
            if len(names) >= channels:
                return names[:channels]
    return [f"Ch{idx + 1}" for idx in range(channels)]


def _task_names(data: np.lib.npyio.NpzFile) -> dict[int, str]:
    tasks = _json_field(data, "task_definitions_json", [])
    names: dict[int, str] = {}
    for item in tasks:
        try:
            cid = int(item.get("class_id", 0) or 0)
        except Exception:
            continue
        if cid > 0:
            names[cid] = str(item.get("class_name") or DEFAULT_EEG8_CLASS_NAMES.get(cid, f"C{cid}"))
    return names


def load_raw_segment(path: str | Path, eeg_channels: int = 32) -> RawSegment:
    """从连续记录 NPZ 解码 raw_stream、采样率、通道名和试次 JSON 元数据。"""
    p = Path(path)
    with np.load(p, allow_pickle=True) as data:
        if "raw_stream" not in data:
            raise ValueError(f"{p} does not contain raw_stream")
        fs = int(np.asarray(data["fs"]).reshape(-1)[0]) if "fs" in data else 250
        raw_all = np.asarray(data["raw_stream"], dtype=np.float32)
        if raw_all.ndim != 2 or raw_all.shape[0] == 0:
            raise ValueError(f"{p} has invalid raw_stream shape {raw_all.shape}")
        n_eeg = min(int(eeg_channels), raw_all.shape[1])
        channel_names = _channel_names(data, raw_all.shape[1])[:n_eeg]
        records = _json_field(data, "trial_records_json", [])
        raw = raw_all[:, :n_eeg].astype(np.float32, copy=True)
        task_names = {**DEFAULT_EEG8_CLASS_NAMES, **_task_names(data)}
    return RawSegment(path=p, raw=raw, fs=fs, channel_names=channel_names, trial_records=records, task_names=task_names)


def scan_raw_segments(paths: str | Path | Iterable[str | Path], count_latest: int | None = None) -> list[Path]:
    """寻找有效连续流文件；可选只保留按修改时间排序的最近 N 个文件。"""
    files = scan_recordings(paths)
    valid = []
    for path in files:
        try:
            with np.load(path, allow_pickle=True) as data:
                if "raw_stream" in data and "trial_records_json" in data:
                    valid.append(path)
        except Exception:
            continue
    valid = sorted(valid, key=lambda path: path.stat().st_mtime)
    if count_latest is not None:
        valid = valid[-int(count_latest) :]
    return valid


def _robust_std(x: np.ndarray) -> np.ndarray:
    """用 1.4826 × MAD 估计抗离群值标准差。"""
    median = np.median(x, axis=0, keepdims=True)
    mad = np.median(np.abs(x - median), axis=0)
    return (1.4826 * mad).astype(np.float64)


def compute_channel_qc(segments: Sequence[RawSegment]) -> dict[str, Any]:
    """跨文件计算每通道有限值比例、稳健幅度及异常标记。

    auto/strict 阈值是项目当前质控规则，不是临床或普适坏导判据。函数返回逐文件
    指标与跨文件摘要供复核；具体剔除由 select_qc_channels 决定。
    """
    if not segments:
        raise ValueError("No segments for channel QC")
    channel_names = list(segments[0].channel_names)
    rows: list[dict[str, Any]] = []
    for segment in segments:
        if list(segment.channel_names) != channel_names:
            raise ValueError(f"Channel names mismatch in {segment.path}")
        raw = np.asarray(segment.raw, dtype=np.float32)
        finite = np.isfinite(raw)
        safe = np.nan_to_num(raw, copy=True)
        std = np.std(safe, axis=0).astype(np.float64)
        robust = _robust_std(safe)
        robust_ptp = (np.percentile(safe, 99.9, axis=0) - np.percentile(safe, 0.1, axis=0)).astype(np.float64)
        finite_ratio = finite.mean(axis=0).astype(np.float64)
        med_std = max(float(np.median(std)), 1e-6)
        med_robust = max(float(np.median(robust)), 1e-6)
        med_ptp = max(float(np.median(robust_ptp)), 1e-6)
        for idx, name in enumerate(channel_names):
            std_ratio = float(std[idx] / med_std)
            robust_ratio = float(robust[idx] / med_robust)
            ptp_ratio = float(robust_ptp[idx] / med_ptp)
            flags = []
            if finite_ratio[idx] < 0.999:
                flags.append("nonfinite")
            if std[idx] < 1e-6 or robust[idx] < 1e-6:
                flags.append("flat")
            if std_ratio > 5.0:
                flags.append("std_ratio_gt_5")
            if robust_ratio > 5.0:
                flags.append("robust_ratio_gt_5")
            if ptp_ratio > 5.0:
                flags.append("ptp_ratio_gt_5")
            rows.append(
                {
                    "file": segment.path.name,
                    "channel_index": idx,
                    "channel_name": name,
                    "raw_std": float(std[idx]),
                    "robust_std": float(robust[idx]),
                    "robust_ptp_0p1_99p9": float(robust_ptp[idx]),
                    "finite_ratio": float(finite_ratio[idx]),
                    "std_ratio": std_ratio,
                    "robust_ratio": robust_ratio,
                    "ptp_ratio": ptp_ratio,
                    "file_flag": bool(flags),
                    "flags": flags,
                }
            )
    summary = []
    for idx, name in enumerate(channel_names):
        subset = [row for row in rows if int(row["channel_index"]) == idx]
        flag_count = int(sum(bool(row["file_flag"]) for row in subset))
        max_std_ratio = float(max(row["std_ratio"] for row in subset))
        max_robust_ratio = float(max(row["robust_ratio"] for row in subset))
        max_ptp_ratio = float(max(row["ptp_ratio"] for row in subset))
        max_raw_std = float(max(row["raw_std"] for row in subset))
        min_finite = float(min(row["finite_ratio"] for row in subset))
        auto_bad = (
            flag_count >= 2
            or max_std_ratio >= 10.0
            or max_robust_ratio >= 10.0
            or max_ptp_ratio >= 10.0
            or min_finite < 0.999
        )
        strict_bad = auto_bad or max_raw_std > 700.0 or flag_count >= 1
        reasons = []
        if flag_count >= 2:
            reasons.append(f"flagged_in_{flag_count}_files")
        if max_std_ratio >= 10.0:
            reasons.append(f"max_std_ratio={max_std_ratio:.2f}")
        if max_robust_ratio >= 10.0:
            reasons.append(f"max_robust_ratio={max_robust_ratio:.2f}")
        if max_ptp_ratio >= 10.0:
            reasons.append(f"max_ptp_ratio={max_ptp_ratio:.2f}")
        if min_finite < 0.999:
            reasons.append(f"min_finite={min_finite:.3f}")
        if strict_bad and not auto_bad:
            reasons.append(f"strict_edge_noise=max_std_{max_raw_std:.1f}_or_single_file_flag")
        summary.append(
            {
                "channel_index": idx,
                "channel_name": name,
                "flag_count": flag_count,
                "max_raw_std": max_raw_std,
                "max_std_ratio": max_std_ratio,
                "max_robust_ratio": max_robust_ratio,
                "max_ptp_ratio": max_ptp_ratio,
                "min_finite_ratio": min_finite,
                "auto_bad": auto_bad,
                "strict_bad": strict_bad,
                "reasons": reasons,
            }
        )
    return {
        "channel_names": channel_names,
        "by_file": rows,
        "summary": summary,
        "rules": {
            "file_flag": "finite<0.999 or flat or std/median>5 or robust_std/median>5 or robust_ptp/median>5",
            "auto_bad": "bad if flagged in >=2 files, or any ratio >=10, or finite<0.999",
            "strict_bad": "auto_bad plus raw_std>700 or any single file flag",
        },
    }


def select_qc_channels(qc: dict[str, Any], mode: str = "auto", manual_bad: Sequence[str] = ()) -> tuple[np.ndarray, list[str], list[str]]:
    """根据关闭、人工、auto 或 strict 策略返回保留索引及通道名称。"""
    names = [str(item) for item in qc["channel_names"]]
    mode_key = str(mode).lower()
    if mode_key in {"off", "none"}:
        bad: list[int] = []
    elif mode_key == "manual":
        wanted = {str(name).strip().upper() for name in manual_bad}
        lookup = {str(name).strip().upper(): idx for idx, name in enumerate(names)}
        missing = sorted(wanted - set(lookup))
        if missing:
            raise ValueError(f"Manual bad channels not found: {missing}")
        bad = sorted(lookup[name] for name in wanted)
    else:
        key = "strict_bad" if mode_key == "strict" else "auto_bad"
        bad = [int(row["channel_index"]) for row in qc["summary"] if bool(row[key])]
    keep = [idx for idx in range(len(names)) if idx not in set(bad)]
    if len(keep) < 3:
        raise ValueError(f"Channel screening left only {len(keep)} channels")
    return np.asarray(keep, dtype=np.int64), [names[idx] for idx in keep], [names[idx] for idx in bad]


def _require_scipy_signal():
    try:
        from scipy import signal  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("scipy is required for continuous Butterworth preprocessing") from exc
    return signal


def _reflect_pad(x: np.ndarray, pad_len: int) -> tuple[np.ndarray, int]:
    """反射扩展时间序列，减少正反向滤波的边界瞬态。"""
    pad_len = int(max(0, min(pad_len, x.shape[0] - 1)))
    if pad_len <= 0:
        return x, 0
    left = 2.0 * x[:1] - x[1 : pad_len + 1][::-1]
    right = 2.0 * x[-1:] - x[-pad_len - 1 : -1][::-1]
    return np.concatenate([left, x, right], axis=0), pad_len


def _sos_zero_phase_filter(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """二阶节形式先正向再反向滤波以抵消相位延迟，并裁去临时 padding。"""
    signal = _require_scipy_signal()
    pad_len = 3 * (2 * len(sos) + 1)
    padded, actual_pad = _reflect_pad(x, pad_len)
    y = signal.sosfilt(sos, padded, axis=0)
    y = signal.sosfilt(sos, y[::-1], axis=0)[::-1]
    if actual_pad:
        y = y[actual_pad:-actual_pad]
    return y


def _linear_detrend_numpy(x: np.ndarray) -> np.ndarray:
    """逐通道拟合并去除线性趋势；单点信号退化为去均值。"""
    if x.shape[0] < 2:
        return x - np.mean(x, axis=0, keepdims=True)
    t = np.linspace(-1.0, 1.0, x.shape[0], dtype=np.float64)
    t -= np.mean(t)
    denom = max(float(np.sum(t * t)), 1e-12)
    y = np.array(x, dtype=np.float64, copy=True)
    y -= np.mean(y, axis=0, keepdims=True)
    for ch in range(y.shape[1]):
        slope = float(np.sum(t * y[:, ch]) / denom)
        y[:, ch] -= t * slope
    return y


def preprocess_raw_segment(raw: np.ndarray, fs: int, config: ContinuousPreprocessConfig) -> np.ndarray:
    """对单文件连续信号去趋势，可选陷波和 Butterworth 零相位带通。"""
    x = _linear_detrend_numpy(np.asarray(raw, dtype=np.float64))
    if not config.apply_filter:
        return x.astype(np.float32)
    signal = _require_scipy_signal()
    if config.bandpass_low < config.line_freq < config.bandpass_high:
        b, a = signal.iirnotch(w0=float(config.line_freq), Q=float(config.notch_q), fs=float(fs))
        notch_sos = np.asarray([[b[0], b[1], b[2], a[0], a[1], a[2]]], dtype=np.float64)
        x = _sos_zero_phase_filter(x, notch_sos)
    nyq = float(fs) / 2.0
    high = min(float(config.bandpass_high), nyq - 1e-3)
    low = max(float(config.bandpass_low), 1e-3)
    if low >= high:
        raise ValueError(f"Invalid bandpass {low}-{high} Hz for fs={fs}")
    sos = signal.butter(
        int(config.filter_order),
        [low / nyq, high / nyq],
        btype="bandpass",
        output="sos",
    )
    return _sos_zero_phase_filter(x, sos).astype(np.float32)


def _valid_trial_record(rec: dict[str, Any]) -> bool:
    """排除无类别、标记坏试次或练习试次的时间记录。"""
    cid = int(rec.get("class_id", 0) or 0)
    return bool(cid > 0 and not rec.get("bad_trial") and not rec.get("is_practice"))


def _extract_segment_epochs(
    segment: RawSegment,
    filtered_raw: np.ndarray,
    keep_channels: np.ndarray,
    config: ContinuousPreprocessConfig,
) -> tuple[list[np.ndarray], list[int], list[str], list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    """按单段记录时间切片、做基线处理并汇报剔除原因。

    ``filtered_raw`` 只包含当前文件的处理后数据；调用方逐文件调用，因此 epoch
    不会跨文件边界。返回值带标签、源记录和对齐诊断，可追溯每个 epoch 的来源。
    """
    fs = int(segment.fs)
    epochs: list[np.ndarray] = []
    class_ids: list[int] = []
    labels: list[str] = []
    records: list[dict[str, Any]] = []
    reject = {
        "bad_or_practice": 0,
        "missing_time": 0,
        "too_short": 0,
        "crosses_source_boundary": 0,
        "nonfinite": 0,
        "baseline_fallback_epoch_start": 0,
    }
    valid_records = [rec for rec in segment.trial_records if _valid_trial_record(rec)]
    if config.align_timeline_to_raw_end and valid_records:
        max_end_ms = max(float(rec.get("trial_end_ms", rec.get("task_end_ms", 0)) or 0) for rec in valid_records)
        max_end_samples = int(round(max_end_ms / 1000.0 * fs))
        timeline_offset_samples = min(0, int(filtered_raw.shape[0]) - max_end_samples)
    else:
        max_end_ms = 0.0
        max_end_samples = 0
        timeline_offset_samples = 0
    diagnostics = {
        "source_file": segment.path.name,
        "raw_samples": int(filtered_raw.shape[0]),
        "max_event_end_ms": float(max_end_ms),
        "max_event_end_samples": int(max_end_samples),
        "timeline_offset_samples": int(timeline_offset_samples),
        "timeline_offset_s": float(timeline_offset_samples / fs),
        "align_timeline_to_raw_end": bool(config.align_timeline_to_raw_end),
    }
    for idx, rec in enumerate(segment.trial_records):
        if not _valid_trial_record(rec):
            reject["bad_or_practice"] += 1
            continue
        cid = int(rec.get("class_id", 0) or 0)
        if "task_start_ms" not in rec or "task_end_ms" not in rec:
            reject["missing_time"] += 1
            continue
        start_ms = float(rec["task_start_ms"]) + float(config.task_offset_s) * 1000.0
        stop_ms = float(rec["task_end_ms"])
        start = int(round(start_ms / 1000.0 * fs)) + timeline_offset_samples
        stop = int(round(stop_ms / 1000.0 * fs)) + timeline_offset_samples
        if start < 0 or stop > filtered_raw.shape[0]:
            reject["crosses_source_boundary"] += 1
            continue
        if stop - start < max(8, int(round(float(config.min_epoch_s) * fs))):
            reject["too_short"] += 1
            continue
        epoch = filtered_raw[start:stop, :][:, keep_channels].T.astype(np.float32)
        if not np.isfinite(epoch).all():
            reject["nonfinite"] += 1
            continue
        baseline_mode = str(config.baseline_mode).lower()
        if baseline_mode in {"off", "none", "false", "0"}:
            baseline = None
        elif baseline_mode == "pre_cue" and "baseline_start_ms" in rec and "cue_onset_ms" in rec:
            b_start = int(round(float(rec["baseline_start_ms"]) / 1000.0 * fs)) + timeline_offset_samples
            b_stop = int(round(float(rec["cue_onset_ms"]) / 1000.0 * fs)) + timeline_offset_samples
            if 0 <= b_start < b_stop <= filtered_raw.shape[0]:
                baseline = filtered_raw[b_start:b_stop, :][:, keep_channels].T.astype(np.float32)
            else:
                baseline = None
        else:
            baseline = None
        if baseline is None and baseline_mode not in {"off", "none", "false", "0"}:
            baseline_n = min(epoch.shape[1], max(1, int(round(float(config.baseline_s) * fs))))
            baseline = epoch[:, :baseline_n]
            reject["baseline_fallback_epoch_start"] += 1
        if baseline is not None:
            epoch = epoch - baseline.mean(axis=1, keepdims=True)
        name = segment.task_names.get(cid, DEFAULT_EEG8_CLASS_NAMES.get(cid, f"C{cid}"))
        epochs.append(epoch.astype(np.float32))
        class_ids.append(cid)
        labels.append(f"C{cid} {name}")
        enriched = dict(rec)
        enriched.update({"source_file": segment.path.name, "source_trial_index": idx})
        records.append(enriched)
    return epochs, class_ids, labels, records, reject, diagnostics


def _normalize_label_mode(label_mode: str) -> str:
    mode = str(label_mode or "all").strip().lower().replace("-", "_")
    aliases = {
        "8": "mi_me8",
        "8class": "mi_me8",
        "mi_me_8": "mi_me8",
        "state": "mi_me_state",
        "body": "mi_me_body",
        "body4": "mi_me_body4",
        "imagery_body": "mi_me_imagery_body",
        "imagery_body4": "mi_me_imagery_body4",
        "execution_body": "mi_me_execution_body",
        "execution_body4": "mi_me_execution_body4",
    }
    mode = aliases.get(mode, mode)
    if mode not in MI_ME_LABEL_MODES:
        raise ValueError(f"Unsupported continuous label_mode: {label_mode}")
    return mode


def _mi_me_state_index(class_id: int) -> int:
    cid = int(class_id)
    if cid in {1, 3, 5, 7}:
        return 0
    if cid in {2, 4, 6, 8}:
        return 1
    raise ValueError(f"Class id {cid} is not a MI/ME class")


def _mi_me_body_index(class_id: int) -> int:
    cid = int(class_id)
    if 1 <= cid <= 8:
        return (cid - 1) // 2
    raise ValueError(f"Class id {cid} is not a MI/ME class")


def _encode_all_classes(class_ids: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    class_order = sorted(set(int(cid) for cid in class_ids.tolist()))
    label_lookup = {cid: idx for idx, cid in enumerate(class_order)}
    y = np.asarray([label_lookup[int(cid)] for cid in class_ids], dtype=np.int64)
    label_names = []
    for cid in class_order:
        matching = [str(label) for label, raw_id in zip(labels, class_ids) if int(raw_id) == cid]
        label_names.append(matching[0] if matching else f"C{cid}")
    return labels, y, label_names


def _apply_mi_me_label_mode(
    eeg: np.ndarray,
    class_ids: np.ndarray,
    labels: np.ndarray,
    records: list[dict[str, Any]],
    label_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[dict[str, Any]], dict[str, Any]]:
    """将想象/执行 × 身体部位标签折叠成用户指定的监督任务。"""
    mode = _normalize_label_mode(label_mode)
    raw_class_ids = np.asarray(class_ids, dtype=np.int64)
    raw_labels = np.asarray(labels, dtype=object)
    keep = np.ones(len(raw_class_ids), dtype=bool)

    if mode in {"all", "mi_me8"}:
        encoded_labels, y, label_names = _encode_all_classes(raw_class_ids, raw_labels)
        return (
            eeg,
            raw_class_ids,
            encoded_labels,
            y,
            label_names,
            list(records),
            {"label_mode": mode, "kept_trials": int(len(raw_class_ids)), "dropped_trials": 0},
        )

    states = np.asarray([_mi_me_state_index(int(cid)) for cid in raw_class_ids], dtype=np.int64)
    bodies = np.asarray([_mi_me_body_index(int(cid)) for cid in raw_class_ids], dtype=np.int64)

    if mode == "mi_me_state":
        y = states
        encoded_labels = np.asarray([MI_ME_STATE_LABELS[int(idx)] for idx in y], dtype=object)
        label_names = list(MI_ME_STATE_LABELS)
    else:
        if mode in {"mi_me_imagery_body", "mi_me_imagery_body4"}:
            keep = states == 0
        elif mode in {"mi_me_execution_body", "mi_me_execution_body4"}:
            keep = states == 1
        elif mode not in {"mi_me_body", "mi_me_body4"}:
            raise ValueError(f"Unsupported continuous label_mode: {label_mode}")
        y = bodies[keep]
        encoded_labels = np.asarray([MI_ME_BODY_LABELS[int(idx)] for idx in y], dtype=object)
        label_names = list(MI_ME_BODY_LABELS)

    if len(set(int(v) for v in y.tolist())) < 2:
        raise ValueError(f"Need at least two labels after label_mode={mode}; got {Counter(encoded_labels)}")

    kept_records = [record for record, item_keep in zip(records, keep) if bool(item_keep)]
    return (
        eeg[keep],
        raw_class_ids[keep],
        encoded_labels,
        y.astype(np.int64),
        label_names,
        kept_records,
        {
            "label_mode": mode,
            "kept_trials": int(np.sum(keep)),
            "dropped_trials": int(np.sum(~keep)),
            "raw_class_counts_before_label_mode": dict(Counter(int(v) for v in raw_class_ids.tolist())),
            "label_counts_after_label_mode": dict(Counter(str(v) for v in encoded_labels.tolist())),
        },
    )


def load_continuous_dataset(
    paths: str | Path | Iterable[str | Path],
    config: ContinuousPreprocessConfig | None = None,
    count_latest: int | None = None,
    label_mode: str = "all",
) -> EEGDataset:
    """端到端读取连续流：文件校验、通道 QC、逐文件预处理/epoch 和标签编码。

    返回数据统一裁成最短 epoch 长度；metadata 保存配置、质控明细、剔除数量和
    时间对齐信息。不同采样率文件不会自动重采样，遇到不一致会报错。
    """
    cfg = config or ContinuousPreprocessConfig()
    files = scan_raw_segments(paths, count_latest=count_latest)
    if not files:
        raise ValueError("No raw_stream NPZ recordings found")
    segments = [load_raw_segment(path, eeg_channels=cfg.eeg_channels) for path in files]
    fs = int(segments[0].fs)
    names = list(segments[0].channel_names)
    for segment in segments:
        if int(segment.fs) != fs:
            raise ValueError(f"Sampling rate mismatch: {segment.path} has {segment.fs}, expected {fs}")
        if list(segment.channel_names) != names:
            raise ValueError(f"Channel layout mismatch in {segment.path}")
    qc = compute_channel_qc(segments)
    mode = "manual" if cfg.bad_channels and cfg.bad_channel_mode == "manual" else cfg.bad_channel_mode
    keep_channels, kept_names, bad_names = select_qc_channels(qc, mode=mode, manual_bad=cfg.bad_channels)

    epochs: list[np.ndarray] = []
    class_ids: list[int] = []
    label_values: list[str] = []
    records: list[dict[str, Any]] = []
    reject_total = Counter()
    segment_timing = []
    for segment in segments:
        filtered = preprocess_raw_segment(segment.raw, segment.fs, cfg)
        seg_epochs, seg_ids, seg_labels, seg_records, seg_reject, seg_timing = _extract_segment_epochs(
            segment,
            filtered,
            keep_channels,
            cfg,
        )
        epochs.extend(seg_epochs)
        class_ids.extend(seg_ids)
        label_values.extend(seg_labels)
        records.extend(seg_records)
        reject_total.update(seg_reject)
        segment_timing.append(seg_timing)
    if not epochs:
        raise ValueError(f"No valid epochs extracted; reject={dict(reject_total)}")
    min_samples = min(epoch.shape[1] for epoch in epochs)
    x = np.stack([epoch[:, :min_samples] for epoch in epochs], axis=0).astype(np.float32)
    class_ids_arr = np.asarray(class_ids, dtype=np.int64)

    peak = np.max(np.abs(x), axis=(1, 2))
    med = float(np.median(peak))
    mad = float(np.median(np.abs(peak - med)) * 1.4826)
    threshold = med + float(cfg.reject_z) * max(mad, 1e-6)
    keep_epoch = peak <= threshold
    if np.sum(keep_epoch) < len(set(class_ids)):
        keep_epoch = np.ones_like(keep_epoch, dtype=bool)
        threshold = float("inf")
    x = x[keep_epoch]
    class_ids_arr = class_ids_arr[keep_epoch]
    labels_arr = np.asarray(label_values, dtype=object)[keep_epoch]
    records = [record for record, keep in zip(records, keep_epoch) if bool(keep)]

    x, class_ids_arr, labels_arr, y, label_names, records, label_mode_info = _apply_mi_me_label_mode(
        x,
        class_ids_arr,
        labels_arr,
        records,
        label_mode=label_mode,
    )
    source_counter = Counter(str(record["source_file"]) for record in records)
    source_counts = [int(source_counter.get(segment.path.name, 0)) for segment in segments]

    metadata = {
        "label_mode": label_mode_info,
        "continuous_preprocessing": {
            "per_recording_filter": bool(cfg.apply_filter),
            "bandpass_hz": [float(cfg.bandpass_low), float(cfg.bandpass_high)],
            "filter_order": int(cfg.filter_order),
            "notch_hz": float(cfg.line_freq),
            "task_offset_s": float(cfg.task_offset_s),
            "baseline_s": float(cfg.baseline_s),
            "baseline_mode": str(cfg.baseline_mode),
            "source_boundary_guard": True,
            "align_timeline_to_raw_end": bool(cfg.align_timeline_to_raw_end),
            "bad_channel_mode": str(cfg.bad_channel_mode),
            "bad_channels_removed": list(bad_names),
            "channels_kept": list(kept_names),
        },
        "channel_qc": qc,
        "epoch_info": {
            "initial_epochs": int(len(peak)),
            "kept_epochs": int(np.sum(keep_epoch)),
            "rejected_by_amplitude": int(np.sum(~keep_epoch)),
            "amplitude_threshold": None if not np.isfinite(threshold) else float(threshold),
            "reject_reasons": dict(reject_total),
            "epoch_samples": int(min_samples),
            "epoch_seconds": float(min_samples / fs),
        },
        "source_timing": segment_timing,
        "trial_records": records,
    }
    return EEGDataset(
        eeg=x,
        labels=labels_arr,
        y=y,
        fs=fs,
        channel_names=list(kept_names),
        label_names=label_names,
        paths=[segment.path for segment in segments],
        source_trial_counts=source_counts,
        metadata=metadata,
    )
