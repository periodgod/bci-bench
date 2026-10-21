"""Causal filtering, streaming windows, runtime measurements, and time drift."""

from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np


def design_causal_bandpass(low_hz: float, high_hz: float, sampling_rate: float,
                           order: int = 4) -> np.ndarray:
    """Design a causal Butterworth band-pass as second-order sections."""
    if not 0 < low_hz < high_hz < sampling_rate / 2:
        raise ValueError("band edges must satisfy 0 < low < high < Nyquist")
    if order < 1:
        raise ValueError("order must be positive")
    try:
        from scipy.signal import butter
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("causal filtering requires scipy") from exc
    return butter(order, (low_hz, high_hz), btype="bandpass", fs=sampling_rate, output="sos")


class CausalSOSFilter:
    """Stateful SOS filter whose result does not depend on chunk boundaries."""

    def __init__(self, sos: np.ndarray, channels: int):
        sections = np.asarray(sos, dtype=np.float64)
        if sections.ndim != 2 or sections.shape[1] != 6 or not np.isfinite(sections).all():
            raise ValueError("sos must be a finite array with shape (sections, 6)")
        if channels < 1:
            raise ValueError("channels must be positive")
        if np.any(sections[:, 3] == 0):
            raise ValueError("each SOS denominator leading coefficient must be nonzero")
        self.sos = sections
        self.channels = int(channels)
        self.reset()

    def reset(self) -> None:
        """Clear filter history at a recording or session boundary."""
        self._state = np.zeros((len(self.sos), self.channels, 2), dtype=np.float64)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Filter one ``(channels, samples)`` chunk, preserving causal state."""
        signal = np.asarray(chunk, dtype=np.float64)
        if signal.ndim != 2 or signal.shape[0] != self.channels:
            raise ValueError("chunk must have shape (configured channels, samples)")
        if signal.shape[1] < 1 or not np.isfinite(signal).all():
            raise ValueError("chunk must contain finite samples")
        try:
            from scipy.signal import sosfilt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("causal filtering requires scipy") from exc
        filtered, self._state = sosfilt(self.sos, signal, axis=-1, zi=self._state)
        return filtered


class StreamWindowBuffer:
    """Emit overlapping windows from arbitrarily sized causal input chunks."""

    def __init__(self, channels: int, window_samples: int, step_samples: int):
        if channels < 1 or window_samples < 1 or step_samples < 1:
            raise ValueError("channels, window_samples, and step_samples must be positive")
        if step_samples > window_samples:
            raise ValueError("step_samples cannot exceed window_samples")
        self.channels = int(channels)
        self.window_samples = int(window_samples)
        self.step_samples = int(step_samples)
        self.reset()

    def reset(self) -> None:
        """Discard buffered samples and restart window offsets at zero."""
        self._buffer = np.empty((self.channels, 0), dtype=np.float64)
        self._buffer_start = 0
        self._total_samples = 0
        self._next_start = 0

    def push(self, chunk: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Append a chunk and return available ``(start_sample, window)`` pairs."""
        signal = np.asarray(chunk, dtype=np.float64)
        if signal.ndim != 2 or signal.shape[0] != self.channels or signal.shape[1] < 1:
            raise ValueError("chunk must have shape (configured channels, positive samples)")
        if not np.isfinite(signal).all():
            raise ValueError("chunk must contain finite samples")
        self._buffer = np.concatenate((self._buffer, signal), axis=1)
        self._total_samples += signal.shape[1]
        windows = []
        while self._next_start + self.window_samples <= self._total_samples:
            offset = self._next_start - self._buffer_start
            windows.append((self._next_start,
                            self._buffer[:, offset:offset + self.window_samples].copy()))
            self._next_start += self.step_samples
        discard = self._next_start - self._buffer_start
        if discard > 0:
            self._buffer = self._buffer[:, discard:]
            self._buffer_start = self._next_start
        return windows


def benchmark_streaming(predict_window: Callable[[np.ndarray], object],
                        windows: Iterable[np.ndarray], *, warmup: int = 1,
                        deadline_ms: float | None = None) -> dict[str, float | int]:
    """Measure inference latency and throughput on the current host.

    Warm-up calls are excluded. Report measurements together with device and
    runtime context; they are not portable model-performance claims.
    """
    if warmup < 0 or (deadline_ms is not None and deadline_ms <= 0):
        raise ValueError("warmup must be non-negative and deadline_ms positive")
    iterator = iter(windows)
    for _ in range(warmup):
        try:
            predict_window(np.asarray(next(iterator)))
        except StopIteration:
            return {"windows": 0, "p50_ms": 0.0, "p95_ms": 0.0,
                    "throughput_windows_s": 0.0, "deadline_misses": 0}
    durations = []
    for window in iterator:
        started = time.perf_counter_ns()
        predict_window(np.asarray(window))
        durations.append((time.perf_counter_ns() - started) / 1e6)
    if not durations:
        return {"windows": 0, "p50_ms": 0.0, "p95_ms": 0.0,
                "throughput_windows_s": 0.0, "deadline_misses": 0}
    values = np.asarray(durations)
    elapsed_s = values.sum() / 1000
    return {
        "windows": len(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "throughput_windows_s": float(len(values) / elapsed_s) if elapsed_s else float("inf"),
        "deadline_misses": int(np.sum(values > deadline_ms)) if deadline_ms is not None else 0,
    }


def temporal_drift_summary(y_true: np.ndarray, y_pred: np.ndarray,
                           time_order: np.ndarray, bins: int = 5) -> list[dict[str, float | int]]:
    """Summarize accuracy by chronological blocks using delayed labels for audit."""
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    timestamps = np.asarray(time_order)
    if truth.ndim != 1 or prediction.shape != truth.shape or timestamps.shape != truth.shape:
        raise ValueError("y_true, y_pred, and time_order must be aligned 1-D arrays")
    if not len(truth) or bins < 1:
        raise ValueError("at least one observation and one bin are required")
    if not np.isfinite(timestamps).all():
        raise ValueError("time_order must be finite")
    ordered = np.argsort(timestamps, kind="stable")
    chunks = [chunk for chunk in np.array_split(ordered, min(bins, len(ordered))) if len(chunk)]
    result = []
    first_ba = None
    for index, selected in enumerate(chunks):
        y = truth[selected]
        pred = prediction[selected]
        recalls = [float(np.mean(pred[y == label] == label)) for label in np.unique(y)]
        balanced = float(np.mean(recalls))
        if first_ba is None:
            first_ba = balanced
        result.append({
            "bin": index,
            "count": int(len(selected)),
            "time_start": float(np.min(timestamps[selected])),
            "time_end": float(np.max(timestamps[selected])),
            "accuracy": float(np.mean(pred == y)),
            "balanced_accuracy": balanced,
            "balanced_accuracy_delta_from_first": balanced - first_ba,
        })
    return result
