"""Evaluation utilities."""

from __future__ import annotations

import time
from collections import Counter
from typing import Sequence

import numpy as np

from .deep import fit_predict_eegnet
from .models import FilterBankLDAModel
from .signal import BANDS, bandpower_from_psd, compute_psd


def stratified_folds(y: np.ndarray, folds: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(y, dtype=np.int64)
    counts = Counter(int(item) for item in labels)
    if len(counts) < 2:
        raise ValueError("At least two classes are required for cross-validation")
    effective = min(int(folds), min(counts.values()))
    if effective < 2:
        raise ValueError(f"Every class needs at least two trials; counts={dict(counts)}")
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(effective)]
    for label in sorted(counts):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        for pos, value in enumerate(idx):
            buckets[pos % effective].append(int(value))
    splits = []
    all_idx = np.arange(len(labels), dtype=np.int64)
    for bucket in buckets:
        test = np.asarray(sorted(bucket), dtype=np.int64)
        mask = np.ones(len(labels), dtype=bool)
        mask[test] = False
        train = all_idx[mask]
        splits.append((train, test))
    return splits


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[int]) -> list[list[int]]:
    lookup = {int(label): idx for idx, label in enumerate(classes)}
    mat = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        mat[lookup[int(truth)], lookup[int(pred)]] += 1
    return mat.tolist()


def metric_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str] | None = None,
    elapsed: float = 0.0,
) -> dict[str, object]:
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    classes = sorted(set(int(v) for v in truth) | set(int(v) for v in pred))
    per_class = {}
    recalls = []
    for label in classes:
        mask = truth == label
        recall = float(np.mean(pred[mask] == label)) if np.any(mask) else 0.0
        key = label_names[label] if label_names and label < len(label_names) else str(label)
        per_class[key] = recall
        recalls.append(recall)
    return {
        "accuracy": float(np.mean(truth == pred)),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "per_class_recall": per_class,
        "confusion_matrix": confusion_matrix(truth, pred, classes),
        "label_order": [label_names[label] if label_names and label < len(label_names) else str(label) for label in classes],
        "elapsed_seconds": float(elapsed),
    }


def cross_validate_filterbank_lda(
    eeg: np.ndarray,
    y: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    label_names: Sequence[str] | None = None,
    folds: int = 5,
    seed: int = 42,
    top_k: int | None = 48,
) -> dict[str, object]:
    labels = np.asarray(y, dtype=np.int64)
    predictions = np.full(len(labels), -1, dtype=np.int64)
    fold_reports = []
    started = time.perf_counter()
    splits = stratified_folds(labels, folds=folds, seed=seed)
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        model = FilterBankLDAModel(top_k=top_k)
        model.fit(eeg[train_idx], labels[train_idx], fs, channel_names)
        pred = model.predict(eeg[test_idx])
        predictions[test_idx] = pred
        fold_reports.append(
            {
                "fold": fold_idx,
                "train_trials": int(len(train_idx)),
                "test_trials": int(len(test_idx)),
                "metrics": metric_summary(labels[test_idx], pred, label_names=label_names),
                "selected_features": model.feature_names_[: min(12, len(model.feature_names_ or []))],
            }
        )
    elapsed = time.perf_counter() - started
    summary = metric_summary(labels, predictions, label_names=label_names, elapsed=elapsed)
    summary.update({"folds": len(splits), "fold_reports": fold_reports, "predictions": predictions.tolist()})
    return summary


def cross_validate_eegnet(
    eeg: np.ndarray,
    y: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    label_names: Sequence[str] | None = None,
    folds: int = 5,
    seed: int = 42,
    epochs: int = 30,
) -> dict[str, object]:
    del channel_names
    labels = np.asarray(y, dtype=np.int64)
    predictions = np.full(len(labels), -1, dtype=np.int64)
    fold_reports = []
    started = time.perf_counter()
    splits = stratified_folds(labels, folds=folds, seed=seed)
    class_count = int(np.max(labels)) + 1
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        pred, meta = fit_predict_eegnet(
            eeg[train_idx],
            labels[train_idx],
            eeg[test_idx],
            fs,
            classes=class_count,
            seed=seed + fold_idx,
            max_epochs=epochs,
        )
        predictions[test_idx] = pred
        fold_reports.append(
            {
                "fold": fold_idx,
                "train_trials": int(len(train_idx)),
                "test_trials": int(len(test_idx)),
                "metrics": metric_summary(labels[test_idx], pred, label_names=label_names),
                "model": meta,
            }
        )
    elapsed = time.perf_counter() - started
    summary = metric_summary(labels, predictions, label_names=label_names, elapsed=elapsed)
    summary.update({"folds": len(splits), "fold_reports": fold_reports, "predictions": predictions.tolist()})
    return summary


def dataset_quality_report(
    eeg: np.ndarray,
    y: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    label_names: Sequence[str] | None = None,
) -> dict[str, object]:
    x = np.asarray(eeg, dtype=np.float32)
    freqs, psd = compute_psd(x, fs)
    band_names, powers = bandpower_from_psd(freqs, psd, BANDS)
    labels = np.asarray(y, dtype=np.int64)
    class_bandpower = {}
    for label in sorted(set(int(v) for v in labels)):
        key = label_names[label] if label_names and label < len(label_names) else str(label)
        class_bandpower[key] = np.mean(powers[labels == label], axis=(0, 1)).astype(float).tolist()
    channel_variance = np.var(x, axis=(0, 2))
    return {
        "shape": [int(v) for v in x.shape],
        "sampling_rate_hz": int(fs),
        "label_counts": dict(Counter(int(v) for v in labels)),
        "label_names": list(label_names or []),
        "channels": list(channel_names),
        "amplitude": {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "abs_p95": float(np.percentile(np.abs(x), 95)),
            "abs_p99": float(np.percentile(np.abs(x), 99)),
        },
        "channel_variance": {
            "min": float(np.min(channel_variance)),
            "median": float(np.median(channel_variance)),
            "max": float(np.max(channel_variance)),
        },
        "band_names": band_names,
        "class_mean_bandpower": class_bandpower,
    }
