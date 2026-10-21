"""本地单段 EEG 数据集的交叉验证、经典模型基准和质量摘要。

本模块中的 ``stratified_folds`` 是按试次随机分层，不会隔离参与者或会话，适用于
同一数据池内的基线比较；若研究问题是跨参与者/跨会话泛化，必须改用
``splits.py`` 中带组的协议，并使用对应分组指标。所有模型需在每折训练部分重新
拟合特征标准化、特征选择、参考矩阵或分类器。
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Sequence

import numpy as np

from .capsule import fit_predict_capsule_net
from .deep import fit_predict_eegnet
from .features import filterbank_spatial_logvar_features, log_bandpower_features, select_top_features
from .models import (
    DiagonalGaussianNB,
    FilterBankLDAModel,
    KNearestNeighborsClassifier,
    NearestCentroidClassifier,
    StandardizedDiagonalLDA,
)
from .minirocket import MiniRocketLDAModel
from .riemann import FilterBankMDMClassifier, FilterBankTangentSpaceLDAModel, RiemannianMDMClassifier, TangentSpaceLDAModel
from .sklearn_ml import EEGFeatureSklearnModel
from .treeboost import EEGFeatureXGBModel
from .signal import BANDS, bandpower_from_psd, compute_psd


def stratified_folds(y: np.ndarray, folds: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """构造确定性的分层试次级折，尽量让每折各类别数量相近。

    有效折数不会超过任一类的样本数；每类至少两个样本才可做交叉验证。注意该
    函数不知道 subject/session 元信息，因此绝不能用它声称参与者隔离评估。
    """
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
    """按给定类别顺序累计真实类别为行、预测类别为列的计数矩阵。"""
    lookup = {int(label): idx for idx, label in enumerate(classes)}
    mat = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        mat[lookup[int(truth)], lookup[int(pred)]] += 1
    return mat.tolist()


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[int]) -> float:
    """分别计算各类别 F1 再等权平均；没有预测正例的类别记 0。"""
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    scores = []
    for label in classes:
        tp = float(np.sum((truth == label) & (pred == label)))
        fp = float(np.sum((truth != label) & (pred == label)))
        fn = float(np.sum((truth == label) & (pred != label)))
        precision = tp / max(tp + fp, 1e-12)
        recall = tp / max(tp + fn, 1e-12)
        if precision + recall <= 0.0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def metric_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str] | None = None,
    elapsed: float = 0.0,
) -> dict[str, object]:
    """汇总硬标签预测的 accuracy、macro BA、macro-F1、召回和混淆矩阵。"""
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
        "macro_f1": _macro_f1(truth, pred, classes),
        "per_class_recall": per_class,
        "confusion_matrix": confusion_matrix(truth, pred, classes),
        "label_order": [label_names[label] if label_names and label < len(label_names) else str(label) for label in classes],
        "elapsed_seconds": float(elapsed),
    }


def _hard_vote(prediction_rows: Sequence[np.ndarray], classes: Sequence[int]) -> np.ndarray:
    """逐样本多数表决；平票时使用最早 voter 的预测以保证确定性。"""
    if not prediction_rows:
        raise ValueError("At least one prediction row is required for voting")
    stacked = np.stack([np.asarray(row, dtype=np.int64) for row in prediction_rows], axis=1)
    class_list = [int(item) for item in classes]
    voted = []
    for row in stacked:
        counts = {label: int(np.sum(row == label)) for label in class_list}
        best_count = max(counts.values())
        tied = {label for label, count in counts.items() if count == best_count}
        if len(tied) == 1:
            voted.append(next(iter(tied)))
            continue
        # Deterministic tie break: trust earlier voter order first.
        for prediction in row:
            if int(prediction) in tied:
                voted.append(int(prediction))
                break
    return np.asarray(voted, dtype=np.int64)


def _fit_predict_vote_models(
    train_eeg: np.ndarray,
    train_y: np.ndarray,
    test_eeg: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    seed: int,
) -> list[tuple[str, np.ndarray, dict[str, object]]]:
    labels = np.asarray(train_y, dtype=np.int64)
    voters: list[tuple[str, object]] = [
        ("fbcsp_lda", FilterBankLDAModel(top_k=48)),
        ("riemann_fb_ts", FilterBankTangentSpaceLDAModel(shrinkage=0.1, top_k=64)),
        ("xgb_eeg", EEGFeatureXGBModel(include_tangent=True, top_k=96, seed=seed, max_depth=2, n_estimators=80)),
        ("minirocket", MiniRocketLDAModel(band=(8.0, 30.0), n_kernels=128, top_k=64, seed=seed)),
    ]
    outputs: list[tuple[str, np.ndarray, dict[str, object]]] = []
    for name, model in voters:
        if isinstance(model, FilterBankLDAModel):
            model.fit(train_eeg, labels, fs, channel_names)
            selected = model.feature_names_[: min(12, len(model.feature_names_ or []))]
        elif isinstance(model, FilterBankTangentSpaceLDAModel):
            model.fit(train_eeg, labels, fs, channel_names)
            selected = model.feature_names_[: min(12, len(model.feature_names_ or []))]
        elif isinstance(model, EEGFeatureXGBModel):
            model.fit(train_eeg, labels, fs, channel_names)
            selected = model.feature_names_[: min(12, len(model.feature_names_ or []))]
        elif isinstance(model, MiniRocketLDAModel):
            model.fit(train_eeg, labels, fs, channel_names)
            selected = model.feature_names_[: min(12, len(model.feature_names_ or []))]
        else:
            raise TypeError(f"Unsupported voter type: {type(model)!r}")
        outputs.append((name, model.predict(test_eeg), {"selected_features": selected}))
    return outputs


def _fit_predict_ml_best_vote_models(
    train_eeg: np.ndarray,
    train_y: np.ndarray,
    test_eeg: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    seed: int,
    include_xgb: bool = True,
) -> list[tuple[str, np.ndarray, dict[str, object]]]:
    labels = np.asarray(train_y, dtype=np.int64)
    voters: list[tuple[str, object]] = [
        ("fbcsp_lda", FilterBankLDAModel(top_k=48)),
        (
            "extra_trees_full_top128",
            EEGFeatureSklearnModel(
                classifier_name="extra_trees",
                feature_mode="full",
                top_k=128,
                seed=seed + 3,
            ),
        ),
        (
            "gaussian_nb_tangent_top64",
            EEGFeatureSklearnModel(
                classifier_name="gaussian_nb",
                feature_mode="tangent",
                top_k=64,
                seed=seed + 8,
            ),
        ),
        (
            "random_forest_fixed_top96",
            EEGFeatureSklearnModel(
                classifier_name="random_forest",
                feature_mode="fixed",
                top_k=96,
                seed=seed + 4,
            ),
        ),
        (
            "rbf_svm_fixed_top96",
            EEGFeatureSklearnModel(
                classifier_name="rbf_svm",
                feature_mode="fixed",
                top_k=96,
                seed=seed + 1,
            ),
        ),
    ]
    if include_xgb:
        voters.append(
            ("xgb_eeg", EEGFeatureXGBModel(include_tangent=True, top_k=96, seed=seed + 1, max_depth=2, n_estimators=80))
        )
    outputs: list[tuple[str, np.ndarray, dict[str, object]]] = []
    for name, model in voters:
        model.fit(train_eeg, labels, fs, channel_names)
        feature_names = getattr(model, "feature_names_", None) or []
        selected = feature_names[: min(12, len(feature_names))]
        outputs.append((name, model.predict(test_eeg), {"selected_features": selected}))
    return outputs


def cross_validate_voting_ensemble(
    eeg: np.ndarray,
    y: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    label_names: Sequence[str] | None = None,
    folds: int = 5,
    seed: int = 42,
    include_capsule: bool = False,
    capsule_epochs: int = 20,
    voter_set: str = "simple",
) -> dict[str, object]:
    """逐折拟合若干 EEG 分类器并组合硬标签多数票。

    每个基础分类器只在当前训练折拟合，测试折仅用于预测。输出同时报告集成与各
    voter 的结果。切分是试次级分层，因此如果同一参与者有多个试次，它们可能跨折；
    这不是跨参与者泛化结果。
    """
    labels = np.asarray(y, dtype=np.int64)
    splits = stratified_folds(labels, folds=folds, seed=seed)
    classes = sorted(set(int(v) for v in labels.tolist()))
    predictions = np.full(len(labels), -1, dtype=np.int64)
    voter_predictions: dict[str, np.ndarray] = {}
    fold_reports = []
    started = time.perf_counter()
    class_count = int(np.max(labels)) + 1

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        fold_seed = int(seed) + fold_idx * 17
        if voter_set == "simple":
            voters = _fit_predict_vote_models(
                eeg[train_idx],
                labels[train_idx],
                eeg[test_idx],
                fs,
                channel_names,
                seed=fold_seed,
            )
        elif voter_set in {"ml_best", "ml_pruned"}:
            voters = _fit_predict_ml_best_vote_models(
                eeg[train_idx],
                labels[train_idx],
                eeg[test_idx],
                fs,
                channel_names,
                seed=int(seed),
                include_xgb=voter_set == "ml_best",
            )
        else:
            raise ValueError(f"Unknown voter_set: {voter_set}")
        if include_capsule:
            capsule_pred, capsule_meta = fit_predict_capsule_net(
                eeg[train_idx],
                labels[train_idx],
                eeg[test_idx],
                fs,
                classes=class_count,
                seed=fold_seed,
                max_epochs=capsule_epochs,
            )
            voters.append(("capsule_net", capsule_pred, {"model": capsule_meta}))
        pred = _hard_vote([item[1] for item in voters], classes=classes)
        predictions[test_idx] = pred
        for voter_name, voter_pred, _ in voters:
            if voter_name not in voter_predictions:
                voter_predictions[voter_name] = np.full(len(labels), -1, dtype=np.int64)
            voter_predictions[voter_name][test_idx] = voter_pred
        fold_reports.append(
            {
                "fold": fold_idx,
                "train_trials": int(len(train_idx)),
                "test_trials": int(len(test_idx)),
                "voters": [
                    {
                        "name": voter_name,
                        "metrics": metric_summary(labels[test_idx], voter_pred, label_names=label_names),
                        **meta,
                    }
                    for voter_name, voter_pred, meta in voters
                ],
                "metrics": metric_summary(labels[test_idx], pred, label_names=label_names),
            }
        )

    elapsed = time.perf_counter() - started
    summary = metric_summary(labels, predictions, label_names=label_names, elapsed=elapsed)
    model_name = "vote_capsule" if include_capsule else "vote_simple"
    if voter_set == "ml_best":
        model_name = "vote_ml_best_capsule" if include_capsule else "vote_ml_best"
    elif voter_set == "ml_pruned":
        model_name = "vote_ml_pruned_capsule" if include_capsule else "vote_ml_pruned"
    summary.update(
        {
            "feature_set": f"hard_vote_{voter_set}",
            "model": model_name,
            "folds": len(splits),
            "fold_reports": fold_reports,
            "predictions": predictions.tolist(),
            "voter_predictions": {name: pred.tolist() for name, pred in voter_predictions.items()},
            "voter_metrics": {
                name: metric_summary(labels, pred, label_names=label_names)
                for name, pred in voter_predictions.items()
            },
        }
    )
    return summary


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
    """对 FBCSP 风格 LDA 做分层试次交叉验证，特征选择在每折内完成。"""
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


def _simple_feature_sets(
    eeg: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
) -> dict[str, tuple[np.ndarray, list[str]]]:
    return {
        "mi_8_30_logvar": log_bandpower_features(
            eeg,
            fs,
            bands=((8.0, 13.0), (13.0, 20.0), (20.0, 30.0)),
            car=True,
            channel_names=channel_names,
        ),
        "fb_4_40_logvar": log_bandpower_features(
            eeg,
            fs,
            bands=((4.0, 8.0), (8.0, 13.0), (13.0, 20.0), (20.0, 30.0), (30.0, 40.0)),
            car=True,
            channel_names=channel_names,
        ),
        "fb_spatial_logvar": filterbank_spatial_logvar_features(eeg, fs, channel_names),
    }


def _simple_model_factories() -> dict[str, object]:
    return {
        "nearest_centroid": NearestCentroidClassifier,
        "diag_lda": StandardizedDiagonalLDA,
        "diag_gaussian_nb": DiagonalGaussianNB,
        "knn_3": lambda: KNearestNeighborsClassifier(k=3),
        "knn_5": lambda: KNearestNeighborsClassifier(k=5),
    }


def cross_validate_simple_models(
    eeg: np.ndarray,
    y: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    label_names: Sequence[str] | None = None,
    folds: int = 5,
    seed: int = 42,
    top_k: int | None = None,
) -> dict[str, object]:
    """Evaluate dependency-light classical models on log-variance features."""
    labels = np.asarray(y, dtype=np.int64)
    splits = stratified_folds(labels, folds=folds, seed=seed)
    features = _simple_feature_sets(eeg, fs, channel_names)
    factories = _simple_model_factories()
    results = []
    started = time.perf_counter()

    for feature_name, (feature_x, feature_names) in features.items():
        for model_name, factory in factories.items():
            predictions = np.full(len(labels), -1, dtype=np.int64)
            fold_reports = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                if top_k is None:
                    selected_idx = np.arange(feature_x.shape[1], dtype=np.int64)
                    train_features = feature_x[train_idx]
                    test_features = feature_x[test_idx]
                else:
                    train_features, selected_idx = select_top_features(feature_x[train_idx], labels[train_idx], top_k)
                    test_features = feature_x[test_idx][:, selected_idx]
                model = factory()
                model.fit(train_features, labels[train_idx])
                pred = model.predict(test_features)
                predictions[test_idx] = pred
                fold_reports.append(
                    {
                        "fold": fold_idx,
                        "train_trials": int(len(train_idx)),
                        "test_trials": int(len(test_idx)),
                        "metrics": metric_summary(labels[test_idx], pred, label_names=label_names),
                        "selected_features": [
                            feature_names[int(idx)] for idx in selected_idx[: min(12, len(selected_idx))]
                        ],
                    }
                )
            summary = metric_summary(labels, predictions, label_names=label_names)
            summary.update(
                {
                    "feature_set": feature_name,
                    "model": model_name,
                    "folds": len(splits),
                    "fold_reports": fold_reports,
                    "predictions": predictions.tolist(),
                }
            )
            results.append(summary)

    results.sort(
        key=lambda item: (
            float(item["balanced_accuracy"]),
            float(item["macro_f1"]),
            float(item["accuracy"]),
        ),
        reverse=True,
    )
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": float(elapsed),
        "folds": len(splits),
        "best": results[0],
        "models": results,
    }


def _riemannian_model_factories() -> dict[str, object]:
    return {
        "mdm_8_30": lambda: RiemannianMDMClassifier(band=(8.0, 30.0), shrinkage=0.1),
        "mdm_4_40": lambda: RiemannianMDMClassifier(band=(4.0, 40.0), shrinkage=0.1),
        "fb_mdm": lambda: FilterBankMDMClassifier(shrinkage=0.1),
        "ts_lda_8_30": lambda: TangentSpaceLDAModel(band=(8.0, 30.0), shrinkage=0.1),
        "ts_lda_4_40": lambda: TangentSpaceLDAModel(band=(4.0, 40.0), shrinkage=0.1),
        "fb_ts_lda_top32": lambda: FilterBankTangentSpaceLDAModel(shrinkage=0.1, top_k=32),
        "fb_ts_lda_top64": lambda: FilterBankTangentSpaceLDAModel(shrinkage=0.1, top_k=64),
        "fb_ts_lda_top96": lambda: FilterBankTangentSpaceLDAModel(shrinkage=0.1, top_k=96),
    }


def cross_validate_riemannian_models(
    eeg: np.ndarray,
    y: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    label_names: Sequence[str] | None = None,
    folds: int = 5,
    seed: int = 42,
) -> dict[str, object]:
    """Evaluate small-sample covariance/Riemannian EEG classifiers."""
    labels = np.asarray(y, dtype=np.int64)
    splits = stratified_folds(labels, folds=folds, seed=seed)
    results = []
    started = time.perf_counter()

    for model_name, factory in _riemannian_model_factories().items():
        predictions = np.full(len(labels), -1, dtype=np.int64)
        fold_reports = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            model = factory()
            if isinstance(model, FilterBankTangentSpaceLDAModel):
                model.fit(eeg[train_idx], labels[train_idx], fs, channel_names)
                selected = model.feature_names_[: min(12, len(model.feature_names_ or []))]
            else:
                model.fit(eeg[train_idx], labels[train_idx], fs)
                selected = []
            pred = model.predict(eeg[test_idx])
            predictions[test_idx] = pred
            fold_reports.append(
                {
                    "fold": fold_idx,
                    "train_trials": int(len(train_idx)),
                    "test_trials": int(len(test_idx)),
                    "metrics": metric_summary(labels[test_idx], pred, label_names=label_names),
                    "selected_features": selected,
                }
            )
        summary = metric_summary(labels, predictions, label_names=label_names)
        summary.update(
            {
                "feature_set": "riemannian_covariance",
                "model": model_name,
                "folds": len(splits),
                "fold_reports": fold_reports,
                "predictions": predictions.tolist(),
            }
        )
        results.append(summary)

    results.sort(
        key=lambda item: (
            float(item["balanced_accuracy"]),
            float(item["macro_f1"]),
            float(item["accuracy"]),
        ),
        reverse=True,
    )
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": float(elapsed),
        "folds": len(splits),
        "best": results[0],
        "models": results,
    }


def _minirocket_model_factories(seed: int) -> dict[str, object]:
    return {
        "mr_8_30_k128_top64": lambda: MiniRocketLDAModel(band=(8.0, 30.0), n_kernels=128, top_k=64, seed=seed),
        "mr_4_40_k128_top64": lambda: MiniRocketLDAModel(band=(4.0, 40.0), n_kernels=128, top_k=64, seed=seed + 1),
        "mr_8_30_k256_top128": lambda: MiniRocketLDAModel(band=(8.0, 30.0), n_kernels=256, top_k=128, seed=seed + 2),
    }


def cross_validate_minirocket_models(
    eeg: np.ndarray,
    y: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    label_names: Sequence[str] | None = None,
    folds: int = 5,
    seed: int = 42,
) -> dict[str, object]:
    """Evaluate MiniRocket-style deterministic convolutional EEG features."""
    labels = np.asarray(y, dtype=np.int64)
    splits = stratified_folds(labels, folds=folds, seed=seed)
    results = []
    started = time.perf_counter()

    for model_name, factory in _minirocket_model_factories(seed).items():
        predictions = np.full(len(labels), -1, dtype=np.int64)
        fold_reports = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            model = factory()
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
        summary = metric_summary(labels, predictions, label_names=label_names)
        summary.update(
            {
                "feature_set": "minirocket_ppv",
                "model": model_name,
                "folds": len(splits),
                "fold_reports": fold_reports,
                "predictions": predictions.tolist(),
            }
        )
        results.append(summary)

    results.sort(
        key=lambda item: (
            float(item["balanced_accuracy"]),
            float(item["macro_f1"]),
            float(item["accuracy"]),
        ),
        reverse=True,
    )
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": float(elapsed),
        "folds": len(splits),
        "best": results[0],
        "models": results,
    }


def _xgb_model_factories(seed: int) -> dict[str, object]:
    return {
        "xgb_eeg_fixed_top64_depth2": lambda: EEGFeatureXGBModel(
            include_tangent=False,
            top_k=64,
            seed=seed,
            max_depth=2,
            n_estimators=80,
        ),
        "xgb_eeg_full_top96_depth2": lambda: EEGFeatureXGBModel(
            include_tangent=True,
            top_k=96,
            seed=seed + 1,
            max_depth=2,
            n_estimators=80,
        ),
        "xgb_eeg_full_top128_depth3": lambda: EEGFeatureXGBModel(
            include_tangent=True,
            top_k=128,
            seed=seed + 2,
            max_depth=3,
            n_estimators=100,
        ),
    }


def cross_validate_xgb_models(
    eeg: np.ndarray,
    y: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    label_names: Sequence[str] | None = None,
    folds: int = 5,
    seed: int = 42,
) -> dict[str, object]:
    """Evaluate XGBoost on EEG-aware features, with split-local feature selection."""
    labels = np.asarray(y, dtype=np.int64)
    splits = stratified_folds(labels, folds=folds, seed=seed)
    results = []
    started = time.perf_counter()

    for model_name, factory in _xgb_model_factories(seed).items():
        predictions = np.full(len(labels), -1, dtype=np.int64)
        fold_reports = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            model = factory()
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
        summary = metric_summary(labels, predictions, label_names=label_names)
        summary.update(
            {
                "feature_set": "eeg_features_xgb",
                "model": model_name,
                "folds": len(splits),
                "fold_reports": fold_reports,
                "predictions": predictions.tolist(),
            }
        )
        results.append(summary)

    results.sort(
        key=lambda item: (
            float(item["balanced_accuracy"]),
            float(item["macro_f1"]),
            float(item["accuracy"]),
        ),
        reverse=True,
    )
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": float(elapsed),
        "folds": len(splits),
        "best": results[0],
        "models": results,
    }


def _ml_suite_factories(seed: int) -> dict[str, object]:
    return {
        "rbf_svm_fixed_top96": lambda: EEGFeatureSklearnModel(
            classifier_name="rbf_svm",
            feature_mode="fixed",
            top_k=96,
            seed=seed + 1,
        ),
        "extra_trees_full_top128": lambda: EEGFeatureSklearnModel(
            classifier_name="extra_trees",
            feature_mode="full",
            top_k=128,
            seed=seed + 3,
        ),
        "random_forest_fixed_top96": lambda: EEGFeatureSklearnModel(
            classifier_name="random_forest",
            feature_mode="fixed",
            top_k=96,
            seed=seed + 4,
        ),
        "hist_gradient_full_top96": lambda: EEGFeatureSklearnModel(
            classifier_name="hist_gradient",
            feature_mode="full",
            top_k=96,
            seed=seed + 5,
        ),
        "decision_tree_fixed_top64": lambda: EEGFeatureSklearnModel(
            classifier_name="decision_tree",
            feature_mode="fixed",
            top_k=64,
            seed=seed + 6,
        ),
        "ada_boost_fixed_top64": lambda: EEGFeatureSklearnModel(
            classifier_name="ada_boost",
            feature_mode="fixed",
            top_k=64,
            seed=seed + 7,
        ),
        "gaussian_nb_tangent_top64": lambda: EEGFeatureSklearnModel(
            classifier_name="gaussian_nb",
            feature_mode="tangent",
            top_k=64,
            seed=seed + 8,
        ),
    }


def cross_validate_ml_suite_models(
    eeg: np.ndarray,
    y: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    label_names: Sequence[str] | None = None,
    folds: int = 5,
    seed: int = 42,
) -> dict[str, object]:
    """Evaluate a suite of sklearn ML classifiers on EEG-aware features."""
    labels = np.asarray(y, dtype=np.int64)
    splits = stratified_folds(labels, folds=folds, seed=seed)
    results = []
    started = time.perf_counter()

    for model_name, factory in _ml_suite_factories(seed).items():
        predictions = np.full(len(labels), -1, dtype=np.int64)
        fold_reports = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            model = factory()
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
        summary = metric_summary(labels, predictions, label_names=label_names)
        summary.update(
            {
                "feature_set": "eeg_features_sklearn",
                "model": model_name,
                "folds": len(splits),
                "fold_reports": fold_reports,
                "predictions": predictions.tolist(),
            }
        )
        results.append(summary)

    results.sort(
        key=lambda item: (
            float(item["balanced_accuracy"]),
            float(item["macro_f1"]),
            float(item["accuracy"]),
        ),
        reverse=True,
    )
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": float(elapsed),
        "folds": len(splits),
        "best": results[0],
        "models": results,
    }


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
    """逐折训练 EEGNet 并汇总折外硬标签预测；仍使用试次级分层切分。"""
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
    """计算幅度、通道方差、频带功率和类别分布等数据质量摘要。

    该报告用于发现明显的单位、坏通道或频谱异常，不会自动判定数据可用于某个
    科学结论，也不会替代原始文件/预处理合同审计。
    """
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
