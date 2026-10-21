"""多分类概率校准、置信度与选择性预测诊断。"""

from __future__ import annotations

import numpy as np


def _validated_probabilities(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs, y = np.asarray(probabilities, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or len(probs) != len(y) or probs.shape[1] < 2:
        raise ValueError("probabilities must have shape (samples, classes) and align with labels")
    if not np.isfinite(probs).all() or np.any(probs < 0) or not np.allclose(probs.sum(1), 1.0, atol=1e-5):
        raise ValueError("probabilities must be finite, non-negative, and sum to one")
    if np.any(y < 0) or np.any(y >= probs.shape[1]):
        raise ValueError("labels must index probability columns")
    return probs, y


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """計算 top-label ECE；每個置信度區間按样本数加权。"""
    probs, y = _validated_probabilities(probabilities, labels)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        mask = (confidence > edges[index] if index == 0 else confidence >= edges[index]) & (confidence <= edges[index + 1])
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return ece


def multiclass_brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """计算多分类 Brier 分数（逐样本类别维平方误差和，再对样本平均）。"""
    probs, y = _validated_probabilities(probabilities, labels)
    target = np.eye(probs.shape[1], dtype=np.float64)[y]
    return float(np.square(probs - target).sum(axis=1).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """仅用验证折 logits 拟合单一正温度；调用方应保留独立测试折评估校准。"""
    from scipy.optimize import minimize_scalar

    scores, y = np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    if scores.ndim != 2 or len(scores) != len(y) or scores.shape[1] < 2 or not np.isfinite(scores).all():
        raise ValueError("logits must be finite (samples, classes) and align with labels")
    if np.any(y < 0) or np.any(y >= scores.shape[1]):
        raise ValueError("labels must index logit columns")

    def nll(log_temperature: float) -> float:
        scaled = scores / np.exp(log_temperature)
        maximum = scaled.max(axis=1, keepdims=True)
        log_normalizer = maximum[:, 0] + np.log(np.exp(scaled - maximum).sum(axis=1))
        return float(np.mean(log_normalizer - scaled[np.arange(len(y)), y]))

    result = minimize_scalar(nll, bounds=(-5.0, 5.0), method="bounded")
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError("temperature optimization failed")
    return float(np.exp(result.x))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """將正溫度縮放套用於 logits，輸出數值穩定的 softmax 概率。"""
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all() or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("logits must be finite 2-D and temperature must be positive")
    scaled = scores / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def risk_coverage_curve(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按置信度从高到低加入预测，返回 coverage 与对应错误率 risk。"""
    probs, y = _validated_probabilities(probabilities, labels)
    order = np.argsort(-probs.max(axis=1), kind="stable")
    errors = (probs.argmax(axis=1)[order] != y[order]).astype(np.float64)
    counts = np.arange(1, len(y) + 1, dtype=np.float64)
    coverage = np.concatenate(([0.0], counts / len(y)))
    risk = np.concatenate(([0.0], np.cumsum(errors) / counts))
    return coverage, risk
