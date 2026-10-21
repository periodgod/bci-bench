"""留出数据上的特征、通道与频带扰动解释工具。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np


def _balanced_accuracy(y: np.ndarray, prediction: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y, prediction))


def grouped_permutation_importance(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: Mapping[str, Sequence[int]],
    *,
    scoring: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_repeats: int = 10,
    random_state: int = 0,
) -> dict[str, dict[str, float]]:
    """在留出特征上联合打乱每组列，报告分数下降均值和标准差。

    ``estimator`` 必须已在训练折拟合。传入训练数据会把重要性解释限制在
    训练拟合表现，容易高估模型依赖；该函数不替调用者创建评估划分。
    """
    values, labels = np.asarray(X), np.asarray(y)
    if values.ndim != 2 or len(values) != len(labels) or not np.isfinite(values).all():
        raise ValueError("X must be finite 2-D features aligned with y")
    if n_repeats < 1 or not groups:
        raise ValueError("n_repeats and groups must be non-empty")
    score_fn = scoring or _balanced_accuracy
    baseline = float(score_fn(labels, np.asarray(estimator.predict(values))))
    rng = np.random.default_rng(random_state)
    output = {}
    for name, columns in groups.items():
        indices = np.asarray(columns, dtype=int)
        if indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0) or np.any(indices >= values.shape[1]):
            raise ValueError(f"invalid feature columns for group {name!r}")
        drops = []
        for _ in range(n_repeats):
            permutation = rng.permutation(len(values))
            perturbed = values.copy()
            perturbed[:, indices] = values[permutation][:, indices]
            drops.append(baseline - float(score_fn(labels, np.asarray(estimator.predict(perturbed)))))
        output[str(name)] = {"mean": float(np.mean(drops)), "std": float(np.std(drops, ddof=0))}
    return output


def channel_ablation_scores(
    predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    channels: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, float]:
    """在留出 epoch 上将指定电极置零，报告 BA 相对基线的变化。

    这是输入扰动诊断，不是重新训练后的通道选择结果。epoch 形状为
    ``(trials, channels, samples)``，不同数据集应在调用方统一通道顺序。
    """
    epochs, labels = np.asarray(X), np.asarray(y)
    if epochs.ndim != 3 or len(epochs) != len(labels) or not np.isfinite(epochs).all():
        raise ValueError("X must be finite (trials, channels, samples) aligned with y")
    channel_groups = channels or {f"channel_{i}": [i] for i in range(epochs.shape[1])}
    baseline = _balanced_accuracy(labels, np.asarray(predict(epochs)))
    result = {"baseline": baseline}
    for name, indices in channel_groups.items():
        selected = np.asarray(indices, dtype=int)
        if selected.ndim != 1 or len(selected) == 0 or np.any(selected < 0) or np.any(selected >= epochs.shape[1]):
            raise ValueError(f"invalid channel indices for group {name!r}")
        ablated = epochs.copy()
        ablated[:, selected, :] = 0
        score = _balanced_accuracy(labels, np.asarray(predict(ablated)))
        result[str(name)] = score - baseline
    return result


def band_ablation_scores(
    predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    sampling_rate: float,
    bands: Mapping[str, tuple[float, float]],
    *,
    order: int = 4,
) -> dict[str, float]:
    """对留出 epoch 施加 Butterworth 带阻后计算 BA 变化。

    每个频带分别移除；该数值表示模型对该频率区域输入的敏感度，不是
    频带的因果效应。通带与标签必须在实验设计阶段确定。
    """
    from scipy.signal import butter, sosfiltfilt

    epochs, labels = np.asarray(X), np.asarray(y)
    nyquist = sampling_rate / 2
    if epochs.ndim != 3 or len(epochs) != len(labels) or not np.isfinite(epochs).all():
        raise ValueError("X must be finite (trials, channels, samples) aligned with y")
    if sampling_rate <= 0 or order < 1 or not bands:
        raise ValueError("sampling_rate, order and bands must be positive/non-empty")
    baseline = _balanced_accuracy(labels, np.asarray(predict(epochs)))
    result = {"baseline": baseline}
    for name, (low, high) in bands.items():
        if not 0 < low < high < nyquist:
            raise ValueError(f"band {name!r} must satisfy 0 < low < high < Nyquist")
        sos = butter(order, (low, high), btype="bandstop", fs=sampling_rate, output="sos")
        filtered = sosfiltfilt(sos, epochs, axis=-1)
        score = _balanced_accuracy(labels, np.asarray(predict(filtered)))
        result[str(name)] = score - baseline
    return result
