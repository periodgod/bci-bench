"""按实验单元计算平衡准确率，并进行参与者配对统计。

``cell_balanced_accuracy`` reproduces the PROTEUS official-style score used by
every 2026-10 report: balanced accuracy inside each participant x session x
interface cell, then an equal-weight mean over cells. A class absent from a
cell contributes recall 0 (``absent="zero"``), matching the historical
``cell_score`` implementations; ``absent="skip"`` gives sklearn-style BA.
"""

# 先分组再平均与把所有窗口拼起来算一次 BA 是两种不同统计量。前者让每个
# participant × session × interface 单元等权，避免窗口较多的单元支配总分。

from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np


def _recalls(y: np.ndarray, pred: np.ndarray, n_classes: int, absent: str) -> list[float]:
    """计算每类召回率；空缺类别按协议记 0 或跳过。

    ``zero`` 会让缺少该类的单元承受惩罚；``skip`` 只平均该单元实际出现的类。
    二者会得到不同 BA，报告时必须记录采用哪一种。
    """
    out = []
    for label in range(n_classes):
        hit = y == label
        if np.any(hit):
            out.append(float(np.mean(pred[hit] == label)))
        elif absent == "zero":
            out.append(0.0)
    return out


def grouped_balanced_accuracy(
    y: np.ndarray, pred: np.ndarray, keys: Sequence[np.ndarray], n_classes: int, absent: str = "zero"
) -> tuple[float, list[dict[str, object]]]:
    """先逐组计算 BA，再对组等权平均，并返回每组明细。

    ``keys`` 是一组与标签逐行对齐的分组数组，例如参与者、会话和接口。
    唯一键组合定义一个 group。返回的明细保留每组类别召回和窗口数，便于审计
    空类别和单元样本量。
    """
    if absent not in ("zero", "skip"):
        raise ValueError("absent must be 'zero' or 'skip'")
    y, pred = np.asarray(y), np.asarray(pred)
    keys = [np.asarray(k).astype(str) for k in keys]
    cells = []
    for group in sorted(set(zip(*(k.tolist() for k in keys)))):
        mask = np.ones(len(y), bool)
        for k, value in zip(keys, group):
            mask &= k == value
        recalls = _recalls(y[mask], pred[mask], n_classes, absent)
        cells.append({"group": list(group), "balanced_accuracy": float(np.mean(recalls)),
                      "class_recalls": recalls, "n_windows": int(mask.sum())})
    return float(np.mean([c["balanced_accuracy"] for c in cells])), cells


def cell_balanced_accuracy(y, pred, subjects, sessions, interfaces, n_classes: int = 3,
                           absent: str = "zero") -> float:
    """PROTEUS 主指标：参与者 × 会话 × 接口单元等权 BA。

    传入的是硬标签 ``pred``，不是概率。三个分组数组必须与 ``y`` 行对齐。
    默认缺类记零，复现历史 cell_score；因此不要与 sklearn 在全体窗口上计算的
    普通 balanced accuracy 混为一谈。
    """
    return grouped_balanced_accuracy(y, pred, (subjects, sessions, interfaces), n_classes, absent)[0]


def stream_mean_balanced_accuracy(y, pred, subjects, sessions, n_classes: int = 3, absent: str = "skip") -> float:
    """NeuralBench 流式指标：先算参与者 × 会话流的 BA，再对流等权平均。"""
    return grouped_balanced_accuracy(y, pred, (subjects, sessions), n_classes, absent)[0]


def summarize_predictions(y, prob_or_pred, subjects, sessions, interfaces, n_classes: int = 3) -> dict[str, object]:
    """汇总一套 OOF 预测。

    ``prob_or_pred`` 可为 ``(N, K)`` 概率矩阵或 ``(N,)`` 硬标签。概率会逐行
    ``argmax`` 成类别；本函数不会校准、平均种子或检查 OOF 是否真的来自折外预测。
    输出同时包含 cell 等权分数、先算参与者再平均的分数、各类召回和单元明细。
    """
    arr = np.asarray(prob_or_pred)
    pred = arr.argmax(1) if arr.ndim == 2 else arr.astype(np.int64)
    y = np.asarray(y)
    cell, cells = grouped_balanced_accuracy(y, pred, (subjects, sessions, interfaces), n_classes)
    subjects = np.asarray(subjects).astype(str)
    per_subject = {p: float(np.mean([c["balanced_accuracy"] for c in cells if c["group"][0] == p]))
                   for p in sorted(set(subjects.tolist()))}
    return {
        "cell_ba": cell,
        "subject_ba": float(np.mean(list(per_subject.values()))),
        "subjects": per_subject,
        "cell_recalls": np.mean([c["class_recalls"] for c in cells], axis=0).tolist(),
        "pooled_recalls": [float(np.mean(pred[y == c] == c)) if np.any(y == c) else 0.0 for c in range(n_classes)],
        "n_cells": len(cells),
        "cells": cells,
    }


def mean_summaries(summaries: Sequence[dict[str, object]]) -> dict[str, object]:
    """Average several ``summarize_predictions`` outputs (e.g. random-init seeds).

    这是先分别对每个种子评分、再对分数求均值；不是把概率先平均后重新评分的
    概率集成。二者回答不同问题，随机初始化对照应使用这里的“种子分数均值”。
    """
    first = summaries[0]
    return {
        "cell_ba": float(np.mean([s["cell_ba"] for s in summaries])),
        "subject_ba": float(np.mean([s["subject_ba"] for s in summaries])),
        "subjects": {p: float(np.mean([s["subjects"][p] for s in summaries])) for p in first["subjects"]},
        "cells": [{**c, "balanced_accuracy": float(np.mean([s["cells"][i]["balanced_accuracy"] for s in summaries]))}
                  for i, c in enumerate(first["cells"])],
    }


def paired_subject_bootstrap(
    left: dict[str, object], right: dict[str, object], n_boot: int = 20000, seed: int = 20261019
) -> dict[str, object]:
    """Paired participant-cluster bootstrap of two ``summarize_predictions`` outputs.

    对参与者有放回抽样，左右两种方法使用同一批抽样参与者（配对）；计算 cell
    区间时再保留每位参与者的 cell 数权重，使估计目标与总体 cell 等权分数一致。
    ``seed`` 决定重抽样序列。区间反映参与者抽样不确定性，不会消除公开参与者
    已参与研发带来的证据等级限制。
    """
    people = sorted(left["subjects"])
    if people != sorted(right["subjects"]):
        raise ValueError("left and right cover different participants")
    delta = np.array([left["subjects"][p] - right["subjects"][p] for p in people])
    weights = np.array([sum(c["group"][0] == p for c in left["cells"]) for p in people])
    draws = np.random.default_rng(seed).integers(0, len(people), size=(n_boot, len(people)))
    sampled, sampled_w = delta[draws], weights[draws]
    cell_boot = (sampled * sampled_w).sum(1) / sampled_w.sum(1)
    return {
        "cell_difference": float(left["cell_ba"] - right["cell_ba"]),
        "cell_cluster_bootstrap_95ci": np.quantile(cell_boot, [0.025, 0.975]).tolist(),
        "subject_difference": float(delta.mean()),
        "subject_bootstrap_95ci": np.quantile(sampled.mean(1), [0.025, 0.975]).tolist(),
        "improved_subjects": int((delta > 1e-12).sum()),
        "unchanged_subjects": int((np.abs(delta) <= 1e-12).sum()),
        "subject_differences": dict(zip(people, delta.tolist())),
        "n_boot": n_boot,
        "seed": seed,
    }


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    """对参与者级配对差值做双侧精确 sign-flip 检验。

    枚举每个差值的正负号组合，计算零假设下绝对均值不小于观测值的比例；参与者
    数最多 20，否则组合数 ``2**n`` 会过大。它检验的是配对差值的符号可交换假设，
    不是独立窗口层面的显著性检验。
    """
    d = np.asarray(differences, dtype=np.float64)
    if len(d) > 20:
        raise ValueError("exact enumeration is limited to 20 participants")
    observed = abs(d.mean())
    signs = np.array(list(product((-1.0, 1.0), repeat=len(d))))
    null = np.abs((signs * d).mean(1))
    return float(np.mean(null >= observed - 1e-12))


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 20000, seed: int = 20261015) -> list[float]:
    """对输入单位有放回重抽样，返回均值的百分位 95% 区间。

    输入每一项应是独立抽样单位（例如参与者差值），不应把同一人的窗口当成
    独立观测传入，否则区间会过窄。
    """
    v = np.asarray(values, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n_boot, len(v)))
    return np.quantile(v[idx].mean(1), [0.025, 0.975]).tolist()
