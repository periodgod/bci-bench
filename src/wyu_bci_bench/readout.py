"""Linear readouts on fixed features, with optional nested participant LOSO for C."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .metrics import cell_balanced_accuracy
from .splits import inner_loso, loso

DEFAULT_CS = (0.003, 0.03, 0.3)


def linear_probe(C: float = 0.003, max_iter: int = 2000):
    """构造标准化 + 类别平衡 L2 LogisticRegression 的 sklearn Pipeline。

    先封装成 Pipeline 可确保 StandardScaler 只在调用 ``fit`` 的训练数据上估计，
    而非提前对全体数据标准化。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter,
                                                              random_state=33))


def probe_proba(train_x, train_y, test_x, C: float = 0.003, max_iter: int = 2000) -> np.ndarray:
    """在训练特征上拟合固定 C 的线性探针，并返回测试行类别概率。"""
    return linear_probe(C, max_iter).fit(train_x, train_y).predict_proba(test_x).astype(np.float32)


def loso_probe(features: np.ndarray, y: np.ndarray, subjects: np.ndarray, C: float = 0.003,
               max_iter: int = 2000) -> np.ndarray:
    """执行固定 C 的外层 LOSO，并把每个留出参与者的预测写回原行位置。

    返回 ``(N,K)`` 概率矩阵；类别数按 ``y`` 最大编号推断，要求类别编号从 0 连续。
    """
    n_classes = int(y.max()) + 1
    probs = np.full((len(y), n_classes), np.nan, np.float32)
    for split in loso(subjects):
        probs[split.test] = probe_proba(features[split.train], y[split.train], features[split.test], C, max_iter)
    return probs


def nested_loso_readout(features: Mapping[str, np.ndarray], y: np.ndarray, subjects: np.ndarray,
                        sessions: np.ndarray, interfaces: np.ndarray, Cs: Sequence[float] = DEFAULT_CS,
                        max_iter: int = 1000):
    """外层 LOSO 评估，内层 LOSO 在训练参与者中选择 C 和特征条件。

    返回 ``(scores, folds, probs)``。每种条件的 ``probs[name]`` 使用其内层选出的 C；
    ``scores["selected_policy"]`` 则按每个外层折选择出的条件拼接预测。测试参与者
    不参与选择。相同分数时按代码中的确定性排序规则打破平局，便于复现。
    """
    n_classes = int(y.max()) + 1
    probs = {k: np.full((len(y), n_classes), np.nan, np.float32) for k in features}
    folds = []
    for outer in loso(subjects):
        rows = []
        for name, X in features.items():
            for C in Cs:
                pred = np.full(len(y), -1, np.int64)
                for inner in inner_loso(subjects, outer.train):
                    clf = linear_probe(C, max_iter).fit(X[inner.train], y[inner.train])
                    pred[inner.test] = clf.predict(X[inner.test])
                m = outer.train
                rows.append((cell_balanced_accuracy(y[m], pred[m], subjects[m], sessions[m], interfaces[m], n_classes),
                             name, C))
        records = {}
        for name, X in features.items():
            score, _, C = max((r for r in rows if r[1] == name), key=lambda r: (r[0], r[2]))
            clf = linear_probe(C, max_iter).fit(X[outer.train], y[outer.train])
            probs[name][outer.test] = clf.predict_proba(X[outer.test])
            t = outer.test
            records[name] = {"C": C, "inner_cell_ba": score, "outer_cell_ba": cell_balanced_accuracy(
                y[t], probs[name][t].argmax(1), subjects[t], sessions[t], interfaces[t], n_classes)}
        best_score, best_name, best_C = max(rows, key=lambda r: (r[0], r[1], r[2]))
        folds.append({"held_out": outer.info["held_out"], "selected_condition": best_name, "selected_C": best_C,
                      "inner_cell_ba": best_score, "conditions": records})
    chosen = np.full(len(y), -1, np.int64)
    for f in folds:
        m = subjects == f["held_out"]
        chosen[m] = probs[f["selected_condition"]][m].argmax(1)
    for name, p in probs.items():
        if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=1e-5):
            raise RuntimeError(f"incomplete or invalid OOF probabilities: {name}")
    scores = {k: cell_balanced_accuracy(y, v.argmax(1), subjects, sessions, interfaces, n_classes)
              for k, v in probs.items()}
    scores["selected_policy"] = cell_balanced_accuracy(y, chosen, subjects, sessions, interfaces, n_classes)
    return scores, folds, probs
