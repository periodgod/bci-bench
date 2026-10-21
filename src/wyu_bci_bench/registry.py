"""One OOF / run-record format for every experiment.

``save_run`` writes ``oof.npz`` (labels, groups and one probability array per
condition) plus ``report.json`` with the protocol, scores and provenance
hashes. ``load_oof`` reads both the new format and the historical files whose
probability keys are bare names, ``*_probabilities`` or seed-stacked arrays.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .grouped import file_sha256
from .metrics import summarize_predictions

META_KEYS = ("y", "subjects", "sessions", "interfaces")


@dataclass(frozen=True)
class OOF:
    """一次运行的折外预测及其逐窗口分组元信息。

    ``probs`` 的每项形状为 ``(N,K)``，或多种子堆叠形状 ``(S,N,K)``。
    OOF 表示每行预测原则上来自未训练该行样本的模型；文件格式本身无法证明这一点，
    仍需要协议报告和生成代码支持。
    """
    y: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    interfaces: np.ndarray
    probs: dict[str, np.ndarray]
    path: Path

    @property
    def meta(self):
        return self.subjects, self.sessions, self.interfaces

    def ensemble(self, name: str) -> np.ndarray:
        """多种子时沿种子轴平均概率；单模型的 ``(N,K)`` 原样返回。"""
        p = self.probs[name]
        return p.mean(0) if p.ndim == 3 else p

    def summary(self, name: str) -> dict[str, object]:
        """对指定预测条件先执行种子概率平均，再计算分组指标。"""
        return summarize_predictions(self.y, self.ensemble(name), *self.meta, n_classes=self.probs[name].shape[-1])


def load_oof(path: str | Path) -> OOF:
    """读取 OOF NPZ，检查必需元数据，并识别二维/三维浮点概率数组。"""
    path = Path(path)
    with np.load(path, allow_pickle=False) as d:
        missing = [k for k in META_KEYS if k not in d]
        if missing:
            raise KeyError(f"{path} lacks {missing}")
        y = d["y"].astype(np.int64)
        meta = [d[k].astype(str) for k in META_KEYS[1:]]
        probs = {}
        for key in d.files:
            if key in META_KEYS:
                continue
            arr = d[key]
            if arr.ndim in (2, 3) and arr.shape[-2] == len(y) and np.issubdtype(arr.dtype, np.floating):
                probs[key.removesuffix("_probabilities")] = arr.astype(np.float32)
    return OOF(y, *meta, probs=probs, path=path)


def validate_probs(name: str, p: np.ndarray, n: int) -> None:
    """检查概率数组的样本数、有限性和每行类别概率和为 1。"""
    if p.shape[-2] != n:
        raise ValueError(f"{name}: {p.shape[-2]} rows, expected {n}")
    if not np.isfinite(p).all() or not np.allclose(p.sum(-1), 1, atol=1e-4):
        raise ValueError(f"{name}: probabilities are incomplete or do not sum to 1")


def save_run(out_dir: str | Path, *, y, subjects, sessions, interfaces, probs: Mapping[str, np.ndarray],
             protocol: Mapping[str, object], sources: Mapping[str, str | Path] = (), extra: Mapping[str, object] = ()
             ) -> dict[str, object]:
    """保存 ``oof.npz`` 与 ``report.json``，并记录协议、分数、环境和来源哈希。

    概率支持 ``(N,K)`` 或 ``(S,N,K)``；后者报告分数前会先平均种子概率。``sources``
    把可读标签映射到输入文件路径，报告中同时记录 SHA-256。该函数能保证文件结构
    和概率基本有效，但不能验证切分过程是否无泄漏，protocol 应描述这一点。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, np.int64)
    for name, p in probs.items():
        validate_probs(name, np.asarray(p), len(y))
    np.savez_compressed(out / "oof.npz", y=y, subjects=np.asarray(subjects).astype(str),
                        sessions=np.asarray(sessions).astype(str), interfaces=np.asarray(interfaces).astype(str),
                        **{k: np.asarray(v, np.float32) for k, v in probs.items()})
    results = {}
    for name, p in probs.items():
        p = np.asarray(p)
        s = summarize_predictions(y, p.mean(0) if p.ndim == 3 else p, subjects, sessions, interfaces, p.shape[-1])
        s.pop("cells")
        results[name] = s
    report = {
        "status": "complete",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": dict(protocol),
        "metric": "Equal mean balanced accuracy over participant x session x interface cells.",
        "results": results,
        "provenance": {label: {"path": str(p), "sha256": file_sha256(p)} for label, p in dict(sources).items()},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        **dict(extra),
    }
    try:
        import torch
        report["environment"]["torch"] = torch.__version__
    except ImportError:
        pass
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
