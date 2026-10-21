"""新 PROTEUS 实验的最小端到端模板。

建议按 ``main`` 的顺序阅读：加载带分组键的缓存、提取特征、运行外层 LOSO、保存
OOF 和协议报告。实验特有逻辑集中在 ``featurize``；若编码器会学习参数，必须在
每个外层训练折内拟合，不能像下面的固定数学特征示例一样对全体 X 一次处理。

    python examples/proteus_loso_template.py --workspace D:/脑电大模型 --out outputs/math_prior_probe

The default featurizer is the fixed math prior with a C=0.003 linear probe,
which reproduces the 2026-10-19 baseline (cell BA 0.5113).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wyu_bci_bench.grouped import load_proteus_windows
from wyu_bci_bench.metrics import paired_subject_bootstrap, summarize_predictions
from wyu_bci_bench.priors import math_prior_features
from wyu_bci_bench.readout import loso_probe
from wyu_bci_bench.registry import save_run


def featurize(X: np.ndarray) -> np.ndarray:
    """固定特征变换示例；有可学习状态的编码器必须改为逐训练折拟合。"""
    return math_prior_features(X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, default=Path("D:/脑电大模型"))
    ap.add_argument("--out", type=Path, default=Path("outputs/math_prior_probe"))
    args = ap.parse_args()
    cache = args.workspace / "challenge2026_track02/proteus_update1/cache/proteus_windows_v1.npz"
    data = load_proteus_windows(cache)

    features = featurize(data.X)
    probs = loso_probe(features, data.y, data.subjects, C=0.003)

    report = save_run(args.out, y=data.y, subjects=data.subjects, sessions=data.sessions,
                      interfaces=data.interfaces, probs={"math_prior": probs},
                      protocol={"split": "10-subject outer LOSO", "readout": "StandardScaler + balanced LR C=0.003",
                                "features": "math prior 1394-D", "evidence": "development only"},
                      sources={"cache": cache, "script": Path(__file__)})
    print("cell BA", round(report["results"]["math_prior"]["cell_ba"], 4))

    # Paired comparison against any other OOF condition on the same rows:
    baseline = summarize_predictions(data.y, np.full_like(probs, 1 / 3) + np.eye(3)[0] * 1e-6, *data.meta)
    diff = paired_subject_bootstrap(summarize_predictions(data.y, probs, *data.meta), baseline)
    print("vs constant class-0:", round(diff["cell_difference"], 4), diff["cell_cluster_bootstrap_95ci"])


if __name__ == "__main__":
    main()
