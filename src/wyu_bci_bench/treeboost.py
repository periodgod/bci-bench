"""XGBoost 与 EEG 频带、空间及 Riemann 切空间特征的组合基线。

切空间参考、Fisher 特征筛选和分类器均在当前训练折拟合。XGBoost 是可选依赖，
本模块在真正构造该模型时才导入它。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .features import filterbank_spatial_logvar_features, log_bandpower_features, select_top_features
from .riemann import covariance_matrices, riemannian_mean, tangent_feature_names, tangent_space


TREEBOOST_BANDS = ((4.0, 8.0), (8.0, 13.0), (13.0, 20.0), (20.0, 30.0), (30.0, 40.0))
TANGENT_BANDS = ((8.0, 30.0), (4.0, 40.0))


def _xgb_classifier(classes: int, seed: int, depth: int = 2, estimators: int = 80):
    """构造单线程、固定种子的多分类 XGBoost 概率模型。"""
    try:
        from xgboost import XGBClassifier  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("xgboost is required for --model xgb; install with `pip install xgboost`") from exc

    return XGBClassifier(
        objective="multi:softprob",
        num_class=int(classes),
        n_estimators=int(estimators),
        max_depth=int(depth),
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.75,
        min_child_weight=2.0,
        reg_alpha=0.05,
        reg_lambda=5.0,
        random_state=int(seed),
        n_jobs=1,
        tree_method="hist",
        eval_metric="mlogloss",
    )


@dataclass
class EEGFeatureXGBModel:
    """折内提取/选择 EEG 特征，再交给 XGBoost 分类的模型包装器。"""

    include_tangent: bool = True
    top_k: int | None = 96
    seed: int = 42
    max_depth: int = 2
    n_estimators: int = 80
    shrinkage: float = 0.1
    classifier_: object | None = None
    selected_indices_: np.ndarray | None = None
    feature_names_: list[str] | None = None
    fs_: int | None = None
    channel_names_: list[str] | None = None
    references_: list[np.ndarray] = field(default_factory=list)

    def _fixed_features(self, eeg: np.ndarray, fs: int, channel_names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
        """生成不依赖样本拟合状态的频带功率与空间方差特征。"""
        bandpower, bandpower_names = log_bandpower_features(
            eeg,
            fs,
            bands=TREEBOOST_BANDS,
            car=True,
            channel_names=channel_names,
        )
        spatial, spatial_names = filterbank_spatial_logvar_features(
            eeg,
            fs,
            channel_names,
            bands=TREEBOOST_BANDS,
            include_channel_power=True,
        )
        return np.concatenate([bandpower, spatial], axis=1).astype(np.float32), bandpower_names + spatial_names

    def _tangent_features(self, eeg: np.ndarray, fs: int, fit: bool) -> tuple[np.ndarray, list[str]]:
        """生成切空间特征；fit=True 时只用当前训练折估计参考协方差。"""
        if self.channel_names_ is None:
            raise RuntimeError("channel_names_ must be set before tangent features")
        parts: list[np.ndarray] = []
        names: list[str] = []
        if fit:
            self.references_ = []
        for band_idx, band in enumerate(TANGENT_BANDS):
            covs = covariance_matrices(eeg, fs, band=band, shrinkage=self.shrinkage)
            if fit:
                reference = riemannian_mean(covs)
                self.references_.append(reference)
            else:
                reference = self.references_[band_idx]
            parts.append(tangent_space(covs, reference))
            names.extend(tangent_feature_names(self.channel_names_, prefix=f"{band[0]:g}-{band[1]:g}Hz:ts:"))
        return np.concatenate(parts, axis=1).astype(np.float32), names

    def _features(self, eeg: np.ndarray, fs: int, fit: bool) -> tuple[np.ndarray, list[str]]:
        """拼接固定特征与可选切空间特征，并保持列名同步。"""
        if self.channel_names_ is None:
            raise RuntimeError("Model channel_names_ are not initialized")
        fixed, names = self._fixed_features(eeg, fs, self.channel_names_)
        if not self.include_tangent:
            return fixed, names
        tangent, tangent_names = self._tangent_features(eeg, fs, fit=fit)
        return np.concatenate([fixed, tangent], axis=1).astype(np.float32), names + tangent_names

    def fit(
        self,
        eeg: np.ndarray,
        y: np.ndarray,
        fs: int,
        channel_names: Sequence[str],
    ) -> "EEGFeatureXGBModel":
        """拟合特征变换、训练折选列和 XGBoost 分类器。"""
        labels = np.asarray(y, dtype=np.int64)
        self.fs_ = int(fs)
        self.channel_names_ = list(channel_names)
        features, names = self._features(eeg, self.fs_, fit=True)
        selected, indices = select_top_features(features, labels, self.top_k)
        classes = int(np.max(labels)) + 1
        self.classifier_ = _xgb_classifier(classes, self.seed, depth=self.max_depth, estimators=self.n_estimators)
        self.classifier_.fit(selected, labels)
        self.selected_indices_ = indices
        self.feature_names_ = [names[int(index)] for index in indices]
        return self

    def transform(self, eeg: np.ndarray) -> np.ndarray:
        """用保存的切空间参考和选列索引转换新 EEG。"""
        if self.fs_ is None or self.selected_indices_ is None:
            raise RuntimeError("Model is not fitted")
        features, _ = self._features(eeg, self.fs_, fit=False)
        return features[:, self.selected_indices_]

    def predict(self, eeg: np.ndarray) -> np.ndarray:
        """输出整数预测，兼容分类器返回标签或二维概率的情况。"""
        if self.classifier_ is None:
            raise RuntimeError("Model is not fitted")
        raw = np.asarray(self.classifier_.predict(self.transform(eeg)))
        if raw.ndim == 2:
            raw = np.argmax(raw, axis=1)
        return np.asarray(raw, dtype=np.int64)
