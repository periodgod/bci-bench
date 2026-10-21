"""PROTEUS three-command networks (M1 / M2) and their training utilities.

Ported verbatim (module construction order included, so seeded initialisation
is unchanged) from ``proteus_update1/math_prior_multiscale_covnet.py`` and
``spatial_energy_ablation_loso.py``. Channel count, sampling rate and window
length are now arguments whose defaults equal the original constants.

- M1 = ``MathPriorMultiScaleCovNet(use_multiscale=True, use_covariance=False)``
  ("current_full_head_2bin")
- M2 = ``PowerFirstMathPriorNet(bins=4)`` ("power_first_4bin")
"""

# 输入约定：raw 是 (N,C,T) 微伏，features 是在当前训练折拟合得到的数学先验特征，
# y 是整数类别。外层测试折不能参与先验标准化器或 LogisticRegression 的拟合。

from __future__ import annotations

import random

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = nn = F = None

from .priors import PRIOR_BANDS


def seed_all(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 随机种子，控制常用随机来源。"""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def sampling_weights(y: np.ndarray, subjects: np.ndarray, mode: str = "multiscale_subject_class_balanced") -> np.ndarray:
    """为每个窗口生成加权抽样权重。

    ``multiscale_class_balanced`` 让每个类别总权重相同；默认模式进一步按
    ``参与者 × 类别`` 分层，使窗口多的组合不会仅凭样本量主导训练。返回值
    尚未归一化，WeightedRandomSampler 会按总和归一化。
    """
    if mode == "multiscale_class_balanced":
        counts = np.bincount(y, minlength=int(y.max()) + 1)
        return 1.0 / counts[y]
    if mode == "multiscale_subject_class_balanced":
        strata = [(s, int(c)) for s, c in zip(subjects, y)]
        counts = {key: strata.count(key) for key in set(strata)}
        return np.asarray([1.0 / counts[key] for key in strata], dtype=np.float64)
    raise ValueError(mode)


if nn is not None:

    class BandCovarianceProjection(nn.Module):
        """把各频带的通道相关矩阵投影到可学习的低秩表示。

        前向先做 FFT 频带筛选，再将每个频带拆成两个时间段，计算通道相关矩阵，
        仅取上三角以避免重复存储对称矩阵。各频带相关向量乘可学习投影矩阵后，
        输出给分类头；该分支是残差专家，不是原始 EEG 的替代物。
        """

        def __init__(self, n_components=4, n_channels=41, n_samples=480, fs=120.0, bands=PRIOR_BANDS[:6]):
            super().__init__()
            self.n_components = n_components
            self.n_channels = n_channels
            self.register_buffer("freqs", torch.fft.rfftfreq(n_samples, d=1 / fs))
            self.register_buffer("bands", torch.tensor(bands, dtype=torch.float32))
            n_upper = n_channels * (n_channels + 1) // 2
            self.register_buffer("upper_i", torch.triu_indices(n_channels, n_channels)[0])
            self.register_buffer("upper_j", torch.triu_indices(n_channels, n_channels)[1])
            self.projection = nn.Parameter(torch.empty(len(bands), n_upper, n_components))
            nn.init.normal_(self.projection, std=1 / np.sqrt(n_upper))
            self.half = n_samples // 2

        def forward(self, x):
            """输入 ``(B,C,T)``，输出各频带、各时间段的低秩协方差特征。"""
            x = x - x.mean(dim=1, keepdim=True)
            x = x / x.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-3)
            window = torch.hann_window(x.shape[-1], periodic=True, device=x.device, dtype=x.dtype)
            spectrum = torch.fft.rfft(x * window, dim=-1)
            freqs = self.freqs.to(dtype=x.dtype)
            pieces = []
            for bi, (low, high) in enumerate(self.bands):
                mask = ((freqs >= low) & (freqs < high)).to(x.dtype)
                filtered = torch.fft.irfft(spectrum * mask[None, None, :], n=x.shape[-1], dim=-1)
                segments = []
                for start in (0, self.half):
                    seg = filtered[..., start:start + self.half]
                    seg = seg - seg.mean(dim=-1, keepdim=True)
                    seg = seg / seg.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-5)
                    corr = torch.matmul(seg, seg.transpose(1, 2)) / float(self.half)
                    segments.append(corr[:, self.upper_i, self.upper_j] @ self.projection[bi])
                pieces.append(torch.cat(segments, dim=-1))
            return torch.cat(pieces, dim=-1)

    class MultiScaleTemporalEnergy(nn.Module):
        """多尺度时域卷积加空间混合，最后统计分段对数能量。

        输入 ``(B,C,T)`` 是 CAR 后的微伏信号；网络内部除以 100 缩小数值范围。
        三组卷积核宽度 15/31/63 捕捉不同时间尺度，再汇总空间滤波器在指定片段
        内的平均平方响应。这里的“能量”是离散信号平方均值，不是物理单位焦耳。
        """

        def __init__(self, filters_per_scale=6, spatial_filters=8, segment_starts=(0, 240),
                     segment_length=240, n_channels=41):
            super().__init__()
            self.segment_starts = tuple(segment_starts)
            self.segment_length = int(segment_length)
            self.temporal = nn.ModuleList([
                nn.Sequential(nn.Conv2d(1, filters_per_scale, kernel_size=(1, k), padding=(0, k // 2), bias=False),
                              nn.BatchNorm2d(filters_per_scale)) for k in (15, 31, 63)])
            self.spatial = nn.ModuleList([
                nn.Sequential(nn.Conv2d(filters_per_scale, spatial_filters, kernel_size=(n_channels, 1), bias=False),
                              nn.BatchNorm2d(spatial_filters)) for _ in (15, 31, 63)])
            self.dropout = nn.Dropout(0.2)
            self.out_features = 3 * spatial_filters * len(self.segment_starts)

        def forward(self, x):
            """将一批 EEG ``(B,C,T)`` 映射为拼接后的多尺度片段特征。"""
            x = (x - x.mean(dim=1, keepdim=True)) / 100.
            x = x.unsqueeze(1)
            outputs = []
            for temporal, spatial in zip(self.temporal, self.spatial):
                z = F.gelu(temporal(x))
                z = F.gelu(spatial(z)).squeeze(2)
                parts = [z[..., s:s + self.segment_length].square().mean(dim=-1).clamp_min(1e-7).log()
                         for s in self.segment_starts]
                outputs.append(torch.cat(parts, dim=-1))
            return self.dropout(torch.cat(outputs, dim=-1))

    class MathPriorMultiScaleCovNet(nn.Module):
        """M1：冻结的数学先验线性分类器，加可门控的时域/协方差残差专家。

        先验分类器只在训练折拟合并冻结；神经分支学习对该基线的补充。门控参数
        经 sigmoid 限制在 0 到 1，负值初始化意味着训练初期残差影响较小。
        两个 use_* 开关用于消融，不会移除先验分支。
        """

        def __init__(self, math_mean, math_scale, lr_coef, lr_intercept, use_multiscale=True,
                     use_covariance=True, segment_starts=(0, 240), segment_length=240, n_classes=3, n_channels=41):
            super().__init__()
            self.register_buffer("math_mean", torch.as_tensor(math_mean, dtype=torch.float32))
            self.register_buffer("math_scale", torch.as_tensor(math_scale, dtype=torch.float32))
            self.use_multiscale = use_multiscale
            self.use_covariance = use_covariance
            if use_multiscale:
                self.multiscale = MultiScaleTemporalEnergy(segment_starts=segment_starts,
                                                           segment_length=segment_length, n_channels=n_channels)
                self.temporal_head = nn.Sequential(nn.Linear(self.multiscale.out_features, 24), nn.GELU(),
                                                   nn.Dropout(0.25), nn.Linear(24, n_classes))
                self.temporal_gate = nn.Parameter(torch.tensor(-1.8))
            if use_covariance:
                self.covariance = BandCovarianceProjection(n_components=4, n_channels=n_channels)
                self.cov_head = nn.Sequential(nn.Linear(6 * 4 * 2, 16), nn.GELU(), nn.Dropout(0.25),
                                              nn.Linear(16, n_classes))
                self.cov_gate = nn.Parameter(torch.tensor(-2.0))
            # Built last so its fold-dependent width does not shift the neural branches' seeded init.
            self.math_prior = nn.Linear(len(math_mean), n_classes)
            with torch.no_grad():
                self.math_prior.weight.copy_(torch.as_tensor(lr_coef, dtype=torch.float32))
                self.math_prior.bias.copy_(torch.as_tensor(lr_intercept, dtype=torch.float32))
            self.math_prior.weight.requires_grad_(False)
            self.math_prior.bias.requires_grad_(False)

        def forward(self, raw, math_features):
            """返回未归一化 logits；训练用交叉熵，推理时再做 softmax。"""
            logits = self.math_prior((math_features - self.math_mean) / self.math_scale)
            if self.use_multiscale:
                logits = logits + torch.sigmoid(self.temporal_gate) * self.temporal_head(self.multiscale(raw))
            if self.use_covariance:
                logits = logits + torch.sigmoid(self.cov_gate) * self.cov_head(self.covariance(raw))
            return logits

    class PowerFirstTemporalEnergy(nn.Module):
        """M2 时域分支：先逐通道统计功率，再进行学习到的空间投影。

        与 M1 先空间混合再求能量的顺序不同，这里先得到每通道、每时间 bin 的
        对数能量，再通过可学习权重组合通道和卷积滤波器。长度不能整除 bins 时，
        时间尾部余数不会进入任何 bin，调用方需遵守实验输入长度合同。
        """

        def __init__(self, bins, n_channels=41):
            super().__init__()
            self.bins = bins
            self.temporal = nn.ModuleList([
                nn.Sequential(nn.Conv2d(1, 6, (1, k), padding=(0, k // 2), bias=False), nn.BatchNorm2d(6))
                for k in (15, 31, 63)])
            self.spatial_weight = nn.Parameter(torch.empty(3, 8, 6, n_channels))
            for weight in self.spatial_weight:
                nn.init.xavier_uniform_(weight)
            self.dropout = nn.Dropout(0.2)
            self.out_features = 3 * 8 * bins

        def forward(self, raw):
            """输入 ``(B,C,T)`` 微伏 EEG，输出按尺度和时间 bin 展平的特征。"""
            x = ((raw - raw.mean(dim=1, keepdim=True)) / 100.).unsqueeze(1)
            outputs = []
            width = x.shape[-1] // self.bins
            for scale, temporal in enumerate(self.temporal):
                z = F.gelu(temporal(x))
                bins = []
                for i in range(self.bins):
                    power = z[..., i * width:(i + 1) * width].square().mean(dim=-1).clamp_min(1e-7).log()
                    bins.append(F.gelu(torch.einsum("bfc,kfc->bk", power, self.spatial_weight[scale])))
                outputs.append(torch.stack(bins, dim=-1).flatten(1))
            return self.dropout(torch.cat(outputs, dim=-1))

    class PowerFirstMathPriorNet(nn.Module):
        """M2：冻结先验分类器与门控 power-first 时域残差相加。"""

        def __init__(self, mean, scale, coef, intercept, bins, n_classes=3, n_channels=41):
            super().__init__()
            self.register_buffer("math_mean", torch.as_tensor(mean, dtype=torch.float32))
            self.register_buffer("math_scale", torch.as_tensor(scale, dtype=torch.float32))
            self.multiscale = PowerFirstTemporalEnergy(bins, n_channels=n_channels)
            self.temporal_head = nn.Sequential(nn.Linear(self.multiscale.out_features, 24), nn.GELU(),
                                               nn.Dropout(0.25), nn.Linear(24, n_classes))
            self.temporal_gate = nn.Parameter(torch.tensor(-1.8))
            self.math_prior = nn.Linear(len(mean), n_classes)
            with torch.no_grad():
                self.math_prior.weight.copy_(torch.as_tensor(coef, dtype=torch.float32))
                self.math_prior.bias.copy_(torch.as_tensor(intercept, dtype=torch.float32))
            self.math_prior.weight.requires_grad_(False)
            self.math_prior.bias.requires_grad_(False)

        def forward(self, raw, math_features):
            """返回先验 logits 与时域残差 logits 的和。"""
            z = (math_features - self.math_mean) / self.math_scale
            return self.math_prior(z) + torch.sigmoid(self.temporal_gate) * self.temporal_head(self.multiscale(raw))


VARIANTS = ("current_full_head_2bin", "power_first_2bin", "power_first_4bin")


def fit_math_prior(train_features: np.ndarray, train_y: np.ndarray, C: float = 0.003):
    """只用传入训练折拟合标准化器和类别平衡 LogisticRegression。

    返回 ``(均值, 尺度, 系数, 截距)`` 四个 float32 数组，供网络冻结为先验权重。
    调用前必须先按 split 取训练行；若混入验证/测试数据，标准化统计和分类边界
    都会泄漏测试信息。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_features)
    prior = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=33)
    prior.fit(scaler.transform(train_features).astype(np.float32), train_y)
    return (scaler.mean_.astype(np.float32), scaler.scale_.clip(1e-3).astype(np.float32),
            prior.coef_.astype(np.float32), prior.intercept_.astype(np.float32))


def make_model(train_features: np.ndarray, train_y: np.ndarray, variant: str = "power_first_4bin"):
    """先拟合折内数学先验，再构造指定 M1/M2；返回的神经残差尚未训练。"""
    args = fit_math_prior(train_features, train_y)
    if variant == "current_full_head_2bin":
        return MathPriorMultiScaleCovNet(*args, use_multiscale=True, use_covariance=False,
                                         segment_starts=(0, 240), segment_length=240)
    if variant in ("power_first_2bin", "power_first_4bin"):
        return PowerFirstMathPriorNet(*args, bins=2 if variant.endswith("2bin") else 4)
    raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")


def train_model(model, raw, features, y, subjects, epochs, seed, device, batch_size=128, lr=4e-4,
                weight_decay=0.05, sampling_mode="multiscale_subject_class_balanced"):
    """按固定 epoch 训练 M1/M2，并按参与者×类别分层有放回抽样。

    ``raw``、``features``、``y`` 必须来自同一训练折且行顺序一致；模型创建时的
    先验也必须只在该折拟合。该函数没有早停，因此 ``epochs`` 应预先确定，不能
    根据外层测试分数挑选。返回切到 eval 模式的模型。
    """
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    seed_all(seed)
    model = model.to(device)
    weights = sampling_weights(y, subjects, sampling_mode)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(y), replacement=True,
                                    generator=torch.Generator().manual_seed(seed + 9000))
    loader = DataLoader(TensorDataset(torch.from_numpy(raw), torch.from_numpy(features),
                                      torch.from_numpy(y.astype(np.int64))), batch_size=batch_size, sampler=sampler)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        model.train()
        for bx, bf, by in loader:
            bx, bf, by = bx.to(device), bf.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(bx, bf), by).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
    return model.eval()


def predict_proba(model, raw, features, device, batch_size=256) -> np.ndarray:
    """分批前向并把 logits 转成 ``(N,K)`` softmax 概率，返回 CPU float32 数组。"""
    model.eval()
    out = []
    with torch.inference_mode():
        for s in range(0, len(raw), batch_size):
            logits = model(torch.from_numpy(raw[s:s + batch_size]).to(device),
                           torch.from_numpy(features[s:s + batch_size]).to(device))
            out.append(torch.softmax(logits, 1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)
