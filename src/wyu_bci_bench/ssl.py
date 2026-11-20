"""Self-supervised objectives (SimCLR / VICReg) on the multiscale temporal encoder.

Ported from ``representation_learning_20261019/ssl_experiment.py``. Augmentations
keep cue-relative time intact (no crop / shift / reversal): gain U(0.9, 1.1),
5 % channel dropout and 1 % RMS Gaussian noise. Task labels are never used.

Published method references for this module:
- TS2Vec (AAAI 2022, CCF-A; DOI 10.1609/aaai.v36i8.20881) motivates the
  hierarchical temporal contrastive objective implemented by ``TS2VecModel``,
  ``hierarchical_contrastive_loss`` and ``train_ts2vec``. This is an EEG-focused
  adaptation, not a reproduction of the authors' full architecture or recipe.
- EEG2Rep (KDD 2024, CCF-A; DOI 10.1145/3637528.3671600), LaBraM (ICLR 2024),
  and BIOT (NeurIPS 2023, CCF-A; DOI 10.52202/075280-3420) are documented
  research candidates, not implemented algorithms in this module.

See ``docs/representation_learning_papers.md`` for official publication links,
method summaries, author repositories, and the integration/reproduction boundaries.
"""

# 自监督阶段只读取信号和参与者分组，不使用任务标签 y。下游评估仍须在每个
# 外层训练折内训练/选择编码器，或明确说明预训练数据是否包含测试参与者。

from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = nn = F = None

if nn is not None:
    from .nets import MultiScaleTemporalEnergy

    class DilatedResidualBlock(nn.Module):
        """保留时间轴长度的扩张卷积残差块，用于逐采样点编码 EEG。"""

        def __init__(self, width: int, dilation: int):
            super().__init__()
            self.conv1 = nn.Conv1d(width, width, kernel_size=3, padding=dilation,
                                   dilation=dilation, bias=False)
            self.norm1 = nn.BatchNorm1d(width)
            self.conv2 = nn.Conv1d(width, width, kernel_size=3, padding=1, bias=False)
            self.norm2 = nn.BatchNorm1d(width)

        def forward(self, x):
            residual = F.gelu(self.norm1(self.conv1(x)))
            residual = self.norm2(self.conv2(residual))
            return F.gelu(x + residual)

    class TS2VecModel(nn.Module):
        """输出每个时间点的向量，供分层时间对比目标训练和下游池化使用。

        输入 ``(批次, 通道, 时间点)``，输出 ``(批次, 时间点, 表征维度)``。
        此轻量 EEG 编码器实现 TS2Vec 的逐时刻表征接口；论文的层级对比损失
        由 ``hierarchical_contrastive_loss`` 计算。它不是作者仓库模型的逐层复刻。
        """

        def __init__(self, n_channels=41, width=64, out_features=64, depth=4):
            super().__init__()
            self.input = nn.Conv1d(n_channels, width, kernel_size=1)
            self.blocks = nn.Sequential(*(DilatedResidualBlock(width, 2 ** i) for i in range(depth)))
            self.projector = nn.Conv1d(width, out_features, kernel_size=1)
            self.out_features = out_features

        def forward(self, x):
            return self.projector(self.blocks(self.input(x))).transpose(1, 2)

        @staticmethod
        def pool(z):
            """将时间点表示做均值池化，得到一个固定宽度的窗口向量。"""
            return z.mean(dim=1)

    class RepresentationModel(nn.Module):
        """时域编码器与自监督投影头。

        训练损失作用于 projector 输出；下游线性探针通常只取 ``encoder`` 输出并
        丢弃投影头。修改滤波器或通道配置会改变编码向量宽度。
        """

        def __init__(self, filters_per_scale=3, spatial_filters=4, proj_hidden=128, proj_out=64, n_channels=41):
            super().__init__()
            self.encoder = MultiScaleTemporalEnergy(filters_per_scale=filters_per_scale,
                                                    spatial_filters=spatial_filters, n_channels=n_channels)
            width = self.encoder.out_features
            self.projector = nn.Sequential(nn.Linear(width, proj_hidden), nn.BatchNorm1d(proj_hidden),
                                           nn.ReLU(), nn.Linear(proj_hidden, proj_out))

        def forward(self, x):
            """返回投影空间中的向量，不是下游任务的类别概率。"""
            return self.projector(self.encoder(x))

    class MaskedEEGAutoencoder(nn.Module):
        """Patch-wise masked autoencoder for unlabeled EEG pretraining.

        Input has shape ``(batch, channels, samples)``. Masked patch values are
        replaced by a learned token before encoding, so the decoder cannot copy
        the hidden signal directly from its input.
        """

        def __init__(self, n_channels=41, patch_size=25, hidden=96,
                     heads=4, depth=3, max_patches=512):
            super().__init__()
            if min(n_channels, patch_size, hidden, heads, depth, max_patches) < 1:
                raise ValueError("autoencoder dimensions must be positive")
            if hidden % heads:
                raise ValueError("hidden size must be divisible by attention heads")
            self.n_channels = n_channels
            self.patch_size = patch_size
            self.max_patches = max_patches
            self.patch_projection = nn.Linear(patch_size, hidden)
            self.channel_embedding = nn.Embedding(n_channels, hidden)
            self.position_embedding = nn.Embedding(max_patches, hidden)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
                batch_first=True, norm_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
            self.decoder = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, patch_size))
            nn.init.normal_(self.mask_token, std=0.02)

        def encode_tokens(self, x):
            """Unmasked tokens using exactly the encoder trained by reconstruction."""
            if x.ndim != 3 or x.shape[1] != self.n_channels or x.shape[-1] % self.patch_size:
                raise ValueError("invalid EEG shape for configured MAE encoder")
            patches = x.unfold(-1, self.patch_size, self.patch_size)
            batch, channels, count, _ = patches.shape
            if count < 2 or count > self.max_patches:
                raise ValueError("patch count must be in [2, max_patches]")
            tokens = self.patch_projection(patches)
            tokens = tokens + self.channel_embedding(torch.arange(channels, device=x.device))[None, :, None]
            tokens = tokens + self.position_embedding(torch.arange(count, device=x.device))[None, None]
            return self.encoder(tokens.reshape(batch, channels * count, -1)).reshape(batch, channels, count, -1)

        def forward(self, x, *, mask_ratio=0.5, generator=None, return_details=False):
            if x.ndim != 3 or x.shape[1] != self.n_channels:
                raise ValueError("x must have shape (batch, configured channels, samples)")
            if x.shape[-1] % self.patch_size:
                raise ValueError("sample count must be divisible by patch_size")
            if not 0.0 < mask_ratio < 1.0:
                raise ValueError("mask_ratio must be between zero and one")
            patches = x.unfold(-1, self.patch_size, self.patch_size)
            batch, channels, count, width = patches.shape
            if count < 2 or count > self.max_patches:
                raise ValueError("patch count must be in [2, max_patches]")
            tokens = self.patch_projection(patches)
            channel_ids = torch.arange(channels, device=x.device)
            position_ids = torch.arange(count, device=x.device)
            positions = self.channel_embedding(channel_ids)[None, :, None] + self.position_embedding(position_ids)[None, None]

            mask_count = min(count - 1, max(1, round(count * mask_ratio)))
            random_order = torch.rand((batch, channels, count), device=x.device,
                                      generator=generator).argsort(dim=-1)
            mask = torch.zeros((batch, channels, count), device=x.device, dtype=torch.bool)
            mask.scatter_(-1, random_order[..., :mask_count], True)
            tokens = torch.where(mask[..., None], self.mask_token.view(1, 1, 1, -1), tokens)
            # Mask signal content, retaining the electrode/time identity to reconstruct.
            tokens = tokens + positions
            encoded = self.encoder(tokens.reshape(batch, channels * count, -1))
            prediction = self.decoder(encoded).reshape(batch, channels, count, width)
            loss = masked_reconstruction_loss(prediction, patches, mask[..., None])
            if return_details:
                return {"loss": loss, "prediction": prediction, "target": patches, "mask": mask}
            return prediction


def augment(x, generator):
    """生成保持提示相对时间位置的视图：随机增益、通道丢弃和 RMS 比例噪声。"""
    gain = 0.9 + 0.2 * torch.rand((len(x), 1, 1), device=x.device, generator=generator)
    keep = (torch.rand((len(x), x.shape[1], 1), device=x.device, generator=generator) >= 0.05).to(x.dtype)
    rms = x.square().mean(dim=(1, 2), keepdim=True).sqrt()
    noise = torch.randn(x.shape, device=x.device, generator=generator) * rms * 0.01
    return x * gain * keep + noise


def simclr_loss(a, b, temperature: float = 0.2):
    """计算批内 NT-Xent：另一视图的同一窗口是正样本，其余窗口是负样本。"""
    z = F.normalize(torch.cat((a, b)), dim=1)
    n = len(a)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(-torch.inf)
    target = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
    return F.cross_entropy(logits, target)


def vicreg_loss(a, b, sim: float = 25., var: float = 25., cov: float = 1.):
    """组合 VICReg 的不变性、方差下限和维度去相关三项损失。

    方差项避免所有样本坍缩成相同向量；协方差项惩罚不同表示维度间的相关性。
    三个权重控制各项量级，不能只比较总 loss 数字来判断表示质量。
    """
    invariance = F.mse_loss(a, b)
    variance = F.relu(1 - torch.sqrt(a.var(0) + 1e-4)).mean() + F.relu(1 - torch.sqrt(b.var(0) + 1e-4)).mean()
    covariance = a.new_tensor(0.)
    for z in (a, b):
        centered = z - z.mean(0)
        matrix = centered.T @ centered / (len(z) - 1)
        off = ~torch.eye(z.shape[1], dtype=torch.bool, device=z.device)
        covariance = covariance + matrix[off].square().sum() / z.shape[1]
    return sim * invariance + var * variance + cov * covariance


def masked_reconstruction_loss(prediction, target, mask):
    """Mean squared error over masked samples only; ``mask`` broadcasts to EEG shape."""
    if torch is None:
        raise ImportError("PyTorch is required for masked reconstruction")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    try:
        selected = torch.broadcast_to(mask.to(device=prediction.device, dtype=torch.bool), prediction.shape)
    except (AttributeError, RuntimeError) as exc:
        raise ValueError("mask must be a torch tensor broadcastable to prediction") from exc
    if not torch.any(selected):
        raise ValueError("mask must select at least one element")
    return F.mse_loss(prediction[selected], target.to(device=prediction.device, dtype=prediction.dtype)[selected])


def predictive_forecast_loss(prediction, future_target, mask=None):
    """MSE for predicting a future signal or latent horizon without labels."""
    if torch is None:
        raise ImportError("PyTorch is required for predictive self-supervision")
    if prediction.shape != future_target.shape:
        raise ValueError("prediction and future_target must have identical shapes")
    target = future_target.to(device=prediction.device, dtype=prediction.dtype)
    if mask is None:
        return F.mse_loss(prediction, target)
    selected = torch.broadcast_to(mask.to(device=prediction.device, dtype=torch.bool), prediction.shape)
    if not torch.any(selected):
        raise ValueError("mask must select at least one prediction target")
    return F.mse_loss(prediction[selected], target[selected])


def cross_channel_contrastive_loss(embedding, temperature: float = 0.2):
    """Use same-window, different-channel features as multi-positive pairs.

    Input is ``(windows, channels, features)``. Other windows are negatives;
    neither task labels nor participant IDs are used.
    """
    if torch is None:
        raise ImportError("PyTorch is required for cross-channel contrastive learning")
    if embedding.ndim != 3 or embedding.shape[0] < 2 or embedding.shape[1] < 2:
        raise ValueError("embedding must have shape (at least 2 windows, at least 2 channels, features)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch, channels, _ = embedding.shape
    z = F.normalize(embedding.reshape(batch * channels, -1), dim=-1)
    similarities = z @ z.T / temperature
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    window_ids = torch.arange(batch, device=z.device).repeat_interleave(channels)
    positives = (window_ids[:, None] == window_ids[None, :]) & ~self_mask
    logits = similarities.masked_fill(self_mask, -torch.inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positives.sum(dim=1)
    return -(log_prob.masked_fill(~positives, 0).sum(dim=1) / positive_count).mean()


def train_masked_autoencoder(model, batches, *, epochs=1, mask_ratio=0.5,
                             learning_rate=1e-3, weight_decay=0.05,
                             device=None, gradient_clip=1.0, epoch_callback=None):
    """Train on re-iterable EEG tensor batches; task-label pairs are rejected.

    Participant splits must be applied to the pretraining corpus before calling
    this routine so held-out-subject access is governed by the chosen protocol.
    """
    if torch is None:
        raise ImportError("PyTorch is required for masked EEG pretraining")
    if epochs < 1 or learning_rate <= 0 or weight_decay < 0 or gradient_clip <= 0:
        raise ValueError("epochs and learning_rate/gradient_clip must be positive; weight_decay non-negative")
    if not 0 < mask_ratio < 1:
        raise ValueError("mask_ratio must be between zero and one")
    if iter(batches) is batches:
        raise ValueError("batches must be re-iterable for multiple epochs")
    run_device = torch.device(device or next(model.parameters()).device)
    model.to(run_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = []
    for _ in range(epochs):
        model.train()
        losses = []
        for batch in batches:
            if not torch.is_tensor(batch):
                raise TypeError("each batch must be an EEG tensor; task labels are not accepted")
            signal = batch.to(run_device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = model(signal, mask_ratio=mask_ratio, return_details=True)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite masked reconstruction loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
        if not losses:
            raise ValueError("each epoch must contain at least one EEG batch")
        history.append(float(np.mean(losses)))
        if epoch_callback is not None:
            epoch_callback(len(history), model.eval())
    return history


def _pair_contrastive_loss(a, b, temperature: float):
    """每个独立组内对比第二维；组与组之间不互当负样本。"""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    z = torch.cat((a, b), dim=1)
    logits = torch.bmm(z, z.transpose(1, 2)) / temperature
    n = a.shape[1]
    diagonal = torch.eye(2 * n, dtype=torch.bool, device=z.device).unsqueeze(0)
    logits = logits.masked_fill(diagonal, -torch.inf)
    target = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
    return F.cross_entropy(logits.reshape(-1, 2 * n), target.repeat(a.shape[0]))


def hierarchical_contrastive_loss(a, b, temperature: float = 0.2, alpha: float = 0.5):
    """TS2Vec 风格的分层时间对比目标，输入形状为 ``(B, T, D)``。

    每个层级同时学习两种关系：同一窗口不同增强视图的同一时间点应接近
    （instance contrast），同一窗口内不同时间点应能区分（temporal contrast）。
    随后沿时间轴用 max pooling 建立更粗的时间尺度，并累计各尺度损失。这个
    目标不需要类别标签。奇数长度的最末时间点会在下采样时被丢弃，与常见
    2 倍层级池化一致；调用者应确保两个视图时间轴对齐且长度至少为 1。
    """
    if a.ndim != 3 or a.shape != b.shape:
        raise ValueError("a and b must have the same (batch, time, feature) shape")
    if a.shape[1] < 1:
        raise ValueError("time dimension must be non-empty")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    losses = []
    while True:
        temporal = (_pair_contrastive_loss(a, b, temperature)
                    if a.shape[1] > 1 else a.sum() * 0.0)
        instance = (_pair_contrastive_loss(a.transpose(0, 1), b.transpose(0, 1), temperature)
                    if a.shape[0] > 1 else a.sum() * 0.0)
        losses.append(alpha * instance + (1.0 - alpha) * temporal)
        if a.shape[1] < 2:
            break
        a = F.max_pool1d(a.transpose(1, 2), kernel_size=2).transpose(1, 2)
        b = F.max_pool1d(b.transpose(1, 2), kernel_size=2).transpose(1, 2)
    return torch.stack(losses).mean()


def cross_subject_supcon(embedding, labels, subjects, temperature: float):
    """Supervised contrastive loss whose positives are same-class windows from *other* participants.

    ``embedding`` 通常先做 L2 归一化；``labels`` 与 ``subjects`` 是同一批次逐行
    对齐的张量。正样本必须来自不同参与者，避免把同一人的相似窗口当作跨参与者
    不变性证据。没有跨参与者正样本的行不参与平均；若整批都没有，返回可反传的
    零损失。
    """
    similarity = embedding @ embedding.T / temperature
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = (labels[:, None] == labels[None, :]) & (subjects[:, None] != subjects[None, :]) & ~diagonal
    logits = similarity.masked_fill(diagonal, torch.finfo(similarity.dtype).min)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    count = positives.sum(dim=1)
    rows = count > 0
    if not rows.any():
        return embedding.sum() * 0.0
    row_loss = -(log_prob.masked_fill(~positives, 0.0).sum(dim=1) / count.clamp_min(1))
    return row_loss[rows].mean()


LOSSES = {"simclr": simclr_loss, "vicreg": vicreg_loss}


@torch.inference_mode() if torch is not None else (lambda f: f)
def encode(model, raw, batch: int = 128) -> np.ndarray:
    """只提取 encoder 表示（不含 projector），分批返回 NumPy 特征矩阵。"""
    model.eval()
    return torch.cat([model.encoder(raw[i:i + batch]).cpu() for i in range(0, len(raw), batch)]).numpy()


def train_ssl(model, raw, people, mode: str, seed: int, epochs: int = 15, batch: int = 128,
              lr: float = 4e-4, weight_decay: float = 0.05) -> list[float]:
    """按参与者均衡抽样训练自监督模型，并返回每轮平均训练损失。

    ``raw`` 必须已放到目标设备；``people`` 长度需与窗口数一致。每轮按参与者
    逆样本数抽样，再对窗口生成两种增强视图。尾部不足一个 batch 的样本会被舍弃，
    以保持批内对比损失的计算形状一致。
    """
    loss_fn = LOSSES[mode]
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    _, ids, counts = np.unique(people, return_inverse=True, return_counts=True)
    weights = torch.as_tensor(1. / counts[ids], dtype=torch.float64)
    sampling_rng = torch.Generator().manual_seed(seed + 9000)
    augmentation_rng = torch.Generator(device=raw.device).manual_seed(seed + 18000)
    trace = []
    for epoch in range(epochs):
        order = torch.multinomial(weights, len(raw), replacement=True, generator=sampling_rng)
        losses = []
        for offset in range(0, len(order) - batch + 1, batch):
            x = raw[order[offset:offset + batch].to(raw.device)]
            a, b = augment(x, augmentation_rng), augment(x, augmentation_rng)
            optimizer.zero_grad(set_to_none=True)
            za, zb = model(torch.cat((a, b))).chunk(2)
            loss = loss_fn(za, zb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {mode} loss at epoch {epoch}")
            loss.backward()
            if not torch.isfinite(nn.utils.clip_grad_norm_(model.parameters(), 1.0)):
                raise RuntimeError("non-finite gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        trace.append(float(np.mean(losses)))
    return trace


def train_ts2vec(model, raw, people, seed: int, epochs: int = 15, batch: int = 128,
                 lr: float = 4e-4, weight_decay: float = 0.05,
                 temperature: float = 0.2, alpha: float = 0.5) -> list[float]:
    """以被试均衡抽样训练逐时间点编码器，返回各轮层级对比损失。"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    _, ids, counts = np.unique(people, return_inverse=True, return_counts=True)
    weights = torch.as_tensor(1. / counts[ids], dtype=torch.float64)
    sampling_rng = torch.Generator().manual_seed(seed + 9000)
    augmentation_rng = torch.Generator(device=raw.device).manual_seed(seed + 18000)
    trace = []
    for epoch in range(epochs):
        order = torch.multinomial(weights, len(raw), replacement=True, generator=sampling_rng)
        losses = []
        for offset in range(0, len(order) - batch + 1, batch):
            x = raw[order[offset:offset + batch].to(raw.device)]
            a, b = augment(x, augmentation_rng), augment(x, augmentation_rng)
            optimizer.zero_grad(set_to_none=True)
            za, zb = model(torch.cat((a, b))).chunk(2)
            loss = hierarchical_contrastive_loss(za, zb, temperature, alpha)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite ts2vec loss at epoch {epoch}")
            loss.backward()
            if not torch.isfinite(nn.utils.clip_grad_norm_(model.parameters(), 1.0)):
                raise RuntimeError("non-finite gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        trace.append(float(np.mean(losses)))
    return trace
