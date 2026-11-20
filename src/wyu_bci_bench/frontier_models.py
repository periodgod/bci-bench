"""Paper-grounded implementations of recently proposed EEG architectures.

These implementations expose the bench epoch convention ``(B, C, T)`` and
state explicitly where the paper does not provide executable code or complete
hyperparameter details. They are architecture baselines, not pretrained models.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DiagonalStateSpace(nn.Module):
    """Differentiable diagonal SSM recurrence for CPU/Windows portability.

    This implements a stable learned diagonal state recurrence. It is not the
    fused selective-scan kernel used by the official Mamba implementation.
    """

    def __init__(self, features: int, state_size: int = 8):
        super().__init__()
        self.features = features
        self.state_size = state_size
        self.input_projection = nn.Linear(features, features * 2)
        self.delta_projection = nn.Linear(features, features)
        self.write_projection = nn.Linear(features, features * state_size)
        self.read_projection = nn.Linear(features, features * state_size)
        self.log_decay = nn.Parameter(torch.zeros(features, state_size))
        self.output_projection = nn.Linear(features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.features:
            raise ValueError(f"expected (batch, sequence, {self.features}) input")
        values, gate = self.input_projection(x).chunk(2, dim=-1)
        delta = F.softplus(self.delta_projection(x)).unsqueeze(-1)
        write = torch.tanh(self.write_projection(x).view(*x.shape[:2], self.features, self.state_size))
        read = torch.tanh(self.read_projection(x).view(*x.shape[:2], self.features, self.state_size))
        state = x.new_zeros((x.shape[0], self.features, self.state_size))
        outputs = []
        for index in range(x.shape[1]):
            step = delta[:, index]
            state = state * torch.exp(-step * F.softplus(self.log_decay)) + step * write[:, index] * values[:, index].unsqueeze(-1)
            outputs.append((state * read[:, index]).sum(dim=-1))
        y = torch.stack(outputs, dim=1) * F.silu(gate)
        return self.output_projection(y)


class StateSpaceBlock(nn.Module):
    def __init__(self, features: int, state_size: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(features)
        self.mixer = DiagonalStateSpace(features, state_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mixer(self.norm(x)))


class STMambaNet(nn.Module):
    """STMambaNet-style six-scale embedding and parallel spatial/temporal SSMs.

    The paper specifies six temporal convolution branches, spatial convolution,
    variance/average pooling, and spatial/temporal Mamba encoders. Exact branch
    widths and several encoder settings are not fully specified in the paper;
    this implementation makes them configurable. Its portable diagonal SSM
    core is an explicit substitute for Mamba's optimized selective scan.
    """

    def __init__(
        self,
        channels: int,
        classes: int,
        *,
        branch_width: int = 4,
        embedding_dim: int = 32,
        kernel_sizes: tuple[int, ...] = (15, 31, 63, 127, 255, 383),
        pool_size: int = 8,
        state_size: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        if channels < 1 or classes < 2 or branch_width < 1 or embedding_dim < 1:
            raise ValueError("channels, classes, and widths must be positive (classes >= 2)")
        if len(kernel_sizes) != 6 or any(k < 1 or k % 2 == 0 for k in kernel_sizes):
            raise ValueError("kernel_sizes must contain six positive odd values")
        if pool_size < 1:
            raise ValueError("pool_size must be positive")
        self.channels = channels
        self.pool_size = pool_size
        self.temporal_convs = nn.ModuleList([
            nn.Conv2d(1, branch_width, kernel_size=(1, k), padding=(0, k // 2), bias=False)
            for k in kernel_sizes
        ])
        merged_width = branch_width * len(kernel_sizes)
        self.temporal_norm = nn.BatchNorm2d(merged_width)
        self.spatial_conv = nn.Conv2d(merged_width, embedding_dim, kernel_size=(channels, 1), bias=False)
        self.spatial_norm = nn.BatchNorm2d(embedding_dim)
        pooled_features = embedding_dim * 2
        self.spatial_encoder = StateSpaceBlock(pooled_features, state_size, dropout)
        self.temporal_encoder = StateSpaceBlock(pooled_features, state_size, dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(pooled_features * 2 * 8, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"expected input (batch, {self.channels}, samples)")
        x = x.unsqueeze(1)
        x = self.temporal_norm(torch.cat([conv(x) for conv in self.temporal_convs], dim=1))
        x = F.elu(self.spatial_norm(self.spatial_conv(x))).squeeze(2)
        x2 = x.square()
        mean = F.avg_pool1d(x, self.pool_size, stride=self.pool_size, ceil_mode=True)
        variance = (F.avg_pool1d(x2, self.pool_size, stride=self.pool_size, ceil_mode=True) - mean.square()).clamp_min(0)
        tokens = torch.cat([mean, variance], dim=1).transpose(1, 2)
        spatial = self.spatial_encoder(tokens)
        temporal = self.temporal_encoder(tokens)
        spatial = F.adaptive_avg_pool1d(spatial.transpose(1, 2), 8)
        temporal = F.adaptive_avg_pool1d(temporal.transpose(1, 2), 8)
        return self.head(torch.cat([spatial, temporal], dim=1).flatten(1))


def make_stmambanet(channels: int, classes: int, **kwargs) -> STMambaNet:
    """Construct the bench STMambaNet architecture baseline."""
    return STMambaNet(channels, classes, **kwargs)


class EEGSimpleConv(nn.Module):
    """EEG-SimpleConv temporal CNN with the author's optional rate conversion."""

    def __init__(self, channels: int, classes: int, *, features: int = 32,
                 conv_blocks: int = 4, kernel_size: int = 7, subjects: int | None = None,
                 sfreq: float | None = None, resampling: float | None = None):
        super().__init__()
        if min(channels, classes, features, conv_blocks, kernel_size) < 1 or classes < 2:
            raise ValueError("channels, classes, features, conv_blocks, and kernel_size must be positive")
        self.channels = channels
        self.resampler = None
        if (sfreq is None) != (resampling is None):
            raise ValueError("sfreq and resampling must be specified together")
        if sfreq is not None:
            if sfreq <= 0 or resampling <= 0:
                raise ValueError("sampling frequencies must be positive")
            try:
                from torchaudio.transforms import Resample
            except ImportError as exc:
                raise ImportError("EEGSimpleConv resampling needs torchaudio") from exc
            self.resampler = Resample(orig_freq=sfreq, new_freq=resampling)
        self.stem = nn.Sequential(
            nn.Conv1d(channels, features, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(features),
            nn.ReLU(),
        )
        blocks = []
        width = features
        for index in range(conv_blocks):
            next_width = width if index == 0 else int(1.414 * width)
            blocks.append(nn.Sequential(
                nn.Conv1d(width, next_width, kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(next_width),
                nn.MaxPool1d(2),
                nn.ReLU(),
                nn.Conv1d(next_width, next_width, kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(next_width),
                nn.ReLU(),
            ))
            width = next_width
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(width, classes)
        self.subject_classifier = nn.Linear(width, subjects) if subjects else None

    def forward(self, x: torch.Tensor):
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"expected input (batch, {self.channels}, samples)")
        if self.resampler is not None:
            x = self.resampler(x.contiguous())
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        features = x.mean(dim=-1)
        logits = self.classifier(features)
        if self.subject_classifier is not None:
            return logits, self.subject_classifier(features)
        return logits


def make_eeg_simpleconv(channels: int, classes: int, **kwargs) -> EEGSimpleConv:
    return EEGSimpleConv(channels, classes, **kwargs)


class _MFTTemporalBranch(nn.Module):
    def __init__(self, width: int, kernel_size: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, width, (1, kernel_size), padding=(0, kernel_size // 2), bias=False),
            nn.BatchNorm2d(width),
            nn.ELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class EEGMFTNet(nn.Module):
    """EEG-MFTNet: multiscale temporal convolutions plus Transformer fusion."""

    def __init__(self, channels: int, samples: int, classes: int, *, dropout: float = 0.3,
                 f1: int = 8, depth_multiplier: int = 2, f2: int = 16,
                 kernel_sizes: tuple[int, ...] = (5, 9, 13, 29, 61, 125),
                 transformer_heads: int = 2, transformer_ff: int = 32):
        super().__init__()
        if min(channels, samples, classes, f1, depth_multiplier, f2) < 1 or classes < 2:
            raise ValueError("channels, samples, classes and widths must be positive")
        if not kernel_sizes or any(k < 1 or k % 2 == 0 for k in kernel_sizes):
            raise ValueError("kernel_sizes must be positive odd values")
        transformer_heads = max(head for head in range(min(transformer_heads, channels), 0, -1)
                                if channels % head == 0)
        self.channels = channels
        self.samples = samples
        self.temporal_branches = nn.ModuleList([
            _MFTTemporalBranch(f1, kernel, 0.5) for kernel in kernel_sizes
        ])
        self.temporal_scales = nn.Parameter(torch.ones(len(kernel_sizes)))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=transformer_heads, dim_feedforward=transformer_ff,
            dropout=0.2, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.temporal_alpha = nn.Parameter(torch.tensor(0.8))
        self.transformer_beta = nn.Parameter(torch.tensor(0.2))
        merged_features = len(kernel_sizes) * f1 + 1
        self.merge_norm = nn.LayerNorm(merged_features)
        self.spatial = nn.Sequential(
            nn.Conv2d(merged_features, merged_features * depth_multiplier,
                      kernel_size=(channels, 1), groups=merged_features, bias=False),
            nn.BatchNorm2d(merged_features * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable_depth = nn.Conv2d(
            merged_features * depth_multiplier, merged_features * depth_multiplier,
            kernel_size=(1, 16), groups=merged_features * depth_multiplier,
            bias=False,
        )
        self.separable_point = nn.Conv2d(merged_features * depth_multiplier, f2, 1, bias=False)
        self.separable_norm = nn.BatchNorm2d(f2)
        self.separable_pool = nn.AvgPool2d((1, 8))
        self.separable_dropout = nn.Dropout(dropout)
        pooled_samples = (samples // 4) // 8
        if pooled_samples < 1:
            raise ValueError("samples must be at least 32 for the paper's pooling stages")
        self.classifier = nn.Linear(f2 * pooled_samples, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.channels, self.samples):
            raise ValueError(f"expected input (batch, {self.channels}, {self.samples})")
        signal = x.unsqueeze(1)
        temporal = torch.cat([
            branch(signal) * self.temporal_scales[index].clamp(0.0, 1.0)
            for index, branch in enumerate(self.temporal_branches)
        ], dim=1)
        transformer_input = x.transpose(1, 2)
        transformer = self.transformer(transformer_input) + transformer_input
        transformer = transformer.transpose(1, 2).unsqueeze(1)
        merged = torch.cat([
            temporal * self.temporal_alpha.clamp(0.0, 1.0),
            transformer * self.transformer_beta.clamp(0.0, 1.0),
        ], dim=1)
        merged = self.merge_norm(merged.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        spatial = self.spatial(merged)
        spatial = F.pad(spatial, (7, 8, 0, 0))
        y = F.elu(self.separable_norm(self.separable_point(self.separable_depth(spatial))))
        y = self.separable_dropout(self.separable_pool(y))
        return self.classifier(y.flatten(1))


def make_eeg_mftnet(channels: int, samples: int, classes: int, **kwargs) -> EEGMFTNet:
    return EEGMFTNet(channels, samples, classes, **kwargs)


class EEGMambaPatchEmbedding(nn.Module):
    """EEGMamba author's temporal/spectral patch embedding (MIT-licensed source)."""

    def __init__(self, patch_size: int = 200, model_dim: int = 200):
        super().__init__()
        if patch_size != 200 or model_dim != 200:
            raise ValueError("the migrated EEGMamba patch projection is fixed to 200-point patches and 200 features")
        self.patch_size = patch_size
        self.model_dim = model_dim
        self.temporal_projection = nn.Sequential(
            nn.Conv2d(1, 25, (1, 49), stride=(1, 25), padding=(0, 24), bias=False),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_projection = nn.Sequential(nn.Linear(101, model_dim, bias=False), nn.Dropout(0.1))
        self.position_conv = nn.Conv2d(model_dim, model_dim, (7, 7), padding=(3, 3), groups=model_dim, bias=False)
        self.mask_encoding = nn.Parameter(torch.zeros(patch_size), requires_grad=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4 or x.shape[-1] != self.patch_size:
            raise ValueError(f"expected EEGMamba patches shaped (batch, channels, segments, {self.patch_size})")
        batch, channels, segments, _ = x.shape
        if mask is not None:
            if mask.shape != x.shape[:3]:
                raise ValueError("mask must have shape (batch, channels, segments)")
            x = torch.where(mask.bool().unsqueeze(-1), self.mask_encoding.to(x), x)
        temporal_input = x.permute(0, 3, 1, 2).reshape(batch, channels * segments, self.patch_size).unsqueeze(1)
        temporal = self.temporal_projection(temporal_input)
        temporal = temporal.permute(0, 2, 1, 3).contiguous().reshape(batch, channels, segments, self.model_dim)
        spectrum = torch.fft.rfft(x, dim=-1, norm="forward").abs()
        spectral = self.spectral_projection(spectrum)
        patch = temporal + spectral
        position = self.position_conv(patch.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return patch + position


class EEGMambaClassifier(nn.Module):
    """Classifier wrapper around the official EEGMamba Mamba-2 encoder."""

    def __init__(self, classes: int, *, channels: int = 22, segments: int = 4, patch_size: int = 200):
        super().__init__()
        if classes < 2 or min(channels, segments) < 1:
            raise ValueError("classes must be >= 2 and channels/segments must be positive")
        try:
            from mamba_ssm.models.config_mamba import MambaConfig
            from mamba_ssm.models.mixer_seq_simple import MixerModel
        except ImportError as exc:  # pragma: no cover - platform-specific optional dependency
            raise ImportError("EEGMamba needs the optional Mamba-SSM CUDA package; install wyu-bci-bench[eegmamba] on a supported Linux/CUDA host") from exc
        self.channels = channels
        self.segments = segments
        self.patch_size = patch_size
        self.patch_embedding = EEGMambaPatchEmbedding(patch_size, 200)
        config = MambaConfig()
        config.ssm_cfg = {"layer": "Mamba2", "headdim": 50, "d_state": 64}
        self.encoder = MixerModel(
            d_model=config.d_model, n_layer=config.n_layer,
            d_intermediate=config.d_intermediate, ssm_cfg=config.ssm_cfg,
            attn_layer_idx=config.attn_layer_idx, attn_cfg=config.attn_cfg,
            rms_norm=config.rms_norm, initializer_cfg=None,
            fused_add_norm=config.fused_add_norm,
            residual_in_fp32=config.residual_in_fp32,
        )
        self.classifier = nn.Linear(200, classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (self.channels, self.segments, self.patch_size):
            raise ValueError(f"expected (batch, {self.channels}, {self.segments}, {self.patch_size})")
        patch = self.patch_embedding(x, mask)
        encoded = self.encoder(patch.reshape(x.shape[0], self.channels * self.segments, 200))
        features = encoded.reshape(x.shape[0], self.channels, self.segments, 200).mean(dim=(1, 2))
        return self.classifier(features)


def make_eegmamba(classes: int, **kwargs) -> EEGMambaClassifier:
    return EEGMambaClassifier(classes, **kwargs)


class EEGM2MambaBlock(nn.Module):
    """Residual Mamba-2 block used by the EEGM2 encoder/decoder."""

    def __init__(self, features: int, *, state_size: int = 64, conv_size: int = 4, expand: int = 2):
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except ImportError as exc:  # pragma: no cover - platform-specific optional dependency
            raise ImportError("EEGM2 needs the optional Mamba-SSM CUDA package; install wyu-bci-bench[eegmamba] on a supported Linux/CUDA host") from exc
        self.mamba = Mamba2(d_model=features, d_state=state_size, d_conv=conv_size, expand=expand)
        self.norm = nn.LayerNorm(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.mamba(x)) + x


class EEGM2(nn.Module):
    """EEGM2 self-supervised U-shaped encoder-decoder reconstruction model."""

    def __init__(self, channels: int, *, output_channels: int | None = None,
                 scale_factor: int = 1, state_size: int = 64, conv_size: int = 4,
                 expand: int = 2, state_block_factory=None):
        super().__init__()
        if channels < 1 or scale_factor < 1 or 64 // scale_factor < 1:
            raise ValueError("channels and scale_factor must be positive and scale_factor <= 64")
        output_channels = output_channels or channels
        if output_channels < 1:
            raise ValueError("output_channels must be positive")
        block_factory = state_block_factory or (
            lambda width: EEGM2MambaBlock(width, state_size=state_size, conv_size=conv_size, expand=expand)
        )
        c1, c2, c3 = 64 // scale_factor, 128 // scale_factor, 256 // scale_factor
        self.channels = channels
        self.input_branches = nn.ModuleList([
            nn.Conv1d(channels, c1, 1), nn.Conv1d(channels, c1, 3, padding=1),
            nn.Conv1d(channels, c1, 7, padding=3),
        ])
        self.input_fuse = nn.Conv1d(c1 * 3, c1, 1)
        self.encoder1 = nn.Sequential(nn.Linear(c1, c1), block_factory(c1))
        self.pool1 = nn.MaxPool1d(2)
        self.encoder2 = nn.Conv1d(c1, c2, 3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.encoder3 = nn.Conv1d(c2, c3, 3, padding=1)
        self.pool3 = nn.MaxPool1d(2)
        self.bottleneck = nn.Sequential(nn.Linear(c3, c3), block_factory(c3), nn.Linear(c3, c3))
        self.decoder3 = nn.Conv1d(c3 * 2, c3, 3, padding=1)
        self.decode_mamba3 = block_factory(c3)
        self.decoder2 = nn.Conv1d(c2 + c3, c2, 3, padding=1)
        self.decode_mamba2 = block_factory(c2)
        self.decoder1 = nn.Conv1d(c1 + c2, c1, 3, padding=1)
        self.output_embedding = nn.Conv1d(c1, output_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.channels or x.shape[-1] < 8:
            raise ValueError(f"expected input (batch, {self.channels}, samples>=8)")
        embedded = self.input_fuse(torch.cat([branch(x) for branch in self.input_branches], dim=1))
        x1 = self.encoder1(embedded.transpose(1, 2)).transpose(1, 2)
        x2 = self.encoder2(self.pool1(x1))
        x3 = self.encoder3(self.pool2(x2))
        latent = self.bottleneck(self.pool3(x3).transpose(1, 2)).transpose(1, 2)
        d3 = F.interpolate(latent, size=x3.shape[-1], mode="linear", align_corners=False)
        d3 = self.decoder3(torch.cat([x3, d3], dim=1))
        d3 = self.decode_mamba3(d3.transpose(1, 2)).transpose(1, 2)
        d2 = F.interpolate(d3, size=x2.shape[-1], mode="linear", align_corners=False)
        d2 = self.decoder2(torch.cat([x2, d2], dim=1))
        d2 = self.decode_mamba2(d2.transpose(1, 2)).transpose(1, 2)
        d1 = F.interpolate(d2, size=x1.shape[-1], mode="linear", align_corners=False)
        return self.output_embedding(self.decoder1(torch.cat([x1, d1], dim=1)))


def eegm2_temporal_spectral_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """EEGM2 temporal L1 plus full-spectrum magnitude reconstruction loss."""
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must have the same batched signal shape")
    temporal = F.l1_loss(prediction, target)
    predicted_spectrum = torch.fft.rfft(prediction, dim=-1, norm="forward").abs()
    target_spectrum = torch.fft.rfft(target, dim=-1, norm="forward").abs()
    return temporal + F.mse_loss(predicted_spectrum, target_spectrum)


def eegm2_representation_summary(features: torch.Tensor) -> torch.Tensor:
    """Nine temporal statistics used by EEGM2's custom linear-probe readout."""
    if features.ndim != 3:
        raise ValueError("features must have shape (batch, channels, time)")
    quantiles = torch.quantile(features, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=features.device), dim=-1)
    values = torch.cat([
        features.min(dim=-1).values.unsqueeze(-1),
        features.max(dim=-1).values.unsqueeze(-1),
        features.mean(dim=-1, keepdim=True),
        features.std(dim=-1, unbiased=False, keepdim=True),
        quantiles.movedim(0, -1),
    ], dim=-1)
    return values.flatten(1)
