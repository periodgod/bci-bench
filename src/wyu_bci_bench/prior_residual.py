"""Bounded class-specific and full-montage convolutional residuals for binary EEG logits."""

import torch
from torch import nn


class TemporalResidualFeatures(nn.Module):
    """Learn spatial mixtures and temporal filters, retaining two time segments."""

    def __init__(self, channels, width, kernel, dilation, pool, energy=False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(channels, width, 1, bias=False),
            nn.Conv1d(width, width, kernel, padding=kernel // 2, groups=width, bias=False),
            nn.BatchNorm1d(width),
            nn.AvgPool1d(pool),
        )
        self.residual = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation, groups=width, bias=False),
            nn.Conv1d(width, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.energy = energy

    def forward(self, x):
        hidden = self.stem(x)
        hidden = hidden + self.residual(hidden)
        values = hidden.square() if self.energy else nn.functional.gelu(hidden)
        halves = nn.functional.adaptive_avg_pool1d(values, 2)
        if self.energy:
            halves = torch.log(halves.clamp_min(1e-6))
        first, last = halves.unbind(-1)
        return torch.cat((first, last, last - first), dim=1)


class PriorResidualEEG(nn.Module):
    """Add two bounded corrections to an externally fitted, frozen binary logit."""

    def __init__(self, task, target_channels, full_channels, width=12, limit=0.75):
        super().__init__()
        kernel, dilation, pool, energy = {
            "mi": (17, 2, 4, True),
            "sub": (15, 3, 2, True),
            "word": (9, 5, 2, False),
        }[task]
        if limit <= 0:
            raise ValueError("logit correction limit must be positive")
        self.target = TemporalResidualFeatures(target_channels, width, kernel, dilation, pool, energy)
        self.full = TemporalResidualFeatures(full_channels, width, 9, 4, 4)
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Dropout(0.25), nn.Linear(3 * width, 1)) for _ in range(2)
        )
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.limit = limit

    def corrections(self, target_eeg, full_eeg):
        target = self.heads[0](self.target(target_eeg)).squeeze(1)
        full = self.heads[1](self.full(full_eeg)).squeeze(1)
        return self.limit * target.tanh(), self.limit * full.tanh()

    def forward(self, target_eeg, full_eeg, prior_logit):
        target, full = self.corrections(target_eeg, full_eeg)
        positive = prior_logit + target + full
        return torch.stack((torch.zeros_like(positive), positive), dim=1)

