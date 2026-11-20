"""Compact convolutional task-phase readout and optional slow-feature head.

This is a controlled experiment architecture, not a published-model reproduction.
Odd padded convolutions preserve temporal indices. Pooling maps input samples
4*j ... 4*j+3 to output j; the half-window split is therefore exact for T % 8=0.
"""
import torch
from torch import nn


class PhaseSlowNet(nn.Module):
    def __init__(self, channels=41, classes=3, width=24, phase=False, slow_features=0):
        super().__init__()
        self.phase = phase
        self.features = nn.Sequential(
            nn.Conv2d(1, width, (1, 31), padding=(0, 15), bias=False),
            nn.BatchNorm2d(width),
            nn.Conv2d(width, width, (channels, 1), groups=width, bias=False),
            nn.BatchNorm2d(width), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.35),
            nn.Conv2d(width, width, (1, 15), padding=(0, 7), groups=width, bias=False),
            nn.Conv2d(width, width, 1, bias=False), nn.BatchNorm2d(width), nn.ELU(),
        )
        self.slow = nn.Sequential(nn.Linear(slow_features, 32), nn.ELU(), nn.Dropout(0.35)) if slow_features else None
        self.classifier = nn.Linear(width * (2 if phase else 1) + (32 if slow_features else 0), classes)

    def forward(self, x, slow=None):
        if x.ndim != 3 or x.shape[-1] % 8:
            raise ValueError("input must be N,C,T with T divisible by eight")
        z = self.features(x[:, None]).squeeze(2)
        if self.phase:
            middle = z.shape[-1] // 2
            z = torch.cat((z[..., :middle].mean(-1), z[..., middle:].mean(-1)), dim=1)
        else:
            z = z.mean(-1)
        if self.slow is not None:
            if slow is None:
                raise ValueError("this model requires slow features")
            z = torch.cat((z, self.slow(slow)), dim=1)
        return self.classifier(z)
