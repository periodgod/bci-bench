"""End-to-end spectral and spatial learning from cue-relative EEG windows."""

import torch
from torch import nn


class SpectralSpatialNet(nn.Module):
    """Learn band filters, spatial projections and changes between two half-windows."""

    def __init__(self, channels, samples=480, fs=120, components=8):
        super().__init__()
        bands = ((1.5, 1.2), (5.5, 1.6), (10.5, 2.1), (17., 3.), (25., 4.), (36., 6.))
        centers, widths = torch.tensor(bands).T
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)
        self.register_buffer("frequencies", torch.fft.rfftfreq(samples, 1 / fs))
        self.register_buffer("edges", torch.triu_indices(components, components, offset=1))
        self.frequency_adjust = nn.Parameter(torch.zeros(len(bands), 2))
        self.spatial = nn.Parameter(torch.randn(len(bands), components, len(channels)) * .025)
        self.common = nn.Parameter(torch.zeros(len(bands), components, 1))
        frontal = ("AF3", "AF4", "F1", "Fz", "F2")
        central = ("C3", "Cz", "C4", "CP3", "CP4")
        with torch.no_grad():
            for band, preferred in enumerate((frontal, frontal, central, central, central, frontal)):
                for component, channel in enumerate(preferred[:components]):
                    if channel in channels:
                        self.spatial[band, component, channels.index(channel)] += 1.
        width = len(bands) * (3 * components + self.edges.shape[1])
        self.encoder = nn.Sequential(nn.BatchNorm1d(width), nn.Linear(width, 48), nn.GELU(), nn.Dropout(.25))
        self.residual = nn.Sequential(nn.Linear(96, 32), nn.GELU(), nn.Dropout(.25), nn.Linear(32, 3))
        self.head = nn.Linear(48, 3)

    def forward(self, x):
        # Project first so the frequency transforms run on a small number of spatial components.
        x = x.float()
        mean = x.mean(dim=1, keepdim=True)
        spatial = torch.nn.functional.normalize(self.spatial, dim=-1)
        projected = torch.einsum("bct,skc->bskt", x - mean, spatial) + self.common * mean[:, None]
        center_delta, width_delta = self.frequency_adjust.tanh().T
        centers = self.centers + self.widths * center_delta
        widths = self.widths * (1 + .5 * width_delta)
        mask = (-.5 * ((self.frequencies - centers[:, None]) / widths[:, None]).square()).exp()
        signal = torch.fft.irfft(torch.fft.rfft(projected.float()) * mask[:, None], n=x.shape[-1])
        halves = signal.reshape(*signal.shape[:-1], 2, -1).permute(0, 3, 1, 2, 4)
        energy = halves.reshape(*halves.shape[:-1], 2, -1).square().mean(-1).clamp_min(1e-7).log().flatten(-2)
        centered = halves - halves.mean(-1, keepdim=True)
        unit = torch.nn.functional.normalize(centered, dim=-1, eps=1e-5)
        covariance = unit @ unit.transpose(-1, -2)
        correlations = covariance[..., self.edges[0], self.edges[1]]
        descriptors = torch.cat((energy, halves.mean(-1), correlations), dim=-1).flatten(2)
        early, late = self.encoder(descriptors.flatten(0, 1)).reshape(len(x), 2, -1).unbind(1)
        mean = (early + late) * .5
        return self.head(mean) + self.residual(torch.cat((mean, late - early), dim=1))
