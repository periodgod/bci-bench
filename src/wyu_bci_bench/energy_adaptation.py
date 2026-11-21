"""Fold-local regional gains inside a frozen power-first EEG classifier."""

import torch
from torch.nn import functional as F


def spatial_energy_terms(energy, weights, regions):
    """Project all channels and each selected region through the same trained weights."""
    whole = torch.einsum("bsfct,skfc->bskt", energy, weights)
    regional = torch.einsum("bsfct,skfc,rc->bsrkt", energy, weights, regions.to(energy))
    return whole, regional


def gain_logits(head, temporal_gate, prior, whole, regional, gains):
    """Apply regional gain corrections before the original projection activation."""
    correction = torch.einsum("bsrkt,srt->bskt", regional, gains)
    return prior + temporal_gate.sigmoid() * head(F.gelu(whole + correction).flatten(1))


def fit_energy_gains(head, temporal_gate, prior, whole, regional, labels, *, by_half, penalty=.1):
    """Fit bounded gain corrections on training rows by regularized cross-entropy."""
    steps = regional.shape[-1]
    if steps % 2:
        raise ValueError("energy time bins must divide evenly into two halves")
    slots = 2 if by_half else 1
    parameters = whole.new_zeros(regional.shape[1:3] + (slots,), requires_grad=True)
    optimizer = torch.optim.LBFGS([parameters], max_iter=200, line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-7, tolerance_change=1e-9)

    def gains():
        return .5 * parameters.tanh().repeat_interleave(steps // slots, dim=-1)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        adjustment = gains()
        logits = gain_logits(head, temporal_gate, prior, whole, regional, adjustment)
        loss = F.cross_entropy(logits, labels) + penalty * adjustment.square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return gains().detach()
