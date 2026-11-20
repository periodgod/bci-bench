"""Explicit labelled-calibration protocols; never mix these with zero-shot OOF."""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from .finetuning import configure_trainable_parameters
from ..splits import ordered_kshot_mask


def personal_partition(subjects, sessions, labels, subject, *, k=5, gap=1):
    """First k examples per class calibrate; next gap examples excluded; score rest.

    Input order must be recording order. For EEGMAT this is offline within-recording
    calibration: both rest/arithmetic labels are given, not prospective deployment.
    """
    if gap < 0:
        raise ValueError('gap must be nonnegative')
    subjects, sessions, labels = map(np.asarray, (subjects, sessions, labels))
    local = subjects == subject
    if not local.any():
        raise ValueError('unknown calibration subject')
    for c in np.unique(labels):
        if np.sum(local & (labels == c)) <= k + gap:
            raise ValueError('not enough rows for calibration, gap and evaluation')
    calibration = ordered_kshot_mask(subjects, sessions, labels, subject, k,
        train_sessions=np.unique(sessions[local]), n_classes=int(labels.max())+1)
    prefix = ordered_kshot_mask(subjects, sessions, labels, subject, k+gap,
        train_sessions=np.unique(sessions[local]), n_classes=int(labels.max())+1)
    return calibration, prefix & ~calibration, local & ~prefix


class PersonalFiLM(nn.Module):
    """Zero-initialized feature-wise affine correction plus personal head update."""
    def __init__(self, population):
        super().__init__()
        self.population = population
        configure_trainable_parameters(population, 'frozen')
        for p in population.class_head.parameters():
            p.requires_grad_(True)
        dim = population.class_head[1].out_features
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    @property
    def class_head(self):
        return self.population.class_head

    def train(self, mode=True):
        super().train(mode)
        self.population.backbone.eval()
        return self

    def forward(self, tokens):
        x = self.population.backbone(tokens)
        x = x * (1+self.gamma) + self.beta
        return self.class_head(x.mean(2).flatten(1))


def add_personal_lora(population, *, rank=8):
    """Freeze a trained global LoRA, stack a zero-initialized personal delta.

    Both branches remain active; only the personal branch and head receive updates.
    This is a transfer of the stacked-adapter idea, not the original joint
    subject-branch training protocol.
    """
    from peft import LoraConfig
    targets = population.backbone.peft_config['default'].target_modules
    config = LoraConfig(r=rank, lora_alpha=2*rank, lora_dropout=0., bias='none', target_modules=targets)
    population.backbone.add_adapter('personal', config)
    # PeftModel.set_adapter accepts only a single name in PEFT0.21; LoraModel
    # supports additive activation of both branches. Freeze explicitly afterwards.
    population.backbone.base_model.set_adapter(['default','personal'])
    configure_trainable_parameters(population, 'frozen')
    for name,p in population.named_parameters():
        if '.personal.' in name or name.startswith('class_head.'):
            p.requires_grad_(True)
    return population


def personal_state(model):
    """Save only subject-fitted tensors; the common population checkpoint is fixed."""
    return {name: p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}


def load_personal_state(model, state):
    params = dict(model.named_parameters())
    expected = {name for name,p in params.items() if p.requires_grad}
    if set(state) != expected:
        raise ValueError('personal checkpoint has different trainable scope')
    with torch.no_grad():
        for name,value in state.items():
            params[name].copy_(value)
