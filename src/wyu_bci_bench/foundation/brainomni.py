"""BrainOmni official RVQ/encoder path with frozen-token caching and explicit tuning.

Sources and the evaluation-dropout correction are in _third_party/brainomni/UPSTREAM.json.
All EEG arrays entering prepare_input are microvolts at 256 Hz. Dataset-specific
filtering and resampling must happen before this function.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .finetuning import attach_peft_adapter, configure_trainable_parameters


def prepare_input(x_uv, *, fs=256):
    """Official per-window EEG CAR and grouped scale, including the VOLT epsilon.

    The official std denominator is std(volts)+1e-5; equivalently std(uV)+10.
    Using std(uV)+1e-5 would silently change pretrained inputs.
    Accept complete windows of at least one second. Normalize observed samples
    here; the official tokenizer pads short windows to its patch length later.
    """
    x = np.asarray(x_uv, dtype=np.float64)
    if fs != 256 or x.ndim != 3 or x.shape[1] < 2 or x.shape[-1] < 256:
        raise ValueError('expected (N,C,T>=256) EEG microvolts at 256 Hz, C>=2')
    if not np.isfinite(x).all():
        raise ValueError('EEG must be finite')
    x = x - x.mean(axis=1, keepdims=True)
    return (x / (x.std(axis=(1, 2), keepdims=True) + 10.0)).astype(np.float32)


def normalize_positions(xyz):
    """Official EEG coordinate centering and sqrt(3*mean(radius^2)) scaling."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2 or not np.isfinite(xyz).all():
        raise ValueError('expected finite (C,3) EEG coordinates')
    xyz = xyz - xyz.mean(axis=0, keepdims=True)
    scale = np.sqrt(3 * np.mean(np.sum(xyz ** 2, axis=1)))
    if scale <= 0:
        raise ValueError('electrode coordinates have no spatial extent')
    return np.concatenate([xyz / scale, np.zeros_like(xyz)], axis=1).astype(np.float32)


def eeg_geometry(channel_names):
    """Standard-1005 head coordinates; these are template, not measured positions."""
    import mne
    names = list(channel_names)
    if len(set(names)) != len(names):
        raise ValueError('channel_names must be unique')
    info = mne.create_info(names, 256, ch_types='eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1005'), on_missing='raise')
    return normalize_positions([ch['loc'][:3] for ch in info['chs']]), np.zeros(len(names), np.int64)


def load_encoder(weights_dir):
    """Strictly load the complete official checkpoint, including its tokenizer."""
    from .._third_party.brainomni.brainomni.model import BrainOmni
    directory = Path(weights_dir)
    config = json.loads((directory / 'model_cfg.json').read_text(encoding='utf-8'))
    model = BrainOmni(**config)
    state = torch.load(directory / 'BrainOmni.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    configure_trainable_parameters(model, 'frozen')
    return model.eval()


@torch.no_grad()
def cache_tokens(encoder, x, pos, sensor_type, *, batch=4, device='cuda'):
    """Cache only fixed tokenizer outputs; no learned normalization or labels used."""
    encoder = encoder.to(device).eval()
    positions = torch.as_tensor(pos, dtype=torch.float32, device=device)
    types = torch.as_tensor(sensor_type, dtype=torch.long, device=device)
    output = []
    for start in range(0, len(x), batch):
        windows = torch.as_tensor(x[start:start + batch], dtype=torch.float32, device=device)
        tokens, _ = encoder.tokenizer.tokenize(windows,
            positions.unsqueeze(0).expand(len(windows), -1, -1),
            types.unsqueeze(0).expand(len(windows), -1), encoder.overlap_ratio)
        output.append(tokens.cpu())
    return torch.cat(output)


class TokenBackbone(nn.Module):
    """Exactly encode() after tokenizer: neuro embedding -> projection -> blocks[:-1]."""
    def __init__(self, encoder):
        super().__init__()
        self.register_buffer('neuro', encoder.tokenizer.encoder.neuros.detach().clone()[None, :, None])
        self.projection = encoder.projection
        self.blocks = nn.ModuleList(list(encoder.blocks[:-1]))

    def forward(self, tokens):
        x = self.projection(tokens + self.neuro.to(tokens))
        for block in self.blocks:
            x = block(x)
        return F.normalize(x, p=2.0, dim=-1, eps=1e-6)


class BrainOmniClassifier(nn.Module):
    """Official mean-time/flatten/MLP head; tokenizer never receives task gradients.

    full updates all blocks actually used by encode(), not the unused final block
    or pretraining head. Each instance must receive a freshly loaded encoder.
    """
    def __init__(self, encoder, *, n_classes=2, strategy='frozen', rank=8, total_steps=None):
        super().__init__()
        if strategy not in {'frozen', 'last_block', 'full', 'lora', 'dora', 'pissa', 'adalora'}:
            raise ValueError('unknown BrainOmni strategy')
        self.strategy = strategy
        self.backbone = TokenBackbone(encoder)
        n_sources = len(encoder.tokenizer.encoder.neuros)
        self.class_head = nn.Sequential(nn.Dropout(0.1),
            nn.Linear(n_sources * encoder.lm_dim, encoder.lm_dim),
            nn.SELU(), nn.Linear(encoder.lm_dim, n_classes))
        configure_trainable_parameters(self.backbone, 'frozen')
        if strategy == 'full':
            configure_trainable_parameters(self.backbone, 'full')
        elif strategy == 'last_block':
            configure_trainable_parameters(self.backbone, 'selected',
                module_names=[f'blocks.{len(self.backbone.blocks)-1}'])
        elif strategy in {'lora', 'dora', 'pissa', 'adalora'}:
            targets = [name for name, module in self.backbone.named_modules()
                       if isinstance(module, nn.Linear) and name.endswith(('.qkv', '.proj'))]
            self.backbone = attach_peft_adapter(self.backbone, method=strategy,
                target_modules=targets, rank=rank, alpha=2 * rank, total_steps=total_steps)

    def train(self, mode=True):
        super().train(mode)
        if self.strategy == 'frozen':
            self.backbone.eval()
        return self

    def forward(self, tokens):
        x = self.backbone(tokens).mean(2).flatten(1)
        return self.class_head(x)


def audit_task_gradients(model, tokens, labels):
    """Fail if an advertised trainable parameter is disconnected from task loss."""
    model.train()
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(tokens), labels).backward()
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    nonfinite = [name for name, p in model.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
    if missing or nonfinite:
        raise RuntimeError(f'task gradient audit failed: missing={missing}, nonfinite={nonfinite}')
    result = dict(trainable_tensors=sum(p.requires_grad for p in model.parameters()),
                  connected=True, finite=True,
                  nonzero_gradient_tensors=sum(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                               for p in model.parameters()))
    model.zero_grad(set_to_none=True)
    return result
