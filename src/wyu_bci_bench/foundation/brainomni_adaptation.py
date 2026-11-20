"""Fold-local BrainOmni adaptation objectives and audited PEFT scheduling."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def adalora_regularization(peft_backbone):
    """PEFT AdaLoraModel's orthogonal penalty for a tensor-returning encoder.

    PEFT only adds this automatically when forward returns an object with .loss.
    BrainOmni returns a tensor, so its task objective must add it explicitly.
    """
    config = peft_backbone.peft_config['default']
    terms = []
    for name, p in peft_backbone.named_parameters():
        if ('lora_A' in name or 'lora_B' in name) and '.default' in name:
            cov = p.float() @ p.float().T if 'lora_A' in name else p.float().T @ p.float()
            terms.append(torch.linalg.matrix_norm(cov - torch.eye(len(cov), device=cov.device)))
    if not terms:
        raise ValueError('no AdaLoRA matrices found')
    return config.orth_reg_weight * torch.stack(terms).mean()


def adalora_objective(model, inputs, labels, indices):
    return F.cross_entropy(model(*inputs), labels) + adalora_regularization(model.backbone)


class AdaLoraStep:
    """Update importance after backward/optimizer.step, before gradients are reset."""
    def __init__(self):
        self.trace = []

    def __call__(self, step, model):
        backbone = model.backbone
        backbone.base_model.update_and_allocate(step)
        cfg = backbone.peft_config['default']
        pattern = cfg.rank_pattern
        if pattern and (step % 10 == 0 or step == cfg.total_step - 1):
            self.trace.append(dict(step=step, total_active_rank=sum(sum(v) for v in pattern.values()),
                per_matrix_rank={k: sum(v) for k, v in pattern.items()}))


def subject_pair_order(subjects, seed):
    """Shuffle pairs from each person; every row exactly once, even counts required.

    Same-subject positives for DEFT. Unlike the official drop-last sampler this
    retains the final partial batch, keeping the number of examples per epoch.
    """
    subjects = np.asarray(subjects)
    rng = np.random.default_rng(seed)
    groups = []
    for subject in np.unique(subjects):
        ix = np.flatnonzero(subjects == subject)
        if len(ix) % 2:
            raise ValueError('subject-paired training requires even rows per subject')
        groups.extend(rng.permutation(ix).reshape(-1, 2))
    return np.stack(groups)[rng.permutation(len(groups))].reshape(-1)


class TokenPoolHead(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, tokens):
        return self.head(tokens.mean(2).flatten(1))


class BrainOmniDEFT(nn.Module):
    """Official DEFT Gaussian model with BrainOmni tokens/head and RVQ reconstruction.

    Reconstruction target is the frozen quantized embedding, not a raw EEG patch.
    All effective backbone blocks are tuned, as in DEFT's full-tuning experiments.
    """
    def __init__(self, classifier):
        super().__init__()
        from .._third_party.deft.gaussian_model import DisentangleV7LiteModel
        base = nn.Module()
        base.backbone = classifier.backbone
        base.classifier = TokenPoolHead(classifier.class_head)
        dim = classifier.class_head[1].out_features
        self.deft = DisentangleV7LiteModel(base, d_model=dim, patch_size=dim)

    @property
    def backbone(self):
        return self.deft.backbone

    @property
    def class_head(self):
        return self.deft.classifier.head

    def forward(self, tokens):
        return self.deft(tokens)['task_logits']

    def loss(self, tokens, labels, subjects):
        from .._third_party.deft.gaussian_model import LossToggles
        outputs = self.deft(tokens)
        # Recursion/precision and metric-distance losses run in FP32 under AMP.
        losses = self.deft.compute_losses(tokens, labels, subjects, outputs,
            task_kind='multiclass', label_smoothing=0.,
            toggles=LossToggles(use_film_reconstruction=True), temperature=.07, var_ratio_r=2.)
        weights = dict(task=1., style_cl=.2, style_invar=.02, var_ratio=.1, recon=.02)
        return sum(weights[k] * v for k, v in losses.items()), losses


class BrainOmniResidualMLP(nn.Module):
    """Parameter-matched task-only residual MLP control; same backbone and head."""
    def __init__(self, classifier, adapter_parameters):
        super().__init__()
        self.backbone = classifier.backbone
        self.class_head = classifier.class_head
        dim = classifier.class_head[1].out_features
        # LayerNorm(2D) + two biased Linear layers: (2D+1)H+3D.
        hidden = max(1, round((adapter_parameters - 3 * dim) / (2 * dim + 1)))
        self.adapter = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, tokens):
        x = self.backbone(tokens)
        x = x + self.adapter(x)
        return self.class_head(x.mean(2).flatten(1))


class BrainOmniMaskedTokens(nn.Module):
    """Original BrainOmni masked RVQ prediction after frozen-token caching.

    Tune only the last block used by encode (penultimate pretraining block).
    The final pretraining block/head remain frozen but propagate loss gradients.
    This avoids adapting an unused block that the downstream path discards.
    """
    def __init__(self, encoder):
        super().__init__()
        from .finetuning import configure_trainable_parameters
        configure_trainable_parameters(encoder, 'frozen')
        self.encoder = encoder
        configure_trainable_parameters(encoder.blocks[-2], 'full')

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.encoder.blocks[-2].train(mode)
        return self

    def forward(self, tokens, indices):
        encoder = self.encoder
        b, c, w, d = tokens.shape
        preserve = torch.rand((b, c, w), device=tokens.device) > encoder.mask_ratio
        random_tokens = tokens.reshape(-1, d)[torch.randperm(b*c*w, device=tokens.device)].reshape_as(tokens)
        x = torch.where(preserve[..., None], tokens, random_tokens)
        keep = ((preserve.float() + torch.rand((b, c, w), device=tokens.device)) > .8)[..., None]
        x = torch.where(keep, x, encoder.mask_token.to(x))
        x = encoder.projection(x + encoder.tokenizer.encoder.neuros.detach()[None, :, None].to(x))
        for block in encoder.blocks:
            x = block(x)
        logits = encoder.predict_head(x).reshape(b, c, w, encoder.num_quantizers_used, -1)
        return encoder.compute_cross_entropy(logits, indices.long(), preserve)[0]


@torch.no_grad()
def cache_rvq_indices(encoder, x, pos, sensor_type, *, batch=4, device='cuda'):
    """Cache original discrete labels; no labels or learned target-domain fitting."""
    encoder.to(device).eval()
    p = torch.as_tensor(pos, dtype=torch.float32, device=device)
    t = torch.as_tensor(sensor_type, dtype=torch.long, device=device)
    result = []
    for start in range(0, len(x), batch):
        windows = torch.as_tensor(x[start:start+batch], dtype=torch.float32, device=device)
        _, indices = encoder.tokenizer.tokenize(windows, p[None].expand(len(windows), -1, -1),
            t[None].expand(len(windows), -1), encoder.overlap_ratio)
        result.append(indices.cpu())
    return torch.cat(result)
