"""SessionFM downstream adaptation primitives for the preregistered K5 protocol."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..priors import MathPriorFeatures
from ..session_causal import causal_center, causal_class_balance
from .sessionfm import SessionFM, session_spectral_features


@dataclass
class SessionFMState:
    """Unlabelled sufficient statistics for one ordered session stream."""
    spectral_sum: torch.Tensor | None = None
    prior_sum: torch.Tensor | None = None
    probability_sum: torch.Tensor | None = None
    count: int = 0

    def reset_state(self) -> None:
        self.spectral_sum = self.prior_sum = self.probability_sum = None
        self.count = 0


class SessionFMDownstream(nn.Module):
    """Frozen training-fold Logistic anchor plus gated SessionFM residual."""

    def __init__(self, backbone: SessionFM, prior: nn.Module | None = None):
        super().__init__()
        self.backbone = backbone
        self.prior = prior or MathPriorFeatures(learn_edges=False)
        for p in self.prior.parameters():
            p.requires_grad_(False)
        for p in self.prior.buffers():
            p.requires_grad_(False)
        self.register_buffer("prior_weight", torch.zeros(backbone.n_classes, 1394))
        self.register_buffer("prior_bias", torch.zeros(backbone.n_classes))
        self.state = SessionFMState()

    @torch.no_grad()
    def fit_prior(self, x: torch.Tensor, y: torch.Tensor, *, keys=None, alpha: float = 32.0,
                  max_iter: int = 1000) -> None:
        """Fit only on caller-supplied training-fold windows and freeze logits."""
        from sklearn.linear_model import LogisticRegression

        features = self.prior(x).detach().cpu().numpy()
        if keys is not None:
            features = causal_center(features, keys, alpha=alpha)
        classifier = LogisticRegression(max_iter=max_iter, random_state=0, C=1.0)
        classifier.fit(features, y.detach().cpu().numpy())
        self.prior_weight.zero_()
        self.prior_bias.zero_()
        self.prior_weight[classifier.classes_] = torch.as_tensor(classifier.coef_, device=self.prior_weight.device,
                                                                  dtype=self.prior_weight.dtype)
        self.prior_bias[classifier.classes_] = torch.as_tensor(classifier.intercept_, device=self.prior_bias.device,
                                                                dtype=self.prior_bias.dtype)

    def prior_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.prior(x)

    def prior_logits(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.prior_weight, self.prior_bias)

    def forward(self, x: torch.Tensor, *, coords=None, baseline=None, n_seen=0,
                prior_features=None, prior_logits=None, mask=None):
        features = self.prior_features(x) if prior_features is None else prior_features
        anchor = self.prior_logits(features) if prior_logits is None else prior_logits
        result = self.backbone(x, coords=coords, baseline=baseline, n_seen=n_seen,
                               prior_logits=anchor, mask=mask)
        result["prior_logits"] = anchor
        result["prior_features"] = features
        return result

    @torch.no_grad()
    def predict_stream(self, x: torch.Tensor, *, coords: torch.Tensor, alpha: float = 32.0,
                       state: SessionFMState | None = None, stateful: bool = True) -> torch.Tensor:
        """Predict one window using past-only SBC, prior centering and CCB."""
        state = self.state if state is None else state
        if x.ndim == 2:
            x = x.unsqueeze(0)
        spec = session_spectral_features(x)
        c = x.shape[1]
        spec_mean = self.backbone.prior_mean[:c]
        if stateful and state.count:
            past = state.spectral_sum / state.count
            w = state.count / (state.count + alpha)
            baseline = w * past + (1 - w) * spec_mean
            n_seen = state.count
        else:
            baseline, n_seen = spec_mean, 0
        pf = self.prior_features(x)
        centered = pf
        if stateful and state.count:
            w = state.count / (state.count + alpha)
            centered = pf - w * state.prior_sum / state.count
        logits = self.prior_logits(centered)
        result = self.backbone(x, coords=coords, baseline=baseline, n_seen=n_seen,
                               prior_logits=logits)
        p = torch.softmax(result["logits"], dim=-1)
        if stateful and state.count:
            mean = state.probability_sum / state.count
            w = state.count / (state.count + alpha)
            prior = (1 - w) / p.shape[-1] + w * mean
            q = p / prior.clamp_min(1e-12)
            p = q / q.sum(-1, keepdim=True)
        if stateful:
            s = spec.detach().reshape(1, c, 4, 10).mean(2).squeeze(0)
            f = pf.detach().squeeze(0)
            state.spectral_sum = s if state.spectral_sum is None else state.spectral_sum + s
            state.prior_sum = f if state.prior_sum is None else state.prior_sum + f
            state.probability_sum = (torch.softmax(result["logits"], -1).squeeze(0) if state.probability_sum is None
                                     else state.probability_sum + torch.softmax(result["logits"], -1).squeeze(0))
            state.count += 1
        return p

    def reset_state(self) -> None:
        self.state.reset_state()


def sessionfm_parameter_groups(model: SessionFMDownstream, *, top_lr: float = 1e-4,
                               head_lr: float = 4e-4, decay: float = 0.65) -> list[dict]:
    """Build depth-wise LR groups; deeper backbone blocks receive top_lr."""
    blocks = list(model.backbone.blocks)
    groups = []
    for depth, block in enumerate(blocks):
        exponent = len(blocks) - 1 - depth
        groups.append({"params": list(block.parameters()), "lr": top_lr * decay ** exponent})
    head = [p for n, p in model.named_parameters() if n.startswith("backbone.pool_query") or
            n.startswith("backbone.residual_head") or n == "backbone.gate"]
    excluded = {id(p) for p in head}
    used = {id(p) for g in groups for p in g["params"]} | excluded
    other_backbone = [p for p in model.backbone.parameters() if id(p) not in used]
    groups.insert(0, {"params": other_backbone, "lr": top_lr * decay ** len(blocks)})
    groups.append({"params": head, "lr": head_lr})
    return groups


def configure_sessionfm_stage(model: SessionFMDownstream, stage: int) -> None:
    """Stage 1 freezes backbone; stage 2 unfreezes backbone and residual head."""
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    for p in model.backbone.parameters():
        p.requires_grad_(stage == 2)
    if stage == 1:
        for p in (model.backbone.pool_query, *model.backbone.residual_head.parameters(), model.backbone.gate):
            p.requires_grad_(True)
    for p in model.prior.parameters():
        p.requires_grad_(False)


def sessionfm_adaptation_loss(result: dict, target_spec: torch.Tensor, y: torch.Tensor,
                              mask: torch.Tensor) -> torch.Tensor:
    """0.9 smoothed CE + 0.1 masked spectral reconstruction MSE."""
    ce = F.cross_entropy(result["logits"], y, label_smoothing=0.1)
    pred = result["spec_reconstruction"]
    selected = mask.reshape(mask.shape[0], mask.shape[1], 4, 2).any(-1)
    if not selected.any():
        raise ValueError("spectral reconstruction mask must select at least one segment")
    mse = (pred - target_spec).square().mean(-1)[selected].mean()
    return 0.9 * ce + 0.1 * mse


def sample_sessionfm_mask(batch: int, channels: int, *, device=None,
                          generator: torch.Generator | None = None,
                          mask_ratio: float = 0.30) -> torch.Tensor:
    """Sample independent 30% segment masks in the model's (B,C,8) token layout."""
    if not 0 < mask_ratio < 1:
        raise ValueError("mask_ratio must be between zero and one")
    count = max(1, min(batch * channels * 8 - 1,
                       round(batch * channels * 8 * mask_ratio)))
    scores = torch.rand((batch * channels * 8,), device=device, generator=generator)
    selected = scores.argsort()[:count]
    mask = torch.zeros((batch * channels * 8,), dtype=torch.bool, device=device)
    mask[selected] = True
    return mask.reshape(batch, channels, 8)


def sessionfm_training_optimizer(model: SessionFMDownstream, stage: int):
    configure_sessionfm_stage(model, stage)
    if stage == 1:
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=4e-4, weight_decay=0.05)
    groups = sessionfm_parameter_groups(model)
    groups = [{**g, "params": [p for p in g["params"] if p.requires_grad]} for g in groups]
    return torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.05)


def fit_sessionfm_lpft(model: SessionFMDownstream, train: dict, validation: dict, *,
                       seed: int, score_fn, device: str = "cpu", batch_size: int = 32,
                       max_epochs: int = 45, patience: int = 8, min_epochs: int = 12):
    """Run the fixed 10-epoch frozen stage and inner-validation-selected FT stage.

    Data dictionaries contain tensors ``x``, ``y``, ``subjects``, ``coords``,
    ``baseline``, ``prior_logits`` and ``target_spec``. ``score_fn`` receives
    validation probabilities and the validation mapping; callers should supply
    the preregistered cell-balanced-accuracy calculation.
    """
    from copy import deepcopy
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    from ..nets import sampling_weights, seed_all

    if not (12 <= min_epochs <= max_epochs <= 45) or patience != 8:
        raise ValueError("stage 2 requires min_epochs >= 12, max_epochs <= 45, patience=8")
    required = {"x", "y", "subjects", "coords", "baseline", "prior_logits", "target_spec"}
    if required - train.keys() or required - validation.keys():
        raise KeyError("train and validation must contain " + ", ".join(sorted(required)))
    seed_all(seed)
    model.to(device)
    y = torch.as_tensor(train["y"], dtype=torch.long)
    tensors = TensorDataset(*(torch.as_tensor(train[k]) for k in
                              ("x", "coords", "baseline", "prior_logits", "target_spec", "y")))
    weights = sampling_weights(np.asarray(train["y"]), np.asarray(train["subjects"]))
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(y),
                                    replacement=True, generator=torch.Generator().manual_seed(seed + 9000))
    loader = DataLoader(tensors, batch_size=batch_size, sampler=sampler)

    optimizer1 = sessionfm_training_optimizer(model, 1)
    optimizer2 = None

    def train_stage(stage: int, epochs: int):
        nonlocal optimizer2
        if stage == 1:
            optimizer = optimizer1
        else:
            if optimizer2 is None:
                optimizer2 = sessionfm_training_optimizer(model, 2)
            optimizer = optimizer2
        history = []
        for _ in range(epochs):
            model.train()
            if stage == 1:
                model.backbone.eval()
                model.backbone.pool_query.requires_grad_(True)
                model.backbone.residual_head.train()
            for bx, bc, bb, bp, bt, by in loader:
                bx, bc, bb, bp, bt, by = (v.to(device) for v in (bx, bc, bb, bp, bt, by))
                mask = sample_sessionfm_mask(len(bx), bx.shape[1], device=bx.device)
                result = model(bx, coords=bc, baseline=bb, prior_logits=bp, mask=mask)
                loss = sessionfm_adaptation_loss(result, bt, by, mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                optimizer.step()
            history.append(float(loss.detach()))
        return history

    first_history = train_stage(1, 10)
    train_stage(2, min_epochs)
    best_score = -float("inf")
    best_epoch, stale, best_state = min_epochs, 0, None
    val_x = torch.as_tensor(validation["x"], dtype=torch.float32)
    for epoch in range(min_epochs, max_epochs + 1):
        model.eval()
        probs = []
        with torch.no_grad():
            for start in range(0, len(val_x), batch_size):
                end = start + batch_size
                out = model(val_x[start:end].to(device),
                            coords=torch.as_tensor(validation["coords"])[start:end].to(device),
                            baseline=torch.as_tensor(validation["baseline"])[start:end].to(device),
                            prior_logits=torch.as_tensor(validation["prior_logits"])[start:end].to(device))
                probs.append(torch.softmax(out["logits"], -1).cpu())
        score = float(score_fn(torch.cat(probs).numpy(), validation))
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch < max_epochs and stale < patience:
            train_stage(2, 1)
        elif stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("inner validation did not select a checkpoint")
    model.load_state_dict(best_state)
    return model.eval(), {"stage1_epochs": 10, "stage1_last_loss": first_history[-1],
                          "best_epoch": best_epoch, "best_inner_score": best_score}


__all__ = ["SessionFMDownstream", "SessionFMState", "sessionfm_parameter_groups",
           "configure_sessionfm_stage", "sessionfm_adaptation_loss", "sample_sessionfm_mask",
           "sessionfm_training_optimizer", "fit_sessionfm_lpft",
           "causal_center", "causal_class_balance"]
