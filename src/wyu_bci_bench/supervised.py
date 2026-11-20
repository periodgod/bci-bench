"""Fold-local supervised training with fixed budgets or a training-loss convergence rule."""
from __future__ import annotations
from contextlib import nullcontext
import time
from copy import deepcopy
from itertools import count
import numpy as np
import torch
from torch.nn import functional as F
from .deep import set_seed


def _autocast_context(device, amp_dtype):
    if amp_dtype is None:
        return nullcontext()
    if amp_dtype != 'bfloat16':
        raise ValueError("amp_dtype must be None or 'bfloat16'")
    if torch.device(device).type != 'cuda' or not torch.cuda.is_bf16_supported():
        raise ValueError('bfloat16 AMP requires a supported CUDA device')
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16)


def balanced_group_class_weights(y, *groups):
    """Equal total loss weight per observed group/class; normalize mean to one."""
    keys = list(zip(*(np.asarray(g).astype(str) for g in groups), np.asarray(y).tolist()))
    counts = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    w = np.asarray([1.0 / counts[key] for key in keys], np.float32)
    return w / w.mean()


def train_classifier(model, inputs, labels, *, epochs, seed, batch=64, lr=1e-3,
                     weight_decay=1e-3, weights=None, device="cuda", parameter_groups=None,
                     epoch_callback=None, amp_dtype=None, objective=None, step_callback=None,
                     batch_order=None, plateau_patience=6, plateau_delta=0.002, min_lr=0.000125):
    """Caller constructs/initializes model with its recorded seed before this call.

    Inputs are a tuple for multi-input models. No test data or validation labels
    are accepted. A callback can evaluate predeclared inner validation epochs.
    Optional CUDA bfloat16 autocast retains FP32 parameters/optimizer state and
    needs no gradient scaling. Default execution is unchanged full precision.
    """
    convergence = epochs is None
    if convergence:
        if objective is not None or parameter_groups is not None:
            raise ValueError("convergence uses weighted cross entropy and one learning rate")
        if not (plateau_patience >= 1 and plateau_delta > 0 and 0 < min_lr <= lr):
            raise ValueError("invalid convergence parameters")
    elif epochs < 1:
        raise ValueError("epochs must be positive or None")
    set_seed(seed)
    _autocast_context(device, amp_dtype)  # validate before allocating or fitting
    gen = torch.Generator().manual_seed(seed)
    model = model.to(device)
    tensors = tuple(torch.as_tensor(x, dtype=torch.float32, device=device) for x in inputs)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    if len(y) < 2 or any(len(x) != len(y) for x in tensors):
        raise ValueError("aligned training inputs with at least two rows are required")
    w = torch.ones(len(y), device=device) if weights is None else torch.as_tensor(weights, dtype=torch.float32, device=device)
    if w.shape != y.shape or not torch.isfinite(w).all() or (w <= 0).any():
        raise ValueError("weights must be finite positive values aligned with labels")
    parameters = parameter_groups if parameter_groups is not None else [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if not convergence:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * int(np.ceil(len(y) / batch)))

    @torch.inference_mode()
    def training_loss():
        model.eval()
        total = torch.zeros((), device=device)
        for start in range(0, len(y), batch):
            ix = slice(start, start + batch)
            with _autocast_context(device, amp_dtype):
                loss = F.cross_entropy(model(*(x[ix] for x in tensors)), y[ix], reduction="none")
            total += (loss * w[ix]).sum()
        result = float(total / w.sum())
        if not np.isfinite(result):
            raise FloatingPointError("non-finite convergence loss")
        return result

    started, epoch_seconds = time.perf_counter(), []
    steps, losses = 0, []
    if convergence:
        initial = best = benchmark = training_loss()
        best_state = deepcopy(model.state_dict())
        loss_curve, learning_rates = [], []
        stale = 0
    for epoch in (count(1) if convergence else range(1, epochs + 1)):
        tick = time.perf_counter()
        model.train()
        order = (torch.randperm(len(y), generator=gen) if batch_order is None
                 else torch.as_tensor(batch_order(epoch), dtype=torch.long))
        if len(order) != len(y) or not torch.equal(order.sort().values, torch.arange(len(y))):
            raise ValueError('batch_order must be a permutation of all training rows')
        order = order.to(device)
        for start in range(0, len(y), batch):
            ix = order[start:start + batch]
            if len(ix) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            with _autocast_context(device, amp_dtype):
                if objective is None:
                    loss = (F.cross_entropy(model(*(x[ix] for x in tensors)), y[ix], reduction="none") * w[ix]).mean()
                else:
                    loss = objective(model, tuple(x[ix] for x in tensors), y[ix], ix)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite supervised loss")
            loss.backward()
            opt.step()
            if step_callback is not None:
                step_callback(steps, model)
            steps += 1
            losses.append(float(loss.detach()))
            if not convergence:
                sched.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        epoch_seconds.append(time.perf_counter() - tick)
        if epoch_callback is not None:
            epoch_callback(epoch, model.eval())
        if convergence:
            value = training_loss()
            learning_rates.append(opt.param_groups[0]["lr"])
            loss_curve.append(value)
            if value < best:
                best, best_state = value, deepcopy(model.state_dict())
            if value < benchmark - plateau_delta:
                benchmark, stale = value, 0
            else:
                stale += 1
                if stale >= plateau_patience:
                    current_lr = learning_rates[-1]
                    if current_lr <= min_lr:
                        break
                    opt.param_groups[0]["lr"] = max(min_lr, 0.25 * current_lr)
                    benchmark, stale = value, 0
    model.eval()
    convergence_report = {}
    if convergence:
        model.load_state_dict(best_state)
        convergence_report = {
            "stop_reason": "training_loss_plateau_at_min_lr", "initial_train_loss": initial,
            "best_train_loss": best, "train_loss_curve": loss_curve, "learning_rates": learning_rates,
        }
    return model, {**convergence_report, "epochs": epoch, "seed": seed, "seconds": time.perf_counter() - started,
                   "amp_dtype": amp_dtype, "optimizer_steps": steps,
                   "first_loss": losses[0], "last_loss": losses[-1],
                   "epoch_seconds": epoch_seconds,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


@torch.inference_mode()
def predict_classifier(model, inputs, *, device="cuda", batch=256, amp_dtype=None):
    model.eval()
    _autocast_context(device, amp_dtype)
    out = []
    for start in range(0, len(inputs[0]), batch):
        xs = tuple(torch.as_tensor(x[start:start + batch], dtype=torch.float32, device=device) for x in inputs)
        with _autocast_context(device, amp_dtype):
            logits = model(*xs)
        out.append(logits.float().softmax(1).cpu().numpy())
    return np.concatenate(out)
