"""Prior-anchored foundation-model residuals with optional causal calibration."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = nn = None

from ..nets import PowerFirstMathPriorNet


if nn is not None:

    class PAFMSCC(nn.Module):
        """M2 plus a gated foundation-model residual head."""

        def __init__(self, m2: PowerFirstMathPriorNet, foundation_dim: int = 200,
                     foundation_encoder: nn.Module | None = None, hidden_dim: int = 64,
                     n_classes: int = 3, foundation_center: bool = False):
            super().__init__()
            if not isinstance(m2, nn.Module):
                raise TypeError("m2 must be a torch module")
            self.m2 = m2
            self.foundation_encoder = foundation_encoder
            self.foundation_center = bool(foundation_center)
            self.foundation_head = nn.Sequential(
                nn.LayerNorm(int(foundation_dim)), nn.Linear(int(foundation_dim), hidden_dim),
                nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden_dim, n_classes),
            )
            self.foundation_gate = nn.Parameter(torch.tensor(-2.5))

        @staticmethod
        def pool_embedding(value: Any) -> torch.Tensor:
            """Convert common encoder outputs to a 2-D batch embedding."""
            if isinstance(value, (tuple, list)):
                value = value[0]
            if not torch.is_tensor(value):
                raise TypeError("foundation encoder must return a tensor or tuple/list")
            if value.ndim == 2:
                return value
            if value.ndim < 2:
                raise ValueError(f"foundation output needs >=2 dimensions, got {tuple(value.shape)}")
            return value.flatten(1, -2).mean(dim=1)

        def encode_foundation(self, foundation_input: torch.Tensor) -> torch.Tensor:
            if self.foundation_encoder is None:
                raise ValueError("no foundation_encoder is attached")
            return self.pool_embedding(self.foundation_encoder(foundation_input))

        def forward(self, raw: torch.Tensor, math_features: torch.Tensor,
                    foundation_input: torch.Tensor | None = None,
                    foundation_embedding: torch.Tensor | None = None) -> torch.Tensor:
            logits = self.m2(raw, math_features)
            if foundation_embedding is None:
                if foundation_input is None:
                    raise ValueError("provide foundation_input or foundation_embedding")
                foundation_embedding = self.encode_foundation(foundation_input)
            if foundation_embedding.ndim != 2:
                foundation_embedding = self.pool_embedding(foundation_embedding)
            if self.foundation_center:
                foundation_embedding = foundation_embedding - foundation_embedding.mean(0, keepdim=True).detach()
            return logits + torch.sigmoid(self.foundation_gate) * self.foundation_head(foundation_embedding)

        def trainable_parameter_count(self) -> tuple[int, int]:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return trainable, total

else:  # pragma: no cover

    class PAFMSCC:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PAFMSCC requires PyTorch")


def train_pafm(model: PAFMSCC, raw: np.ndarray, math_features: np.ndarray,
               foundation_embeddings: np.ndarray, y: np.ndarray, subjects: np.ndarray,
               *, epochs: int, seed: int, device: str = "cpu", batch_size: int = 128,
               learning_rate: float = 4e-4, backbone_learning_rate: float = 1e-4,
               weight_decay: float = 0.05, foundation_is_input: bool = False):
    """Train one isolated fold from embeddings or prepared model inputs."""
    if torch is None:
        raise ImportError("train_pafm requires PyTorch")
    from ..nets import seed_all, sampling_weights
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    seed_all(seed)
    model = model.to(device)
    target = torch.as_tensor(np.asarray(y, dtype=np.int64))
    sampler = WeightedRandomSampler(
        torch.as_tensor(sampling_weights(np.asarray(y), np.asarray(subjects)), dtype=torch.double),
        len(target), replacement=True, generator=torch.Generator().manual_seed(seed + 9000),
    )
    foundation_tensor = torch.as_tensor(foundation_embeddings, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(raw, dtype=torch.float32),
                      torch.as_tensor(math_features, dtype=torch.float32),
                      foundation_tensor, target),
        batch_size=batch_size, sampler=sampler,
    )
    encoder_ids = ({id(p) for p in model.foundation_encoder.parameters()}
                   if model.foundation_encoder is not None else set())
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    encoder_params = [p for p in model.parameters() if p.requires_grad and id(p) in encoder_ids]
    groups = [{"params": head_params, "lr": learning_rate}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": backbone_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    for _ in range(int(epochs)):
        model.train()
        for bx, bf, bz, by in loader:
            optimizer.zero_grad(set_to_none=True)
            if foundation_is_input:
                logits = model(bx.to(device), bf.to(device), foundation_input=bz.to(device))
            else:
                logits = model(bx.to(device), bf.to(device), foundation_embedding=bz.to(device))
            loss = nn.functional.cross_entropy(logits, by.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval()


def train_pafm_early_stopping(
    model: PAFMSCC,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    device: str = "cpu",
    batch_size: int = 32,
    learning_rate: float = 4e-4,
    backbone_learning_rate: float = 1e-4,
    weight_decay: float = 0.05,
    foundation_is_input: bool = False,
):
    """Fit one isolated fold and select the epoch on grouped validation BA.

    ``train`` and ``validation`` each contain ``raw``, ``math_features``,
    ``foundation``, ``y``, ``subjects``, ``sessions`` and ``interfaces`` arrays.
    The returned report contains the validation score history and selected epoch.
    """
    if torch is None:
        raise ImportError("train_pafm_early_stopping requires PyTorch")
    from copy import deepcopy
    from ..metrics import cell_balanced_accuracy
    from ..nets import sampling_weights, seed_all
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    required = {"raw", "math_features", "foundation", "y", "subjects", "sessions", "interfaces"}
    for label, dataset in (("train", train), ("validation", validation)):
        missing = required - set(dataset)
        if missing:
            raise KeyError(f"{label} data lacks {sorted(missing)}")
        lengths = {len(dataset[key]) for key in required}
        if len(lengths) != 1:
            raise ValueError(f"{label} arrays have inconsistent lengths")

    seed_all(seed)
    model = model.to(device)
    train_y = np.asarray(train["y"], dtype=np.int64)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sampling_weights(train_y, np.asarray(train["subjects"])), dtype=torch.double),
        len(train_y), replacement=True, generator=torch.Generator().manual_seed(seed + 9000),
    )
    train_tensors = TensorDataset(
        torch.as_tensor(train["raw"], dtype=torch.float32),
        torch.as_tensor(train["math_features"], dtype=torch.float32),
        torch.as_tensor(train["foundation"], dtype=torch.float32),
        torch.as_tensor(train_y),
    )
    loader = DataLoader(train_tensors, batch_size=batch_size, sampler=sampler)
    encoder_ids = ({id(p) for p in model.foundation_encoder.parameters()}
                   if model.foundation_encoder is not None else set())
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    encoder_params = [p for p in model.parameters() if p.requires_grad and id(p) in encoder_ids]
    groups = [{"params": head_params, "lr": learning_rate}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": backbone_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    if model.foundation_encoder is not None and not encoder_params:
        model.foundation_encoder.eval()

    val_foundation = torch.as_tensor(validation["foundation"], dtype=torch.float32)
    val_raw = torch.as_tensor(validation["raw"], dtype=torch.float32)
    val_math = torch.as_tensor(validation["math_features"], dtype=torch.float32)

    history = []
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        if model.foundation_encoder is not None and not encoder_params:
            model.foundation_encoder.eval()
        for bx, bf, bz, by in loader:
            optimizer.zero_grad(set_to_none=True)
            if foundation_is_input:
                logits = model(bx.to(device), bf.to(device), foundation_input=bz.to(device))
            else:
                logits = model(bx.to(device), bf.to(device), foundation_embedding=bz.to(device))
            loss = nn.functional.cross_entropy(logits, by.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()

        model.eval()
        probabilities = []
        with torch.inference_mode():
            for start in range(0, len(val_raw), batch_size):
                if foundation_is_input:
                    logits = model(val_raw[start:start + batch_size].to(device),
                                   val_math[start:start + batch_size].to(device),
                                   foundation_input=val_foundation[start:start + batch_size].to(device))
                else:
                    logits = model(val_raw[start:start + batch_size].to(device),
                                   val_math[start:start + batch_size].to(device),
                                   foundation_embedding=val_foundation[start:start + batch_size].to(device))
                probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
        probs = np.concatenate(probabilities)
        score = cell_balanced_accuracy(
            validation["y"], probs.argmax(1), validation["subjects"],
            validation["sessions"], validation["interfaces"], n_classes=probs.shape[1],
        )
        history.append({"epoch": epoch, "validation_cell_ba": float(score)})
        if score > best_score:
            best_score, best_epoch, stale = float(score), epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    return model.eval(), {"best_epoch": best_epoch, "best_validation_cell_ba": best_score,
                          "history": history, "stopped_epoch": history[-1]["epoch"]}

