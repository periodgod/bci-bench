"""Parameter-efficient and conventional fine-tuning helpers for EEG backbones.

The helpers only control which parameters receive gradients or attach an adapter.
They do not choose an EEG input contract, data split, optimizer schedule, or loss.
Those remain experiment decisions and must be made inside the training fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class TrainableParameterCount:
    """Trainable/total parameter counts, useful for comparing adaptation cost."""

    trainable: int
    total: int

    @property
    def fraction(self) -> float:
        """Fraction of model parameters that can receive gradients."""
        return self.trainable / self.total if self.total else 0.0


def configure_trainable_parameters(
    model: Any,
    strategy: str,
    *,
    module_names: Iterable[str] = (),
) -> TrainableParameterCount:
    """Set ``requires_grad`` for full, frozen, or explicitly selected modules.

    ``strategy`` is one of ``full``, ``frozen`` or ``selected``. For ``selected``,
    give exact names from ``model.named_modules()`` such as ``head`` or
    ``encoder.blocks.3``. A selected parent includes all of its descendants.
    Explicit names prevent accidental unfreezing caused by guessed layer naming.
    """
    if not hasattr(model, "named_parameters") or not hasattr(model, "named_modules"):
        raise TypeError("model must be a PyTorch module")
    names = tuple(dict.fromkeys(str(name) for name in module_names))
    if strategy not in {"full", "frozen", "selected"}:
        raise ValueError("strategy must be 'full', 'frozen', or 'selected'")
    if strategy == "selected" and not names:
        raise ValueError("selected strategy requires one or more exact module_names")

    known = {name for name, _ in model.named_modules()}
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"unknown module_names: {unknown}")

    for parameter_name, parameter in model.named_parameters():
        module_name = parameter_name.rpartition(".")[0]
        trainable = strategy == "full" or (
            strategy == "selected"
            and any(module_name == name or module_name.startswith(name + ".") for name in names)
        )
        parameter.requires_grad_(trainable)

    return count_trainable_parameters(model)


def count_trainable_parameters(model: Any) -> TrainableParameterCount:
    """Count all and trainable scalar parameters in a PyTorch module."""
    total = trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return TrainableParameterCount(trainable=trainable, total=total)


def attach_peft_adapter(
    model: Any,
    *,
    method: str,
    target_modules: str | list[str],
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    total_steps: int | None = None,
) -> Any:
    """Attach a PEFT adapter supported by the installed ``peft`` package.

    Supported methods are standard LoRA, DoRA, PiSSA-initialized LoRA, and
    AdaLoRA. The target module names must be checked against the actual backbone.
    QLoRA is intentionally not included: it additionally requires a compatible
    quantization stack (for example bitsandbytes), absent from this workspace.
    """
    if rank < 1 or alpha < 1:
        raise ValueError("rank and alpha must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if not target_modules or (isinstance(target_modules, list) and not target_modules):
        raise ValueError("target_modules must name at least one backbone module")
    if method == "adalora" and (total_steps is None or total_steps < 1):
        raise ValueError("AdaLoRA requires total_steps > 0 for its rank-allocation schedule")
    try:
        from peft import AdaLoraConfig, LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError("PEFT methods require the optional 'peft' package") from exc

    common = dict(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                  target_modules=target_modules, bias="none")
    if method == "lora":
        config = LoraConfig(**common)
    elif method == "dora":
        config = LoraConfig(**common, use_dora=True)
    elif method == "pissa":
        config = LoraConfig(**common, init_lora_weights="pissa")
    elif method == "adalora":
        adalora = {key: value for key, value in common.items() if key != "r"}
        config = AdaLoraConfig(**adalora, target_r=rank, init_r=rank * 2, total_step=total_steps,
            tinit=int(total_steps * .1), tfinal=max(1, int(total_steps * .2)), deltaT=10)
    else:
        raise ValueError("method must be one of: lora, dora, pissa, adalora")
    return get_peft_model(model, config)
