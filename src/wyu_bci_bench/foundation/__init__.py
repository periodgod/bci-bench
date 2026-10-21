"""Pretrained EEG foundation-model adapters with explicit input contracts.

Each adapter states the exact transformation from our window cache to the
model's expected input. Only contracts audited against the original paper and
official loader are marked ``verified``; using an unverified one raises, so an
old mismatched mapping (e.g. the pre-2026-10-19 two-patch, unscaled CBraMod
input) cannot silently come back.
"""

from __future__ import annotations

from dataclasses import dataclass

from .finetuning import (TrainableParameterCount, attach_peft_adapter,
                         configure_trainable_parameters, count_trainable_parameters)

__all__ = ["InputContract", "CONTRACTS", "get_contract", "require_verified",
           "TrainableParameterCount", "configure_trainable_parameters",
           "count_trainable_parameters", "attach_peft_adapter",
           "BIOT_16_CHANNELS", "BIOT_18_CHANNELS", "biot_prepare_input",
           "load_biot_encoder", "extract_biot_features",
           "EEGPT_CHANNELS", "eegpt_prepare_input", "load_eegpt_encoder",
           "extract_eegpt_features"]


@dataclass(frozen=True)
class InputContract:
    """基础模型输入适配合同及其审计状态。

    ``steps`` 按执行顺序描述从 bench 窗口到模型输入的转换；``source`` 指向核验
    所依据的论文/官方加载器或本地复现实验。``verified`` 为 False 时，调用方不能
    把该模型用于正式实验，避免形状看似兼容但单位、采样率或 patch 定义错误。
    """
    model: str
    verified: bool
    steps: tuple[str, ...]
    output: str
    source: str
    notes: str = ""


CONTRACTS: dict[str, InputContract] = {
    "cbramod": InputContract(
        model="CBraMod (ICLR 2025, arXiv:2412.07236)",
        verified=True,
        steps=("CAR over channels", "microvolts / 100 (official downstream loaders)",
               "resample 480 @ 120 Hz -> 800 @ 200 Hz (scipy.signal.resample)",
               "reshape to (N, C, 4, 200): four 1-second patches", "frozen encoder, proj_out = Identity",
               "mean over channels and patches"),
        output="200-D window embedding",
        source="representation_learning_20261019/cbramod_full4s_probe.py; 复现报告.md section 4",
        notes="Upsampling does not restore content above the cache's 60 Hz Nyquist limit.",
    ),
    "labram": InputContract(
        model="LaBraM (ICLR 2024, arXiv:2405.18765)", verified=False, steps=(), output="",
        source="pilot_labram/ (BCIC-IV-2a line); PROTEUS probe labram_proteus_probe.py not re-audited",
        notes="Audit channel vocabulary, 200 Hz patching and amplitude scaling before use.",
    ),
    "eegpt": InputContract(
        model="EEGPT (NeurIPS 2024)", verified=True,
        steps=("require exactly the official 58-channel EEGPT montage by channel name",
               "require a 4-second epoch and resample to 256 Hz (1024 samples)",
               "apply an explicit caller-supplied amplitude multiplier; no universal source scale is assumed",
               "strictly load target_encoder weights into the official EEGTransformer architecture",
               "encode to (batch, 16 time patches, 4 summary tokens, 512 features)",
               "mean-pool time patches and summary tokens to a 512-D frozen window feature"),
        output="512-D pooled frozen feature; raw encoder tokens are 16 x 4 x 512",
        source="EEGPT official README.md, pretrain/engine_pretraining.py, downstream/Modules/models/EEGPT_mcae_finetune.py; local checkpoint strict-load audit 2026-10-21",
        notes="The amplitude multiplier is mandatory and dataset-specific. The authors' pretraining preparation merges sources with differing unit conversions, so this adapter does not infer one universal scale. Strict loading verifies key/shape compatibility, not checkpoint provenance. The adapter exposes encoder-only features; it does not claim equivalence to each downstream classifier's predictor/reconstructor path.",
    ),
    "biot": InputContract(
        model="BIOT (NeurIPS 2023)", verified=True,
        steps=("select an exact ordered checkpoint montage (16 or 18 bipolar channels)",
               "resample each epoch to the checkpoint's 200 Hz rate",
               "normalize each channel by its per-epoch 95th percentile absolute amplitude",
               "construct BIOTEncoder with n_fft=200 and hop_length=100 (or checkpoint-matched values)",
               "load the official encoder state dict strictly and freeze it",
               "apply channel identity tokens, STFT magnitude patches, linear attention, and temporal mean pooling"),
        output="256-D window embedding",
        source="https://github.com/ycq091044/BIOT (README.md, model/biot.py, utils.py, run_example.py)",
        notes="Only official 16/18-channel checkpoint montages are supported. Signal channel names must exactly match the selected ordered montage; checkpoint weights must be supplied. The official TUAB loader uses 10 s windows; other durations need protocol-specific validation.",
    ),
}


def get_contract(name: str) -> InputContract:
    """按规范名称读取合同；未知名称会列出当前已登记模型。"""
    try:
        return CONTRACTS[name]
    except KeyError as exc:
        raise KeyError(f"no contract for {name!r}; known: {sorted(CONTRACTS)}") from exc


def require_verified(name: str) -> InputContract:
    """仅返回已完成输入合同核验的模型；否则明确阻止运行。"""
    contract = get_contract(name)
    if not contract.verified:
        raise NotImplementedError(
            f"{name} input contract is not verified ({contract.notes}). Audit it against the paper and "
            f"official loader, record the steps in wyu_bci_bench.foundation.CONTRACTS, then set verified=True.")
    return contract


from .biot import BIOT_16_CHANNELS, BIOT_18_CHANNELS
from .biot import extract_features as extract_biot_features
from .biot import load_encoder as load_biot_encoder
from .biot import prepare_input as biot_prepare_input
from .eegpt import EEGPT_CHANNELS
from .eegpt import extract_features as extract_eegpt_features
from .eegpt import load_encoder as load_eegpt_encoder
from .eegpt import prepare_input as eegpt_prepare_input
