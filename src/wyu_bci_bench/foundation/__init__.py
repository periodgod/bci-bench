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
    "brainomni": InputContract(
        model="BrainOmni Tiny/Base (NeurIPS 2025, arXiv:2505.18185)", verified=True,
        steps=("caller supplies finite EEG windows in microvolts at256Hz, >=512 samples, and scalp-channel geometry",
               "per-window CAR across EEG channels; divide by std(uV)+10, equivalent to official std(volts)+1e-5",
               "center EEG xyz and divide by sqrt(3*mean(radius^2)); EEG orientation zeros and sensor_type0",
               "strictly load complete official checkpoint including frozen RVQ tokenizer",
               "official tokenizer unfold512 with25% overlap and padding; quantized vectors plus detached source embeddings",
               "projection then blocks[:-1], L2-normalize feature dimension; mean time and flatten for official-style MLP"),
        output="(N,16,W,lm_dim) embeddings; token-level classifier supports frozen/last_block/full/LoRA",
        source="OpenTSLab/BrainOmni commit340d6b5aba886af76b217272cdb3251651e9bf16: factory/utils.py, brainomni/model.py, downstream/model.py; real Tiny strict-load and cached-vs-direct audit2026-11-14",
        notes="Vendored SDPA evaluation dropout corrected to0; training unchanged. Torch distributed replaces DeepSpeed collectives for single-process use. Dataset filtering must be explicit; standard1005 geometry is a template. Full tuning excludes tokenizer, unused final block and pretraining head. Contract verification is not evidence of superiority.",
    ),
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
        model="LaBraM (ICLR 2024, arXiv:2405.18765)", verified=True,
        steps=("named channel IDs using official standard_1020 vocabulary plus CLS index0",
               "PROTEUS microvolt 4s input at120Hz; resample_poly5/3 to200Hz then divide by100",
               "reshape (N,C,4,200); encoder checkpoint with explicitly audited missing/unexpected heads",
               "return all tokens; no VQ tokenizer needed for downstream encoder"),
        output="(N,1+4*C,200) tokens",
        source="official LaBraM engine_for_finetuning.py and utils.py retrieved2026-11-09; historical PROTEUS wrapper audited against them",
        notes="Historical PROTEUS wrapper omitted /100. Corrected transfer contract is evaluated separately; checkpoint key compatibility alone does not establish performance.",
    ),
    "eegpt": InputContract(
        model="EEGPT (NeurIPS 2024)", verified=True,
        steps=("require unique named channels from the official 58-channel EEGPT pretrained montage; subsets supported",
               "require a 4-second epoch and resample to 256 Hz (1024 samples)",
               "apply an explicit caller-supplied amplitude multiplier; no universal source scale is assumed",
               "strictly load target_encoder weights into the official EEGTransformer architecture",
               "encode to (batch, 16 time patches, 4 summary tokens, 512 features)",
               "mean-pool time patches and summary tokens to a 512-D frozen window feature"),
        output="512-D pooled frozen feature; raw encoder tokens are 16 x 4 x 512",
        source="EEGPT official README.md, pretrain/engine_pretraining.py, downstream/Modules/models/EEGPT_mcae_finetune.py and linear_probe_EEGPT_BCIC2A.py (19-channel subset); strict-load audit 2026-10-21, subset source audit 2026-11-09",
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
from .bnci2015 import (BNCI2015_004_CHANNELS, BNCI2015_004_EVENT_IDS, BNCI2015_004_TASKS,
                       audit_bnci2015_004, make_grouped_dataset)
from .pa_fm import PAFMSCC, train_pafm, train_pafm_early_stopping
from .sessionfm import SessionFM, SessionBaselineState, session_spectral_features
from .sessionfm_downstream import (SessionFMDownstream, SessionFMState,
                                   configure_sessionfm_stage, sample_sessionfm_mask,
                                   sessionfm_adaptation_loss, sessionfm_parameter_groups,
                                   sessionfm_training_optimizer)
