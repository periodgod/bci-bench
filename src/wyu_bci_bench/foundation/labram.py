"""Audited LaBraM encoder adapter; downstream inference needs no VQ tokenizer."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.signal import resample_poly

# Official utils.standard_1020 order (first 95 named scalp/reference electrodes).
CHANNELS = tuple(('FP1 FPZ FP2 AF9 AF7 AF5 AF3 AF1 AFZ AF2 AF4 AF6 AF8 AF10 '
 'F9 F7 F5 F3 F1 FZ F2 F4 F6 F8 F10 FT9 FT7 FC5 FC3 FC1 FCZ FC2 FC4 FC6 FT8 FT10 '
 'T9 T7 C5 C3 C1 CZ C2 C4 C6 T8 T10 TP9 TP7 CP5 CP3 CP1 CPZ CP2 CP4 CP6 TP8 TP10 '
 'P9 P7 P5 P3 P1 PZ P2 P4 P6 P8 P10 PO9 PO7 PO5 PO3 PO1 POZ PO2 PO4 PO6 PO8 PO10 '
 'O1 OZ O2 O9 CB1 CB2 IZ O10 T3 T5 T4 T6 M1 M2 A1 A2').split())


def channel_ids(names):
    names=tuple(str(n).strip().upper() for n in names)
    if not names or len(set(names)) != len(names) or set(names)-set(CHANNELS):
        raise ValueError('channels must be unique known LaBraM scalp/reference names')
    return [0]+[CHANNELS.index(n)+1 for n in names]


def prepare_input(x, fs, names):
    from . import require_verified
    require_verified('labram')
    x=np.asarray(x,dtype=np.float32)
    channel_ids(names)
    if x.ndim!=3 or x.shape[1]!=len(names) or x.shape[-1]!=480 or fs!=120 or not np.isfinite(x).all():
        raise ValueError('audited PROTEUS input is finite (N,C,480) microvolts at 120 Hz')
    # Official engine_for_finetuning.py divides input microvolts by 100.
    return (resample_poly(x,5,3,axis=-1)/100).reshape(len(x),len(names),4,200).astype(np.float32)


def load_encoder(model_file, weights=None, *, seed=0):
    from . import require_verified
    require_verified('labram')
    spec=importlib.util.spec_from_file_location('wyu_bci_bench_official_labram',Path(model_file))
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    torch.manual_seed(seed)
    encoder=module.labram_base_patch200_200(pretrained=False,num_classes=4,
        drop_rate=0.,drop_path_rate=.1,attn_drop_rate=0.,drop_block_rate=None,
        use_mean_pooling=True,init_scale=.001,use_rel_pos_bias=True,
        use_abs_pos_emb=True,init_values=.1,qkv_bias=False)
    if weights is not None:
        state=torch.load(weights,map_location='cpu',weights_only=True)
        report=encoder.load_state_dict(state,strict=False)
        missing={'fc_norm.weight','fc_norm.bias','head.weight','head.bias'}
        unexpected={'mask_token','norm.weight','norm.bias','lm_head.weight','lm_head.bias'}
        if set(report.missing_keys)!=missing or set(report.unexpected_keys)!=unexpected:
            raise RuntimeError(f'LaBraM encoder checkpoint mismatch: {report}')
    encoder.eval()
    for parameter in encoder.parameters(): parameter.requires_grad_(False)
    return encoder
