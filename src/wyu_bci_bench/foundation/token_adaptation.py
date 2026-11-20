"""Position-preserving features and last-block adaptation of audited encoders."""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn


def eegpt_proteus_input(x, fs, names):
    """Within-window EMS then CAR, explicit unitless scale1, resample to256Hz.

    EMS follows the authors' BCIC downstream loader; FFT resampling and direct
    named 41-electrode input are this task's transfer choices. No labels or
    other windows enter the transform.
    """
    from braindecode.preprocessing import exponential_moving_standardize
    from .eegpt import prepare_input
    x=np.asarray(x,dtype=np.float32)
    standardized=np.stack([exponential_moving_standardize(a,factor_new=.001,init_block_size=None,eps=1e-4) for a in x])
    standardized-=standardized.mean(axis=1,keepdims=True)
    return prepare_input(standardized,fs,names,amplitude_scale=1.)


@torch.inference_mode()
def tokens_and_prefix(encoder, x, kind, ids):
    """Return official output and input to final block; frozen prefix is eval-only."""
    encoder.eval(); captured=[]
    def capture(module,args): captured.append(args[0].detach())
    hook=encoder.blocks[-1].register_forward_pre_hook(capture)
    try:
        if kind=='eegpt':
            tokens=encoder(x,ids)
            prefix=captured[0].reshape(len(x),16,x.shape[1]+4,512)
        elif kind=='labram':
            if encoder.rel_pos_bias is not None:
                raise ValueError('cached prefix requires no shared relative-position bias')
            tokens=encoder.forward_features(x,input_chans=ids,return_all_tokens=True)
            prefix=captured[0]
        else: raise ValueError(kind)
    finally: hook.remove()
    return tokens,prefix


def pool_tokens(tokens,kind,mode):
    """Global or fixed four-period summaries; named channel IDs handled upstream."""
    if kind=='eegpt':
        z=tokens.mean(-2)  # batch,16,512
        return z.mean(1) if mode=='global' else z.reshape(len(z),4,4,512).mean(2).flatten(1)
    if kind=='labram':
        z=tokens[:,1:].reshape(len(tokens),-1,4,200).mean(1)
        return z.mean(1) if mode=='global' else z.flatten(1)
    raise ValueError(kind)


class LastBlockClassifier(nn.Module):
    def __init__(self,encoder,kind,classes=3):
        super().__init__(); self.kind=kind
        self.block=copy.deepcopy(encoder.blocks[-1])
        self.norm=copy.deepcopy(encoder.norm)
        self.fc_norm=copy.deepcopy(encoder.fc_norm) if kind=='labram' else nn.Identity()
        for p in self.block.parameters(): p.requires_grad_(True)
        # Retain frozen final normalization to isolate one-block adaptation.
        for module in (self.norm,self.fc_norm):
            for p in module.parameters(): p.requires_grad_(False)
        self.head=nn.Linear(2048 if kind=='eegpt' else 800,classes)

    def encode(self,prefix):
        if self.kind=='eegpt':
            b=prefix.shape[0]
            z=self.block(prefix.flatten(0,1))[:,-4:]
            return self.norm(z).reshape(b,16,4,512)
        return self.fc_norm(self.norm(self.block(prefix)))

    def forward(self,prefix):
        return self.head(pool_tokens(self.encode(prefix),self.kind,'time'))
