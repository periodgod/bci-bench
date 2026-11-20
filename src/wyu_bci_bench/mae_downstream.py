"""Task readout of the exact same masked EEG encoder, with frozen control."""
import numpy as np
import torch
from torch import nn


class MaskedEEGClassifier(nn.Module):
    def __init__(self,autoencoder,n_patches,classes=3,*,frozen=False):
        super().__init__();self.autoencoder=autoencoder;self.frozen=frozen
        for p in autoencoder.parameters():p.requires_grad_(not frozen)
        for p in autoencoder.decoder.parameters():p.requires_grad_(False)
        autoencoder.mask_token.requires_grad_(False)
        self.head=nn.Linear(autoencoder.patch_projection.out_features*n_patches,classes)

    def train(self,mode=True):
        super().train(mode)
        if self.frozen:self.autoencoder.eval()
        return self

    def encode(self,x):return self.autoencoder.encode_tokens(x).mean(1).flatten(1)

    def forward(self,x):return self.head(self.encode(x))


@torch.inference_mode()
def extract_features(model,x,device='cuda',batch=64):
    model=model.to(device).eval();out=[]
    for start in range(0,len(x),batch):
        out.append(model.encode(torch.as_tensor(x[start:start+batch],dtype=torch.float32,device=device)).cpu().numpy())
    return np.concatenate(out)
