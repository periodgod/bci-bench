"""Portable architecture references for additional 2024-2026 EEG candidates.

Several candidate names/descriptions lack a verified paper identity. These
implementations are descriptor-level references, not claimed author replicas.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .frontier_models import StateSpaceBlock


def _load_geometric_csp_models():
    try:
        from ._third_party.geometric_csp.model import Graph_CSPNet_Basic, Tensor_CSPNet_Basic
        from ._third_party.geometric_csp.modules import Graph_BiMap
    except ImportError as exc:
        raise ImportError("Graph/Tensor CSPNet adapters need geoopt; install wyu-bci-bench[geometric-csp]") from exc
    return Graph_CSPNet_Basic, Tensor_CSPNet_Basic, Graph_BiMap


class MIMamba(nn.Module):
    """MI-Mamba: local spatial-temporal convolution, global Mamba, FC classifier."""
    def __init__(self, channels, classes, width=32, state_size=16, kernel_size=15):
        super().__init__(); self.channels=channels
        self.local=nn.Sequential(nn.Conv2d(1,width,(channels,kernel_size),padding=(0,kernel_size//2),bias=False),
                                 nn.BatchNorm2d(width),nn.ELU())
        self.global_mamba=FRSMambaStateSpaceMixer(width,state_size=state_size,use_fourier=False)
        self.classifier=nn.Sequential(nn.Linear(width,width),nn.ELU(),nn.Linear(width,classes))
    def forward(self,x):
        if x.ndim!=3 or x.shape[1]!=self.channels: raise ValueError("invalid EEG shape")
        tokens=self.local(x.unsqueeze(1)).squeeze(2).transpose(1,2)
        return self.classifier(self.global_mamba(tokens).mean(1))


class MultiScalePhysiologicalTokenizer(nn.Module):
    """Three temporal scales followed by spatial, token-fusion and pooling stages."""
    def __init__(self,channels,embedding=64,f1=10,depth_multiplier=3,pool1=8,pool2=8,dropout=0.3):
        super().__init__(); temporal=f1*3; fused=temporal*depth_multiplier
        self.temporal=nn.ModuleList([nn.Conv2d(1,f1,(1,k),padding="same",bias=False) for k in (36,24,18)])
        self.temporal_norm=nn.BatchNorm2d(temporal)
        self.spatial=nn.Conv2d(temporal,fused,(channels,1),groups=temporal,bias=False)
        self.spatial_norm=nn.BatchNorm2d(fused)
        self.pool1=nn.MaxPool2d((1,pool1),stride=(1,pool1)); self.dropout1=nn.Dropout(dropout)
        self.fusion_depth=nn.Conv2d(fused,fused,(1,16),padding="same",groups=fused,bias=False)
        self.fusion_point=nn.Conv2d(fused,fused,1,bias=False); self.fusion_norm=nn.BatchNorm2d(fused)
        self.pool2=nn.MaxPool2d((1,pool2),stride=(1,pool2)); self.dropout2=nn.Dropout(dropout)
        self.project=nn.Identity() if fused==embedding else nn.Linear(fused,embedding)
        self.activation=nn.ELU()
    def forward(self,x):
        if x.ndim==3: x=x.unsqueeze(1)
        if x.ndim!=4 or x.shape[1]!=1 or x.shape[2]!=self.spatial.kernel_size[0]:
            raise ValueError("tokenizer expects (batch, channels, samples) or (batch, 1, channels, samples)")
        x=self.temporal_norm(self.activation(torch.cat([branch(x) for branch in self.temporal],1)))
        x=self.dropout1(self.pool1(self.activation(self.spatial_norm(self.spatial(x)))))
        x=self.dropout2(self.pool2(self.activation(self.fusion_norm(self.fusion_point(self.fusion_depth(x))))))
        return self.project(x.squeeze(2).transpose(1,2))


class FourierRhythmicStateModeling(nn.Module):
    """Blockwise complex Fourier mixing and low/high rhythmic context extraction."""
    def __init__(self,embedding,*,num_blocks=None,sparsity_threshold=0.01,low_ratio=0.5,mode="full"):
        super().__init__(); self.embedding=embedding; self.mode=mode; self.sparsity_threshold=sparsity_threshold; self.low_ratio=low_ratio
        blocks=num_blocks or next((k for k in (8,6,5,4,3,2,1) if embedding%k==0),1)
        if embedding%blocks: raise ValueError("num_blocks must divide embedding width")
        self.blocks=blocks; block_size=embedding//blocks
        self.weight=nn.Parameter(0.02*torch.randn(blocks,block_size,block_size,2))
        self.real_scale=nn.Parameter(0.02*torch.randn(2,blocks,block_size,1))
        self.imag_scale=nn.Parameter(0.02*torch.randn(2,blocks,block_size,1))
        self.bias=nn.Parameter(0.02*torch.randn(2,blocks,block_size))
        self.rhythm_filter=nn.Parameter(torch.ones(1,embedding,1))
        self.context=nn.Sequential(nn.Conv1d(2*embedding,2*embedding,1,bias=False),nn.BatchNorm1d(2*embedding),nn.SiLU(),nn.Conv1d(2*embedding,embedding,1))
        self.norm=nn.LayerNorm(embedding)
    def forward(self,tokens):
        seq=tokens.transpose(1,2); n=seq.shape[-1]
        spectrum=torch.fft.rfft(seq.float(),dim=-1,norm="ortho"); original=spectrum
        b,c,f=spectrum.shape; spectrum=spectrum.reshape(b,self.blocks,self.embedding//self.blocks,f)
        weight=torch.view_as_complex(self.weight.contiguous()); mixed=torch.einsum("bkif,kio->bkof",spectrum,weight)
        real=F.silu(mixed.real*self.real_scale[0].unsqueeze(0)-mixed.imag*self.real_scale[1].unsqueeze(0)+self.bias[0,:,:,None])
        imag=F.silu(mixed.imag*self.imag_scale[0].unsqueeze(0)+mixed.real*self.imag_scale[1].unsqueeze(0)+self.bias[1,:,:,None])
        mixed=torch.view_as_complex(F.softshrink(torch.stack((real,imag),-1),lambd=self.sparsity_threshold)).reshape(b,c,f)
        mixed=mixed*self.rhythm_filter+original
        enhanced=torch.fft.irfft(mixed,n=n,dim=-1,norm="ortho").to(seq.dtype)+seq
        low_bins=max(1,min(f-1,int(round(f*self.low_ratio)))) if f>1 else 1
        low_mask=torch.zeros_like(mixed); low_mask[...,:low_bins]=1; high_mask=1-low_mask
        low=torch.fft.irfft(mixed*low_mask,n=n,dim=-1,norm="ortho")+seq
        high=torch.fft.irfft(mixed*high_mask,n=n,dim=-1,norm="ortho")+seq
        if self.mode=="low_only": used=torch.cat((low,torch.zeros_like(high)),1); residual=low
        elif self.mode=="high_only": used=torch.cat((torch.zeros_like(low),high),1); residual=high
        else: used=torch.cat((low,high),1); residual=enhanced
        context=self.norm((self.context(used)+residual).transpose(1,2))
        return low.transpose(1,2),high.transpose(1,2),context


class FRSMambaStateSpaceMixer(nn.Module):
    """Mamba selective state-space mixer with Fourier-conditioned input/residual gates."""
    def __init__(self,embedding,state_size=16,conv_size=3,use_fourier=True,mode="full"):
        super().__init__(); inner=2*embedding; rank=(embedding+15)//16
        self.embedding=embedding; self.inner=inner; self.state_size=state_size; self.use_fourier=use_fourier; self.mode=mode
        self.in_proj=nn.Linear(embedding,2*inner)
        self.conv=nn.Conv1d(inner,inner,conv_size,groups=inner,padding=conv_size-1)
        self.x_proj=nn.Linear(inner,rank+2*state_size,bias=False); self.dt_proj=nn.Linear(rank,inner)
        self.A_log=nn.Parameter(torch.arange(1,state_size+1,dtype=torch.float32).log().repeat(inner,1))
        self.skip=nn.Parameter(torch.ones(inner)); self.out_proj=nn.Linear(inner,embedding)
        self.context_norm=nn.LayerNorm(embedding); self.token_mod=nn.Linear(embedding,inner); self.residual_mod=nn.Linear(embedding,inner)
    def forward(self,tokens,rhythmic_context=None):
        length=tokens.shape[1]; value,residual=self.in_proj(tokens).chunk(2,-1)
        if self.use_fourier and self.mode!="none" and rhythmic_context is not None:
            ctx=self.context_norm(rhythmic_context); value=value*(1+torch.sigmoid(self.token_mod(ctx))); residual=residual+self.residual_mod(ctx)
        value=F.silu(self.conv(value.transpose(1,2))[...,:length].transpose(1,2))
        delta,b,c=self.x_proj(value).split((self.dt_proj.in_features,self.state_size,self.state_size),-1)
        delta=F.softplus(self.dt_proj(delta)); decay=-self.A_log.float().exp()
        state=value.new_zeros((value.shape[0],self.inner,self.state_size),dtype=torch.float32); outputs=[]
        vf=value.float(); df=delta.float(); bf=b.float(); cf=c.float()
        for t in range(length):
            da=torch.exp(df[:,t,:,None]*decay[None])
            dbu=df[:,t,:,None]*bf[:,t,None,:]*vf[:,t,:,None]
            state=da*state+dbu
            outputs.append((state*cf[:,t,None,:]).sum(-1))
        mixed=torch.stack(outputs,1).to(value.dtype)+value*self.skip.to(value)
        return self.out_proj(mixed*F.silu(residual))


class FRSMambaBlock(nn.Module):
    def __init__(self,embedding,dropout=0.3,use_fourier=True,mode="full",state_size=16,
                 num_blocks=None,sparsity_threshold=0.01,low_ratio=0.5):
        super().__init__(); self.norm=nn.LayerNorm(embedding); self.use_fourier=use_fourier; self.mode=mode
        self.rhythm=FourierRhythmicStateModeling(embedding,mode=mode,num_blocks=num_blocks,
                                                 sparsity_threshold=sparsity_threshold,low_ratio=low_ratio) if use_fourier else None
        self.mixer=FRSMambaStateSpaceMixer(embedding,state_size=state_size,use_fourier=use_fourier,mode=mode)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        norm=self.norm(x); context=None
        if self.rhythm is not None and self.mode!="none": _,_,context=self.rhythm(norm)
        return x+self.dropout(self.mixer(norm,context))


class CFSPMNet(nn.Module):
    """Fourier-guided Spatial-Patch Mamba with shared/private prototype selection."""
    def __init__(self,channels=22,classes=2,embedding=64,depth=2,*,f1=10,depth_multiplier=3,
                 pool1=8,pool2=8,dropout=0.3,use_fourier=True,rhythm_mode="full",
                 rhythm_num_blocks=None,rhythm_sparsity_threshold=0.01,rhythm_low_ratio=0.5,
                 prototype_confidence=0.8,prototype_consistency=0.0):
        super().__init__(); self.channels=channels; self.embedding=embedding; self.prototype_confidence=prototype_confidence; self.prototype_consistency=prototype_consistency
        self.physiological_tokenizer=MultiScalePhysiologicalTokenizer(channels,embedding,f1,depth_multiplier,pool1,pool2,dropout)
        self.position=nn.Parameter(torch.randn(1,256,embedding)*0.02)
        self.encoder=nn.Sequential(*[FRSMambaBlock(embedding,dropout,use_fourier,rhythm_mode,
                                                  num_blocks=rhythm_num_blocks,
                                                  sparsity_threshold=rhythm_sparsity_threshold,
                                                  low_ratio=rhythm_low_ratio) for _ in range(depth)])
        self.final_norm=nn.LayerNorm(embedding); self.head=nn.Linear(embedding,classes)
        self.prototype_matcher=SharedPrivatePrototypeMatcher(classes)
        self.rhythm_num_blocks=rhythm_num_blocks; self.rhythm_sparsity_threshold=rhythm_sparsity_threshold; self.rhythm_low_ratio=rhythm_low_ratio
    def encode(self,x):
        tokens=self.physiological_tokenizer(x)
        if tokens.shape[1]>self.position.shape[1]: raise ValueError("token count exceeds configured position encoding capacity")
        tokens=tokens*math.sqrt(self.embedding)+self.position[:,:tokens.shape[1]].to(tokens)
        return self.final_norm(self.encoder(tokens))
    def forward(self,x,return_features=False):
        sequence=self.encode(x); features=sequence.mean(1); logits=self.head(features)
        return {"logits":logits,"features":features,"sequence":sequence} if return_features else logits

    def select_pseudo_labels(self,source_features,source_labels,source_domains,target_features,
                             *,target_domain=None,confidence_threshold=0.8,consistency_threshold=0.0):
        return self.prototype_matcher(source_features,source_labels,source_domains,target_features,
                                      target_domain=target_domain,confidence_threshold=confidence_threshold,
                                      consistency_threshold=consistency_threshold)


class SharedPrivatePrototypeMatcher(nn.Module):
    """Filter target pseudo-labels using semantic confidence and shared/private prototypes."""
    def __init__(self,classes):
        super().__init__(); self.classes=classes
    def forward(self,source_features,source_labels,source_domains,target_features,*,target_domain=None,
                confidence_threshold=0.8,consistency_threshold=0.0):
        if source_features.ndim!=2 or target_features.ndim!=2 or source_features.shape[1]!=target_features.shape[1]:
            raise ValueError("source and target embeddings must be 2-D with matching feature size")
        if source_features.shape[0] == 0 or target_features.shape[0] == 0:
            raise ValueError("source and target feature batches must be non-empty")
        if source_labels.shape!=(source_features.shape[0],) or source_domains.shape!=source_labels.shape:
            raise ValueError("source labels and domains must align with source embeddings")
        shared=[]; private=[]; shared_valid=[]; private_valid=[]
        for label in range(self.classes):
            shared_mask=source_labels.eq(label)
            shared_valid.append(shared_mask.any())
            shared.append(source_features[shared_mask].mean(0) if shared_mask.any() else torch.zeros_like(source_features[0]))
            domain_prototypes=[]
            for domain in torch.unique(source_domains):
                if target_domain is not None and torch.equal(domain, torch.as_tensor(target_domain,device=domain.device)):
                    continue
                mask=shared_mask & source_domains.eq(domain)
                if mask.any(): domain_prototypes.append(source_features[mask].mean(0))
            private_valid.append(bool(domain_prototypes))
            private.append(torch.stack(domain_prototypes).mean(0) if domain_prototypes else torch.zeros_like(source_features[0]))
        shared=torch.nn.functional.normalize(torch.stack(shared),dim=-1)
        private=torch.nn.functional.normalize(torch.stack(private),dim=-1)
        target=torch.nn.functional.normalize(target_features,dim=-1)
        shared_score=target@shared.T; private_score=target@private.T
        shared_score[:,~torch.stack(shared_valid)] = -torch.inf
        private_score[:,~torch.tensor(private_valid,device=private_score.device)] = -torch.inf
        shared_label=shared_score.argmax(-1); private_label=private_score.argmax(-1)
        semantic_confidence=torch.softmax(shared_score,dim=-1).amax(-1)
        consistency=(shared_score.gather(1,shared_label[:,None]).squeeze(1)+
                     private_score.gather(1,private_label[:,None]).squeeze(1))*0.5
        consistent=shared_label.eq(private_label) & (consistency>=consistency_threshold)
        accepted=consistent & semantic_confidence.ge(confidence_threshold) & torch.isfinite(consistency)
        return {"labels":shared_label,"mask":accepted,"semantic_confidence":semantic_confidence,
                "shared_private_consistency":consistency}


class DualExpertTransformer(nn.Module):
    def __init__(self,channels,classes,width=32,heads=4):
        super().__init__(); self.channels=channels
        heads=max(h for h in range(min(heads,width),0,-1) if width%h==0)
        def branch():
            layer=nn.TransformerEncoderLayer(width,heads,2*width,batch_first=True,activation="gelu")
            return nn.Sequential(nn.Linear(channels,width),nn.TransformerEncoder(layer,1))
        self.a,self.b=branch(),branch(); self.guide_a=nn.Linear(width,width); self.guide_b=nn.Linear(width,width)
        self.gate=nn.Linear(2*width,2); self.head=nn.Linear(width,classes)
    def forward(self,x,return_gate=False):
        if x.ndim!=3 or x.shape[1]!=self.channels: raise ValueError("invalid EEG shape")
        t=x.transpose(1,2); a,b=self.a(t).mean(1),self.b(t).mean(1)
        a=a*torch.sigmoid(self.guide_a(b)); b=b*torch.sigmoid(self.guide_b(a))
        w=self.gate(torch.cat((a,b),-1)).softmax(-1)
        y=self.head(w[:,:1]*a+w[:,1:]*b); return (y,w) if return_gate else y


class _BidirectionalEEGStateBlock(nn.Module):
    """Small input-dependent diagonal SSM in forward and reverse directions."""
    def __init__(self,width,state_size=16):
        super().__init__(); self.width=width; self.state_size=state_size
        self.norm=nn.LayerNorm(width)
        self.in_proj=nn.Linear(width,2*width)
        self.delta=nn.Linear(width,width)
        self.state_in=nn.Linear(width,width*state_size,bias=False)
        self.state_out=nn.Linear(width*state_size,width,bias=False)
        self.log_decay=nn.Parameter(torch.zeros(width,state_size))
        self.out_proj=nn.Linear(width,width)
    def _scan(self,x):
        batch,length,width=x.shape
        delta=F.softplus(self.delta(x)).clamp(max=5.)
        drive=self.state_in(x).view(batch,length,width,self.state_size)
        decay=torch.exp(-delta.unsqueeze(-1)*F.softplus(self.log_decay).unsqueeze(0).unsqueeze(0))
        state=x.new_zeros((batch,width,self.state_size)); outputs=[]
        for step in range(length):
            state=decay[:,step]*state+drive[:,step]*delta[:,step].unsqueeze(-1)
            outputs.append(self.state_out(state.reshape(batch,-1)))
        return torch.stack(outputs,dim=1)
    def forward(self,x):
        residual=x; x=self.norm(x); content,gate=self.in_proj(x).chunk(2,-1)
        forward=self._scan(content)
        backward=self._scan(content.flip(1)).flip(1)
        return residual+self.out_proj((forward+backward)*torch.sigmoid(gate))


class EEGMamba2024(nn.Module):
    """2024 multi-task EEGMamba: ST-Adaptive, bidirectional SSM, task MoE.

    This is an independent PyTorch reconstruction from arXiv:2407.20254;
    it is distinct from the 2025 EEGMamba foundation-model package.
    """
    def __init__(self,channels,classes,tasks,width=48,experts=4,top_k=2,
                 state_size=16,temporal_kernel=31):
        super().__init__()
        if min(channels,classes,tasks,width,experts,top_k,state_size,temporal_kernel)<1 or classes<2:
            raise ValueError("all dimensions must be positive and classes must be >= 2")
        if top_k>experts: raise ValueError("top_k cannot exceed expert count")
        self.channels=channels; self.width=width; self.tasks=tasks; self.top_k=top_k
        self.spatial_temporal=nn.Sequential(
            nn.Conv2d(1,width,(1,temporal_kernel),padding=(0,temporal_kernel//2),bias=False),
            nn.BatchNorm2d(width),nn.GELU())
        self.class_token=nn.Parameter(torch.zeros(1,1,width))
        self.state_blocks=nn.ModuleList([_BidirectionalEEGStateBlock(width,state_size) for _ in range(2)])
        self.task_embedding=nn.Embedding(tasks,width)
        self.router=nn.Linear(2*width,experts)
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width)) for _ in range(experts)])
        self.universal=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))
        self.classifier=nn.Linear(width,classes)
    def forward(self,x,task_ids=None,return_gates=False):
        if x.ndim!=3 or x.shape[1]!=self.channels: raise ValueError("invalid EEG shape")
        batch=x.shape[0]
        tokens=self.spatial_temporal(x.unsqueeze(1)).mean(dim=2).transpose(1,2)
        token=torch.cat((self.class_token.expand(batch,-1,-1),tokens),dim=1)
        for block in self.state_blocks: token=block(token)
        pooled=token[:,0]
        if task_ids is None: task=torch.zeros(batch,dtype=torch.long,device=x.device)
        else:
            task_ids=torch.as_tensor(task_ids,device=x.device,dtype=torch.long)
            if task_ids.ndim==0: task_ids=task_ids.expand(batch)
            if task_ids.shape!=(batch,) or torch.any((task_ids<0)|(task_ids>=self.tasks)):
                raise ValueError("task_ids must identify one configured task per epoch")
            task=task_ids
        condition=self.task_embedding(task)
        scores=self.router(torch.cat((pooled,condition),dim=-1))
        top_values,top_indices=scores.topk(self.top_k,dim=-1)
        sparse=torch.full_like(scores,-torch.inf).scatter(1,top_indices,top_values).softmax(-1)
        expert_outputs=torch.stack([expert(pooled) for expert in self.experts],dim=1)
        task_output=(sparse.unsqueeze(-1)*expert_outputs).sum(1)
        universal_weight=1.-sparse.max(dim=-1,keepdim=True).values
        features=task_output+universal_weight*self.universal(pooled)
        logits=self.classifier(features)
        return (logits,sparse,universal_weight) if return_gates else logits


class PTSM(nn.Module):
    """Physiology-aware/task-invariant model described in arXiv:2508.11357."""
    def __init__(self, channels, classes, width=32, subject_classes=0,
                 mask_target=0.5, dropout=0.3):
        super().__init__()
        if min(channels, classes, width) < 1 or classes < 2:
            raise ValueError("channels, classes, and width must be positive")
        if subject_classes < 0:
            raise ValueError("subject_classes cannot be negative")
        if not 0<mask_target<1:
            raise ValueError("mask_target must be between zero and one")
        self.channels=channels; self.width=width
        self.mask_target=mask_target
        self.personal_spatial=nn.Sequential(nn.Linear(channels,width),nn.GELU(),nn.Linear(width,channels))
        self.shared_spatial=nn.Sequential(nn.Linear(channels,width),nn.GELU(),nn.Linear(width,channels))
        self.personal_temporal=nn.Conv1d(channels,1,31,padding=15)
        self.shared_temporal=nn.Conv1d(channels,1,31,padding=15)
        self.spatial_fusion_logit=nn.Parameter(torch.zeros(()))
        self.temporal_fusion_logit=nn.Parameter(torch.zeros(()))
        self.temporal_encoder=nn.Sequential(
            nn.Conv1d(channels,width,7,padding=3,bias=False),nn.BatchNorm1d(width),nn.ELU(),nn.Dropout(dropout),
            nn.Conv1d(width,2*width,5,padding=2,bias=False),nn.BatchNorm1d(2*width),nn.ELU(),nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.shared_encoder=nn.Sequential(nn.Linear(2*width,2*width),nn.ELU(),nn.Dropout(dropout),
                                          nn.Linear(2*width,width),nn.LayerNorm(width),nn.ELU())
        self.task_projection=nn.Sequential(nn.Linear(width,width),nn.LayerNorm(width),nn.ELU())
        self.subject_projection=nn.Sequential(nn.Linear(width,width),nn.LayerNorm(width),nn.ELU())
        self.classifier=nn.Linear(width,classes)
        self.subject_classifier=nn.Linear(width,subject_classes) if subject_classes else None

    def forward(self,x,mask=None,return_features=False):
        if x.ndim!=3 or x.shape[1]!=self.channels:
            raise ValueError(f"expected input (batch, {self.channels}, samples)")
        batch,_,samples=x.shape
        if mask is not None:
            if mask.shape not in {(batch,samples),(batch,self.channels,samples)}:
                raise ValueError("mask must have shape (batch, samples) or (batch, channels, samples)")
            x=x.masked_fill(mask.bool() if mask.ndim==3 else mask[:,None,:].bool(),0)
        channel_summary=x.mean(-1)
        personal_s=torch.sigmoid(self.personal_spatial(channel_summary)).unsqueeze(-1)
        shared_s=torch.sigmoid(self.shared_spatial(channel_summary)).unsqueeze(-1)
        personal_t=torch.sigmoid(self.personal_temporal(x))
        shared_t=torch.sigmoid(self.shared_temporal(x))
        spatial_weight=torch.sigmoid(self.spatial_fusion_logit)
        temporal_weight=torch.sigmoid(self.temporal_fusion_logit)
        spatial_mask=spatial_weight*personal_s+(1-spatial_weight)*shared_s
        temporal_mask=temporal_weight*personal_t+(1-temporal_weight)*shared_t
        masked=x*spatial_mask*temporal_mask
        temporal_features=self.temporal_encoder(masked)
        shared=self.shared_encoder(temporal_features)
        task=self.task_projection(shared)
        subject=self.subject_projection(shared)
        logits=self.classifier(task)
        if return_features:
            subject_logits=self.subject_classifier(subject) if self.subject_classifier is not None else None
            return {"logits":logits,"task_features":task,"subject_features":subject,
                    "subject_logits":subject_logits,"shared_features":shared,
                    "spatial_masks":(personal_s,shared_s,spatial_mask),
                    "temporal_masks":(personal_t,shared_t,temporal_mask)}
        return logits

    def training_loss(self,x,task_labels,subject_labels=None,*,
                      contrastive_temperature=0.2,mask_similarity_weight=0.1,
                      mask_sparsity_weight=0.01,mask_size_weight=0.1,
                      orthogonality_weight=0.1,covariance_weight=0.1,
                      information_weight=0.01,sparsity_weight=0.001,
                      contrastive_weight=0.1):
        return ptsm_objective(self(x,return_features=True),task_labels,subject_labels,
                              temperature=contrastive_temperature,
                              mask_target=self.mask_target,
                              mask_similarity_weight=mask_similarity_weight,
                              mask_sparsity_weight=mask_sparsity_weight,
                              mask_size_weight=mask_size_weight,
                              orthogonality_weight=orthogonality_weight,
                              covariance_weight=covariance_weight,
                              information_weight=information_weight,
                              sparsity_weight=sparsity_weight,
                              contrastive_weight=contrastive_weight)


def _supervised_contrastive_loss(features,labels,temperature):
    if labels is None or features.shape[0]<2:
        return features.new_zeros(())
    features=F.normalize(features,dim=-1)
    logits=features@features.T/temperature
    logits=logits-logits.max(dim=-1,keepdim=True).values.detach()
    eye=torch.eye(features.shape[0],dtype=torch.bool,device=features.device)
    positives=labels[:,None].eq(labels[None,:]) & ~eye
    valid=positives.any(-1)
    if not valid.any():
        return features.new_zeros(())
    log_prob=logits-torch.logsumexp(logits.masked_fill(eye,-torch.inf),dim=-1,keepdim=True)
    return -(log_prob.masked_fill(~positives,0).sum(-1)[valid]/positives.sum(-1)[valid]).mean()


def ptsm_objective(outputs,task_labels,subject_labels=None,*,
                   temperature=0.2,mask_target=0.5,
                   mask_similarity_weight=0.1,mask_sparsity_weight=0.01,
                   mask_size_weight=0.1,orthogonality_weight=0.1,
                   covariance_weight=0.1,information_weight=0.01,
                   sparsity_weight=0.001,contrastive_weight=0.1):
    """PTSM task/subject objectives, disentanglement and mask regularizers."""
    if temperature<=0:
        raise ValueError("temperature must be positive")
    task=outputs["task_features"]; subject=outputs["subject_features"]
    task_loss=F.cross_entropy(outputs["logits"],task_labels)
    subject_loss=task.new_zeros(())
    if subject_labels is not None:
        if outputs["subject_logits"] is None:
            raise ValueError("PTSM must be configured with subject_classes for subject supervision")
        subject_loss=F.cross_entropy(outputs["subject_logits"],subject_labels)
    task_unit=F.normalize(task,dim=-1); subject_unit=F.normalize(subject,dim=-1)
    vector_orthogonality=(task_unit*subject_unit).sum(-1).abs().mean()
    task_centered=task-task.mean(0,keepdim=True); subject_centered=subject-subject.mean(0,keepdim=True)
    cross_cov=task_centered.transpose(0,1)@subject_centered/max(1,task.shape[0]-1)
    task_cov=task_centered.T@task_centered/max(1,task.shape[0]-1)
    subject_cov=subject_centered.T@subject_centered/max(1,subject.shape[0]-1)
    covariance_decorrelation=(torch.linalg.matrix_norm(cross_cov)+1e-8)/(
        torch.linalg.matrix_norm(task_cov)*torch.linalg.matrix_norm(subject_cov)+1e-8)
    information_loss=-(torch.trace(task_cov)+torch.trace(subject_cov))
    feature_sparsity=task.abs().mean()+subject.abs().mean()
    task_contrast=_supervised_contrastive_loss(task,task_labels,temperature)
    subject_contrast=_supervised_contrastive_loss(subject,subject_labels,temperature)
    contrastive=(task_contrast+subject_contrast)*0.5
    personal_spatial,shared_spatial,_=outputs["spatial_masks"]
    personal_temporal,shared_temporal,_=outputs["temporal_masks"]
    personal=torch.cat((personal_spatial.flatten(1),personal_temporal.flatten(1)),dim=1)
    common=torch.cat((shared_spatial.flatten(1),shared_temporal.flatten(1)),dim=1)
    mask_similarity=(F.normalize(personal,dim=-1)*F.normalize(common,dim=-1)).sum(-1).mean()
    mask_sparsity=personal.abs().mean()+common.abs().mean()
    mask_size=(personal.mean(-1)-mask_target).abs().mean()+(common.mean(-1)-mask_target).abs().mean()
    total=(task_loss+subject_loss+orthogonality_weight*vector_orthogonality+
           covariance_weight*covariance_decorrelation+information_weight*information_loss+
           sparsity_weight*feature_sparsity+contrastive_weight*contrastive+
           mask_similarity_weight*mask_similarity+mask_sparsity_weight*mask_sparsity+
           mask_size_weight*mask_size)
    return {"loss":total,"task_loss":task_loss,"subject_loss":subject_loss,
            "orthogonality_loss":vector_orthogonality,"covariance_loss":covariance_decorrelation,
            "information_loss":information_loss,"feature_sparsity_loss":feature_sparsity,
            "contrastive_loss":contrastive,"mask_similarity_loss":mask_similarity,
            "mask_sparsity_loss":mask_sparsity,"mask_size_loss":mask_size}


class EEGTriNetPlusPlus(nn.Module):
    """Paper-described spatial/spectral CNN, BiLSTM, Transformer and MAML head."""
    def __init__(self, channels, classes, width=32, patch_size=64, patch_stride=32,
                 lstm_hidden=32, transformer_layers=2, heads=4):
        super().__init__()
        if min(channels, classes, width, patch_size, patch_stride, lstm_hidden) < 1 or classes < 2:
            raise ValueError("invalid EEG-TriNet++ dimensions")
        self.channels = channels
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.spectral_spatial = nn.Sequential(
            nn.Conv2d(1, width, (channels, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(width), nn.GELU(),
            nn.Conv2d(width, width, (1, 5), padding=(0, 2), groups=width, bias=False),
            nn.Conv2d(width, width, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.temporal = nn.LSTM(width, lstm_hidden, batch_first=True,
                                bidirectional=True)
        model_dim = 2 * lstm_hidden
        heads = max(h for h in range(min(heads, model_dim), 0, -1) if model_dim % h == 0)
        layer = nn.TransformerEncoderLayer(model_dim, heads, 4 * model_dim,
                                           batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, transformer_layers)
        self.head = nn.Linear(model_dim, classes)

    def encode(self, x):
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"expected (batch, {self.channels}, samples)")
        if x.shape[-1] < self.patch_size:
            x = F.pad(x, (0, self.patch_size - x.shape[-1]))
        patches = x.unfold(-1, self.patch_size, self.patch_stride).permute(0, 2, 1, 3)
        spectrum = torch.fft.rfft(patches, dim=-1, norm="forward").abs()
        b, count = spectrum.shape[:2]
        tokens = self.spectral_spatial(spectrum.reshape(b * count, 1, self.channels, -1)).flatten(1)
        sequence, _ = self.temporal(tokens.reshape(b, count, -1))
        return self.transformer(sequence)

    def forward(self, x, return_embedding=False):
        embedding = self.encode(x).mean(1)
        return embedding if return_embedding else self.head(embedding)

    def adapt(self, support_x, support_y, *, steps=1, inner_lr=1e-2, create_graph=True):
        """Return differentiably adapted MAML parameters for one support episode."""
        if steps < 1 or inner_lr <= 0:
            raise ValueError("steps and inner_lr must be positive")
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        for _ in range(steps):
            logits = torch.func.functional_call(self, (params, buffers), (support_x,))
            loss = F.cross_entropy(logits, support_y)
            gradients = torch.autograd.grad(loss, tuple(params.values()), create_graph=create_graph)
            params = {name: param - inner_lr * grad
                      for (name, param), grad in zip(params.items(), gradients)}
        return params


def maml_episode_loss(model, support_x, support_y, query_x, query_y, *, steps=1,
                      inner_lr=1e-2, create_graph=True):
    """MAML support adaptation followed by query cross-entropy for meta-training."""
    params = model.adapt(support_x, support_y, steps=steps, inner_lr=inner_lr,
                         create_graph=create_graph)
    logits = torch.func.functional_call(model, (params, dict(model.named_buffers())), (query_x,))
    return F.cross_entropy(logits, query_y)


class SPDTokenTransformer(nn.Module):
    """Unified BWSPD/Log-Euclidean/Euclidean Transformer from arXiv:2601.21521.

    Input may be raw ``(B,C,T)`` epochs, which are converted to overlapping
    window covariances, or precomputed ``(B,tokens,C,C)`` SPD covariances. The
    latter is the required contract for the paper's multi-band tokenization.
    """
    def __init__(self,channels,classes,width=64,eps=1e-6,patch_size=438,
                 patch_stride=None,embedding="logeuclidean",bn_embed=True,
                 depth=4,heads=8,feedforward=512,max_tokens=512):
        super().__init__()
        if embedding not in {"bwspd","logeuclidean","euclidean"}:
            raise ValueError("embedding must be bwspd, logeuclidean, or euclidean")
        if min(channels,classes,width,patch_size,depth,max_tokens)<1 or classes<2 or eps<=0:
            raise ValueError("invalid SPD Transformer dimensions or epsilon")
        heads=max(h for h in range(min(heads,width),0,-1) if width%h==0)
        self.channels=channels; self.eps=eps; self.patch_size=patch_size
        self.patch_stride=patch_stride or patch_size; self.embedding=embedding
        self.max_tokens=max_tokens
        self.proj=nn.Linear(channels*(channels+1)//2,width)
        self.position=nn.Parameter(torch.zeros(1,max_tokens,width))
        self.bn_embed=nn.BatchNorm1d(width) if bn_embed else nn.Identity()
        layer=nn.TransformerEncoderLayer(width,heads,feedforward,batch_first=True,
                                         activation="gelu",norm_first=False)
        self.enc=nn.TransformerEncoder(layer,depth)
        self.head=nn.Linear(width,classes)
        upper=torch.triu_indices(channels,channels)
        self.register_buffer("upper_i",upper[0],persistent=False)
        self.register_buffer("upper_j",upper[1],persistent=False)

    def _covariances(self,x):
        if x.ndim==4:
            if tuple(x.shape[-2:])!=(self.channels,self.channels):
                raise ValueError("precomputed covariance input must have shape (batch, tokens, channels, channels)")
            return (x+x.transpose(-1,-2))*0.5
        if x.ndim!=3 or x.shape[1]!=self.channels:
            raise ValueError(f"expected raw EEG (batch, {self.channels}, samples) or SPD covariance tokens")
        if x.shape[-1]<self.patch_size:
            x=F.pad(x,(0,self.patch_size-x.shape[-1]))
        patches=x.unfold(-1,self.patch_size,self.patch_stride).transpose(1,2)
        patches=patches-patches.mean(-1,keepdim=True)
        cov=patches@patches.transpose(-1,-2)/max(1,self.patch_size-1)
        eye=torch.eye(self.channels,device=x.device,dtype=x.dtype)
        return cov+self.eps*eye

    def _embed_spd(self,cov):
        cov=((cov+cov.transpose(-1,-2))*0.5).float()
        if self.embedding=="bwspd":
            values,vectors=torch.linalg.eigh(cov)
            values=values.clamp_min(self.eps)
            matrix=(vectors*values.sqrt().unsqueeze(-2))@vectors.transpose(-1,-2)
        elif self.embedding=="logeuclidean":
            values,vectors=torch.linalg.eigh(cov)
            values=values.clamp_min(self.eps)
            matrix=(vectors*values.log().unsqueeze(-2))@vectors.transpose(-1,-2)
        else:
            matrix=cov
        return matrix[...,self.upper_i,self.upper_j]

    def forward(self,x,return_tokens=False):
        cov=self._covariances(x)
        tokens=self.proj(self._embed_spd(cov).to(self.proj.weight.dtype))
        if tokens.shape[1]>self.max_tokens:
            raise ValueError("SPD token sequence exceeds maximum position capacity")
        tokens=tokens+self.position[:,:tokens.shape[1]].to(tokens)
        tokens=self.bn_embed(tokens.transpose(1,2)).transpose(1,2)
        encoded=self.enc(tokens)
        logits=self.head(encoded.mean(1))
        return {"logits":logits,"tokens":encoded,"covariances":cov} if return_tokens else logits


class AGTCNet(nn.Module):
    def __init__(self,channels,classes,width=16,adjacency=None):
        super().__init__(); self.channels=channels
        if adjacency is not None and torch.as_tensor(adjacency).shape!=(channels,channels): raise ValueError("adjacency must be square over channels")
        self.register_buffer("adjacency_bias",None if adjacency is None else torch.as_tensor(adjacency,dtype=torch.float32))
        self.node_projection=nn.Linear(2,width); self.query=nn.Linear(width,width,bias=False); self.key=nn.Linear(width,width,bias=False)
        self.graph=nn.Conv1d(channels,width,1)
        self.temporal_branches=nn.ModuleList([nn.Sequential(nn.Conv1d(width,width,k,padding=k//2),nn.BatchNorm1d(width),nn.GELU()) for k in (7,15,31)])
        self.temporal_fusion=nn.Sequential(nn.Conv1d(width*3,width,1),nn.BatchNorm1d(width),nn.GELU())
        self.head=nn.Linear(width,classes)
    def forward(self,x):
        if x.ndim!=3 or x.shape[1]!=self.channels: raise ValueError("invalid EEG shape")
        node_stats=torch.stack((x.mean(-1),x.std(-1,unbiased=False)),-1)
        nodes=self.node_projection(node_stats); q=self.query(nodes); k=self.key(nodes)
        scores=q@k.transpose(1,2)/(q.shape[-1]**0.5)
        if self.adjacency_bias is not None: scores=scores+self.adjacency_bias.to(scores)
        adjacency=scores.softmax(-1)
        graph_signal=torch.bmm(adjacency,x)
        spatial=self.graph(graph_signal)
        temporal=self.temporal_fusion(torch.cat([branch(spatial) for branch in self.temporal_branches],1))
        return self.head(temporal.mean(-1))


class GeometryEEGNet(nn.Module):
    def __init__(self,channels,classes,rank=16,eps=1e-4):
        super().__init__(); self.channels=channels; self.rank=min(rank,channels); self.eps=eps
        self.proj=nn.Parameter(torch.randn(channels,self.rank)/channels**0.5); self.head=nn.Linear(self.rank*(self.rank+1)//2,classes)
    def forward(self,x):
        if x.ndim!=3 or x.shape[1]!=self.channels: raise ValueError("invalid EEG shape")
        z=self.proj.T@(x-x.mean(-1,keepdim=True)); c=z@z.transpose(1,2)/max(1,x.shape[-1]-1)+self.eps*torch.eye(self.rank,device=x.device,dtype=x.dtype)
        v,q=torch.linalg.eigh(c.float()); g=(q*v.clamp_min(self.eps).log().unsqueeze(1))@q.transpose(1,2)
        i,j=torch.triu_indices(self.rank,self.rank,device=x.device); return self.head(g.to(x.dtype)[:,i,j])


class PhaseSpaceSPDNet(nn.Module):
    """Phase-space augmented covariance followed by the Phase-SPDNet stack.

    Raw epochs are converted into a phase-space covariance using Takens-style
    delayed channel vectors. Precomputed covariance tensors can also be passed
    directly when the caller uses the MOABB preprocessing pipeline.
    """
    def __init__(self,channels,classes,delay=2,rank=None,order=2,eps=1e-4,
                 batch_norm=False):
        super().__init__()
        if min(channels,classes,delay,order)<1 or classes<2:
            raise ValueError("channels, delay, order, and classes must be positive")
        self.channels=channels; self.delay=delay; self.order=order; self.eps=eps
        self.input_dim=channels*(order+1)
        self.rank=max(1,min(self.input_dim//2,rank or self.input_dim//2))
        self.bimap=nn.Parameter(torch.empty(self.input_dim,self.rank,dtype=torch.float64))
        nn.init.orthogonal_(self.bimap)
        self.batch_norm=nn.BatchNorm1d(self.rank*(self.rank+1)//2,dtype=torch.float64) if batch_norm else None
        self.classifier=nn.Linear(self.rank*(self.rank+1)//2,classes,dtype=torch.float64)

    def phase_covariance(self,x):
        if x.ndim!=3 or x.shape[1]!=self.channels:
            raise ValueError(f"expected raw EEG shaped (batch, {self.channels}, samples)")
        offsets=torch.arange(self.order+1,device=x.device)*self.delay
        needed=int(offsets[-1])+1
        if x.shape[-1]<needed:
            x=F.pad(x,(0,needed-x.shape[-1]))
        taps=[x[...,int(offset):x.shape[-1]-int(offsets[-1]-offset) if int(offsets[-1]-offset)>0 else None]
              for offset in offsets]
        phase=torch.cat(taps,dim=1)
        phase=phase-phase.mean(-1,keepdim=True)
        return phase@phase.transpose(1,2)/max(1,phase.shape[-1]-1)

    def forward(self,x,return_features=False):
        if x.ndim==3:
            cov=self.phase_covariance(x)
        elif x.ndim==4 and x.shape[-1]==x.shape[-2] and x.shape[1]==1:
            cov=x[:,0]
            if cov.shape[-1]!=self.input_dim:
                raise ValueError(f"expected covariance dimension {self.input_dim}")
        else:
            raise ValueError("expected raw EEG or (batch, 1, phase_dim, phase_dim) covariance")
        cov=(cov+cov.transpose(-1,-2))*0.5
        eye=torch.eye(self.input_dim,device=cov.device,dtype=cov.dtype)
        cov=cov+eye*self.eps
        w=self.bimap.to(cov)
        mapped=w.T@cov@w
        values,vectors=torch.linalg.eigh((mapped+mapped.transpose(-1,-2)).double())
        mapped=(vectors*values.clamp_min(self.eps).unsqueeze(-2))@vectors.transpose(-1,-2)
        eigenvalues,eigenvectors=torch.linalg.eigh(mapped)
        log_cov=(eigenvectors*eigenvalues.clamp_min(self.eps).log().unsqueeze(-2))@eigenvectors.transpose(-1,-2)
        tri=torch.triu_indices(self.rank,self.rank,device=cov.device)
        features=log_cov[:,tri[0],tri[1]]
        if self.batch_norm is not None and features.shape[0]>1:
            features=self.batch_norm(features)
        logits=self.classifier(features)
        return (logits,features) if return_features else logits


def geometric_consistency_loss(a,b):
    if a.shape!=b.shape or a.ndim!=3: raise ValueError("views must have equal batched EEG shapes")
    def logcov(x):
        x=x-x.mean(-1,keepdim=True); c=x@x.transpose(1,2)/max(1,x.shape[-1]-1); v,q=torch.linalg.eigh(((c+c.transpose(1,2))/2).float())
        return (q*v.clamp_min(1e-5).log().unsqueeze(1))@q.transpose(1,2)
    return (logcov(a)-logcov(b)).square().mean()


def _multi_gaussian_kernel(a,b,bandwidths):
    squared_distance=torch.cdist(a.float(),b.float()).square()
    kernels=[torch.exp(-squared_distance/(2*float(sigma)**2)) for sigma in bandwidths]
    return torch.stack(kernels).mean(0)


def cs_divergence_loss(source,target,sigma=None,*,bandwidths=(5.,6.,7.,8.)):
    """Kernel Cauchy-Schwarz divergence estimator used by BFM-MSDA."""
    if source.ndim!=2 or target.ndim!=2 or source.shape[1]!=target.shape[1]:
        raise ValueError("feature dimensions must match")
    if source.shape[0]==0 or target.shape[0]==0:
        raise ValueError("CS divergence needs non-empty source and target batches")
    scales=(float(sigma),) if sigma is not None else bandwidths
    if not scales or any(float(scale)<=0 for scale in scales):
        raise ValueError("kernel bandwidths must be positive")
    kss=_multi_gaussian_kernel(source,source,scales).mean().clamp_min(1e-8)
    ktt=_multi_gaussian_kernel(target,target,scales).mean().clamp_min(1e-8)
    kst=_multi_gaussian_kernel(source,target,scales).mean().clamp_min(1e-8)
    return -2*kst.log()+kss.log()+ktt.log()


def conditional_cs_divergence(source_features,target_features,source_probabilities,
                              target_probabilities,*,feature_bandwidths=(5.,6.,7.,8.),
                              probability_bandwidths=(5.,6.,7.,8.)):
    """Conditional CS divergence over joint feature/prediction KDEs."""
    if source_features.ndim!=2 or target_features.ndim!=2 or source_features.shape[1]!=target_features.shape[1]:
        raise ValueError("source and target features must be 2-D with matching dimensions")
    if source_probabilities.shape[0]!=source_features.shape[0] or target_probabilities.shape[0]!=target_features.shape[0]:
        raise ValueError("probabilities must align with their feature batches")
    if source_probabilities.shape[1]!=target_probabilities.shape[1]:
        raise ValueError("source and target class dimensions must match")
    kss=_multi_gaussian_kernel(source_features,source_features,feature_bandwidths)
    ktt=_multi_gaussian_kernel(target_features,target_features,feature_bandwidths)
    kst=_multi_gaussian_kernel(source_features,target_features,feature_bandwidths)
    kts=kst.transpose(0,1)
    lss=_multi_gaussian_kernel(source_probabilities,source_probabilities,probability_bandwidths)
    ltt=_multi_gaussian_kernel(target_probabilities,target_probabilities,probability_bandwidths)
    lst=_multi_gaussian_kernel(source_probabilities,target_probabilities,probability_bandwidths)
    lts=lst.transpose(0,1)
    eps=1e-8
    self_s=((kss*lss).sum(-1)/(kss.sum(-1).square()+eps)).sum().clamp_min(eps)
    self_t=((ktt*ltt).sum(-1)/(ktt.sum(-1).square()+eps)).sum().clamp_min(eps)
    cross_st=((kst*lst).sum(-1)/(kss.sum(-1)*kst.sum(-1)+eps)).sum().clamp_min(eps)
    cross_ts=((kts*lts).sum(-1)/(ktt.sum(-1)*kts.sum(-1)+eps)).sum().clamp_min(eps)
    cs_st=-2*cross_st.log()+self_s.log()+self_t.log()
    cs_ts=-2*cross_ts.log()+self_s.log()+self_t.log()
    return (cs_st+cs_ts)*0.5


class BFMMSDA(nn.Module):
    """BFM-guided source selection plus feature-CS and decision-CCS alignment.

    Foundation embeddings are supplied separately from the task encoder, so a
    frozen LaBraM/other verified BFM can rank source subjects without being
    coupled to the downstream classifier implementation.
    """
    def __init__(self,encoder,feature_dim,classes,*,top_k=3,temperature=0.1,
                 alpha=0.4,beta=1.4,warm_up=0):
        super().__init__()
        if top_k<1 or temperature<=0 or alpha<0 or beta<0 or warm_up<0:
            raise ValueError("invalid BFM-MSDA hyperparameters")
        self.encoder=encoder; self.top_k=top_k; self.temperature=temperature
        self.alpha=alpha; self.beta=beta; self.warm_up=warm_up
        self.classifier=nn.Linear(feature_dim,classes)
    def encode(self,x):
        features=self.encoder(x)
        if isinstance(features,(tuple,list)): features=features[0]
        if isinstance(features,dict): features=features.get("features",features.get("logits"))
        if features is None: raise ValueError("encoder must return a feature tensor")
        if features.ndim>2: features=features.flatten(1)
        if features.ndim!=2: raise ValueError("encoder features must reduce to a matrix")
        return features
    def forward(self,x,return_features=False):
        features=self.encode(x); logits=self.classifier(features)
        return {"logits":logits,"features":features} if return_features else logits
    def select_sources(self,source_bfm_embeddings,source_domains,target_bfm_embeddings):
        if source_bfm_embeddings.ndim!=2 or target_bfm_embeddings.ndim!=2 or source_bfm_embeddings.shape[1]!=target_bfm_embeddings.shape[1]:
            raise ValueError("BFM embeddings must be matrices with matching feature dimensions")
        if source_domains.shape!=(source_bfm_embeddings.shape[0],):
            raise ValueError("source_domains must align with source BFM embeddings")
        domains=torch.unique(source_domains,sorted=True)
        if domains.numel()==0: raise ValueError("at least one source domain is required")
        target=F.normalize(target_bfm_embeddings.mean(0),dim=0)
        source_means=torch.stack([source_bfm_embeddings[source_domains.eq(d)].mean(0) for d in domains])
        similarities=F.normalize(source_means,dim=-1)@target
        count=min(self.top_k,domains.numel()); selected=similarities.topk(count).indices
        chosen_domains=domains[selected]; chosen_scores=similarities[selected]
        weights=torch.softmax(chosen_scores/self.temperature,dim=0)
        return {"domains":chosen_domains,"similarities":chosen_scores,"weights":weights}
    def adaptation_loss(self,source_x,source_labels,source_domains,target_x,
                        source_bfm_embeddings,target_bfm_embeddings,*,epoch=None):
        if source_x.shape[0]!=source_labels.shape[0] or source_x.shape[0]!=source_domains.shape[0]:
            raise ValueError("source inputs, labels, and domains must align")
        if source_bfm_embeddings.shape[0]!=source_x.shape[0]:
            raise ValueError("source BFM embeddings must align with source examples")
        selection=self.select_sources(source_bfm_embeddings,source_domains,target_bfm_embeddings)
        target_features=self.encode(target_x)
        target_probabilities=self.classifier(target_features).softmax(-1)
        cls_losses=[]; feature_losses=[]; decision_losses=[]
        for domain in selection["domains"]:
            mask=source_domains.eq(domain); features=self.encode(source_x[mask])
            logits=self.classifier(features); probabilities=logits.softmax(-1)
            cls_losses.append(F.cross_entropy(logits,source_labels[mask]))
            feature_losses.append(cs_divergence_loss(features,target_features))
            decision_losses.append(conditional_cs_divergence(features,target_features,
                                                              probabilities,target_probabilities))
        weights=selection["weights"]
        cls_loss=(weights*torch.stack(cls_losses)).sum()
        cs_loss=(weights*torch.stack(feature_losses)).sum()
        ccs_loss=(weights*torch.stack(decision_losses)).sum()
        active=epoch is None or epoch>=self.warm_up
        loss=cls_loss+(self.alpha*cs_loss+self.beta*ccs_loss if active else 0.0)
        return {"loss":loss,"classification_loss":cls_loss,"cs_loss":cs_loss,
                "ccs_loss":ccs_loss,"selection":selection,"alignment_active":active}


class SubjectAlignedEncoder(nn.Module):
    """Subject-specific encoders with a shared classifier, as in Lopes et al. (2026)."""
    def __init__(self,channels,classes,subjects,width=32,*,adapter_rank=0):
        super().__init__(); self.channels=channels; self.subjects=subjects; self.adapter_rank=adapter_rank
        def make_encoder():
            return nn.Sequential(nn.Conv1d(channels,width,15,padding=7),nn.BatchNorm1d(width),nn.ELU(),
                                 nn.Conv1d(width,width,7,padding=3),nn.BatchNorm1d(width),nn.ELU())
        self.encoders=nn.ModuleList([make_encoder() for _ in range(subjects)])
        self.head=nn.Linear(width,classes)
        if adapter_rank:
            if adapter_rank < 1 or adapter_rank > channels: raise ValueError("adapter_rank must be in [1, channels]")
            self.adapter_left=nn.Parameter(torch.zeros(subjects,channels,adapter_rank))
            self.adapter_right=nn.Parameter(torch.zeros(subjects,adapter_rank,channels))
            nn.init.normal_(self.adapter_left,std=channels**-0.5)
            nn.init.normal_(self.adapter_right,std=channels**-0.5)
        else:
            self.register_parameter("adapter_left",None); self.register_parameter("adapter_right",None)
    def forward(self,x,subject_ids=None):
        if x.ndim!=3 or x.shape[1]!=self.channels: raise ValueError(f"expected (batch, {self.channels}, samples)")
        if subject_ids is None:
            if self.subjects != 1: raise ValueError("subject_ids are required for subject-specific encoders")
            subject_ids=torch.zeros(x.shape[0],dtype=torch.long,device=x.device)
        if subject_ids.shape!=(x.shape[0],) or subject_ids.dtype not in (torch.int32,torch.int64):
            raise ValueError("subject_ids must be an integer vector of shape (batch,)")
        if torch.any(subject_ids<0) or torch.any(subject_ids>=self.subjects): raise ValueError("subject id outside configured encoder bank")
        logits=x.new_empty((x.shape[0],self.head.out_features))
        for subject in torch.unique(subject_ids):
            idx=subject_ids.eq(subject); eeg=x[idx]
            if self.adapter_rank:
                left=self.adapter_left[subject].to(eeg); right=self.adapter_right[subject].to(eeg)
                eeg=eeg+torch.einsum("rc,brt->bct",right,torch.einsum("cr,bct->brt",left,eeg))
            features=self.encoders[int(subject)](eeg).mean(-1)
            logits[idx]=self.head(features)
        return logits


class _BiMap(nn.Module):
    """Per-feature Stiefel BiMap used by the Tensor/Graph CSPNet papers."""

    def __init__(self, features, input_dim, output_dim):
        super().__init__()
        self.features = features
        self.input_dim = input_dim
        self.output_dim = output_dim
        work_dim = max(input_dim, output_dim)
        self.work_dim = work_dim
        self.weight = nn.Parameter(torch.empty(features, work_dim, output_dim, dtype=torch.float64))
        with torch.no_grad():
            for index in range(features):
                q, _ = torch.linalg.qr(torch.randn(work_dim, output_dim, dtype=torch.float64))
                self.weight[index].copy_(q[:, :output_dim])

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.features or x.shape[-2:] != (self.input_dim, self.input_dim):
            raise ValueError("BiMap expects (batch, features, input_dim, input_dim)")
        if self.work_dim != self.input_dim:
            pad = self.work_dim - self.input_dim
            x = F.pad(x, (0, pad, 0, pad))
            if pad:
                eye = torch.eye(self.work_dim, device=x.device, dtype=x.dtype)
                eye[-pad:, -pad:] = 0
                eye[-pad:, -pad:] = torch.eye(pad, device=x.device, dtype=x.dtype)
                x = x + eye[None, None]
        w = self.weight.to(x)
        return torch.einsum("hni,bhnm,hmj->bhij", w, x, w)


class _ReEig(nn.Module):
    def __init__(self, floor=1e-4):
        super().__init__(); self.floor = floor
    def forward(self, x):
        vals, vecs = torch.linalg.eigh(((x + x.transpose(-1, -2)) * 0.5).float())
        return ((vecs * vals.clamp_min(self.floor).unsqueeze(-2)) @ vecs.transpose(-1, -2)).to(x.dtype)


class _LogEig(nn.Module):
    def forward(self, x):
        vals, vecs = torch.linalg.eigh(((x + x.transpose(-1, -2)) * 0.5).float())
        return ((vecs * vals.clamp_min(1e-6).log().unsqueeze(-2)) @ vecs.transpose(-1, -2)).to(x.dtype)


class _BatchNormSPD(nn.Module):
    """Log-Euclidean SPD normalization with train-fold running center."""
    def __init__(self, features, dim, momentum=0.1):
        super().__init__(); self.momentum = momentum
        self.register_buffer("running_log_mean", torch.zeros(features, dim, dim, dtype=torch.float64))
        self.weight_factor = nn.Parameter(torch.eye(dim, dtype=torch.float64).expand(features, -1, -1).clone())
        self.logeig = _LogEig()
    def forward(self, x):
        logx = self.logeig(x)
        if self.training:
            mean = logx.mean(dim=(0,))
            with torch.no_grad(): self.running_log_mean.lerp_(mean.detach(), self.momentum)
        else:
            mean = self.running_log_mean
        values, vectors = torch.linalg.eigh(((mean + mean.transpose(-1, -2)) * 0.5).float())
        inv_root = (vectors * values.mul(-0.5).exp().unsqueeze(-2)) @ vectors.transpose(-1, -2)
        centered = torch.einsum("hni,bhnm,hmj->bhij", inv_root.to(x), x, inv_root.to(x))
        factor = self.weight_factor.to(x)
        affine = factor @ factor.transpose(-1, -2)
        affine_values, affine_vectors = torch.linalg.eigh(affine.float())
        root = (affine_vectors * affine_values.clamp_min(1e-6).sqrt().unsqueeze(-2)) @ affine_vectors.transpose(-1, -2)
        return torch.einsum("hni,bhnm,hmj->bhij", root.to(x), centered, root.to(x))


class TensorCSPNetReference(nn.Module):
    """Tensor-CSPNet architecture: temporal/frequency BiMaps then temporal CNN.

    Input is the paper's covariance tensor ``(B, windows, bands, C, C)``.
    The default 2 windows, 9 bands and 22 channels match its BCIC setting.
    """
    def __init__(self, channels=22, classes=4, *, windows=2, bands=9,
                 dims=(22, 36, 36, 22), tcn_channels=16, mlp=False):
        super().__init__()
        if dims[0] != channels or dims[-1] != channels:
            raise ValueError("Tensor-CSPNet dims must start and end at channels")
        self.channels, self.windows, self.bands = channels, windows, bands
        feature_count = windows * bands
        self.maps = nn.ModuleList([
            _BiMap(feature_count, dims[0], dims[1]),
            _BiMap(feature_count, dims[1], dims[2]),
            _BiMap(feature_count, dims[2], dims[3]),
        ])
        self.norm = _BatchNormSPD(feature_count, dims[-1])
        self.reeig = _ReEig()
        self.logeig = _LogEig()
        self.temporal = nn.Conv2d(1, tcn_channels, (2, bands * channels * channels),
                                  stride=(1, channels * channels), dtype=torch.float64)
        in_features = tcn_channels
        if mlp:
            self.head = nn.Sequential(nn.Linear(in_features, in_features), nn.ReLU(),
                                      nn.Linear(in_features, in_features), nn.ReLU(),
                                      nn.Linear(in_features, classes)).double()
        else:
            self.head = nn.Linear(in_features, classes).double()

    def forward(self, x):
        expected = (self.windows, self.bands, self.channels, self.channels)
        if x.ndim != 5 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"expected (batch, {expected}) covariance tensor")
        b = x.shape[0]
        x = x.reshape(b, self.windows * self.bands, self.channels, self.channels).double()
        for layer in self.maps:
            x = self.reeig(layer(x))
        x = self.logeig(self.norm(x))
        x = x.reshape(b, 1, self.windows, self.bands * self.channels * self.channels)
        return self.head(self.temporal(x).flatten(1))


class GraphCSPNetReference(nn.Module):
    """Graph-CSPNet graph BiMap plus SPD log readout.

    Input is the paper's segment covariance tensor ``(B, segments, C, C)``;
    ``graph_matrix`` is the segment graph used by the first graph-BiMap layer.
    """
    def __init__(self, channels=22, classes=4, *, segments=4, graph_matrix=None,
                 dims=(22, 36, 36, 22), mlp=False):
        super().__init__()
        if dims[0] != channels or dims[-1] != channels:
            raise ValueError("Graph-CSPNet dims must start and end at channels")
        self.channels, self.segments = channels, segments
        graph = torch.eye(segments, dtype=torch.float64) if graph_matrix is None else torch.as_tensor(graph_matrix, dtype=torch.float64)
        if graph.shape != (segments, segments):
            raise ValueError("graph_matrix must have shape (segments, segments)")
        self.register_buffer("graph_matrix", graph)
        self.maps = nn.ModuleList([
            _BiMap(segments, dims[0], dims[1]),
            _BiMap(segments, dims[1], dims[2]),
            _BiMap(segments, dims[2], dims[3]),
        ])
        self.norm = _BatchNormSPD(segments, dims[-1])
        self.reeig, self.logeig = _ReEig(), _LogEig()
        features = segments * channels * channels
        if mlp:
            self.head = nn.Sequential(nn.Linear(features, channels), nn.ReLU(), nn.Linear(channels, channels),
                                      nn.ReLU(), nn.Linear(channels, classes)).double()
        else:
            self.head = nn.Linear(features, classes).double()

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.segments, self.channels, self.channels):
            raise ValueError("expected (batch, segments, channels, channels) covariance tensor")
        x = x.double()
        adjacency = self.graph_matrix.to(device=x.device, dtype=x.dtype)
        x = torch.einsum("ij,bjmn->bimn", adjacency, x)
        for layer in self.maps:
            x = self.reeig(layer(x))
        x = self.logeig(self.norm(x))
        return self.head(x.flatten(1))


class _DCTBlock(nn.Module):
    """Learned congruence map W^T X W with SPD eigenvalue rectification."""
    def __init__(self,in_dim,out_dim,eps=1e-5):
        super().__init__(); self.weight=nn.Parameter(torch.empty(in_dim,out_dim)); self.eps=eps
        nn.init.orthogonal_(self.weight)
    def forward(self,x):
        w=self.weight.to(x)
        y=w.transpose(-1,-2)@x@w
        y=(y+y.transpose(-1,-2))*0.5
        values,vectors=torch.linalg.eigh(y.float())
        return (vectors*values.clamp_min(self.eps).unsqueeze(-2))@vectors.transpose(-1,-2)


class GADCN(nn.Module):
    """DCT, DLDCT, or DDCT-UNet congruence model from arXiv:2511.18940."""
    def __init__(self,channels,classes,rank=16,width=64,eps=1e-4,variant="ddct_unet"):
        super().__init__()
        if variant not in {"dct","dldct","ddct_unet"}:
            raise ValueError("variant must be dct, dldct, or ddct_unet")
        self.channels=channels; self.rank=min(rank,channels); self.eps=eps; self.variant=variant
        self.input_projection=nn.Parameter(torch.empty(channels,self.rank))
        nn.init.orthogonal_(self.input_projection)
        if variant=="dct":
            dims=(self.rank,max(2,self.rank//2))
        elif variant=="dldct":
            dims=(self.rank,max(2,3*self.rank//4),max(2,self.rank//2))
        else:
            dims=(self.rank,max(2,3*self.rank//4),max(2,self.rank//2),max(2,self.rank//4))
        self.encoder=nn.ModuleList([_DCTBlock(a,b,eps) for a,b in zip(dims,dims[1:])])
        self.decoder=nn.ModuleList([_DCTBlock(b,a,eps) for a,b in reversed(list(zip(dims,dims[1:])) )]) if variant=="ddct_unet" else nn.ModuleList()
        final_dim=dims[0] if variant=="ddct_unet" else dims[-1]
        tangent=final_dim*(final_dim+1)//2
        self.fisher_head=nn.Sequential(nn.Linear(tangent,width),nn.GELU(),nn.Linear(width,classes))
    def encode_covariance(self,x):
        if x.ndim==3:
            if x.shape[1]!=self.channels: raise ValueError("invalid EEG channel count")
            z=self.input_projection.to(x).T@(x-x.mean(-1,keepdim=True))
            x=z@z.transpose(1,2)/max(1,z.shape[-1]-1)
        if x.ndim!=3 or x.shape[-1]!=x.shape[-2]:
            raise ValueError("expected raw EEG (batch, channels, samples) or covariance (batch, channels, channels)")
        if x.shape[-1]!=self.rank:
            if x.shape[-1]!=self.channels: raise ValueError("covariance dimension must match channels or rank")
            p=self.input_projection.to(x); x=p.T@x@p
        x=(x+x.transpose(-1,-2))*0.5+torch.eye(self.rank,device=x.device,dtype=x.dtype)*self.eps
        skips=[]
        for block in self.encoder:
            skips.append(x); x=block(x)
        if self.decoder:
            for index,block in enumerate(self.decoder):
                x=block(x)
                skip=skips[-2-index] if index < len(skips)-1 else None
                if skip is not None and skip.shape==x.shape:
                    x=x+skip
            values,vectors=torch.linalg.eigh(x.float())
            x=(vectors*values.clamp_min(self.eps).log().unsqueeze(-2))@vectors.transpose(-1,-2)
        else:
            values,vectors=torch.linalg.eigh(x.float())
            x=(vectors*values.clamp_min(self.eps).log().unsqueeze(-2))@vectors.transpose(-1,-2)
        i,j=torch.triu_indices(x.shape[-1],x.shape[-1],device=x.device)
        return x.to(dtype=self.fisher_head[0].weight.dtype)[:,i,j]
    def forward(self,x): return self.fisher_head(self.encode_covariance(x))
    def consistency_loss(self,view_a,view_b): return geometric_consistency_loss(view_a,view_b)


class TensorCSPNet(nn.Module):
    """Adapter for the licensed Tensor-CSPNet author model implementation."""
    def __init__(self, channels=22, classes=4, *, windows=2, bands=9, mlp=False):
        super().__init__()
        if (channels, classes) not in {(22, 4), (20, 2)}:
            raise ValueError("author Tensor-CSPNet implementation supports BCIC (22,4) or KU (20,2)")
        if windows != 2 or bands != 9:
            raise ValueError("author Tensor-CSPNet experiment configuration uses 2 windows and 9 frequency bands")
        dataset = "BCIC" if channels == 22 else "KU"
        self.channels, self.windows, self.bands = channels, windows, bands
        _, Tensor_CSPNet_Basic, _ = _load_geometric_csp_models()
        self.model = Tensor_CSPNet_Basic(channel_num=windows * bands, mlp=mlp, dataset=dataset)
    def forward(self, x):
        expected = (self.windows, self.bands, self.channels, self.channels)
        if x.ndim != 5 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"expected (batch, {expected}) covariance tensor")
        return self.model(x.double())


class GraphCSPNet(nn.Module):
    """Adapter for the licensed Graph-CSPNet author model implementation."""
    def __init__(self, channels=22, classes=4, *, segments=4, graph_matrix=None, mlp=False):
        super().__init__()
        if (channels, classes) not in {(22, 4), (20, 2)}:
            raise ValueError("author Graph-CSPNet implementation supports BCIC (22,4) or KU (20,2)")
        dataset = "BCIC" if channels == 22 else "KU"
        graph = torch.eye(segments, dtype=torch.float64) if graph_matrix is None else torch.as_tensor(graph_matrix, dtype=torch.float64)
        if graph.shape != (segments, segments):
            raise ValueError("graph_matrix must have shape (segments, segments)")
        self.channels, self.segments = channels, segments
        Graph_CSPNet_Basic, _, _ = _load_geometric_csp_models()
        self.model = Graph_CSPNet_Basic(channel_num=segments, P=graph, mlp=mlp, dataset=dataset)
    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.segments, self.channels, self.channels):
            raise ValueError("expected (batch, segments, channels, channels) covariance tensor")
        x = x.double()
        _, _, Graph_BiMap = _load_geometric_csp_models()
        for layer in self.model.modules():
            if isinstance(layer, Graph_BiMap):
                layer._P = layer._P.to(device=x.device, dtype=x.dtype)
        return self.model(x)
