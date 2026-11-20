"""BrainOmni LoRA with shared A/task-specific B; task IDs are known dataset IDs."""
from __future__ import annotations
import copy
import torch
from torch import nn


class BrainOmniMultiTask(nn.Module):
    """Two-task adapter controls using maintained PEFT layers.

    Heads and adapter B matrices remain task-specific. With share_a=False,
    both A/B are task-specific (the matched joint-training control).
    No subject identity or target labels are used to route adapters.
    """
    def __init__(self, population, *, share_a):
        super().__init__()
        from peft import LoraConfig
        self.backbone = population.backbone
        self.heads = nn.ModuleList([population.class_head, copy.deepcopy(population.class_head)])
        source = self.backbone.peft_config['default']
        cfg = LoraConfig(r=source.r, lora_alpha=source.lora_alpha,
            lora_dropout=0., bias='none', target_modules=source.target_modules)
        self.backbone.add_adapter('task1', cfg)
        self.share_a = bool(share_a)
        for module in self.backbone.modules():
            if hasattr(module,'lora_A') and 'default' in module.lora_A and 'task1' in module.lora_A:
                if share_a:
                    module.lora_A['task1'].weight = module.lora_A['default'].weight
                else:
                    # Same initial directions; sharing is the only A-policy difference.
                    module.lora_A['task1'].load_state_dict(module.lora_A['default'].state_dict())
        self.activate(0)

    def activate(self, task):
        self.backbone.base_model.set_adapter('default' if task==0 else 'task1')
        # PEFT toggles inactive adapters' requires_grad. Both graphs from a mixed
        # minibatch must remain differentiable until the combined backward call.
        for name,p in self.backbone.named_parameters():
            p.requires_grad_('lora_A.' in name or 'lora_B.' in name)

    def forward(self, tokens, task_ids):
        task_ids=task_ids.long()
        if task_ids.ndim!=1 or len(task_ids)!=len(tokens) or not torch.isin(task_ids,
            torch.tensor([0,1],device=task_ids.device)).all():
            raise ValueError('each row requires known task0 or task1')
        outputs=[];indices=[]
        for task in (0,1):
            ix=torch.nonzero(task_ids==task).flatten()
            if len(ix):
                self.activate(task)
                features=self.backbone(tokens[ix]).mean(2).flatten(1)
                outputs.append(self.heads[task](features));indices.append(ix)
        order=torch.cat(indices)
        return torch.cat(outputs)[order.argsort()]
