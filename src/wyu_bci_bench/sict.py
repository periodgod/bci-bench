"""Session-isolated causal context decoder used by the SICT experiment."""

from __future__ import annotations

import torch
from torch import nn


class CausalContextBlock(nn.Module):
    """Pre-norm cross-attention block with learned relative-age bias."""

    def __init__(self, width: int = 64, heads: int = 4, mlp_width: int = 128,
                 age_buckets: int = 16):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.age_bias = nn.Embedding(age_buckets, heads)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, mlp_width), nn.GELU(), nn.Linear(mlp_width, width))

    def forward(self, query: torch.Tensor, memory: torch.Tensor,
                age_buckets: torch.Tensor, *, uniform_attention: bool = False) -> torch.Tensor:
        if query.ndim != 2 or memory.ndim != 2 or query.shape != (1, self.width):
            raise ValueError("query must be (1,width) and memory must be (history,width)")
        if age_buckets.shape != (len(memory),):
            raise ValueError("age_buckets must align with memory")
        if len(memory) == 0:
            return query
        q = self.query(self.query_norm(query)).view(1, self.heads, self.head_width)
        normalized = self.memory_norm(memory)
        k = self.key(normalized).view(len(memory), self.heads, self.head_width)
        v = self.value(normalized).view(len(memory), self.heads, self.head_width)
        scores = torch.einsum("qhd,khd->hqk", q, k) / (self.head_width ** 0.5)
        scores = scores + self.age_bias(age_buckets).T[:, None, :]
        weights = torch.softmax(scores, dim=-1)
        if uniform_attention:
            weights = torch.full_like(weights, 1.0 / len(memory))
        context = torch.einsum("hqk,khd->qhd", weights, v).reshape(1, self.width)
        hidden = query + self.output(context)
        return hidden + self.ff(self.ff_norm(hidden))

    def forward_sequence(self, queries: torch.Tensor, memory: torch.Tensor, visible: torch.Tensor,
                         ages: torch.Tensor, *, uniform_attention: bool = False) -> torch.Tensor:
        n = len(queries)
        q = self.query(self.query_norm(queries)).view(n, self.heads, self.head_width).transpose(0, 1)
        normalized = self.memory_norm(memory)
        k = self.key(normalized).view(n, self.heads, self.head_width).transpose(0, 1)
        v = self.value(normalized).view(n, self.heads, self.head_width).transpose(0, 1)
        scores = torch.matmul(q, k.transpose(1, 2)) / (self.head_width ** 0.5)
        scores = scores + self.age_bias(ages).permute(2, 0, 1)
        scores = scores.masked_fill(~visible.unsqueeze(0), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        if uniform_attention:
            weights = visible.to(weights.dtype).unsqueeze(0)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1)
        context = torch.matmul(weights, v).transpose(0, 1).reshape(n, self.width)
        has_context = visible.any(-1, keepdim=True)
        hidden = queries + self.output(context) * has_context
        return hidden + self.ff(self.ff_norm(hidden)) * has_context


class SICTAdapter(nn.Module):
    """Causal residual decoder over PCA, frozen M2 hidden state, and source logits.

    Each call predicts one window from a query of shape ``(91,)`` and an already
    truncated history of shape ``(0..memory, 91)``. The caller appends the current
    query only after prediction, which makes session reset and stream isolation
    explicit at the evaluator boundary.
    """

    def __init__(self, memory: int = 128, input_width: int = 91, width: int = 64,
                 heads: int = 4, mlp_width: int = 128, age_buckets: int = 16,
                 n_classes: int = 3):
        super().__init__()
        if memory < 1 or input_width < 1 or age_buckets < 2:
            raise ValueError("memory/input_width must be positive and age_buckets >= 2")
        self.memory = int(memory)
        self.age_buckets = int(age_buckets)
        self.input = nn.Linear(input_width, width)
        self.blocks = nn.ModuleList([
            CausalContextBlock(width, heads, mlp_width, age_buckets) for _ in range(2)
        ])
        self.output = nn.Linear(width, n_classes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def age_ids(self, history_length: int, device: torch.device) -> torch.Tensor:
        ages = torch.arange(history_length, 0, -1, device=device)
        return torch.clamp(torch.floor(torch.log2(ages.float())).long(), max=self.age_buckets - 1)

    def forward_step(self, query: torch.Tensor, history: torch.Tensor | None = None,
                     *, ablation: str = "none", shuffle_seed: int = 20261031) -> torch.Tensor:
        if query.ndim != 1 or query.shape[0] != self.input.in_features:
            raise ValueError(f"query must have shape ({self.input.in_features},)")
        if history is None:
            history = query.new_empty((0, query.numel()))
        if history.ndim != 2 or history.shape[1] != query.numel():
            raise ValueError("history must be (n,input_width)")
        if len(history) > self.memory:
            history = history[-self.memory:]
        if ablation not in {"none", "uniform", "shuffle", "no_memory"}:
            raise ValueError(f"unknown ablation: {ablation}")
        if ablation == "no_memory":
            history = history[:0]
        ages = self.age_ids(len(history), query.device)
        if ablation == "shuffle" and len(history) > 1:
            generator = torch.Generator(device=history.device).manual_seed(shuffle_seed)
            order = torch.randperm(len(history), generator=generator, device=history.device)
            history = history[order]
            ages = ages[order]
        q = self.input(query.unsqueeze(0))
        if not len(history):
            return query.new_zeros(self.output.out_features)
        mem = self.input(history)
        for block in self.blocks:
            q = block(q, mem, ages, uniform_attention=ablation == "uniform")
        return self.output(q).squeeze(0)

    def forward_sequence(self, inputs: torch.Tensor, stream_ids: torch.Tensor,
                         *, ablation: str = "none", shuffle_seed: int = 20261031) -> torch.Tensor:
        """Predict a concatenated set of streams using only same-stream prefixes."""
        if inputs.ndim != 2 or inputs.shape[1] != self.input.in_features:
            raise ValueError(f"inputs must have shape (n,{self.input.in_features})")
        if stream_ids.shape != (len(inputs),):
            raise ValueError("stream_ids must align with inputs")
        if ablation not in {"none", "uniform", "shuffle", "no_memory"}:
            raise ValueError(f"unknown ablation: {ablation}")
        n = len(inputs)
        q_pos = torch.arange(n, device=inputs.device)[:, None]
        k_pos = torch.arange(n, device=inputs.device)[None, :]
        distance = q_pos - k_pos
        same_stream = stream_ids[:, None] == stream_ids[None, :]
        visible = same_stream & (distance > 0) & (distance <= self.memory)
        if ablation == "no_memory":
            return inputs.new_zeros((n, self.output.out_features))
        ages = torch.zeros((n, n), dtype=torch.long, device=inputs.device)
        ages[visible] = torch.clamp(torch.floor(torch.log2(distance[visible].float())).long(),
                                    max=self.age_buckets - 1)
        memory = self.input(inputs)
        queries = memory
        for block in self.blocks:
            queries = block.forward_sequence(queries, memory, visible, ages,
                                             uniform_attention=ablation == "uniform")
        residual = self.output(queries)
        return residual * visible.any(-1, keepdim=True)


class SICTStream:
    """Small explicit stream state; a new instance resets all historical context."""

    def __init__(self, adapter: SICTAdapter, *, ablation: str = "none"):
        self.adapter = adapter
        self.ablation = ablation
        self._history: list[torch.Tensor] = []

    def reset_state(self) -> None:
        self._history.clear()

    def predict_one(self, query: torch.Tensor) -> torch.Tensor:
        history = (torch.stack(self._history) if self._history
                   else query.new_empty((0, query.numel())))
        result = self.adapter.forward_step(query, history, ablation=self.ablation)
        self._history.append(query.detach().clone())
        if len(self._history) > self.adapter.memory:
            self._history.pop(0)
        return result
