"""Model E: sparse Mixture-of-Experts decoder (Mixtral-style).

Same attention as A (GQA + RoPE + QK-norm, pre-norm RMSNorm), but each FFN is replaced by
n_experts SwiGLU experts and a linear router; every token goes to its top_k experts and their
outputs are mixed with softmax weights over the selected logits. Dropless: each expert
processes exactly the tokens routed to it (gather -> expert -> weighted scatter-add).

Training adds two router losses (read from model.aux after a forward pass):
  lb : Switch-Transformer load balancing, E * sum_i f_i * P_i (f_i = share of routing slots,
       P_i = mean router probability), minimum 1 when the load is uniform
  z  : router z-loss, mean(logsumexp(router logits)^2), keeps router logits small (fp16)
"""
import dataclasses

import torch
import torch.nn as nn

from .decoder import DecoderAPI
from .layers import Attention, RMSNorm, SwiGLU, build_rope_cache, init_weights, padded_vocab


@dataclasses.dataclass
class MoEConfig:
    vocab_size: int
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    n_experts: int = 8
    top_k: int = 2
    d_ff_expert: int = 1024
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    qk_norm: bool = True
    bias: bool = False


class MoEFFN(nn.Module):
    def __init__(self, d, d_ff, n_experts, top_k, bias=False):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d, d_ff, bias=bias) for _ in range(n_experts)])
        self.lb = self.z = None
        self.last_load = None

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        logits = self.router(flat).float()
        top_v, top_i = logits.topk(self.top_k, dim=-1)
        w = top_v.softmax(-1)
        out = torch.zeros(flat.shape, dtype=torch.float32, device=flat.device)
        for e, expert in enumerate(self.experts):
            rows, slot = (top_i == e).nonzero(as_tuple=True)
            if rows.numel():
                y = expert(flat[rows])
                out.index_add_(0, rows, y.float() * w[rows, slot, None])
        n = flat.shape[0]
        load = torch.zeros(self.n_experts, device=flat.device).scatter_add_(
            0, top_i.reshape(-1), torch.ones(n * self.top_k, device=flat.device)) / (n * self.top_k)
        self.lb = self.n_experts * (load * logits.softmax(-1).mean(0)).sum()
        self.z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        self.last_load = load.detach()
        return out.to(x.dtype).view(shape)


class MoEBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm = RMSNorm(c.d_model)
        self.attn = Attention(c.d_model, c.n_heads, c.n_kv_heads, "causal", bias=c.bias, qk_norm=c.qk_norm)
        self.ffn_norm = RMSNorm(c.d_model)
        self.moe = MoEFFN(c.d_model, c.d_ff_expert, c.n_experts, c.top_k, bias=c.bias)

    def forward(self, x, rope_cs, cache=None, offset=0):
        x = x + self.attn(self.attn_norm(x), rope_cs, cache=cache, offset=offset)
        return x + self.moe(self.ffn_norm(x))


class MoEDecoderLM(nn.Module):
    def __init__(self, c: MoEConfig):
        super().__init__()
        self.cfg = c
        self.tok_emb = nn.Embedding(padded_vocab(c.vocab_size), c.d_model)
        self.blocks = nn.ModuleList([MoEBlock(c) for _ in range(c.n_layers)])
        self.norm = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, padded_vocab(c.vocab_size), bias=False)
        self.lm_head.weight = self.tok_emb.weight
        cos, sin = build_rope_cache(c.max_seq_len, c.d_model // c.n_heads, c.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        init_weights(self, c.n_layers)
        self.aux = {}

    def forward(self, ids, caches=None, offset=0, last_only=False):
        x = self.tok_emb(ids)
        cs = (self.rope_cos, self.rope_sin)
        for i, blk in enumerate(self.blocks):
            x = blk(x, cs, cache=None if caches is None else caches[i], offset=offset)
        self.aux = {"lb": torch.stack([b.moe.lb for b in self.blocks]).mean(),
                    "z": torch.stack([b.moe.z for b in self.blocks]).mean()}
        if last_only:
            x = x[:, -1:]
        return self.lm_head(self.norm(x))[..., :self.cfg.vocab_size]
