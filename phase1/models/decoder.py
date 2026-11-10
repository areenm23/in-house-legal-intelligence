"""Model A: decoder-only Transformer LM (Llama-style).

Pre-norm RMSNorm blocks, GQA attention with RoPE and QK-norm (fp16 stability on T4), SwiGLU
FFN, tied input/output embeddings. Builds on the Assignment-3 decoder (RoPE, GQA).
"""
import dataclasses

import torch
import torch.nn as nn

from .layers import Attention, RMSNorm, SwiGLU, build_rope_cache, init_weights, padded_vocab


@dataclasses.dataclass
class DecoderConfig:
    vocab_size: int
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 2048
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    qk_norm: bool = True
    bias: bool = False


class DecoderBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm = RMSNorm(c.d_model)
        self.attn = Attention(c.d_model, c.n_heads, c.n_kv_heads, "causal", bias=c.bias, qk_norm=c.qk_norm)
        self.ffn_norm = RMSNorm(c.d_model)
        self.ffn = SwiGLU(c.d_model, c.d_ff, bias=c.bias)

    def forward(self, x, rope_cs, cache=None, offset=0):
        x = x + self.attn(self.attn_norm(x), rope_cs, cache=cache, offset=offset)
        return x + self.ffn(self.ffn_norm(x))


class DecoderLM(nn.Module):
    def __init__(self, c: DecoderConfig):
        super().__init__()
        self.cfg = c
        self.tok_emb = nn.Embedding(padded_vocab(c.vocab_size), c.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(c) for _ in range(c.n_layers)])
        self.norm = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, padded_vocab(c.vocab_size), bias=False)
        self.lm_head.weight = self.tok_emb.weight
        cos, sin = build_rope_cache(c.max_seq_len, c.d_model // c.n_heads, c.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        init_weights(self, c.n_layers)

    def forward(self, ids, caches=None, offset=0, last_only=False):
        """ids: (B, T) -> logits (B, T, vocab_size) (or (B, 1, V) with last_only).
        caches: list of per-layer dicts for KV-cached decoding (filled in place)."""
        x = self.tok_emb(ids)
        cs = (self.rope_cos, self.rope_sin)
        for i, blk in enumerate(self.blocks):
            x = blk(x, cs, cache=None if caches is None else caches[i], offset=offset)
        if last_only:
            x = x[:, -1:]
        return self.lm_head(self.norm(x))[..., :self.cfg.vocab_size]


class DecoderAPI:
    """KV-cached prefill/step for phase1.eval.generate."""

    def __init__(self, model):
        self.m = model

    def prefill(self, ids):
        caches = [{} for _ in self.m.blocks]
        logits = self.m(ids, caches=caches, offset=0, last_only=True)
        return logits[:, -1], (caches, ids.shape[1])

    def step(self, state, nxt):
        caches, pos = state
        logits = self.m(nxt[:, None], caches=caches, offset=pos)
        return logits[:, -1], (caches, pos + 1)
