"""Model B: encoder-decoder Transformer trained on the continuation objective.

The encoder reads the prefix bidirectionally; the decoder writes the continuation with causal
self-attention plus cross-attention to the encoder output. Same block design as A (pre-norm
RMSNorm, RoPE, QK-norm, SwiGLU); one embedding shared by encoder, decoder and output head.
Biases are kept in every Linear layer so BitFit has something to train in Phase 4.
"""
import dataclasses

import torch
import torch.nn as nn

from .layers import Attention, RMSNorm, SwiGLU, build_rope_cache, init_weights, padded_vocab

BOS_ID = 2


@dataclasses.dataclass
class EncDecConfig:
    vocab_size: int
    d_model: int = 768
    n_enc_layers: int = 8
    n_dec_layers: int = 8
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 2048
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    qk_norm: bool = True
    bias: bool = True


class EncoderBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm = RMSNorm(c.d_model)
        self.attn = Attention(c.d_model, c.n_heads, c.n_kv_heads, "bidir", bias=c.bias, qk_norm=c.qk_norm)
        self.ffn_norm = RMSNorm(c.d_model)
        self.ffn = SwiGLU(c.d_model, c.d_ff, bias=c.bias)

    def forward(self, x, rope_cs):
        x = x + self.attn(self.attn_norm(x), rope_cs)
        return x + self.ffn(self.ffn_norm(x))


class DecoderBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_norm = RMSNorm(c.d_model)
        self.self_attn = Attention(c.d_model, c.n_heads, c.n_kv_heads, "causal", bias=c.bias, qk_norm=c.qk_norm)
        self.cross_norm = RMSNorm(c.d_model)
        self.cross_attn = Attention(c.d_model, c.n_heads, c.n_kv_heads, "cross", bias=c.bias, qk_norm=c.qk_norm)
        self.ffn_norm = RMSNorm(c.d_model)
        self.ffn = SwiGLU(c.d_model, c.d_ff, bias=c.bias)

    def forward(self, x, memory, rope_cs, cache=None, offset=0):
        sc = cc = None
        if cache is not None:
            sc, cc = cache.setdefault("self", {}), cache.setdefault("cross", {})
        x = x + self.self_attn(self.self_norm(x), rope_cs, cache=sc, offset=offset)
        x = x + self.cross_attn(self.cross_norm(x), memory=memory, cache=cc)
        return x + self.ffn(self.ffn_norm(x))


class EncoderDecoderLM(nn.Module):
    def __init__(self, c: EncDecConfig):
        super().__init__()
        self.cfg = c
        self.tok_emb = nn.Embedding(padded_vocab(c.vocab_size), c.d_model)
        self.encoder = nn.ModuleList([EncoderBlock(c) for _ in range(c.n_enc_layers)])
        self.enc_norm = RMSNorm(c.d_model)
        self.decoder = nn.ModuleList([DecoderBlock(c) for _ in range(c.n_dec_layers)])
        self.dec_norm = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, padded_vocab(c.vocab_size), bias=False)
        self.lm_head.weight = self.tok_emb.weight
        cos, sin = build_rope_cache(c.max_seq_len, c.d_model // c.n_heads, c.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        init_weights(self, c.n_enc_layers + c.n_dec_layers)

    def encode(self, src):
        x = self.tok_emb(src)
        cs = (self.rope_cos, self.rope_sin)
        for blk in self.encoder:
            x = blk(x, cs)
        return self.enc_norm(x)

    def decode(self, tgt, memory, caches=None, offset=0):
        x = self.tok_emb(tgt)
        cs = (self.rope_cos, self.rope_sin)
        for i, blk in enumerate(self.decoder):
            x = blk(x, memory, cs, cache=None if caches is None else caches[i], offset=offset)
        return self.lm_head(self.dec_norm(x))[..., :self.cfg.vocab_size]

    def forward(self, src, tgt):
        """src: (B, S) prefix, tgt: (B, T) decoder input (<s> + continuation[:-1]) -> logits (B, T, V)."""
        return self.decode(tgt, self.encode(src))


class EncDecAPI:
    """Encoder run once; decoder self-attention KV cache; cross-attention K/V computed once."""

    def __init__(self, model):
        self.m = model

    def prefill(self, src):
        memory = self.m.encode(src)
        caches = [{} for _ in self.m.decoder]
        bos = torch.full((src.shape[0], 1), BOS_ID, dtype=src.dtype, device=src.device)
        logits = self.m.decode(bos, memory, caches, offset=0)
        return logits[:, -1], (memory, caches, 1)

    def step(self, state, nxt):
        memory, caches, pos = state
        logits = self.m.decode(nxt[:, None], memory, caches, offset=pos)
        return logits[:, -1], (memory, caches, pos + 1)
