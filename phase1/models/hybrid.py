"""Model F: hybrid Mamba-2 + attention LM (Samba / Jamba style).

The stack is n_groups repetitions of [mamba_per_group x Mamba-2 block, 1 x attention block]
(runs/run_F.py uses 1:1, as in Samba; Jamba uses 7:1).
Mamba-2 blocks (D's mixer: norm -> selective SSM -> residual, no FFN) carry most of the
sequence mixing in linear time; the attention blocks (A's block: GQA + RoPE + QK-norm + SwiGLU)
give exact recall over the whole window. Decoding keeps an SSM/conv state per Mamba block and
a KV cache per attention block.
"""
import dataclasses

import torch
import torch.nn as nn

from .decoder import DecoderBlock, DecoderConfig
from .layers import RMSNorm, build_rope_cache, init_weights, padded_vocab
from .mamba2 import Mamba2Block, Mamba2Config


@dataclasses.dataclass
class HybridConfig:
    vocab_size: int
    d_model: int = 768
    n_groups: int = 5
    mamba_per_group: int = 3
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 2048
    d_state: int = 64
    expand: int = 2
    headdim: int = 64
    chunk_size: int = 64
    max_seq_len: int = 2048
    rope_base: float = 10000.0


class HybridLM(nn.Module):
    def __init__(self, c: HybridConfig):
        super().__init__()
        self.cfg = c
        mc = Mamba2Config(vocab_size=c.vocab_size, d_model=c.d_model, d_state=c.d_state, expand=c.expand,
                          headdim=c.headdim, chunk_size=c.chunk_size)
        dc = DecoderConfig(vocab_size=c.vocab_size, d_model=c.d_model, n_heads=c.n_heads, n_kv_heads=c.n_kv_heads,
                           d_ff=c.d_ff)
        layers, kinds = [], []
        for _ in range(c.n_groups):
            for _ in range(c.mamba_per_group):
                layers.append(Mamba2Block(mc))
                kinds.append("mamba")
            layers.append(DecoderBlock(dc))
            kinds.append("attn")
        self.layers, self.kinds = nn.ModuleList(layers), kinds
        self.tok_emb = nn.Embedding(padded_vocab(c.vocab_size), c.d_model)
        self.norm = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, padded_vocab(c.vocab_size), bias=False)
        self.lm_head.weight = self.tok_emb.weight
        cos, sin = build_rope_cache(c.max_seq_len, c.d_model // c.n_heads, c.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        init_weights(self, len(layers))
        for blk, k in zip(self.layers, kinds):
            if k == "mamba":
                blk.mixer.conv1d.reset_parameters()

    def _head(self, x):
        return self.lm_head(self.norm(x))[..., :self.cfg.vocab_size]

    def forward(self, ids):
        x = self.tok_emb(ids)
        cs = (self.rope_cos, self.rope_sin)
        for blk, k in zip(self.layers, self.kinds):
            x = blk(x) if k == "mamba" else blk(x, cs)
        return self._head(x)

    def prefill(self, ids):
        x = self.tok_emb(ids)
        cs = (self.rope_cos, self.rope_sin)
        states = []
        for blk, k in zip(self.layers, self.kinds):
            if k == "mamba":
                h, st = blk.mixer(blk.norm(x), return_state=True)
                x = x + h
            else:
                st = {}
                x = blk(x, cs, cache=st, offset=0)
            states.append(st)
        return self._head(x[:, -1:]), (states, ids.shape[1])

    def step(self, ids, state):
        states, pos = state
        x = self.tok_emb(ids)
        cs = (self.rope_cos, self.rope_sin)
        new = []
        for blk, k, st in zip(self.layers, self.kinds, states):
            if k == "mamba":
                h, st = blk.mixer.step(blk.norm(x), st)
                x = x + h
            else:
                x = blk(x, cs, cache=st, offset=pos)
            new.append(st)
        return self._head(x), (new, pos + 1)


class HybridAPI:
    def __init__(self, model):
        self.m = model

    def prefill(self, ids):
        logits, st = self.m.prefill(ids)
        return logits[:, -1], st

    def step(self, st, nxt):
        logits, st = self.m.step(nxt[:, None], st)
        return logits[:, -1], st
