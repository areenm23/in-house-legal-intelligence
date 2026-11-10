"""Model C: RetNet (Sun et al., 2023, "Retentive Network").

Multi-scale retention with a fixed decay per head, gamma_h = 1 - 2^(-5-h), xPos-style rotation
(RoPE) on queries and keys, per-head GroupNorm and a swish output gate; SwiGLU FFN; pre-norm.

Training uses the parallel form

    D[n,m] = gamma^(n-m) (n >= m, else 0),  D~ = D / sqrt(rowsum D)
    R = (Q K^T / sqrt(d_h)) * D~,           R~ = R / max(|rowsum R|, 1),   Y = R~ V

Decoding uses the equivalent recurrence, with the two row normalisations reproduced exactly:

    S_n = gamma S_{n-1} + k_n^T v_n,   z_n = gamma z_{n-1} + k_n,   c_n = (1 - gamma^(n+1)) / (1 - gamma)
    y_n = (q_n S_n / sqrt(c_n)) / max(|q_n . z_n| / sqrt(c_n), 1)
"""
import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, SwiGLU, apply_rope, build_rope_cache, init_weights, padded_vocab


@dataclasses.dataclass
class RetNetConfig:
    vocab_size: int
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 6
    d_ff: int = 2048
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    bias: bool = False


class MultiScaleRetention(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, h = c.d_model, c.n_heads
        self.h, self.dh = h, d // h
        self.q_proj = nn.Linear(d, d, bias=c.bias)
        self.k_proj = nn.Linear(d, d, bias=c.bias)
        self.v_proj = nn.Linear(d, d, bias=c.bias)
        self.g_proj = nn.Linear(d, d, bias=c.bias)
        self.o_proj = nn.Linear(d, d, bias=c.bias)
        self.o_proj.is_residual_out = True
        self.group_norm = nn.GroupNorm(h, d, eps=1e-5)
        gammas = 1.0 - 2.0 ** (-5.0 - torch.arange(h, dtype=torch.float32))
        self.register_buffer("log_gamma", gammas.log(), persistent=False)

    def _heads(self, t):
        B, T, _ = t.shape
        return t.view(B, T, self.h, self.dh).transpose(1, 2)

    def _qkv(self, x, rope_cs, offset):
        q = apply_rope(self._heads(self.q_proj(x)), *rope_cs, offset=offset)
        k = apply_rope(self._heads(self.k_proj(x)), *rope_cs, offset=offset) * self.dh ** -0.5
        v = self._heads(self.v_proj(x))
        return q, k, v

    def _out(self, x, y):
        """y: (B, H, T, dh) retention output -> gated, group-normed projection."""
        B, H, T, dh = y.shape
        y = y.transpose(1, 2).reshape(B * T, H * dh)
        y = self.group_norm(y.float()).view(B, T, H * dh).to(x.dtype)
        return self.o_proj(F.silu(self.g_proj(x)) * y)

    def decay(self, T, device):
        """(H, T, T) decay matrix D, its row sums, both float32."""
        n = torch.arange(T, device=device, dtype=torch.float32)
        diff = n[:, None] - n[None, :]
        lg = self.log_gamma.to(device)[:, None, None]
        D = torch.where(diff >= 0, torch.exp(diff.clamp(min=0) * lg), torch.zeros((), device=device))
        return D, D.sum(-1, keepdim=True)

    def forward(self, x, rope_cs, offset=0):
        q, k, v = self._qkv(x, rope_cs, offset)
        T = x.shape[1]
        D, dsum = self.decay(T, x.device)
        Dn = (D / dsum.sqrt()).to(q.dtype)
        R = (q @ k.transpose(-1, -2)) * Dn
        denom = R.float().sum(-1, keepdim=True).abs().clamp(min=1.0).to(R.dtype)
        y = (R / denom) @ v
        return self._out(x, y)

    def prefill(self, x, rope_cs):
        """Parallel pass over the prompt that also returns the recurrent state after it."""
        q, k, v = self._qkv(x, rope_cs, 0)
        T = x.shape[1]
        D, dsum = self.decay(T, x.device)
        Dn = (D / dsum.sqrt()).to(q.dtype)
        R = (q @ k.transpose(-1, -2)) * Dn
        denom = R.float().sum(-1, keepdim=True).abs().clamp(min=1.0).to(R.dtype)
        y = (R / denom) @ v
        w = D[:, -1, :].to(torch.float32)
        with torch.autocast(device_type=x.device.type, enabled=False):
            S = torch.einsum("ht,bhtk,bhtv->bhkv", w, k.float(), v.float())
            z = torch.einsum("ht,bhtk->bhk", w, k.float())
        return self._out(x, y), {"S": S, "z": z, "n": T}

    def step(self, x, rope_cs, state):
        """x: (B, 1, d), state from prefill/step -> output (B, 1, d), new state."""
        n = state["n"]
        q, k, v = self._qkv(x, rope_cs, n)
        q, k, v = q[:, :, 0].float(), k[:, :, 0].float(), v[:, :, 0].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            g = self.log_gamma.exp().to(x.device)[None, :]
            S = state["S"] * g[..., None, None] + k[..., :, None] * v[..., None, :]
            z = state["z"] * g[..., None] + k
            c = ((1 - g ** (n + 1)) / (1 - g)).sqrt()
            num = torch.einsum("bhk,bhkv->bhv", q, S) / c[..., None]
            den = ((q * z).sum(-1) / c).abs().clamp(min=1.0)
            y = (num / den[..., None])[:, :, None, :].to(x.dtype)
        return self._out(x, y), {"S": S, "z": z, "n": n + 1}


class RetNetBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ret_norm = RMSNorm(c.d_model)
        self.retention = MultiScaleRetention(c)
        self.ffn_norm = RMSNorm(c.d_model)
        self.ffn = SwiGLU(c.d_model, c.d_ff, bias=c.bias)

    def forward(self, x, rope_cs):
        x = x + self.retention(self.ret_norm(x), rope_cs)
        return x + self.ffn(self.ffn_norm(x))

    def prefill(self, x, rope_cs):
        h, st = self.retention.prefill(self.ret_norm(x), rope_cs)
        x = x + h
        return x + self.ffn(self.ffn_norm(x)), st

    def step(self, x, rope_cs, st):
        h, st = self.retention.step(self.ret_norm(x), rope_cs, st)
        x = x + h
        return x + self.ffn(self.ffn_norm(x)), st


class RetNetLM(nn.Module):
    def __init__(self, c: RetNetConfig):
        super().__init__()
        self.cfg = c
        self.tok_emb = nn.Embedding(padded_vocab(c.vocab_size), c.d_model)
        self.blocks = nn.ModuleList([RetNetBlock(c) for _ in range(c.n_layers)])
        self.norm = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, padded_vocab(c.vocab_size), bias=False)
        self.lm_head.weight = self.tok_emb.weight
        cos, sin = build_rope_cache(c.max_seq_len, c.d_model // c.n_heads, c.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        init_weights(self, c.n_layers)

    def _head(self, x):
        return self.lm_head(self.norm(x))[..., :self.cfg.vocab_size]

    def forward(self, ids):
        x = self.tok_emb(ids)
        cs = (self.rope_cos, self.rope_sin)
        for blk in self.blocks:
            x = blk(x, cs)
        return self._head(x)

    def forward_recurrent(self, ids):
        """Token-by-token recurrence (for checking it equals forward)."""
        logits, states = self.prefill(ids[:, :1])
        outs = [logits]
        for t in range(1, ids.shape[1]):
            logits, states = self.step(ids[:, t:t + 1], states)
            outs.append(logits)
        return torch.cat(outs, dim=1)

    def prefill(self, ids):
        x = self.tok_emb(ids)
        cs = (self.rope_cos, self.rope_sin)
        states = []
        for blk in self.blocks:
            x, st = blk.prefill(x, cs)
            states.append(st)
        return self._head(x[:, -1:]), states

    def step(self, ids, states):
        x = self.tok_emb(ids)
        cs = (self.rope_cos, self.rope_sin)
        new = []
        for blk, st in zip(self.blocks, states):
            x, st = blk.step(x, cs, st)
            new.append(st)
        return self._head(x), new


class RetNetAPI:
    def __init__(self, model):
        self.m = model

    def prefill(self, ids):
        logits, states = self.m.prefill(ids)
        return logits[:, -1], states

    def step(self, states, nxt):
        logits, states = self.m.step(nxt[:, None], states)
        return logits[:, -1], states
