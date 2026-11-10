"""Building blocks shared by the four Phase-1 models.

RoPE (`build_rope_cache`, `rotate_half`, `apply_rope`) is the Assignment-3 implementation
(CS781 Assm-3/src/q4/rope.py), unchanged apart from taking an explicit position offset.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rope_cache(max_seq_len: int, d_head: int, base: float = 10000.0):
    """(cos, sin) of shape (max_seq_len, d_head); split-half convention (channel i pairs with i + d/2)."""
    assert d_head % 2 == 0, "RoPE requires an even head dimension"
    i = torch.arange(0, d_head // 2, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (2.0 * i / d_head))
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    angles = torch.cat([freqs, freqs], dim=-1)
    return angles.cos(), angles.sin()


def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin, offset=0):
    """x: (B, H, T, d_head); rotates positions offset .. offset+T-1."""
    T = x.shape[-2]
    if offset + T > cos.shape[0]:
        raise ValueError(f"position {offset + T} exceeds the RoPE cache ({cos.shape[0]})")
    cos = cos[offset:offset + T].to(dtype=x.dtype, device=x.device).view(1, 1, T, -1)
    sin = sin[offset:offset + T].to(dtype=x.dtype, device=x.device).view(1, 1, T, -1)
    return x * cos + rotate_half(x) * sin


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d, hidden, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(d, hidden, bias=bias)
        self.up_proj = nn.Linear(d, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, d, bias=bias)
        self.down_proj.is_residual_out = True

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def init_weights(model, n_residual_layers, std=0.02):
    """GPT-2 style: N(0, std) everywhere, residual output projections scaled by 1/sqrt(2 * layers)."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            s = std / math.sqrt(2 * n_residual_layers) if getattr(m, "is_residual_out", False) else std
            nn.init.normal_(m.weight, mean=0.0, std=s)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)


def padded_vocab(v, multiple=64):
    return ((v + multiple - 1) // multiple) * multiple


class Attention(nn.Module):
    """Multi-head attention with GQA, optional QK-norm and RoPE.

    Q/K/V/O are separate named Linear layers (q_proj, k_proj, v_proj, o_proj) so that LoRA,
    LoRMA and IA3 can target them in Phases 2-7.

    kind = "causal" (decoder self-attention), "bidir" (encoder self-attention) or "cross".
    """

    def __init__(self, d, n_heads, n_kv_heads, kind, bias=False, qk_norm=True, rope=True):
        super().__init__()
        assert d % n_heads == 0 and n_heads % n_kv_heads == 0
        self.kind, self.h, self.hkv = kind, n_heads, n_kv_heads
        self.dh = d // n_heads
        self.use_rope = rope and kind != "cross"
        self.q_proj = nn.Linear(d, n_heads * self.dh, bias=bias)
        self.k_proj = nn.Linear(d, n_kv_heads * self.dh, bias=bias)
        self.v_proj = nn.Linear(d, n_kv_heads * self.dh, bias=bias)
        self.o_proj = nn.Linear(n_heads * self.dh, d, bias=bias)
        self.o_proj.is_residual_out = True
        self.q_norm = RMSNorm(self.dh) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.dh) if qk_norm else nn.Identity()

    def _heads(self, t, n):
        B, T, _ = t.shape
        return t.view(B, T, n, self.dh).transpose(1, 2)

    def kv(self, src, rope_cs=None, offset=0):
        """Projected (and, for self-attention, rotated) keys/values: (B, Hkv, T, dh) each."""
        k = self.k_norm(self._heads(self.k_proj(src), self.hkv))
        v = self._heads(self.v_proj(src), self.hkv)
        if self.use_rope:
            k = apply_rope(k, *rope_cs, offset=offset)
        return k, v

    def forward(self, x, rope_cs=None, memory=None, cache=None, offset=0):
        """x: (B, T, d). memory: encoder output (cross). cache: dict with 'k','v' (self) or the
        precomputed memory k/v (cross); updated in place for self-attention. offset: position of x[:, 0]."""
        B, T, _ = x.shape
        q = self.q_norm(self._heads(self.q_proj(x), self.h))
        if self.use_rope:
            q = apply_rope(q, *rope_cs, offset=offset)
        if self.kind == "cross":
            if cache is not None and "k" in cache:
                k, v = cache["k"], cache["v"]
            else:
                k, v = self.kv(memory)
                if cache is not None:
                    cache["k"], cache["v"] = k, v
        else:
            k, v = self.kv(x, rope_cs, offset)
            if cache is not None:
                if "k" in cache:
                    k = torch.cat([cache["k"], k], dim=2)
                    v = torch.cat([cache["v"], v], dim=2)
                cache["k"], cache["v"] = k, v
        if self.hkv != self.h:
            k = k.repeat_interleave(self.h // self.hkv, dim=1)
            v = v.repeat_interleave(self.h // self.hkv, dim=1)
        causal = self.kind == "causal" and T > 1
        if causal and k.shape[2] != T:
            raise ValueError("multi-token decoding steps with a non-empty cache are not supported")
        y = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), is_causal=causal)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.h * self.dh))
