"""Model D: Mamba-2 selective state-space LM (Dao & Gu, 2024, "Transformers are SSMs").

Pure PyTorch (no custom CUDA kernels, so it runs on any Kaggle GPU). Training uses the chunked
SSD algorithm (the paper's minimal reference, written without einops); decoding uses the
equivalent per-token recurrence with a convolution state and an SSM state per layer.

Mixer, per layer:
    [z, xBC, dt] = in_proj(x)
    xBC = silu(causal depthwise conv1d(xBC));  x, B, C = split(xBC)
    dt = softplus(dt + dt_bias);  A = -exp(A_log)                (one scalar decay per head)
    h_t = exp(dt_t A) h_{t-1} + dt_t x_t B_t^T,   y_t = h_t C_t + D x_t
    out = out_proj(RMSNorm(y * silu(z)))
Blocks are pre-norm residual mixers with no separate FFN (as in Mamba).
"""
import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import RMSNorm, init_weights, padded_vocab


@dataclasses.dataclass
class Mamba2Config:
    vocab_size: int
    d_model: int = 768
    n_layers: int = 24
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    chunk_size: int = 64
    dt_min: float = 1e-3
    dt_max: float = 1e-1


def segsum(x):
    """x: (..., T) -> (..., T, T) with out[i, j] = sum_{j < k <= i} x[k] for i >= j, -inf above the diagonal."""
    T = x.size(-1)
    xr = x[..., None].expand(*x.shape, T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    xr = xr.masked_fill(~mask, 0)
    out = torch.cumsum(xr, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    return out.masked_fill(~mask, -torch.inf)


def ssd_chunked(X, A, B, C, chunk):
    """Chunked SSD scan (float32).

    X: (b, T, h, p)  inputs already multiplied by dt
    A: (b, T, h)     dt * A (log decay per step, <= 0)
    B, C: (b, T, n)  shared by all heads (one group)
    Returns Y (b, T, h, p) and the final state (b, h, p, n). T must be a multiple of chunk.
    """
    b, T, h, p = X.shape
    n = B.shape[-1]
    c = T // chunk
    X = X.reshape(b, c, chunk, h, p)
    A = A.reshape(b, c, chunk, h).permute(0, 3, 1, 2)
    B = B.reshape(b, c, chunk, n)
    C = C.reshape(b, c, chunk, n)
    A_cum = torch.cumsum(A, dim=-1)

    L = torch.exp(segsum(A))
    CB = torch.einsum("bcln,bcsn->bcls", C, B)
    W = CB[:, None] * L
    Y_diag = torch.einsum("bhcls,bcshp->bclhp", W, X)

    decay_states = torch.exp(A_cum[..., -1:] - A_cum)
    states = torch.einsum("bcln,bhcl,bclhp->bchpn", B, decay_states, X)

    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cum[..., -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final = new_states[:, :-1], new_states[:, -1]

    Y_off = torch.einsum("bcln,bchpn,bhcl->bclhp", C, states, torch.exp(A_cum))
    return (Y_diag + Y_off).reshape(b, T, h, p), final


class Mamba2Mixer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.d_inner = c.expand * c.d_model
        assert self.d_inner % c.headdim == 0
        self.nheads = self.d_inner // c.headdim
        self.conv_dim = self.d_inner + 2 * c.d_state
        self.in_proj = nn.Linear(c.d_model, 2 * self.d_inner + 2 * c.d_state + self.nheads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, c.d_conv, groups=self.conv_dim,
                                padding=c.d_conv - 1, bias=True)
        dt = torch.exp(torch.rand(self.nheads) * (math.log(c.dt_max) - math.log(c.dt_min)) + math.log(c.dt_min))
        dt = dt.clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, c.d_model, bias=False)
        self.out_proj.is_residual_out = True

    def _split(self, zxbcdt):
        return torch.split(zxbcdt, [self.d_inner, self.conv_dim, self.nheads], dim=-1)

    def _ssm_inputs(self, xBC, dt):
        x, B, C = torch.split(xBC, [self.d_inner, self.c.d_state, self.c.d_state], dim=-1)
        dt = F.softplus(dt.float() + self.dt_bias.float())
        A = -torch.exp(self.A_log.float())
        return x, B, C, dt, A

    def _finish(self, y, z):
        return self.out_proj(self.norm(y * F.silu(z.float())).to(z.dtype))

    def forward(self, u, return_state=False):
        b, T, _ = u.shape
        z, xBC, dt = self._split(self.in_proj(u))
        xBC_raw = xBC
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2))[..., :T].transpose(1, 2))
        x, B, C, dt, A = self._ssm_inputs(xBC, dt)
        x = x.float().reshape(b, T, self.nheads, self.c.headdim)
        pad = (-T) % self.c.chunk_size
        X, Ad, Bf, Cf = x * dt[..., None], dt * A, B.float(), C.float()
        if pad:
            X = F.pad(X, (0, 0, 0, 0, 0, pad))
            Ad = F.pad(Ad, (0, 0, 0, pad))
            Bf, Cf = F.pad(Bf, (0, 0, 0, pad)), F.pad(Cf, (0, 0, 0, pad))
        with torch.autocast(device_type=u.device.type, enabled=False):
            if self.training and torch.is_grad_enabled():
                Y, final = checkpoint(ssd_chunked, X, Ad, Bf, Cf, self.c.chunk_size, use_reentrant=False)
            else:
                Y, final = ssd_chunked(X, Ad, Bf, Cf, self.c.chunk_size)
        Y = Y[:, :T] + x * self.D.float()[:, None]
        out = self._finish(Y.reshape(b, T, self.d_inner), z)
        if not return_state:
            return out
        K = self.c.d_conv
        conv_state = F.pad(xBC_raw.transpose(1, 2), (max(0, K - 1 - T), 0))[..., -(K - 1):]
        return out, {"conv": conv_state.float(), "ssm": final}

    def step(self, u, state):
        """u: (b, 1, d) -> (b, 1, d), new state."""
        z, xBC, dt = self._split(self.in_proj(u[:, 0]))
        window = torch.cat([state["conv"], xBC.float()[..., None]], dim=-1)
        xc = (window * self.conv1d.weight[:, 0].float()).sum(-1) + self.conv1d.bias.float()
        x, B, C, dt, A = self._ssm_inputs(F.silu(xc), dt)
        x = x.reshape(-1, self.nheads, self.c.headdim).float()
        with torch.autocast(device_type=u.device.type, enabled=False):
            dA = torch.exp(dt * A)
            h = state["ssm"] * dA[..., None, None] + (x * dt[..., None])[..., None] * B.float()[:, None, None, :]
            y = torch.einsum("bhpn,bn->bhp", h, C.float()) + x * self.D.float()[:, None]
        out = self._finish(y.reshape(-1, 1, self.d_inner), z[:, None])
        return out, {"conv": window[..., 1:], "ssm": h}


class Mamba2Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = RMSNorm(c.d_model)
        self.mixer = Mamba2Mixer(c)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class Mamba2LM(nn.Module):
    def __init__(self, c: Mamba2Config):
        super().__init__()
        self.cfg = c
        self.tok_emb = nn.Embedding(padded_vocab(c.vocab_size), c.d_model)
        self.blocks = nn.ModuleList([Mamba2Block(c) for _ in range(c.n_layers)])
        self.norm = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, padded_vocab(c.vocab_size), bias=False)
        self.lm_head.weight = self.tok_emb.weight
        init_weights(self, c.n_layers)
        for blk in self.blocks:
            blk.mixer.conv1d.reset_parameters()

    def _head(self, x):
        return self.lm_head(self.norm(x))[..., :self.cfg.vocab_size]

    def forward(self, ids):
        x = self.tok_emb(ids)
        for blk in self.blocks:
            x = blk(x)
        return self._head(x)

    def prefill(self, ids):
        x = self.tok_emb(ids)
        states = []
        for blk in self.blocks:
            h, st = blk.mixer(blk.norm(x), return_state=True)
            x = x + h
            states.append(st)
        return self._head(x[:, -1:]), states

    def step(self, ids, states):
        x = self.tok_emb(ids)
        new = []
        for blk, st in zip(self.blocks, states):
            h, st = blk.mixer.step(blk.norm(x), st)
            x = x + h
            new.append(st)
        return self._head(x), new

    def forward_recurrent(self, ids):
        logits, states = self.prefill(ids[:, :1])
        outs = [logits]
        for t in range(1, ids.shape[1]):
            logits, states = self.step(ids[:, t:t + 1], states)
            outs.append(logits)
        return torch.cat(outs, dim=1)


class Mamba2API:
    def __init__(self, model):
        self.m = model

    def prefill(self, ids):
        logits, states = self.m.prefill(ids)
        return logits[:, -1], states

    def step(self, states, nxt):
        logits, states = self.m.step(nxt[:, None], states)
        return logits[:, -1], states
