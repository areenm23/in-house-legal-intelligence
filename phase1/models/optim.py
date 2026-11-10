"""Optimizers and LR schedules for the Phase-1 runs.

`newton_schulz_orthogonalize` and the Muon update rule are the Assignment-3 code
(CS781 Assm-3/src/q4/optim_muon.py). `MuonAdamW` puts Muon (2-D hidden weights) and AdamW
(embeddings, norms, biases, everything else) in ONE torch.optim.Optimizer, so the runner's
GradScaler, LambdaLR and checkpointing work unchanged.
"""
import math

import torch
import torch.nn as nn


@torch.no_grad()
def newton_schulz_orthogonalize(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def _no_decay(name, p):
    return p.ndim < 2 or name.endswith(".bias") or "norm" in name


def adamw_groups(model, weight_decay):
    """Weight decay on matrices only (not on norms, biases, SSM scalars)."""
    decay, no_decay, seen = [], [], set()
    for n, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (no_decay if _no_decay(n, p) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def make_adamw(model, lr, weight_decay, betas=(0.9, 0.95), eps=1e-8):
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(adamw_groups(model, weight_decay), lr=lr, betas=betas, eps=eps, fused=fused)


class MuonAdamW(torch.optim.Optimizer):
    """Groups with use_muon=True: momentum + Newton-Schulz (A-3 Muon). Others: AdamW."""

    def __init__(self, groups):
        defaults = dict(lr=1e-3, use_muon=False, momentum=0.95, ns_steps=5, betas=(0.9, 0.95), eps=1e-8,
                        weight_decay=0.0)
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if g["use_muon"]:
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"] = torch.zeros_like(p.grad)
                    buf = st["momentum_buffer"]
                    buf.mul_(g["momentum"]).add_(p.grad)
                    u = newton_schulz_orthogonalize(buf, steps=g["ns_steps"])
                    u = u * max(1.0, p.size(0) / p.size(1)) ** 0.5
                    p.add_(u.to(p.dtype), alpha=-g["lr"])
                else:
                    if "step" not in st:
                        st["step"] = torch.zeros((), dtype=torch.float32)
                        st["exp_avg"] = torch.zeros_like(p)
                        st["exp_avg_sq"] = torch.zeros_like(p)
                    st["step"] += 1
                    t = st["step"].item()
                    b1, b2 = g["betas"]
                    if g["weight_decay"]:
                        p.mul_(1 - g["lr"] * g["weight_decay"])
                    st["exp_avg"].lerp_(p.grad, 1 - b1)
                    st["exp_avg_sq"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    denom = (st["exp_avg_sq"] / (1 - b2 ** t)).sqrt_().add_(g["eps"])
                    p.addcdiv_(st["exp_avg"], denom, value=-g["lr"] / (1 - b1 ** t))


def make_muon_adamw(model, muon_lr, adam_lr, weight_decay, momentum=0.95):
    """Muon on 2-D weights inside the blocks; AdamW on the (tied) embedding, norms and biases."""
    muon, decay, no_decay, seen = [], [], [], set()
    emb = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    for n, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim == 2 and id(p) not in emb:
            muon.append(p)
        elif _no_decay(n, p):
            no_decay.append(p)
        else:
            decay.append(p)
    return MuonAdamW([
        {"params": muon, "use_muon": True, "lr": muon_lr, "momentum": momentum},
        {"params": decay, "lr": adam_lr, "weight_decay": weight_decay},
        {"params": no_decay, "lr": adam_lr, "weight_decay": 0.0},
    ])


def warmup_cosine(total_steps, warmup, final_frac):
    """LR multiplier: linear warm-up, then cosine decay to final_frac of the peak."""
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        t = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * t))
    return f
