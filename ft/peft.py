"""Fine-tuning / parameter-efficient adaptation methods, written once for every Phase-1 backbone.

    info = apply_method(backbone, "lorma", r=8, alpha=8, targets="attn")
    # backbone parameters are frozen except what the method trains; the task head stays trainable

Methods
  full         every backbone parameter trains
  partial      only the top `k` layers (+ final norm) train
  lora         y = W0 x + b + (alpha/r) B A x                      (Hu et al., 2021), B = 0 at init
  lorma        LoRMA+ : y = (I + (alpha/r) B A) W0 x + b          (Bihany, Patel & Modi, 2025)
               computed as h + (alpha/r) B(A h), h = W0 x; B = 0 at init (exact identity)
  lorma_pi     LoRMA-pi : y = rot(B A) W0 x + b, rot cyclically shifts row i of BA by i places;
               initialised so rot(BA) = I (B[:,0] = 1, A[0,0] = 1, rest 0)
  ia3          y = l * (W0 x + b) on the target outputs, l = 1 at init (Liu et al., 2022)
  bitfit       only bias terms train (Ben Zaken et al., 2022); Linear layers without a bias get a
               zero-initialised one (add_missing=True) so bias-free backbones can use it too
  houlsby      bottleneck adapter after the mixing sublayer and after the FFN of every layer
  pfeiffer     one bottleneck adapter per layer, after the FFN (or the mixer for Mamba blocks)
               adapter(h) = h + U act(D h), U = 0 at init                (Houlsby 2019, Pfeiffer 2020)
  rslora       LoRA with scale alpha / sqrt(r) instead of alpha / r     (Kalajdzievski, 2023)
  adalora      y = W0 x + b + s P diag(E) Q x, SVD-form update; orthogonality penalty on P, Q;
               singular values E pruned by smoothed sensitivity |E * dL/dE| to a shrinking total
               rank budget (cubic schedule)                            (Zhang et al., 2023)
  qlora        frozen base weight stored as 4-bit NF4 (blocks of 64, absmax scale), dequantised
               on the fly, + LoRA in full precision                   (Dettmers et al., 2023)
  prompt       P trainable input vectors prepended to the embeddings     (Lester et al., 2021)
  prefix       P trainable vectors at the input of every layer (deep prompts, P-Tuning v2 /
               prefix-tuning family; works for attention, retention and SSM layers alike)
  frozen       nothing in the backbone trains (linear probe)
  gradual      gradual unfreezing (Howard & Ruder, 2018): starts frozen; GradualUnfreezer makes one layer
               group trainable at a time, top first (final norm + top layer, ..., embeddings last)

Layer-wise LR decay (discriminative fine-tuning) is an optimiser setting, not a method: layer_depth()
gives each backbone parameter its depth so the trainer can scale its LR by decay ** (top - depth).

Targets for lora / lorma / ia3: "attn" = the sequence-mixing projections, "all" = those plus
the FFN projections; or an explicit list of leaf module names.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTN_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "g_proj", "in_proj", "out_proj"}
FFN_NAMES = {"gate_proj", "up_proj", "down_proj"}
IA3_NAMES = {"k_proj", "v_proj", "up_proj", "in_proj"}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0, rs=False):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / (math.sqrt(r) if rs else r)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), self.A.to(x.dtype)), self.B.to(x.dtype))

    def delta_weight(self):
        return self.scale * self.B @ self.A

    def merged(self):
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        lin.weight.data = self.base.weight.data + self.delta_weight().to(self.base.weight.dtype)
        if self.base.bias is not None:
            lin.bias.data = self.base.bias.data.clone()
        return lin


class LoRMALinear(nn.Module):
    """Multiplicative low-rank update of the output space: W = M W0 with M = I + s B A (plus) or rot(BA) (pi)."""

    def __init__(self, base: nn.Linear, r=8, alpha=8, mode="plus", dropout=0.0):
        super().__init__()
        assert mode in ("plus", "pi")
        self.base, self.r, self.mode = base, r, mode
        d = base.out_features
        self.scale = alpha / r if mode == "plus" else 1.0
        self.A = nn.Parameter(torch.empty(r, d))
        self.B = nn.Parameter(torch.zeros(d, r))
        if mode == "plus":
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        else:
            nn.init.zeros_(self.A)
            with torch.no_grad():
                self.B[:, 0] = 1.0
                self.A[0, 0] = 1.0
            idx = (torch.arange(d)[None, :] - torch.arange(d)[:, None]) % d
            self.register_buffer("rot_idx", idx, persistent=False)
        self.drop = nn.Dropout(dropout)

    def _wx(self, x):
        h = F.linear(x, self.base.weight)
        return h

    def multiplier(self):
        M = self.B @ self.A
        if self.mode == "plus":
            return torch.eye(M.shape[0], device=M.device, dtype=M.dtype) + self.scale * M
        return torch.gather(M, 1, self.rot_idx)

    def forward(self, x):
        h = self._wx(x)
        if self.mode == "plus":
            y = h + self.scale * F.linear(F.linear(self.drop(h), self.A.to(h.dtype)), self.B.to(h.dtype))
        else:
            R = torch.gather(self.B @ self.A, 1, self.rot_idx)
            y = F.linear(self.drop(h), R.to(h.dtype))
        if self.base.bias is not None:
            y = y + self.base.bias.to(y.dtype)
        return y

    def delta_weight(self):
        return self.multiplier() @ self.base.weight - self.base.weight

    def merged(self):
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        lin.weight.data = (self.multiplier() @ self.base.weight.float()).to(self.base.weight.dtype)
        if self.base.bias is not None:
            lin.bias.data = self.base.bias.data.clone()
        return lin


class IA3Linear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.scale = nn.Parameter(torch.ones(base.out_features))

    def forward(self, x):
        return self.base(x) * self.scale.to(x.dtype)


class Adapter(nn.Module):
    def __init__(self, d, bottleneck=64, dropout=0.0):
        super().__init__()
        self.down = nn.Linear(d, bottleneck)
        self.up = nn.Linear(bottleneck, d)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        return h + self.up(self.drop(F.gelu(self.down(h))))


class WithAdapter(nn.Module):
    """Runs a sublayer and applies an adapter to its output (before the residual add)."""

    def __init__(self, inner, d, bottleneck=64, dropout=0.0):
        super().__init__()
        self.inner = inner
        self.adapter = Adapter(d, bottleneck, dropout)

    def forward(self, *args, **kw):
        return self.adapter(self.inner(*args, **kw))

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["inner"], name)


class AdaLoRALinear(nn.Module):
    """W0 x + b + s * P diag(E * mask) Q x.  P: (out, r), E: (r,), Q: (r, in); E = 0 at init."""

    def __init__(self, base: nn.Linear, r=12, alpha=None, dropout=0.0):
        super().__init__()
        self.base, self.r = base, r
        self.scale = (alpha or r) / r
        self.P = nn.Parameter(torch.randn(base.out_features, r) * 0.02)
        self.E = nn.Parameter(torch.zeros(r))
        self.Q = nn.Parameter(torch.randn(r, base.in_features) * 0.02)
        self.register_buffer("mask", torch.ones(r))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        e = (self.E * self.mask).to(x.dtype)
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), self.Q.to(x.dtype)) * e, self.P.to(x.dtype))

    def orth_penalty(self):
        I = torch.eye(self.r, device=self.P.device)
        return ((self.P.t() @ self.P - I) ** 2).sum() + ((self.Q @ self.Q.t() - I) ** 2).sum()

    def delta_weight(self):
        return self.scale * self.P @ torch.diag(self.E * self.mask) @ self.Q

    def merged(self):
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        lin.weight.data = self.base.weight.data + self.delta_weight().to(self.base.weight.dtype)
        if self.base.bias is not None:
            lin.bias.data = self.base.bias.data.clone()
        return lin


class RankAllocator:
    """AdaLoRA budget schedule: full rank for t_init steps, cubic decrease of the total number of kept
    singular values to `target` by total - t_final, then fixed. Importance of each singular value =
    EMA of |E * grad E| (sensitivity) times EMA of its deviation (uncertainty), as in the paper."""

    def __init__(self, modules, target_r, total_steps, t_init_frac=0.1, t_final_frac=0.2, every=10,
                 beta1=0.85, beta2=0.85):
        self.mods = list(modules)
        self.n0 = sum(m.r for m in self.mods)
        self.target = target_r * len(self.mods)
        self.fracs = (t_init_frac, t_final_frac)
        self.retarget(total_steps)
        self.every, self.b1, self.b2 = every, beta1, beta2
        self.sens = [torch.zeros(m.r, device=m.E.device) for m in self.mods]
        self.unc = [torch.zeros(m.r, device=m.E.device) for m in self.mods]

    def retarget(self, total_steps):
        """New total step count (time-budgeted runs shrink their plan after measuring the speed)."""
        self.T = total_steps
        self.t_i, self.t_f = int(self.fracs[0] * total_steps), int(self.fracs[1] * total_steps)

    def budget(self, step):
        if step <= self.t_i:
            return self.n0
        if step >= self.T - self.t_f:
            return self.target
        frac = 1 - (step - self.t_i) / max(1, self.T - self.t_f - self.t_i)
        return int(self.target + (self.n0 - self.target) * frac ** 3)

    @torch.no_grad()
    def step(self, step):
        """Call after backward (gradients present), before optimizer.step()."""
        for i, m in enumerate(self.mods):
            if m.E.grad is None:
                continue
            s = (m.E * m.E.grad).abs()
            self.sens[i] = self.b1 * self.sens[i] + (1 - self.b1) * s
            self.unc[i] = self.b2 * self.unc[i] + (1 - self.b2) * (s - self.sens[i]).abs()
        if step % self.every or step <= self.t_i:
            return None
        b = self.budget(step)
        scores = torch.cat([self.sens[i] * self.unc[i] for i in range(len(self.mods))])
        if b >= scores.numel():
            return b
        thr = torch.kthvalue(scores.float().cpu(), scores.numel() - b).values.item()
        for i, m in enumerate(self.mods):
            keep = (self.sens[i] * self.unc[i] > thr).float()
            m.mask.copy_(keep)
            m.E.mul_(keep)
        return b


NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                    0.7229568362236023, 1.0])


def nf4_quantize(w, block=64):
    """w (out, in) -> (packed uint8 codes, fp16 absmax per block, shape). Nearest NF4 level per value."""
    flat = w.detach().float().reshape(-1)
    pad = (-flat.numel()) % block
    flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block)
    absmax = blocks.abs().max(1).values.clamp(min=1e-12)
    normed = blocks / absmax[:, None]
    codes = (normed[..., None] - NF4.to(w.device)).abs().argmin(-1).to(torch.uint8).view(-1)
    packed = (codes[0::2] << 4) | codes[1::2]
    return packed, absmax.half(), tuple(w.shape)


def nf4_dequantize(packed, absmax, shape, block=64, dtype=torch.float32):
    codes = torch.stack([packed >> 4, packed & 15], 1).view(-1).long()
    vals = NF4.to(packed.device)[codes].view(-1, block) * absmax.float()[:, None]
    n = shape[0] * shape[1]
    return vals.view(-1)[:n].view(shape).to(dtype)


class QLoRALinear(nn.Module):
    """4-bit NF4 frozen base (the fp32 weight is dropped) + LoRA adapters."""

    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0, block=64):
        super().__init__()
        packed, absmax, shape = nf4_quantize(base.weight, block)
        self.register_buffer("w_packed", packed)
        self.register_buffer("w_absmax", absmax)
        self.shape, self.block = shape, block
        self.in_features, self.out_features = base.in_features, base.out_features
        self.bias = None if base.bias is None else nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def weight(self, dtype=torch.float32):
        return nf4_dequantize(self.w_packed, self.w_absmax, self.shape, self.block, dtype)

    def forward(self, x):
        y = F.linear(x, self.weight(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
        return y + self.scale * F.linear(F.linear(self.drop(x), self.A.to(x.dtype)), self.B.to(x.dtype))


class GradualUnfreezer:
    """Gradual unfreezing (ULMFiT): the backbone starts frozen and the task head trains alone; as training
    progresses one more layer group becomes trainable every `frac` of the planned steps, top first
    (final norm + top layer, then the next layer down, ..., embeddings last), up to `max_groups` groups.
    The optimiser must already hold every parameter (frozen ones have no grad and are skipped)."""

    def __init__(self, bb, frac=0.08, max_groups=None, start_frac=0.0):
        layers = bb.layers()
        self.groups = [list(bb.final_norm().parameters()) + list(layers[-1].parameters())]
        self.groups += [list(blk.parameters()) for blk in reversed(layers[:-1])]
        self.groups.append(list(bb.embedding().parameters()))
        self.max = len(self.groups) if max_groups is None else max(1, min(max_groups, len(self.groups)))
        self.frac, self.start = max(frac, 1e-6), start_frac
        self.open = 0

    def params(self):
        return [p for g in self.groups[:self.max] for p in g]

    def step(self, step, total):
        """-> number of open groups after this step (call before the forward pass of `step`)."""
        prog = step / max(1, total)
        want = 0 if prog < self.start else min(self.max, 1 + int((prog - self.start) / self.frac))
        while self.open < want:
            for p in self.groups[self.open]:
                p.requires_grad_(True)
            self.open += 1
        return self.open


def layer_depth(bb, name):
    """Depth of backbone parameter `name` (relative to bb) for layer-wise LR decay: 0 = embeddings,
    i + 1 = layer i, n_layers + 1 = final norm, prompts and anything else on top."""
    import re
    m = re.search(r"\.(blocks|encoder|layers)\.(\d+)\.", "." + name)
    if m:
        return int(m.group(2)) + 1
    if "tok_emb" in name:
        return 0
    return len(bb.layers()) + 1


def peft_extra_loss(bb, adalora_orth=0.1):
    """Regularisers some methods need (AdaLoRA orthogonality); 0 otherwise."""
    mods = [m for m in bb.modules() if isinstance(m, AdaLoRALinear)]
    if not mods:
        return None
    return adalora_orth * sum(m.orth_penalty() for m in mods) / len(mods)


def _target_set(targets):
    if targets == "attn":
        return ATTN_NAMES
    if targets == "all":
        return ATTN_NAMES | FFN_NAMES
    return set(targets)


def _replace_linears(root, names, make):
    """Replace every nn.Linear whose leaf name is in `names` (inside root) with make(linear)."""
    n = 0
    for parent in list(root.modules()):
        if isinstance(parent, (LoRALinear, LoRMALinear, IA3Linear, AdaLoRALinear, QLoRALinear)):
            continue
        for leaf, child in list(parent.named_children()):
            if leaf in names and isinstance(child, nn.Linear):
                setattr(parent, leaf, make(child))
                n += 1
    return n


def _mixer_and_ffn(bb, blk):
    """(name of the sequence-mixing sublayer, name of the FFN sublayer or None) of one layer."""
    for mix in ("attn", "retention", "mixer", "self_attn"):
        if hasattr(blk, mix):
            break
    else:
        raise ValueError(f"no mixing sublayer in {type(blk).__name__}")
    ffn = "ffn" if hasattr(blk, "ffn") else ("moe" if hasattr(blk, "moe") else None)
    return mix, ffn


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, train


def apply_method(bb, method, r=8, alpha=None, targets="attn", k=4, bottleneck=64, n_prompt=16,
                 dropout=0.0, add_missing=True, init_ids=None, target_r=4):
    """Freeze/wrap the backbone in place. Returns {"method", "trainable", "total", ...}."""
    lm = bb.lm
    for p in bb.parameters():
        p.requires_grad_(False)
    info = {"method": method}
    names = _target_set(targets)
    if method == "full":
        for p in bb.parameters():
            p.requires_grad_(True)
    elif method == "frozen":
        pass
    elif method == "gradual":
        info["groups"] = len(bb.layers()) + 1
    elif method == "partial":
        for blk in bb.layers()[-k:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for p in bb.final_norm().parameters():
            p.requires_grad_(True)
        info["k"] = k
    elif method in ("lora", "rslora"):
        info["wrapped"] = _replace_linears(lm, names, lambda l: LoRALinear(l, r, alpha or 2 * r, dropout,
                                                                           rs=method == "rslora"))
    elif method == "adalora":
        info["wrapped"] = _replace_linears(lm, names, lambda l: AdaLoRALinear(l, r, alpha, dropout))
        info["target_r"] = target_r
    elif method == "qlora":
        info["wrapped"] = _replace_linears(lm, names, lambda l: QLoRALinear(l, r, alpha or 2 * r, dropout))
    elif method in ("lorma", "lorma_pi"):
        mode = "plus" if method == "lorma" else "pi"
        info["wrapped"] = _replace_linears(lm, names, lambda l: LoRMALinear(l, r, alpha or r, mode, dropout))
    elif method == "ia3":
        info["wrapped"] = _replace_linears(lm, IA3_NAMES if targets == "attn" else names, IA3Linear)
    elif method == "bitfit":
        added = 0
        for m in lm.modules():
            if isinstance(m, nn.Linear) and m.bias is None and add_missing and m is not getattr(lm, "lm_head", None):
                m.bias = nn.Parameter(torch.zeros(m.out_features, device=m.weight.device))
                added += 1
        for n_, p in lm.named_parameters():
            if n_.endswith("bias") or n_.endswith("dt_bias"):
                p.requires_grad_(True)
        info["added_biases"] = added
    elif method in ("houlsby", "pfeiffer"):
        d = bb.d_model
        n = 0
        for blk in bb.layers():
            mix, ffn = _mixer_and_ffn(bb, blk)
            spots = [ffn or mix] if method == "pfeiffer" else [s for s in (mix, ffn) if s]
            for s in spots:
                setattr(blk, s, WithAdapter(getattr(blk, s), d, bottleneck, dropout).to(next(blk.parameters()).device))
                n += 1
        info["adapters"] = n
        info["bottleneck"] = bottleneck
    elif method == "prompt":
        emb = bb.embedding().weight
        ids = init_ids if init_ids is not None else torch.randint(4, 32000, (n_prompt,))
        bb.prompt = nn.Parameter(emb[ids.to(emb.device)].detach().clone())
        info["n_prompt"] = n_prompt
    elif method == "prefix":
        ids = init_ids if init_ids is not None else torch.randint(4, 32000, (n_prompt,))
        dev = bb.embedding().weight.device
        init = bb.layer_inputs(ids.to(dev)[None])[:, 0]
        bb.deep_prompt = nn.Parameter(init.detach().clone())
        info["n_prompt"] = n_prompt
    else:
        raise ValueError(f"unknown method {method!r}")
    for m in bb.modules():
        if isinstance(m, (LoRALinear, LoRMALinear, QLoRALinear)):
            m.A.requires_grad_(True)
            m.B.requires_grad_(True)
        elif isinstance(m, AdaLoRALinear):
            for p_ in (m.P, m.E, m.Q):
                p_.requires_grad_(True)
        elif isinstance(m, IA3Linear):
            m.scale.requires_grad_(True)
        elif isinstance(m, Adapter):
            for p in m.parameters():
                p.requires_grad_(True)
    if bb.prompt is not None:
        bb.prompt.requires_grad_(True)
    if bb.deep_prompt is not None:
        bb.deep_prompt.requires_grad_(True)
    if method in ("lora", "rslora", "adalora", "qlora", "lorma", "lorma_pi", "ia3"):
        info["targets"] = sorted(names) if method != "ia3" or targets != "attn" else sorted(IA3_NAMES)
        info["r"] = r if method != "ia3" else None
    info["total"], info["trainable"] = count_params(bb)
    return info


def merge_all(bb):
    """Fold LoRA / LoRMA wrappers back into plain Linear layers (for inference / export)."""
    n = 0
    for parent in list(bb.modules()):
        for leaf, child in list(parent.named_children()):
            if isinstance(child, (LoRALinear, LoRMALinear, AdaLoRALinear)):
                setattr(parent, leaf, child.merged().to(child.base.weight.device))
                n += 1
    return n


def trainable_state(module):
    """state_dict restricted to trainable parameters (+ every buffer-free tensor of wrappers)."""
    keep = {n for n, p in module.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in module.state_dict().items() if k in keep}
