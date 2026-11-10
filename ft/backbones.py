"""Phase-1 language models as feature extractors for the downstream phases.

    bb = load_backbone("A", roots=["/kaggle/input", "results"])     # newest checkpoint found under roots
    h = bb(ids)                    # (B, T) -> (B, T, d) contextual vectors (final norm applied)
    h = bb(ids, echo=True)         # causal models: read the window twice, return the second copy
    bb.set_bidirectional(True)     # attention models only: drop the causal mask

Keys (REGISTRY) map to the Phase-1 run file that builds the model and the run NAME whose
checkpoint is loaded. Checkpoints are found, in order of preference, as
  <root>/**/backbones/<NAME>.pt                 (fp16 weights-only export, export_backbones())
  <root>/**/runs/<NAME>/ckpt/step_*.pt          (Phase-1 training checkpoint, highest step)
  <root>/**/runs/<NAME>/ckpt/milestone_*.pt     (weights-only token milestone, largest)

Prompts (prompt / deep-prompt tuning, ft.peft) are handled here because they change the input
sequence: `self.prompt` (P, d) is prepended to the embeddings; `self.deep_prompt` (L, P, d)
replaces the first P positions at the input of every layer (P-Tuning v2 style prefix). The
returned hidden states never include prompt positions.
"""
import glob
import os
import re

import torch
import torch.nn as nn

from phase1.models.layers import Attention, build_rope_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB_SIZE = 32768

REGISTRY = {
    "A": ("runs/run_A_cont.py", "A_decoder_cont"),
    "A0": ("runs/run_A.py", "A_decoder"),
    "Amuon": ("runs/run_A_muon.py", "A_decoder_muon"),
    "Amuon_cont": ("runs/run_A_muon_cont.py", "A_decoder_muon_cont"),
    "As": ("runs/run_A_small.py", "A_small"),
    "B": ("runs/run_B.py", "B_encdec"),
    "B_cont": ("runs/run_B_cont.py", "B_encdec_cont"),
    "C": ("runs/run_C.py", "C_retnet"),
    "C_cont": ("runs/run_C_cont.py", "C_retnet_cont"),
    "D": ("runs/run_D.py", "D_mamba2"),
    "E": ("runs/run_E.py", "E_moe"),
    "F": ("runs/run_F.py", "F_hybrid"),
    "G": ("runs/run_G.py", "G_mlm_encoder"),
}


def family_of(model):
    from phase1.models.decoder import DecoderLM
    from phase1.models.encdec import EncoderDecoderLM
    from phase1.models.hybrid import HybridLM
    from phase1.models.mamba2 import Mamba2LM
    from phase1.models.mlm import EncoderMLM
    from phase1.models.moe import MoEDecoderLM
    from phase1.models.retnet import RetNetLM
    for cls, fam in ((DecoderLM, "decoder"), (MoEDecoderLM, "moe"), (EncoderDecoderLM, "encdec"),
                     (EncoderMLM, "mlm"), (RetNetLM, "retnet"), (Mamba2LM, "mamba"), (HybridLM, "hybrid")):
        if isinstance(model, cls):
            return fam
    raise TypeError(f"unknown backbone class {type(model).__name__}")


CAUSAL_FAMILIES = {"decoder", "moe", "retnet", "mamba", "hybrid"}
ATTENTION_FAMILIES = {"decoder", "moe", "hybrid", "encdec", "mlm"}


class Backbone(nn.Module):
    def __init__(self, lm, key="?", name="?", max_len=4096):
        super().__init__()
        self.key, self.name = key, name
        self.family = family_of(lm)
        if self.family == "encdec":
            lm.decoder = nn.ModuleList()
            lm.dec_norm = nn.Identity()
        self.lm = lm
        self.d_model = lm.cfg.d_model
        self.bidirectional = self.family in ("encdec", "mlm")
        self.prompt = None
        self.deep_prompt = None
        if hasattr(lm, "rope_cos") and lm.rope_cos.shape[0] < max_len:
            dh = lm.rope_cos.shape[1]
            base = getattr(lm.cfg, "rope_base", 10000.0)
            cos, sin = build_rope_cache(max_len, dh, base)
            lm.rope_cos, lm.rope_sin = cos, sin

    def layers(self):
        lm = self.lm
        if self.family == "encdec":
            return list(lm.encoder)
        if self.family == "hybrid":
            return list(lm.layers)
        return list(lm.blocks)

    def final_norm(self):
        return self.lm.enc_norm if self.family == "encdec" else self.lm.norm

    def embedding(self):
        return self.lm.tok_emb

    @property
    def is_causal(self):
        return not self.bidirectional

    def set_bidirectional(self, on=True):
        if self.family in ("encdec", "mlm"):
            if not on:
                raise ValueError(f"{self.family} encoder is always bidirectional")
            return
        if self.family not in ATTENTION_FAMILIES:
            raise ValueError(f"{self.family} has no attention mask to remove (use echo instead)")
        n = 0
        for m in self.lm.modules():
            if isinstance(m, Attention) and m.kind in ("causal", "bidir"):
                m.kind = "bidir" if on else "causal"
                n += 1
        self.bidirectional = on
        return n

    def _run_layer(self, blk, x, cs, kind=None):
        fam = self.family
        if fam in ("mamba",) or (fam == "hybrid" and kind == "mamba"):
            return blk(x)
        return blk(x, cs)

    def forward(self, ids, echo=False):
        """ids: (B, T) -> (B, T, d)."""
        T = ids.shape[1]
        if echo:
            ids = torch.cat([ids, ids], dim=1)
        x = self.lm.tok_emb(ids)
        P = 0
        if self.prompt is not None:
            P = self.prompt.shape[0]
            x = torch.cat([self.prompt.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1), x], dim=1)
        elif self.deep_prompt is not None:
            P = self.deep_prompt.shape[1]
            x = torch.cat([x.new_zeros(x.shape[0], P, x.shape[2]), x], dim=1)
        cs = (self.lm.rope_cos, self.lm.rope_sin) if hasattr(self.lm, "rope_cos") else None
        kinds = self.lm.kinds if self.family == "hybrid" else [None] * len(self.layers())
        for i, (blk, kind) in enumerate(zip(self.layers(), kinds)):
            if self.deep_prompt is not None:
                dp = self.deep_prompt[i].to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
                x = torch.cat([dp, x[:, P:]], dim=1)
            x = self._run_layer(blk, x, cs, kind)
        x = self.final_norm()(x)
        return x[:, x.shape[1] - T:]

    @torch.no_grad()
    def layer_inputs(self, ids):
        """Hidden state entering every layer, (L, B, T, d) (used to initialise deep prompts)."""
        x = self.lm.tok_emb(ids)
        cs = (self.lm.rope_cos, self.lm.rope_sin) if hasattr(self.lm, "rope_cos") else None
        kinds = self.lm.kinds if self.family == "hybrid" else [None] * len(self.layers())
        outs = []
        for blk, kind in zip(self.layers(), kinds):
            outs.append(x)
            x = self._run_layer(blk, x, cs, kind)
        return torch.stack(outs)


def _step(path):
    m = re.search(r"_(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def find_checkpoint(name, roots):
    for root in roots:
        hits = glob.glob(os.path.join(root, "**", "backbones", f"{name}.pt"), recursive=True)
        if hits:
            return sorted(hits)[0]
    for pattern in ("step_*.pt", "milestone_*.pt"):
        hits = []
        for root in roots:
            hits += glob.glob(os.path.join(root, "**", "runs", name, "ckpt", pattern), recursive=True)
        if hits:
            return max(hits, key=_step)
    return None


def build_lm(key):
    from phase1.train.launch import load_run
    run_file, name = REGISTRY[key]
    run = load_run(os.path.join(ROOT, run_file))
    assert run.NAME == name, (run.NAME, name)
    return run.build_model(VOCAB_SIZE), name


def tiny_lm(key):
    """Small random model of the same family as `key` (tests / CPU smoke runs)."""
    from phase1.models.decoder import DecoderConfig, DecoderLM
    from phase1.models.encdec import EncDecConfig, EncoderDecoderLM
    from phase1.models.hybrid import HybridConfig, HybridLM
    from phase1.models.mamba2 import Mamba2Config, Mamba2LM
    from phase1.models.mlm import EncoderMLM, MLMConfig
    from phase1.models.moe import MoEConfig, MoEDecoderLM
    from phase1.models.retnet import RetNetConfig, RetNetLM
    V, name = VOCAB_SIZE, REGISTRY[key][1]
    fam = key.rstrip("0").split("_")[0]
    if fam in ("A", "Amuon", "As"):
        return DecoderLM(DecoderConfig(V, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128)), name
    if fam == "B":
        return EncoderDecoderLM(EncDecConfig(V, d_model=64, n_enc_layers=2, n_dec_layers=1, n_heads=4, n_kv_heads=2,
                                             d_ff=128)), name
    if fam == "C":
        return RetNetLM(RetNetConfig(V, d_model=64, n_layers=2, n_heads=2, d_ff=128)), name
    if fam == "D":
        return Mamba2LM(Mamba2Config(V, d_model=64, n_layers=2, d_state=16, expand=2, headdim=16, chunk_size=16)), name
    if fam == "E":
        return MoEDecoderLM(MoEConfig(V, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, n_experts=4, top_k=2,
                                      d_ff_expert=64)), name
    if fam == "F":
        return HybridLM(HybridConfig(V, d_model=64, n_groups=2, mamba_per_group=1, n_heads=4, n_kv_heads=2, d_ff=128,
                                     d_state=16, headdim=16, chunk_size=16)), name
    if fam == "G":
        return EncoderMLM(MLMConfig(V, d_model=64, n_layers=2, n_heads=4, n_kv_heads=4, d_ff=128)), name
    raise KeyError(key)


def load_state(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    return {k: v.float() if torch.is_floating_point(v) else v for k, v in sd.items()}


def load_backbone(key, roots=(), ckpt=None, random_init=False, max_len=4096, tiny=False):
    """key from REGISTRY. random_init=True skips the checkpoint (from-scratch ablation); tiny=True
    builds a small random model of the same family (tests)."""
    lm, name = tiny_lm(key) if tiny else build_lm(key)
    src = None
    if not (random_init or tiny):
        src = ckpt or find_checkpoint(name, list(roots))
        if src is None:
            raise FileNotFoundError(f"no checkpoint for {key} ({name}) under {list(roots)}")
        missing, unexpected = lm.load_state_dict(load_state(src), strict=False)
        missing = [k for k in missing if not k.endswith(("rope_cos", "rope_sin"))]
        if missing or unexpected:
            raise RuntimeError(f"{name}: missing {missing[:5]} unexpected {unexpected[:5]}")
    bb = Backbone(lm, key, name, max_len=max_len)
    bb.ckpt_path = src
    return bb


def export_backbones(keys, roots, out_dir):
    """fp16 weights-only copies (half the size of fp32, no optimizer state) -> out_dir/<NAME>.pt."""
    os.makedirs(out_dir, exist_ok=True)
    done = {}
    for key in keys:
        _, name = REGISTRY[key]
        src = find_checkpoint(name, roots)
        if src is None:
            print(f"skip {key}: no checkpoint for {name}")
            continue
        sd = load_state(src)
        big = max(v.abs().max().item() for v in sd.values() if torch.is_floating_point(v))
        assert big < 6e4, f"{name}: weight {big} overflows fp16"
        half = {k: v.half() if torch.is_floating_point(v) else v for k, v in sd.items()}
        dst = os.path.join(out_dir, f"{name}.pt")
        torch.save({"model": half, "source": os.path.relpath(src, ROOT) if src.startswith(ROOT) else src,
                    "step": _step(src)}, dst)
        done[key] = {"name": name, "source": src, "bytes": os.path.getsize(dst)}
        print(f"{key}: {src} -> {dst} ({os.path.getsize(dst) / 1e6:.0f} MB)")
    return done
