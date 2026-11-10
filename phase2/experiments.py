"""Named L-NER experiments. Each is DEFAULTS updated by its own entries (and by --set overrides).

Backbone keys are ft.backbones.REGISTRY keys (A = A_decoder_cont, As = A_small, B = B_encdec
encoder, C = C_retnet, D = D_mamba2, G = G_mlm_encoder, ...).

Required-six map (course 8.2): A_full (full FT), A_lorma (LoRMA, compulsory), B_bitfit (BitFit),
C_pfeiffer (adapters), D_lora (LoRA), As_kd (knowledge distillation). Everything else is an
ablation or a candidate replacement for one of those rows.
"""
import copy

DEFAULTS = dict(
    split="edge",
    window=512,
    stride=256,
    batch=8, micro=0,
    eval_batch=8,
    epochs=20, patience=5, eval_every=1,
    lr=1e-4, head_lr=1e-3, weight_decay=0.01, warmup=0.06, grad_clip=1.0,
    dropout=0.1, head="linear", head_hidden=512, crf=False,
    bidir=False,
    echo=False,
    class_weight="none",
    o_weight=1.0,
    oversample=1,
    decode="viterbi",
    trim=True,
    dev_frac=0.15, seed=13, predict_test=True, save_soft=False, amp=True,
    method="full", method_args={}, kd=None, random_init=False,
    time_cap=40,
)

_E = {}


def _add(name, **kw):
    _E[name] = kw


_add("A_full", backbone="A", method="full", bidir=True, save_soft=True)
_add("A_full_causal", backbone="A", method="full", bidir=False)
_add("A_full_echo", backbone="A", method="full", echo=True)
_add("A_full_crf", backbone="A", method="full", bidir=True, crf=True)
_add("A_full_cw", backbone="A", method="full", bidir=True, class_weight="sqrt_inv", oversample=2)
_add("A_full_nosplit", backbone="A", method="full", bidir=True, split="none")
_add("A_full_w1024", backbone="A", method="full", bidir=True, window=1024, stride=512, batch=4)
_add("A_scratch", backbone="A", method="full", bidir=True, random_init=True)
_add("A_lorma", backbone="A", method="lorma", method_args=dict(r=16, alpha=16, targets="all"), lr=5e-4,
     bidir=True, save_soft=True)
_add("A_lorma_attn", backbone="A", method="lorma", method_args=dict(r=16, alpha=16, targets="attn"), lr=5e-4,
     bidir=True)
_add("A_lorma_pi", backbone="A", method="lorma_pi", method_args=dict(r=8, targets="attn"), lr=5e-4, bidir=True)
_add("A_lora", backbone="A", method="lora", method_args=dict(r=16, alpha=32, targets="all"), lr=5e-4, bidir=True)
_add("A_ia3", backbone="A", method="ia3", lr=3e-3, bidir=True)
_add("A_bitfit", backbone="A", method="bitfit", lr=3e-3, bidir=True)
_add("A_rslora", backbone="A", method="rslora", method_args=dict(r=16, alpha=16, targets="all"), lr=5e-4, bidir=True)
_add("A_adalora", backbone="A", method="adalora", method_args=dict(r=12, target_r=4, targets="all"), lr=1e-3,
     bidir=True)
_add("A_qlora", backbone="A", method="qlora", method_args=dict(r=16, alpha=32, targets="all"), lr=5e-4, bidir=True)
_add("A_houlsby", backbone="A", method="houlsby", method_args=dict(bottleneck=64), lr=1e-3, bidir=True)
_add("A_partial", backbone="A", method="partial", method_args=dict(k=4), lr=2e-4, bidir=True)
_add("A_prefix", backbone="A", method="prefix", method_args=dict(n_prompt=32), lr=5e-3, bidir=True)
_add("A_prompt", backbone="A", method="prompt", method_args=dict(n_prompt=32), lr=1e-2, bidir=True)
_add("A_frozen_bilstm", backbone="A", method="frozen", head="bilstm", head_lr=2e-3, bidir=False)

_add("B_bitfit", backbone="B", method="bitfit", lr=1e-3)
_add("B_full", backbone="B", method="full", save_soft=True)
_add("B_lorma", backbone="B", method="lorma", method_args=dict(r=16, alpha=16, targets="all"), lr=5e-4)
_add("B_lora", backbone="B", method="lora", method_args=dict(r=16, alpha=32, targets="all"), lr=5e-4)
_add("B_houlsby", backbone="B", method="houlsby", method_args=dict(bottleneck=64), lr=1e-3)

_add("C_pfeiffer", backbone="C", method="pfeiffer", method_args=dict(bottleneck=64), lr=1e-3, echo=True)
_add("C_full_echo", backbone="C", method="full", echo=True)
_add("C_full", backbone="C", method="full")
_add("C_lorma", backbone="C", method="lorma", method_args=dict(r=16, alpha=16, targets="all"), lr=5e-4, echo=True)
_add("C_lorma_bilstm", backbone="C", method="lorma", method_args=dict(r=16, alpha=16, targets="all"), lr=5e-4,
     head="bilstm")

_add("D_lora", backbone="D", method="lora", method_args=dict(r=16, alpha=32, targets="attn"), lr=5e-4, echo=True,
     micro=4, time_cap=60)
_add("D_full_echo", backbone="D", method="full", echo=True, micro=4)
_add("D_full", backbone="D", method="full", micro=4)
_add("D_lorma", backbone="D", method="lorma", method_args=dict(r=16, alpha=16, targets="attn"), lr=5e-4, echo=True,
     micro=4)
_add("D_bitfit", backbone="D", method="bitfit", lr=1e-3, echo=True, micro=4)
_add("D_lora_bilstm", backbone="D", method="lora", method_args=dict(r=16, alpha=32, targets="attn"), lr=5e-4,
     head="bilstm", micro=4)

_add("G_full", backbone="G", method="full", save_soft=True)
_add("G_lorma", backbone="G", method="lorma", method_args=dict(r=16, alpha=16, targets="all"), lr=5e-4)
_add("G_lora", backbone="G", method="lora", method_args=dict(r=16, alpha=32, targets="all"), lr=5e-4)

_add("As_full", backbone="As", method="full", bidir=True)
_add("As_kd", backbone="As", method="full", bidir=True, kd=dict(teacher="A_full", alpha=0.5, T=2.0))
_add("As_kd_G", backbone="As", method="full", bidir=True, kd=dict(teacher="G_full", alpha=0.5, T=2.0))

for _k in ("A", "B", "C", "D", "G", "As"):
    _add(f"smoke_{_k}", backbone=_k, method="lorma", method_args=dict(r=4, targets="attn"), lr=5e-4,
         epochs=1, window=128, stride=64, batch=2, bidir=_k in ("A", "As"), echo=_k in ("C", "D"),
         save_soft=_k == "A", save_weights=False)


def names():
    return sorted(_E)


def get_experiment(name, overrides=None):
    if name not in _E:
        raise KeyError(f"unknown experiment {name!r}; known: {', '.join(names())}")
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(copy.deepcopy(_E[name]))
    if overrides:
        for k, v in overrides.items():
            if k.startswith("kd.") and cfg.get("kd"):
                cfg["kd"][k[3:]] = v
            elif k.startswith("method_args."):
                cfg["method_args"][k[12:]] = v
            else:
                cfg[k] = v
    cfg["name"] = overrides.get("name", name) if overrides else name
    return cfg
