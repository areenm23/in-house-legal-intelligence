"""Named legal-statute-identification experiments: DEFAULTS updated by each entry (and by --set overrides).

Backbone keys are ft.backbones.REGISTRY keys (A = A_decoder_cont, As = A_small, B = B_encdec encoder,
C = C_retnet, D = D_mamba2, G = G_mlm_encoder).

Phase-5 technique list (course 11.2.1): full fine-tuning, partial / layer-wise fine-tuning, QLoRA, LoRMA
(compulsory), AdaLoRA, adapters, prefix tuning, prompt tuning, knowledge distillation. Here: full |
partial (top-k layers; llrd = layer-wise LR) | qlora | lorma | adalora | houlsby | prefix | prompt | KD
(As_* with kd=...). Gradual unfreezing and BitFit are not on this list.

Required-six map (provisional, the grids confirm it):
  A_lorma (LoRMA) | B_full (full) | C_houlsby (adapters) | D_partial (partial) | G_qlora (QLoRA) | As_kd (KD)
"""
import copy

from phase4.experiments import DEFAULTS as P4_DEFAULTS

DEFAULTS = dict(P4_DEFAULTS)
DEFAULTS.update(
    ctx="first",
    eval_docs=3000,
    loss="bce",
    pos_weight="sqrt",
    pos_weight_cap=30.0,
    focal=2.0,
    asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05,
    label_mix=False,
    class_weight=None,
)

_E = {}


def _add(name, **kw):
    _E[name] = kw


LORMA = dict(r=16, alpha=16, targets="all")
LORA = dict(r=16, alpha=32, targets="all")
ADALORA = dict(r=12, target_r=4, targets="all")
ADAPTER = dict(bottleneck=64)
PREFIX = dict(n_prompt=32)
EXTRACT = dict(max_chunks=16, train_docs=16000)
L1024 = dict(seq_len=1024, seq_batch=8, eval_batch=16)
SLOW = dict(eval_docs=1500, n_evals=4)
CAUSAL = dict(pool="mean_last")

_add("A_lorma", backbone="A", method="lorma", method_args=LORMA, lr=5e-4, bidir=True, save_soft=True, extract=EXTRACT)
_add("B_full", backbone="B", method="full", lr=5e-5, budget_min=30)
_add("C_houlsby", backbone="C", method="houlsby", method_args=ADAPTER, lr=1e-3, budget_min=45, **L1024, **CAUSAL)
_add("D_partial", backbone="D", method="partial", method_args=dict(k=6), lr=1e-4, budget_min=50, **L1024, **CAUSAL,
     **SLOW)
_add("G_qlora", backbone="G", method="qlora", method_args=LORA, lr=5e-4)
_add("As_kd", backbone="As", method="full", lr=5e-5, bidir=True, budget_min=30,
     kd=dict(teacher="A_lorma", alpha=0.5, T=2.0))

_add("G_lorma", backbone="G", method="lorma", method_args=LORMA, lr=5e-4, save_soft=True, extract=EXTRACT)
_add("G_full", backbone="G", method="full", lr=3e-5)
_add("G_adalora", backbone="G", method="adalora", method_args=ADALORA, lr=1e-3)
_add("G_prefix", backbone="G", method="prefix", method_args=PREFIX, lr=5e-3)
_add("G_houlsby", backbone="G", method="houlsby", method_args=ADAPTER, lr=1e-3)
_add("A_full", backbone="A", method="full", lr=3e-5, bidir=True)
_add("A_qlora", backbone="A", method="qlora", method_args=LORA, lr=5e-4, bidir=True)
_add("A_adalora", backbone="A", method="adalora", method_args=ADALORA, lr=1e-3, bidir=True)
_add("A_partial", backbone="A", method="partial", method_args=dict(k=4), lr=1e-4, bidir=True)
_add("A_llrd", backbone="A", method="full", lr=1e-4, llrd=0.85, bidir=True)
_add("A_houlsby", backbone="A", method="houlsby", method_args=ADAPTER, lr=1e-3, bidir=True)
_add("A_prefix", backbone="A", method="prefix", method_args=PREFIX, lr=5e-3, bidir=True)
_add("A_prompt", backbone="A", method="prompt", method_args=PREFIX, lr=1e-2, bidir=True)
_add("A_scratch", backbone="A", method="full", lr=3e-4, bidir=True, random_init=True)
_add("A_frozen", backbone="A", method="frozen", bidir=True, budget_min=15)
_add("B_lorma", backbone="B", method="lorma", method_args=LORMA, lr=5e-4, budget_min=30)
_add("B_houlsby", backbone="B", method="houlsby", method_args=ADAPTER, lr=1e-3, budget_min=30)
_add("B_prefix", backbone="B", method="prefix", method_args=PREFIX, lr=5e-3, budget_min=30)
_add("B_qlora", backbone="B", method="qlora", method_args=LORA, lr=5e-4, budget_min=30)
_add("C_lorma", backbone="C", method="lorma", method_args=LORMA, lr=5e-4, budget_min=45, **L1024, **CAUSAL)
_add("C_prompt", backbone="C", method="prompt", method_args=PREFIX, lr=1e-2, budget_min=45, **L1024, **CAUSAL)
_add("C_prefix", backbone="C", method="prefix", method_args=PREFIX, lr=5e-3, budget_min=45, **L1024, **CAUSAL)
_add("C_houlsby_512", backbone="C", method="houlsby", method_args=ADAPTER, lr=1e-3, budget_min=45, **CAUSAL)
_add("D_lorma", backbone="D", method="lorma", method_args=dict(r=16, alpha=16, targets="attn"), lr=5e-4,
     budget_min=50, **CAUSAL, **SLOW)
_add("D_prefix", backbone="D", method="prefix", method_args=PREFIX, lr=5e-3, budget_min=50, **CAUSAL, **SLOW)
_add("D_prompt", backbone="D", method="prompt", method_args=PREFIX, lr=1e-2, budget_min=50, **CAUSAL, **SLOW)
_add("D_partial_512", backbone="D", method="partial", method_args=dict(k=6), lr=1e-4, budget_min=50, **CAUSAL, **SLOW)

_add("As_full", backbone="As", method="full", lr=5e-5, bidir=True, budget_min=30)
_add("As_kd_G", backbone="As", method="full", lr=5e-5, bidir=True, budget_min=30,
     kd=dict(teacher="G_lorma", alpha=0.5, T=2.0))
_add("As_kd_feat", backbone="As", method="full", lr=5e-5, bidir=True, budget_min=30,
     kd=dict(teacher="A_lorma", alpha=0.5, T=2.0, feat=0.5))

_AB = dict(backbone="G", method="lorma", method_args=LORMA, lr=5e-4, max_train_docs=16000, budget_min=22, **SLOW)
_add("G_ab_first", **_AB)
_add("G_ab_last", **_AB, ctx="last")
_add("G_ab_middle", **_AB, ctx="middle")
_add("G_ab_headtail", **_AB, ctx="headtail")
_add("G_ab_first1024", **_AB, **L1024)
_add("G_ab_random2", **_AB, ctx="hier", n_chunks=2, eval_chunks=8, chunk_pick="random", eval_pick="spread")
_add("G_ab_spread4_gru", **_AB, ctx="hier", n_chunks=4, eval_chunks=8, chunk_pick="spread", eval_pick="spread", agg="gru")
_add("G_ab_nopw", **_AB, pos_weight="none")
_add("G_ab_focal", **_AB, loss="focal", pos_weight="none")
_add("G_ab_asl", **_AB, loss="asl", pos_weight="none")
_add("G_ab_mix", **_AB, label_mix=True)

_SMOKE = dict(method="lorma", method_args=dict(r=4, targets="attn"), lr=5e-4, epochs=1, seq_len=128, batch=8,
              seq_batch=8, eval_batch=16, max_train_docs=64, eval_docs=32, budget_min=1.5, n_evals=2, probe_steps=3,
              reserve_min=1, log_every=2, save_weights=False)
for _k in ("A", "B", "C", "D", "G", "As"):
    _add(f"smoke_{_k}", backbone=_k, **_SMOKE, bidir=_k in ("A", "As"), **(CAUSAL if _k in ("C", "D") else {}))
_E["smoke_G"].update(save_soft=True, extract=dict(max_chunks=4, train_docs=32))
_add("smoke_G_hier", backbone="G", **{**_SMOKE, "ctx": "hier", "n_chunks": 3, "eval_chunks": 4, "agg": "gru",
                                      "label_mix": True, "loss": "asl"})
_add("smoke_As_kd", backbone="As", **{**_SMOKE, "method": "full", "method_args": {}}, bidir=True,
     kd=dict(teacher="smoke_G", alpha=0.5, T=2.0, feat=0.5))


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
            elif k.startswith("extract.") and cfg.get("extract"):
                cfg["extract"][k[8:]] = v
            else:
                cfg[k] = v
    cfg["name"] = overrides.get("name", name) if overrides else name
    return cfg
