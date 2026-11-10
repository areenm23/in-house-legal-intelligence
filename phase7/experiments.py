"""Named prior-case-retrieval experiments: DEFAULTS updated by each entry (and by --set overrides).

Backbone keys are ft.backbones.REGISTRY keys (A = A_decoder_cont, As = A_small, B = B_encdec encoder, C = C_retnet,
D = D_mamba2, G = G_mlm_encoder).

Phase-7 technique list (course 13.2.1): full fine-tuning, partial / layer-wise fine-tuning, QLoRA, LoRMA
(compulsory), prefix tuning, prompt tuning, IA3, knowledge distillation. Pooling, similarity, hard negatives and the
BM25 hybrid are methodological choices (not techniques).

Required-six map (provisional):
  A_lorma (LoRMA) | B_full (full) | C_ia3 (IA3) | D_partial (partial) | G_qlora (QLoRA) | As_kd (KD, teacher G_lorma)
"""
import copy

DEFAULTS = dict(
    seq_len=512,
    train_chunks=2, train_pick="spread",
    eval_chunks=8, eval_pick="spread",
    eval_chunks_fast=2,
    pool="mean", agg="mean",
    proj_dim=256, dropout=0.0, tau=0.05,
    q_batch=6, n_hard=1, hard_pool=30, hard_skip=0,
    seq_batch=16, eval_batch=32,
    epochs=12, budget_min=35, probe_steps=10, n_evals=5, patience=3, reserve_min=12, log_every=20,
    max_train_queries=None,
    lr=2e-4, head_lr=1e-3, weight_decay=0.01, warmup=0.06, grad_clip=1.0,
    bm25_k1=1.2, bm25_b=0.75, bm25_max_tokens=None,
    bidir=False, seed=13, predict_test=True, save_emb=False, amp=True,
    method="lorma", method_args={}, kd=None, random_init=False,
)

_E = {}


def _add(name, **kw):
    _E[name] = kw


LORMA = dict(r=16, alpha=16, targets="all")
LORA = dict(r=16, alpha=32, targets="all")
PREFIX = dict(n_prompt=32)
CAUSAL = dict(pool="mean_last")
SLOW = dict(train_chunks=1, eval_chunks=4, budget_min=45, reserve_min=25)

_add("A_lorma", backbone="A", method="lorma", method_args=LORMA, lr=5e-4, bidir=True, save_emb=True)
_add("B_full", backbone="B", method="full", lr=3e-5)
_add("C_ia3", backbone="C", method="ia3", lr=3e-3, **CAUSAL)
_add("D_partial", backbone="D", method="partial", method_args=dict(k=6), lr=1e-4, **CAUSAL, **SLOW)
_add("G_qlora", backbone="G", method="qlora", method_args=LORA, lr=5e-4)
_add("As_kd", backbone="As", method="full", lr=5e-5, bidir=True, kd=dict(teacher="G_lorma", alpha=0.5, T=1.0))

_add("G_lorma", backbone="G", method="lorma", method_args=LORMA, lr=5e-4, save_emb=True)
_add("G_full", backbone="G", method="full", lr=3e-5)
_add("G_ia3", backbone="G", method="ia3", lr=3e-3)
_add("G_prefix", backbone="G", method="prefix", method_args=PREFIX, lr=5e-3)
_add("G_partial", backbone="G", method="partial", method_args=dict(k=4), lr=1e-4)
_add("A_full", backbone="A", method="full", lr=3e-5, bidir=True)
_add("A_qlora", backbone="A", method="qlora", method_args=LORA, lr=5e-4, bidir=True)
_add("A_ia3", backbone="A", method="ia3", lr=3e-3, bidir=True)
_add("A_prefix", backbone="A", method="prefix", method_args=PREFIX, lr=5e-3, bidir=True)
_add("A_prompt", backbone="A", method="prompt", method_args=PREFIX, lr=1e-2, bidir=True)
_add("A_frozen", backbone="A", method="frozen", bidir=True, budget_min=15)
_add("A_scratch", backbone="A", method="full", lr=3e-4, bidir=True, random_init=True)
_add("B_lorma", backbone="B", method="lorma", method_args=LORMA, lr=5e-4)
_add("B_ia3", backbone="B", method="ia3", lr=3e-3)
_add("C_lorma", backbone="C", method="lorma", method_args=LORMA, lr=5e-4, **CAUSAL)
_add("C_prompt", backbone="C", method="prompt", method_args=PREFIX, lr=1e-2, **CAUSAL)
_add("D_lorma", backbone="D", method="lorma", method_args=dict(r=16, alpha=16, targets="attn"), lr=5e-4, **CAUSAL, **SLOW)
_add("As_full", backbone="As", method="full", lr=5e-5, bidir=True)
_add("As_kd_A", backbone="As", method="full", lr=5e-5, bidir=True, kd=dict(teacher="A_lorma", alpha=0.5, T=1.0))

_AB = dict(backbone="G", method="lorma", method_args=LORMA, lr=5e-4, budget_min=25)
_add("G_ab_ref", **_AB)
_add("G_ab_nohard", **_AB, n_hard=0)
_add("G_ab_hard3", **_AB, n_hard=3, q_batch=4)
_add("G_ab_first", **_AB, train_pick="first", eval_pick="first")
_add("G_ab_maxpool", **_AB, pool="max")
_add("G_ab_attnagg", **_AB, agg="attn")
_add("G_ab_1chunk", **_AB, train_chunks=1)

_SMOKE = dict(method="lorma", method_args=dict(r=4, targets="attn"), lr=5e-4, epochs=2, seq_len=64, train_chunks=2,
              eval_chunks=3, eval_chunks_fast=2, q_batch=3, seq_batch=8, eval_batch=16, budget_min=1.0, n_evals=2,
              probe_steps=2, reserve_min=1, log_every=2, save_weights=False)
for _k in ("A", "B", "C", "D", "G", "As"):
    _add(f"smoke_{_k}", backbone=_k, **_SMOKE, bidir=_k in ("A", "As"), **(CAUSAL if _k in ("C", "D") else {}))
_E["smoke_G"]["save_emb"] = True
_add("smoke_As_kd", backbone="As", **{**_SMOKE, "method": "full", "method_args": {}}, bidir=True,
     kd=dict(teacher="smoke_G", alpha=0.5, T=1.0))


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
