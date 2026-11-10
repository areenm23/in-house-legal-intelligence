"""Named legal-summarisation experiments: DEFAULTS updated by each entry (and by --set overrides).

Backbones (full LMs, ft.backbones.REGISTRY keys): A = A_decoder_cont, As = A_small, B = B_encdec (encoder +
decoder), C = C_retnet, D = D_mamba2. G (MLM encoder) has no decoder and is not used.

Phase-6 technique list (course 12.2.1): Houlsby adapters, Pfeiffer adapters, prefix tuning, prompt tuning,
LoRA, QLoRA, AdaLoRA, LoRMA (compulsory), knowledge distillation. Adapters only on A / As / B (see
genlm.py). Required-six map (provisional):
  A_lorma (LoRMA) | B_lora (LoRA) | C_qlora (QLoRA) | D_adalora (AdaLoRA) | B_houlsby (Houlsby) | As_kd (KD)
"""
import copy

DEFAULTS = dict(
    ctx="central",
    budget=1536,
    head_tokens=256, lead_frac=0.25,
    max_target=1024,
    batch=8, batch_tokens=12000,
    epochs=4, budget_min=45, probe_steps=10, n_evals=5, patience=2, reserve_min=18, log_every=20,
    eval_docs=48,
    gen_dev_docs=100,
    max_train_docs=None,
    lr=5e-4, weight_decay=0.01, warmup=0.05, grad_clip=1.0, smoothing=0.0,
    len_ratio=None,
    len_scale=1.0,
    len_min=128, len_max=1024, min_frac=0.8, max_frac=1.2,
    temperature=0.0, top_p=1.0, no_repeat_ngram=4, repetition_penalty=1.0, gen_batch=16, fallback_words=150,
    bertscore=True, predict_test=True, amp=True, seed=13,
    method="lora", method_args={}, kd=None, random_init=False,
)

_E = {}


def _add(name, **kw):
    _E[name] = kw


LORA = dict(r=16, alpha=32, targets="all")
LORMA = dict(r=16, alpha=16, targets="all")
ADALORA = dict(r=12, target_r=4, targets="all")
ADAPTER = dict(bottleneck=64)
B_IN = dict(budget=2048)
SLOW = dict(budget=1024, max_target=512, batch_tokens=6000, budget_min=55, reserve_min=30, gen_batch=24)
C_FIT = dict(budget=1024, max_target=512, batch_tokens=3100)

_add("A_lorma", backbone="A", method="lorma", method_args=LORMA, lr=5e-4)
_add("B_lora", backbone="B", method="lora", method_args=LORA, lr=5e-4, **B_IN)
_add("C_qlora", backbone="C", method="qlora", method_args=LORA, lr=5e-4, budget_min=50, **C_FIT)
_add("D_adalora", backbone="D", method="adalora", method_args=dict(r=12, target_r=4, targets="attn"), lr=1e-3, **SLOW)
_add("B_houlsby", backbone="B", method="houlsby", method_args=ADAPTER, lr=1e-3, **B_IN)
_add("As_kd", backbone="As", method="full", lr=1e-4, budget_min=40, kd=dict(teacher="A_lorma", alpha=0.5, T=2.0))

LONG = dict(epochs=8, budget_min=120)
_add("A_lorma_long", backbone="A", method="lorma", method_args=LORMA, lr=5e-4, **LONG)
_add("B_lora_long", backbone="B", method="lora", method_args=LORA, lr=5e-4, **B_IN, **LONG)
_add("C_qlora_long", backbone="C", method="qlora", method_args=LORA, lr=5e-4, **LONG, **C_FIT)
_add("D_adalora_long", backbone="D", method="adalora", method_args=dict(r=12, target_r=4, targets="attn"), lr=1e-3,
     **{**SLOW, "epochs": 8, "budget_min": 100})
_add("B_houlsby_long", backbone="B", method="houlsby", method_args=ADAPTER, lr=1e-3, **B_IN, **LONG)
_add("As_kd_long", backbone="As", method="full", lr=1e-4, epochs=8, budget_min=90,
     kd=dict(teacher="A_lorma_long", alpha=0.5, T=2.0))
_add("A_full_long", backbone="A", method="full", lr=5e-5, save_weights=False, **LONG)

_add("A_lora", backbone="A", method="lora", method_args=LORA, lr=5e-4)
_add("A_qlora", backbone="A", method="qlora", method_args=LORA, lr=5e-4)
_add("A_adalora", backbone="A", method="adalora", method_args=ADALORA, lr=1e-3)
_add("A_houlsby", backbone="A", method="houlsby", method_args=ADAPTER, lr=1e-3)
_add("A_pfeiffer", backbone="A", method="pfeiffer", method_args=ADAPTER, lr=1e-3)
_add("A_full", backbone="A", method="full", lr=5e-5, save_weights=False)
_add("B_lorma", backbone="B", method="lorma", method_args=LORMA, lr=5e-4, **B_IN)
_add("B_adalora", backbone="B", method="adalora", method_args=ADALORA, lr=1e-3, **B_IN)
_add("B_qlora", backbone="B", method="qlora", method_args=LORA, lr=5e-4, **B_IN)
_add("B_pfeiffer", backbone="B", method="pfeiffer", method_args=ADAPTER, lr=1e-3, **B_IN)
_add("C_lora", backbone="C", method="lora", method_args=LORA, lr=5e-4, budget_min=50, **C_FIT)
_add("C_lorma", backbone="C", method="lorma", method_args=LORMA, lr=5e-4, budget_min=50, **C_FIT)
_add("D_lora", backbone="D", method="lora", method_args=dict(r=16, alpha=32, targets="attn"), lr=5e-4, **SLOW)
_add("D_lorma", backbone="D", method="lorma", method_args=dict(r=16, alpha=16, targets="attn"), lr=5e-4, **SLOW)

_add("As_full", backbone="As", method="full", lr=1e-4, budget_min=40)
_add("As_kd_B", backbone="As", method="full", lr=1e-4, budget_min=40, kd=dict(teacher="B_lora", alpha=0.5, T=2.0))
_add("As_scratch", backbone="As", method="full", lr=3e-4, budget_min=40, random_init=True)

_AB = dict(backbone="B", method="lora", method_args=LORA, lr=5e-4, max_train_docs=3000, budget_min=25, **B_IN,
           gen_dev_docs=80)
_add("B_ab_central", **_AB)
_add("B_ab_head", **_AB, ctx="head")
_add("B_ab_headtail", **_AB, ctx="headtail", head_tokens=512)
_add("B_ab_oracle", **_AB, ctx="oracle")
_add("B_ab_dancer", **{**_AB, "ctx": "dancer", "budget": 1024})
_add("B_ab_1024", **{**_AB, "budget": 1024})

_SMOKE = dict(method="lora", method_args=dict(r=4, targets="attn"), lr=5e-4, epochs=1, budget=96, max_target=48,
              batch=4, batch_tokens=800, max_train_docs=24, eval_docs=6, gen_dev_docs=6, budget_min=1.0, n_evals=2,
              probe_steps=2, reserve_min=1, log_every=2, len_min=8, len_max=24, gen_batch=4, bertscore=False)
for _k in ("A", "B", "C", "D", "As"):
    _add(f"smoke_{_k}", backbone=_k, **_SMOKE)
_add("smoke_B_houlsby", backbone="B", **{**_SMOKE, "method": "houlsby", "method_args": dict(bottleneck=8)})
_add("smoke_A_dancer", backbone="A", **{**_SMOKE, "ctx": "dancer", "method": "lorma"})
_add("smoke_As_kd", backbone="As", **{**_SMOKE, "method": "full", "method_args": {}},
     kd=dict(teacher="smoke_A", alpha=0.5, T=2.0))


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
