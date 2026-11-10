"""Named rhetorical-role experiments: DEFAULTS updated by each entry (and by --set overrides).

Backbone keys are ft.backbones.REGISTRY keys (A = A_decoder_cont, As = A_small, B = B_encdec encoder,
C = C_retnet, D = D_mamba2, G = G_mlm_encoder).

Phase-3 technique list (course 9.2.1): LoRA, QLoRA, AdaLoRA, rsLoRA, LoRMA (compulsory), prefix,
prompt, IA3, KD. Full fine-tuning is not on the list: *_full runs are reference points only
("can PEFT approach full adaptation?").

Required-six map (provisional, the grid confirms it):
  A_lorma (LoRMA) | B_rslora (rsLoRA) | C_ia3 (IA3) | D_lora (LoRA) | G_adalora (AdaLoRA) | As_kd (KD)
"""
import copy

DEFAULTS = dict(
    ctx="win",
    window=512, stride=256,
    chunk_tokens=2048,
    max_sent_tokens=256,
    pool="mean",
    head="bilstm", head_hidden=512, crf=False, dropout=0.1,
    pos_feat=True,
    dom_feat=True,
    train_domains=["CL", "IT"],
    bidir=False,
    echo=False,
    batch=4, eval_batch=8,
    epochs=15, patience=4, eval_every=1,
    lr=1e-4, head_lr=1e-3, weight_decay=0.01, warmup=0.06, grad_clip=1.0,
    class_weight="sqrt_inv",
    focal=0.0, smoothing=0.0,
    tune_bias=True,
    seed=13, predict_test=True, save_soft=False, amp=True,
    method="full", method_args={}, kd=None, random_init=False,
    time_cap=35,
)

_E = {}


def _add(name, **kw):
    _E[name] = kw


LORA = dict(r=16, alpha=32, targets="all")
LORMA = dict(r=16, alpha=16, targets="all")

_add("A_lorma", backbone="A", method="lorma", method_args=LORMA, lr=5e-4, bidir=True, save_soft=True)
_add("B_rslora", backbone="B", method="rslora", method_args=dict(r=16, alpha=16, targets="all"), lr=5e-4)
_add("C_ia3", backbone="C", method="ia3", lr=3e-3)
_add("D_lora", backbone="D", method="lora", method_args=dict(r=16, alpha=32, targets="attn"), lr=5e-4,
     chunk_tokens=1024, time_cap=45)
_add("G_adalora", backbone="G", method="adalora", method_args=dict(r=12, target_r=4, targets="all"), lr=1e-3)
_add("As_kd", backbone="As", method="full", bidir=True, kd=dict(teacher="G_lora", alpha=0.5, T=2.0))

_add("G_lora", backbone="G", method="lora", method_args=LORA, lr=5e-4, save_soft=True)
_add("G_lorma", backbone="G", method="lorma", method_args=LORMA, lr=5e-4, save_soft=True)
_add("G_rslora", backbone="G", method="rslora", method_args=dict(r=16, alpha=16, targets="all"), lr=5e-4)
_add("G_ia3", backbone="G", method="ia3", lr=3e-3)
_add("G_prefix", backbone="G", method="prefix", method_args=dict(n_prompt=32), lr=5e-3)
_add("G_full", backbone="G", method="full", save_soft=True)
_add("A_lora", backbone="A", method="lora", method_args=LORA, lr=5e-4, bidir=True)
_add("A_qlora", backbone="A", method="qlora", method_args=LORA, lr=5e-4, bidir=True)
_add("A_adalora", backbone="A", method="adalora", method_args=dict(r=12, target_r=4, targets="all"), lr=1e-3,
     bidir=True)
_add("A_ia3", backbone="A", method="ia3", lr=3e-3, bidir=True)
_add("A_prefix", backbone="A", method="prefix", method_args=dict(n_prompt=32), lr=5e-3, bidir=True)
_add("A_prompt", backbone="A", method="prompt", method_args=dict(n_prompt=32), lr=1e-2, bidir=True)
_add("A_full", backbone="A", method="full", bidir=True, save_soft=True)
_add("A_scratch", backbone="A", method="full", bidir=True, random_init=True)
_add("A_frozen", backbone="A", method="frozen", bidir=False)
_add("B_lorma", backbone="B", method="lorma", method_args=LORMA, lr=5e-4)
_add("B_lora", backbone="B", method="lora", method_args=LORA, lr=5e-4)
_add("C_lorma", backbone="C", method="lorma", method_args=LORMA, lr=5e-4)
_add("C_lora", backbone="C", method="lora", method_args=LORA, lr=5e-4)
_add("C_prompt", backbone="C", method="prompt", method_args=dict(n_prompt=32), lr=1e-2)
_add("D_lorma", backbone="D", method="lorma", method_args=dict(r=16, alpha=16, targets="attn"), lr=5e-4,
     chunk_tokens=1024, time_cap=45)
_add("D_ia3", backbone="D", method="ia3", lr=3e-3, chunk_tokens=1024, time_cap=45)
_add("D_prefix", backbone="D", method="prefix", method_args=dict(n_prompt=32), lr=5e-3, chunk_tokens=1024,
     time_cap=45)

_G = dict(backbone="G", method="lora", method_args=LORA, lr=5e-4)
_add("G_lora_sent", **_G, ctx="sent", head="linear", pos_feat=False)
_add("G_lora_sent_bilstm", **_G, ctx="sent")
_add("G_lora_win_linear", **_G, head="linear", pos_feat=False)
_add("G_lora_crf", **_G, crf=True)
_add("G_lora_attnpool", **_G, pool="attn")
_add("G_lora_w1024", **_G, window=1024, stride=512, chunk_tokens=3072)
_add("G_lora_nocw", **_G, class_weight="none")
_add("G_lora_focal", **_G, focal=2.0)
_add("G_lora_CL", **_G, train_domains=["CL"])
_add("G_lora_IT", **_G, train_domains=["IT"])
_add("A_lorma_causal", backbone="A", method="lorma", method_args=LORMA, lr=5e-4, bidir=False)
_add("A_lorma_sent", backbone="A", method="lorma", method_args=LORMA, lr=5e-4, bidir=True, ctx="sent",
     head="linear", pos_feat=False)

_add("As_full", backbone="As", method="full", bidir=True)
_add("As_kd_A", backbone="As", method="full", bidir=True, kd=dict(teacher="A_lorma", alpha=0.5, T=2.0))
_add("As_lora_kd", backbone="As", method="lora", method_args=LORA, lr=5e-4, bidir=True,
     kd=dict(teacher="G_lora", alpha=0.5, T=2.0))

for _k in ("A", "B", "C", "D", "G", "As"):
    _add(f"smoke_{_k}", backbone=_k, method="lorma", method_args=dict(r=4, targets="attn"), lr=5e-4,
         epochs=1, window=128, stride=64, chunk_tokens=256, batch=2, bidir=_k in ("A", "As"),
         save_soft=_k == "G", save_weights=False, time_cap=10)
_add("smoke_As_kd", backbone="As", method="full", bidir=True, epochs=1, window=128, stride=64, chunk_tokens=256,
     batch=2, kd=dict(teacher="smoke_G", alpha=0.5, T=2.0), save_weights=False, time_cap=10)


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
