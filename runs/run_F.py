"""F: hybrid Mamba-2 + attention (Samba-style 1:1), ~85M parameters: 6 x [1 Mamba-2 block + 1 attention block]
(d 768; Mamba-2 as in D; attention block as in A). 1:1 rather than Jamba's 3:1 because round 1 measured
pure-PyTorch Mamba-2 at ~3.5k tok/s on a T4 (A: ~13.4k); 3:1 would have seen only ~165M tokens in a session."""
from phase1.models.hybrid import HybridAPI, HybridConfig, HybridLM
from phase1.models.optim import make_adamw, warmup_cosine

NAME = "F_hybrid"
KIND = "causal"
CFG = dict(
    batch_size=8,
    seq_len=1024,
    grad_accum=16,
    total_steps="auto",
    grad_clip=1.0,
    eval_every=200,
    eval_batches=4,
    eval_batch_size=8,
    token_milestones=[int(2.5e8), int(5e8)],
    mix=None,
    seed=1234,
)


def build_model(vocab_size):
    return HybridLM(HybridConfig(vocab_size=vocab_size, d_model=768, n_groups=6, mamba_per_group=1, n_heads=12,
                                 n_kv_heads=4, d_ff=2048, d_state=64, expand=2, headdim=64, chunk_size=64))


def make_optimizer(model):
    return make_adamw(model, lr=6e-4, weight_decay=0.1)


def make_lr_lambda(total_steps):
    return warmup_cosine(total_steps, warmup=min(300, max(1, total_steps // 20)), final_frac=0.1)


def decoder_api(model):
    return HybridAPI(model)
