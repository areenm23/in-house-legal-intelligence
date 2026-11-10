"""A-small: the same decoder design at ~64M parameters (10 x 640): the student for later distillation."""
from phase1.models.decoder import DecoderConfig, DecoderLM, DecoderAPI
from phase1.models.optim import make_adamw, warmup_cosine

NAME = "A_small"
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
    return DecoderLM(DecoderConfig(vocab_size=vocab_size, d_model=640, n_layers=10, n_heads=10, n_kv_heads=2,
                                   d_ff=1728))


def make_optimizer(model):
    return make_adamw(model, lr=8e-4, weight_decay=0.1)


def make_lr_lambda(total_steps):
    return warmup_cosine(total_steps, warmup=min(300, max(1, total_steps // 20)), final_frac=0.1)


def decoder_api(model):
    return DecoderAPI(model)
