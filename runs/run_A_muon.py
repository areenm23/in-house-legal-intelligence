"""A trained with Muon (Assignment-3 implementation) on the hidden matrices + AdamW on the rest.
Same architecture, data and token windows as run_A.py: an optimizer comparison."""
from phase1.models.decoder import DecoderConfig, DecoderLM, DecoderAPI
from phase1.models.optim import make_muon_adamw, warmup_cosine

NAME = "A_decoder_muon"
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
    return DecoderLM(DecoderConfig(vocab_size=vocab_size, d_model=768, n_layers=12, n_heads=12, n_kv_heads=4,
                                   d_ff=2048))


def make_optimizer(model):
    return make_muon_adamw(model, muon_lr=0.02, adam_lr=6e-4, weight_decay=0.1)


def make_lr_lambda(total_steps):
    return warmup_cosine(total_steps, warmup=min(300, max(1, total_steps // 20)), final_frac=0.1)


def decoder_api(model):
    return DecoderAPI(model)
