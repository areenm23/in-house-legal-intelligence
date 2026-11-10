"""G: bidirectional encoder (ModernBERT-style) trained with masked LM, ~110M parameters
(12 x 768, 12 heads, SwiGLU 2048). 30% random masking on 75% of rows, a fully masked suffix of
32-256 tokens on the other 25% (so it can write continuations by filling masks left to right)."""
from phase1.models.mlm import EncoderMLM, MLMConfig, MLMFillAPI, mask_id_for, mlm_batch, mlm_loss
from phase1.models.optim import make_adamw, warmup_cosine

NAME = "G_mlm_encoder"
KIND = "mlm"
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

MASK_PROB, SUFFIX_FRAC, SUFFIX_SPAN = 0.3, 0.25, (32, 256)
DECODES = ["decode/mlm_fill_greedy.json"]


def build_model(vocab_size):
    return EncoderMLM(MLMConfig(vocab_size=vocab_size, d_model=768, n_layers=12, n_heads=12, n_kv_heads=12,
                                d_ff=2048))


def train_batch(shards, index, cfg=CFG):
    return mlm_batch(shards, cfg["batch_size"], cfg["seq_len"], cfg["seed"], index, cfg["mix"],
                     mask_id_for(shards.meta["vocab_size"]), MASK_PROB, SUFFIX_FRAC, SUFFIX_SPAN)


def eval_batch(shards, seed, index, mix, cfg=CFG):
    return mlm_batch(shards, cfg["eval_batch_size"], cfg["seq_len"], seed, index, mix,
                     mask_id_for(shards.meta["vocab_size"]), MASK_PROB, SUFFIX_FRAC, SUFFIX_SPAN)


loss_fn = mlm_loss


def make_optimizer(model):
    return make_adamw(model, lr=6e-4, weight_decay=0.1)


def make_lr_lambda(total_steps):
    return warmup_cosine(total_steps, warmup=min(300, max(1, total_steps // 20)), final_frac=0.1)


def decoder_api(model):
    return MLMFillAPI(model, n_masks=224, block=4)
