"""B: encoder-decoder Transformer, continuation objective, ~138M parameters (8 enc + 8 dec x 768, biases on)."""
from phase1.models.encdec import EncDecConfig, EncoderDecoderLM, EncDecAPI
from phase1.models.optim import make_adamw, warmup_cosine

NAME = "B_encdec"
KIND = "encdec"
CFG = dict(
    batch_size=8,
    src_len=768,
    tgt_len=256,
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
    return EncoderDecoderLM(EncDecConfig(vocab_size=vocab_size, d_model=768, n_enc_layers=8, n_dec_layers=8,
                                         n_heads=12, n_kv_heads=4, d_ff=2048, bias=True))


def make_optimizer(model):
    return make_adamw(model, lr=6e-4, weight_decay=0.1)


def make_lr_lambda(total_steps):
    return warmup_cosine(total_steps, warmup=min(300, max(1, total_steps // 20)), final_frac=0.1)


def decoder_api(model):
    return EncDecAPI(model)
