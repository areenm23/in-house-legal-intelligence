"""E: sparse Mixture-of-Experts decoder, ~271M parameters in total, ~101M active per token
(12 x 768, GQA 12q/4kv; 8 SwiGLU experts of 1024 per layer, top-2 routing). Active compute per
token matches A, so A vs E isolates the effect of conditional computation."""
import torch.nn.functional as F

from phase1.models.decoder import DecoderAPI
from phase1.models.moe import MoEConfig, MoEDecoderLM
from phase1.models.optim import make_adamw, warmup_cosine

NAME = "E_moe"
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

LB_COEF, Z_COEF = 0.01, 1e-3


def build_model(vocab_size):
    return MoEDecoderLM(MoEConfig(vocab_size=vocab_size, d_model=768, n_layers=12, n_heads=12, n_kv_heads=4,
                                  n_experts=8, top_k=2, d_ff_expert=1024))


def loss_fn(model, batch, device):
    x, y = (t.to(device, non_blocking=True) for t in batch)
    logits = model(x)
    ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
    if not model.training:
        return ce
    return ce + LB_COEF * model.aux["lb"] + Z_COEF * model.aux["z"]


def make_optimizer(model):
    return make_adamw(model, lr=6e-4, weight_decay=0.1)


def make_lr_lambda(total_steps):
    return warmup_cosine(total_steps, warmup=min(300, max(1, total_steps // 20)), final_frac=0.1)


def decoder_api(model):
    return DecoderAPI(model)
