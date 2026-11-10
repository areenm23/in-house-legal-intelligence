"""Model G: bidirectional Transformer encoder trained with masked language modelling
(ModernBERT-style: pre-norm RMSNorm, RoPE, QK-norm, SwiGLU, no causal mask, tied output head).

Objective (mlm_batch): every window is masked one of two ways, chosen per row:
  * random masking: mask_prob of the tokens are selected; 80% -> <mask>, 10% -> random token,
    10% unchanged (BERT);
  * suffix masking: the last L tokens (L uniform in span) are all <mask>. This teaches the
    encoder to fill the end of a document from its beginning, which is how it writes Phase-1
    continuations; it is still a masked-LM objective, and the encoder stays bidirectional for
    Phases 2-7.
Loss is cross-entropy on the selected positions only (the head runs only there).

Generation (MLMFillAPI): the prompt is followed by n_masks <mask> tokens; masks are filled left
to right. One forward pass predicts the next `block` masks at once, so a 224-token
continuation takes 224/block passes (no KV cache is possible for a bidirectional model).
"""
import dataclasses

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..train.data import _windows
from .encdec import EncoderBlock
from .layers import RMSNorm, build_rope_cache, init_weights, padded_vocab

N_EXTRA_SPECIALS = 13
FIRST_REGULAR_ID = 4


def mask_id_for(vocab_size):
    return vocab_size - N_EXTRA_SPECIALS


@dataclasses.dataclass
class MLMConfig:
    vocab_size: int
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12
    d_ff: int = 2048
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    qk_norm: bool = True
    bias: bool = False


class EncoderMLM(nn.Module):
    def __init__(self, c: MLMConfig):
        super().__init__()
        self.cfg = c
        self.mask_id = mask_id_for(c.vocab_size)
        self.tok_emb = nn.Embedding(padded_vocab(c.vocab_size), c.d_model)
        self.blocks = nn.ModuleList([EncoderBlock(c) for _ in range(c.n_layers)])
        self.norm = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, padded_vocab(c.vocab_size), bias=False)
        self.lm_head.weight = self.tok_emb.weight
        cos, sin = build_rope_cache(c.max_seq_len, c.d_model // c.n_heads, c.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        init_weights(self, c.n_layers)

    def hidden(self, ids):
        """(B, T) -> contextual vectors (B, T, d); every position sees the whole sequence."""
        x = self.tok_emb(ids)
        cs = (self.rope_cos, self.rope_sin)
        for blk in self.blocks:
            x = blk(x, cs)
        return self.norm(x)

    def logits(self, h):
        return self.lm_head(h)[..., :self.cfg.vocab_size]

    def forward(self, ids):
        return self.logits(self.hidden(ids))


def mlm_batch(shards, batch_size, seq_len, seed, index, mix, mask_id, mask_prob=0.3, suffix_frac=0.25,
              span=(32, 256)):
    """Same token windows as causal_batch (seed, index); returns (input ids, labels with -100 = ignore)."""
    w = _windows(shards, batch_size, seq_len, seed, index, mix).numpy()
    rng = np.random.default_rng([seed, index, 7])
    B, T = w.shape
    sel = rng.random((B, T)) < mask_prob
    r = rng.random((B, T))
    x = w.copy()
    x[sel & (r < 0.8)] = mask_id
    rand = sel & (r >= 0.8) & (r < 0.9)
    x[rand] = rng.integers(FIRST_REGULAR_ID, mask_id, int(rand.sum()))
    for b in np.nonzero(rng.random(B) < suffix_frac)[0]:
        L = int(rng.integers(span[0], span[1] + 1))
        sel[b] = False
        sel[b, T - L:] = True
        x[b] = w[b]
        x[b, T - L:] = mask_id
    labels = np.where(sel, w, -100)
    return torch.from_numpy(x), torch.from_numpy(labels)


def mlm_loss(model, batch, device):
    x, labels = (t.to(device, non_blocking=True) for t in batch)
    h = model.hidden(x)
    sel = labels != -100
    logits = model.logits(h[sel])
    return F.cross_entropy(logits.float(), labels[sel])


class MLMFillAPI:
    """prefill/step for phase1.eval.generate: left-to-right mask filling, `block` masks per pass."""

    def __init__(self, model, n_masks=224, block=4):
        self.m, self.n, self.block = model, n_masks, block

    def _logits(self, st):
        p, x = st["pos"], st["x"]
        if p >= x.shape[1]:
            return torch.zeros(x.shape[0], self.m.cfg.vocab_size, device=x.device)
        if st["cache"] is None or p >= st["start"] + st["cache"].shape[1]:
            h = self.m.hidden(x)
            lg = self.m.logits(h[:, p:p + self.block]).float()
            lg[..., self.m.mask_id:] = float("-inf")
            st["cache"], st["start"] = lg, p
        return st["cache"][:, p - st["start"]]

    def prefill(self, ids):
        masks = torch.full((ids.shape[0], self.n), self.m.mask_id, dtype=ids.dtype, device=ids.device)
        st = {"x": torch.cat([ids, masks], 1), "pos": ids.shape[1], "cache": None, "start": 0}
        return self._logits(st), st

    def step(self, st, nxt):
        if st["pos"] < st["x"].shape[1]:
            st["x"][:, st["pos"]] = nxt
        st["pos"] += 1
        return self._logits(st), st
