"""Tests for the batching, resumable runner and generation driver (toy models only).

  python -m pytest phase1/tests -q
"""
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from phase1.eval.generate import CausalNoCache, DecodeConfig, EncDecNoCache, generate, sample_next
from phase1.train.data import BOS_ID, TokenShards, causal_batch, continuation_batch, fixed_eval_batches
from phase1.train.runner import RunConfig, causal_lm_loss, train

V = 64


@pytest.fixture()
def shards(tmp_path):
    rng = np.random.default_rng(0)
    meta = {"sources": {}}
    for split in ("train", "val"):
        os.makedirs(tmp_path / split)
    for src, n in [("acts", 5000), ("tribunals", 3000)]:
        for split in ("train", "val"):
            rng.integers(1, V, n).astype(np.uint16).tofile(tmp_path / split / f"{src}.bin")
        meta["sources"][src] = {"train": {"tokens": n}, "val": {"tokens": n}}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return str(tmp_path)


class Toy(nn.Module):
    """Causal toy LM with dropout so RNG restoration matters on resume."""
    def __init__(self):
        super().__init__()
        self.emb, self.drop, self.out = nn.Embedding(V, 16), nn.Dropout(0.1), nn.Linear(16, V)

    def forward(self, x):
        h = self.emb(x)
        h = h.cumsum(1) / torch.arange(1, h.shape[1] + 1).view(1, -1, 1)
        return self.out(self.drop(h))


def test_batches_are_deterministic_and_shifted(shards):
    sh = TokenShards(shards, "train")
    x1, y1 = causal_batch(sh, 4, 32, seed=0, index=7)
    x2, y2 = causal_batch(sh, 4, 32, seed=0, index=7)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert torch.equal(x1[:, 1:], y1[:, :-1])
    assert not torch.equal(x1, causal_batch(sh, 4, 32, seed=0, index=8)[0])


def test_continuation_uses_same_windows(shards):
    sh = TokenShards(shards, "train")
    x, y = causal_batch(sh, 3, 47, seed=1, index=2)
    window = torch.cat([x, y[:, -1:]], dim=1)
    src, tgt_in, tgt_out = continuation_batch(sh, 3, 30, 18, seed=1, index=2)
    assert torch.equal(torch.cat([src, tgt_out], 1), window)
    assert (tgt_in[:, 0] == BOS_ID).all() and torch.equal(tgt_in[:, 1:], tgt_out[:, :-1])


def test_mix_weights_select_sources(shards):
    sh = TokenShards(shards, "val")
    ev = fixed_eval_batches(sh, lambda s, seed, i, mix: causal_batch(s, 2, 16, seed, i, mix), 2)
    assert set(ev) == {"acts", "tribunals"} and len(ev["acts"]) == 2
    assert np.allclose(sh.weights({"acts": 3, "tribunals": 1}), [0.75, 0.25])


def _run(shards, out_dir, total_steps, max_hours=11.0):
    torch.manual_seed(0)
    model = Toy()
    sh, val = TokenShards(shards, "train"), TokenShards(shards, "val")
    cfg = RunConfig(out_dir=out_dir, total_steps=total_steps, grad_accum=2, grad_clip=1.0, eval_every=3,
                    log_every=1, max_hours=max_hours, token_milestones=[200], seed=0)
    res = train(model, lambda m: torch.optim.AdamW(m.parameters(), lr=1e-2),
                lambda s: min(1.0, (s + 1) / 3), causal_lm_loss,
                lambda i: causal_batch(sh, 2, 16, seed=0, index=i),
                fixed_eval_batches(val, lambda s, seed, i, mix: causal_batch(s, 2, 16, seed, i, mix), 1),
                tokens_per_batch=2 * 16, cfg=cfg)
    return model, res


def test_resume_matches_uninterrupted(shards, tmp_path):
    straight, _ = _run(shards, str(tmp_path / "a"), 8)
    _run(shards, str(tmp_path / "b"), 4)
    resumed, res = _run(shards, str(tmp_path / "b"), 8)
    assert res["step"] == 8
    for (n, p), q in zip(straight.state_dict().items(), resumed.state_dict().values()):
        assert torch.equal(p, q), f"{n} differs after resume"
    log = [json.loads(l) for l in open(tmp_path / "b" / "log.jsonl")]
    assert any("eval" in r for r in log) and log[-1]["stopped"] == "total_steps"
    assert os.path.exists(tmp_path / "b" / "ckpt" / "milestone_200.pt")
    assert len([f for f in os.listdir(tmp_path / "b" / "ckpt") if f.startswith("step_")]) <= 2


def test_max_hours_stops_and_checkpoints(shards, tmp_path):
    _, res = _run(shards, str(tmp_path / "c"), 100, max_hours=0.0)
    assert res["stopped"] == "max_hours" and res["step"] == 1
    assert os.path.exists(tmp_path / "c" / "ckpt" / "step_1.pt")


def word_tokenizer():
    """Tiny whitespace-aware tokenizer whose ids are < V, for driving the toy model."""
    vocab = {"<|endoftext|>": 0, "<pad>": 1, "<s>": 2}
    for i in range(3, V):
        vocab[f"Ġw{i}"] = i
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<pad>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def test_sample_next_filters():
    g = torch.Generator().manual_seed(0)
    logits = torch.tensor([[0.0, 5.0, 4.0, -1.0]])
    assert sample_next(logits, [[]], DecodeConfig(temperature=0), g).item() == 1
    for _ in range(20):
        assert sample_next(logits, [[]], DecodeConfig(top_k=2), g).item() in (1, 2)
        assert sample_next(logits, [[]], DecodeConfig(top_p=0.5), g).item() == 1
    assert sample_next(logits, [[1, 2, 1]], DecodeConfig(temperature=0, no_repeat_ngram=2), g).item() != 2
    assert sample_next(logits, [[1]], DecodeConfig(temperature=0, repetition_penalty=100.0), g).item() == 2


def test_generate_produces_128_words():
    torch.manual_seed(0)
    model = Toy().eval()
    with torch.no_grad():
        model.out.bias[0] = -1e4
    tok = word_tokenizer()
    prompts = {str(i): {"text": " ".join(f"w{3 + (i * j) % (V - 3)}" for j in range(40 + i))} for i in range(5)}
    preds, stats = generate(prompts, tok, CausalNoCache(model), DecodeConfig(max_new_tokens=200, seed=1),
                            max_prompt_tokens=64, batch_size=2, device="cpu")
    assert set(preds) == set(prompts)
    assert all(len(p["text"].split()) == 128 for p in preds.values())
    assert stats["short_outputs"] == 0 and stats["prompt_tokens_dropped"] > 0


def test_generate_encdec_adapter():
    torch.manual_seed(0)
    dec = Toy().eval()
    emb = nn.Embedding(V, 16)
    fwd = lambda src, tgt: dec(tgt) + dec.out(emb(src).mean(1, keepdim=True))
    tok = word_tokenizer()
    prompts = {"a": {"text": "w5 w6 w7 w8"}}
    preds, _ = generate(prompts, tok, EncDecNoCache(fwd), DecodeConfig(temperature=0, max_new_tokens=20),
                        max_prompt_tokens=64, device="cpu")
    assert len(preds["a"]["text"].split()) <= 128
