"""Tests for the A-1 BPE adapter and the pipeline running on it (EOS = </s> = 3).

  python -m pytest phase1/tests -q
"""
import json
import os
import random

import numpy as np
import pytest
import torch

from phase1.data.a1_bpe import A1BPETokenizer, A1Tokenizer, ByteBPE, _fast_fit, load_tokenizer, special_ids
from phase1.data.common import SOURCES, is_val_doc
from phase1.eval.generate import CausalNoCache, DecodeConfig, generate
from phase1.tests.test_pipeline import make_doc, raw, run

TEXTS = ["the court held that section 302 IPC applies", "the appellant filed Crl.A. No. 12/2019 in the court",
         "न्यायालय ने कहा ₹500 \"quoted\" back\\slash", "the the the court court held held"] * 3


def test_fast_fit_matches_a1_fit():
    slow = ByteBPE(vocab_size=120).fit(TEXTS)
    fast = _fast_fit(ByteBPE(vocab_size=120), TEXTS)
    assert fast.merges == slow.merges and fast.itos == slow.itos


def test_seeded_vocab_keeps_a1_merges_and_has_every_byte():
    plain = _fast_fit(ByteBPE(vocab_size=400), TEXTS)
    tok = A1Tokenizer.train(TEXTS, vocab_size=400)
    n = len(tok.bpe.merges)
    assert n > 0 and tok.bpe.merges == plain.merges[:n]
    assert all(f"B{b:02X}" in tok.stoi for b in range(256))
    ids = tok.encode("😀 ¿?").ids
    assert tok.unk_id not in ids and tok.decode(ids) == "😀 ¿?"


def test_ids_match_a1_encoder_and_specials():
    tok = A1Tokenizer.train(TEXTS, vocab_size=300)
    ref = A1BPETokenizer(tok.bpe)
    for t in TEXTS + ["unseen wordsss here"]:
        assert tok.encode(t).ids == ref.encode(t)
    assert (tok.pad_id, tok.unk_id, tok.bos_id, tok.eos_id) == (0, 1, 2, 3)
    assert tok.id_to_token(tok.mask_id) == "<mask>" and tok.get_vocab_size() == len(tok.bpe.itos) + 13
    assert special_ids(tok) == {"eos": 3, "bos": 2, "pad": 0}
    text = "Section 302  IPC\n\nthe   court"
    assert tok.decode(tok.encode(text).ids) == " ".join(text.split())
    assert tok.decode(tok.encode("the court").ids + [tok.mask_id]) == "the court"


def test_save_load_and_detect(tmp_path):
    tok = A1Tokenizer.train(TEXTS, vocab_size=300)
    p = str(tmp_path / "a1.json")
    tok.save(p)
    back = load_tokenizer(p)
    assert isinstance(back, A1Tokenizer) and back.itos == tok.itos
    assert all(back.encode(t).ids == tok.encode(t).ids for t in TEXTS)


def test_parallel_encode_matches_serial():
    rng = random.Random(0)
    docs = [make_doc(rng, rng.randint(20, 200)) for _ in range(300)]
    tok = A1Tokenizer.train(docs[:50], vocab_size=500)
    try:
        par = tok.encode_batch(docs, workers=2, chunk=16)
    finally:
        tok.close()
    assert [e.ids for e in par] == [tok.encode(d).ids for d in docs]


@pytest.fixture(scope="module")
def built_a1(raw, tmp_path_factory):
    base, docs, test = raw
    work = tmp_path_factory.mktemp("work_a1")
    tok_path = str(work / "tok" / "a1.json")
    run("phase1.data.train_tokenizer", "--impl", "a1", "--raw-dir", str(base), "--vocab-size", "400",
        "--mb-per-source", "1", "--val-frac", "0.2", "--eval-mb", "1",
        "--test-file", str(base / "Phase_1_Pretraining" / "Test" / "test.json"), "--out", tok_path)
    shards = str(work / "shards")
    run("phase1.data.make_shards", "--raw-dir", str(base), "--tokenizer", tok_path,
        "--budget", *[f"{s}=1e9" for s in SOURCES], "--keep-frac", *[f"{s}=1" for s in SOURCES],
        "--val-tokens-per-source", "1000000000", "--val-frac", "0.2", "--dev-docs-per-source", "5",
        "--batch-docs", "40", "--workers", "2", "--out", shards)
    return tok_path, shards


def test_a1_report(built_a1):
    tok_path, _ = built_a1
    rep = json.load(open(tok_path.replace(".json", ".report.json"), encoding="utf-8"))
    assert rep["special_ids"]["eos"] == 3 and rep["trained_on"]["impl"] == "a1"
    assert all(v["roundtrip_failures"] == 0 and v["unk_tokens"] == 0 for v in rep["sources"].values())
    assert rep["test_prompt_tokens"]["median"] >= 512


def test_a1_shards_decode_to_normalised_documents(raw, built_a1):
    _, docs, _ = raw
    tok_path, shards = built_a1
    tok = load_tokenizer(tok_path)
    meta = json.load(open(os.path.join(shards, "meta.json"), encoding="utf-8"))
    assert (meta["eos_id"], meta["bos_id"], meta["tokenizer_impl"]) == (3, 2, "a1")
    for src in SOURCES:
        expected = {"train": [], "val": []}
        for i, rec in enumerate(docs[src]):
            if rec["text"].strip():
                expected["val" if is_val_doc(src, i, 0.2) else "train"].append(rec["text"])
        for split in ("train", "val"):
            ids = np.fromfile(os.path.join(shards, split, f"{src}.bin"), dtype=np.uint16)
            starts = np.load(os.path.join(shards, split, f"{src}.idx.npy"))
            assert len(starts) == len(expected[split])
            ends = list(starts[1:]) + [len(ids)]
            for s, e, text in zip(starts, ends, expected[split]):
                assert ids[e - 1] == 3 and 3 not in ids[s:e - 1]
                assert tok.decode(ids[s:e - 1].tolist()) == " ".join(text.split())


def test_generate_stops_on_a1_eos():
    tok = A1Tokenizer.train(TEXTS, vocab_size=300)
    n = tok.get_vocab_size()
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Embedding(n, 8), torch.nn.Linear(8, n)).eval()
    with torch.no_grad():
        model[1].bias[3] = 1e4
    prompts = {"a": {"text": "the court held"}, "b": {"text": "the appellant filed"}}
    preds, stats = generate(prompts, tok, CausalNoCache(model), DecodeConfig(max_new_tokens=50),
                            max_prompt_tokens=64, device="cpu")
    assert all(p["text"] == "" for p in preds.values()) and stats["short_outputs"] == 2


def test_prep_driver_end_to_end(raw, tmp_path):
    base, _, test = raw
    cfg = {"t": {"tokenizer": {"vocab_size": 400, "mb_per_source": 1}, "val_frac": 0.2,
                 "budgets": {"_help": "x", "acts": "all", "tribunals": 5000, "high_courts": 0},
                 "val_tokens_per_source": 1e9, "dev_examples_per_source": 2},
           "blank": {"tokenizer": {"vocab_size": None, "mb_per_source": 1}, "val_frac": 0.2,
                     "budgets": {"acts": 1}, "val_tokens_per_source": 1, "dev_examples_per_source": 1}}
    cfg_path = tmp_path / "data.json"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "p1data"
    run("phase1.data.prep", "--config", str(cfg_path), "--section", "t", "--raw-dir", str(base),
        "--out", str(out), "--skip-download", "--workers", "2", "--eval-mb", "1")
    meta = json.load(open(out / "shards" / "meta.json", encoding="utf-8"))
    assert set(meta["sources"]) == {"acts", "tribunals"} and meta["eos_id"] == 3
    assert meta["sources"]["tribunals"]["train"]["tokens"] < meta["sources"]["acts"]["train"]["tokens"]
    assert len(json.load(open(out / "dev" / "prompts.json", encoding="utf-8"))) == 4
    assert json.load(open(out / "test.json", encoding="utf-8")) == test
    man = json.load(open(out / "prep_manifest.json", encoding="utf-8"))
    assert set(man["timings_s"]) == {"profile", "tokenizer", "shards", "dev_set"}
    with pytest.raises(AssertionError, match="blank values: tokenizer.vocab_size"):
        run("phase1.data.prep", "--config", str(cfg_path), "--section", "blank", "--raw-dir", str(base),
            "--out", str(tmp_path / "x"), "--skip-download")


def test_suppress_eos_reaches_target_words():
    tok = A1Tokenizer.train(TEXTS, vocab_size=300)
    n = tok.get_vocab_size()
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Embedding(n, 8), torch.nn.Linear(8, n)).eval()
    with torch.no_grad():
        model[1].bias[3] = 1e4
        model[1].bias[tok.encode("the").ids[-1]] = 1e3
    preds, stats = generate({"a": {"text": "the court held"}}, tok, CausalNoCache(model),
                            DecodeConfig(temperature=0, max_new_tokens=600, suppress_eos=True),
                            max_prompt_tokens=64, device="cpu")
    assert len(preds["a"]["text"].split()) == 128 and stats["short_outputs"] == 0
