"""End-to-end tests of the Phase-1 tooling on synthetic data laid out like the HF dataset.

  python -m pytest phase1/tests -q
"""
import json
import os
import random
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from tokenizers import Tokenizer

from phase1.checks.model_checks import (check_causal, check_encdec_decoder_causal, check_equivalent,
                                        count_params)
from phase1.data.common import SOURCES, is_val_doc, iter_json_records, source_file
from phase1.eval.submission import load_prediction, to_prediction_text, validate, write_submission

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORDS = ("the court held that section 302 IPC appellant respondent petition dismissed "
         "Hon'ble A.I.R. 1973 SC 1461 Crl.A. No. 12/2019 \"quoted\" backslash\\ न्यायालय ₹500").split()


def make_doc(rng, n_words):
    out = []
    for i in range(n_words):
        out.append(rng.choice(WORDS))
        out.append("\n" if i % 37 == 36 else ("  " if i % 11 == 0 else " "))
    return "".join(out).strip()


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    base = tmp_path_factory.mktemp("raw")
    rng = random.Random(1)
    docs = {}
    for src in SOURCES:
        recs = [{"id": f"{src}{i}", "text": make_doc(rng, rng.randint(750, 1500))} for i in range(60)]
        recs.insert(3, {"id": "empty", "text": "   "})
        docs[src] = recs
        path = source_file(str(base), src)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
    test_dir = base / "Phase_1_Pretraining" / "Test"
    test_dir.mkdir(parents=True)
    test = {str(i): {"text": make_doc(rng, 512)} for i in range(1, 11)}
    (test_dir / "test.json").write_text(json.dumps(test, ensure_ascii=False), encoding="utf-8")
    return base, docs, test


def run(*args):
    r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.mark.parametrize("chunk", [1, 7, 64, 1 << 22])
def test_stream_matches_json_load(raw, chunk):
    base, docs, _ = raw
    path = source_file(str(base), "acts")
    with open(path, encoding="utf-8") as f:
        expected = json.load(f)
    assert list(iter_json_records(path, chunk_chars=chunk)) == expected


def test_stream_jsonl_and_empty(tmp_path):
    p = tmp_path / "x.jsonl"
    recs = [{"text": "a\nb"}, {"text": "ü \"q\""}]
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    assert list(iter_json_records(str(p), chunk_chars=3)) == recs
    e = tmp_path / "e.json"
    e.write_text(" [ ] ", encoding="utf-8")
    assert list(iter_json_records(str(e))) == []


def test_stream_truncated_array_raises(tmp_path):
    p = tmp_path / "t.json"
    p.write_text('[{"text": "a"}, {"text": "b', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        list(iter_json_records(str(p), chunk_chars=4))


@pytest.fixture(scope="module")
def built(raw, tmp_path_factory):
    base, docs, test = raw
    work = tmp_path_factory.mktemp("work")
    tok_path = str(work / "tok" / "bpe.json")
    run("phase1.data.train_tokenizer", "--impl", "hf", "--raw-dir", str(base), "--vocab-size", "400",
        "--mb-per-source", "1", "--val-frac", "0.2", "--eval-mb", "1",
        "--test-file", str(base / "Phase_1_Pretraining" / "Test" / "test.json"), "--out", tok_path)
    shards = str(work / "shards")
    run("phase1.data.make_shards", "--raw-dir", str(base), "--tokenizer", tok_path,
        "--budget", *[f"{s}=1e9" for s in SOURCES], "--keep-frac", *[f"{s}=1" for s in SOURCES],
        "--val-tokens-per-source", "1000000000", "--val-frac", "0.2", "--dev-docs-per-source", "5",
        "--batch-docs", "7", "--out", shards)
    return work, tok_path, shards


def test_tokenizer_roundtrip_and_specials(built):
    _, tok_path, _ = built
    tok = Tokenizer.from_file(tok_path)
    assert tok.id_to_token(0) == "<|endoftext|>"
    text = "Section 302 IPC\n  न्यायालय ₹500 \"x\"\\"
    assert tok.decode(tok.encode(text).ids) == text
    rep = json.load(open(tok_path.replace(".json", ".report.json"), encoding="utf-8"))
    assert all(v["roundtrip_failures"] == 0 for v in rep["sources"].values())
    assert "test_prompt_tokens" in rep


def test_offsets_cover_words(built):
    _, tok_path, _ = built
    tok = Tokenizer.from_file(tok_path)
    text = "The appellant  filed Crl.A. No. 12/2019."
    enc = tok.encode(text)
    for (a, b), t in zip(enc.offsets, enc.tokens):
        assert text[a:b].strip() == text[a:b], f"offset of {t!r} includes whitespace: {text[a:b]!r}"


def test_shards_decode_to_the_right_documents(raw, built):
    base, docs, _ = raw
    _, tok_path, shards = built
    tok = Tokenizer.from_file(tok_path)
    meta = json.load(open(os.path.join(shards, "meta.json"), encoding="utf-8"))
    for src in SOURCES:
        expected = {"train": [], "val": []}
        for i, rec in enumerate(docs[src]):
            if rec["text"].strip():
                expected["val" if is_val_doc(src, i, 0.2) else "train"].append(rec["text"])
        for split in ("train", "val"):
            ids = np.fromfile(os.path.join(shards, split, f"{src}.bin"), dtype=np.uint16)
            starts = np.load(os.path.join(shards, split, f"{src}.idx.npy"))
            assert len(starts) == len(expected[split]) == meta["sources"][src][split]["docs"]
            assert len(ids) == meta["sources"][src][split]["tokens"]
            ends = list(starts[1:]) + [len(ids)]
            for s, e, text in zip(starts, ends, expected[split]):
                assert ids[e - 1] == 0, "each document must end with <|endoftext|>"
                assert tok.decode(ids[s:e - 1].tolist()) == text


def test_budget_caps_train_tokens(raw, built, tmp_path):
    base, _, _ = raw
    _, tok_path, _ = built
    out = str(tmp_path / "small")
    run("phase1.data.make_shards", "--raw-dir", str(base), "--tokenizer", tok_path,
        "--budget", "acts=3000", "--keep-frac", "acts=1", "--val-tokens-per-source", "0",
        "--val-frac", "0.2", "--dev-docs-per-source", "0", "--batch-docs", "1", "--out", out)
    meta = json.load(open(os.path.join(out, "meta.json"), encoding="utf-8"))
    n = meta["sources"]["acts"]["train"]["tokens"]
    starts = np.load(os.path.join(out, "train", "acts.idx.npy"))
    assert starts[-1] < 3000 <= n, (starts[-1], n)


def test_dev_set_windows(built):
    work, _, shards = built
    out = str(work / "dev")
    run("phase1.eval.make_dev_set", "--dev-docs", os.path.join(shards, "dev_docs"),
        "--per-source", "3", "--out", out)
    prompts = json.load(open(os.path.join(out, "prompts.json"), encoding="utf-8"))
    truth = json.load(open(os.path.join(out, "truth.json"), encoding="utf-8"))
    assert set(prompts) == set(truth) and len(prompts) == 3 * len(SOURCES)
    docs = {}
    for src in SOURCES:
        for line in open(os.path.join(shards, "dev_docs", f"{src}.jsonl"), encoding="utf-8"):
            d = json.loads(line)
            docs[f"{d['source']}-{d['index']}"] = d["text"]
    for k in prompts:
        p, r = prompts[k]["text"], truth[k]["text"]
        assert len(p.split()) == 512 and len(r.split()) == 128
        doc = docs[k]
        i = doc.index(p)
        rest = doc[i + len(p):]
        assert rest.lstrip().startswith(r), "reference must directly follow the prompt"


def test_submission_roundtrip(raw, tmp_path):
    _, _, test = raw
    pred = {k: {"text": to_prediction_text("word " * 300)} for k in test}
    assert all(len(v["text"].split()) == 128 for v in pred.values())
    assert validate(pred, test) == []
    _, zip_path = write_submission(pred, str(tmp_path))
    assert zipfile.ZipFile(zip_path).namelist() == ["prediction.json"]
    assert load_prediction(zip_path) == pred


@pytest.mark.parametrize("breakage", ["missing", "extra", "empty", "long"])
def test_submission_rejects(raw, breakage):
    _, _, test = raw
    pred = {k: {"text": "a b c"} for k in test}
    if breakage == "missing":
        pred.pop("1")
    elif breakage == "extra":
        pred["999"] = {"text": "x"}
    elif breakage == "empty":
        pred["1"] = {"text": "  "}
    else:
        pred["1"] = {"text": "w " * 129}
    with pytest.raises(ValueError):
        validate(pred, test)


def test_submission_warns_on_prompt_echo(raw):
    _, _, test = raw
    pred = {k: {"text": to_prediction_text(test[k]["text"])} for k in test}
    assert any("starts with the prompt" in w for w in validate(pred, test))


def test_starter_sample_submission_is_valid_shape():
    path = os.path.join(ROOT, "course_files", "starter", "Starter Code", "submission.zip")
    if not os.path.exists(path):
        pytest.skip("starter zip not unpacked")
    pred = load_prediction(path)
    ids = {str(i) for i in range(1, 1001)}
    assert set(pred) == ids
    validate(pred, {k: {"text": "unrelated prompt"} for k in ids})


class ToyPrefixMean(nn.Module):
    """Causal toy: position t sees the running mean of embeddings 0..t."""
    def __init__(self, v, d=16, leak=False):
        super().__init__()
        self.emb, self.out, self.leak = nn.Embedding(v, d), nn.Linear(d, v), leak

    def forward(self, x):
        h = self.emb(x)
        if self.leak:
            mix = h.mean(1, keepdim=True).expand_as(h)
        else:
            mix = h.cumsum(1) / torch.arange(1, h.shape[1] + 1).view(1, -1, 1)
        return self.out(h + mix)

    def forward_recurrent(self, x):
        h, state, outs = self.emb(x), 0, []
        for t in range(h.shape[1]):
            state = state + h[:, t]
            outs.append(self.out(h[:, t] + state / (t + 1)))
        return torch.stack(outs, 1)


def test_causal_check_passes_and_catches_leak():
    v = 50
    check_causal(ToyPrefixMean(v).eval(), v)
    with pytest.raises(AssertionError, match="causality leak"):
        check_causal(ToyPrefixMean(v, leak=True).eval(), v)


def test_equivalence_check():
    v, m = 50, ToyPrefixMean(50).eval()
    check_equivalent(m, m.forward_recurrent, v)
    with pytest.raises(AssertionError, match="disagree"):
        check_equivalent(m, lambda x: m.forward_recurrent(x) + 0.01, v)


def test_encdec_check():
    v, dec = 50, ToyPrefixMean(50).eval()
    src_emb = nn.Embedding(v, 16)
    good = lambda s, t: dec(t) + dec.out(src_emb(s).mean(1, keepdim=True))
    check_encdec_decoder_causal(good, v)
    with pytest.raises(AssertionError, match="encoder input"):
        check_encdec_decoder_causal(lambda s, t: dec(t), v)


def test_count_params_tied_and_untied():
    v, d = 100, 8
    m = ToyPrefixMean(v, d)
    c = count_params(m, vocab_size=v)
    assert c["embedding"] == v * d and c["untied_output_head"] == d * v + v
    assert c["total"] == c["embedding"] + c["untied_output_head"] + c["non_embedding"]
    m.out.weight = m.emb.weight
    c = count_params(m, vocab_size=v)
    assert c["untied_output_head"] == 0 and c["total"] == v * d + v
