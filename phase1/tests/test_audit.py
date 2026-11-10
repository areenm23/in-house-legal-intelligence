"""phase1.data.audit on a synthetic corpus with planted duplicates and dev/test leaks.

The shards and dev set come from the real make_shards / make_dev_set, so the audit's rebuilt
training subset is checked against what make_shards actually wrote.
"""
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest

from phase1.data.a1_bpe import A1Tokenizer
from phase1.data.audit import minhash, ngram_hashes
from phase1.data.common import SOURCES, is_val_doc, source_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VAL_FRAC = 0.15


def _vocab(rng, n=3000):
    letters = "abcdefghijklmnopqrstuvwxyz"
    return list({"".join(rng.choice(letters) for _ in range(rng.randint(2, 9))) for _ in range(n)})


def _doc(rng, vocab, n):
    words = [vocab[min(int(rng.paretovariate(1.1)) - 1, len(vocab) - 1)] if rng.random() < 0.5 else rng.choice(vocab)
             for _ in range(n)]
    return "\n".join(" ".join(words[i:i + 12]) for i in range(0, n, 12))


def _run(*args):
    r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


@pytest.fixture(scope="module")
def audited(tmp_path_factory):
    base = tmp_path_factory.mktemp("audit")
    raw, ref, out = base / "raw", base / "ref", base / "out"
    rng = random.Random(7)
    vocab = _vocab(rng)
    docs = {s: [_doc(rng, vocab, rng.randint(720, 1400)) for _ in range(80)] for s in SOURCES}
    hc = docs["high_courts"]
    hc[50] = hc[10]
    w = hc[20].split()
    w[100:103] = ["zzqx", "qqzx", "xxqz"]
    hc[60] = " ".join(w)
    hc[5] = "   "
    hc[6] = "a short one"
    acts = docs["acts"]
    v = next(i for i in range(len(acts)) if is_val_doc("acts", i, VAL_FRAC))
    j = next(i for i in range(len(acts), 200) if not is_val_doc("acts", i, VAL_FRAC))
    while len(acts) < j:
        acts.append(_doc(rng, vocab, 800))
    acts.append(acts[v])
    t = next(i for i in range(len(acts)) if not is_val_doc("acts", i, VAL_FRAC) and i != j)
    for s, d in docs.items():
        path = source_file(str(raw), s)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{"text": x} for x in d], f)
    test_dir = raw / "Phase_1_Pretraining" / "Test"
    test_dir.mkdir(parents=True)
    leak = " ".join(acts[t].split()[100:612])
    novel = " ".join(_doc(random.Random(99), [f"nov{k}" for k in range(500)], 512).split())
    (test_dir / "test.json").write_text(json.dumps({"1": {"text": leak}, "2": {"text": novel}}), encoding="utf-8")

    tok = A1Tokenizer.train([x for d in docs.values() for x in d[:20]], 800)
    (ref / "tokenizer").mkdir(parents=True)
    tok_path = ref / "tokenizer" / "a1bpe_800.json"
    tok.save(str(tok_path))
    shards = base / "shards"
    _run("phase1.data.make_shards", "--raw-dir", str(raw), "--tokenizer", str(tok_path),
         "--budget", "acts=1e12", "high_courts=40000", "supreme_court=1e12", "tribunals=1e12",
         "--keep-frac", "acts=1", "high_courts=0.9", "supreme_court=1", "tribunals=0.5",
         "--val-tokens-per-source", "100000000", "--val-frac", str(VAL_FRAC), "--dev-docs-per-source", "1000",
         "--dev-min-words", "700", "--batch-docs", "8", "--workers", "1", "--out", str(shards))
    _run("phase1.eval.make_dev_set", "--dev-docs", str(shards / "dev_docs"), "--per-source", "100",
         "--out", str(ref / "dev"))
    (ref / "shards").mkdir()
    (ref / "shards" / "meta.json").write_text((shards / "meta.json").read_text(encoding="utf-8"), encoding="utf-8")
    _run("phase1.data.audit", "--raw-dir", str(raw), "--ref", str(ref), "--out", str(out), "--workers", "2",
         "--batch-docs", "8")
    with open(out / "audit.json", encoding="utf-8") as f:
        res = json.load(f)
    with open(out / "per_example_overlap.json", encoding="utf-8") as f:
        per = json.load(f)
    return res, per, out, {"dev_copy": f"acts-{v}", "novel_types": len(set(novel.split()))}


def test_training_subset_rebuilt_exactly(audited):
    res = audited[0]
    hc = json.loads(json.dumps(res["membership_check"]["high_courts"]))
    assert res["size"]["high_courts"]["by_membership"]["cut_by_budget"]["docs"] > 0, "budget cut-off not exercised"
    for s in SOURCES:
        assert res["membership_check"][s]["exact"], (s, res["membership_check"][s])
    assert hc["train_docs"] < 80


def test_duplicates_found(audited):
    res = audited[0]
    hc = res["duplicates"]["per_source"]["high_courts"]
    assert hc["exact_dup_docs"] == 1
    assert hc["near_dup_docs"] == 2
    assert res["duplicates"]["per_source"]["supreme_court"]["near_dup_docs"] == 0
    assert res["duplicates"]["dev_docs_with_exact_copy_in_train"] >= 1
    assert res["duplicates"]["dev_docs_with_near_dup_in_train"] >= 1
    assert res["dev_dup_per_doc"][audited[3]["dev_copy"]]["exact_copy_in_train"]


def test_overlap_found(audited):
    res, per = audited[0], audited[1]
    assert per["test_prompt:1"]["best_train_doc"][0] == pytest.approx(1.0)
    assert per["test_prompt:2"]["best_train_doc"][0] == 0
    assert per[f"dev_truth:{audited[3]['dev_copy']}"]["best_train_doc"][0] == pytest.approx(1.0)
    assert res["overlap"]["test_prompt"]["best_train_doc_ge_0.95"] == 1
    oov = res["vocab"]["oov"]["test_prompts_vs_train"]
    assert oov["oov_types"] == audited[3]["novel_types"]


def test_flags_tokenizer_and_outputs(audited):
    res, _, out, _ = audited
    assert res["quality"]["high_courts"]["short"]["docs"] == 1
    assert res["size"]["high_courts"]["empty_docs"] == 1
    tz = res["tokenizer"]
    for s in SOURCES:
        assert 1.0 <= tz["per_source"][s]["fertility_sample"] < 10
        assert tz["per_source"][s]["fertility_full_pass"] == pytest.approx(
            res["size"][s]["tokens"] / res["size"][s]["words"])
    assert res["vocab"]["all"]["types"] > 1000
    assert os.path.exists(out / "audit.md") and len(os.listdir(out / "figures")) == 8


def test_minhash_estimates_jaccard():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2 ** 63, 2000, dtype=np.uint64)
    b = a.copy()
    b[:400] = rng.integers(0, 2 ** 63, 400, dtype=np.uint64)
    est = (minhash(a) == minhash(b)).mean()
    assert abs(est - 1600 / 2400) < 0.15
    assert (minhash(a) == minhash(a[::-1])).all()
    wh = np.arange(1, 20, dtype=np.uint64)
    assert len(ngram_hashes(wh, 13)) == 7 and len(ngram_hashes(wh[:5], 13)) == 0
