"""Phase-7 tests: loading formats, F1@7 parity with the official evaluator, ranking (self exclusion, exactly 7),
BM25 against a brute-force formula, hard negatives, validation, every experiment on tiny backbones, and an end-to-end
run (train -> KD from saved embeddings -> ensemble) with valid zips.

    python -m pytest phase7/tests -q
"""
import json
import os
import sys
import zipfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from phase7 import data as D
from phase7 import synth
from phase7.experiments import get_experiment, names
from phase7.score import BM25, at_k, hard_negatives, top_k, validate

TOK = os.path.join(ROOT, "results", "results", "p1data_small", "tokenizer", "a1bpe_32755.json")
EVAL = os.path.join(ROOT, "course_files", "eval", "Evaluation Script", "evaluate_phase7.py")
need_tok = pytest.mark.skipif(not os.path.exists(TOK), reason="Phase-1 tokenizer not present")


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory):
    return synth.write(str(tmp_path_factory.mktemp("p7")))


@pytest.fixture(scope="module")
def data(synth_root, tmp_path_factory):
    return D.load_data(synth_root, TOK, str(tmp_path_factory.mktemp("c")), workers=1)


@pytest.mark.parametrize("raw", [
    {"q1": {"text": ["a b.", "c."], "relevant_candidates": ["x", "", "y"]}},
    [{"query_id": "q1", "sentences": ["a b.", "c."], "relevant": ["x", "y"]}],
    {"queries": [{"id": "q1", "text": "a b. c.", "relevant_candidates": ["x", "y"]}]},
])
def test_normalize(raw):
    assert D.normalize(raw) == [("q1", "a b. c.", ["x", "y"])]


@need_tok
def test_load(data):
    tr, dv, te = data["train"], data["dev"], data["test"]
    assert len(tr.queries) == 12 and len(tr.pool) == 61 and te.rel is None and dv.labelled
    assert all("" not in r for r in tr.rel) and all(tr.rel_rows(q) for q in range(len(tr.queries)))


def test_at_k_matches_official(tmp_path):
    if not os.path.exists(EVAL):
        pytest.skip("official evaluator not unpacked")
    import subprocess
    rng = np.random.default_rng(0)
    pool = [f"c{i}" for i in range(40)]
    gold = {f"q{i}": list(rng.choice(pool, size=int(rng.integers(1, 12)), replace=False)) for i in range(15)}
    pred = {q: list(rng.choice(pool, size=7, replace=False)) for q in gold}
    json.dump({q: {"text": ["x"], "relevant_candidates": g} for q, g in gold.items()}, open(tmp_path / "t.json", "w"))
    json.dump(pred, open(tmp_path / "p.json", "w"))
    json.dump(pool, open(tmp_path / "c.json", "w"))
    r = subprocess.run([sys.executable, EVAL, "--truth", str(tmp_path / "t.json"), "--prediction", str(tmp_path / "p.json"),
                        "--candidate-ids", str(tmp_path / "c.json")], capture_output=True, text=True)
    off = float([ln for ln in r.stdout.splitlines() if ln.startswith("Micro-F1@7")][0].split(":")[1])
    assert abs(off - at_k(pred, gold)["f1"]) < 1e-9


def test_top_k_and_validate():
    s = np.array([[5.0, 4, 3, 2, 1, 0, -1, -2, -3], [0, 1, 2, 3, 4, 5, 6, 7, 8]])
    q_ids, p_ids = ["a", "b"], ["a", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "b"]
    out = top_k(s, q_ids, p_ids)
    assert out["a"] == ["x1", "x2", "x3", "x4", "x5", "x6", "x7"] and "b" not in out["b"] and len(out["b"]) == 7
    assert validate(out, q_ids, p_ids)
    for bad in ({"a": out["a"]}, {"a": out["a"][:6], "b": out["b"]}, {"a": ["zz"] * 7, "b": out["b"]},
                {"a": out["a"][:6] + [out["a"][0]], "b": out["b"]}):
        with pytest.raises(ValueError):
            validate(bad, q_ids, p_ids)


def test_bm25_matches_formula():
    class S:
        def __init__(self, docs):
            self.docs = [np.array(d, dtype=np.uint16) for d in docs]

        def __len__(self):
            return len(self.docs)

        def doc(self, i):
            return self.docs[i]

    pool = S([[1, 2, 2, 3], [2, 4, 4, 4, 5], [6, 7], [1, 1, 1, 8, 9, 2]])
    qs = S([[1, 2, 9], [4, 7, 7]])
    k1, b = 1.2, 0.75
    got = BM25(pool, k1, b).scores(qs)
    N, avg = 4, np.mean([len(d) for d in pool.docs])
    for qi, q in enumerate(qs.docs):
        for di, d in enumerate(pool.docs):
            want = 0.0
            for t in set(q.tolist()):
                df = sum(t in x.tolist() for x in pool.docs)
                tf = d.tolist().count(t)
                idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
                want += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg))
            assert abs(got[qi, di] - want) < 1e-4
    hn = hard_negatives(got, [[0], [1]], ["q0", "q1"], ["p0", "p1", "p2", "p3"], n=2)
    assert 0 not in hn[0] and 1 not in hn[1] and all(len(h) == 2 for h in hn)


def test_experiments_resolve():
    p7 = {"full", "partial", "qlora", "lorma", "prefix", "prompt", "ia3", "frozen"}
    for n in names():
        assert get_experiment(n)["method"] in p7, n
    six = [get_experiment(n) for n in ("A_lorma", "B_full", "C_ia3", "D_partial", "G_qlora", "As_kd")]
    assert {c["backbone"] for c in six} >= {"A", "B", "C", "D"} and len({c["method"] for c in six[:5]}) == 5 and six[5]["kd"]


QUICK = dict(tiny=True, tokenizer=TOK, epochs=1, budget_min=None, n_evals=1, probe_steps=1, predict_test=False,
             save_emb=False, save_weights=False, seq_len=32, train_chunks=2, eval_chunks=2, eval_chunks_fast=1, q_batch=3,
             max_train_queries=6, log_every=1000)


@need_tok
@pytest.mark.parametrize("exp", [n for n in names() if not n.startswith("smoke") and not get_experiment(n)["kd"]])
def test_every_experiment_runs(exp, synth_root, tmp_path):
    from phase7.train import run
    r = run(get_experiment(exp, dict(QUICK, seq_batch=4)), synth_root, [os.path.join(ROOT, "results")], str(tmp_path))
    assert r["steps"] == 2 and 0 <= r["dev_hybrid"]["f1"] <= 1


@need_tok
def test_end_to_end_tiny(synth_root, data, tmp_path):
    from phase7 import ensemble
    from phase7.session import summarize
    from phase7.train import run
    out = str(tmp_path / "runs")
    roots = [os.path.join(ROOT, "results")]
    r = run(get_experiment("smoke_G", {"tiny": True, "tokenizer": TOK}), synth_root, roots, out)
    assert os.path.exists(os.path.join(out, "smoke_G", "emb_train.npz")) and r["dev_hybrid"]["f1"] >= r["dev_bm25"]["f1"] - 1e-9
    run(get_experiment("smoke_As_kd", {"tiny": True, "tokenizer": TOK}), synth_root, roots, out)
    run(get_experiment("smoke_D", {"tiny": True, "tokenizer": TOK}), synth_root, roots, out)
    ensemble.main(["--runs", "smoke_G", "smoke_As_kd", "--data", synth_root, "--roots", *roots, "--out", out, "--name", "ens"])
    te = data["test"]
    for z in ("smoke_G/submission.zip", "smoke_G/submission_dense.zip", "smoke_As_kd/submission.zip", "smoke_D/submission.zip",
              "ens/submission.zip"):
        with zipfile.ZipFile(os.path.join(out, z)) as f:
            assert f.namelist() == ["prediction.json"]
            pj = json.loads(f.read("prediction.json"))
            assert validate(pj, te.queries.ids, te.pool.ids) and all(q not in v for q, v in pj.items())
    assert {"smoke_G", "ens"} <= {x["run"] for x in summarize(out)}


def _p7_run(root, nb, run, backbone, method, f1, kd=None, pred=None):
    d = root / nb / "runs_p7" / run
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"backbone": backbone, "method": method, "kd": kd and {"teacher": kd},
                                               "dev_hybrid": {"f1": f1, "precision": f1, "recall": f1},
                                               "dev_dense": {"f1": f1 - 0.01}, "alpha": 0.6}))
    pred = pred or {q: [f"c{i}" for i in range(7)] for q in ("q1", "q2")}
    with zipfile.ZipFile(d / "submission.zip", "w") as z:
        z.writestr("prediction.json", json.dumps(pred))


def test_upload_plan_p7(tmp_path):
    from phase6.upload import collect, plan
    from phase7 import upload
    R = tmp_path / "res"
    _p7_run(R, "s1", "A_lorma", "A", "lorma", 0.20)
    _p7_run(R, "s1", "G_qlora", "G", "qlora", 0.25)
    _p7_run(R, "s1", "As_kd", "As", "full", 0.22, kd="G_lorma")
    _p7_run(R, "s1", "ens_GA", "A+G", "ensemble", 0.27)
    _p7_run(R, "s1", "smoke_G", "G", "lorma", 0.15)
    _p7_run(R, "s2", "B_full", "B", "full", 0.15)
    _p7_run(R, "s2", "C_ia3", "C", "ia3", 0.12, pred={"q1": ["c1"] * 7, "q2": [f"c{i}" for i in range(7)]})
    _p7_run(R, "s2", "D_partial", "D", "partial", 0.10, pred={"q1": [f"c{i}" for i in range(7)]})
    order, rejected, missing = plan(collect(str(R), metrics=upload.metrics, score="f1_hybrid"), 40, upload.SIX, upload.check)
    assert [(r["slot"], r["run"]) for r in order] == [("LoRMA", "A_lorma"), ("full", "B_full"), ("QLoRA", "G_qlora"),
                                                      ("KD", "As_kd"), ("", "ens_GA")]
    assert missing == ["IA3", "partial"] and {r for r, _ in rejected} == {"C_ia3", "D_partial"}
    assert order[0]["metrics"]["p@7"] == 0.20


def test_fusion_end_to_end(synth_root, data, tmp_path):
    from phase7 import fusion
    from phase7.score import BM25
    tr, dv, te = data["train"], data["dev"], data["test"]
    runs = tmp_path / "runs" / "nb" / "runs_p7"
    for name, noise in (("good", 0.0), ("noisy", 3.0)):
        d = runs / name
        d.mkdir(parents=True)
        rng = np.random.default_rng(0)
        for s, sp_ in (("dev", dv), ("test", te)):
            dense = BM25(sp_.pool).scores(sp_.queries) + noise * rng.standard_normal((len(sp_.queries), len(sp_.pool)))
            np.savez(d / f"scores_{s}.npz", dense=dense.astype(np.float16), q_ids=np.array(sp_.queries.ids),
                     p_ids=np.array(sp_.pool.ids))
        (d / "result.json").write_text(json.dumps({"dev_hybrid": {"f1": 0.5 if name == "good" else 0.1}}))
    rep = fusion.run(synth_root, str(tmp_path / "runs"), [ROOT], str(tmp_path / "out"), tok_path=TOK)
    assert rep["dense_runs"] == ["good", "noisy"]
    assert rep["dev_f1_fused_tuned"] >= rep["dev_dense_top1"] > 0.3
    for v in ("fused", "dense", "lexical"):
        with zipfile.ZipFile(tmp_path / "out" / v / "submission.zip") as z:
            pred = json.loads(z.read("prediction.json"))
        assert set(pred) == set(te.queries.ids) and all(len(set(x)) == 7 for x in pred.values())
        assert all(q not in x for q, x in pred.items())


def test_fusion_features():
    from phase7 import fusion
    assert fusion.latest_year("decided in 1987, citing a 1952 case") == 1987 and fusion.latest_year("no year") == 0

    class S:
        def __init__(self, docs):
            self.docs = [np.array(d, np.uint16) for d in docs]

        def __len__(self):
            return len(self.docs)

        def doc(self, i):
            return self.docs[i]

    pool = S([[1, 2, 3, 4], [4, 3, 2, 1], [9, 9, 9, 9]])
    q = S([[1, 2, 3]])
    s2 = fusion.bm25(fusion.ngram_matrix(pool, 2), fusion.ngram_matrix(q, 2), 1.2, 0.75)
    assert s2[0, 0] > 0 and s2[0, 1] == 0 and s2[0, 2] == 0
