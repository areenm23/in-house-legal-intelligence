"""Phase-4 tests: format detection, cache, scorer parity (sklearn + official evaluator), validation,
document views, model encoding (no padding, every family), gradual unfreezing / layer-wise LR, KD,
experiments, and end-to-end tiny runs (train -> hier -> KD -> ensemble) with valid submission zips.

    python -m pytest phase4/tests -q
"""
import json
import os
import sys
import zipfile

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from ft.backbones import load_backbone
from ft.peft import GradualUnfreezer, apply_method, layer_depth
from phase4 import data as D
from phase4 import synth
from phase4.experiments import get_experiment, names
from phase4.model import CJPEModel
from phase4.score import fit_threshold, prf, validate

TOK = os.path.join(ROOT, "results", "results", "p1data_small", "tokenizer", "a1bpe_32755.json")
EVAL = os.path.join(ROOT, "course_files", "eval", "Evaluation Script", "evaluate_phase4.py")
need_tok = pytest.mark.skipif(not os.path.exists(TOK), reason="Phase-1 tokenizer not present")
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory):
    return synth.write(str(tmp_path_factory.mktemp("p4")), docs=80, seed=1)


@pytest.fixture(scope="module")
def data(synth_root, tmp_path_factory):
    return D.load_data(synth_root, TOK, str(tmp_path_factory.mktemp("cache")), workers=1)


@pytest.mark.parametrize("raw", [
    [{"id": "c1", "text": "a b c", "label": 1}, {"id": "c2", "text": "d e", "label": 0}],
    {"c1": {"text": "a b c", "label": 1}, "c2": {"text": "d e", "label": 0}},
    {"data": [{"case_id": "c1", "case_text": "a b c", "gold": "1"}, {"case_id": "c2", "case_text": "d e", "gold": "0"}]},
    [{"name": "c1", "text": ["a b", "c"], "label": True}, {"name": "c2", "text": ["d", "e"], "label": False}],
    [{"id": "c1", "document": "a b c", "outcome": "accepted"}, {"id": "c2", "document": "d e", "outcome": "rejected"}],
])
def test_normalize_formats(raw):
    assert D.normalize(raw) == [("c1", "a b c", 1), ("c2", "d e", 0)]


def test_normalize_bad_label():
    with pytest.raises(ValueError):
        D.normalize([{"id": "x", "text": "a", "label": 2}])
    assert D.normalize([{"id": "x", "text": "a", "label": -1}], labels=False) == [("x", "a", None)]


def test_jsonl_and_missing_test_ids(tmp_path):
    root = tmp_path / "Phase_4_CJPE"
    (root / "Train").mkdir(parents=True)
    (root / "Test").mkdir()
    (root / "Train" / "train.json").write_text("\n".join(json.dumps({"id": f"t{i}", "text": "x y", "label": i % 2})
                                                         for i in range(4)))
    (root / "Test" / "test.json").write_text(json.dumps([{"text": "no id"}]))
    assert D.find_root(str(tmp_path)) == str(root)
    recs, _ = D.read_split(str(root), "train")
    assert [r[2] for r in recs] == [0, 1, 0, 1]
    with pytest.raises(ValueError):
        D.read_split(str(root), "test")


@need_tok
def test_load_data_and_cache(synth_root, data, tmp_path):
    tr, dv, te = data["train"], data["dev"], data["test"]
    assert len(tr) == 80 and tr.labelled and dv.labelled and not te.labelled
    assert te.ids[-1] == "test_empty" and len(te.doc(len(te) - 1)) == 1
    from phase1.data.a1_bpe import load_tokenizer
    raw = json.load(open(os.path.join(synth_root, "Train", "train.json")))
    tok = load_tokenizer(TOK)
    for i in (0, 5, 17):
        assert tr.doc(i).tolist() == tok.encode(raw[i]["text"]).ids
        assert tr.y[i] == raw[i]["label"] and tr.ids[i] == raw[i]["id"]
    first = D.load_data(synth_root, TOK, str(tmp_path / "c2"), workers=1)
    assert first["files"] and len(os.listdir(tmp_path / "c2")) == 6
    again = D.load_data(synth_root, TOK, str(tmp_path / "c2"), workers=1)
    assert not again["files"]
    for s in ("train", "dev", "test"):
        assert np.array_equal(again[s].tok, data[s].tok) and np.array_equal(again[s].off, data[s].off)
        assert again[s].ids == data[s].ids and np.array_equal(again[s].y, data[s].y) and again[s].h == data[s].h
    texts = [r["text"] for r in raw]
    par = D.encode_texts(texts * 4, TOK, workers=2, part=16)
    assert all(np.array_equal(a, b) for a, b in zip(par[:len(texts)], D.encode_texts(texts, TOK, workers=1)))


def test_stratified_subset():
    y = np.array([0] * 700 + [1] * 300)
    s = D.stratified_subset(y, 100, seed=0)
    assert len(s) == 100 and abs(y[s].mean() - 0.3) < 0.02 and len(set(s)) == 100
    assert np.array_equal(D.stratified_subset(y, None), np.arange(1000))
    assert np.array_equal(s, D.stratified_subset(y, 100, seed=0))


def test_prf_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for k in range(30):
        yt = (rng.random(200) < 0.3 + 0.02 * k).astype(int)
        yp = np.where(rng.random(200) < 0.7, yt, rng.integers(0, 2, 200)) if k % 5 else np.zeros(200, int)
        s = prf(yt, yp)
        assert abs(s["macro_f1"] - sk.f1_score(yt, yp, average="macro", labels=[0, 1], zero_division=0)) < 1e-12
        assert abs(s["precision"] - sk.precision_score(yt, yp, average="binary", pos_label=1, zero_division=0)) < 1e-12
        assert abs(s["recall"] - sk.recall_score(yt, yp, average="binary", pos_label=1, zero_division=0)) < 1e-12


@pytest.mark.skipif(not os.path.exists(EVAL), reason="official evaluator not unpacked")
def test_official_parity():
    pytest.importorskip("sklearn")
    from phase4.report import official_parity
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 150)
    pred = np.where(rng.random(150) < 0.6, y, 1 - y)
    off, ours, err = official_parity([f"d{i}" for i in range(150)], y, pred)
    assert err == "" and abs(off - ours) < 1e-5


def test_validate_catches_errors():
    ids = ["a", "b", "c"]
    ok = [{"id": i, "pred": 0} for i in ids]
    assert validate(ok, ids)
    bad = [
        [],
        ok[:2],
        ok + [{"id": "a", "pred": 1}],
        [{"id": "a", "pred": True}] + ok[1:],
        [{"id": "a", "pred": 2}] + ok[1:],
        [{"id": "a", "pred": 1.0}] + ok[1:],
        [{"id": 1, "pred": 0}] + ok[1:],
        [{"id": "a"}] + ok[1:],
    ]
    for pj in bad:
        with pytest.raises(ValueError):
            validate(pj, ids)


def test_fit_threshold_never_hurts():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.35).astype(int)
    p = np.clip(0.3 + 0.25 * y + rng.normal(0, 0.2, 500), 0, 1)
    t, f = fit_threshold(p, y)
    assert f >= prf(y, (p >= 0.5).astype(int))["macro_f1"] - 1e-12 and 0.05 <= t <= 0.95


@pytest.mark.parametrize("n,L", [(1, 8), (8, 8), (9, 8), (16, 8), (17, 8), (100, 8), (1000, 64)])
def test_chunk_starts_cover(n, L):
    st = D.chunk_starts(n, L)
    cov = np.zeros(n, int)
    for s in st:
        cov[s:s + L] += 1
        assert min(L, n) == len(range(s, min(s + L, n)))
    assert cov.min() >= 1 and st[0] == 0 and st[-1] == max(0, n - L) and np.all(np.diff(st) > 0)


def test_views():
    t = np.arange(1000, dtype=np.uint16)
    L = 100
    (s,), _ = D.views(t, "first", L)
    assert s.tolist() == list(range(100))
    (s,), p = D.views(t, "last", L)
    assert s.tolist() == list(range(900, 1000)) and abs(p[0] - 0.95) < 1e-9
    (s,), _ = D.views(t, "middle", L)
    assert s.tolist() == list(range(450, 550))
    (s,), _ = D.views(t, "headtail", L, head_tokens=30)
    assert s.tolist() == list(range(30)) + list(range(930, 1000))
    seqs, pos = D.views(t, "hier", L, k=4, pick="last")
    assert [q[0] for q in seqs] == [600, 700, 800, 900] and all(len(q) == L for q in seqs) and np.all(np.diff(pos) > 0)
    seqs, _ = D.views(t, "hier", L, k=3, pick="spread")
    assert [q[0] for q in seqs] == [0, 400, 900]
    seqs, _ = D.views(t, "hier", L, k=4, pick="random", rng=np.random.default_rng(0))
    assert len(seqs) == 4 and all(len(q) == L for q in seqs)
    seqs, _ = D.views(t[:30], "hier", L, k=4)
    assert len(seqs) == 1 and len(seqs[0]) == 30
    for ctx in ("first", "last", "middle", "headtail"):
        (s,), _ = D.views(t[:30], ctx, L)
        assert s.tolist() == list(range(30))
    (s,), _ = D.views(np.zeros(0, np.uint16), "last", L)
    assert s.tolist() == [D.PLACEHOLDER]


@pytest.mark.parametrize("key", ["A", "B", "C", "D", "G", "As"])
def test_encode_matches_backbone(key):
    torch.manual_seed(0)
    bb = load_backbone(key, tiny=True)
    apply_method(bb, "frozen")
    m = CJPEModel(bb, pool="mean_last", agg="attn", dropout=0.0).eval()
    rng = np.random.default_rng(0)
    seqs = [rng.integers(4, 30000, n).astype(np.int64) for n in (16, 16, 7, 16, 3)]
    with torch.no_grad():
        got = m.encode(seqs, CPU, False, seq_batch=2)
        for s, g in zip(seqs, got):
            h = bb(torch.from_numpy(s)[None])[0]
            assert torch.allclose(g, torch.cat([h.mean(0), h[-1]]), atol=1e-4)
        lg, dv = m([(seqs[:2], np.array([0.2, 0.9])), (seqs[2:3], np.array([0.5])), (seqs[3:], np.array([0.1, 0.6]))],
                   CPU, False, 4)
    assert lg.shape == (3, 2) and dv.shape == (3, m.d_doc)
    assert torch.allclose(dv[1], got[2], atol=1e-5)


@pytest.mark.parametrize("agg", ["mean", "max", "attn", "gru"])
@pytest.mark.parametrize("pool", ["mean", "max", "last", "attn"])
def test_pool_agg_shapes(pool, agg):
    bb = load_backbone("G", tiny=True)
    m = CJPEModel(bb, pool=pool, agg=agg, len_feat=True, head="mlp")
    seqs = [np.arange(4, 20, dtype=np.int64)] * 3
    lg, dv = m([(seqs, np.array([0.1, 0.5, 0.9])), (seqs[:1], np.array([0.5]))], CPU, False, 8, n_tok=[16, 50000])
    assert lg.shape == (2, 2) and torch.isfinite(lg).all()
    lg.sum().backward()


def test_gradual_unfreezer_and_layer_depth():
    bb = load_backbone("A", tiny=True)
    apply_method(bb, "gradual")
    assert not any(p.requires_grad for p in bb.parameters())
    u = GradualUnfreezer(bb, frac=0.25)
    L = len(bb.layers())
    assert len(u.groups) == L + 1
    assert u.step(0, 100) == 1 and all(p.requires_grad for p in bb.layers()[-1].parameters())
    assert not any(p.requires_grad for p in bb.layers()[0].parameters())
    assert u.step(30, 100) == 2 and all(p.requires_grad for p in bb.layers()[0].parameters())
    assert not bb.embedding().weight.requires_grad
    assert u.step(99, 100) == L + 1 and bb.embedding().weight.requires_grad
    u2 = GradualUnfreezer(bb, frac=0.1, max_groups=1)
    assert u2.step(99, 100) == 1
    names = dict(bb.named_parameters())
    assert layer_depth(bb, "lm.tok_emb.weight") == 0
    assert layer_depth(bb, next(n for n in names if n.startswith("lm.blocks.1."))) == 2
    assert layer_depth(bb, "lm.norm.weight") == L + 1


def test_param_groups_llrd():
    from phase4.train import param_groups
    bb = load_backbone("G", tiny=True)
    apply_method(bb, "full")
    m = CJPEModel(bb)
    cfg = dict(lr=1e-3, head_lr=5e-3, weight_decay=0.01, llrd=0.5)
    groups = param_groups(m, cfg)
    lrs = sorted({g["lr"] for g in groups})
    L = len(bb.layers())
    want = sorted({1e-3 * 0.5 ** k for k in range(L + 2)} | {5e-3})
    assert np.allclose(lrs, want)
    n = sum(p.numel() for g in groups for p in g["params"])
    assert n == sum(p.numel() for p in m.parameters())
    bb2 = load_backbone("G", tiny=True)
    apply_method(bb2, "gradual")
    m2 = CJPEModel(bb2)
    u = GradualUnfreezer(bb2)
    g2 = param_groups(m2, dict(cfg, llrd=None), u.params())
    assert sum(p.numel() for g in g2 for p in g["params"]) == sum(p.numel() for p in m2.parameters())


def test_kd_terms_and_micro_batches():
    from phase4.train import kd_terms, micro_batches
    t = torch.randn(6, 2)
    assert kd_terms(t.clone(), t, 2.0).abs().max() < 1e-6 and (kd_terms(torch.zeros(6, 2), t * 3, 2.0) > 0).all()
    vs = [([0] * k, None) for k in (4, 4, 1, 8, 2, 2)]
    mb = micro_batches(vs, 8)
    assert sum(mb, []) == list(range(6)) and all(sum(len(vs[j][0]) for j in g) <= 8 or len(g) == 1 for g in mb)


def test_experiments_resolve():
    phase4_list = {"full", "partial", "gradual", "bitfit", "lorma", "adalora", "houlsby", "pfeiffer", "frozen"}
    for n in names():
        cfg = get_experiment(n)
        assert cfg["backbone"] in ("A", "As", "B", "C", "D", "G")
        assert cfg["method"] in phase4_list, n
    six = ["A_lorma", "B_bitfit", "C_houlsby", "D_partial", "G_gradual", "As_kd"]
    cfgs = [get_experiment(n) for n in six]
    assert {c["backbone"] for c in cfgs} >= {"A", "B", "C", "D"}
    assert len({c["method"] for c in cfgs[:5]}) == 5 and cfgs[5]["kd"] and "lorma" in {c["method"] for c in cfgs}
    assert get_experiment("G_lorma", {"extract.max_chunks": 8})["extract"]["max_chunks"] == 8


QUICK = dict(tiny=True, tokenizer=TOK, epochs=1, max_train_docs=16, batch=8, eval_docs=8, n_evals=1, probe_steps=1,
             budget_min=None, predict_test=False, save_soft=False, extract=None, save_weights=False, eval_chunks=2,
             seq_len=48, head_tokens=12, log_every=1000)


@need_tok
@pytest.mark.parametrize("exp", [n for n in names() if not n.startswith("smoke") and not get_experiment(n)["kd"]])
def test_every_experiment_runs(exp, synth_root, tmp_path):
    from phase4.train import run
    over = dict(QUICK, seq_batch=min(get_experiment(exp)["seq_batch"], 4))
    r = run(get_experiment(exp, over), synth_root, [os.path.join(ROOT, "results")], str(tmp_path))
    assert r["steps"] == 2 and 0 <= r["dev"]["macro_f1"] <= 1
    if get_experiment(exp)["method"] == "gradual":
        assert r["params_trainable"] == r["params_total"]


@need_tok
def test_end_to_end_tiny(synth_root, data, tmp_path):
    from phase4 import ensemble, hier
    from phase4.train import run
    out = str(tmp_path / "runs")
    roots = [os.path.join(ROOT, "results")]
    r = run(get_experiment("smoke_G", {"tiny": True, "tokenizer": TOK}), synth_root, roots, out)
    assert 0 <= r["dev"]["macro_f1"] <= 1 and r["soft_labels"] == r["train_docs"]
    for f in ("soft_labels.npz", "chunks_train.npz", "chunks_dev.npz", "chunks_test.npz", "submission_thr.zip"):
        assert os.path.exists(os.path.join(out, "smoke_G", f)), f
    res = hier.run_all(os.path.join(out, "smoke_G"), out, variants=["attn", "gru", "q1", "last4"], epochs=3)
    assert set(res) == {"attn", "gru", "q1", "last4"}
    r2 = run(get_experiment("smoke_As_kd", {"tiny": True, "tokenizer": TOK}), synth_root, roots, out)
    assert r2["dev"]["macro_f1"] >= 0 and r2["kd"]["teacher"] == "smoke_G"
    r3 = run(get_experiment("smoke_G_hier", {"tiny": True, "tokenizer": TOK}), synth_root, roots, out)
    assert r3["n_chunks"] == 3
    te = data["test"]
    for z in ("smoke_G/submission.zip", "smoke_G/submission_thr.zip", "smoke_G__h_gru/submission.zip",
              "smoke_As_kd/submission.zip", "smoke_G_hier/submission.zip"):
        with zipfile.ZipFile(os.path.join(out, z)) as f:
            assert f.namelist() == ["prediction.json"]
            assert validate(json.loads(f.read("prediction.json")), te.ids)
    e = ensemble.main(["--runs", "smoke_G", "smoke_G__h_gru", "smoke_As_kd", "--out", out, "--name", "ens"])
    assert "dev" in e and os.path.exists(os.path.join(out, "ens", "submission_thr.zip"))
    from phase4.session import summarize
    rows = summarize(out)
    assert {"smoke_G", "smoke_G__h_gru", "ens"} <= {x["run"] for x in rows}
