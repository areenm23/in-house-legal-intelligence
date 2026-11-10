"""Phase-3 tests: format detection, scorer parity (sklearn + official evaluator), chunks/windows, model
context modes, KD, end-to-end tiny runs with valid submission zips.

    python -m pytest phase3/tests -q
"""
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from ft.backbones import load_backbone
from ft.peft import apply_method
from phase3 import data as D
from phase3 import synth
from phase3.experiments import get_experiment, names
from phase3.model import RRModel
from phase3.score import fit_bias, prf, score, validate

TOK = os.path.join(ROOT, "results", "results", "p1data_small", "tokenizer", "a1bpe_32755.json")
EVAL = os.path.join(ROOT, "course_files", "eval", "Evaluation Script", "evaluate_phase3.py")
need_tok = pytest.mark.skipif(not os.path.exists(TOK), reason="Phase-1 tokenizer not present")
LABELS = synth.LABELS


@pytest.fixture(scope="module")
def tok():
    from phase1.data.a1_bpe import load_tokenizer
    return load_tokenizer(TOK)


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory):
    return synth.write(str(tmp_path_factory.mktemp("p3")), docs=6, seed=1)


@pytest.mark.parametrize("raw", [
    [{"id": "d1", "text": ["a b.", "c d."], "labels": ["Fact", "Issue"]}],
    {"d1": {"text": ["a b.", "c d."], "labels": ["Fact", "Issue"]}},
    {"data": [{"doc_id": "d1", "sentences": ["a b.", "c d."], "label": [0, 1]}]},
    [{"id": "d1", "text": [{"text": "a b."}, {"text": "c d."}], "labels": [{"label": "Fact"}, {"label": "Issue"}]}],
    [{"id": "d1", "text": "a b.\nc d.", "labels": [["Fact"], ["Issue"]]}],
])
def test_normalize_formats(raw):
    recs = D.normalize_docs(raw, LABELS)
    assert recs == [("d1", ["a b.", "c d."], ["Fact", "Issue"])]


def test_normalize_mismatch_raises():
    with pytest.raises(ValueError):
        D.normalize_docs([{"id": "x", "text": ["a"], "labels": ["Fact", "Fact"]}], LABELS)


@pytest.mark.parametrize("v", [LABELS, {l: i for i, l in enumerate(LABELS)}, {str(i): l for i, l in enumerate(LABELS)},
                               {"labels": LABELS}])
def test_label_vocab_formats(tmp_path, v):
    p = tmp_path / "label_vocab.json"
    p.write_text(json.dumps(v))
    assert D.load_labels(str(p)) == LABELS


def test_load_split_synth(synth_root):
    data = D.load_split(synth_root)
    assert data["labels"] == LABELS
    assert len(data["train"]["CL"]) == 6 and len(data["dev"]["IT"]) == 2
    assert any(len(d) == 0 for d in data["test"]["IT"])
    assert all(d.labels is None for d in D.all_docs(data["test"]))
    assert D.find_root(os.path.dirname(synth_root)) == synth_root


def test_prf_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(20):
        yt = rng.integers(0, 13, 300)
        yp = np.where(rng.random(300) < 0.6, yt, rng.integers(0, 10, 300))
        s = prf(yt, yp, 13)
        L = list(range(13))
        assert abs(s["macro_f1"] - sk.f1_score(yt, yp, labels=L, average="macro", zero_division=0)) < 1e-12
        assert abs(s["macro_p"] - sk.precision_score(yt, yp, labels=L, average="macro", zero_division=0)) < 1e-12
        assert abs(s["macro_r"] - sk.recall_score(yt, yp, labels=L, average="macro", zero_division=0)) < 1e-12


@need_tok
@pytest.mark.skipif(not os.path.exists(EVAL), reason="official evaluator not unpacked")
def test_official_parity(synth_root):
    pytest.importorskip("sklearn")
    from phase3.report import official_parity
    data = D.load_split(synth_root)
    dev = D.all_docs(data["dev"])
    rng = np.random.default_rng(3)
    pred = {d.key: np.where(rng.random(len(d)) < 0.5, d.y, rng.integers(0, 13, len(d))) for d in dev}
    off, ours, err = official_parity(dev, pred, LABELS, data["files"]["labels"])
    assert err == "" and abs(off - ours) < 1e-5


def test_validate_catches_errors(synth_root):
    data = D.load_split(synth_root)
    test = D.all_docs(data["test"])
    ok = {d.id: ["Fact"] * len(d) for d in test}
    assert validate(ok, test, LABELS)
    bad_len = dict(ok); bad_len[test[0].id] = ["Fact"] * (len(test[0]) + 1)
    bad_lab = dict(ok); bad_lab[test[0].id] = ["fact"] * len(test[0])
    missing = dict(ok); missing.pop(test[0].id)
    for pj in (bad_len, bad_lab, missing):
        with pytest.raises(ValueError):
            validate(pj, test, LABELS)


def test_fit_bias_never_hurts():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 400)
    lp = np.log(np.full((400, 5), 0.1)) + rng.normal(0, 1, (400, 5))
    lp[np.arange(400), y] += 0.8
    b, f = fit_bias(lp, y, 5)
    assert f >= prf(y, lp.argmax(1), 5)["macro_f1"] - 1e-12


@need_tok
def test_encode_and_chunks(tok, synth_root):
    data = D.encode_all(D.load_split(synth_root), tok)
    for d in D.all_docs(data["train"]):
        assert d.n_tok == len(d.ids) == len(d.sent_of)
        assert np.all(np.diff(d.sent_of) >= 0) and set(d.sent_of.tolist()) == set(range(len(d)))
        for cap in (16, 64, 4096):
            for rng in (None, np.random.default_rng(0)):
                ch = D.chunks(d, cap, rng)
                assert ch[0][0] == 0 and ch[-1][1] == len(d)
                assert all(a[1] == b[0] for a, b in zip(ch, ch[1:]))
                r = D.sentence_token_ranges(d)
                for s0, s1 in ch:
                    assert s1 - s0 == 1 or (r[s1 - 1, 1] - r[s0, 0]) <= cap
    d = D.all_docs(data["train"])[1]
    assert d.ids.tolist() == tok.encode(" ".join(s for s in d.sents)).ids


@pytest.mark.parametrize("n,size,stride", [(5, 8, 4), (8, 8, 4), (9, 8, 4), (100, 32, 16), (100, 32, 32), (50, 0, 0)])
def test_windows_cover(n, size, stride):
    w = D.windows(n, size, stride)
    cov = np.zeros(n, int)
    for s, e in w:
        cov[s:e] += 1
        assert size <= 0 or e - s == min(size, n)
    assert cov.min() >= 1 and w[-1][1] == n


def _doc(tok, sents):
    d = D.RRDoc("x", "CL", sents, ["Fact"] * len(sents), {l: i for i, l in enumerate(LABELS)})
    return d.encode(tok)


@need_tok
@pytest.mark.parametrize("key", ["A", "B", "C", "D", "G"])
def test_model_context_modes(tok, key):
    torch.manual_seed(0)
    d = _doc(tok, ["The appellant filed an appeal.", "The issue is jurisdiction under section 10.", "Appeal allowed."])
    bb = load_backbone(key, tiny=True)
    apply_method(bb, "frozen")
    cfg = dict(window=512, stride=256, eval_batch=8)
    m = RRModel(bb, 13, ctx="win", head="linear", pos_feat=False, dom_feat=False, dropout=0.0).eval()
    with torch.no_grad():
        H = bb(torch.from_numpy(d.ids.astype(np.int64))[None])[0]
        sof = torch.from_numpy(d.sent_of.astype(np.int64))
        want = torch.stack([H[sof == i].mean(0) for i in range(len(d))])
        got = m(d, 0, len(d), 0, cfg, torch.device("cpu"), False)
        assert torch.allclose(got, m.out(want), atol=1e-4)
        m.ctx = "sent"
        want = torch.stack([bb(torch.from_numpy(d.ids[d.sent_of == i].astype(np.int64))[None])[0].mean(0)
                            for i in range(len(d))])
        got = m(d, 0, len(d), 0, cfg, torch.device("cpu"), False)
        assert torch.allclose(got, m.out(want), atol=1e-4)
        m.ctx = "win"
        part = m(d, 1, 3, 0, cfg, torch.device("cpu"), False)
        assert part.shape == (2, 13)


@need_tok
def test_overlapping_windows_equal_single_window_for_bag_model(tok):
    """With a per-token model (embedding only, no mixing), overlapping-window averaging must give exactly
    the single-pass vectors: checks the overlap bookkeeping."""
    d = _doc(tok, ["word " * 30, "other words here " * 10, "end."])
    bb = load_backbone("G", tiny=True)
    bb.lm.blocks = torch.nn.ModuleList()
    apply_method(bb, "frozen")
    m = RRModel(bb, 13, head="linear", pos_feat=False, dom_feat=False, dropout=0.0).eval()
    with torch.no_grad():
        a = m(d, 0, len(d), 0, dict(window=4096, stride=4096, eval_batch=8), torch.device("cpu"), False)
        b = m(d, 0, len(d), 0, dict(window=16, stride=5, eval_batch=3), torch.device("cpu"), False)
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("pool", ["mean", "max", "attn"])
def test_pool_shapes(pool):
    bb = load_backbone("G", tiny=True)
    m = RRModel(bb, 13, pool=pool)
    H = torch.randn(10, bb.d_model)
    sof = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])
    v = m._pool(H, sof, 4)
    assert v.shape == (4, bb.d_model)
    if pool == "max":
        assert torch.allclose(v[1], H[2:5].max(0).values)


def test_kd_loss_zero_when_equal():
    from phase3.train import kd_loss
    t = torch.log_softmax(torch.randn(7, 13), -1)
    assert kd_loss(t.clone(), t, 2.0).abs() < 1e-6
    assert kd_loss(torch.zeros(7, 13), t, 2.0) > 0


def test_experiments_resolve():
    for n in names():
        cfg = get_experiment(n)
        assert cfg["backbone"] in ("A", "As", "B", "C", "D", "G")
    techs = {get_experiment(n)["method"] for n in ("A_lorma", "B_rslora", "C_ia3", "D_lora", "G_adalora")}
    assert techs == {"lorma", "rslora", "ia3", "lora", "adalora"} and get_experiment("As_kd")["kd"]


@need_tok
def test_end_to_end_tiny(synth_root, tmp_path):
    out = str(tmp_path / "runs")
    from phase3.train import run
    roots = [os.path.join(ROOT, "results")]
    r = run(get_experiment("smoke_G", {"tiny": True}), synth_root, roots, out)
    assert 0 <= r["dev"]["macro_f1"] <= 1 and os.path.exists(os.path.join(out, "smoke_G", "soft_labels.npz"))
    r2 = run(get_experiment("smoke_As_kd", {"tiny": True}), synth_root, roots, out)
    assert r2["dev"]["macro_f1"] >= 0
    r3 = run(get_experiment("smoke_G", {"tiny": True}), synth_root, roots, out, mode="full")
    assert r3["best_epoch"] >= 1 and "dev" not in r3
    data = D.load_split(synth_root)
    test = D.all_docs(data["test"])
    for z in ("smoke_G/submission.zip", "smoke_G/submission_bias.zip", "smoke_G__full/submission.zip"):
        with zipfile.ZipFile(os.path.join(out, z)) as f:
            assert f.namelist() == ["prediction.json"]
            assert validate(json.loads(f.read("prediction.json")), test, LABELS)
    from phase3 import ensemble
    res = ensemble.main(["--runs", "smoke_G", "smoke_As_kd", "--data", synth_root, "--roots", *roots, "--out", out,
                         "--name", "ens", "--bias"])
    assert "dev" in res and os.path.exists(os.path.join(out, "ens", "submission_bias.zip"))
