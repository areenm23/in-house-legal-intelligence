"""Phase-5 tests: multi-label loading, clean dev, scorer parity (sklearn + official evaluator, including its
label-dimension rule), thresholds, validation, losses / KD, every experiment on tiny backbones, and an
end-to-end run (train -> hier -> KD -> ensemble) with valid zips.

    python -m pytest phase5/tests -q
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

from phase5 import data as D
from phase5 import synth
from phase5.experiments import get_experiment, names
from phase5.score import (apply, cross_fit, decide, fit_global, fit_per_label, metrics,
                          to_prediction_json, validate)

TOK = os.path.join(ROOT, "results", "results", "p1data_small", "tokenizer", "a1bpe_32755.json")
EVAL = os.path.join(ROOT, "course_files", "eval", "Evaluation Script", "evaluate_phase5.py")
need_tok = pytest.mark.skipif(not os.path.exists(TOK), reason="Phase-1 tokenizer not present")


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory):
    return synth.write(str(tmp_path_factory.mktemp("p5")), docs=120, seed=1)


@pytest.fixture(scope="module")
def data(synth_root, tmp_path_factory):
    return D.load_data(synth_root, TOK, str(tmp_path_factory.mktemp("cache")), workers=1)


@pytest.mark.parametrize("raw", [
    [{"id": "c1", "text": "a b", "labels": [7, 1, 7]}, {"id": "c2", "text": "c", "labels": [3]}],
    {"c1": {"text": "a b", "labels": ["1", "7"]}, "c2": {"text": "c", "labels": 3}},
    {"data": [{"case_id": "c1", "facts": ["a", "b"], "statutes": [1.0, 7]}, {"case_id": "c2", "facts": "c", "statutes": [3]}]},
])
def test_normalize_formats(raw):
    assert D.normalize(raw) == [("c1", "a b", [1, 7]), ("c2", "c", [3])]


@pytest.mark.parametrize("bad", [[-1], [True], ["IPC 302"], [1.5]])
def test_bad_labels(bad):
    with pytest.raises(ValueError):
        D.normalize([{"id": "x", "text": "a", "labels": bad}])
    assert D.normalize([{"id": "x", "text": "a", "labels": bad}], labels=False) == [("x", "a", None)]


@need_tok
def test_load_data_clean_dev(synth_root, data):
    tr, dv, te = data["train"], data["dev"], data["test"]
    raw = json.load(open(os.path.join(synth_root, "Train", "dev.json")))
    assert len(dv) == len(raw) and data["n_labels"] == 12
    keep = {dv.ids[i] for i in data["dev_clean"]}
    assert not keep & set(te.ids) and len(keep) == len(data["dev_clean"]) == len(raw) - 3
    Y = dv.Y(12)
    for i in (0, 4):
        assert sorted(np.where(Y[i])[0].tolist()) == raw[i]["labels"] == dv.labels(i).tolist()
    assert te.ids[-1] == "test_empty" and not te.labelled and tr.labelled


def test_freq_buckets():
    b = D.freq_buckets(np.array([100, 0, 50, 1, 10, 5]))
    assert b[0] == "frequent" and b[1] == "unseen" and b[3] == "rare"


def _rand(n=200, L=15, seed=0):
    rng = np.random.default_rng(seed)
    Y = (rng.random((n, L)) < np.linspace(0.4, 0.01, L)).astype(np.uint8)
    P = np.where(rng.random((n, L)) < 0.85, Y, 1 - Y).astype(np.uint8)
    return Y, P


def test_metrics_match_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    for s in range(10):
        Y, P = _rand(seed=s)
        m = metrics(Y, P, dim="all")
        assert abs(m["macro_f1"] - sk.f1_score(Y, P, average="macro", zero_division=0)) < 1e-12
        assert abs(m["micro_f1"] - sk.f1_score(Y, P, average="micro", zero_division=0)) < 1e-12
        assert abs(m["subset_acc"] - sk.accuracy_score(Y, P)) < 1e-12


@pytest.mark.skipif(not os.path.exists(EVAL), reason="official evaluator not unpacked")
def test_official_parity_and_label_dimension():
    pytest.importorskip("sklearn")
    from phase5.report import official_parity
    Y, P = _rand(seed=3)
    Y[:, -3:] = 0
    P[:, -3:] = 0
    ids = [f"d{i}" for i in range(len(Y))]
    vals, err = official_parity(ids, [np.where(r)[0].tolist() for r in Y], P)
    m = metrics(Y, P)
    assert err == "" and int(vals["Labels"]) == Y.shape[1] - 3
    assert abs(float(vals["Macro-F1"]) - m["macro_f1"]) < 1e-5 and abs(float(vals["Micro-F1"]) - m["micro_f1"]) < 1e-5


def test_thresholds():
    rng = np.random.default_rng(1)
    Y = (rng.random((400, 8)) < 0.15).astype(np.uint8)
    Pr = np.clip(0.1 + 0.4 * Y + rng.normal(0, 0.15, Y.shape), 0, 1)
    t, f = fit_global(Y, Pr)
    assert f >= metrics(Y, decide(Pr, 0.5))["macro_f1"] - 1e-12
    pl = fit_per_label(Y, Pr, t)
    assert pl.shape == (8,) and metrics(Y, decide(Pr, pl))["macro_f1"] >= f - 0.02
    cv = cross_fit(Y, Pr)
    assert set(cv) == {"fixed", "global", "global_atleast1", "per_label"} and 0 <= cv["global"]["macro_f1"] <= 1
    D1 = apply("global_atleast1", 0.99, Pr)
    assert (D1.sum(1) >= 1).all()


def test_validate_and_max_id():
    ids = ["a", "b"]
    Dm = np.array([[1, 0, 1, 1], [0, 0, 0, 0]], np.uint8)
    pj = to_prediction_json(ids, Dm, max_id=2)
    assert pj == [{"id": "a", "pred": [0, 2]}, {"id": "b", "pred": []}] and validate(pj, ids, 3)
    for bad in ([{"id": "a", "pred": [0, 0]}, pj[1]], [{"id": "a", "pred": [True]}, pj[1]], [{"id": "a", "pred": [3]}, pj[1]],
                [{"id": "a", "pred": 1}, pj[1]], pj[:1], [{"id": 1, "pred": []}, pj[1]]):
        with pytest.raises(ValueError):
            validate(bad, ids, 3)


def test_losses_and_kd():
    from phase5.train import kd_terms, ml_loss
    from phase4.experiments import DEFAULTS as _
    cfg = get_experiment("G_lorma")
    z = torch.randn(5, 7, requires_grad=True)
    Y = (torch.rand(5, 7) < 0.3).float()
    for loss in ("bce", "focal", "asl"):
        out = ml_loss(z, Y, dict(cfg, loss=loss))
        assert out.shape == (5,) and torch.isfinite(out).all()
        out.sum().backward()
    t = torch.randn(5, 7)
    assert kd_terms(t.clone(), t, 2.0).abs().max() < 1e-5 and (kd_terms(torch.zeros(5, 7), 3 * t, 2.0) > 0).all()


def test_experiments_resolve():
    p5_list = {"full", "partial", "qlora", "lorma", "adalora", "houlsby", "prefix", "prompt", "frozen"}
    for n in names():
        assert get_experiment(n)["method"] in p5_list, n
    six = [get_experiment(n) for n in ("A_lorma", "B_full", "C_houlsby", "D_partial", "G_qlora", "As_kd")]
    assert {c["backbone"] for c in six} >= {"A", "B", "C", "D"} and len({c["method"] for c in six[:5]}) == 5
    assert six[5]["kd"] and "lorma" in {c["method"] for c in six}


QUICK = dict(tiny=True, tokenizer=TOK, epochs=1, max_train_docs=16, batch=8, eval_docs=8, n_evals=1, probe_steps=1,
             budget_min=None, predict_test=False, save_soft=False, extract=None, save_weights=False, eval_chunks=2,
             seq_len=48, head_tokens=12, log_every=1000)


@need_tok
@pytest.mark.parametrize("exp", [n for n in names() if not n.startswith("smoke") and not get_experiment(n)["kd"]])
def test_every_experiment_runs(exp, synth_root, tmp_path):
    from phase5.train import run
    over = dict(QUICK, seq_batch=min(get_experiment(exp)["seq_batch"], 4))
    r = run(get_experiment(exp, over), synth_root, [os.path.join(ROOT, "results")], str(tmp_path))
    assert r["steps"] == 2 and 0 <= r["dev_cv"]["global"]["macro_f1"] <= 1


@need_tok
def test_end_to_end_tiny(synth_root, data, tmp_path):
    from phase5 import ensemble, hier
    from phase5.session import summarize
    from phase5.train import run
    out = str(tmp_path / "runs")
    roots = [os.path.join(ROOT, "results")]
    r = run(get_experiment("smoke_G", {"tiny": True, "tokenizer": TOK}), synth_root, roots, out)
    assert r["max_pred_id"] == 10 and r["dev_clean_docs"] == len(data["dev_clean"])
    hier.run_all(os.path.join(out, "smoke_G"), out, variants=["attn", "gru", "q2"], epochs=3)
    run(get_experiment("smoke_As_kd", {"tiny": True, "tokenizer": TOK}), synth_root, roots, out)
    run(get_experiment("smoke_G_hier", {"tiny": True, "tokenizer": TOK}), synth_root, roots, out)
    te = data["test"]
    for z in ("smoke_G/submission.zip", "smoke_G/submission_pl.zip", "smoke_G/submission_g1.zip",
              "smoke_G__h_gru/submission.zip", "smoke_As_kd/submission.zip", "smoke_G_hier/submission_pl.zip"):
        with zipfile.ZipFile(os.path.join(out, z)) as f:
            assert f.namelist() == ["prediction.json"]
            pj = json.loads(f.read("prediction.json"))
            assert validate(pj, te.ids, 11) and all(max(x["pred"], default=0) <= 10 for x in pj)
    e = ensemble.main(["--runs", "smoke_G", "smoke_G__h_gru", "smoke_As_kd", "--out", out, "--name", "ens"])
    assert "dev_cv" in e and os.path.exists(os.path.join(out, "ens", "submission_pl.zip"))
    assert {"smoke_G", "smoke_G__h_gru", "ens"} <= {x["run"] for x in summarize(out)}
