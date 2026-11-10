"""Tests for phase1.train.launch with a toy run file (test fixture values, not training advice).

  python -m pytest phase1/tests -q
"""
import json
import os
import shutil

import numpy as np
import pytest

from phase1.tests.test_pipeline import raw, run

V = 64
RUN_FILE = '''
import torch, torch.nn as nn
NAME = "toy"
KIND = "{kind}"
CFG = dict(batch_size=2, seq_len=16, src_len=8, tgt_len=8, grad_accum=2, total_steps={steps}, grad_clip=1.0,
           eval_every=3, eval_batches=1, eval_batch_size=2, token_milestones=[64], mix=None, seed={seed})
class Toy(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.emb, self.out = nn.Embedding(v, 8), nn.Linear(8, v)
    def forward(self, x, tgt=None):
        h = self.emb(x if tgt is None else tgt)
        if tgt is not None:
            h = h + self.emb(x).mean(1, keepdim=True)
        return self.out(h.cumsum(1) / torch.arange(1, h.shape[1] + 1).view(1, -1, 1))
def build_model(vocab_size):
    return Toy(vocab_size)
def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-2)
def lr_lambda(step):
    return 1.0
'''


@pytest.fixture()
def data(tmp_path):
    d = tmp_path / "p1data" / "shards"
    rng = np.random.default_rng(0)
    meta = {"vocab_size": V, "eos_id": 3, "sources": {}}
    for split in ("train", "val"):
        os.makedirs(d / split)
    for src in ("acts", "tribunals"):
        for split in ("train", "val"):
            rng.integers(4, V, 4000).astype(np.uint16).tofile(d / split / f"{src}.bin")
        meta["sources"][src] = {"train": {"tokens": 4000}, "val": {"tokens": 4000}}
    (d / "meta.json").write_text(json.dumps(meta))
    return str(tmp_path / "p1data")


def write_run(tmp_path, kind="causal", steps=4, seed=0):
    p = tmp_path / f"run_{kind}_{steps}.py"
    p.write_text(RUN_FILE.format(kind=kind, steps=steps, seed=seed))
    return str(p)


def launch(run_file, data, out, *extra, inputs="none"):
    return run("phase1.train.launch", "--run", run_file, "--data", data, "--out", out, "--inputs", inputs,
               "--log-every", "1", "--allow-any-size", *extra)


@pytest.mark.parametrize("kind", ["causal", "encdec"])
def test_pilot_reports_throughput(tmp_path, data, kind):
    out = launch(write_run(tmp_path, kind), data, str(tmp_path / "runs"), "--pilot", "5")
    rep = json.load(open(tmp_path / "runs" / "toy_pilot" / "pilot.json"))
    assert rep["steps"] == 5 and rep["tok_per_s"] > 0 and rep["tokens_per_step"] == 2 * 2 * 16
    assert "PILOT" in out and not os.path.exists(tmp_path / "runs" / "toy_pilot" / "ckpt")


def test_blank_values_refused(tmp_path, data):
    p = tmp_path / "blank.py"
    p.write_text(RUN_FILE.format(kind="causal", steps=None, seed=0))
    with pytest.raises(AssertionError, match=r"blank: \['total_steps'\]"):
        launch(str(p), data, str(tmp_path / "runs"))


def test_resume_from_attached_previous_session(tmp_path, data):
    session1 = tmp_path / "s1"
    launch(write_run(tmp_path, steps=4), data, str(session1 / "runs"))
    inputs = tmp_path / "kaggle_input"
    shutil.copytree(session1, inputs / "cs781-a-s1")
    out = launch(write_run(tmp_path, steps=8), data, str(tmp_path / "s2" / "runs"), inputs=str(inputs))
    assert "resume: copied" in out and "resumed from" in out
    log = [json.loads(l) for l in open(tmp_path / "s2" / "runs" / "toy" / "log.jsonl")]
    assert log[-1]["step"] == 8 and log[-1]["stopped"] == "total_steps"
    assert [r["step"] for r in log if "stopped" in r] == [4, 8]
    import torch
    launch(write_run(tmp_path, steps=8), data, str(tmp_path / "straight"))
    a = torch.load(tmp_path / "straight" / "toy" / "ckpt" / "step_8.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "s2" / "runs" / "toy" / "ckpt" / "step_8.pt", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_predict_test_split_writes_valid_zip(raw, tmp_path):
    import zipfile
    base, _, test = raw
    cfg = {"t": {"tokenizer": {"vocab_size": 400, "mb_per_source": 1}, "val_frac": 0.2,
                 "budgets": {"acts": "all"}, "val_tokens_per_source": 1e9, "dev_examples_per_source": 2}}
    (tmp_path / "data.json").write_text(json.dumps(cfg))
    data = str(tmp_path / "p1data")
    run("phase1.data.prep", "--config", str(tmp_path / "data.json"), "--section", "t", "--raw-dir", str(base),
        "--out", data, "--skip-download", "--workers", "1", "--eval-mb", "1")
    run_file = write_run(tmp_path, steps=2)
    launch(run_file, data, str(tmp_path / "runs"))
    ckpt = str(tmp_path / "runs" / "toy" / "ckpt" / "step_2.pt")
    (tmp_path / "dec.json").write_text(json.dumps({"temperature": 1.0, "suppress_eos": True,
                                                    "max_new_tokens": 40, "max_prompt_tokens": 32}))
    out = run("phase1.eval.predict", "--run", run_file, "--ckpt", ckpt, "--data", data,
              "--decode", str(tmp_path / "dec.json"), "--split", "test", "--out", str(tmp_path / "pred"))
    assert "no-cache adapter" in out
    z = zipfile.ZipFile(tmp_path / "pred" / "submission.zip")
    pred = json.loads(z.read("prediction.json"))
    assert set(pred) == set(test) and all(p["text"].strip() for p in pred.values())
    rec = json.load(open(tmp_path / "pred" / "predict_record.json"))
    assert rec["step"] == 2 and rec["decode"]["suppress_eos"] is True and rec["empty_outputs"] == 0
