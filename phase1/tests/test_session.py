"""End-to-end check of what one Kaggle session does, on synthetic data and a tiny model:
prep (no profiling pass) -> phase1.job (time-boxed training with total_steps='auto', then
test submission per decoding config). Dev scoring needs bert-score; it may fail locally and
the job must carry on regardless."""
import json
import os
import time
import zipfile

from phase1.tests.test_pipeline import ROOT, raw, run

RUN_FILE = '''
from phase1.models.decoder import DecoderConfig, DecoderLM, DecoderAPI
from phase1.models.optim import make_adamw, warmup_cosine
NAME = "tiny_A"
KIND = "causal"
CFG = dict(batch_size=4, seq_len=64, grad_accum=2, total_steps="auto", grad_clip=1.0, eval_every=5,
           eval_batches=1, eval_batch_size=2, token_milestones=[], mix=None, seed=0, max_steps=12)
def build_model(vocab_size):
    return DecoderLM(DecoderConfig(vocab_size=vocab_size, d_model=32, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=48))
def make_optimizer(model):
    return make_adamw(model, lr=1e-3, weight_decay=0.1)
def make_lr_lambda(total_steps):
    return warmup_cosine(total_steps, warmup=2, final_frac=0.1)
def decoder_api(model):
    return DecoderAPI(model)
'''


def test_prep_then_job(raw, tmp_path):
    base, _, test = raw
    cfg = {"t": {"tokenizer": {"vocab_size": 400, "mb_per_source": 1}, "profile": False, "val_frac": 0.1,
                 "budgets": {"acts": "all", "tribunals": 20000}, "val_tokens_per_source": 2000,
                 "dev_examples_per_source": 2}}
    (tmp_path / "data.json").write_text(json.dumps(cfg))
    data = str(tmp_path / "p1data")
    out = run("phase1.data.prep", "--config", str(tmp_path / "data.json"), "--section", "t",
              "--raw-dir", str(base), "--out", data, "--skip-download", "--workers", "1")
    assert "keep" in out and os.path.exists(os.path.join(data, "shards", "meta.json"))
    meta = json.load(open(os.path.join(data, "shards", "meta.json")))
    assert meta["sources"]["tribunals"]["train"]["tokens"] >= 20000 * 0.8

    rf = tmp_path / "run_tiny.py"
    rf.write_text(RUN_FILE)
    dec = tmp_path / "greedy.json"
    dec.write_text(json.dumps({"temperature": 0, "no_repeat_ngram": 3, "suppress_eos": True,
                               "max_new_tokens": 200, "max_prompt_tokens": 40, "batch_size": 4}))
    now = time.time()
    out = run("phase1.job", "--run", str(rf), "--data", data, "--out", str(tmp_path / "runs"),
              "--inputs", "none", "--train-until", str(now + 600), "--hard-stop", str(now + 1500),
              "--decodes", str(dec), "--preds", str(tmp_path / "preds"), "--subs", str(tmp_path / "subs"),
              "--allow-any-size", "--log-every", "1")
    plan = json.load(open(tmp_path / "runs" / "tiny_A" / "plan.json"))
    assert plan["total_steps"] == 12
    summ = json.load(open(tmp_path / "subs" / "tiny_A__summary.json"))
    assert summ["train_exit"] == 0
    zips = [r["zip"] for r in summ["results"] if r.get("zip")]
    assert len(zips) == 1 and "step12" in zips[0]
    with zipfile.ZipFile(zips[0]) as z:
        pred = json.loads(z.read("prediction.json"))
    assert set(pred) == set(test) and all(len(v["text"].split()) > 0 for v in pred.values())


MLM_RUN = '''
from phase1.models.mlm import EncoderMLM, MLMConfig, MLMFillAPI, mask_id_for, mlm_batch, mlm_loss
from phase1.models.optim import make_adamw, warmup_cosine
NAME = "tiny_G"
KIND = "mlm"
CFG = dict(batch_size=4, seq_len=64, grad_accum=2, total_steps="auto", grad_clip=1.0, eval_every=5,
           eval_batches=1, eval_batch_size=2, token_milestones=[], mix=None, seed=0, max_steps=8)
DECODES = [{decode!r}]
def build_model(vocab_size):
    return EncoderMLM(MLMConfig(vocab_size=vocab_size, d_model=32, n_layers=1, n_heads=2, n_kv_heads=2, d_ff=48))
def train_batch(shards, index, cfg=CFG):
    return mlm_batch(shards, cfg["batch_size"], cfg["seq_len"], cfg["seed"], index, cfg["mix"],
                     mask_id_for(shards.meta["vocab_size"]), 0.3, 0.25, (8, 32))
def eval_batch(shards, seed, index, mix, cfg=CFG):
    return mlm_batch(shards, cfg["eval_batch_size"], cfg["seq_len"], seed, index, mix,
                     mask_id_for(shards.meta["vocab_size"]), 0.3, 0.25, (8, 32))
loss_fn = mlm_loss
def make_optimizer(model):
    return make_adamw(model, lr=1e-3, weight_decay=0.1)
def make_lr_lambda(total_steps):
    return warmup_cosine(total_steps, warmup=2, final_frac=0.1)
def decoder_api(model):
    return MLMFillAPI(model, n_masks=40, block=4)
'''

CONT_RUN = RUN_FILE.replace('NAME = "tiny_A"', 'NAME = "tiny_A_cont"\nINIT_FROM = "tiny_A"').replace("seed=0", "seed=7")


def test_mlm_job_and_continued_training(raw, tmp_path):
    base, _, test = raw
    cfg = {"t": {"tokenizer": {"vocab_size": 400, "mb_per_source": 1}, "profile": False, "val_frac": 0.1,
                 "budgets": {"acts": "all"}, "val_tokens_per_source": 2000, "dev_examples_per_source": 2}}
    (tmp_path / "data.json").write_text(json.dumps(cfg))
    data = str(tmp_path / "p1data")
    run("phase1.data.prep", "--config", str(tmp_path / "data.json"), "--section", "t", "--raw-dir", str(base),
        "--out", data, "--skip-download", "--workers", "1")
    data2 = str(tmp_path / "p1data2")
    out = run("phase1.data.prep", "--config", str(tmp_path / "data.json"), "--section", "t", "--raw-dir", str(base),
              "--out", data2, "--skip-download", "--workers", "1", "--tokenizer-from", os.path.join(data, "tokenizer"))
    assert "not refitted" in out
    for split in ("train", "val"):
        assert open(os.path.join(data, "shards", split, "acts.bin"), "rb").read() == \
            open(os.path.join(data2, "shards", split, "acts.bin"), "rb").read()

    dec = tmp_path / "fill.json"
    dec.write_text(json.dumps({"temperature": 0, "no_repeat_ngram": 3, "suppress_eos": True,
                               "max_new_tokens": 40, "max_prompt_tokens": 40, "batch_size": 4}))
    (tmp_path / "run_g.py").write_text(MLM_RUN.format(decode=str(dec)))
    now = time.time()
    common = ["--train-until", str(now + 600), "--hard-stop", str(now + 1500), "--preds", str(tmp_path / "preds"),
              "--subs", str(tmp_path / "subs"), "--allow-any-size", "--log-every", "1"]
    run("phase1.job", "--run", str(tmp_path / "run_g.py"), "--data", data, "--out", str(tmp_path / "runs"),
        "--inputs", "none", "--decodes", "unused.json", *common)
    summ = json.load(open(tmp_path / "subs" / "tiny_G__summary.json"))
    assert summ["train_exit"] == 0 and [r["decode"] for r in summ["results"]][0] == "fill"
    zips = [r["zip"] for r in summ["results"] if r.get("zip")]
    with zipfile.ZipFile(zips[0]) as z:
        pred = json.loads(z.read("prediction.json"))
    assert set(pred) == set(test) and all(v["text"].split() for v in pred.values())

    (tmp_path / "run_a.py").write_text(RUN_FILE)
    (tmp_path / "run_a_cont.py").write_text(CONT_RUN)
    run("phase1.train.launch", "--run", str(tmp_path / "run_a.py"), "--data", data, "--out",
        str(tmp_path / "prev" / "runs"), "--inputs", "none", "--deadline", str(now + 600), "--allow-any-size")
    gdec = tmp_path / "greedy.json"
    gdec.write_text(json.dumps({"temperature": 0, "no_repeat_ngram": 3, "suppress_eos": True,
                                "max_new_tokens": 60, "max_prompt_tokens": 40, "batch_size": 4}))
    now = time.time()
    out = run("phase1.job", "--run", str(tmp_path / "run_a_cont.py"), "--data", data, "--out", str(tmp_path / "runs2"),
              "--inputs", str(tmp_path / "prev"), "--train-until", str(now + 600), "--hard-stop", str(now + 1500),
              "--decodes", str(gdec), "--preds", str(tmp_path / "preds2"), "--subs", str(tmp_path / "subs2"),
              "--allow-any-size")
    assert "init: weights from" in out and "step_12.pt" in out
    info = json.load(open(tmp_path / "runs2" / "tiny_A_cont" / "init_from.json"))
    assert info["step"] == 12
    summ = json.load(open(tmp_path / "subs2" / "tiny_A_cont__summary.json"))
    assert [r.get("zip", "").endswith("tiny_A_regen__greedy__step12.zip") for r in summ["init_results"]][0]


def test_decoding_sweep_with_mbr(raw, tmp_path, monkeypatch):
    """Sweep = dev runs for every decode (incl. MBR), ranking, test zips for the top-k (fake scorer)."""
    base, _, test = raw
    cfg = {"t": {"tokenizer": {"vocab_size": 400, "mb_per_source": 1}, "profile": False, "val_frac": 0.1,
                 "budgets": {"acts": "all"}, "val_tokens_per_source": 2000, "dev_examples_per_source": 3}}
    (tmp_path / "data.json").write_text(json.dumps(cfg))
    data = str(tmp_path / "p1data")
    run("phase1.data.prep", "--config", str(tmp_path / "data.json"), "--section", "t", "--raw-dir", str(base),
        "--out", data, "--skip-download", "--workers", "1")
    (tmp_path / "run_a.py").write_text(RUN_FILE)
    now = time.time()
    run("phase1.train.launch", "--run", str(tmp_path / "run_a.py"), "--data", data, "--out", str(tmp_path / "runs"),
        "--inputs", "none", "--deadline", str(now + 600), "--allow-any-size")
    decs = []
    for name, d in {"g": {"temperature": 0}, "s": {"temperature": 0.9},
                    "m": {"temperature": 0.9, "mbr_samples": 3, "mbr_utility": "rougeL"}}.items():
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({**d, "suppress_eos": True, "max_new_tokens": 40, "max_prompt_tokens": 40,
                                 "batch_size": 6}))
        decs.append(str(p))
    monkeypatch.setenv("PHASE1_FAKE_SCORER", "1")
    out = run("phase1.sweep", "--run", str(tmp_path / "run_a.py"),
              "--ckpt", str(tmp_path / "runs" / "tiny_A" / "ckpt" / "step_12.pt"), "--data", data,
              "--decodes", *decs, "--top-k", "2", "--hard-stop", str(time.time() + 1200),
              "--preds", str(tmp_path / "preds"), "--subs", str(tmp_path / "subs"))
    assert "mbr_mean_agreement" in out
    summ = json.load(open(tmp_path / "subs" / "tiny_A__sweep.json"))
    assert len(summ["dev"]) == 3 and all("scores" in r for r in summ["dev"])
    assert len(summ["test"]) == 2 and summ["test"][0]["decode"] == summ["ranking"][0][0]
    with zipfile.ZipFile(summ["test"][0]["zip"]) as z:
        pred = json.loads(z.read("prediction.json"))
    assert set(pred) == set(test)
