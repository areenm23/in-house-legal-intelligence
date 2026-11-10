"""Phase-6 tests: loading / dev split / cache, input selection and dancer alignment, batches (targets and masks),
the generative wrapper on every family (training logits = the model's own forward, cached generation = no-cache
greedy), PEFT methods on full LMs (adapters on B's decoder), official ROUGE parity, validation, and end-to-end tiny
runs (train -> KD -> re-decode) with valid zips.

    python -m pytest phase6/tests -q
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

from ft.peft import apply_method
from phase6 import data as D
from phase6.experiments import get_experiment, names
from phase6.genlm import EOS, SEP, generate, group_pairs, load_genlm, make_batch
from phase6.score import rouge, validate

TOK = os.path.join(ROOT, "results", "results", "p1data_small", "tokenizer", "a1bpe_32755.json")
EVAL = os.path.join(ROOT, "course_files", "eval", "Evaluation Script", "evaluate_phase6.py")
need_tok = pytest.mark.skipif(not os.path.exists(TOK), reason="Phase-1 tokenizer not present")
CPU = torch.device("cpu")
W = ("the appellant filed an appeal against the order of the high court which dismissed the writ petition on the "
     "ground of delay the court held that section of the act applies").split()


def _doc(rng, n):
    return ". ".join(" ".join(rng.choice(W) for _ in range(rng.integers(6, 18))).capitalize() for _ in range(n)) + "."


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory):
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("p6") / "Phase_6_Summ"
    (root / "Train").mkdir(parents=True)
    (root / "Test").mkdir()
    tr = [{"id": f"d{i}", "document": _doc(rng, int(rng.integers(3, 40))), "summary": _doc(rng, int(rng.integers(1, 5)))}
          for i in range(60)]
    te = [{"id": f"t{i}", "document": _doc(rng, int(rng.integers(3, 30)))} for i in range(7)] + [{"id": "t_empty", "document": ""}]
    (root / "Train" / "train.json").write_text(json.dumps(tr))
    (root / "Test" / "test.json").write_text(json.dumps(te))
    return str(root)


@pytest.fixture(scope="module")
def data(synth_root, tmp_path_factory):
    return D.load_data(synth_root, TOK, str(tmp_path_factory.mktemp("c")), workers=1)


@pytest.mark.parametrize("raw", [
    [{"id": "a", "document": "x y.", "summary": "s."}],
    {"a": {"text": ["x y."], "headnote": "s."}},
    {"data": [{"doc_id": "a", "judgment": "x y.", "reference": ["s."]}]},
])
def test_normalize(raw):
    assert D.normalize(raw) == [("a", "x y.", "s.")]


def test_split_sentences():
    assert D.split_sentences("The court held. Appeal allowed; costs paid.\nNext line; Done") == \
        ["The court held.", "Appeal allowed; costs paid.", "Next line;", "Done"]


@need_tok
def test_load_data(data):
    tr, dv, te = data["train"], data["dev"], data["test"]
    assert len(tr) + len(dv) == 60 and len(dv) >= 2 and not set(tr.ids) & set(dv.ids)
    assert len(te) == 8 and te.summ is None and dv.ref is not None and len(dv.summ) == len(dv)
    from phase1.data.a1_bpe import load_tokenizer
    tok = load_tokenizer(TOK)
    assert dv.summ[0].tolist() == tok.encode(dv.ref[0]).ids


@need_tok
def test_cached_load_shares_one_summary_array(synth_root, data, tmp_path):
    cache = str(tmp_path / "c")
    D.load_data(synth_root, TOK, cache, workers=1)
    again = D.load_data(synth_root, TOK, cache, workers=1)
    for s in ("train", "dev"):
        assert [x.tolist() for x in again[s].summ] == [x.tolist() for x in data[s].summ]
        assert len({id(x.base) for x in again[s].summ}) == 1


def test_select_input():
    rng = np.random.default_rng(0)
    sents = [rng.integers(4, 300, int(rng.integers(5, 30))).astype(np.uint16) for _ in range(50)]
    full = np.concatenate(sents)
    for how in ("head", "headtail", "central", "oracle"):
        x = D.select_input(sents, 200, how, head_tokens=50, summary=sents[30])
        assert len(x) <= 200 and x.dtype == np.int64
    assert D.select_input(sents, 200, "head").tolist() == full[:200].tolist()
    ht = D.select_input(sents, 200, "headtail", head_tokens=50)
    assert ht[:50].tolist() == full[:50].tolist() and ht[50:].tolist() == full[-150:].tolist()
    assert D.select_input(sents[:2], 10 ** 6, "central").tolist() == np.concatenate(sents[:2]).tolist()
    oracle = D.select_input(sents, 200, "oracle", summary=sents[30])
    assert sents[30].tolist() == oracle[:len(sents[30])].tolist() or any(
        oracle[i:i + len(sents[30])].tolist() == sents[30].tolist() for i in range(len(oracle)))
    ch = D.chunk_sentences(sents, 120)
    assert ch[0][0] == 0 and ch[-1][1] == 50 and all(a[1] == b[0] for a, b in zip(ch, ch[1:]))
    pairs = D.dancer_pairs(sents, [sents[2], sents[45]], 120)
    assert sum(len(t) for _, t in pairs) == len(sents[2]) + len(sents[45])


def test_make_batch_decoder_and_encdec():
    pairs = [(np.array([5, 6, 7]), np.array([8, 9])), (np.array([5, 6]), np.array([8, 9, 10, 11]))]
    b = make_batch("decoder", pairs, CPU, max_target=3)
    assert b["ids"][0, :6].tolist() == [5, 6, 7, SEP, 8, 9] and b["targets"][0, 3:6].tolist() == [8, 9, EOS]
    assert b["mask"][1].sum() == 3 and b["targets"][1, 2:5].tolist() == [8, 9, 10]
    e = make_batch("encdec", [(np.array([5, 6]), np.array([8, 9]))] * 2, CPU, 10)
    assert e["tgt_in"][0, :3].tolist() == [2, 8, 9] and e["targets"][0, :3].tolist() == [8, 9, EOS]
    with pytest.raises(ValueError):
        make_batch("encdec", pairs, CPU, 10)
    groups = group_pairs("encdec", [(np.zeros(n), np.zeros(3)) for n in (4, 4, 7, 4)], 8, 10 ** 6)
    assert sorted(map(sorted, groups)) == [[0, 1, 3], [2]]


@pytest.mark.parametrize("key", ["A", "B", "C", "D", "As"])
def test_target_logits_match_forward_and_generation(key):
    torch.manual_seed(0)
    g = load_genlm(key, tiny=True)
    apply_method(g, "lora", r=4, targets="attn")
    for m in g.modules():
        if hasattr(m, "B") and isinstance(getattr(m, "B"), torch.nn.Parameter):
            torch.nn.init.normal_(m.B, std=0.05)
    g.eval()
    x, y = np.array([7, 8, 9, 10, 11]), np.array([20, 21, 22])
    b = make_batch(g.family, [(x, y)], CPU, 10)
    with torch.no_grad():
        lg, tgt = g.target_logits(b)
        if g.family == "encdec":
            ref = g.lm(b["src"], b["tgt_in"])[0]
        else:
            ref = g.lm(b["ids"])[0][len(x):len(x) + len(y) + 1]
    assert torch.allclose(lg, ref, atol=1e-4) and tgt.tolist() == [20, 21, 22, EOS]
    out = generate(g, [x, x], [(5, 5), (5, 5)], CPU, batch_size=2, amp=False, log=None)
    assert out[0] == out[1] and len(out[0]) == 5
    from phase1.eval.generate import DecodeConfig
    plain = generate(g, [x], [(4, 4)], CPU, DecodeConfig(temperature=0.0), amp=False, log=None)[0]
    seq = list(x) if g.family == "encdec" else list(x) + [SEP]
    with torch.no_grad():
        for k, tok_ in enumerate(plain):
            if g.family == "encdec":
                lgt = g.lm(torch.tensor([list(x)]), torch.tensor([[2] + plain[:k]]))[0, -1].clone()
            else:
                lgt = g.lm(torch.tensor([seq]))[0, -1].clone()
            lgt[[EOS] + list(range(32755, 32768)) + [0, 2, 1]] = -1e4
            assert int(lgt.argmax()) == tok_
            seq.append(tok_)


@pytest.mark.parametrize("key,method", [(k, m) for k in ("A", "As", "B", "C", "D") for m in ("lora", "lorma", "adalora")]
                         + [("A", "houlsby"), ("B", "houlsby"), ("B", "pfeiffer"), ("As", "full")])
def test_build_model_puts_everything_on_the_device(key, method):
    from phase6.train import build_model
    args = {"r": 4, "targets": "attn"} if method in ("lora", "lorma") else \
        {"r": 4, "target_r": 2, "targets": "attn"} if method == "adalora" else {"bottleneck": 8} if method != "full" else {}
    g, _ = build_model({"backbone": key, "method": method, "method_args": args, "tiny": True}, [], torch.device("meta"))
    off = [n for n, t in list(g.named_parameters()) + list(g.named_buffers()) if t.device.type != "meta"]
    assert not off, off[:5]


def test_c_experiments_fit_retnet_memory():
    for n in names():
        cfg = get_experiment(n)
        if cfg["backbone"] == "C" and not n.startswith("smoke"):
            assert cfg["budget"] + cfg["max_target"] <= 1536 and cfg["batch_tokens"] <= 3100, n


def _fake_run(root, nb, run, backbone, method, bert, kd=None, ids=("1", "2", "3"), scored=True, summary="s"):
    d = root / nb / "runs_p6" / run
    d.mkdir(parents=True)
    dev = {"bertscore_f1": bert, "rouge1": 0.4, "rougeL": 0.2, "words_mean": 300.0} if scored else {"error": "x"}
    (d / "result.json").write_text(json.dumps({"backbone": backbone, "method": method, "kd": kd and {"teacher": kd},
                                               "dev": dev}))
    (d / "dev_pred.json").write_text(json.dumps([{}] * 100))
    with zipfile.ZipFile(d / "submission.zip", "w") as z:
        z.writestr("prediction.json", json.dumps([{"id": i, "summary": summary} for i in ids]))


def test_upload_plan(tmp_path):
    from phase6 import upload
    R = tmp_path / "res"
    _fake_run(R, "s1", "A_lorma", "A", "lorma", 0.80, scored=False)
    _fake_run(R, "r1", "A_lorma", "A", "lorma", 0.84)
    _fake_run(R, "r1", "A_lorma__dec_s13", "A", "lorma", 0.85)
    _fake_run(R, "r1", "B_lora", "B", "lora", 0.83)
    _fake_run(R, "s1", "B_houlsby", "B", "houlsby", 0.82)
    _fake_run(R, "r1", "As_kd", "As", "full", 0.81, kd="A_lorma")
    _fake_run(R, "s2", "C_qlora", "C", "qlora", 0.79)
    _fake_run(R, "s4", "A_full", "A", "full", 0.86)
    _fake_run(R, "s4", "A_qlora", "A", "qlora", 0.70, ids=("1", "2"))
    _fake_run(R, "s4", "A_pfeiffer", "A", "pfeiffer", 0.75, summary="")
    order, rejected, missing = upload.plan(upload.collect(str(R)), max_n=40)
    assert [(r["slot"], r["run"]) for r in order] == [
        ("LoRMA", "A_lorma__dec_s13"), ("LoRA", "B_lora"), ("QLoRA", "C_qlora"), ("Houlsby", "B_houlsby"),
        ("KD", "As_kd"), ("", "A_full"), ("", "A_lorma")]
    assert missing == ["AdaLoRA"] and {r for r, _ in rejected} == {"A_qlora", "A_pfeiffer"}
    out = tmp_path / "up"
    upload.write(order[:3], str(out))
    assert sorted(os.listdir(out)) == ["01_A_lorma__dec_s13.zip", "02_B_lora.zip", "03_C_qlora.zip", "manifest.json",
                                       "manifest.md"]
    assert len(upload.plan(upload.collect(str(R)), max_n=2)[0]) == 2


def test_generate_splits_batch_on_oom():
    torch.manual_seed(0)
    g = load_genlm("A", tiny=True)
    apply_method(g, "lora", r=4, targets="attn")
    x = np.array([7, 8, 9, 10, 11])
    inputs, lengths = [x, x, x, x], [(4, 4), (5, 5), (6, 6), (7, 7)]
    want = generate(g, inputs, lengths, CPU, batch_size=4, amp=False, log=None)
    real_api = g.api

    class Flaky:
        def __init__(self):
            self.api = real_api()

        def prefill(self, ids):
            if len(ids) > 1:
                raise torch.cuda.OutOfMemoryError("simulated")
            return self.api.prefill(ids)

        def step(self, *a):
            return self.api.step(*a)

    g.api = Flaky
    assert generate(g, inputs, lengths, CPU, batch_size=4, amp=False, log=None) == want
    g.api = lambda: type("Always", (), {"prefill": lambda self, ids: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("x"))})()
    with pytest.raises(torch.cuda.OutOfMemoryError):
        generate(g, [x], [(3, 3)], CPU, amp=False, log=None)


def test_adapters_on_full_lms():
    g = load_genlm("B", tiny=True)
    info = apply_method(g, "houlsby", bottleneck=8)
    assert info["adapters"] == 2 * len(g.layers())
    g2 = load_genlm("A", tiny=True)
    apply_method(g2, "pfeiffer", bottleneck=8)
    out = generate(g2, [np.array([5, 6, 7])], [(3, 3)], CPU, amp=False, log=None)
    assert len(out[0]) == 3
    from phase6.train import build_model
    with pytest.raises(ValueError):
        build_model(dict(get_experiment("smoke_C"), tiny=True, method="houlsby", method_args={}), [], CPU)


def test_rouge_matches_official(tmp_path):
    pytest.importorskip("rouge_score")
    if not os.path.exists(EVAL):
        pytest.skip("official evaluator not unpacked")
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "bert_score.py").write_text("import torch\ndef score(p, r, **kw):\n    t = torch.ones(len(p))\n    return t, t, t\n")
    preds = ["The appeal is allowed and the order is set aside.", "Writ petition dismissed on delay.", "x"]
    refs = ["The appeal was allowed; the impugned order set aside.", "The petition is dismissed.", "y z"]
    json.dump({f"d{i}": r for i, r in enumerate(refs)}, open(tmp_path / "t.json", "w"))
    json.dump([{"id": f"d{i}", "summary": p} for i, p in enumerate(preds)], open(tmp_path / "p.json", "w"))
    env = dict(os.environ, PYTHONPATH=str(stub))
    r = subprocess.run([sys.executable, EVAL, "--truth", str(tmp_path / "t.json"), "--prediction", str(tmp_path / "p.json")],
                       capture_output=True, text=True, env=env)
    vals = dict(ln.split(": ") for ln in r.stdout.strip().splitlines())
    r1, rl = rouge(preds, refs)
    assert abs(float(vals["ROUGE-1"]) - r1) < 1e-6 and abs(float(vals["ROUGE-L"]) - rl) < 1e-6


def test_validate():
    ids = ["a", "b"]
    assert validate([{"id": "a", "summary": "x"}, {"id": "b", "summary": "y"}], ids)
    for bad in ([{"id": "a", "summary": " "}, {"id": "b", "summary": "y"}], [{"id": "a", "summary": "x"}],
                [{"id": "a", "summary": "x"}, {"id": "a", "summary": "y"}], [{"id": 1, "summary": "x"}, {"id": "b", "summary": "y"}]):
        with pytest.raises(ValueError):
            validate(bad, ids)


def test_experiments_resolve():
    p6 = {"houlsby", "pfeiffer", "prefix", "prompt", "lora", "qlora", "adalora", "lorma", "full"}
    for n in names():
        c = get_experiment(n)
        assert c["method"] in p6 and c["backbone"] in ("A", "As", "B", "C", "D")
        if c["method"] in ("houlsby", "pfeiffer"):
            assert c["backbone"] in ("A", "As", "B"), n
    six = [get_experiment(n) for n in ("A_lorma", "B_lora", "C_qlora", "D_adalora", "B_houlsby", "As_kd")]
    assert {c["backbone"] for c in six} >= {"A", "B", "C", "D"} and len({c["method"] for c in six[:5]}) == 5
    assert six[5]["kd"]


@pytest.mark.parametrize("combo", sorted({(get_experiment(n)["backbone"], get_experiment(n)["method"],
                                            json.dumps(get_experiment(n)["method_args"], sort_keys=True))
                                           for n in names() if not n.startswith("smoke")}))
def test_every_method_backbone_trains(combo):
    from phase6.train import build_model
    key, method, args = combo
    args = json.loads(args)
    if "bottleneck" in args:
        args["bottleneck"] = 8
    g, info = build_model(dict(backbone=key, method=method, method_args=args, tiny=True), [], CPU)
    b = make_batch(g.family, [(np.array([5, 6, 7, 8]), np.array([9, 10]))], CPU, 10)
    lg, y = g.target_logits(b)
    torch.nn.functional.cross_entropy(lg, y).backward()
    assert info["trainable"] > 0 and any(p.grad is not None for p in g.parameters() if p.requires_grad)


@need_tok
def test_end_to_end_tiny(synth_root, data, tmp_path):
    pytest.importorskip("rouge_score")
    from phase6.session import summarize
    from phase6.train import redecode, run
    out = str(tmp_path / "runs")
    roots = [os.path.join(ROOT, "results")]
    ov = {"tiny": True, "tokenizer": TOK}
    r = run(get_experiment("smoke_A", ov), synth_root, roots, out)
    assert r["dev"]["rouge1"] >= 0 and os.path.exists(os.path.join(out, "smoke_A", "adapter.pt"))
    run(get_experiment("smoke_As_kd", ov), synth_root, roots, out)
    run(get_experiment("smoke_B_houlsby", ov), synth_root, roots, out)
    run(get_experiment("smoke_A_dancer", ov), synth_root, roots, out)
    redecode(os.path.join(out, "smoke_A"), {"len_ratio": 0.3}, "smoke_A__dec_x", synth_root, roots, out)
    te = data["test"]
    for z in ("smoke_A", "smoke_As_kd", "smoke_B_houlsby", "smoke_A_dancer", "smoke_A__dec_x"):
        with zipfile.ZipFile(os.path.join(out, z, "submission.zip")) as f:
            assert f.namelist() == ["prediction.json"]
            assert validate(json.loads(f.read("prediction.json")), te.ids)
    assert {"smoke_A", "smoke_A__dec_x"} <= {x["run"] for x in summarize(out)}
