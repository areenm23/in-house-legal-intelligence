"""Phase-2 / ft library tests: offsets, oracle, scorer parity, PEFT identities, CRF, end-to-end tiny run.

    python -m pytest phase2/tests -q
"""
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from ft import peft as P
from ft.backbones import REGISTRY, load_backbone
from ft.heads import CRF, constrained_decode, viterbi
from phase2 import data as D
from phase2 import synth
from phase2.score import errors, score, to_prediction_json, validate

TOK = os.path.join(ROOT, "results", "results", "p1data_small", "tokenizer", "a1bpe_32755.json")
EVAL = os.path.join(ROOT, "course_files", "eval", "Evaluation Script", "evaluate_phase2.py")
need_tok = pytest.mark.skipif(not os.path.exists(TOK), reason="Phase-1 tokenizer not present")


@pytest.fixture(scope="module")
def tok():
    from phase1.data.a1_bpe import load_tokenizer
    return load_tokenizer(TOK)


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory):
    return synth.write(str(tmp_path_factory.mktemp("p2")), docs=10, test_docs=4, seed=1)


TEXTS = ["IN THE SUPREME COURT OF INDIA\n\nRam Singh,  (Appellant)  v.  Śrī Gopal ...Respondent",
         "Section 302/34 I.P.C. read with Art. 21; dated 22.03.2011 — “quoted” text.",
         "M/s. Bharat Steel Pvt. Ltd., Adv. ½ naïve café x"]


@need_tok
@pytest.mark.parametrize("split", ["none", "edge", "punct"])
def test_offsets_cover_text(tok, split):
    ot = D.OffsetTokenizer(tok, split)
    for t in TEXTS:
        ids, s, e = ot.encode(t)
        assert len(ids) == len(s) == len(e)
        assert (e >= s).all() and (s[1:] >= s[:-1]).all()
        covered = np.zeros(len(t), dtype=int)
        for a, b in zip(s, e):
            covered[a:b] += 1
        for i, c in enumerate(t):
            assert covered[i] == (0 if c.isspace() else 1), (split, i, c)
        for i, a, b in zip(ids, s, e):
            raw = ot.enc._bytes.get(ot.enc.itos[i], b"")
            raw = raw[:-1] if raw.endswith(b" ") else raw
            if b > a:
                assert t[a:b].encode("utf-8").startswith(raw) or raw.startswith(t[a:b].encode("utf-8")[:1]) or \
                    raw in t[a:b].encode("utf-8")


@need_tok
def test_split_none_matches_phase1(tok):
    ot = D.OffsetTokenizer(tok, "none")
    for t in TEXTS:
        assert list(ot.encode(t)[0]) == tok.encode(t).ids


def test_word_pieces():
    assert D.word_pieces("Singh,", "edge") == [(0, 5), (5, 6)]
    assert D.word_pieces("(Appellant).", "edge") == [(0, 1), (1, 10), (10, 11), (11, 12)]
    assert D.word_pieces("...", "edge") == [(0, 3)]
    assert D.word_pieces("I.P.C.", "punct") == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    assert D.word_pieces("abc", "edge") == [(0, 3)]


@need_tok
def test_oracle_synthetic(tok, synth_root):
    d = D.load_split(synth_root)
    ts = D.TagSet(d["labels"])
    ot = D.OffsetTokenizer(tok, "edge")
    pred = {}
    for k, t in d["texts"].items():
        ids, s, e = ot.encode(t)
        tags, st = D.tag_tokens(s, e, d["gold"][k], ts)
        assert st["unrepresentable"] == 0 and st["conflicts"] == 0
        pred[k] = D.spans_from_tags(tags, s, e, ts)
    truth = {k: {(a, b, l) for a, b, l, _ in v} for k, v in d["gold"].items()}
    assert score(truth, pred, d["labels"])["macro_f1"] == 1.0


def test_windows_cover():
    for n, size, stride in [(10, 4, 2), (5, 8, 4), (100, 32, 32), (33, 32, 16)]:
        w = D.windows(n, size, stride)
        cov = np.zeros(n)
        for a, b in w:
            assert b - a == min(size, n)
            cov[a:b] = 1
        assert cov.all()
    for off in range(7):
        w = D.windows(50, 8, 8, offset=off)
        assert all(b - a == 8 for a, b in w) and min(a for a, _ in w) == 0 and max(b for _, b in w) == 50


def test_dev_split_deterministic():
    ids = [str(i) for i in range(40)]
    a = D.dev_split(ids, 0.15)
    assert a == D.dev_split(list(reversed(ids)), 0.15)
    assert len(a[1]) == 6 and not set(a[0]) & set(a[1])


def test_score_matches_official(tmp_path, synth_root):
    if not os.path.exists(EVAL):
        pytest.skip("course evaluator not present")
    d = D.load_split(synth_root)
    truth = {k: {(a, b, l) for a, b, l, _ in v} for k, v in d["gold"].items()}
    rng = np.random.default_rng(0)
    pred = {}
    for k, v in truth.items():
        out = set()
        for a, b, l in v:
            r = rng.random()
            if r < 0.6:
                out.add((a, b, l))
            elif r < 0.75:
                out.add((a, b + 1, l))
            elif r < 0.85:
                out.add((a, b, d["labels"][(d["labels"].index(l) + 1) % 12]))
        out.add((0, 3, "DATE"))
        pred[k] = out
    tj, pj = tmp_path / "t.json", tmp_path / "p.json"
    json.dump(to_prediction_json(truth), open(tj, "w"))
    json.dump(to_prediction_json(pred), open(pj, "w"))
    validate(json.load(open(pj)), list(truth), d["labels"])
    lab = os.path.join(synth_root, "Train", "labels.txt")
    outp = subprocess.run([sys.executable, EVAL, "--truth", str(tj), "--prediction", str(pj), "--labels", lab],
                          capture_output=True, text=True, check=True).stdout
    official = float([x for x in outp.splitlines() if x.startswith("Strict Macro-F1")][0].split()[-1])
    assert abs(official - score(truth, pred, d["labels"])["macro_f1"]) < 1e-6
    er = errors(truth, pred)["kinds"]
    assert er["boundary_end"] > 0 and er["label_confusion"] > 0 and er["spurious"] > 0


def test_validate_rejects():
    labs = synth.LABELS
    with pytest.raises(ValueError):
        validate({"1": [[3, 3, "DATE"]]}, ["1"], labs)
    with pytest.raises(ValueError):
        validate({"1": [[0, 3, "DATE"], [0, 3, "DATE"]]}, ["1"], labs)
    with pytest.raises(ValueError):
        validate({"1": [[0, 3, "XX"]]}, ["1"], labs)
    with pytest.raises(ValueError):
        validate({"1": []}, ["1", "2"], labs)
    assert validate({"1": [[0, 3, "DATE", "abc"]], "2": []}, ["1", "2"], labs)


FAMILY_KEYS = ["A", "B", "C", "D", "E", "F", "G"]


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_backbone_shapes_and_echo(key):
    torch.manual_seed(0)
    bb = load_backbone(key, tiny=True).eval()
    x = torch.randint(4, 30000, (2, 24))
    h = bb(x)
    assert h.shape == (2, 24, bb.d_model)
    assert bb(x, echo=True).shape == (2, 24, bb.d_model)
    if bb.is_causal:
        y = x.clone()
        y[:, 20] = 7
        assert torch.allclose(bb(x)[:, :20], bb(y)[:, :20], atol=1e-5)
        assert not torch.allclose(bb(x, echo=True)[:, :20], bb(y, echo=True)[:, :20], atol=1e-5)


@pytest.mark.parametrize("key", ["A", "E", "F"])
def test_bidirectional_switch(key):
    torch.manual_seed(0)
    bb = load_backbone(key, tiny=True).eval()
    x = torch.randint(4, 30000, (1, 16))
    y = x.clone()
    y[:, 12] = 9
    assert bb.set_bidirectional(True) > 0
    assert not torch.allclose(bb(x)[:, :12], bb(y)[:, :12], atol=1e-5)
    bb.set_bidirectional(False)
    assert torch.allclose(bb(x)[:, :12], bb(y)[:, :12], atol=1e-5)


def test_bidirectional_rejected_for_ssm():
    with pytest.raises(ValueError):
        load_backbone("D", tiny=True).set_bidirectional(True)


METHODS = [("lora", {}), ("lorma", {}), ("lorma_pi", {"r": 4}), ("ia3", {}), ("houlsby", {}), ("pfeiffer", {}),
           ("bitfit", {}), ("partial", {"k": 1}), ("full", {}), ("frozen", {})]


@pytest.mark.parametrize("key", ["A", "B", "C", "D", "F", "G"])
@pytest.mark.parametrize("method,args", METHODS)
def test_peft_identity_at_init(key, method, args):
    torch.manual_seed(0)
    bb = load_backbone(key, tiny=True).eval()
    x = torch.randint(4, 30000, (2, 20))
    with torch.no_grad():
        ref = bb(x).clone()
    info = P.apply_method(bb, method, **args)
    with torch.no_grad():
        out = bb(x)
    assert torch.allclose(out, ref, atol=1e-4), f"{key}/{method} changes the function at init"
    tr = {n for n, p in bb.named_parameters() if p.requires_grad}
    if method == "frozen":
        assert not tr
    elif method == "full":
        assert len(tr) == len(list(bb.parameters()))
    else:
        assert 0 < info["trainable"] < info["total"]
    if method in ("lora", "lorma", "lorma_pi"):
        assert all(n.endswith((".A", ".B")) for n in tr)
        assert info["wrapped"] > 0
    if method == "bitfit":
        assert all(n.endswith("bias") for n in tr)
    if tr:
        bb.train()
        loss = bb(x).float().pow(2).mean()
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in bb.parameters() if p.requires_grad)


@pytest.mark.parametrize("method", ["lora", "lorma", "lorma_pi"])
def test_merge_equals_wrapped(method):
    torch.manual_seed(0)
    bb = load_backbone("A", tiny=True).eval()
    P.apply_method(bb, method, r=4)
    with torch.no_grad():
        for m in bb.modules():
            if isinstance(m, (P.LoRALinear, P.LoRMALinear)):
                m.A.add_(0.05 * torch.randn_like(m.A))
                m.B.add_(0.05 * torch.randn_like(m.B))
    x = torch.randint(4, 30000, (2, 12))
    with torch.no_grad():
        a = bb(x).clone()
        assert P.merge_all(bb) > 0
        b = bb(x)
    assert torch.allclose(a, b, atol=1e-4)


def test_lorma_pi_rank_inflation():
    lin = torch.nn.Linear(16, 16, bias=False)
    m = P.LoRMALinear(lin, r=1, mode="pi")
    assert torch.allclose(m.multiplier(), torch.eye(16))
    with torch.no_grad():
        m.A.copy_(torch.randn_like(m.A))
        m.B.copy_(torch.randn_like(m.B))
    assert torch.linalg.matrix_rank(m.multiplier()) > 1


@pytest.mark.parametrize("method", ["prompt", "prefix"])
@pytest.mark.parametrize("key", ["A", "C", "D", "G"])
def test_prompts(method, key):
    torch.manual_seed(0)
    bb = load_backbone(key, tiny=True)
    info = P.apply_method(bb, method, n_prompt=5)
    x = torch.randint(4, 30000, (2, 10))
    h = bb(x)
    assert h.shape == (2, 10, bb.d_model)
    assert info["trainable"] == (5 * bb.d_model if method == "prompt" else len(bb.layers()) * 5 * bb.d_model)
    h.float().pow(2).mean().backward()
    p = bb.prompt if method == "prompt" else bb.deep_prompt
    assert p.grad is not None and p.grad.abs().sum() > 0


def _brute(em, trans, start, end, allowed=None, start_ok=None):
    T, n = em.shape
    best, arg = -1e18, None
    for path in itertools.product(range(n), repeat=T):
        s = start[path[0]] + em[0, path[0]] + end[path[-1]]
        for t in range(1, T):
            s += trans[path[t - 1], path[t]] + em[t, path[t]]
        if s > best:
            best, arg = s, list(path)
    return best, arg


def test_crf_nll_and_viterbi_bruteforce():
    torch.manual_seed(0)
    n, T = 3, 4
    crf = CRF(n)
    with torch.no_grad():
        crf.trans.normal_()
        crf.start.normal_()
        crf.end.normal_()
    em = torch.randn(1, T, n)
    tags = torch.tensor([[0, 2, 1, 1]])
    scores = []
    for path in itertools.product(range(n), repeat=T):
        s = crf.start[path[0]] + em[0, 0, path[0]] + crf.end[path[-1]]
        for t in range(1, T):
            s = s + crf.trans[path[t - 1], path[t]] + em[0, t, path[t]]
        scores.append(s)
    logZ = torch.logsumexp(torch.stack(scores), 0)
    gold = crf.start[0] + em[0, 0, 0] + crf.trans[0, 2] + em[0, 1, 2] + crf.trans[2, 1] + em[0, 2, 1] + \
        crf.trans[1, 1] + em[0, 3, 1] + crf.end[1]
    nll = crf.nll(em, tags, torch.ones(1, T, dtype=torch.bool)) * T
    assert torch.allclose(nll, logZ - gold, atol=1e-4)
    _, arg = _brute(em[0].numpy(), crf.trans.detach().numpy(), crf.start.detach().numpy(), crf.end.detach().numpy())
    assert crf.decode(em[0]) == arg


def test_constrained_decode_valid_bio():
    ts = D.TagSet(synth.LABELS)
    allowed, start = ts.constraints()
    rng = np.random.default_rng(0)
    for _ in range(20):
        lp = torch.log_softmax(torch.from_numpy(rng.normal(size=(30, ts.n)) * 3), -1)
        path = constrained_decode(lp, allowed, start)
        assert not (path[0] > 0 and not ts.is_b(path[0]))
        for a, b in zip(path, path[1:]):
            assert allowed[a, b]


@need_tok
@pytest.mark.parametrize("exp,over", [
    ("smoke_A", {}),
    ("smoke_C", {}),
    ("smoke_B", {"method": "bitfit"}),
    ("smoke_D", {"crf": True}),
    ("smoke_G", {"method": "prefix", "method_args": {"n_prompt": 4}}),
])
def test_end_to_end_tiny(tmp_path, synth_root, exp, over):
    from phase2.experiments import get_experiment
    from phase2.train import run
    cfg = get_experiment(exp, {"tiny": True, "tokenizer": TOK, "epochs": 2, **over})
    res = run(cfg, synth_root, [], str(tmp_path), "dev", device=torch.device("cpu"))
    out = tmp_path / exp
    assert (out / "submission.zip").exists() and (out / "result.json").exists()
    assert 0.0 <= res["dev"]["macro_f1"] <= 1.0
    pj = json.load(open(out / "prediction.json", encoding="utf-8"))
    d = D.load_split(synth_root)
    validate(pj, list(d["test"]), d["labels"])


@need_tok
def test_kd_teacher_student(tmp_path, synth_root):
    from phase2.experiments import get_experiment
    from phase2.train import run
    cpu = torch.device("cpu")
    t = get_experiment("smoke_A", {"tiny": True, "tokenizer": TOK, "name": "teach", "save_soft": True})
    run(t, synth_root, [], str(tmp_path), "full", device=cpu)
    assert (tmp_path / "teach__full" / "soft_labels.npz").exists()
    s = get_experiment("smoke_A", {"tiny": True, "tokenizer": TOK, "name": "stud", "backbone": "As",
                                   "kd": {"teacher": "teach", "alpha": 0.5, "T": 2.0}})
    res = run(s, synth_root, [], str(tmp_path), "dev", device=cpu)
    assert "dev" in res


@pytest.mark.parametrize("key", ["A", "C", "D"])
@pytest.mark.parametrize("method", ["rslora", "adalora", "qlora"])
def test_new_methods_init_and_grads(key, method):
    torch.manual_seed(0)
    bb = load_backbone(key, tiny=True).eval()
    x = torch.randint(4, 30000, (2, 16))
    with torch.no_grad():
        ref = bb(x).clone()
    info = P.apply_method(bb, method, r=4, targets="all")
    with torch.no_grad():
        out = bb(x)
    if method == "qlora":
        assert ((out - ref).norm() / ref.norm()) < (0.3 if key == "D" else 0.15)
        q = next(m for m in bb.modules() if isinstance(m, P.QLoRALinear))
        xi = torch.randn(3, q.in_features)
        exact = torch.nn.functional.linear(xi, q.weight(), q.bias)
        assert torch.allclose(q(xi), exact, atol=1e-5)
    else:
        assert torch.allclose(out, ref, atol=1e-4)
    assert info["wrapped"] > 0 and 0 < info["trainable"] < info["total"]
    bb.train()
    loss = bb(x).float().pow(2).mean()
    reg = P.peft_extra_loss(bb)
    if method == "adalora":
        assert reg is not None and reg.item() > 0
        loss = loss + reg
    loss.backward()
    assert all(p.grad is not None for p in bb.parameters() if p.requires_grad)


def test_nf4_roundtrip_error():
    torch.manual_seed(0)
    w = torch.randn(96, 80) * 0.05
    packed, absmax, shape = P.nf4_quantize(w)
    assert packed.dtype == torch.uint8 and packed.numel() == w.numel() // 2
    back = P.nf4_dequantize(packed, absmax, shape)
    assert back.shape == w.shape
    assert ((back - w).norm() / w.norm()) < 0.12
    lv = P.NF4.repeat(4)[None].repeat(2, 1)
    p2, a2, s2 = P.nf4_quantize(lv)
    assert torch.allclose(P.nf4_dequantize(p2, a2, s2), lv, atol=1e-6)


def test_adalora_rank_allocator_prunes():
    torch.manual_seed(0)
    bb = load_backbone("A", tiny=True)
    P.apply_method(bb, "adalora", r=6, targets="attn")
    mods = [m for m in bb.modules() if isinstance(m, P.AdaLoRALinear)]
    alloc = P.RankAllocator(mods, target_r=2, total_steps=50, every=5)
    opt = torch.optim.SGD([p for p in bb.parameters() if p.requires_grad], lr=0.1)
    x = torch.randint(4, 30000, (2, 12))
    for step in range(1, 51):
        opt.zero_grad()
        (bb(x).float().pow(2).mean() + P.peft_extra_loss(bb)).backward()
        alloc.step(step)
        opt.step()
    kept = sum(int(m.mask.sum()) for m in mods)
    assert kept <= 2 * len(mods) + 1 and kept < 6 * len(mods)
    assert alloc.budget(0) == 6 * len(mods) and alloc.budget(50) == 2 * len(mods)
