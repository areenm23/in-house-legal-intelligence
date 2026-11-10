"""Checks on the four Phase-1 architectures: size limits, causality, and that the cached /
recurrent decoding paths compute the same logits as the full forward pass."""
import importlib.util
import os

import pytest
import torch

from phase1.checks.model_checks import (check_causal, check_encdec_decoder_causal, check_equivalent,
                                        check_param_range, count_params)
from phase1.models.decoder import DecoderAPI, DecoderConfig, DecoderLM
from phase1.models.encdec import EncDecAPI, EncDecConfig, EncoderDecoderLM
from phase1.models.mamba2 import Mamba2API, Mamba2Config, Mamba2LM
from phase1.models.retnet import RetNetAPI, RetNetConfig, RetNetLM
from phase1.models.optim import make_muon_adamw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V = 101


def tiny(kind):
    torch.manual_seed(0)
    if kind == "A":
        m = DecoderLM(DecoderConfig(vocab_size=V, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=96,
                                    max_seq_len=256))
        return m, DecoderAPI(m)
    if kind == "B":
        m = EncoderDecoderLM(EncDecConfig(vocab_size=V, d_model=64, n_enc_layers=2, n_dec_layers=2, n_heads=4,
                                          n_kv_heads=2, d_ff=96, max_seq_len=256))
        return m, EncDecAPI(m)
    if kind == "C":
        m = RetNetLM(RetNetConfig(vocab_size=V, d_model=64, n_layers=2, n_heads=4, d_ff=96, max_seq_len=256))
        return m, RetNetAPI(m)
    m = Mamba2LM(Mamba2Config(vocab_size=V, d_model=64, n_layers=2, d_state=16, headdim=16, chunk_size=8))
    return m, Mamba2API(m)


def cached_logits(api, ids, split):
    """Prefill ids[:, :split], then feed the rest one token at a time; logits for positions split-1 .. T-1."""
    logits, st = api.prefill(ids[:, :split])
    outs = [logits]
    for t in range(split, ids.shape[1]):
        logits, st = api.step(st, ids[:, t])
        outs.append(logits)
    return torch.stack(outs, 1)


@pytest.mark.parametrize("kind", ["A", "C", "D"])
def test_causal_models(kind):
    m, api = tiny(kind)
    m.eval()
    check_causal(lambda x: m(x), V, seq_len=40)
    ids = torch.randint(0, V, (3, 37), generator=torch.Generator().manual_seed(1))
    full = m(ids)
    for split in (1, 9, 20):
        got = cached_logits(api, ids, split)
        assert torch.allclose(got, full[:, split - 1:], atol=2e-4, rtol=1e-3), (kind, split,
                                                                                (got - full[:, split - 1:]).abs().max())


@pytest.mark.parametrize("kind", ["C", "D"])
def test_recurrent_equals_parallel(kind):
    m, _ = tiny(kind)
    m.eval()
    check_equivalent(lambda x: m(x), lambda x: m.forward_recurrent(x), V, seq_len=30, atol=2e-4)


def test_encdec():
    m, api = tiny("B")
    m.eval()
    check_encdec_decoder_causal(lambda s, t: m(s, t), V)
    g = torch.Generator().manual_seed(2)
    src, cont = torch.randint(0, V, (2, 20), generator=g), torch.randint(0, V, (2, 12), generator=g)
    tgt_in = torch.cat([torch.full((2, 1), 2), cont[:, :-1]], 1)
    full = m(src, tgt_in)
    logits, st = api.prefill(src)
    outs = [logits]
    for t in range(cont.shape[1] - 1):
        logits, st = api.step(st, cont[:, t])
        outs.append(logits)
    assert torch.allclose(torch.stack(outs, 1), full, atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("kind", ["A", "B", "C", "D"])
def test_training_step_reduces_loss(kind):
    m, _ = tiny(kind)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(3)
    x = torch.randint(0, V, (4, 33), generator=g)
    losses = []
    for _ in range(15):
        logits = m(x[:, :16], x[:, 16:32]) if kind == "B" else m(x[:, :-1])
        y = x[:, 17:33] if kind == "B" else x[:, 1:]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.5


def test_muon_adamw_steps_and_saves():
    m, _ = tiny("A")
    opt = make_muon_adamw(m, muon_lr=0.02, adam_lr=1e-3, weight_decay=0.1)
    x = torch.randint(0, V, (2, 17))
    for _ in range(3):
        loss = torch.nn.functional.cross_entropy(m(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    sd = opt.state_dict()
    opt2 = make_muon_adamw(m, muon_lr=0.02, adam_lr=1e-3, weight_decay=0.1)
    opt2.load_state_dict(sd)
    assert len(opt2.state) == len(opt.state)


def load_run(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "runs", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", ["run_A", "run_B", "run_C", "run_D", "run_A_muon", "run_A_small"])
def test_run_files_build_in_range(name):
    run = load_run(name)
    m = run.build_model(32768)
    n = check_param_range(m)
    print(name, {k: round(v / 1e6, 2) for k, v in count_params(m, 32768).items()})
    assert run.make_lr_lambda(1000)(999) == pytest.approx(0.1, abs=1e-3)
    run.make_optimizer(m)
    run.decoder_api(m)


from phase1.models.hybrid import HybridAPI, HybridConfig, HybridLM
from phase1.models.mlm import EncoderMLM, MLMConfig, MLMFillAPI, mask_id_for, mlm_batch
from phase1.models.moe import MoEConfig, MoEDecoderLM


def tiny2(kind):
    torch.manual_seed(0)
    if kind == "E":
        m = MoEDecoderLM(MoEConfig(vocab_size=V, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, n_experts=4,
                                   top_k=2, d_ff_expert=48, max_seq_len=256))
        return m, DecoderAPI(m)
    m = HybridLM(HybridConfig(vocab_size=V, d_model=64, n_groups=2, mamba_per_group=2, n_heads=4, n_kv_heads=2,
                              d_ff=96, d_state=16, headdim=16, chunk_size=8, max_seq_len=256))
    return m, HybridAPI(m)


@pytest.mark.parametrize("kind", ["E", "F"])
def test_round2_causal_and_cached(kind):
    m, api = tiny2(kind)
    m.eval()
    check_causal(lambda x: m(x), V, seq_len=40)
    ids = torch.randint(0, V, (3, 37), generator=torch.Generator().manual_seed(1))
    full = m(ids)
    for split in (1, 9, 20):
        got = cached_logits(api, ids, split)
        assert torch.allclose(got, full[:, split - 1:], atol=2e-4, rtol=1e-3), (kind, split)


def test_moe_routes_and_balances():
    m, _ = tiny2("E")
    x = torch.randint(0, V, (4, 33))
    m(x)
    assert m.aux["lb"].item() >= 0.99 and m.aux["z"].item() > 0
    load = m.blocks[0].moe.last_load
    assert torch.isclose(load.sum(), torch.tensor(1.0)) and (load > 0).sum() >= 2
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(15):
        lg = m(x[:, :-1])
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, V), x[:, 1:].reshape(-1)) + 0.01 * m.aux["lb"]
        opt.zero_grad()
        loss.backward()
        assert m.blocks[0].moe.router.weight.grad is not None
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.5


def tiny_mlm():
    torch.manual_seed(0)
    Vm = 120
    return EncoderMLM(MLMConfig(vocab_size=Vm, d_model=64, n_layers=2, n_heads=4, n_kv_heads=4, d_ff=96,
                                max_seq_len=256)), Vm


def test_mlm_is_bidirectional_and_fill_api_matches_full_forward():
    m, Vm = tiny_mlm()
    m.eval()
    x = torch.randint(4, mask_id_for(Vm), (2, 30))
    base = m(x)
    y = x.clone()
    y[:, -1] = (y[:, -1] + 1) % mask_id_for(Vm)
    assert (m(y)[:, 0] - base[:, 0]).abs().max() > 1e-4
    api = MLMFillAPI(m, n_masks=6, block=2)
    logits, st = api.prefill(x)
    full = m(st["x"].clone())
    assert torch.allclose(logits, full[:, 30].clamp(min=-1e9), atol=1e-4) or torch.allclose(
        logits[:, :m.mask_id], full[:, 30, :m.mask_id], atol=1e-4)
    nxt = torch.tensor([5, 6])
    logits2, st = api.step(st, nxt)
    assert torch.allclose(logits2[:, :m.mask_id], full[:, 31, :m.mask_id], atol=1e-4)
    logits3, st = api.step(st, nxt)
    ref = m(st["x"].clone())
    assert torch.allclose(logits3[:, :m.mask_id], ref[:, 32, :m.mask_id], atol=1e-4)
    assert torch.isinf(logits3[:, m.mask_id:]).all()
    assert (st["x"][:, 30:32] == nxt[:, None]).all()


class _FakeShards:
    def __init__(self, n, vocab):
        rng = np.random.default_rng(0)
        self.arrays = {"acts": rng.integers(4, vocab - 13, n).astype(np.uint16)}
        self.names = ["acts"]
        self.meta = {"vocab_size": vocab}

    def weights(self, mix=None):
        return np.array([1.0])


def test_mlm_batch_masks():
    sh = _FakeShards(50_000, 32768)
    mid = mask_id_for(32768)
    x, lab = mlm_batch(sh, 64, 256, seed=0, index=3, mix=None, mask_id=mid)
    x2, lab2 = mlm_batch(sh, 64, 256, seed=0, index=3, mix=None, mask_id=mid)
    assert torch.equal(x, x2) and torch.equal(lab, lab2)
    sel = lab != -100
    suffix_rows = [b for b in range(64) if (x[b, -32:] == mid).all() and (lab[b, :-256] == -100).all()
                   and sel[b, -1]]
    assert 5 <= len(suffix_rows) <= 30
    frac = sel.float().mean().item()
    assert 0.2 < frac < 0.45
    assert ((x == mid) <= sel).all()
    assert (x[~sel] == lab[~sel].clamp(min=0))[lab[~sel] != -100].all()


import numpy as np


@pytest.mark.parametrize("name", ["run_E", "run_F", "run_G", "run_A_cont", "run_B_cont", "run_C_cont",
                                  "run_D_cont", "run_A_muon_cont"])
def test_round2_run_files(name):
    run = load_run(name)
    m = run.build_model(32768)
    check_param_range(m)
    print(name, {k: round(v / 1e6, 2) for k, v in count_params(m, 32768).items()})
    run.make_optimizer(m)
    run.decoder_api(m)
    assert run.make_lr_lambda(1000)(999) == pytest.approx(0.1, abs=1e-3)
    if name.endswith("_cont"):
        assert run.INIT_FROM and run.NAME == run.INIT_FROM + "_cont" and run.CFG["seed"] != 1234


def test_recurrent_state_stays_fp32_under_autocast():
    """Under CUDA autocast the state einsums must stay fp32 (fp16 gives garbage/NaN RetNet text).
    CPU autocast (bf16) also lowers einsum/bmm, so it reproduces the dtype problem."""
    for kind in ("C", "D"):
        m, api = tiny(kind)
        m.eval()
        ids = torch.randint(0, V, (2, 20))
        with torch.no_grad():
            ref, _ = api.prefill(ids)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits, st = api.prefill(ids)
                for _ in range(3):
                    logits, st = api.step(st, logits.argmax(-1))
        states = st if isinstance(st, list) else st[0]
        for s in states:
            for v in s.values():
                if torch.is_tensor(v) and v.is_floating_point():
                    assert v.dtype == torch.float32, (kind, v.dtype)


def test_replanner_moves_total_to_measured_speed(tmp_path, monkeypatch):
    import json as _json
    from phase1.train import launch
    clock = [1000.0]
    monkeypatch.setattr(launch.time, "time", lambda: clock[0])
    plan = {"total_steps": 1000}
    path = tmp_path / "plan.json"

    class C:
        total_steps = 1000
    rp = launch.Replanner(plan, str(path), deadline=1000.0 + 1180 * 2.0 + 180, margin_s=180)
    lam = rp.lr_lambda(lambda total: (lambda s: s / total))
    for step in range(1, 102):
        rp(step, C)
        clock[0] += 2.0
    assert C.total_steps == rp.total and 1150 < rp.total < 1200
    assert lam(rp.total) == 1.0
    assert _json.load(open(path))["replans"][0]["step"] == 101
