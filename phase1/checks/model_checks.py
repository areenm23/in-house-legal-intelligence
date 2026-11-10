"""Architecture-agnostic checks to run on every Phase-1 model before spending GPU hours on it.

The checks only call the model through small functions, so they work for any design:

  causal LM (decoder, RetNet, Mamba):   forward(input_ids[B,T]) -> logits[B,T,V]
  encoder-decoder:                      forward(src_ids[B,S], tgt_ids[B,T]) -> logits[B,T,V]

    from phase1.checks.model_checks import (count_params, check_param_range, check_causal,
        check_encdec_decoder_causal, check_equivalent, measure_throughput)
    print(count_params(model, vocab_size=V))
    check_causal(lambda x: model(x), vocab_size=V)
    check_equivalent(lambda x: model(x), lambda x: model.forward_recurrent(x), vocab_size=V)  # RetNet/Mamba
    print(measure_throughput(step_fn, make_batch, tokens_per_batch))

The failure modes these catch are the expensive ones: a mask/scan/conv-padding bug that
lets position t see the future (loss looks great, generation is garbage), a recurrent form
that disagrees with the parallel form, and a model outside the 50M-300M limit.
"""
import time

import torch
import torch.nn as nn


def count_params(model, vocab_size=None):
    """Total / embedding / output-head / non-embedding parameter counts (shared tensors counted once)."""
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    emb_ids, emb = set(), 0
    for m in model.modules():
        if isinstance(m, nn.Embedding) and id(m.weight) not in emb_ids:
            emb_ids.add(id(m.weight))
            emb += m.weight.numel()
    head = 0
    if vocab_size is not None:
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.out_features == vocab_size and id(m.weight) not in emb_ids:
                emb_ids.add(id(m.weight))
                head += m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
    return {"total": total, "embedding": emb, "untied_output_head": head,
            "non_embedding": total - emb - head}


def check_param_range(model, lo=50_000_000, hi=300_000_000):
    n = count_params(model)["total"]
    assert lo <= n <= hi, f"{n / 1e6:.2f}M parameters is outside the {lo / 1e6:.0f}M-{hi / 1e6:.0f}M limit"
    return n


@torch.no_grad()
def check_causal(forward, vocab_size, seq_len=64, batch=2, cuts=(0, 1, 17, None), atol=1e-4,
                 device="cpu", seed=0):
    """Changing tokens after position t must not change logits at positions <= t.

    Run with the model in eval mode (no dropout) and in float32. Returns the largest
    difference seen at allowed positions; raises AssertionError on a leak.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab_size, (batch, seq_len), generator=g).to(device)
    base = forward(x).float()
    assert base.shape[:2] == (batch, seq_len), f"expected logits [B,T,V], got {tuple(base.shape)}"
    worst = 0.0
    for t in cuts:
        t = seq_len // 2 if t is None else t
        y = x.clone()
        y[:, t + 1:] = torch.randint(0, vocab_size, (batch, seq_len - t - 1), generator=g).to(device)
        diff = (forward(y).float()[:, :t + 1] - base[:, :t + 1]).abs().max().item()
        worst = max(worst, diff)
        assert diff <= atol, f"causality leak: logits at positions <= {t} changed by {diff:.3g} when later tokens changed"
    y = x.clone()
    y[:, 5] = (y[:, 5] + 1) % vocab_size
    assert (forward(y).float()[:, 5] - base[:, 5]).abs().max().item() > atol, \
        "logits at t do not depend on token t (model ignores its input?)"
    return worst


@torch.no_grad()
def check_encdec_decoder_causal(forward, vocab_size, src_len=48, tgt_len=32, batch=2, atol=1e-4,
                                device="cpu", seed=0):
    """Decoder side of an encoder-decoder: future target tokens must not leak; source must matter."""
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, vocab_size, (batch, src_len), generator=g).to(device)
    worst = check_causal(lambda tgt: forward(src, tgt), vocab_size, seq_len=tgt_len, batch=batch,
                         atol=atol, device=device, seed=seed + 1)
    tgt = torch.randint(0, vocab_size, (batch, tgt_len), generator=g).to(device)
    base = forward(src, tgt).float()
    src2 = src.clone()
    src2[:, -1] = (src2[:, -1] + 1) % vocab_size
    assert (forward(src2, tgt).float() - base).abs().max().item() > atol, \
        "decoder output does not depend on the encoder input (cross-attention not wired?)"
    return worst


@torch.no_grad()
def check_equivalent(forward_a, forward_b, vocab_size, seq_len=64, batch=2, atol=1e-3, rtol=1e-3,
                     device="cpu", seed=0):
    """Two computations of the same model must agree, e.g. RetNet parallel vs recurrent vs
    chunkwise, Mamba parallel scan vs step-by-step, or full forward vs KV-cached decoding."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab_size, (batch, seq_len), generator=g).to(device)
    a, b = forward_a(x).float(), forward_b(x).float()
    assert a.shape == b.shape, f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"
    diff = (a - b).abs()
    bad = diff > atol + rtol * b.abs()
    assert not bad.any(), (f"forms disagree at {bad.sum().item()} entries; max abs diff {diff.max().item():.3g}; "
                           f"first bad position {bad.nonzero()[0].tolist()}")
    return diff.max().item()


def measure_throughput(step_fn, make_batch, tokens_per_batch, steps=20, warmup=5):
    """Training tokens/sec for one full step (forward+backward+optimizer), excluding warm-up.

    step_fn(batch) must do the whole training step; make_batch() returns a batch already on
    the device. Reports peak CUDA memory when on GPU. Measure every architecture with the
    same sequence length, batch size and precision.
    """
    cuda = torch.cuda.is_available()
    for _ in range(warmup):
        step_fn(make_batch())
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    for _ in range(steps):
        step_fn(make_batch())
    if cuda:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t
    out = {"tokens_per_sec": steps * tokens_per_batch / dt, "sec_per_step": dt / steps}
    if cuda:
        out["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return out
