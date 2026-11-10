"""Architecture-agnostic, resumable training loop for Kaggle/Colab sessions.

The caller supplies the model, the optimizer, the LR schedule and every hyperparameter value;
this module only runs the loop and makes it survive session limits:

    from phase1.train.runner import RunConfig, train, causal_lm_loss
    from phase1.train.data import TokenShards, causal_batch, fixed_eval_batches

    shards, val = TokenShards(SHARDS, "train"), TokenShards(SHARDS, "val")
    cfg = RunConfig(out_dir="/kaggle/working/run_A", total_steps=..., grad_accum=..., grad_clip=...,
                    eval_every=..., max_hours=11.0, token_milestones=[...])
    train(model,
          make_optimizer=lambda m: torch.optim.AdamW(m.parameters(), lr=..., weight_decay=...),
          lr_lambda=lambda step: ...,                       # multiplier on the base LR
          loss_fn=causal_lm_loss,                           # or encdec_loss
          train_batch=lambda index: causal_batch(shards, B, T, seed=0, index=index),
          eval_batches=fixed_eval_batches(val, lambda sh, s, i, mix: causal_batch(sh, B, T, s, i, mix), n),
          tokens_per_batch=B * T, cfg=cfg)

Calling train() again with the same out_dir resumes from the latest checkpoint (model,
optimizer, LR schedule, GradScaler, step, RNG). Data batches are a function of the step
index, so nothing else needs restoring. Logs go to <out_dir>/log.jsonl.
"""
import dataclasses
import json
import math
import os
import time

import torch
import torch.nn.functional as F


@dataclasses.dataclass
class RunConfig:
    out_dir: str
    total_steps: int
    grad_accum: int = 1
    grad_clip: float | None = None
    amp: bool = True
    eval_every: int = 500
    log_every: int = 20
    ckpt_every_minutes: float = 30.0
    max_hours: float = 11.0
    keep_last: int = 2
    token_milestones: list = dataclasses.field(default_factory=list)
    seed: int = 0
    max_bad_steps: int = 50


def causal_lm_loss(model, batch, device):
    x, y = (t.to(device, non_blocking=True) for t in batch)
    logits = model(x)
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))


def encdec_loss(model, batch, device):
    src, tgt_in, tgt_out = (t.to(device, non_blocking=True) for t in batch)
    logits = model(src, tgt_in)
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), tgt_out.reshape(-1))


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _ckpt_dir(cfg):
    d = os.path.join(cfg.out_dir, "ckpt")
    os.makedirs(d, exist_ok=True)
    return d


def _latest(cfg):
    d = _ckpt_dir(cfg)
    steps = sorted(int(f[5:-3]) for f in os.listdir(d) if f.startswith("step_") and f.endswith(".pt"))
    return os.path.join(d, f"step_{steps[-1]}.pt") if steps else None


def _save(cfg, model, opt, sched, scaler, step, tokens, wall):
    d = _ckpt_dir(cfg)
    _atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                  "scaler": scaler.state_dict(), "step": step, "tokens": tokens, "wall_seconds": wall,
                  "rng_cpu": torch.get_rng_state(),
                  "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                  "config": dataclasses.asdict(cfg)}, os.path.join(d, f"step_{step}.pt"))
    old = sorted(int(f[5:-3]) for f in os.listdir(d) if f.startswith("step_") and f.endswith(".pt"))
    for s in old[:-cfg.keep_last]:
        os.remove(os.path.join(d, f"step_{s}.pt"))


@torch.no_grad()
def evaluate(model, loss_fn, eval_batches, device, amp):
    """Mean loss per source over the fixed eval batches, plus the token-weighted overall mean."""
    was_training = model.training
    model.eval()
    out = {}
    for src, batches in eval_batches.items():
        losses = []
        for b in batches:
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                losses.append(loss_fn(model, b, device).item())
        out[src] = sum(losses) / len(losses)
    out["mean"] = sum(out.values()) / len(out)
    model.train(was_training)
    return out


def train(model, make_optimizer, lr_lambda, loss_fn, train_batch, eval_batches, tokens_per_batch, cfg, on_step=None):
    """on_step(step, cfg): optional hook after every optimizer step; may change cfg.total_steps (re-planning)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = cfg.amp and device == "cuda"
    os.makedirs(cfg.out_dir, exist_ok=True)
    model.to(device).train()
    opt = make_optimizer(model)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step, tokens, wall_before = 0, 0, 0.0

    latest = _latest(cfg)
    if latest:
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        step, tokens, wall_before = ck["step"], ck["tokens"], ck["wall_seconds"]
        torch.set_rng_state(ck["rng_cpu"])
        if ck["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        print(f"resumed from {latest} at step {step} ({tokens:,} tokens)")
    else:
        torch.manual_seed(cfg.seed)

    log = open(os.path.join(cfg.out_dir, "log.jsonl"), "a", encoding="utf-8")
    milestones = sorted(m for m in cfg.token_milestones if m > tokens)
    t_session, t_ckpt, t_log = time.time(), time.time(), time.time()
    tok_since_log, bad_steps, stop_reason = 0, 0, "total_steps"

    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    while step < cfg.total_steps:
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(cfg.grad_accum):
            batch = train_batch(step * cfg.grad_accum + micro)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = loss_fn(model, batch, device) / cfg.grad_accum
            scaler.scale(loss).backward()
            loss_sum += loss.item()
        scaler.unscale_(opt)
        grad_norm = (torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
                     if cfg.grad_clip else None)
        if amp:
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skipped = scaler.get_scale() < scale_before
        elif math.isfinite(loss_sum):
            opt.step()
            skipped = False
        else:
            skipped = True
        bad_steps = bad_steps + 1 if skipped else 0
        if bad_steps >= cfg.max_bad_steps:
            stop_reason = f"{bad_steps} consecutive non-finite steps"
            break
        sched.step()
        step += 1
        tokens += tokens_per_batch * cfg.grad_accum
        tok_since_log += tokens_per_batch * cfg.grad_accum

        if step % cfg.log_every == 0:
            now = time.time()
            write({"step": step, "tokens": tokens, "loss": loss_sum, "lr": sched.get_last_lr()[0],
                   "grad_norm": grad_norm, "scale": scaler.get_scale() if amp else None,
                   "tok_per_s": tok_since_log / (now - t_log), "skipped": skipped})
            t_log, tok_since_log = now, 0
        if step % cfg.eval_every == 0 or step == cfg.total_steps:
            ev = evaluate(model, loss_fn, eval_batches, device, amp)
            write({"step": step, "tokens": tokens, "eval": ev})
            print(f"step {step} tokens {tokens:,} train {loss_sum:.4f} val {ev['mean']:.4f}")
        while milestones and tokens >= milestones[0]:
            m = milestones.pop(0)
            _atomic_save({"model": model.state_dict(), "step": step, "tokens": tokens},
                         os.path.join(_ckpt_dir(cfg), f"milestone_{m}.pt"))
        wall = wall_before + time.time() - t_session
        if time.time() - t_ckpt >= cfg.ckpt_every_minutes * 60:
            _save(cfg, model, opt, sched, scaler, step, tokens, wall)
            t_ckpt = time.time()
        if on_step is not None:
            on_step(step, cfg)
        if (time.time() - t_session) / 3600 >= cfg.max_hours:
            stop_reason = "max_hours"
            break

    wall = wall_before + time.time() - t_session
    _save(cfg, model, opt, sched, scaler, step, tokens, wall)
    write({"step": step, "tokens": tokens, "stopped": stop_reason, "wall_seconds": wall})
    log.close()
    print(f"stopped at step {step} ({tokens:,} tokens): {stop_reason}")
    return {"step": step, "tokens": tokens, "stopped": stop_reason}
