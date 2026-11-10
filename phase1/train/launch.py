"""Run one training job described by a run file on the prepared shards.

  python -m phase1.train.launch --run runs/run_A.py --data <p1data dir> --out <runs dir>
  python -m phase1.train.launch --run runs/run_A.py --data ... --out ... --pilot 200

A run file is plain Python that defines (see runs/run_A.py):
  NAME           output folder name, e.g. "A_decoder_v1"
  KIND           "causal" (forward(ids) -> logits) or "encdec" (forward(src, tgt) -> logits)
  CFG            dict: batch_size, seq_len (causal) or src_len + tgt_len (encdec), grad_accum,
                 total_steps, grad_clip, eval_every, eval_batches, eval_batch_size,
                 token_milestones, mix, seed
  build_model(vocab_size) -> nn.Module
  make_optimizer(model)   -> torch optimizer
  lr_lambda(step)         -> multiplier on the optimizer's base LR
  optional: loss_fn(model, batch, device), train_batch(shards, index), eval_batch(shards, seed, index, mix)

This file only wires those into phase1.train.runner.train(); it sets no hyperparameter.

Resume: before training, the newest checkpoint of the same NAME found under --inputs
(default /kaggle/input, i.e. previous sessions' outputs attached as notebook inputs) is
copied into <out>/<NAME> if it is newer than what is there; train() then resumes from it.

--pilot N: run N steps into <out>/<NAME>_pilot, then report tokens/s, peak GPU memory and
tokens per hour (written to pilot.json), so the step budget can be chosen from a measurement.
"""
import argparse
import glob
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import statistics
import time

import torch

from ..checks.model_checks import check_param_range, count_params
from .data import TokenShards, causal_batch, continuation_batch, fixed_eval_batches
from .runner import RunConfig, causal_lm_loss, encdec_loss, train

REQUIRED = {"causal": ["batch_size", "seq_len"], "encdec": ["batch_size", "src_len", "tgt_len"],
            "mlm": ["batch_size", "seq_len"]}
COMMON = ["grad_accum", "total_steps", "grad_clip", "eval_every", "eval_batches", "eval_batch_size",
          "token_milestones", "mix", "seed"]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_run(path):
    spec = importlib.util.spec_from_file_location("run_" + re.sub(r"\W", "_", os.path.basename(path)), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("NAME", "KIND", "CFG", "build_model", "make_optimizer"):
        if not hasattr(mod, name):
            raise SystemExit(f"{path}: missing {name}")
    if not (hasattr(mod, "lr_lambda") or hasattr(mod, "make_lr_lambda")):
        raise SystemExit(f"{path}: missing lr_lambda (or make_lr_lambda(total_steps))")
    if mod.KIND not in REQUIRED:
        raise SystemExit(f"{path}: KIND must be one of {list(REQUIRED)}")
    keys = REQUIRED[mod.KIND] + COMMON
    absent = [k for k in keys if k not in mod.CFG]
    if absent:
        raise SystemExit(f"{path}: CFG is missing {absent}")
    blank = [k for k in keys if mod.CFG[k] is None and k not in ("mix", "grad_clip")]
    if blank:
        raise SystemExit(f"{path}: CFG values still blank: {blank}")
    if mod.CFG["total_steps"] == "auto" and not hasattr(mod, "make_lr_lambda"):
        raise SystemExit(f"{path}: total_steps='auto' needs make_lr_lambda(total_steps)")
    if mod.KIND == "mlm":
        missing = [f for f in ("loss_fn", "train_batch", "eval_batch") if not hasattr(mod, f)]
        if missing:
            raise SystemExit(f"{path}: KIND='mlm' needs {missing}")
    return mod


def batch_fns(run, shards, val, eval_needed=True):
    c = run.CFG
    takes_cfg = lambda f: "cfg" in inspect.signature(f).parameters
    if hasattr(run, "train_batch"):
        if takes_cfg(run.train_batch):
            train_batch = lambda i: run.train_batch(shards, i, cfg=c)
        else:
            train_batch = lambda i: run.train_batch(shards, i)
    elif run.KIND == "causal":
        train_batch = lambda i: causal_batch(shards, c["batch_size"], c["seq_len"], c["seed"], i, c["mix"])
    else:
        train_batch = lambda i: continuation_batch(shards, c["batch_size"], c["src_len"], c["tgt_len"],
                                                   c["seed"], i, c["mix"])
    eb = c["eval_batch_size"]
    if hasattr(run, "eval_batch"):
        make_eval = (lambda sh, seed, i, mix: run.eval_batch(sh, seed, i, mix, cfg=c)) if takes_cfg(run.eval_batch)             else run.eval_batch
    elif run.KIND == "causal":
        make_eval = lambda sh, seed, i, mix: causal_batch(sh, eb, c["seq_len"], seed, i, mix)
    else:
        make_eval = lambda sh, seed, i, mix: continuation_batch(sh, eb, c["src_len"], c["tgt_len"], seed, i, mix)
    tokens = c["batch_size"] * (c["src_len"] + c["tgt_len"] if run.KIND == "encdec" else c["seq_len"])
    evals = fixed_eval_batches(val, make_eval, c["eval_batches"]) if eval_needed else None
    return train_batch, evals, tokens


def _micro_seconds(model, loss_fn, batches, device, amp):
    """Seconds per forward+backward micro-batch (median of the timed ones; no optimizer step)."""
    times = []
    for i, b in enumerate(batches):
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = loss_fn(model, b, device)
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= 2:
            times.append(time.perf_counter() - t)
    model.zero_grad(set_to_none=True)
    return statistics.median(times)


class Replanner:
    """Re-fits total_steps to the measured speed so the LR schedule ends at the deadline.

    The probe in make_plan times forward+backward only; real steps also pay for the optimizer,
    data loading, evaluation and checkpointing. At steps 100, 300, then every 500, the remaining time
    is divided by the measured seconds per step. Changing the total mid-run shifts the cosine
    curve slightly; plan.json records every re-plan.
    """

    def __init__(self, plan, plan_path, deadline, margin_s=180):
        self.plan, self.path, self.deadline, self.margin = plan, plan_path, deadline, margin_s
        self.total = plan["total_steps"]
        self.t0 = self.s0 = None

    def lr_lambda(self, make):
        return lambda s: make(self.total)(s)

    def __call__(self, step, cfg):
        now = time.time()
        if self.t0 is None:
            self.t0, self.s0 = now, step
            return
        done = step - self.s0
        if not (done == 100 or done == 300 or (done > 300 and done % 500 == 0)):
            return
        sec = (now - self.t0) / done
        new = step + max(1, int((self.deadline - now - self.margin) / sec))
        if self.plan.get("max_steps"):
            new = min(new, self.plan["max_steps"])
        if abs(new - self.total) > max(2, 0.01 * self.total):
            print(f"replan at step {step}: {sec:.2f} s/step measured -> total_steps {self.total} -> {new}", flush=True)
            self.total = cfg.total_steps = new
            self.plan["total_steps"] = new
            self.plan.setdefault("replans", []).append({"step": step, "sec_per_step": round(sec, 3), "total": new})
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.plan, f, indent=2)


class SimpleRun:
    """A run module seen through a replaced CFG (micro-batch / accumulation / steps from the plan)."""

    def __init__(self, run, cfg):
        self._run, self.CFG = run, cfg

    def __getattr__(self, k):
        return getattr(self._run, k)


def make_plan(run, model, shards, val, run_dir, deadline, loss_fn, probe_batches=8, safety=0.93):
    """Fix the micro-batch size and total_steps of a time-boxed run; saved to plan.json, reused on resume.

    The micro-batch is halved (grad_accum doubled, so tokens per step are unchanged) until a
    forward+backward fits with room left for the AdamW moments. total_steps is what fits before
    the deadline at the measured speed, so the LR schedule finishes when the time does.
    """
    path = os.path.join(run_dir, "plan.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            plan = json.load(f)
        print(f"plan (existing): {plan}")
        return plan
    c = dict(run.CFG)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    model.to(device).train()
    n_params = sum(p.numel() for p in model.parameters())
    while True:
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            tb, _, tokens = batch_fns(SimpleRun(run, c), shards, val, eval_needed=False)
            sec = _micro_seconds(model, loss_fn, [tb(10_000_000 + i) for i in range(probe_batches)], device, amp)
            if device == "cuda":
                need = torch.cuda.max_memory_allocated() + 8 * n_params
                total = torch.cuda.get_device_properties(0).total_memory
                if need > 0.90 * total:
                    raise torch.cuda.OutOfMemoryError(f"needs ~{need / 1e9:.1f} of {total / 1e9:.1f} GB")
            break
        except torch.cuda.OutOfMemoryError as e:
            model.zero_grad(set_to_none=True)
            if c["batch_size"] == 1:
                raise
            print(f"micro-batch {c['batch_size']} does not fit ({str(e)[:100]}); halving it")
            c["batch_size"] //= 2
            c["grad_accum"] *= 2
    step_s = sec * c["grad_accum"] * 1.03
    eval_s = sec / 3 * c["eval_batches"] * len(val.names) * c["eval_batch_size"] / c["batch_size"]
    step_s += eval_s / c["eval_every"]
    left = deadline - time.time() - 120
    steps = max(1, int(left * safety / step_s))
    if c.get("max_steps"):
        steps = min(steps, int(c["max_steps"]))
    plan = {"batch_size": c["batch_size"], "grad_accum": c["grad_accum"], "total_steps": steps,
            "max_steps": int(c["max_steps"]) if c.get("max_steps") else None,
            "sec_per_step": round(step_s, 4), "tokens_per_step": tokens * c["grad_accum"],
            "planned_tokens": steps * tokens * c["grad_accum"],
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
            "made": time.strftime("%Y-%m-%d %H:%M:%S"), "deadline": deadline}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    print(f"plan: {json.dumps(plan)}")
    return plan


def _ckpt_step(path):
    m = re.search(r"step_(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def restore_from_inputs(name, run_dir, inputs):
    """Copy the newest <inputs>/**/<name>/ (by checkpoint step) into run_dir if it is ahead of run_dir."""
    local = max([_ckpt_step(p) for p in glob.glob(os.path.join(run_dir, "ckpt", "step_*.pt"))], default=-1)
    best, best_step = None, local
    for p in glob.glob(os.path.join(inputs, "**", name, "ckpt", "step_*.pt"), recursive=True):
        s = _ckpt_step(p)
        if s > best_step:
            best, best_step = os.path.dirname(os.path.dirname(p)), s
    if best is None:
        print(f"resume: nothing newer than step {local} under {inputs}")
        return local
    shutil.rmtree(run_dir, ignore_errors=True)
    shutil.copytree(best, run_dir, ignore=shutil.ignore_patterns("*.tmp"))
    ck = os.path.join(run_dir, "ckpt")
    for p in glob.glob(os.path.join(ck, "step_*.pt")):
        if _ckpt_step(p) != best_step:
            os.remove(p)
    print(f"resume: copied {best} (step {best_step}) -> {run_dir}")
    return best_step


def init_from_inputs(model, init_name, run_dir, inputs):
    """Continued training: load the weights of the newest <inputs>/**/<init_name>/ckpt/step_*.pt
    (or of the file init_name itself) into a fresh run; skipped once this run has its own checkpoints."""
    if glob.glob(os.path.join(run_dir, "ckpt", "step_*.pt")):
        return None
    cands = [init_name] if os.path.isfile(init_name) else         glob.glob(os.path.join(inputs, "**", init_name, "ckpt", "step_*.pt"), recursive=True)
    if not cands:
        raise SystemExit(f"INIT_FROM={init_name!r}: no checkpoint under {inputs} (attach that session's output)")
    src = max(cands, key=_ckpt_step)
    ck = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    info = {"init_from": src, "step": ck.get("step"), "tokens": ck.get("tokens")}
    with open(os.path.join(run_dir, "init_from.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"init: weights from {src} (step {ck.get('step')}, {ck.get('tokens') or 0:,} tokens)")
    return info


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True, help="prep output dir (contains shards/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", default="/kaggle/input", help="where earlier sessions' outputs are attached")
    ap.add_argument("--pilot", type=int, default=0, help="N steps, then report throughput")
    ap.add_argument("--max-hours", type=float, default=11.0, help="stop cleanly before the 12 h session limit")
    ap.add_argument("--ckpt-every-minutes", type=float, default=30.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--deadline", type=float, default=0,
                    help="unix time by which training must end (required for total_steps='auto')")
    ap.add_argument("--allow-any-size", action="store_true", help="skip the 50M-300M check (local smoke tests)")
    args = ap.parse_args()

    run = load_run(args.run)
    shards_dir = os.path.join(args.data, "shards")
    shards, val = TokenShards(shards_dir, "train"), TokenShards(shards_dir, "val")
    vocab = shards.meta["vocab_size"]
    torch.manual_seed(run.CFG["seed"])
    model = run.build_model(vocab)
    params = count_params(model, vocab_size=vocab)
    print(f"[{run.NAME}] parameters: " + ", ".join(f"{k}={v / 1e6:.2f}M" for k, v in params.items()))
    if not args.allow_any_size:
        check_param_range(model)

    name = run.NAME + ("_pilot" if args.pilot else "")
    run_dir = os.path.join(args.out, name)
    if not args.pilot and os.path.isdir(args.inputs):
        restore_from_inputs(name, run_dir, args.inputs)
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(args.run, os.path.join(run_dir, "run_file.py"))
    if getattr(run, "INIT_FROM", None) and not args.pilot:
        init_from_inputs(model, run.INIT_FROM, run_dir, args.inputs)
    loss_fn = getattr(run, "loss_fn", None) or (causal_lm_loss if run.KIND == "causal" else encdec_loss)
    lr_lambda = getattr(run, "lr_lambda", None)
    replanner = None
    max_hours = args.max_hours
    if run.CFG["total_steps"] == "auto":
        if args.pilot:
            run = SimpleRun(run, dict(run.CFG, total_steps=args.pilot))
            lr_lambda = run.make_lr_lambda(args.pilot)
        else:
            if not args.deadline:
                raise SystemExit("total_steps='auto' needs --deadline")
            plan = make_plan(run, model, shards, val, run_dir, args.deadline, loss_fn)
            run = SimpleRun(run, dict(run.CFG, batch_size=plan["batch_size"], grad_accum=plan["grad_accum"],
                                      total_steps=plan["total_steps"]))
            replanner = Replanner(plan, os.path.join(run_dir, "plan.json"), args.deadline)
            lr_lambda = replanner.lr_lambda(run.make_lr_lambda)
            max_hours = min(max_hours, max(0.01, (args.deadline - time.time()) / 3600))
    elif lr_lambda is None:
        lr_lambda = run.make_lr_lambda(args.pilot or run.CFG["total_steps"])
    train_batch, eval_batches, tokens_per_batch = batch_fns(run, shards, val)
    c = run.CFG
    cfg = RunConfig(out_dir=run_dir, total_steps=args.pilot or c["total_steps"], grad_accum=c["grad_accum"],
                    grad_clip=c["grad_clip"], eval_every=args.pilot or c["eval_every"], log_every=args.log_every,
                    ckpt_every_minutes=args.ckpt_every_minutes, max_hours=max_hours,
                    token_milestones=[] if args.pilot else list(c["token_milestones"]), seed=c["seed"])
    info = {"name": run.NAME, "kind": run.KIND, "params": params, "cfg": c, "run_file_sha1": file_hash(args.run),
            "vocab_size": vocab, "data": os.path.abspath(args.data), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "pilot": args.pilot,
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    res = train(model, run.make_optimizer, lr_lambda, loss_fn, train_batch, eval_batches,
                tokens_per_batch, cfg, on_step=replanner)
    print(f"[{run.NAME}] {res}")

    if args.pilot:
        log = [json.loads(line) for line in open(os.path.join(run_dir, "log.jsonl"), encoding="utf-8")]
        rates = [r["tok_per_s"] for r in log if "tok_per_s" in r][1:]
        rate = statistics.median(rates) if rates else None
        report = {"name": run.NAME, "steps": res["step"], "tokens_per_step": tokens_per_batch * c["grad_accum"],
                  "tok_per_s": rate, "tokens_per_hour": rate * 3600 if rate else None,
                  "sec_per_step": tokens_per_batch * c["grad_accum"] / rate if rate else None,
                  "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
                  "skipped_steps": sum(bool(r.get("skipped")) for r in log), "params": params, "gpu": info["gpu"]}
        with open(os.path.join(run_dir, "pilot.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        shutil.rmtree(os.path.join(run_dir, "ckpt"), ignore_errors=True)
        print("PILOT " + json.dumps(report))


if __name__ == "__main__":
    main()
