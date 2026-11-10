"""Train / evaluate / predict one court-judgment-prediction experiment.

    python -m phase4.train --exp G_lorma --data DATA_ROOT --roots /kaggle/input results --out runs_p4
           [--set lr=3e-4 budget_min=20] [--max-minutes 60]

Training docs = train.json (optionally a class-stratified subset of max_train_docs). Model selection on
a fixed stratified subset of dev (eval_docs, the same for every run); final scores on the whole dev set;
predictions for test. Training is time-budgeted: after `probe_steps` optimiser steps the run measures
its speed and shrinks the step plan (and the LR schedule with it) to fit `budget_min` minutes of
training; with --max-minutes (session deadline) it also stops early enough to finish its outputs.

Outputs in <out>/<exp>/: config.json, log.jsonl, result.json (dev macro-F1 / P / R / accuracy at argmax
and at the dev-tuned threshold (optimistic), confusion, F1 by length quartile, most confident FP/FN,
params, memory, speed), dev_pred.json, submission.zip (argmax) and submission_thr.zip (dev-tuned
threshold), scores_{dev,test}.npz (logits by id, for ensembles), adapter.pt (head + trainable weights,
small methods only), with save_soft soft_labels.npz (logits + document vectors on the training docs: the
KD teacher signal), and with extract chunks_{train,dev,test}.npz (chunk vectors over whole documents,
input of phase4.hier).
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ft.backbones import load_backbone
from ft.peft import AdaLoRALinear, GradualUnfreezer, RankAllocator, apply_method, layer_depth, peft_extra_loss, \
    trainable_state
from phase3.train import find_tokenizer, sha12
from . import data as D
from .experiments import get_experiment
from .model import CJPEModel
from .score import breakdown, fit_threshold, prf, probs, write_submission

TOKENIZER_SHA = "0c9231a52bd5"


def class_weights(y, kind):
    if kind in (None, "none"):
        return None
    cnt = np.bincount(np.asarray(y, dtype=np.int64), minlength=2).astype(np.float64)
    w = cnt.sum() / (2 * np.maximum(cnt, 1))
    if kind == "sqrt":
        w = np.sqrt(w)
    elif kind != "balanced":
        raise ValueError(kind)
    return torch.tensor(w, dtype=torch.float32)


def param_groups(model, cfg, extra=()):
    """AdamW groups: backbone (decay / no decay, per layer when llrd is set: lr * llrd ** (top - depth)),
    head (+ KD projection). `extra`: backbone params that are frozen now but will train (gradual)."""
    bb, decay = model.bb, cfg.get("llrd")
    top = len(bb.layers()) + 1
    extra = {id(p) for p in extra}
    groups = {}
    for n, p in model.named_parameters():
        if not (p.requires_grad or id(p) in extra):
            continue
        if not n.startswith("bb."):
            key = ("head", 0)
        else:
            depth = layer_depth(bb, n[3:]) if decay else top
            key = ("nodecay" if (p.ndim < 2 or "prompt" in n) else "decay", depth)
        groups.setdefault(key, []).append(p)
    out = []
    for (kind, depth), ps in sorted(groups.items()):
        if kind == "head":
            out.append({"params": ps, "lr": cfg["head_lr"], "weight_decay": 0.01})
        else:
            s = decay ** (top - depth) if decay else 1.0
            out.append({"params": ps, "lr": cfg["lr"] * s, "weight_decay": cfg["weight_decay"] if kind == "decay" else 0.0})
    return out


def kd_terms(s_logits, t_logits, T):
    """Per-document T^2 KL(softmax(t/T) || softmax(s/T)) -> (n,)."""
    pt = F.softmax(t_logits / T, -1)
    return (pt * (F.log_softmax(t_logits / T, -1) - F.log_softmax(s_logits / T, -1))).sum(-1) * T * T


def find_run_file(out_root, run, fname):
    cands = [os.path.join(out_root, run, fname)]
    cands += sorted(glob.glob(os.path.join("/kaggle/input", "**", "runs_p4", run, fname), recursive=True))
    return next((c for c in cands if os.path.exists(c)), None)


def micro_batches(views, limit):
    """Group consecutive documents so each group has at most `limit` sequences (at least one document)."""
    out, cur, n = [], [], 0
    for j, (seqs, _pos) in enumerate(views):
        if cur and n + len(seqs) > limit:
            out.append(cur)
            cur, n = [], 0
        cur.append(j)
        n += len(seqs)
    if cur:
        out.append(cur)
    return out


def make_view(split, i, cfg, train, rng=None):
    k = cfg["n_chunks"] if train else cfg["eval_chunks"]
    pick = cfg["chunk_pick"] if train else cfg["eval_pick"]
    return D.views(split.doc(i), cfg["ctx"], cfg["seq_len"], cfg["head_tokens"], k, pick, cfg.get("stride"), rng)


@torch.no_grad()
def predict(model, split, idx, cfg, device):
    """-> (logits (n, 2) float32, document vectors (n, d_doc) float16, sequences encoded, seconds)."""
    was = model.training
    model.eval()
    t0 = time.time()
    logits, vecs, n_seq = [], [], 0
    idx = list(idx)
    step = max(1, cfg["eval_batch"] // max(1, cfg["eval_chunks"] if cfg["ctx"] == "hier" else 1))
    nt = split.n_tok()
    for j in range(0, len(idx), step):
        ids = idx[j:j + step]
        vs = [make_view(split, i, cfg, False) for i in ids]
        n_seq += sum(len(s) for s, _ in vs)
        lg, dv = model(vs, device, cfg["amp"], cfg["eval_batch"], nt[ids])
        logits.append(lg.cpu().numpy())
        vecs.append(dv.float().cpu().numpy().astype(np.float16))
    model.train(was)
    d = model.d_doc
    return (np.concatenate(logits) if logits else np.zeros((0, 2), np.float32),
            np.concatenate(vecs) if vecs else np.zeros((0, d), np.float16), n_seq, time.time() - t0)


@torch.no_grad()
def extract_chunks(model, split, idx, cfg, device, max_chunks, path):
    """Chunk vectors (pooled backbone output, before the position embedding) of whole documents, at most
    max_chunks per document (spread evenly when there are more) -> npz(vecs, off, pos, ids, y, n_tok)."""
    was = model.training
    model.eval()
    L = cfg["seq_len"]
    vecs, pos, off = [], [], [0]
    buf, buf_n = [], 0

    def flush():
        nonlocal buf, buf_n
        if buf:
            vecs.append(model.encode(buf, device, cfg["amp"], cfg["eval_batch"]).float().cpu().numpy().astype(np.float16))
        buf, buf_n = [], 0

    nt = split.n_tok()
    for i in idx:
        t = split.doc(i).astype(np.int64)
        st = D.chunk_starts(len(t), L, cfg.get("stride"))
        sel = D.pick_chunks(len(st), max_chunks, "spread")
        for s in st[sel]:
            q = t[s:s + L]
            buf.append(q)
            pos.append((s + len(q) / 2) / max(1, len(t)))
        off.append(off[-1] + len(sel))
        if len(buf) >= 4 * cfg["eval_batch"]:
            flush()
    flush()
    model.train(was)
    np.savez(path, vecs=np.concatenate(vecs) if vecs else np.zeros((0, model.d_chunk), np.float16),
             off=np.array(off, dtype=np.int64), pos=np.array(pos, dtype=np.float32),
             ids=np.array([split.ids[i] for i in idx]), y=split.y[list(idx)], n_tok=nt[list(idx)])
    return off[-1]


def save_scores(path, ids, logits, vecs=None):
    kw = {"ids": np.array(ids), "logits": np.asarray(logits, dtype=np.float32)}
    if vecs is not None:
        kw["vecs"] = np.asarray(vecs, dtype=np.float16)
    np.savez_compressed(path, **kw)


def load_scores(path):
    z = np.load(path)
    out = {"ids": [str(i) for i in z["ids"].tolist()], "logits": z["logits"].astype(np.float32)}
    if "vecs" in z.files:
        out["vecs"] = z["vecs"].astype(np.float32)
    return out


def run(cfg, data_root, roots, out_root, max_minutes=None, device=None):
    t0 = time.time()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(cfg)
    cfg["amp"] = cfg.get("amp", True) and device.type == "cuda"
    exp = cfg["name"]
    out = os.path.join(out_root, exp)
    os.makedirs(out, exist_ok=True)
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    tok_path = cfg.get("tokenizer") or find_tokenizer(roots)
    if sha12(tok_path) != TOKENIZER_SHA:
        print(f"WARNING: tokenizer {sha12(tok_path)} != Phase-1 {TOKENIZER_SHA}")
    data = D.load_data(data_root, tok_path, os.path.join(out_root, "_cache"))
    tr, dv, te = data["train"], data["dev"], data["test"]
    if dv is None or not dv.labelled:
        hold = D.stratified_subset(tr.y, max(1, len(tr) // 10), seed=12345)
        rest = np.setdiff1d(np.arange(len(tr)), hold)
        tr, dv = D.take(tr, rest, "train"), D.take(tr, hold, "dev")
        print(f"no labelled dev split: held out {len(dv)} training documents")
    tr_idx = D.stratified_subset(tr.y, cfg["max_train_docs"], seed=cfg["subset_seed"])
    ev_idx = D.stratified_subset(dv.y, cfg["eval_docs"], seed=0)

    teacher = None
    if cfg.get("kd"):
        kd = cfg["kd"]
        src = find_run_file(out_root, kd["teacher"], "soft_labels.npz")
        if src is None:
            raise FileNotFoundError(f"KD teacher soft labels not found for {kd['teacher']}")
        t = load_scores(src)
        tmap = {i: k for k, i in enumerate(t["ids"])}
        n0 = len(tr_idx)
        tr_idx = np.array([i for i in tr_idx if tr.ids[i] in tmap], dtype=np.int64)
        teacher = {"map": tmap, "logits": t["logits"], "vecs": t.get("vecs"), "src": src}
        print(f"KD teacher {src}: {len(tr_idx)}/{n0} training docs have teacher logits")
        if len(tr_idx) < 0.5 * n0:
            raise RuntimeError("KD teacher covers fewer than half of the training documents")

    bb = load_backbone(cfg["backbone"], roots, random_init=cfg.get("random_init", False), tiny=cfg.get("tiny", False))
    if cfg["bidir"]:
        bb.set_bidirectional(True)
    bb.to(device)
    info = apply_method(bb, cfg["method"], **dict(cfg.get("method_args", {})))
    model = CJPEModel(bb, cfg["pool"], cfg["agg"], cfg["head"], cfg["dropout"], cfg["head_hidden"], cfg["pos_feat"],
                      cfg["len_feat"], cfg["echo"]).to(device)
    feat_w = (cfg.get("kd") or {}).get("feat", 0.0)
    if teacher is not None and feat_w:
        if teacher["vecs"] is None:
            raise ValueError("teacher soft_labels.npz has no document vectors (needed for kd.feat)")
        model.kd_proj = nn.Linear(model.d_doc, teacher["vecs"].shape[1]).to(device)
    unfreezer = GradualUnfreezer(bb, cfg["unfreeze_frac"], cfg.get("unfreeze_max")) if cfg["method"] == "gradual" else None
    opt = torch.optim.AdamW(param_groups(model, cfg, unfreezer.params() if unfreezer else ()), betas=(0.9, 0.98),
                            eps=1e-6)
    train_names = {n for n, p in model.named_parameters()
                   if p.requires_grad or (unfreezer and any(p is q for q in unfreezer.params()))}
    n_total = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("kd_proj"))
    n_train_final = sum(p.numel() for n, p in model.named_parameters() if n in train_names and not n.startswith("kd_proj"))
    print(f"[{exp}] backbone {bb.name} ({bb.ckpt_path}) method {info} -> trainable {n_train_final:,} / {n_total:,}; "
          f"ctx {cfg['ctx']} L={cfg['seq_len']}; train docs {len(tr_idx)} (pos rate {tr.y[tr_idx].mean():.3f}), "
          f"dev {len(dv)} (eval subset {len(ev_idx)}), test {len(te)}", flush=True)

    cw = class_weights(tr.y[tr_idx], cfg["class_weight"])
    cw = cw.to(device) if cw is not None else None
    B = cfg["batch"]
    steps_per_epoch = math.ceil(len(tr_idx) / B)
    plan = {"total": max(1, steps_per_epoch * cfg["epochs"])}
    warm = lambda: max(1, int(cfg["warmup"] * plan["total"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm() if s < warm() else max(0.0, (plan["total"] - s) / max(1, plan["total"] - warm())))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    ada = [m for m in model.modules() if isinstance(m, AdaLoRALinear)]
    allocator = RankAllocator(ada, info.get("target_r", 4), plan["total"]) if ada else None
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**cfg, "peft_info": info, "tokenizer": tok_path, "backbone_ckpt": bb.ckpt_path,
                   "train_docs": len(tr_idx), "dev_docs": len(dv), "eval_docs": len(ev_idx), "test_docs": len(te)},
                  f, indent=1)
    logf = open(os.path.join(out, "log.jsonl"), "a")

    def log(rec):
        rec["t"] = round(time.time() - t0, 1)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    budget_s = cfg["budget_min"] * 60 if cfg.get("budget_min") else None
    final_docs = len(dv) + len(te) + (len(tr_idx) if cfg.get("save_soft") else 0)
    ex = cfg.get("extract") or {}
    reserve = {"s": cfg["reserve_min"] * 60, "seq_s": None}

    def update_reserve(n_seq, secs):
        reserve["seq_s"] = n_seq / max(secs, 1e-6)
        per_doc = cfg["eval_chunks"] if cfg["ctx"] == "hier" else 1
        n_ex = 0
        if ex:
            n_ex = (min(ex.get("train_docs") or len(tr_idx), len(tr_idx)) + len(dv) + len(te)) * min(10, ex.get("max_chunks", 48))
        reserve["s"] = (final_docs * per_doc + n_ex) / reserve["seq_s"] * 1.2 + 60

    best = {"f1": -1.0, "step": 0, "state": None}
    bad = step = tokens = 0
    t_train = 0.0
    times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    seq_batch = cfg["seq_batch"]
    next_eval = max(1, plan["total"] // cfg["n_evals"])
    stop_reason = None
    nt = tr.n_tok()
    model.train()
    ep = 0
    while stop_reason is None and ep < cfg["epochs"]:
        ep += 1
        perm = rng.permutation(tr_idx)
        for b0 in range(0, len(perm), B):
            if step >= plan["total"]:
                stop_reason = "plan"
                break
            batch = perm[b0:b0 + B]
            if unfreezer is not None:
                unfreezer.step(step, plan["total"])
            ts = time.time()
            vs = [make_view(tr, i, cfg, True, rng) for i in batch]
            y_all = torch.as_tensor(tr.y[batch].astype(np.int64), device=device)
            W = float(cw[y_all].sum()) if cw is not None else float(len(batch))
            has_t = np.array([tr.ids[i] in teacher["map"] for i in batch]) if teacher is not None else None
            n_t = max(1, int(has_t.sum())) if has_t is not None else 1
            while True:
                opt.zero_grad(set_to_none=True)
                tot = 0.0
                try:
                    for mb in micro_batches(vs, seq_batch):
                        lg, dvec = model([vs[j] for j in mb], device, cfg["amp"], seq_batch, nt[batch[mb]])
                        y = y_all[mb]
                        ce = F.cross_entropy(lg, y, weight=cw, reduction="none", label_smoothing=cfg["smoothing"])
                        if cfg["focal"]:
                            pt = torch.softmax(lg, -1).gather(1, y[:, None]).squeeze(1)
                            ce = ce * (1 - pt) ** cfg["focal"]
                        loss = ce.sum() / W
                        if teacher is not None:
                            a = cfg["kd"].get("alpha", 0.5)
                            m = torch.as_tensor(has_t[mb], device=device)
                            rows = [teacher["map"].get(tr.ids[batch[j]], 0) for j in mb]
                            tl = torch.as_tensor(teacher["logits"][rows], device=device)
                            kd_l = (kd_terms(lg, tl, cfg["kd"].get("T", 2.0)) * m).sum() / n_t
                            loss = a * loss + (1 - a) * kd_l
                            if feat_w:
                                tv = torch.as_tensor(teacher["vecs"][rows], device=device)
                                cos = F.cosine_similarity(model.kd_proj(dvec.float()), tv, dim=-1)
                                loss = loss + feat_w * ((1 - cos) * m).sum() / n_t
                        reg = peft_extra_loss(model.bb)
                        if reg is not None:
                            loss = loss + reg * len(mb) / len(batch)
                        scaler.scale(loss).backward()
                        tot += float(loss.detach())
                    break
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if seq_batch <= 1:
                        raise
                    seq_batch = max(1, seq_batch // 2)
                    print(f"OOM -> seq_batch {seq_batch}", flush=True)
            scaler.unscale_(opt)
            if allocator is not None:
                allocator.step(step + 1)
            if cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            tokens += sum(len(s) for seqs, _ in vs for s in seqs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - ts
            t_train += dt
            times.append(dt)
            if budget_s and (step == cfg["probe_steps"] or step == next_eval):
                recent = times[-50:] if step > cfg["probe_steps"] else times[min(5, len(times) - 1):]
                sps = float(np.mean(recent))
                fit = step + int(max(0.0, budget_s - t_train) / max(sps, 1e-6))
                if fit < plan["total"]:
                    plan["total"] = max(step + 1, fit)
                    next_eval = min(next_eval, step + max(1, (plan["total"] - step) // max(1, cfg["n_evals"])))
                    if allocator is not None:
                        allocator.retarget(plan["total"])
                    print(f"plan: {sps:.2f} s/step -> {plan['total']} steps (budget {cfg['budget_min']} min, "
                          f"{plan['total'] * B / max(1, len(tr_idx)):.2f} epochs)", flush=True)
            if step % cfg["log_every"] == 0:
                log({"step": step, "epoch": ep, "loss": round(tot, 5), "lr": sched.get_last_lr()[0],
                     "tok_s": round(tokens / max(t_train, 1e-6)), "seq_batch": seq_batch,
                     "open_groups": unfreezer.open if unfreezer else None})
            last = step >= plan["total"]
            if step >= next_eval or last:
                lg, _v, n_seq, secs = predict(model, dv, ev_idx, cfg, device)
                update_reserve(n_seq, secs)
                pred = lg.argmax(1)
                s = prf(dv.y[ev_idx], pred)
                rec = {"step": step, "epoch": round(step * B / max(1, len(tr_idx)), 3), "eval_f1": round(s["macro_f1"], 4),
                       "eval_acc": round(s["accuracy"], 4), "eval_pos_rate": round(s["pred_pos_rate"], 3),
                       "eval_s": round(secs, 1), "plan": plan["total"]}
                if s["macro_f1"] > best["f1"]:
                    best = {"f1": s["macro_f1"], "step": step,
                            "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in train_names}}
                    bad = 0
                else:
                    bad += 1
                log(rec)
                next_eval = step + max(1, plan["total"] // cfg["n_evals"])
                if cfg["patience"] and bad >= cfg["patience"]:
                    stop_reason = "patience"
                    break
            if max_minutes and time.time() - t0 > max_minutes * 60 - reserve["s"]:
                stop_reason = "deadline"
                break
    stop_reason = stop_reason or "epochs"
    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=False)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None
    print(f"[{exp}] training stopped ({stop_reason}) at step {step}; best eval F1 {best['f1']:.4f} at step "
          f"{best['step']}", flush=True)

    result = {"exp": exp, "backbone": cfg["backbone"], "method": cfg["method"], "ctx": cfg["ctx"],
              "seq_len": cfg["seq_len"], "n_chunks": cfg["n_chunks"] if cfg["ctx"] == "hier" else 1,
              "pool": cfg["pool"], "agg": cfg["agg"], "kd": cfg.get("kd"),
              "params_total": n_total, "params_trainable": n_train_final, "peft_info": info,
              "train_docs": len(tr_idx), "steps": step, "epochs_seen": round(step * B / max(1, len(tr_idx)), 3),
              "stop": stop_reason, "best_step": best["step"], "best_eval_f1": round(best["f1"], 6),
              "train_tok_s": round(tokens / max(t_train, 1e-6), 1), "train_minutes": round(t_train / 60, 2),
              "peak_mem_gib": peak}

    def save_result():
        result["minutes_total"] = round((time.time() - t0) / 60, 2)
        with open(os.path.join(out, "result.json"), "w") as f:
            json.dump(result, f, indent=1)

    all_dev = np.arange(len(dv))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    lg, vec, n_seq, secs = predict(model, dv, all_dev, cfg, device)
    result["infer_docs_s"] = round(len(dv) / max(secs, 1e-6), 2)
    result["infer_peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None
    p1 = probs(lg)
    pred = lg.argmax(1)
    result["dev"] = prf(dv.y, pred)
    thr, _ = fit_threshold(p1, dv.y)
    result["dev_thr"] = {"threshold": thr, **prf(dv.y, (p1 >= thr).astype(np.int64))}
    result["errors"] = breakdown(dv.y, pred, p1, dv.n_tok(), dv.ids)
    with open(os.path.join(out, "dev_pred.json"), "w", encoding="utf-8") as f:
        json.dump([{"id": i, "gold": int(g), "pred": int(p), "p1": round(float(q), 5)}
                   for i, g, p, q in zip(dv.ids, dv.y, pred, p1)], f)
    save_scores(os.path.join(out, "scores_dev.npz"), dv.ids, lg)
    save_result()
    if cfg.get("save_soft"):
        lg_t, vec_t, _n, _s = predict(model, tr, tr_idx, cfg, device)
        save_scores(os.path.join(out, "soft_labels.npz"), [tr.ids[i] for i in tr_idx], lg_t, vec_t)
        result["soft_labels"] = len(tr_idx)
    sw = cfg.get("save_weights", "auto")
    n_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if sw is True or (sw == "auto" and n_now < 20e6):
        torch.save({"trainable": {k: v.half() for k, v in trainable_state(model).items()}, "config": cfg,
                    "peft_info": info}, os.path.join(out, "adapter.pt"))
    if cfg["predict_test"]:
        lg_te, _v, _n, _s = predict(model, te, np.arange(len(te)), cfg, device)
        p1_te = probs(lg_te)
        write_submission(out, te.ids, lg_te.argmax(1))
        write_submission(out, te.ids, (p1_te >= thr).astype(np.int64), "submission_thr.zip")
        save_scores(os.path.join(out, "scores_test.npz"), te.ids, lg_te)
        result["test_pos_rate"] = round(float(lg_te.argmax(1).mean()), 4)
        result["test_pos_rate_thr"] = round(float((p1_te >= thr).mean()), 4)
    save_result()
    if ex:
        t1 = time.time()
        n_ex = min(ex.get("train_docs") or len(tr_idx), len(tr_idx))
        sub = tr_idx[D.stratified_subset(tr.y[tr_idx], n_ex, seed=1)] if n_ex < len(tr_idx) else tr_idx
        mc = ex.get("max_chunks", 48)
        n_c = {"train": extract_chunks(model, tr, sub, cfg, device, mc, os.path.join(out, "chunks_train.npz")),
               "dev": extract_chunks(model, dv, all_dev, cfg, device, mc, os.path.join(out, "chunks_dev.npz")),
               "test": extract_chunks(model, te, np.arange(len(te)), cfg, device, mc, os.path.join(out, "chunks_test.npz"))}
        result["chunks"] = {**n_c, "minutes": round((time.time() - t1) / 60, 2), "max_chunks": mc}
        save_result()
    d = result["dev"]
    print(f"[{exp}] done in {result['minutes_total']} min: dev macro-F1 {d['macro_f1']:.4f} (P {d['precision']:.3f} "
          f"R {d['recall']:.3f}; thr {thr:.2f} -> {result['dev_thr']['macro_f1']:.4f})", flush=True)
    del model, bb, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--data", required=True, help="Phase_4_CJPE folder (or its parent)")
    ap.add_argument("--roots", nargs="+", default=["results"], help="where to look for checkpoints/tokenizer")
    ap.add_argument("--out", default="runs_p4")
    ap.add_argument("--set", nargs="*", help="config overrides key=value (JSON values)")
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args(argv)
    root = D.find_root(a.data)
    if root is None:
        sys.exit(f"no Phase_4_CJPE data under {a.data}")
    run(get_experiment(a.exp, parse_set(a.set)), root, a.roots, a.out, a.max_minutes)


if __name__ == "__main__":
    main()
