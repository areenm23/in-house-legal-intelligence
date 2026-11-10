"""Train / evaluate / predict one legal-statute-identification experiment (multi-label).

    python -m phase5.train --exp G_lorma --data DATA_ROOT --roots /kaggle/input results --out runs_p5
           [--set lr=3e-4 budget_min=20] [--max-minutes 60]

Same skeleton as phase4.train (views, time budget, periodic evaluation, OOM back-off, gradual / LLRD
plumbing), with a multi-label head (one logit per statute, bias initialised to the label prior), losses
bce / focal / asl, optional label-interaction layer, KD on per-label logits, and threshold strategies.

All dev use is on the *clean* dev rows (data.clean_dev: no test documents, no duplicates). Model selection
= macro-F1 at the best global threshold on a fixed clean-dev subset. Final dev numbers: `dev_fixed` (0.5,
nothing fitted), `dev_cv` (2-fold cross-fitted thresholds: honest), `dev_fit` (fitted on all clean dev:
optimistic). Test zips: submission.zip (global threshold), submission_pl.zip (per-label thresholds),
submission_g1.zip (global + at least one label per document). Predicted ids never exceed the largest id
in dev gold (the evaluator's label dimension comes from the test truth).
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
from ft.peft import AdaLoRALinear, GradualUnfreezer, RankAllocator, apply_method, peft_extra_loss, trainable_state
from phase3.train import find_tokenizer, sha12
from phase4.model import CJPEModel
from phase4.train import micro_batches, param_groups, predict, save_scores, load_scores, make_view
from . import data as D
from .experiments import get_experiment
from .score import cross_fit, fit_global, fit_thresholds, apply, metrics, per_label_table, sigmoid, write_submission

TOKENIZER_SHA = "0c9231a52bd5"


class LSIModel(CJPEModel):
    """CJPEModel with one logit per statute and an optional label-interaction layer
    z' = z + W sigmoid(z) (W = 0 at init: starts as independent binary relevance)."""

    def __init__(self, bb, n_labels, label_mix=False, **kw):
        super().__init__(bb, n_classes=n_labels, **kw)
        self.mix = nn.Linear(n_labels, n_labels, bias=False) if label_mix else None
        if self.mix is not None:
            nn.init.zeros_(self.mix.weight)

    def head(self, Dv, n_tok=None):
        z = super().head(Dv, n_tok)
        if self.mix is not None:
            z = z + self.mix(torch.sigmoid(z))
        return z

    def init_prior(self, prior):
        last = self.out if isinstance(self.out, nn.Linear) else self.out[-1]
        p = np.clip(np.asarray(prior, dtype=np.float64), 1e-4, 0.5)
        with torch.no_grad():
            last.bias.copy_(torch.tensor(np.log(p / (1 - p)), dtype=torch.float32))


def pos_weights(Y, kind, cap=30.0):
    if kind in (None, "none"):
        return None
    pos = Y.sum(0).astype(np.float64)
    neg = len(Y) - pos
    w = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    if kind == "sqrt":
        w = np.sqrt(w)
    elif kind != "inv":
        raise ValueError(kind)
    return torch.tensor(np.clip(w, 1.0, cap), dtype=torch.float32)


def ml_loss(z, Y, cfg, pw=None):
    """Per-document loss (summed over labels) -> (n,)."""
    kind = cfg["loss"]
    if kind == "asl":
        gn, gp, clip = cfg["asl_gamma_neg"], cfg["asl_gamma_pos"], cfg["asl_clip"]
        p = torch.sigmoid(z)
        pm = (p - clip).clamp(min=0) if clip else p
        lp = Y * (1 - p) ** gp * torch.log(p.clamp(min=1e-8))
        ln = (1 - Y) * pm ** gn * torch.log((1 - pm).clamp(min=1e-8))
        return -(lp + ln).sum(1)
    bce = F.binary_cross_entropy_with_logits(z, Y, pos_weight=pw, reduction="none")
    if kind == "focal":
        p = torch.sigmoid(z)
        pt = Y * p + (1 - Y) * (1 - p)
        bce = bce * (1 - pt) ** cfg["focal"]
    elif kind != "bce":
        raise ValueError(kind)
    return bce.sum(1)


def kd_terms(z, t, T):
    """Per-document T^2 * sum_l BCE(sigmoid(z/T), sigmoid(t/T)) minus the teacher's entropy (= binary KL)."""
    pt = torch.sigmoid(t / T)
    kl = F.binary_cross_entropy_with_logits(z / T, pt, reduction="none") - \
        F.binary_cross_entropy_with_logits(t / T, pt, reduction="none")
    return kl.sum(1) * T * T


def find_run_file(out_root, run, fname):
    cands = [os.path.join(out_root, run, fname)]
    cands += sorted(glob.glob(os.path.join("/kaggle/input", "**", "runs_p5", run, fname), recursive=True))
    return next((c for c in cands if os.path.exists(c)), None)


def confusions(Y, Dm, top=15):
    """Most-missed / most-spurious labels and (missed gold, predicted instead) pairs within documents."""
    Y, Dm = np.asarray(Y, bool), np.asarray(Dm, bool)
    fn, fp = Y & ~Dm, ~Y & Dm
    pairs = {}
    for i in np.where(fn.any(1) & fp.any(1))[0]:
        for a in np.where(fn[i])[0]:
            for b in np.where(fp[i])[0]:
                pairs[(int(a), int(b))] = pairs.get((int(a), int(b)), 0) + 1
    mm = np.argsort(-fn.sum(0))[:top]
    sp = np.argsort(-fp.sum(0))[:top]
    return {"most_missed": [{"label": int(l), "fn": int(fn[:, l].sum()), "support": int(Y[:, l].sum())} for l in mm],
            "most_spurious": [{"label": int(l), "fp": int(fp[:, l].sum())} for l in sp],
            "missed_vs_predicted": [{"gold": a, "pred": b, "docs": c}
                                    for (a, b), c in sorted(pairs.items(), key=lambda kv: -kv[1])[:top]]}


@torch.no_grad()
def extract_chunks(model, split, idx, cfg, device, max_chunks, path):
    """Chunk vectors of whole documents (<= max_chunks, spread) + label lists -> npz for phase5.hier."""
    was = model.training
    model.eval()
    L = cfg["seq_len"]
    vecs, pos, off, buf = [], [], [0], []

    def flush():
        if buf:
            vecs.append(model.encode(buf, device, cfg["amp"], cfg["eval_batch"]).float().cpu().numpy().astype(np.float16))
            buf.clear()

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
    labs = [split.labels(i) for i in idx]
    lab_off = np.zeros(len(idx) + 1, dtype=np.int64)
    lab_off[1:] = np.cumsum([len(x) for x in labs])
    np.savez(path, vecs=np.concatenate(vecs) if vecs else np.zeros((0, model.d_chunk), np.float16),
             off=np.array(off, dtype=np.int64), pos=np.array(pos, dtype=np.float32),
             ids=np.array([split.ids[i] for i in idx]), n_tok=split.n_tok()[list(idx)], lab_off=lab_off,
             lab=np.concatenate(labs) if labs else np.zeros(0, np.int16))
    return off[-1]


def final_scores(Yd, lg, n_labels, train_counts, buckets, seed=0):
    """Every threshold strategy on clean dev: fixed / cross-fitted (honest) / fitted (optimistic)."""
    Pr = sigmoid(lg)
    th = fit_thresholds(Yd, Pr)
    res = {"dev_fixed": metrics(Yd, apply("fixed", 0.5, Pr)), "dev_cv": cross_fit(Yd, Pr, seed),
           "dev_fit": {s: metrics(Yd, apply(s, t, Pr)) for s, t in th.items()},
           "thresholds": {"global": th["global"], "global_atleast1": th["global_atleast1"],
                          "per_label": [round(float(x), 3) for x in th["per_label"]]}}
    Dg = apply("global", th["global"], Pr)
    res["per_label"], res["by_bucket"] = per_label_table(Yd, Dg, train_counts, buckets)
    res["confusions"] = confusions(Yd, Dg)
    return res, th, Pr


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
    tr, dv, te, NL = data["train"], data["dev"], data["test"], data["n_labels"]
    if dv is None or not dv.labelled:
        raise FileNotFoundError("Phase 5 needs a labelled dev.json")
    dc = data["dev_clean"]
    max_id = int(max(int(dv.labels(i).max()) for i in dc if len(dv.labels(i)))) if len(dc) else NL - 1
    tr_idx = D.random_subset(len(tr), cfg["max_train_docs"], seed=cfg["subset_seed"])
    ev_idx = dc[D.random_subset(len(dc), cfg["eval_docs"], seed=0)]
    Ytr = tr.Y(NL)
    train_counts = Ytr[tr_idx].sum(0)
    buckets = D.freq_buckets(train_counts)
    Yev, Ydc = dv.Y(NL, ev_idx), dv.Y(NL, dc)

    teacher = None
    if cfg.get("kd"):
        src = find_run_file(out_root, cfg["kd"]["teacher"], "soft_labels.npz")
        if src is None:
            raise FileNotFoundError(f"KD teacher soft labels not found for {cfg['kd']['teacher']}")
        t = load_scores(src)
        tmap = {i: k for k, i in enumerate(t["ids"])}
        n0 = len(tr_idx)
        tr_idx = np.array([i for i in tr_idx if tr.ids[i] in tmap], dtype=np.int64)
        if t["logits"].shape[1] != NL or len(tr_idx) < 0.5 * n0:
            raise RuntimeError(f"KD teacher {src} does not match ({t['logits'].shape}, {len(tr_idx)}/{n0} docs)")
        teacher = {"map": tmap, "logits": t["logits"], "vecs": t.get("vecs"), "src": src}
        print(f"KD teacher {src}: {len(tr_idx)}/{n0} training docs", flush=True)

    bb = load_backbone(cfg["backbone"], roots, random_init=cfg.get("random_init", False), tiny=cfg.get("tiny", False))
    if cfg["bidir"]:
        bb.set_bidirectional(True)
    bb.to(device)
    info = apply_method(bb, cfg["method"], **dict(cfg.get("method_args", {})))
    model = LSIModel(bb, NL, cfg["label_mix"], pool=cfg["pool"], agg=cfg["agg"], head=cfg["head"],
                     dropout=cfg["dropout"], hidden=cfg["head_hidden"], pos_feat=cfg["pos_feat"],
                     len_feat=cfg["len_feat"], echo=cfg["echo"]).to(device)
    model.init_prior(train_counts / max(1, len(tr_idx)))
    feat_w = (cfg.get("kd") or {}).get("feat", 0.0)
    if teacher is not None and feat_w:
        model.kd_proj = nn.Linear(model.d_doc, teacher["vecs"].shape[1]).to(device)
    unfreezer = GradualUnfreezer(bb, cfg["unfreeze_frac"], cfg.get("unfreeze_max")) if cfg["method"] == "gradual" else None
    opt = torch.optim.AdamW(param_groups(model, cfg, unfreezer.params() if unfreezer else ()), betas=(0.9, 0.98), eps=1e-6)
    train_names = {n for n, p in model.named_parameters()
                   if p.requires_grad or (unfreezer and any(p is q for q in unfreezer.params()))}
    n_total = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("kd_proj"))
    n_train = sum(p.numel() for n, p in model.named_parameters() if n in train_names and not n.startswith("kd_proj"))
    print(f"[{exp}] backbone {bb.name} ({bb.ckpt_path}) method {info} -> trainable {n_train:,} / {n_total:,}; "
          f"{NL} labels (predict ids <= {max_id}); ctx {cfg['ctx']} L={cfg['seq_len']}; train {len(tr_idx)}, "
          f"clean dev {len(dc)}/{len(dv)} (eval {len(ev_idx)}), test {len(te)}", flush=True)

    pw = pos_weights(Ytr[tr_idx], cfg["pos_weight"], cfg["pos_weight_cap"])
    pw = pw.to(device) if pw is not None else None
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
        json.dump({**cfg, "peft_info": info, "tokenizer": tok_path, "backbone_ckpt": bb.ckpt_path, "n_labels": NL,
                   "max_pred_id": max_id, "train_docs": len(tr_idx), "dev_clean": len(dc), "dev_rows": len(dv),
                   "eval_docs": len(ev_idx), "test_docs": len(te)}, f, indent=1)
    logf = open(os.path.join(out, "log.jsonl"), "a")

    def log(rec):
        rec["t"] = round(time.time() - t0, 1)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    budget_s = cfg["budget_min"] * 60 if cfg.get("budget_min") else None
    ex = cfg.get("extract") or {}
    final_docs = len(dc) + len(te) + (len(tr_idx) if cfg.get("save_soft") else 0)
    reserve = {"s": cfg["reserve_min"] * 60}

    def update_reserve(n_seq, secs):
        per_doc = cfg["eval_chunks"] if cfg["ctx"] == "hier" else 1
        n_ex = (min(ex.get("train_docs") or len(tr_idx), len(tr_idx)) + len(dc) + len(te)) * min(6, ex.get("max_chunks", 32)) if ex else 0
        reserve["s"] = (final_docs * per_doc + n_ex) / max(n_seq / max(secs, 1e-6), 1e-6) * 1.2 + 60

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
            Yb = torch.as_tensor(Ytr[batch].astype(np.float32), device=device)
            has_t = np.array([tr.ids[i] in teacher["map"] for i in batch]) if teacher is not None else None
            n_t = max(1, int(has_t.sum())) if has_t is not None else 1
            while True:
                opt.zero_grad(set_to_none=True)
                tot = 0.0
                try:
                    for mb in micro_batches(vs, seq_batch):
                        z, dvec = model([vs[j] for j in mb], device, cfg["amp"], seq_batch, nt[batch[mb]])
                        loss = ml_loss(z, Yb[mb], cfg, pw).sum() / len(batch)
                        if teacher is not None:
                            a = cfg["kd"].get("alpha", 0.5)
                            m = torch.as_tensor(has_t[mb], device=device)
                            rows = [teacher["map"].get(tr.ids[batch[j]], 0) for j in mb]
                            tl = torch.as_tensor(teacher["logits"][rows], device=device)
                            loss = a * loss + (1 - a) * (kd_terms(z, tl, cfg["kd"].get("T", 2.0)) * m).sum() / n_t
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
                     "tok_s": round(tokens / max(t_train, 1e-6)), "seq_batch": seq_batch})
            last = step >= plan["total"]
            if step >= next_eval or last:
                lg, _v, n_seq, secs = predict(model, dv, ev_idx, cfg, device)
                update_reserve(n_seq, secs)
                Pr = sigmoid(lg)
                tg, f_g = fit_global(Yev, Pr)
                f_05 = metrics(Yev, apply("fixed", 0.5, Pr))["macro_f1"]
                rec = {"step": step, "epoch": round(step * B / max(1, len(tr_idx)), 3), "eval_f1": round(f_g, 4),
                       "eval_thr": tg, "eval_f1_05": round(f_05, 4), "eval_s": round(secs, 1), "plan": plan["total"]}
                if f_g > best["f1"]:
                    best = {"f1": f_g, "step": step,
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
    print(f"[{exp}] training stopped ({stop_reason}) at step {step}; best eval macro-F1 {best['f1']:.4f} at step "
          f"{best['step']}", flush=True)

    result = {"exp": exp, "backbone": cfg["backbone"], "method": cfg["method"], "ctx": cfg["ctx"],
              "seq_len": cfg["seq_len"], "n_chunks": cfg["n_chunks"] if cfg["ctx"] == "hier" else 1,
              "pool": cfg["pool"], "agg": cfg["agg"], "loss": cfg["loss"], "pos_weight": cfg["pos_weight"],
              "label_mix": cfg["label_mix"], "kd": cfg.get("kd"), "n_labels": NL, "max_pred_id": max_id,
              "params_total": n_total, "params_trainable": n_train, "peft_info": info,
              "train_docs": len(tr_idx), "steps": step, "epochs_seen": round(step * B / max(1, len(tr_idx)), 3),
              "stop": stop_reason, "best_step": best["step"], "best_eval_f1": round(best["f1"], 6),
              "train_tok_s": round(tokens / max(t_train, 1e-6), 1), "train_minutes": round(t_train / 60, 2),
              "peak_mem_gib": peak, "dev_clean_docs": len(dc), "dev_rows": len(dv)}

    def save_result():
        result["minutes_total"] = round((time.time() - t0) / 60, 2)
        with open(os.path.join(out, "result.json"), "w") as f:
            json.dump(result, f, indent=1)

    lg, _v, _n, secs = predict(model, dv, dc, cfg, device)
    result["infer_docs_s"] = round(len(dc) / max(secs, 1e-6), 2)
    fs, th, Pr = final_scores(Ydc, lg, NL, train_counts, buckets, seed)
    result.update(fs)
    with open(os.path.join(out, "dev_pred.json"), "w", encoding="utf-8") as f:
        Dg = apply("global", th["global"], Pr)
        json.dump([{"id": dv.ids[i], "gold": [int(x) for x in dv.labels(i)], "pred": [int(x) for x in np.where(r)[0]]}
                   for i, r in zip(dc, Dg)], f)
    save_scores(os.path.join(out, "scores_dev.npz"), [dv.ids[i] for i in dc], lg)
    save_result()
    if cfg.get("save_soft"):
        lg_t, vec_t, _n, _s = predict(model, tr, tr_idx, cfg, device)
        save_scores(os.path.join(out, "soft_labels.npz"), [tr.ids[i] for i in tr_idx], lg_t, vec_t)
        result["soft_labels"] = len(tr_idx)
    sw = cfg.get("save_weights", "auto")
    if sw is True or (sw == "auto" and sum(p.numel() for p in model.parameters() if p.requires_grad) < 20e6):
        torch.save({"trainable": {k: v.half() for k, v in trainable_state(model).items()}, "config": cfg,
                    "peft_info": info}, os.path.join(out, "adapter.pt"))
    if cfg["predict_test"]:
        lg_te, _v, _n, _s = predict(model, te, np.arange(len(te)), cfg, device)
        Pt = sigmoid(lg_te)
        result["test"] = {"global": write_submission(out, te.ids, apply("global", th["global"], Pt), max_id),
                          "per_label": write_submission(out, te.ids, apply("per_label", th["per_label"], Pt), max_id,
                                                        "submission_pl.zip"),
                          "global_atleast1": write_submission(out, te.ids, apply("global_atleast1", th["global_atleast1"], Pt),
                                                              max_id, "submission_g1.zip")}
        save_scores(os.path.join(out, "scores_test.npz"), te.ids, lg_te)
    save_result()
    if ex:
        t1 = time.time()
        n_ex = min(ex.get("train_docs") or len(tr_idx), len(tr_idx))
        sub = tr_idx[D.random_subset(len(tr_idx), n_ex, seed=1)]
        mc = ex.get("max_chunks", 32)
        result["chunks"] = {
            "train": extract_chunks(model, tr, sub, cfg, device, mc, os.path.join(out, "chunks_train.npz")),
            "dev": extract_chunks(model, dv, dc, cfg, device, mc, os.path.join(out, "chunks_dev.npz")),
            "test": extract_chunks(model, te, np.arange(len(te)), cfg, device, mc, os.path.join(out, "chunks_test.npz")),
            "minutes": round((time.time() - t1) / 60, 2), "max_chunks": mc}
        save_result()
    cv = result["dev_cv"]
    print(f"[{exp}] done in {result['minutes_total']} min: clean-dev macro-F1 fixed {result['dev_fixed']['macro_f1']:.4f}, "
          f"cv global {cv['global']['macro_f1']:.4f}, cv per-label {cv['per_label']['macro_f1']:.4f} "
          f"(micro {cv['global']['micro_f1']:.4f})", flush=True)
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
    ap.add_argument("--data", required=True, help="Phase_5_LSI folder (or its parent)")
    ap.add_argument("--roots", nargs="+", default=["results"])
    ap.add_argument("--out", default="runs_p5")
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args(argv)
    root = D.find_root(a.data)
    if root is None:
        sys.exit(f"no Phase_5_LSI data under {a.data}")
    run(get_experiment(a.exp, parse_set(a.set)), root, a.roots, a.out, a.max_minutes)


if __name__ == "__main__":
    main()
