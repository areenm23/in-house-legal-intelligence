"""Train / evaluate / predict one L-NER experiment.

    python -m phase2.train --exp A_lorma --data DATA_ROOT --roots /kaggle/input results --out runs_p2 [--mode dev]
           [--set lr=3e-4 epochs=10] [--max-minutes 60]

Modes
  dev   train on fold1+fold2 minus a fixed 15% doc-level dev split, pick the epoch with the best dev
        strict macro-F1, then predict the test set with that model
  full  train on every labelled document for `epochs_full` epochs (default: best dev epoch of the
        matching dev run, scaled by the extra data, else `epochs`), then predict the test set
  f12 / f21  train on one fold, evaluate on the other (cross-fold check)

Outputs in <out>/<exp>[__<mode>]/: config.json, log.jsonl, result.json (dev scores per label,
error breakdown, params, memory, speed), dev_pred.json, prediction.json, submission.zip,
scores_{dev,test}.npz (per-token log-probs, for ensembling), head + trainable weights (adapter.pt)
and, with save_soft, soft_labels.npz (teacher log-probs on every labelled document, for KD).
"""
import argparse
import copy
import glob
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ft.backbones import load_backbone
from ft.heads import CRF, TokenHead
from ft.peft import AdaLoRALinear, RankAllocator, apply_method, peft_extra_loss, trainable_state
from phase1.data.a1_bpe import load_tokenizer
from . import data as D
from .decode import apply_trim, decode_doc, doc_scores, fit_trim_rules
from .experiments import get_experiment
from .score import errors, score, to_prediction_json, validate

TOKENIZER_SHA = "0c9231a52bd5"


class NERModel(nn.Module):
    def __init__(self, bb, tagset, head="linear", crf=False, echo=False, dropout=0.1, head_hidden=512):
        super().__init__()
        self.bb = bb
        self.echo = echo
        self.head = TokenHead(bb.d_model, tagset.n, head, dropout, head_hidden)
        self.crf = CRF(tagset.n, *tagset.constraints()) if crf else None
        self.head_tags = tagset.n

    def scores(self, x):
        return self.head(self.bb(x, echo=self.echo))


def kd_loss(s, t_logp, T):
    """T^2 KL(softmax(t/T) || softmax(s/T)) per token (mean). t_logp: teacher log-probs at T=1."""
    pt = F.softmax(t_logp / T, -1)
    return (pt * (F.log_softmax(t_logp / T, -1) - F.log_softmax(s / T, -1))).sum(-1).mean() * T * T


def find_tokenizer(roots):
    for r in roots:
        for p in sorted(glob.glob(os.path.join(r, "**", "a1bpe_*.json"), recursive=True)):
            if not p.endswith(".report.json"):
                return p
    raise FileNotFoundError(f"no a1bpe_*.json tokenizer under {roots}")


def sha12(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()[:12]


def load_docs(data, tok_path, split, cache_dir, tagset):
    os.makedirs(cache_dir, exist_ok=True)
    key = f"docs_{split}_{sha12(tok_path)}_{len(data['texts'])}_{len(data['test'])}.pkl"
    path = os.path.join(cache_dir, key)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    otok = D.OffsetTokenizer(load_tokenizer(tok_path), split)
    lab = {d: D.Doc(d, t, otok, tagset, data["gold"][d]) for d, t in data["texts"].items()}
    test = {d: D.Doc(d, t, otok) for d, t in data["test"].items()}
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump((lab, test), f)
    os.replace(tmp, path)
    return lab, test


def class_weights(docs, n, kind, o_weight):
    w = torch.ones(n)
    if kind == "sqrt_inv":
        cnt = np.bincount(np.concatenate([d.tags for d in docs]), minlength=n).astype(np.float64)
        ent = cnt[1:][cnt[1:] > 0]
        ref = np.median(ent) if len(ent) else 1.0
        w = torch.tensor(np.clip(np.sqrt(ref / np.maximum(cnt, 1)), 0.2, 10.0), dtype=torch.float32)
    w[0] = o_weight
    return w


def train_windows(docs, size, rng, oversample=1):
    out = []
    for d in docs:
        n = len(d)
        if n == 0:
            continue
        off = int(rng.integers(0, size)) if n > size else 0
        for s, e in D.windows(n, size, size, offset=off):
            reps = oversample if (oversample > 1 and (d.tags[s:e] > 0).any()) else 1
            out += [(d, s, e)] * reps
    rng.shuffle(out)
    return out


def batches(wins, bs):
    by_len = {}
    for w in wins:
        by_len.setdefault(w[2] - w[1], []).append(w)
    out = []
    for L, ws in by_len.items():
        for i in range(0, len(ws), bs):
            out.append(ws[i:i + bs])
    random.shuffle(out)
    return out


def param_groups(model, lr, head_lr, wd):
    groups = {"bb_decay": [], "bb_nodecay": [], "head": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("head") or n.startswith("crf"):
            groups["head"].append(p)
        elif p.ndim < 2 or "prompt" in n:
            groups["bb_nodecay"].append(p)
        else:
            groups["bb_decay"].append(p)
    return [g for g in ({"params": groups["bb_decay"], "lr": lr, "weight_decay": wd},
                        {"params": groups["bb_nodecay"], "lr": lr, "weight_decay": 0.0},
                        {"params": groups["head"], "lr": head_lr, "weight_decay": 0.01}) if g["params"]]


def decode_mode(model, cfg):
    return "crf" if model.crf is not None else cfg["decode"]


def predict_docs(model, docs, cfg, device, tagset, trim_rules):
    preds, raw = {}, {}
    model.eval()
    for d in docs:
        if len(d) == 0:
            raw[d.id] = np.zeros((0, model.head_tags), dtype=np.float32)
            preds[d.id] = set()
            continue
        sc = doc_scores(model, d, cfg["window"], cfg["stride"], device, batch=cfg["eval_batch"], amp=cfg["amp"])
        raw[d.id] = sc
        sp = decode_doc(sc, d, tagset, decode_mode(model, cfg), model.crf)
        if trim_rules:
            sp = apply_trim(sp, d.text, trim_rules)
        preds[d.id] = sp
    model.train()
    return preds, raw


def save_scores(path, raw):
    np.savez_compressed(path, **{k: v.astype(np.float16) for k, v in raw.items()})


def write_submission(out_dir, pred, test_docs, labels):
    pj = to_prediction_json(pred, {d.id: d.text for d in test_docs})
    validate(pj, [d.id for d in test_docs], labels)
    with open(os.path.join(out_dir, "prediction.json"), "w", encoding="utf-8") as f:
        json.dump(pj, f, ensure_ascii=False)
    with zipfile.ZipFile(os.path.join(out_dir, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(out_dir, "prediction.json"), "prediction.json")
    return sum(len(v) for v in pj.values())


def run(cfg, data_root, roots, out_root, mode="dev", max_minutes=None, device=None):
    t0 = time.time()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(cfg)
    cfg["amp"] = cfg.get("amp", True) and device.type == "cuda"
    exp = cfg["name"]
    out = os.path.join(out_root, exp if mode == "dev" else f"{exp}__{mode}")
    os.makedirs(out, exist_ok=True)
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    data = D.load_split(data_root)
    labels = data["labels"]
    tagset = D.TagSet(labels)
    tok_path = cfg.get("tokenizer") or find_tokenizer(roots)
    if sha12(tok_path) != TOKENIZER_SHA:
        print(f"WARNING: tokenizer {sha12(tok_path)} != Phase-1 {TOKENIZER_SHA}")
    lab, test = load_docs(data, tok_path, cfg["split"], os.path.join(out_root, "_cache"), tagset)
    ids = sorted(lab)
    if mode == "dev":
        tr_ids, dev_ids = D.dev_split(ids, cfg["dev_frac"])
    elif mode == "full":
        tr_ids, dev_ids = ids, []
    elif mode in ("f12", "f21"):
        a, b = (1, 2) if mode == "f12" else (2, 1)
        tr_ids = [d for d in ids if data["fold"][d] == a]
        dev_ids = [d for d in ids if data["fold"][d] == b]
    else:
        raise ValueError(mode)
    tr_docs, dev_docs = [lab[d] for d in tr_ids], [lab[d] for d in dev_ids]
    trim_rules = fit_trim_rules(data["texts"], {d: data["gold"][d] for d in tr_ids}) if cfg["trim"] else {}

    epochs = cfg["epochs"]
    if mode == "full":
        epochs = cfg.get("epochs_full") or epochs
        cands = [os.path.join(out_root, exp, "result.json")]
        cands += sorted(glob.glob(os.path.join("/kaggle/input", "**", exp, "result.json"), recursive=True))
        dev_res = next((c for c in cands if os.path.exists(c)), None)
        if not cfg.get("epochs_full") and dev_res:
            be = json.load(open(dev_res)).get("best_epoch")
            print(f"full mode: best dev epoch {be} from {dev_res}")
            if be:
                epochs = max(1, int(math.ceil(be / (1 - cfg["dev_frac"]))))
    bb = load_backbone(cfg["backbone"], roots, random_init=cfg.get("random_init", False), tiny=cfg.get("tiny", False))
    if cfg["bidir"]:
        bb.set_bidirectional(True)
    bb.to(device)
    margs = dict(cfg.get("method_args", {}))
    info = apply_method(bb, cfg["method"], **margs)
    model = NERModel(bb, tagset, cfg["head"], cfg["crf"], cfg["echo"], cfg["dropout"], cfg["head_hidden"]).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{exp}/{mode}] backbone {bb.name} ({bb.ckpt_path}) method {info} -> trainable {n_train:,} / {n_total:,}",
          flush=True)

    teacher_soft = None
    if cfg.get("kd"):
        kd = cfg["kd"]
        cands = [os.path.join(out_root, f"{kd['teacher']}__full", "soft_labels.npz"),
                 os.path.join(out_root, kd["teacher"], "soft_labels.npz")]
        cands += glob.glob(os.path.join("/kaggle/input", "**", kd["teacher"] + "*", "soft_labels.npz"), recursive=True)
        src = next((c for c in cands if os.path.exists(c)), None)
        if src is None:
            raise FileNotFoundError(f"KD teacher soft labels not found: {cands[:2]}")
        z = np.load(src)
        teacher_soft = {k: z[k].astype(np.float32) for k in z.files}
        for d in tr_docs:
            assert teacher_soft[d.id].shape[0] == len(d), f"teacher tokens differ for {d.id} (split mode?)"
        print(f"KD teacher: {src}")

    cw = class_weights(tr_docs, tagset.n, cfg["class_weight"], cfg["o_weight"]).to(device)
    opt = torch.optim.AdamW(param_groups(model, cfg["lr"], cfg["head_lr"], cfg["weight_decay"]), betas=(0.9, 0.98),
                            eps=1e-6)
    n_win = len(train_windows(tr_docs, cfg["window"], np.random.default_rng(0), cfg["oversample"]))
    steps_per_epoch = math.ceil(n_win / cfg["batch"]) + 4
    plan = {"total": max(1, steps_per_epoch * epochs)}
    warm = max(1, int(cfg["warmup"] * plan["total"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (plan["total"] - s) / max(1, plan["total"] - warm)))
    total = plan["total"]
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    ada = [m for m in model.modules() if isinstance(m, AdaLoRALinear)]
    allocator = RankAllocator(ada, info.get("target_r", 4), total) if ada else None
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**cfg, "mode": mode, "epochs_run": epochs, "peft_info": info, "tokenizer": tok_path,
                   "backbone_ckpt": bb.ckpt_path, "train_docs": len(tr_docs), "dev_docs": len(dev_docs)}, f, indent=1)
    logf = open(os.path.join(out, "log.jsonl"), "a")

    def log(rec):
        rec["t"] = round(time.time() - t0, 1)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    micro = cfg["micro"] or cfg["batch"]
    best = {"f1": -1.0, "epoch": 0, "state": None}
    bad = 0
    step = 0
    tok_seen = 0
    t_train = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    stop = False
    ep = 0
    while ep < epochs:
        ep += 1
        model.train()
        t_ep = time.time()
        losses = []
        for bt in batches(train_windows(tr_docs, cfg["window"], rng, cfg["oversample"]), cfg["batch"]):
            x = torch.stack([torch.from_numpy(d.ids[s:e].astype(np.int64)) for d, s, e in bt])
            y = torch.stack([torch.from_numpy(d.tags[s:e]) for d, s, e in bt])
            tsoft = (torch.stack([torch.from_numpy(teacher_soft[d.id][s:e]) for d, s, e in bt])
                     if teacher_soft is not None else None)
            while True:
                try:
                    opt.zero_grad(set_to_none=True)
                    tot_loss = 0.0
                    for i in range(0, len(bt), micro):
                        xb, yb = x[i:i + micro].to(device), y[i:i + micro].to(device)
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg["amp"]):
                            sc = model.scores(xb)
                        sc = sc.float()
                        if model.crf is not None:
                            loss = model.crf.nll(sc, yb, torch.ones_like(yb, dtype=torch.bool))
                        else:
                            loss = F.cross_entropy(sc.reshape(-1, tagset.n), yb.reshape(-1), weight=cw)
                        if tsoft is not None:
                            a = cfg["kd"].get("alpha", 0.5)
                            loss = a * loss + (1 - a) * kd_loss(sc.reshape(-1, tagset.n),
                                                                tsoft[i:i + micro].to(device).reshape(-1, tagset.n),
                                                                cfg["kd"].get("T", 2.0))
                        reg = peft_extra_loss(model.bb)
                        if reg is not None:
                            loss = loss + reg
                        loss = loss * xb.shape[0] / len(bt)
                        scaler.scale(loss).backward()
                        tot_loss += float(loss.detach())
                    break
                except torch.cuda.OutOfMemoryError:
                    if micro == 1:
                        raise
                    micro = max(1, micro // 2)
                    torch.cuda.empty_cache()
                    print(f"OOM -> micro-batch {micro}", flush=True)
            scaler.unscale_(opt)
            if allocator is not None:
                allocator.step(step + 1)
            if cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            tok_seen += x.numel() * (2 if cfg["echo"] else 1)
            losses.append(tot_loss)
            if max_minutes and (time.time() - t0) / 60 > max_minutes:
                stop = True
                break
        t_train += time.time() - t_ep
        if ep == 1 and cfg.get("time_cap") and mode != "full":
            per_ep = (time.time() - t0) / 60
            fit = max(2, int(cfg["time_cap"] / max(per_ep, 1e-3)))
            if fit < epochs:
                epochs = fit
                plan["total"] = max(step + 1, steps_per_epoch * epochs)
                print(f"time_cap {cfg['time_cap']} min: {per_ep:.1f} min/epoch -> {epochs} epochs", flush=True)
        rec = {"epoch": ep, "step": step, "loss": float(np.mean(losses)) if losses else None,
               "lr": sched.get_last_lr()[0], "tok_s": round(tok_seen / max(t_train, 1e-6)), "micro": micro}
        if dev_docs and (ep % cfg["eval_every"] == 0 or ep == epochs or stop):
            pred, _ = predict_docs(model, dev_docs, cfg, device, tagset, trim_rules)
            sc = score({d.id: d.gold for d in dev_docs}, pred, labels)
            rec.update(dev_f1=round(sc["macro_f1"], 4), dev_p=round(sc["macro_p"], 4), dev_r=round(sc["macro_r"], 4))
            if sc["macro_f1"] > best["f1"]:
                best = {"f1": sc["macro_f1"], "epoch": ep,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                                  if k in {n for n, p in model.named_parameters() if p.requires_grad}}}
                bad = 0
            else:
                bad += 1
        log(rec)
        if stop or (dev_docs and cfg["patience"] and bad >= cfg["patience"]):
            break
    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=False)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None

    result = {"exp": exp, "mode": mode, "backbone": cfg["backbone"], "method": cfg["method"],
              "params_total": n_total, "params_trainable": n_train, "peft_info": info,
              "best_epoch": best["epoch"] if dev_docs else epochs, "epochs_run": ep, "steps": step,
              "train_tok_s": round(tok_seen / max(t_train, 1e-6)), "train_minutes": round(t_train / 60, 2),
              "peak_mem_gib": peak, "trim_rules": {k: sorted(v) for k, v in trim_rules.items()}}
    if dev_docs:
        t1 = time.time()
        pred, raw = predict_docs(model, dev_docs, cfg, device, tagset, trim_rules)
        result["infer_tok_s"] = round(sum(len(d) for d in dev_docs) / max(time.time() - t1, 1e-6))
        truth = {d.id: d.gold for d in dev_docs}
        result["dev"] = score(truth, pred, labels)
        result["dev_untrimmed"] = score(truth, {d.id: decode_doc(raw[d.id], d, tagset, decode_mode(model, cfg),
                                                                 model.crf) for d in dev_docs},
                                        labels)["macro_f1"] if trim_rules else None
        result["dev_errors"] = errors(truth, pred)
        with open(os.path.join(out, "dev_pred.json"), "w", encoding="utf-8") as f:
            json.dump(to_prediction_json(pred, {d.id: d.text for d in dev_docs}), f, ensure_ascii=False)
        save_scores(os.path.join(out, "scores_dev.npz"), raw)
    def save_result():
        result["minutes_total"] = round((time.time() - t0) / 60, 2)
        with open(os.path.join(out, "result.json"), "w") as f:
            json.dump(result, f, indent=1)

    save_result()
    if cfg.get("save_soft"):
        _, raw = predict_docs(model, [lab[d] for d in ids], cfg, device, tagset, {})
        lp = {k: torch.log_softmax(torch.from_numpy(v), -1).numpy() for k, v in raw.items()}
        save_scores(os.path.join(out, "soft_labels.npz"), lp)
    sw = cfg.get("save_weights", "auto")
    if sw is True or (sw == "auto" and n_train < 20e6):
        torch.save({"trainable": {k: v.half() for k, v in trainable_state(model).items()}, "config": cfg,
                    "peft_info": info}, os.path.join(out, "adapter.pt"))
    if cfg["predict_test"]:
        pred, raw = predict_docs(model, list(test.values()), cfg, device, tagset, trim_rules)
        result["test_entities"] = write_submission(out, pred, list(test.values()), labels)
        save_scores(os.path.join(out, "scores_test.npz"), raw)
    save_result()
    msg = f"[{exp}/{mode}] done in {result['minutes_total']} min"
    if dev_docs:
        msg += f": dev strict macro-F1 {result['dev']['macro_f1']:.4f} (epoch {best['epoch']})"
    print(msg, flush=True)
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
    ap.add_argument("--data", required=True, help="Phase_2_LNER folder (or its parent)")
    ap.add_argument("--roots", nargs="+", default=["results"], help="where to look for checkpoints/tokenizer")
    ap.add_argument("--out", default="runs_p2")
    ap.add_argument("--mode", default="dev", choices=["dev", "full", "f12", "f21"])
    ap.add_argument("--set", nargs="*", help="config overrides key=value (JSON values)")
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args(argv)
    root = D.find_root(a.data)
    if root is None:
        sys.exit(f"no Phase_2_LNER data under {a.data}")
    cfg = get_experiment(a.exp, parse_set(a.set))
    run(cfg, root, a.roots, a.out, a.mode, a.max_minutes)


if __name__ == "__main__":
    main()
