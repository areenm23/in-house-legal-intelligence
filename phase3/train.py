"""Train / evaluate / predict one rhetorical-role experiment.

    python -m phase3.train --exp G_lora --data DATA_ROOT --roots /kaggle/input results --out runs_p3 [--mode dev]
           [--set lr=3e-4 epochs=10] [--max-minutes 60]

Modes
  dev   train on {CL,IT}_train (cfg train_domains), pick the epoch with the best macro-F1 on CL_dev + IT_dev
        pooled (the official metric), then predict the test set with that model
  full  train on train + dev for `epochs_full` epochs (default: best dev epoch of the matching dev run,
        scaled by the extra data), then predict the test set

Outputs in <out>/<exp>[__full]/: config.json, log.jsonl, result.json (dev scores overall / per domain /
per label, confusion matrix, params, memory, speed), dev_pred.json, prediction.json, submission.zip
(+ submission_bias.zip when tune_bias), scores_{dev,test}.npz (per-sentence log-probs keyed DOMAIN/id,
for ensembling), adapter.pt (head + trainable weights) and, with save_soft, soft_labels.npz (log-probs
on every labelled document, the KD teacher signal).
"""
import argparse
import glob
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time

import numpy as np
import torch

from ft.backbones import load_backbone
from ft.peft import AdaLoRALinear, RankAllocator, apply_method, peft_extra_loss, trainable_state
from phase1.data.a1_bpe import load_tokenizer
from . import data as D
from .experiments import get_experiment
from .model import RRModel
from .score import fit_bias, score, to_prediction_json, write_submission

TOKENIZER_SHA = "0c9231a52bd5"
DOM_ID = {"CL": 0, "IT": 1}


def kd_loss(s, t_logp, T):
    """T^2 KL(softmax(t/T) || softmax(s/T)) per sentence (mean). t_logp: teacher log-probs at T=1."""
    F = torch.nn.functional
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


def load_data(root, tok_path, max_sent_tokens, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"p3_{D.cache_key(root, tok_path, max_sent_tokens)}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    data = D.encode_all(D.load_split(root), load_tokenizer(tok_path), max_sent_tokens)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp, path)
    return data


def class_weights(docs, n, kind):
    if kind == "none":
        return None
    cnt = D.label_counts(docs, n).astype(np.float64)
    ref = np.median(cnt[cnt > 0]) if (cnt > 0).any() else 1.0
    w = ref / np.maximum(cnt, 1)
    w = np.sqrt(w) if kind == "sqrt_inv" else w
    return torch.tensor(np.clip(w, 0.1, 10.0), dtype=torch.float32)


def param_groups(model, lr, head_lr, wd):
    groups = {"bb_decay": [], "bb_nodecay": [], "head": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not n.startswith("bb."):
            groups["head"].append(p)
        elif p.ndim < 2 or "prompt" in n:
            groups["bb_nodecay"].append(p)
        else:
            groups["bb_decay"].append(p)
    return [g for g in ({"params": groups["bb_decay"], "lr": lr, "weight_decay": wd},
                        {"params": groups["bb_nodecay"], "lr": lr, "weight_decay": 0.0},
                        {"params": groups["head"], "lr": head_lr, "weight_decay": 0.01}) if g["params"]]


def make_chunks(docs, max_tokens, rng):
    out = [(d, s0, s1) for d in docs for s0, s1 in D.chunks(d, max_tokens, rng)]
    rng.shuffle(out)
    return out


@torch.no_grad()
def predict(model, docs, cfg, device):
    """-> ({key: label ids}, {key: (n_sent, n) log-probs}). Whole documents, overlapping windows."""
    model.eval()
    preds, raw = {}, {}
    for d in docs:
        if len(d) == 0:
            preds[d.key] = np.zeros(0, dtype=np.int64)
            raw[d.key] = np.zeros((0, model.n), dtype=np.float32)
            continue
        sc = model(d, 0, len(d), DOM_ID.get(d.domain, 0), cfg, device, cfg["amp"])
        raw[d.key] = torch.log_softmax(sc, -1).cpu().numpy()
        preds[d.key] = model.decode(sc)
    model.train()
    return preds, raw


def save_scores(path, raw):
    np.savez_compressed(path, **{k.replace("/", "__"): v.astype(np.float16) for k, v in raw.items()})


def load_scores(path):
    z = np.load(path)
    return {k.replace("__", "/", 1): z[k].astype(np.float32) for k in z.files}


def find_dev_result(out_root, exp):
    cands = [os.path.join(out_root, exp, "result.json")]
    cands += sorted(glob.glob(os.path.join("/kaggle/input", "**", "runs_p3", exp, "result.json"), recursive=True))
    return next((c for c in cands if os.path.exists(c)), None)


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

    tok_path = cfg.get("tokenizer") or find_tokenizer(roots)
    if sha12(tok_path) != TOKENIZER_SHA:
        print(f"WARNING: tokenizer {sha12(tok_path)} != Phase-1 {TOKENIZER_SHA}")
    data = load_data(data_root, tok_path, cfg["max_sent_tokens"], os.path.join(out_root, "_cache"))
    labels = data["labels"]
    n = len(labels)
    doms = cfg["train_domains"]
    dev_docs = D.all_docs(data["dev"])
    if mode == "dev":
        tr_docs = D.all_docs(data["train"], doms)
    elif mode == "full":
        tr_docs = D.all_docs(data["train"], doms) + D.all_docs(data["dev"], doms)
        dev_docs = []
    else:
        raise ValueError(mode)
    test_docs = D.all_docs(data["test"])

    epochs = cfg["epochs"]
    dev_res = find_dev_result(out_root, exp) if mode == "full" else None
    if mode == "full":
        epochs = cfg.get("epochs_full") or epochs
        if not cfg.get("epochs_full") and dev_res:
            be = json.load(open(dev_res)).get("best_epoch")
            n_tr = sum(len(d) for d in D.all_docs(data["train"], doms))
            scale = sum(len(d) for d in tr_docs) / max(1, n_tr)
            print(f"full mode: best dev epoch {be} from {dev_res} (data x{scale:.2f})")
            if be:
                epochs = max(1, int(math.ceil(be * min(scale, 1.5))))
    bb = load_backbone(cfg["backbone"], roots, random_init=cfg.get("random_init", False), tiny=cfg.get("tiny", False))
    if cfg["bidir"]:
        bb.set_bidirectional(True)
    bb.to(device)
    info = apply_method(bb, cfg["method"], **dict(cfg.get("method_args", {})))
    model = RRModel(bb, n, cfg["ctx"], cfg["pool"], cfg["head"], cfg["crf"], cfg["echo"], cfg["dropout"],
                    cfg["head_hidden"], cfg["pos_feat"], cfg["dom_feat"]).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{exp}/{mode}] backbone {bb.name} ({bb.ckpt_path}) method {info} -> trainable {n_train:,} / {n_total:,}; "
          f"train docs {len(tr_docs)} ({sum(len(d) for d in tr_docs)} sentences), dev docs {len(dev_docs)}",
          flush=True)

    teacher = None
    if cfg.get("kd"):
        kd = cfg["kd"]
        folders = [kd["teacher"], kd["teacher"] + "__full"]
        if mode == "full":
            folders.reverse()
        cands = []
        for fo in folders:
            cands.append(os.path.join(out_root, fo, "soft_labels.npz"))
            cands += sorted(glob.glob(os.path.join("/kaggle/input", "**", "runs_p3", fo, "soft_labels.npz"),
                                      recursive=True))
        if mode == "dev":
            cands = [c for c in cands if not os.path.dirname(c).endswith("__full")]
        src = next((c for c in cands if os.path.exists(c)), None)
        if src is None:
            raise FileNotFoundError(f"KD teacher soft labels not found: {cands[:2]}")
        teacher = load_scores(src)
        for d in tr_docs:
            assert teacher[d.key].shape == (len(d), n), f"teacher scores do not match {d.key}"
        print(f"KD teacher: {src}")

    cw = class_weights(tr_docs, n, cfg["class_weight"])
    cw = cw.to(device) if cw is not None else None
    opt = torch.optim.AdamW(param_groups(model, cfg["lr"], cfg["head_lr"], cfg["weight_decay"]), betas=(0.9, 0.98),
                            eps=1e-6)
    chunk_tokens = cfg["chunk_tokens"]
    steps_per_epoch = math.ceil(len(make_chunks(tr_docs, chunk_tokens, np.random.default_rng(0))) / cfg["batch"]) + 2
    plan = {"total": max(1, steps_per_epoch * epochs)}
    warm = max(1, int(cfg["warmup"] * plan["total"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (plan["total"] - s) / max(1, plan["total"] - warm)))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    ada = [m for m in model.modules() if isinstance(m, AdaLoRALinear)]
    allocator = RankAllocator(ada, info.get("target_r", 4), plan["total"]) if ada else None
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**cfg, "mode": mode, "epochs_run": epochs, "peft_info": info, "tokenizer": tok_path,
                   "backbone_ckpt": bb.ckpt_path, "train_docs": len(tr_docs), "dev_docs": len(dev_docs),
                   "labels": labels}, f, indent=1)
    logf = open(os.path.join(out, "log.jsonl"), "a")

    def log(rec):
        rec["t"] = round(time.time() - t0, 1)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    best = {"f1": -1.0, "epoch": 0, "state": None}
    bad = step = sent_seen = 0
    t_train = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    stop = False
    ep = 0
    train_names = {k for k, p in model.named_parameters() if p.requires_grad}
    while ep < epochs:
        ep += 1
        model.train()
        t_ep = time.time()
        losses = []
        ch = make_chunks(tr_docs, chunk_tokens, rng)
        for i in range(0, len(ch), cfg["batch"]):
            bt = ch[i:i + cfg["batch"]]
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            try:
                for d, s0, s1 in bt:
                    sc = model(d, s0, s1, DOM_ID.get(d.domain, 0), cfg, device, cfg["amp"])
                    y = torch.from_numpy(d.y[s0:s1]).to(device)
                    loss = model.loss(sc, y, cw, cfg["focal"], cfg["smoothing"])
                    if teacher is not None:
                        a = cfg["kd"].get("alpha", 0.5)
                        loss = a * loss + (1 - a) * kd_loss(sc, torch.from_numpy(teacher[d.key][s0:s1]).to(device),
                                                            cfg["kd"].get("T", 2.0))
                    reg = peft_extra_loss(model.bb)
                    if reg is not None:
                        loss = loss + reg
                    loss = loss / len(bt)
                    scaler.scale(loss).backward()
                    tot += float(loss.detach())
                    sent_seen += s1 - s0
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if chunk_tokens <= cfg["window"] // 2:
                    raise
                chunk_tokens //= 2
                print(f"OOM -> chunk_tokens {chunk_tokens} (from the next epoch)", flush=True)
                continue
            scaler.unscale_(opt)
            if allocator is not None:
                allocator.step(step + 1)
            if cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            losses.append(tot)
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
               "lr": sched.get_last_lr()[0], "sent_s": round(sent_seen / max(t_train, 1e-6), 1),
               "chunk_tokens": chunk_tokens}
        if dev_docs and (ep % cfg["eval_every"] == 0 or ep == epochs or stop):
            pred, _ = predict(model, dev_docs, cfg, device)
            sc = score(dev_docs, pred, labels)
            rec.update(dev_f1=round(sc["macro_f1"], 4), dev_acc=round(sc["accuracy"], 4),
                       **{f"dev_f1_{k}": round(v["macro_f1"], 4) for k, v in sc["domains"].items()})
            if sc["macro_f1"] > best["f1"]:
                best = {"f1": sc["macro_f1"], "epoch": ep,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                                  if k in train_names}}
                bad = 0
            else:
                bad += 1
        log(rec)
        if stop or (dev_docs and cfg["patience"] and bad >= cfg["patience"]):
            break
    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=False)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None

    result = {"exp": exp, "mode": mode, "backbone": cfg["backbone"], "method": cfg["method"], "ctx": cfg["ctx"],
              "head": cfg["head"], "crf": cfg["crf"], "train_domains": doms,
              "params_total": n_total, "params_trainable": n_train, "peft_info": info,
              "best_epoch": best["epoch"] if dev_docs else epochs, "epochs_run": ep, "steps": step,
              "train_sent_s": round(sent_seen / max(t_train, 1e-6), 1), "train_minutes": round(t_train / 60, 2),
              "peak_mem_gib": peak}
    bias = None
    if dev_docs:
        t1 = time.time()
        pred, raw = predict(model, dev_docs, cfg, device)
        result["infer_sent_s"] = round(sum(len(d) for d in dev_docs) / max(time.time() - t1, 1e-6), 1)
        result["dev"] = score(dev_docs, pred, labels)
        with open(os.path.join(out, "dev_pred.json"), "w", encoding="utf-8") as f:
            json.dump(to_prediction_json(dev_docs, pred, labels), f, ensure_ascii=False)
        save_scores(os.path.join(out, "scores_dev.npz"), raw)
        if cfg["tune_bias"] and model.crf is None:
            keys = [d.key for d in dev_docs if len(d)]
            lp = np.concatenate([raw[k] for k in keys])
            y = np.concatenate([d.y for d in dev_docs if len(d)])
            bias, f_b = fit_bias(lp, y, n)
            result["bias"] = bias.tolist()
            result["dev_bias_f1_optimistic"] = round(f_b, 6)
    elif mode == "full" and cfg["tune_bias"] and dev_res:
        b = json.load(open(dev_res)).get("bias")
        bias = np.array(b) if b is not None else None
        if bias is not None:
            result["bias"] = b
            result["bias_from"] = dev_res

    def save_result():
        result["minutes_total"] = round((time.time() - t0) / 60, 2)
        with open(os.path.join(out, "result.json"), "w") as f:
            json.dump(result, f, indent=1)

    save_result()
    if cfg.get("save_soft"):
        lab = D.all_docs(data["train"]) + D.all_docs(data["dev"])
        _, raw = predict(model, lab, cfg, device)
        save_scores(os.path.join(out, "soft_labels.npz"), raw)
    sw = cfg.get("save_weights", "auto")
    if sw is True or (sw == "auto" and n_train < 20e6):
        torch.save({"trainable": {k: v.half() for k, v in trainable_state(model).items()}, "config": cfg,
                    "peft_info": info}, os.path.join(out, "adapter.pt"))
    if cfg["predict_test"] and test_docs:
        pred, raw = predict(model, test_docs, cfg, device)
        result["test_sentences"] = write_submission(out, test_docs, pred, labels)
        save_scores(os.path.join(out, "scores_test.npz"), raw)
        if bias is not None:
            bdir = os.path.join(out, "bias")
            os.makedirs(bdir, exist_ok=True)
            pb = {k: (v + bias).argmax(1) if len(v) else np.zeros(0, dtype=np.int64) for k, v in raw.items()}
            write_submission(bdir, test_docs, pb, labels)
            os.replace(os.path.join(bdir, "submission.zip"), os.path.join(out, "submission_bias.zip"))
        result["test_pred_dist"] = {labels[i]: int(c) for i, c in enumerate(
            np.bincount(np.concatenate([pred[d.key] for d in test_docs if len(d)] or [np.zeros(0, int)]),
                        minlength=n))}
    save_result()
    msg = f"[{exp}/{mode}] done in {result['minutes_total']} min"
    if dev_docs:
        msg += f": dev macro-F1 {result['dev']['macro_f1']:.4f} (epoch {best['epoch']}; " + ", ".join(
            f"{k} {v['macro_f1']:.4f}" for k, v in result["dev"]["domains"].items()) + ")"
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
    ap.add_argument("--data", required=True, help="Phase_3_RR folder (or its parent)")
    ap.add_argument("--roots", nargs="+", default=["results"], help="where to look for checkpoints/tokenizer")
    ap.add_argument("--out", default="runs_p3")
    ap.add_argument("--mode", default="dev", choices=["dev", "full"])
    ap.add_argument("--set", nargs="*", help="config overrides key=value (JSON values)")
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args(argv)
    root = D.find_root(a.data)
    if root is None:
        sys.exit(f"no Phase_3_RR data under {a.data}")
    run(get_experiment(a.exp, parse_set(a.set)), root, a.roots, a.out, a.mode, a.max_minutes)


if __name__ == "__main__":
    main()
