"""Train / generate / score one legal-summarisation experiment.

    python -m phase6.train --exp B_lora --data DATA_ROOT --roots /kaggle/input results --out runs_p6 [--set ...]
    python -m phase6.train --decode runs_p6/B_lora --set len_ratio=0.12 --name B_lora__dec_r12   # new decoding only

Training pairs: (selected document tokens, summary tokens) per training document (ctx = head / headtail /
central / oracle), or chunk-level pairs (ctx = dancer). Loss = token cross-entropy on the summary (+ EOS);
with kd=dict(teacher=RUN, alpha, T), plus T^2 KL to the teacher's next-token distribution on the same
summary positions (word-level KD; the teacher is rebuilt from RUN/config.json + RUN/adapter.pt, so it
must be a PEFT run). Model selection = teacher-forced loss on a fixed dev subset. At the end: generation
for `gen_dev_docs` dev documents (scored with the official ROUGE-1 / ROUGE-L / BERTScore definitions) and
for every test document -> submission.zip.

Length control: target tokens per document = clip(len_ratio x document tokens, len_min, len_max), with
len_ratio defaulting to the training median; generation may stop after min_frac x target and must stop at
max_frac x target. For dancer, each chunk gets its share of the document's target.
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from ft.peft import AdaLoRALinear, RankAllocator, apply_method, peft_extra_loss, trainable_state
from phase1.data.a1_bpe import load_tokenizer
from phase1.eval.generate import DecodeConfig
from phase3.train import find_tokenizer, sha12
from . import data as D
from .experiments import get_experiment
from .genlm import generate, group_pairs, kd_terms, load_genlm, make_batch
from .score import clean, evaluate, novel_bigrams, write_submission

TOKENIZER_SHA = "0c9231a52bd5"
ADAPTER_METHODS = ("houlsby", "pfeiffer")


def find_run_file(out_root, run, fname):
    cands = [os.path.join(out_root, run, fname)]
    cands += sorted(glob.glob(os.path.join("/kaggle/input", "**", "runs_p6", run, fname), recursive=True))
    return next((c for c in cands if os.path.exists(c)), None)


def build_model(cfg, roots, device):
    g = load_genlm(cfg["backbone"], roots, random_init=cfg.get("random_init", False), tiny=cfg.get("tiny", False))
    if cfg["method"] in ADAPTER_METHODS and g.family in ("retnet", "mamba"):
        raise ValueError(f"{cfg['method']} adapters are not supported for {g.family} generation (use a Linear-wrapper method)")
    g.to(device)
    info = apply_method(g, cfg["method"], **dict(cfg.get("method_args", {})))
    g.to(device)
    return g, info


def load_run_model(run_dir, roots, device, tiny=None):
    """Rebuild a finished PEFT run (config.json + adapter.pt) for KD or re-decoding."""
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    if tiny is not None:
        cfg["tiny"] = tiny
    ap = os.path.join(run_dir, "adapter.pt")
    if not os.path.exists(ap):
        raise FileNotFoundError(f"{run_dir}: no adapter.pt (full fine-tunes are not stored; use a PEFT run)")
    g, _ = build_model(cfg, roots, device)
    sd = torch.load(ap, map_location="cpu", weights_only=False)["trainable"]
    missing = [k for k in sd if k not in dict(g.named_parameters())]
    if missing:
        raise RuntimeError(f"{run_dir}: adapter keys not in the model: {missing[:3]}")
    g.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    return g, cfg


def summary_sentences(tok, ref):
    return [np.asarray(tok.encode(s).ids, dtype=np.int64) for s in D.split_sentences(ref)]


def make_pairs(docs, idx, cfg, idf, tok):
    """-> list of (input tokens, target tokens, doc index)."""
    out = []
    for i in idx:
        sents = docs.sents[i]
        y = docs.summ[i].astype(np.int64) if docs.summ is not None else None
        if cfg["ctx"] == "dancer":
            for x, t in D.dancer_pairs(sents, summary_sentences(tok, docs.ref[i]), cfg["budget"]):
                out.append((x, t, i))
        else:
            out.append((D.select_input(sents, cfg["budget"], cfg["ctx"], idf, cfg["head_tokens"], cfg["lead_frac"], y),
                        y, i))
    return out


def gen_inputs(docs, idx, cfg, idf, len_ratio):
    """Generation inputs + (min, max) new tokens per input; dancer -> one input per chunk.
    -> (inputs, lengths, owner doc per input)."""
    inputs, lengths, owner = [], [], []
    for i in idx:
        sents = docs.sents[i]
        n = sum(len(s) for s in sents)
        target = int(np.clip(cfg.get("len_scale", 1.0) * len_ratio * n, cfg["len_min"], cfg["len_max"]))
        if cfg["ctx"] == "dancer":
            ch = D.chunk_sentences(sents, cfg["budget"]) or [(0, 0)]
            for a, b in ch:
                x = np.concatenate(sents[a:b])[:cfg["budget"]].astype(np.int64) if b > a else np.array([3])
                share = max(8, int(target * len(x) / max(1, n)))
                inputs.append(x)
                lengths.append((int(cfg["min_frac"] * share), max(8, int(cfg["max_frac"] * share))))
                owner.append(i)
        else:
            ctx = "central" if cfg["ctx"] == "oracle" else cfg["ctx"]
            x = D.select_input(sents, cfg["budget"], ctx, idf, cfg["head_tokens"], cfg["lead_frac"])
            inputs.append(x if len(x) else np.array([3]))
            lengths.append((int(cfg["min_frac"] * target), max(8, int(cfg["max_frac"] * target))))
            owner.append(i)
    return inputs, lengths, owner


def summarise(g, docs, idx, cfg, idf, len_ratio, tok, device):
    inputs, lengths, owner = gen_inputs(docs, idx, cfg, idf, len_ratio)
    dec = DecodeConfig(temperature=cfg["temperature"], top_p=cfg["top_p"], no_repeat_ngram=cfg["no_repeat_ngram"],
                       repetition_penalty=cfg["repetition_penalty"], seed=cfg["seed"])
    toks = generate(g, inputs, lengths, device, dec, cfg["gen_batch"], cfg["amp"])
    per = {i: [] for i in idx}
    for o, t in zip(owner, toks):
        per[o] += t
    out = []
    for i in idx:
        s = clean(tok.decode(per[i]))
        if not s:
            s = " ".join(tok.decode(docs.doc_tokens(i)[:400].tolist()).split()[:cfg["fallback_words"]]) or "Judgment."
        out.append(s)
    return out, sum(len(t) for t in toks)


@torch.no_grad()
def dev_loss(g, pairs, cfg, device):
    was = g.training
    g.eval()
    tot = n = 0.0
    for grp in group_pairs(g.family, [(p[0], p[1]) for p in pairs], cfg["batch"], cfg["batch_tokens"]):
        b = make_batch(g.family, [(pairs[i][0], pairs[i][1]) for i in grp], device, cfg["max_target"])
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg["amp"]):
            lg, y = g.target_logits(b)
        tot += float(F.cross_entropy(lg, y, reduction="sum"))
        n += len(y)
    g.train(was)
    return tot / max(1, n)


def teacher_logits(tg, tcfg, docs, rows, idf, device, max_target):
    """Teacher next-token logits on the student's target positions, row by row (row-major order)."""
    outs = []
    for i, y in rows:
        x = D.select_input(docs.sents[i], tcfg["budget"], tcfg["ctx"] if tcfg["ctx"] != "dancer" else "head", idf,
                           tcfg["head_tokens"], tcfg["lead_frac"], y)
        b = make_batch(tg.family, [(x, y)], device, max_target)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            lg, _ = tg.target_logits(b)
        outs.append(lg.float())
    return torch.cat(outs)


def finish(g, cfg, data, tok, idf, len_ratio, out, result, device, t0):
    """Generate dev subset + test, score dev, write files."""
    dv, te = data["dev"], data["test"]
    gd = list(range(min(cfg["gen_dev_docs"], len(dv))))
    t1 = time.time()
    preds, n_new = summarise(g, dv, gd, cfg, idf, len_ratio, tok, device)
    result["gen_dev_seconds"] = round(time.time() - t1, 1)
    result["gen_tokens_per_s"] = round(n_new / max(1e-6, time.time() - t1), 1)
    refs = [dv.ref[i] for i in gd]
    with open(os.path.join(out, "dev_pred.json"), "w", encoding="utf-8") as f:
        json.dump([{"id": dv.ids[i], "summary": p, "reference": r} for i, p, r in zip(gd, preds, refs)], f,
                  ensure_ascii=False)
    if cfg["predict_test"]:
        t1 = time.time()
        tp, _ = summarise(g, te, list(range(len(te))), cfg, idf, len_ratio, tok, device)
        result["test_docs"] = write_submission(out, te.ids, tp)
        result["gen_test_seconds"] = round(time.time() - t1, 1)
    result["minutes_total"] = round((time.time() - t0) / 60, 2)
    json.dump(result, open(os.path.join(out, "result.json"), "w"), indent=1)
    try:
        src = [" ".join(tok.decode(dv.doc_tokens(i)[:20000].tolist()).split()) for i in gd]
        result["dev"] = evaluate(preds, refs, with_bert=cfg["bertscore"])
        result["dev"]["novel_bigrams"] = novel_bigrams(preds, src)
    except Exception as e:
        print(f"[{cfg['name']}] scoring failed: {type(e).__name__}: {e}", flush=True)
        result["dev"] = {"error": f"scoring failed: {type(e).__name__}: {e}"}
    result["minutes_total"] = round((time.time() - t0) / 60, 2)
    json.dump(result, open(os.path.join(out, "result.json"), "w"), indent=1)
    return result


def length_ratio(data, cfg):
    if cfg.get("len_ratio"):
        return float(cfg["len_ratio"])
    tr = data["train"]
    r = [len(tr.summ[i]) / max(1, sum(len(s) for s in tr.sents[i])) for i in range(len(tr))]
    return float(np.median(r))


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
    tok = load_tokenizer(tok_path)
    data = D.load_data(data_root, tok_path, os.path.join(out_root, "_cache"))
    tr, dv = data["train"], data["dev"]
    idf = D.IDF(tr) if cfg["ctx"] in ("central", "oracle") else None
    len_ratio = length_ratio(data, cfg)
    tr_idx = np.arange(len(tr)) if not cfg["max_train_docs"] else \
        np.sort(rng.choice(len(tr), size=min(cfg["max_train_docs"], len(tr)), replace=False))
    t1 = time.time()
    pairs = make_pairs(tr, tr_idx, cfg, idf, tok)
    ev_pairs = make_pairs(dv, range(min(cfg["eval_docs"], len(dv))), cfg, idf, tok)
    print(f"[{exp}] {len(pairs)} training pairs from {len(tr_idx)} docs ({time.time() - t1:.0f}s), dev {len(dv)} "
          f"(loss on {len(ev_pairs)} pairs, generation on {min(cfg['gen_dev_docs'], len(dv))} docs), test "
          f"{len(data['test'])}; length ratio {len_ratio:.3f}", flush=True)

    g, info = build_model(cfg, roots, device)
    teacher = None
    if cfg.get("kd"):
        tdir = os.path.dirname(find_run_file(out_root, cfg["kd"]["teacher"], "config.json") or "")
        if not tdir:
            raise FileNotFoundError(f"KD teacher run {cfg['kd']['teacher']} not found")
        tg, tcfg = load_run_model(tdir, roots, device, tiny=cfg.get("tiny"))
        tg.eval()
        for p in tg.parameters():
            p.requires_grad_(False)
        tidf = D.IDF(tr) if tcfg["ctx"] in ("central", "oracle") else None
        teacher = {"g": tg, "cfg": tcfg, "idf": tidf, "dir": tdir}
        print(f"KD teacher {tdir} ({tg.name}, {tcfg['method']}, ctx {tcfg['ctx']})", flush=True)
    n_total = sum(p.numel() for p in g.parameters())
    n_train = sum(p.numel() for p in g.parameters() if p.requires_grad)
    print(f"[{exp}] {g.name} ({g.ckpt_path}) {info} -> trainable {n_train:,} / {n_total:,}", flush=True)
    decay = [p for n, p in g.named_parameters() if p.requires_grad and p.ndim >= 2]
    nodecay = [p for n, p in g.named_parameters() if p.requires_grad and p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": nodecay, "weight_decay": 0.0}], lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-6)
    groups = group_pairs(g.family, [(p[0], p[1]) for p in pairs], cfg["batch"], cfg["batch_tokens"])
    plan = {"total": max(1, len(groups) * cfg["epochs"])}
    warm = lambda: max(1, int(cfg["warmup"] * plan["total"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm() if s < warm() else max(0.0, (plan["total"] - s) / max(1, plan["total"] - warm())))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    ada = [m for m in g.modules() if isinstance(m, AdaLoRALinear)]
    allocator = RankAllocator(ada, info.get("target_r", 4), plan["total"]) if ada else None
    json.dump({**cfg, "peft_info": info, "tokenizer": tok_path, "backbone_ckpt": g.ckpt_path, "len_ratio_used": len_ratio,
               "train_docs": int(len(tr_idx)), "train_pairs": len(pairs)}, open(os.path.join(out, "config.json"), "w"),
              indent=1)
    logf = open(os.path.join(out, "log.jsonl"), "a")

    def log(rec):
        rec["t"] = round(time.time() - t0, 1)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    budget_s = cfg["budget_min"] * 60 if cfg.get("budget_min") else None
    reserve_s = cfg["reserve_min"] * 60
    best = {"loss": float("inf"), "step": 0, "state": None}
    train_names = {n for n, p in g.named_parameters() if p.requires_grad}
    step = tokens = bad = 0
    t_train, times = 0.0, []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    next_eval = max(1, plan["total"] // cfg["n_evals"])
    stop = None
    max_tgt = cfg["max_target"]
    ep = 0
    g.train()
    while stop is None and ep < cfg["epochs"]:
        ep += 1
        for gi in rng.permutation(len(groups)):
            if step >= plan["total"]:
                stop = "plan"
                break
            grp = groups[gi]
            ts = time.time()
            rows = [(pairs[i][0], pairs[i][1]) for i in grp]
            while True:
                opt.zero_grad(set_to_none=True)
                try:
                    b = make_batch(g.family, rows, device, max_tgt)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg["amp"]):
                        lg, y = g.target_logits(b)
                    loss = F.cross_entropy(lg, y, label_smoothing=cfg["smoothing"])
                    if teacher is not None:
                        tl = teacher_logits(teacher["g"], teacher["cfg"], tr, [(pairs[i][2], pairs[i][1]) for i in grp],
                                            teacher["idf"], device, max_tgt)
                        a = cfg["kd"].get("alpha", 0.5)
                        loss = a * loss + (1 - a) * kd_terms(lg, tl, cfg["kd"].get("T", 2.0))
                    reg = peft_extra_loss(g)
                    if reg is not None:
                        loss = loss + reg
                    scaler.scale(loss).backward()
                    break
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if max_tgt <= 128:
                        raise
                    max_tgt = int(max_tgt * 0.75)
                    print(f"OOM -> max_target {max_tgt}", flush=True)
            scaler.unscale_(opt)
            if allocator is not None:
                allocator.step(step + 1)
            torch.nn.utils.clip_grad_norm_([p for p in g.parameters() if p.requires_grad], cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            tokens += int(b["mask"].sum()) + (int(b["src"].numel()) if "src" in b else int((b["ids"] != 0).sum()))
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - ts
            t_train += dt
            times.append(dt)
            if budget_s and (step == cfg["probe_steps"] or step == next_eval):
                sps = float(np.mean(times[-50:] if step > cfg["probe_steps"] else times[min(3, len(times) - 1):]))
                fit = step + int(max(0.0, budget_s - t_train) / max(sps, 1e-6))
                if fit < plan["total"]:
                    plan["total"] = max(step + 1, fit)
                    next_eval = min(next_eval, step + max(1, (plan["total"] - step) // cfg["n_evals"]))
                    if allocator is not None:
                        allocator.retarget(plan["total"])
                    print(f"plan: {sps:.2f} s/step -> {plan['total']} steps ({plan['total'] / len(groups):.2f} epochs)",
                          flush=True)
            if step % cfg["log_every"] == 0:
                log({"step": step, "epoch": ep, "loss": round(float(loss.detach()), 4), "lr": sched.get_last_lr()[0],
                     "tok_s": round(tokens / max(t_train, 1e-6))})
            if step >= next_eval or step >= plan["total"]:
                vl = dev_loss(g, ev_pairs, cfg, device)
                log({"step": step, "epoch": round(step / len(groups), 3), "dev_loss": round(vl, 4), "plan": plan["total"]})
                if vl < best["loss"]:
                    best = {"loss": vl, "step": step,
                            "state": {k: v.detach().cpu().clone() for k, v in g.state_dict().items() if k in train_names}}
                    bad = 0
                else:
                    bad += 1
                next_eval = step + max(1, plan["total"] // cfg["n_evals"])
                if cfg["patience"] and bad >= cfg["patience"]:
                    stop = "patience"
                    break
            if max_minutes and time.time() - t0 > max_minutes * 60 - reserve_s:
                stop = "deadline"
                break
    stop = stop or "epochs"
    if best["state"] is not None:
        g.load_state_dict(best["state"], strict=False)
    if teacher is not None:
        del teacher["g"]
        torch.cuda.empty_cache() if device.type == "cuda" else None
    result = {"exp": exp, "backbone": cfg["backbone"], "family": g.family, "method": cfg["method"], "ctx": cfg["ctx"],
              "budget": cfg["budget"], "kd": cfg.get("kd"), "params_total": n_total, "params_trainable": n_train,
              "peft_info": info, "train_docs": int(len(tr_idx)), "train_pairs": len(pairs), "steps": step,
              "epochs_seen": round(step / max(1, len(groups)), 3), "stop": stop, "best_step": best["step"],
              "best_dev_loss": round(best["loss"], 5), "train_tok_s": round(tokens / max(t_train, 1e-6), 1),
              "train_minutes": round(t_train / 60, 2), "len_ratio": len_ratio,
              "peak_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None}
    sw = cfg.get("save_weights", "auto")
    if sw is True or (sw == "auto" and n_train < 30e6):
        torch.save({"trainable": {k: v.half() for k, v in trainable_state(g).items()}, "config": cfg, "peft_info": info},
                   os.path.join(out, "adapter.pt"))
    print(f"[{exp}] training stopped ({stop}) at step {step}; best dev loss {best['loss']:.4f}", flush=True)
    result = finish(g, cfg, data, tok, idf, len_ratio, out, result, device, t0)
    d = result.get("dev") or {}
    print(f"[{exp}] done in {result['minutes_total']} min: dev BERTScore {d.get('bertscore_f1')} ROUGE-1 "
          f"{d.get('rouge1')} ROUGE-L {d.get('rougeL')} words {d.get('words_mean')}", flush=True)
    del g, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def redecode(run_dir, overrides, name, data_root, roots, out_root=None, device=None):
    """Re-generate a finished PEFT run with new decoding / length settings (no training)."""
    t0 = time.time()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g, cfg = load_run_model(run_dir, roots, device)
    cfg.update(overrides)
    cfg["amp"] = cfg.get("amp", True) and device.type == "cuda"
    cfg["name"] = name
    out = os.path.join(out_root or os.path.dirname(os.path.normpath(run_dir)), name)
    os.makedirs(out, exist_ok=True)
    tok = load_tokenizer(cfg["tokenizer"] if os.path.exists(cfg.get("tokenizer", "")) else find_tokenizer(roots))
    data = D.load_data(data_root, cfg["tokenizer"] if os.path.exists(cfg.get("tokenizer", "")) else find_tokenizer(roots),
                       os.path.join(os.path.dirname(out), "_cache"))
    idf = D.IDF(data["train"]) if cfg["ctx"] in ("central", "oracle") else None
    lr = length_ratio(data, cfg)
    base = json.load(open(os.path.join(run_dir, "result.json")))
    result = {k: base.get(k) for k in ("backbone", "family", "method", "ctx", "budget", "kd", "params_total",
                                       "params_trainable", "train_minutes", "train_tok_s", "peak_mem_gib")}
    result.update(exp=name, decoded_from=os.path.basename(os.path.normpath(run_dir)), overrides=overrides, len_ratio=lr)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    return finish(g, cfg, data, tok, idf, lr, out, result, device, t0)


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
    ap.add_argument("--exp")
    ap.add_argument("--decode", help="run folder to re-decode (with --set and --name)")
    ap.add_argument("--name")
    ap.add_argument("--data", required=True)
    ap.add_argument("--roots", nargs="+", default=["results"])
    ap.add_argument("--out", default="runs_p6")
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args(argv)
    root = D.find_root(a.data)
    if root is None:
        sys.exit(f"no Phase_6_Summ data under {a.data}")
    if a.decode:
        return redecode(a.decode, parse_set(a.set), a.name or os.path.basename(a.decode) + "__dec", root, a.roots, a.out)
    return run(get_experiment(a.exp, parse_set(a.set)), root, a.roots, a.out, a.max_minutes)


if __name__ == "__main__":
    main()
