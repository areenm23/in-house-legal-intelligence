"""Train / evaluate / rank one prior-case-retrieval experiment (dense bi-encoder on a Phase-1 backbone).

    python -m phase7.train --exp G_lorma --data DATA_ROOT --roots /kaggle/input results --out runs_p7 [--set ...]

Encoder: phase4.model.CJPEModel's chunk encoder (K chunks of seq_len tokens, pooled, aggregated) + a linear
projection, L2-normalised; query and candidate share it. Similarity = cosine / learned temperature.
Loss: InfoNCE per query over (its sampled positive, every other in-batch document, `n_hard` BM25 hard negatives
per query); other relevant candidates of the same query are masked out. KD (kd=dict(teacher=RUN, alpha, T)):
KL from the teacher's similarity distribution over the same batch (teacher embeddings of every train query /
candidate are saved by the teacher run, save_emb=True) to the student's.
Ranking: dense scores, BM25 scores, and a hybrid alpha * z(dense) + (1 - alpha) * z(BM25) with alpha chosen on
dev. Zips: submission.zip (hybrid), submission_dense.zip. Model selection: dev micro-F1@7 (dense) on the full dev
pool with eval_chunks_fast chunks per document.
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
from ft.peft import AdaLoRALinear, RankAllocator, apply_method, peft_extra_loss, trainable_state
from phase3.train import find_tokenizer, sha12
from phase4.data import views
from phase4.model import CJPEModel
from . import data as D
from .experiments import get_experiment
from .score import BM25, at_k, diagnostics, hard_negatives, top_k, write_submission, zscore

TOKENIZER_SHA = "0c9231a52bd5"


class BiEncoder(nn.Module):
    def __init__(self, bb, proj_dim=256, pool="mean", agg="mean", dropout=0.0, tau=0.05):
        super().__init__()
        self.enc = CJPEModel(bb, pool=pool, agg=agg, dropout=dropout, n_classes=1)
        self.proj = nn.Linear(self.enc.d_doc, proj_dim)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))

    @property
    def bb(self):
        return self.enc.bb

    def embed(self, vs, device, amp, seq_batch):
        _, Dv = self.enc(vs, device, amp, seq_batch)
        return F.normalize(self.proj(Dv.float()), dim=-1)

    def scale(self):
        return torch.exp(-self.log_tau).clamp(max=100.0)


def doc_views(store, rows, cfg, train, rng=None):
    k = cfg["train_chunks"] if train else cfg["eval_chunks"]
    pick = cfg["train_pick"] if train else cfg["eval_pick"]
    return [views(store.doc(i), "hier", cfg["seq_len"], k=k, pick=pick, rng=rng) for i in rows]


@torch.no_grad()
def embed_all(model, store, cfg, device, chunks=None):
    was = model.training
    model.eval()
    c = dict(cfg, eval_chunks=chunks or cfg["eval_chunks"])
    step = max(1, c["eval_batch"] // max(1, c["eval_chunks"]))
    out = []
    for j in range(0, len(store), step):
        rows = range(j, min(len(store), j + step))
        out.append(model.embed(doc_views(store, rows, c, False), device, c["amp"], c["eval_batch"]).float().cpu().numpy())
    model.train(was)
    return np.concatenate(out) if out else np.zeros((0, model.proj.out_features), np.float32)


def find_run_file(out_root, run, fname):
    cands = [os.path.join(out_root, run, fname)]
    cands += sorted(glob.glob(os.path.join("/kaggle/input", "**", "runs_p7", run, fname), recursive=True))
    return next((c for c in cands if os.path.exists(c)), None)


def gold_of(split):
    return {q: [c for c in (split.rel[i] or [])] for i, q in enumerate(split.queries.ids)}


def evaluate(scores, split):
    pred = top_k(scores, split.queries.ids, split.pool.ids)
    res = at_k(pred, gold_of(split))
    res.update(diagnostics(scores, [split.rel_rows(q) for q in range(len(split.queries))]))
    return res, pred


def best_alpha(dense, bm, split, grid=np.round(np.linspace(0, 1, 11), 2)):
    zd, zb = zscore(dense), zscore(bm)
    best = (-1.0, 1.0)
    for a in grid:
        f = at_k(top_k(a * zd + (1 - a) * zb, split.queries.ids, split.pool.ids), gold_of(split))["f1"]
        if f > best[0] + 1e-12:
            best = (f, float(a))
    return best[1], best[0]


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
    if dv is None:
        raise FileNotFoundError("Phase 7 needs Dev/dev.json + Dev/dev_candidates.json")
    t1 = time.time()
    bm = {s: BM25(d.pool, cfg["bm25_k1"], cfg["bm25_b"], cfg["bm25_max_tokens"]).scores(d.queries)
          for s, d in (("train", tr), ("dev", dv), ("test", te))}
    q_rows = [q for q in range(len(tr.queries)) if tr.rel_rows(q)]
    if cfg["max_train_queries"]:
        q_rows = sorted(rng.choice(q_rows, size=min(cfg["max_train_queries"], len(q_rows)), replace=False).tolist())
    hard = hard_negatives(bm["train"], [tr.rel_rows(q) for q in range(len(tr.queries))], tr.queries.ids, tr.pool.ids,
                          n=cfg["hard_pool"], skip=cfg["hard_skip"])
    bm25_dev, _ = evaluate(bm["dev"], dv)
    print(f"[{exp}] BM25 ({time.time() - t1:.0f}s): dev F1@7 {bm25_dev['f1']:.4f} R@100 {bm25_dev['recall@100']}; "
          f"{len(q_rows)} training queries, pools train {len(tr.pool)} dev {len(dv.pool)} test {len(te.pool)}", flush=True)

    teacher = None
    if cfg.get("kd"):
        src = find_run_file(out_root, cfg["kd"]["teacher"], "emb_train.npz")
        if src is None:
            raise FileNotFoundError(f"KD teacher embeddings not found for {cfg['kd']['teacher']} (save_emb=True)")
        z = np.load(src)
        if [str(x) for x in z["q_ids"]] != tr.queries.ids or [str(x) for x in z["p_ids"]] != tr.pool.ids:
            raise RuntimeError("teacher embeddings do not match this training split")
        teacher = {"q": torch.from_numpy(z["q"].astype(np.float32)).to(device),
                   "p": torch.from_numpy(z["p"].astype(np.float32)).to(device), "scale": float(z["scale"]), "src": src}
        print(f"KD teacher {src}", flush=True)

    bb = load_backbone(cfg["backbone"], roots, random_init=cfg.get("random_init", False), tiny=cfg.get("tiny", False))
    if cfg["bidir"]:
        bb.set_bidirectional(True)
    bb.to(device)
    info = apply_method(bb, cfg["method"], **dict(cfg.get("method_args", {})))
    model = BiEncoder(bb, cfg["proj_dim"], cfg["pool"], cfg["agg"], cfg["dropout"], cfg["tau"]).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{exp}] {bb.name} ({bb.ckpt_path}) {info} -> trainable {n_train:,} / {n_total:,}", flush=True)
    groups = {"bb_decay": [], "bb_nodecay": [], "head": []}
    for n, p in model.named_parameters():
        if p.requires_grad:
            k = "head" if not n.startswith("enc.bb.") else ("bb_nodecay" if p.ndim < 2 or "prompt" in n else "bb_decay")
            groups[k].append(p)
    opt = torch.optim.AdamW([g for g in ({"params": groups["bb_decay"], "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
                                         {"params": groups["bb_nodecay"], "lr": cfg["lr"], "weight_decay": 0.0},
                                         {"params": groups["head"], "lr": cfg["head_lr"], "weight_decay": 0.0}) if g["params"]],
                            betas=(0.9, 0.98), eps=1e-6)
    Bq = cfg["q_batch"]
    steps_per_epoch = math.ceil(len(q_rows) / Bq)
    plan = {"total": max(1, steps_per_epoch * cfg["epochs"])}
    warm = lambda: max(1, int(cfg["warmup"] * plan["total"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm() if s < warm() else max(0.0, (plan["total"] - s) / max(1, plan["total"] - warm())))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    ada = [m for m in model.modules() if isinstance(m, AdaLoRALinear)]
    allocator = RankAllocator(ada, info.get("target_r", 4), plan["total"]) if ada else None
    json.dump({**cfg, "peft_info": info, "tokenizer": tok_path, "backbone_ckpt": bb.ckpt_path, "train_queries": len(q_rows),
               "bm25_dev": bm25_dev}, open(os.path.join(out, "config.json"), "w"), indent=1)
    logf = open(os.path.join(out, "log.jsonl"), "a")

    def log(rec):
        rec["t"] = round(time.time() - t0, 1)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    def eval_dense(chunks):
        e_q = embed_all(model, dv.queries, cfg, device, chunks)
        e_p = embed_all(model, dv.pool, cfg, device, chunks)
        return e_q @ e_p.T

    budget_s = cfg["budget_min"] * 60 if cfg.get("budget_min") else None
    best = {"f1": -1.0, "step": 0, "state": None}
    train_names = {n for n, p in model.named_parameters() if p.requires_grad}
    step = tokens = bad = 0
    t_train, times = 0.0, []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    next_eval = max(1, plan["total"] // cfg["n_evals"])
    stop, seq_batch, ep = None, cfg["seq_batch"], 0
    model.train()
    while stop is None and ep < cfg["epochs"]:
        ep += 1
        perm = rng.permutation(q_rows)
        for b0 in range(0, len(perm), Bq):
            if step >= plan["total"]:
                stop = "plan"
                break
            qs = perm[b0:b0 + Bq]
            pos = [int(rng.choice(tr.rel_rows(q))) for q in qs]
            negs = [int(j) for q in qs for j in (rng.choice(hard[q], size=min(cfg["n_hard"], len(hard[q])), replace=False)
                                                  if hard[q] and cfg["n_hard"] else [])]
            docs = list(dict.fromkeys(pos + negs))
            col = {d: k for k, d in enumerate(docs)}
            ts = time.time()
            while True:
                opt.zero_grad(set_to_none=True)
                try:
                    eq = model.embed(doc_views(tr.queries, qs, cfg, True, rng), device, cfg["amp"], seq_batch)
                    ed = model.embed(doc_views(tr.pool, docs, cfg, True, rng), device, cfg["amp"], seq_batch)
                    S = (eq @ ed.T) * model.scale()
                    mask = torch.zeros_like(S, dtype=torch.bool)
                    for r, q in enumerate(qs):
                        for j in tr.rel_rows(q):
                            if j in col and col[j] != col[pos[r]]:
                                mask[r, col[j]] = True
                    S = S.masked_fill(mask, -1e4)
                    target = torch.tensor([col[p] for p in pos], device=device)
                    loss = F.cross_entropy(S, target)
                    if teacher is not None:
                        St = (teacher["q"][torch.as_tensor(qs, device=device)] @ teacher["p"][torch.as_tensor(docs, device=device)].T) \
                            * teacher["scale"]
                        St = St.masked_fill(mask, -1e4)
                        T = cfg["kd"].get("T", 1.0)
                        kl = F.kl_div(F.log_softmax(S / T, -1), F.log_softmax(St / T, -1), log_target=True,
                                      reduction="batchmean") * T * T
                        a = cfg["kd"].get("alpha", 0.5)
                        loss = a * loss + (1 - a) * kl
                    reg = peft_extra_loss(model.bb)
                    if reg is not None:
                        loss = loss + reg
                    scaler.scale(loss).backward()
                    break
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if cfg["train_chunks"] <= 1 and len(qs) <= 2:
                        raise
                    if cfg["train_chunks"] > 1:
                        cfg["train_chunks"] -= 1
                    else:
                        qs, pos = qs[:len(qs) // 2], pos[:len(pos) // 2]
                    print(f"OOM -> train_chunks {cfg['train_chunks']}, queries per step {len(qs)}", flush=True)
            scaler.unscale_(opt)
            if allocator is not None:
                allocator.step(step + 1)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            tokens += (len(qs) + len(docs)) * cfg["train_chunks"] * cfg["seq_len"]
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
                    print(f"plan: {sps:.2f} s/step -> {plan['total']} steps ({plan['total'] / steps_per_epoch:.2f} epochs)",
                          flush=True)
            if step % cfg["log_every"] == 0:
                log({"step": step, "epoch": ep, "loss": round(float(loss.detach()), 4), "tau": round(float(1 / model.scale()), 4),
                     "lr": sched.get_last_lr()[0]})
            if step >= next_eval or step >= plan["total"]:
                res, _ = evaluate(eval_dense(cfg["eval_chunks_fast"]), dv)
                log({"step": step, "epoch": round(step / steps_per_epoch, 3), "dev_f1@7": round(res["f1"], 4),
                     "dev_r@100": res["recall@100"], "plan": plan["total"]})
                if res["f1"] > best["f1"]:
                    best = {"f1": res["f1"], "step": step,
                            "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in train_names}}
                    bad = 0
                else:
                    bad += 1
                next_eval = step + max(1, plan["total"] // cfg["n_evals"])
                if cfg["patience"] and bad >= cfg["patience"]:
                    stop = "patience"
                    break
            if max_minutes and time.time() - t0 > max_minutes * 60 - cfg["reserve_min"] * 60:
                stop = "deadline"
                break
    stop = stop or "epochs"
    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=False)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None

    t1 = time.time()
    dense_dev = eval_dense(cfg["eval_chunks"])
    infer_s = time.time() - t1
    r_dense, _ = evaluate(dense_dev, dv)
    alpha, _ = best_alpha(dense_dev, bm["dev"], dv)
    hyb_dev = alpha * zscore(dense_dev) + (1 - alpha) * zscore(bm["dev"])
    r_hyb, _ = evaluate(hyb_dev, dv)
    result = {"exp": exp, "backbone": cfg["backbone"], "method": cfg["method"], "kd": cfg.get("kd"), "pool": cfg["pool"],
              "agg": cfg["agg"], "seq_len": cfg["seq_len"], "train_chunks": cfg["train_chunks"], "eval_chunks": cfg["eval_chunks"],
              "n_hard": cfg["n_hard"], "params_total": n_total, "params_trainable": n_train, "peft_info": info,
              "train_queries": len(q_rows), "steps": step, "epochs_seen": round(step / max(1, steps_per_epoch), 3),
              "stop": stop, "best_step": best["step"], "best_eval_f1": round(best["f1"], 6),
              "train_tok_s": round(tokens / max(t_train, 1e-6), 1), "train_minutes": round(t_train / 60, 2),
              "peak_mem_gib": peak, "embed_docs_s": round((len(dv.queries) + len(dv.pool)) / max(infer_s, 1e-6), 2),
              "dev_bm25": bm25_dev, "dev_dense": r_dense, "alpha": alpha, "dev_hybrid": r_hyb, "tau": float(1 / model.scale().detach())}
    np.savez_compressed(os.path.join(out, "scores_dev.npz"), dense=dense_dev.astype(np.float16), bm25=bm["dev"].astype(np.float16),
                        q_ids=np.array(dv.queries.ids), p_ids=np.array(dv.pool.ids))
    json.dump(result, open(os.path.join(out, "result.json"), "w"), indent=1)
    if cfg["predict_test"]:
        dense_te = embed_all(model, te.queries, cfg, device) @ embed_all(model, te.pool, cfg, device).T
        hyb_te = alpha * zscore(dense_te) + (1 - alpha) * zscore(bm["test"])
        write_submission(out, top_k(hyb_te, te.queries.ids, te.pool.ids), te.queries.ids, te.pool.ids)
        write_submission(out, top_k(dense_te, te.queries.ids, te.pool.ids), te.queries.ids, te.pool.ids, "submission_dense.zip")
        np.savez_compressed(os.path.join(out, "scores_test.npz"), dense=dense_te.astype(np.float16),
                            bm25=bm["test"].astype(np.float16), q_ids=np.array(te.queries.ids), p_ids=np.array(te.pool.ids))
    if cfg.get("save_emb"):
        np.savez_compressed(os.path.join(out, "emb_train.npz"), q=embed_all(model, tr.queries, cfg, device).astype(np.float16),
                            p=embed_all(model, tr.pool, cfg, device).astype(np.float16), q_ids=np.array(tr.queries.ids),
                            p_ids=np.array(tr.pool.ids), scale=float(model.scale()))
    if cfg.get("save_weights", "auto") is True or (cfg.get("save_weights", "auto") == "auto" and n_train < 20e6):
        torch.save({"trainable": {k: v.half() for k, v in trainable_state(model).items()}, "config": cfg}, os.path.join(out, "adapter.pt"))
    result["minutes_total"] = round((time.time() - t0) / 60, 2)
    json.dump(result, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(f"[{exp}] done in {result['minutes_total']} min: dev F1@7 dense {r_dense['f1']:.4f}, hybrid {r_hyb['f1']:.4f} "
          f"(alpha {alpha}), BM25 {bm25_dev['f1']:.4f}", flush=True)
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
    ap.add_argument("--data", required=True)
    ap.add_argument("--roots", nargs="+", default=["results"])
    ap.add_argument("--out", default="runs_p7")
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args(argv)
    root = D.find_root(a.data)
    if root is None:
        sys.exit(f"no Phase_7_PCR data under {a.data}")
    run(get_experiment(a.exp, parse_set(a.set)), root, a.roots, a.out, a.max_minutes)


if __name__ == "__main__":
    main()
