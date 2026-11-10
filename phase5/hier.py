"""Two-stage hierarchical statute identification and section probes on the chunk vectors a finished run
extracted (train.py `extract=`: chunks_{train,dev,test}.npz with label lists). Multi-label version of
phase4.hier: same variants (mean / max / attn / gru / lastN / firstN / q1..q4), a BCE head with one
output per statute (sqrt positive weights, prior-initialised bias), early stopping on 10% of the
stage-2 training documents (macro-F1 at the best global threshold), thresholds cross-fitted on clean dev
exactly as in train.py.

    python -m phase5.hier --run runs_p5/G_lorma [--variants mean attn gru q1 q2 q3 q4]

Writes <out>/<run>__h_<variant>/ like a normal run (result.json, scores_{dev,test}.npz, dev_pred.json,
and zips for the document-level variants).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase4.hier import Aggregator as _Agg, kind_of, select
from phase4.train import save_scores
from .score import apply, fit_global, sigmoid, write_submission
from .train import final_scores
from .data import freq_buckets

VARIANTS = ["mean", "max", "attn", "gru", "last1", "first1", "first4", "q1", "q2", "q3", "q4"]
SUBMIT = {"mean", "max", "attn", "gru"}


class Aggregator(_Agg):
    def __init__(self, d, kind, n_labels, hidden=256, dropout=0.1):
        super().__init__(d, kind, hidden, dropout)
        self.out = nn.Linear(self.out.in_features, n_labels)


def load_chunks(path, L):
    z = np.load(path)
    lab_off, lab = z["lab_off"], z["lab"]
    Y = np.zeros((len(z["ids"]), L), np.uint8)
    for i in range(len(Y)):
        lb = lab[lab_off[i]:lab_off[i + 1]]
        Y[i, lb[lb < L]] = 1
    return {"vecs": z["vecs"].astype(np.float32), "off": z["off"], "pos": z["pos"].astype(np.float32),
            "ids": [str(i) for i in z["ids"].tolist()], "Y": Y, "n_tok": z["n_tok"]}


def batches(C, docs, variant, bs, rng=None):
    order = np.asarray(docs) if rng is None else rng.permutation(docs)
    d = C["vecs"].shape[1]
    for b in range(0, len(order), bs):
        ids = order[b:b + bs]
        sels = [select(C["pos"][C["off"][i]:C["off"][i + 1]], variant) + C["off"][i] for i in ids]
        K = max(len(s) for s in sels)
        X = np.zeros((len(ids), K, d), np.float32)
        P = np.zeros((len(ids), K), np.float32)
        M = np.zeros((len(ids), K), bool)
        for r, s in enumerate(sels):
            X[r, :len(s)], P[r, :len(s)], M[r, :len(s)] = C["vecs"][s], C["pos"][s], True
        yield ids, torch.from_numpy(X), torch.from_numpy(P), torch.from_numpy(M)


@torch.no_grad()
def predict(model, C, variant, device, idx=None, bs=256):
    model.eval()
    idx = np.arange(len(C["ids"])) if idx is None else np.asarray(idx)
    out = np.zeros((len(idx), model.out.out_features), np.float32)
    k = 0
    for ids, X, P, M in batches(C, idx, variant, bs):
        out[k:k + len(ids)] = model(X.to(device), P.to(device), M.to(device)).float().cpu().numpy()
        k += len(ids)
    model.train()
    return out


def train_variant(tr, variant, device, L, epochs=40, patience=5, lr=1e-3, bs=64, seed=0, val_frac=0.1):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n = len(tr["ids"])
    perm = rng.permutation(n)
    val_idx, fit_idx = np.sort(perm[:max(1, int(val_frac * n))]), np.sort(perm[max(1, int(val_frac * n)):])
    model = Aggregator(tr["vecs"].shape[1], kind_of(variant), L).to(device)
    Yf = tr["Y"][fit_idx]
    prior = np.clip(Yf.mean(0), 1e-4, 0.5)
    with torch.no_grad():
        model.out.bias.copy_(torch.tensor(np.log(prior / (1 - prior)), dtype=torch.float32))
    pos = Yf.sum(0).astype(np.float64)
    pw = torch.tensor(np.clip(np.sqrt(np.where(pos > 0, (len(Yf) - pos) / np.maximum(pos, 1), 1)), 1, 30),
                      dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best, bad, state, hist = -1.0, 0, None, []
    for _ep in range(epochs):
        for ids, X, P, M in batches(tr, fit_idx, variant, bs, rng):
            y = torch.as_tensor(tr["Y"][ids].astype(np.float32), device=device)
            loss = F.binary_cross_entropy_with_logits(model(X.to(device), P.to(device), M.to(device)), y,
                                                      pos_weight=pw, reduction="none").sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        _t, f = fit_global(tr["Y"][val_idx], sigmoid(predict(model, tr, variant, device, val_idx)))
        hist.append(round(f, 4))
        if f > best:
            best, bad, state = f, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return model, best, hist


def run_all(run_dir, out_root=None, variants=None, epochs=40, device=None):
    t0 = time.time()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = os.path.basename(os.path.normpath(run_dir))
    out_root = out_root or os.path.dirname(os.path.normpath(run_dir))
    base = json.load(open(os.path.join(run_dir, "result.json")))
    L, max_id = base["n_labels"], base["max_pred_id"]
    files = {s: os.path.join(run_dir, f"chunks_{s}.npz") for s in ("train", "dev", "test")}
    missing = [f for f in files.values() if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"{run}: no chunk vectors ({missing[0]}); run it with extract=...")
    tr, dv, te = (load_chunks(files[s], L) for s in ("train", "dev", "test"))
    train_counts = tr["Y"].sum(0)
    buckets = freq_buckets(train_counts)
    variants = variants or VARIANTS
    if isinstance(variants, str):
        variants = variants.split("+")
    results = {}
    for v in variants:
        t1 = time.time()
        model, val_f1, hist = train_variant(tr, v, device, L, epochs=epochs)
        lg = predict(model, dv, v, device)
        fs, th, Pr = final_scores(dv["Y"], lg, L, train_counts, buckets)
        name = f"{run}__h_{v}"
        out = os.path.join(out_root, name)
        os.makedirs(out, exist_ok=True)
        n_agg = sum(p.numel() for p in model.parameters())
        res = {"exp": name, "stage1": run, "backbone": base.get("backbone"), "method": base.get("method"),
               "ctx": f"hier2:{v}", "seq_len": base.get("seq_len"), "loss": "bce", "n_labels": L, "max_pred_id": max_id,
               "params_trainable": n_agg, "params_total": (base.get("params_total") or 0) + n_agg,
               "train_docs": len(tr["ids"]), "val_f1": round(val_f1, 6), "val_hist": hist, **fs}
        save_scores(os.path.join(out, "scores_dev.npz"), dv["ids"], lg)
        Dg = apply("global", th["global"], Pr)
        with open(os.path.join(out, "dev_pred.json"), "w", encoding="utf-8") as f:
            json.dump([{"id": i, "gold": [int(x) for x in np.where(g)[0]], "pred": [int(x) for x in np.where(p)[0]]}
                       for i, g, p in zip(dv["ids"], dv["Y"], Dg)], f)
        lg_te = predict(model, te, v, device)
        save_scores(os.path.join(out, "scores_test.npz"), te["ids"], lg_te)
        if v in SUBMIT:
            Pt = sigmoid(lg_te)
            res["test"] = {"global": write_submission(out, te["ids"], apply("global", th["global"], Pt), max_id),
                           "per_label": write_submission(out, te["ids"], apply("per_label", th["per_label"], Pt), max_id,
                                                         "submission_pl.zip")}
        res["train_minutes"] = round((time.time() - t1) / 60, 2)
        with open(os.path.join(out, "result.json"), "w") as f:
            json.dump(res, f, indent=1)
        results[v] = res
        print(f"[{name}] clean-dev macro-F1 cv global {res['dev_cv']['global']['macro_f1']:.4f}, cv per-label "
              f"{res['dev_cv']['per_label']['macro_f1']:.4f} (val {val_f1:.4f})", flush=True)
    summ = {v: {"f1_cv_global": round(r["dev_cv"]["global"]["macro_f1"], 4),
                "f1_cv_perlabel": round(r["dev_cv"]["per_label"]["macro_f1"], 4), "val_f1": r["val_f1"]}
            for v, r in results.items()}
    summ["stage1_f1_cv_global"] = ((base.get("dev_cv") or {}).get("global") or {}).get("macro_f1")
    summ["minutes"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(run_dir, "hier_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args(argv)
    return run_all(a.run, a.out, a.variants, a.epochs)


if __name__ == "__main__":
    main()
