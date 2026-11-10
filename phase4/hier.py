"""Two-stage hierarchical models (ILDC-paper style) and section probes on the chunk vectors a finished run
extracted (train.py `extract=`: chunks_{train,dev,test}.npz, every chunk of a document, at most
max_chunks spread evenly over longer documents).

    python -m phase4.hier --run runs_p4/G_lorma [--variants mean attn gru q1 q2 q3 q4]

Stage 1 = the fine-tuned run (it read one view, e.g. the last 512 tokens). Stage 2 = a small aggregator
trained on the frozen chunk vectors of the *whole* document:
  mean / max      mean / max of all chunk vectors -> linear
  attn            attention pooling over all chunks (+ position embedding) -> linear
  gru             BiGRU over the chunk sequence + attention -> linear
  lastN / firstN  attention over the last / first N chunks only
  q1..q4          section probes: mean of the chunks whose centre lies in that quarter of the document
                  (nearest chunk when a short document has none there) -> linear
Selection uses 10% of the stage-1 training docs held out from stage 2 (early stopping), so the dev set is
only used for the final scores. Caveat: stage 1 was trained on the training documents, so
their chunk vectors are fitted features (the usual stacking caveat); dev/test vectors are not.

Each variant writes <out>/<run>__h_<variant>/ with result.json (same fields as train.py, so it shows in
the session summary and can join ensembles), scores_{dev,test}.npz, dev_pred.json and, for the
document-level variants, submission.zip / submission_thr.zip.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .score import breakdown, fit_threshold, prf, probs, write_submission
from .train import save_scores

VARIANTS = ["mean", "max", "attn", "gru", "last1", "last4", "first1", "first4", "q1", "q2", "q3", "q4"]
SUBMIT = {"mean", "attn", "gru", "last4"}
N_POS = 16


def load_chunks(path):
    z = np.load(path)
    return {"vecs": z["vecs"].astype(np.float32), "off": z["off"], "pos": z["pos"].astype(np.float32),
            "ids": [str(i) for i in z["ids"].tolist()], "y": z["y"].astype(np.int64), "n_tok": z["n_tok"]}


def select(pos, variant):
    """Indices (into one document's chunks) used by a variant."""
    k = len(pos)
    if variant in ("mean", "max", "attn", "gru"):
        return np.arange(k)
    if variant.startswith("last"):
        return np.arange(max(0, k - int(variant[4:])), k)
    if variant.startswith("first"):
        return np.arange(min(int(variant[5:]), k))
    if variant in ("q1", "q2", "q3", "q4"):
        q = int(variant[1]) - 1
        lo, hi = q / 4, (q + 1) / 4
        sel = np.where((pos >= lo) & ((pos < hi) if q < 3 else (pos <= hi)))[0]
        return sel if len(sel) else np.array([int(np.argmin(np.abs(pos - (lo + hi) / 2)))])
    raise ValueError(variant)


def kind_of(variant):
    if variant in ("mean", "max", "gru"):
        return variant
    return "mean" if variant.startswith("q") else "attn"


class Aggregator(nn.Module):
    def __init__(self, d, kind, hidden=256, dropout=0.1):
        super().__init__()
        self.kind = kind
        self.norm = nn.LayerNorm(d)
        self.pos = nn.Embedding(N_POS, d)
        nn.init.zeros_(self.pos.weight)
        if kind == "gru":
            self.gru = nn.GRU(d, hidden // 2, batch_first=True, bidirectional=True)
            d = hidden
        self.att = nn.Linear(d, 1) if kind in ("attn", "gru") else None
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, 2)

    def forward(self, X, P, M):
        """X (B, K, d) chunk vectors, P (B, K) relative positions, M (B, K) bool mask -> logits (B, 2)."""
        x = self.norm(X) + self.pos((P * (N_POS - 1)).round().long().clamp(0, N_POS - 1))
        m = M.float()
        if self.kind == "gru":
            lens = M.sum(1).clamp(min=1).cpu()
            pk = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
            x, _ = nn.utils.rnn.pad_packed_sequence(self.gru(pk)[0], batch_first=True, total_length=X.shape[1])
        if self.kind == "mean":
            v = (x * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        elif self.kind == "max":
            v = x.masked_fill(~M[..., None], -1e4).max(1).values
        else:
            a = torch.softmax(self.att(x).squeeze(-1).masked_fill(~M, -1e4), 1)
            v = (a[..., None] * x).sum(1)
        return self.out(self.drop(v))


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
            X[r, :len(s)] = C["vecs"][s]
            P[r, :len(s)] = C["pos"][s]
            M[r, :len(s)] = True
        yield ids, torch.from_numpy(X), torch.from_numpy(P), torch.from_numpy(M)


@torch.no_grad()
def predict(model, C, variant, device, bs=256):
    model.eval()
    out = np.zeros((len(C["ids"]), 2), np.float32)
    for ids, X, P, M in batches(C, np.arange(len(C["ids"])), variant, bs):
        out[ids] = model(X.to(device), P.to(device), M.to(device)).float().cpu().numpy()
    model.train()
    return out


def train_variant(tr, variant, device, epochs=30, patience=4, lr=1e-3, bs=64, seed=0, val_frac=0.1):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n = len(tr["ids"])
    val = np.zeros(n, bool)
    for c in (0, 1):
        ci = np.where(tr["y"] == c)[0]
        val[rng.choice(ci, size=max(1, int(round(val_frac * len(ci)))), replace=False)] = True
    fit_idx, val_idx = np.where(~val)[0], np.where(val)[0]
    model = Aggregator(tr["vecs"].shape[1], kind_of(variant)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    cnt = np.bincount(tr["y"][fit_idx], minlength=2).astype(np.float64)
    cw = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), dtype=torch.float32, device=device)
    best, bad, state, hist = -1.0, 0, None, []
    for ep in range(epochs):
        for ids, X, P, M in batches(tr, fit_idx, variant, bs, rng):
            y = torch.as_tensor(tr["y"][ids], device=device)
            loss = F.cross_entropy(model(X.to(device), P.to(device), M.to(device)), y, weight=cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        lg = np.zeros((len(val_idx), 2), np.float32)
        model.eval()
        with torch.no_grad():
            for j, (ids, X, P, M) in enumerate(batches(tr, val_idx, variant, 256)):
                lg[j * 256:j * 256 + len(ids)] = model(X.to(device), P.to(device), M.to(device)).float().cpu().numpy()
        model.train()
        f = prf(tr["y"][val_idx], lg.argmax(1))["macro_f1"]
        hist.append(round(f, 4))
        if f > best:
            best, bad, state = f, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return model, best, hist


def run_all(run_dir, out_root=None, variants=None, epochs=30, device=None):
    t0 = time.time()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = os.path.basename(os.path.normpath(run_dir))
    out_root = out_root or os.path.dirname(os.path.normpath(run_dir))
    files = {s: os.path.join(run_dir, f"chunks_{s}.npz") for s in ("train", "dev", "test")}
    missing = [f for f in files.values() if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"{run}: no chunk vectors ({missing[0]}); run it with extract=...")
    tr, dv, te = (load_chunks(files[s]) for s in ("train", "dev", "test"))
    base = json.load(open(os.path.join(run_dir, "result.json")))
    variants = variants or VARIANTS
    if isinstance(variants, str):
        variants = variants.split("+")
    results = {}
    for v in variants:
        t1 = time.time()
        model, val_f1, hist = train_variant(tr, v, device, epochs=epochs)
        lg = predict(model, dv, v, device)
        p1, pred = probs(lg), lg.argmax(1)
        thr, _ = fit_threshold(p1, dv["y"])
        name = f"{run}__h_{v}"
        out = os.path.join(out_root, name)
        os.makedirs(out, exist_ok=True)
        n_agg = sum(p.numel() for p in model.parameters())
        res = {"exp": name, "stage1": run, "backbone": base.get("backbone"), "method": base.get("method"),
               "ctx": f"hier2:{v}", "seq_len": base.get("seq_len"), "n_chunks": "all", "agg": kind_of(v),
               "params_trainable": n_agg, "params_total": (base.get("params_total") or 0) + n_agg,
               "train_docs": len(tr["ids"]), "val_f1": round(val_f1, 6), "val_hist": hist,
               "dev": prf(dv["y"], pred), "dev_thr": {"threshold": thr, **prf(dv["y"], (p1 >= thr).astype(np.int64))},
               "errors": breakdown(dv["y"], pred, p1, dv["n_tok"], dv["ids"]),
               "chunks_per_doc_dev": round(float(np.mean([len(select(dv["pos"][dv["off"][i]:dv["off"][i + 1]], v))
                                                           for i in range(len(dv["ids"]))])), 2)}
        save_scores(os.path.join(out, "scores_dev.npz"), dv["ids"], lg)
        with open(os.path.join(out, "dev_pred.json"), "w", encoding="utf-8") as f:
            json.dump([{"id": i, "gold": int(g), "pred": int(p), "p1": round(float(q), 5)}
                       for i, g, p, q in zip(dv["ids"], dv["y"], pred, p1)], f)
        lg_te = predict(model, te, v, device)
        save_scores(os.path.join(out, "scores_test.npz"), te["ids"], lg_te)
        if v in SUBMIT:
            write_submission(out, te["ids"], lg_te.argmax(1))
            write_submission(out, te["ids"], (probs(lg_te) >= thr).astype(np.int64), "submission_thr.zip")
        res["test_pos_rate"] = round(float(lg_te.argmax(1).mean()), 4)
        res["train_minutes"] = round((time.time() - t1) / 60, 2)
        with open(os.path.join(out, "result.json"), "w") as f:
            json.dump(res, f, indent=1)
        results[v] = res
        print(f"[{name}] dev macro-F1 {res['dev']['macro_f1']:.4f} (val {val_f1:.4f}, thr {thr:.2f} -> "
              f"{res['dev_thr']['macro_f1']:.4f})", flush=True)
    summ = {v: {"dev_f1": round(r["dev"]["macro_f1"], 4), "val_f1": r["val_f1"], "chunks_per_doc_dev": r["chunks_per_doc_dev"]}
            for v, r in results.items()}
    summ["stage1_dev_f1"] = (base.get("dev") or {}).get("macro_f1")
    summ["minutes"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(run_dir, "hier_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder with chunks_{train,dev,test}.npz")
    ap.add_argument("--out", default=None, help="where the <run>__h_<variant> folders go (default: next to the run)")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args(argv)
    return run_all(a.run, a.out, a.variants, a.epochs)


if __name__ == "__main__":
    main()
