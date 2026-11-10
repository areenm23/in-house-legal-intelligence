"""Phase-5 scoring: the official evaluate_phase5.py numbers (sklearn macro-F1 / micro-F1 over the label
dimension with zero_division=0, subset accuracy), per-label scores, threshold strategies, and
prediction.json writing + validation.

Thresholds (P = sigmoid scores, (n, L)):
  fixed      P >= 0.5
  global     one threshold t for every label, chosen for macro-F1
  per_label  one threshold per label, chosen for that label's F1 (labels with fewer than `min_pos`
             positives fall back to the global threshold)
  atleast1   any strategy + "if a document gets no label, give it its top-scoring label"
Honest estimates: thresholds are fitted on one half of the documents and scored on the other, both ways
(`cross_fit`), so the dev number is not fitted on itself. Final thresholds for test are fitted on all dev.
"""
import json
import os
import zipfile

import numpy as np

GRID = np.round(np.arange(0.02, 0.96, 0.02), 2)


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def counts(Y, P):
    """Y, P (n, L) 0/1 -> tp, fp, fn per label."""
    Y, P = np.asarray(Y, bool), np.asarray(P, bool)
    return (Y & P).sum(0), (~Y & P).sum(0), (Y & ~P).sum(0)


def f1_per_label(Y, P):
    tp, fp, fn = counts(Y, P)
    den = 2 * tp + fp + fn
    return np.divide(2 * tp, den, out=np.zeros(len(tp), dtype=np.float64), where=den > 0)


def metrics(Y, P, dim="official"):
    """Official numbers + extras. Y, P (n, L) 0/1. dim="official": like evaluate_phase5.py, the label
    dimension is (largest gold id present in Y) + 1, so labels above it are ignored; dim="all": all L."""
    Y, P = np.asarray(Y, np.uint8), np.asarray(P, np.uint8)
    if dim == "official" and Y.size:
        present_ids = np.where(Y.sum(0) > 0)[0]
        n_dim = int(present_ids.max()) + 1 if len(present_ids) else Y.shape[1]
        Y, P = Y[:, :n_dim], P[:, :n_dim]
    tp, fp, fn = counts(Y, P)
    f = f1_per_label(Y, P)
    mic = 2 * tp.sum() / max(1, 2 * tp.sum() + fp.sum() + fn.sum())
    sup = Y.sum(0)
    present = sup > 0
    return {"macro_f1": float(f.mean()) if len(f) else 0.0, "micro_f1": float(mic),
            "subset_acc": float((Y == P).all(1).mean()) if len(Y) else 0.0,
            "macro_f1_present": float(f[present].mean()) if present.any() else 0.0,
            "micro_p": float(tp.sum() / max(1, tp.sum() + fp.sum())), "micro_r": float(tp.sum() / max(1, tp.sum() + fn.sum())),
            "labels_per_doc_pred": float(P.sum(1).mean()) if len(P) else 0.0,
            "labels_per_doc_gold": float(Y.sum(1).mean()) if len(Y) else 0.0,
            "empty_pred_docs": int((P.sum(1) == 0).sum()), "n": int(len(Y)), "labels_present": int(present.sum())}


def per_label_table(Y, P, train_counts, buckets):
    tp, fp, fn = counts(Y, P)
    f = f1_per_label(Y, P)
    rows = [{"label": l, "train": int(train_counts[l]), "dev_support": int(tp[l] + fn[l]), "bucket": buckets[l],
             "f1": round(float(f[l]), 4), "tp": int(tp[l]), "fp": int(fp[l]), "fn": int(fn[l])} for l in range(len(f))]
    by = {}
    for b in ("frequent", "medium", "rare", "unseen"):
        m = [r["f1"] for r in rows if r["bucket"] == b and r["dev_support"] > 0]
        by[b] = {"labels": len(m), "macro_f1": round(float(np.mean(m)), 4) if m else None}
    return rows, by


def decide(Pr, thr, atleast1=False):
    """Pr (n, L) probabilities, thr scalar or (L,) -> 0/1 matrix."""
    D = (Pr >= np.asarray(thr)).astype(np.uint8)
    if atleast1 and len(D):
        empty = D.sum(1) == 0
        D[np.where(empty)[0], Pr[empty].argmax(1)] = 1
    return D


def fit_global(Y, Pr, atleast1=False):
    best = (-1.0, 0.5)
    for t in GRID:
        f = metrics(Y, decide(Pr, t, atleast1))["macro_f1"]
        if f > best[0] + 1e-12:
            best = (f, float(t))
    return best[1], best[0]


def fit_per_label(Y, Pr, t_global, min_pos=3):
    """Per-label threshold maximising that label's F1 (ties -> the global threshold side)."""
    Y = np.asarray(Y, bool)
    L = Y.shape[1]
    thr = np.full(L, t_global, dtype=np.float64)
    for l in range(L):
        if Y[:, l].sum() < min_pos:
            continue
        p, y = Pr[:, l], Y[:, l]
        best = (-1.0, t_global)
        for t in GRID:
            d = p >= t
            tp, fp, fn = (d & y).sum(), (d & ~y).sum(), (~d & y).sum()
            f = 2 * tp / max(1, 2 * tp + fp + fn)
            if f > best[0] + 1e-12 or (abs(f - best[0]) <= 1e-12 and abs(t - t_global) < abs(best[1] - t_global)):
                best = (f, float(t))
        thr[l] = best[1]
    return thr


def fit_thresholds(Y, Pr, min_pos=3):
    """-> dict of strategy -> threshold(s) fitted on (Y, Pr)."""
    tg, _ = fit_global(Y, Pr)
    tg1, _ = fit_global(Y, Pr, atleast1=True)
    return {"fixed": 0.5, "global": tg, "global_atleast1": tg1, "per_label": fit_per_label(Y, Pr, tg, min_pos)}


def apply(strategy, thr, Pr):
    return decide(Pr, thr, atleast1=strategy.endswith("atleast1"))


def cross_fit(Y, Pr, seed=0, min_pos=3):
    """2-fold honest estimate of every strategy's macro-F1: fit on one half, score the other, both ways;
    the two halves' decisions are pooled into one prediction matrix and scored once."""
    n = len(Y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    halves = [perm[: n // 2], perm[n // 2:]]
    pooled = {}
    for a, b in ((0, 1), (1, 0)):
        th = fit_thresholds(Y[halves[a]], Pr[halves[a]], min_pos)
        for s, t in th.items():
            pooled.setdefault(s, np.zeros_like(Y, dtype=np.uint8))[halves[b]] = apply(s, t, Pr[halves[b]])
    return {s: metrics(Y, D) for s, D in pooled.items()}


def to_prediction_json(ids, D, max_id=None):
    out = []
    for i, row in zip(ids, D):
        lab = [int(x) for x in np.where(row)[0] if max_id is None or x <= max_id]
        out.append({"id": str(i), "pred": lab})
    return out


def validate(pj, ids, n_labels=None):
    """Official checks: non-empty list, one record per test id, string ids, unique non-negative int labels
    (below n_labels when given)."""
    if not isinstance(pj, list) or not pj:
        raise ValueError("prediction must be a non-empty JSON list")
    seen = set()
    for n, r in enumerate(pj):
        if not isinstance(r, dict) or "id" not in r or "pred" not in r:
            raise ValueError(f"item {n} must contain id and pred")
        if not isinstance(r["id"], str) or not isinstance(r["pred"], list) or r["id"] in seen:
            raise ValueError(f"invalid or duplicate item {n}")
        labs = r["pred"]
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 or (n_labels and x >= n_labels) for x in labs) \
                or len(set(labs)) != len(labs):
            raise ValueError(f"invalid/duplicate label in item {n}: {labs}")
        seen.add(r["id"])
    want = {str(i) for i in ids}
    if seen != want:
        raise ValueError(f"id mismatch: missing {len(want - seen)}, extra {len(seen - want)}")
    return True


def write_submission(out_dir, ids, D, max_id, name="submission.zip"):
    pj = to_prediction_json(ids, D, max_id)
    validate(pj, ids, max_id + 1)
    path = os.path.join(out_dir, name.replace("submission", "prediction").replace(".zip", ".json"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pj, f)
    with zipfile.ZipFile(os.path.join(out_dir, name), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, "prediction.json")
    return {"docs": len(pj), "labels_per_doc": round(sum(len(r["pred"]) for r in pj) / max(1, len(pj)), 3),
            "empty": sum(not r["pred"] for r in pj)}


def write_truth(ids, label_lists, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"id": str(i), "gold": [int(x) for x in g]} for i, g in zip(ids, label_lists)], f)
