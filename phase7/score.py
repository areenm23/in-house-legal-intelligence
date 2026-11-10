"""Phase-7 scoring (official evaluate_phase7.py definitions: micro P / R / F1 over the top-7 of every query,
against the full gold set), ranking diagnostics, BM25 over A-1 token ids, rank fusion, prediction.json writing
and validation (exactly 7 unique ids from the split's pool per query, every query present).
"""
import json
import os
import zipfile

import numpy as np
import scipy.sparse as sp

K = 7


def at_k(pred, gold, k=K):
    """pred / gold: {qid: [ids]} -> official micro numbers."""
    tp = fp = fn = 0
    for q, g in gold.items():
        g, p = set(g), set(pred[q][:k])
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    return {"f1": 2 * tp / max(1, 2 * tp + fp + fn), "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "tp": tp, "fp": fp, "fn": fn}


def diagnostics(scores, rel_rows, ks=(7, 20, 100)):
    """scores (n_q, n_p), rel_rows: list of relevant pool rows -> recall@k (macro over queries), MRR, MAP."""
    order = np.argsort(-scores, axis=1)
    out = {f"recall@{k}": [] for k in ks}
    mrr, ap = [], []
    for q, rr in enumerate(rel_rows):
        if not rr:
            continue
        rank = np.empty(scores.shape[1], dtype=np.int64)
        rank[order[q]] = np.arange(scores.shape[1])
        rk = np.sort(rank[rr])
        for k in ks:
            out[f"recall@{k}"].append(float((rk < k).mean()))
        mrr.append(1.0 / (rk[0] + 1))
        ap.append(float(np.mean([(j + 1) / (r + 1) for j, r in enumerate(rk)])))
    res = {k: round(float(np.mean(v)), 4) if v else None for k, v in out.items()}
    res["mrr"] = round(float(np.mean(mrr)), 4) if mrr else None
    res["map"] = round(float(np.mean(ap)), 4) if ap else None
    return res


def top_k(scores, q_ids, p_ids, k=K):
    """scores (n_q, n_p) -> {qid: k pool ids}, a query's own id excluded."""
    pidx = {d: i for i, d in enumerate(p_ids)}
    s = scores.copy()
    for qi, q in enumerate(q_ids):
        if q in pidx:
            s[qi, pidx[q]] = -np.inf
    part = np.argpartition(-s, min(k, s.shape[1] - 1), axis=1)[:, :k + 1]
    out = {}
    for qi, q in enumerate(q_ids):
        cand = part[qi][np.argsort(-s[qi, part[qi]])]
        out[q] = [p_ids[j] for j in cand if np.isfinite(s[qi, j])][:k]
    return out


def zscore(s):
    mu, sd = s.mean(1, keepdims=True), s.std(1, keepdims=True)
    return (s - mu) / np.maximum(sd, 1e-9)


def term_matrix(store, rows=None, V=32768, max_tokens=None):
    rows = range(len(store)) if rows is None else rows
    data, ind, ptr = [], [], [0]
    for i in rows:
        t = store.doc(i)
        if max_tokens:
            t = t[:max_tokens]
        u, c = np.unique(t.astype(np.int64), return_counts=True)
        ind.append(u)
        data.append(c)
        ptr.append(ptr[-1] + len(u))
    return sp.csr_matrix((np.concatenate(data).astype(np.float32) if data else np.zeros(0, np.float32),
                          np.concatenate(ind) if ind else np.zeros(0, np.int64), np.array(ptr)), shape=(len(ptr) - 1, V))


class BM25:
    """Okapi BM25 over a pool; queries use their unique terms (binary) or counts."""

    def __init__(self, pool_store, k1=1.2, b=0.75, max_tokens=None):
        C = term_matrix(pool_store, max_tokens=max_tokens)
        N = C.shape[0]
        df = np.bincount(C.indices, minlength=C.shape[1])
        self.idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        dl = np.asarray(C.sum(1)).ravel()
        Kd = k1 * (1 - b + b * dl / max(1e-9, dl.mean()))
        C = C.tocoo()
        w = C.data * (k1 + 1) / (C.data + Kd[C.row]) * self.idf[C.col]
        self.W = sp.csr_matrix((w.astype(np.float32), (C.row, C.col)), shape=C.shape).T.tocsr()
        self.max_tokens = max_tokens

    def scores(self, query_store, binary=True):
        Q = term_matrix(query_store, max_tokens=self.max_tokens)
        if binary:
            Q.data[:] = 1.0
        return np.asarray((Q @ self.W).todense(), dtype=np.float32)


def hard_negatives(scores, rel_rows, q_ids, p_ids, n=20, skip=0):
    """Top-n non-relevant pool rows per query (after `skip` ranks), excluding the query itself."""
    pidx = {d: i for i, d in enumerate(p_ids)}
    out = []
    order = np.argsort(-scores, axis=1)
    for qi, rr in enumerate(rel_rows):
        bad = set(rr) | ({pidx[q_ids[qi]]} if q_ids[qi] in pidx else set())
        out.append([int(j) for j in order[qi][skip:skip + n + len(bad) + 5] if j not in bad][:n])
    return out


def validate(pred, q_ids, pool_ids):
    if not isinstance(pred, dict):
        raise ValueError("prediction must map query ids to lists")
    pool = set(pool_ids)
    if set(pred) != set(q_ids):
        raise ValueError(f"query id mismatch: missing {len(set(q_ids) - set(pred))}, extra {len(set(pred) - set(q_ids))}")
    for q, v in pred.items():
        if not isinstance(v, list) or len(v) != K or len(set(v)) != K:
            raise ValueError(f"{q}: need exactly {K} unique ids, got {v}")
        if any(not isinstance(c, str) or c not in pool for c in v):
            raise ValueError(f"{q}: id outside the candidate pool")
    return True


def write_submission(out_dir, pred, q_ids, pool_ids, name="submission.zip"):
    validate(pred, q_ids, pool_ids)
    path = os.path.join(out_dir, name.replace("submission", "prediction").replace(".zip", ".json"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({q: pred[q] for q in q_ids}, f)
    with zipfile.ZipFile(os.path.join(out_dir, name), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, "prediction.json")
    return len(pred)
