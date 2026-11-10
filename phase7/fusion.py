"""Phase-7 score fusion on CPU: the dense scores saved by finished runs + lexical n-gram BM25 + a citation prior +
a date check, with fusion weights tuned on dev micro-F1@7 (no GPU, no retraining).

    python -m phase7.fusion --data data/phase7 --runs "Phase 7 outs" --roots kaggle/out/cs781-p2-backbones --out runs_p7/fusion

Features (each a (queries x pool) matrix, z-scored per query before weighting):
  dense     mean of the z-scored dense scores of the best `top` runs by dev hybrid F1 (top tuned in 1, 3, 5, 8)
  bm25_n    Okapi BM25 over token n-grams (n = 1, 2, 3; hashed), k1 / b tuned on dev
  cited     log(1 + times the candidate is a relevant case of a labelled query); dev uses train labels only, test
            uses train + dev labels; a candidate is matched by id or by text hash
  later_*   1 if the candidate is newer than the query: both 'DATE OF JUDGMENT' headers (later_dated), or a dated
            query and an undated candidate that mentions a later year (later_year); a prior case cannot be newer
Weights: coordinate ascent on dev F1@7; the honest number is a 2-fold cross-fit over dev queries.
Outputs: one test zip per variant (fused, dense only, best BM25 only) + fusion.json with weights and diagnostics.
"""
import argparse
import glob
import json
import os
import re
from collections import Counter

import numpy as np
import scipy.sparse as sp

from . import data as D
from .score import at_k, top_k, write_submission, zscore

BUCKETS = 1 << 22
_YEAR = re.compile(r"\b(1[89]\d\d|20[0-2]\d)\b")


def ngram_matrix(store, n, buckets=BUCKETS):
    """(docs x buckets) counts of hashed token n-grams."""
    data, ind, ptr = [], [], [0]
    for i in range(len(store)):
        t = store.doc(i).astype(np.int64)
        if len(t) < n:
            g = np.zeros(0, np.int64)
        else:
            g = t[:len(t) - n + 1].copy()
            for j in range(1, n):
                g = g * 32771 + t[j:len(t) - n + 1 + j]
            g %= buckets
        u, c = np.unique(g, return_counts=True)
        ind.append(u)
        data.append(c)
        ptr.append(ptr[-1] + len(u))
    return sp.csr_matrix((np.concatenate(data).astype(np.float32) if data else np.zeros(0, np.float32),
                          np.concatenate(ind) if ind else np.zeros(0, np.int64), np.array(ptr)),
                         shape=(len(store), buckets))


def bm25(C_pool, C_q, k1, b):
    N = C_pool.shape[0]
    df = np.bincount(C_pool.indices, minlength=C_pool.shape[1])
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
    dl = np.asarray(C_pool.sum(1)).ravel()
    C = C_pool.tocoo()
    Kd = k1 * (1 - b + b * dl / max(1e-9, dl.mean()))
    W = sp.csr_matrix(((C.data * (k1 + 1) / (C.data + Kd[C.row]) * idf[C.col]).astype(np.float32), (C.row, C.col)),
                      shape=C_pool.shape)
    Q = C_q.copy()
    Q.data[:] = 1.0
    return np.asarray((Q @ W.T).todense(), dtype=np.float32)


def citation_counts(splits):
    """Counter over candidate ids and over candidate text hashes of the relevant cases of these labelled splits."""
    by_id, by_hash = Counter(), Counter()
    for s in splits:
        h = {d: s.h_p[i] for i, d in enumerate(s.pool.ids)}
        for rel in s.rel:
            for c in rel or []:
                by_id[c] += 1
                if c in h:
                    by_hash[h[c]] += 1
    return by_id, by_hash


def cited_feature(split, counts):
    by_id, by_hash = counts
    v = np.array([max(by_id.get(d, 0), by_hash.get(split.h_p[i], 0)) for i, d in enumerate(split.pool.ids)], np.float32)
    return np.log1p(np.broadcast_to(v, (len(split.queries), len(v))).copy())


def latest_year(text):
    ys = [int(y) for y in _YEAR.findall(text or "")]
    return max(ys) if ys else 0


def raw_texts(root, split):
    part, stem = D.SPLITS[split]
    q = {r[0]: r[1] for r in D.normalize(D._json(D._find(root, part, f"{stem}.json")), with_rel=False)}
    p = {r[0]: r[1] for r in D.normalize(D._json(D._find(root, part, f"{stem}_candidates.json")), with_rel=False)}
    return q, p


_DATE = re.compile(r"DATE OF JUDGMENT\s*:?\s*(\d{1,2})/(\d{1,2})/(\d{4})")


def judgment_date(text):
    """yyyymmdd from the 'DATE OF JUDGMENT: dd/mm/yyyy' header, 0 if absent."""
    m = _DATE.search((text or "")[:4000])
    return int(m.group(3)) * 10000 + int(m.group(2)) * 100 + int(m.group(1)) if m else 0


def later_features(split, q_text, p_text):
    """later_dated: both judgments dated and the candidate is newer; later_year: the query is dated, the candidate
    is not, and the candidate mentions a year after the query's (it cannot be older than a year it mentions)."""
    qd = np.array([judgment_date(q_text.get(d)) for d in split.queries.ids])
    pd = np.array([judgment_date(p_text.get(d)) for d in split.pool.ids])
    py = np.array([latest_year(p_text.get(d)) for d in split.pool.ids])
    dated = (qd[:, None] > 0) & (pd[None, :] > 0) & (pd[None, :] > qd[:, None])
    by_year = (qd[:, None] > 0) & (pd[None, :] == 0) & (py[None, :] > qd[:, None] // 10000)
    return {"later_dated": dated.astype(np.float32), "later_year": by_year.astype(np.float32)}


def dense_runs(runs_dir, split_name, sp_):
    """-> [(dev hybrid F1, run name, dense matrix)] for every run whose saved scores match this split's order."""
    out = []
    for f in sorted(glob.glob(os.path.join(runs_dir, "**", f"scores_{split_name}.npz"), recursive=True)):
        d = os.path.dirname(f)
        rp = os.path.join(d, "result.json")
        if os.path.basename(d).startswith("smoke_") or not os.path.exists(rp):
            continue
        z = np.load(f)
        if [str(x) for x in z["q_ids"]] != sp_.queries.ids or [str(x) for x in z["p_ids"]] != sp_.pool.ids:
            continue
        r = json.load(open(rp))
        out.append(((r.get("dev_hybrid") or {}).get("f1") or 0.0, os.path.basename(d), z["dense"].astype(np.float32)))
    return out


def mean_dense(runs, names):
    sel = [m for _f, n, m in runs if n in names]
    return np.mean([zscore(m) for m in sel], 0) if sel else None


def f1(scores, split, rows=None):
    q = split.queries.ids if rows is None else [split.queries.ids[i] for i in rows]
    s = scores if rows is None else scores[rows]
    gold = {qq: [c for c in (split.rel[split.queries.index[qq]] or [])] for qq in q}
    return at_k(top_k(s, q, split.pool.ids), gold)["f1"]


GRID = {"later_dated": (0.0, -1.0, -3.0, -100.0), "later_year": (0.0, -1.0, -3.0, -100.0)}
DEFAULT_GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)


def combine(feats, w):
    return sum(w[k] * feats[k] for k in w if w[k])


def tune(feats, split, rows=None, rounds=3):
    w = {k: (1.0 if k == "dense" else 0.0) for k in feats}
    if "dense" not in feats:
        w[next(iter(feats))] = 1.0
    best = f1(combine(feats, w), split, rows)
    for _ in range(rounds):
        changed = False
        for k in feats:
            for v in GRID.get(k, DEFAULT_GRID):
                cand = {**w, k: v}
                if not any(cand.values()):
                    continue
                s = f1(combine(feats, cand), split, rows)
                if s > best + 1e-12:
                    best, w, changed = s, cand, True
        if not changed:
            break
    return w, best


def cross_fit(feats, split, seed=0):
    idx = np.random.default_rng(seed).permutation(len(split.queries))
    a, b = np.sort(idx[: len(idx) // 2]), np.sort(idx[len(idx) // 2:])
    out = []
    for fit, ev in ((a, b), (b, a)):
        w, _ = tune(feats, split, fit)
        out.append(f1(combine(feats, w), split, ev))
    return float(np.mean(out))


def run(data_root, runs_dir, roots, out, tok_path=None, ngrams=(1, 2, 3), k1s=(1.2, 1.6, 2.0, 3.0), bs=(0.75, 0.9, 1.0)):
    from phase3.train import find_tokenizer
    os.makedirs(out, exist_ok=True)
    tok_path = tok_path or find_tokenizer(roots)
    data = D.load_data(data_root, tok_path, os.path.join(out, "_cache"))
    tr, dv, te = data["train"], data["dev"], data["test"]
    rep = {"runs_dir": runs_dir}

    rd, rt = dense_runs(runs_dir, "dev", dv), dense_runs(runs_dir, "test", te)
    names_t = {n for _f, n, _m in rt}
    ranked = [n for _f, n, _m in sorted(rd, key=lambda x: -x[0]) if n in names_t]
    rep["dense_runs"] = ranked
    dense = {}
    if ranked:
        best_top, best_f = 1, -1.0
        for t in (1, 3, 5, 8):
            m = mean_dense(rd, set(ranked[:t]))
            s = f1(m, dv)
            rep[f"dev_dense_top{t}"] = round(s, 4)
            if s > best_f:
                best_top, best_f = t, s
        rep["dense_top"] = best_top
        dense = {"dev": mean_dense(rd, set(ranked[:best_top])), "test": mean_dense(rt, set(ranked[:best_top]))}

    lex = {}
    for n in ngrams:
        Cd, Cdq = ngram_matrix(dv.pool, n), ngram_matrix(dv.queries, n)
        best = (-1.0, None)
        for k1 in k1s:
            for b in bs:
                s = f1(bm25(Cd, Cdq, k1, b), dv)
                if s > best[0]:
                    best = (s, (k1, b))
        k1, b = best[1]
        rep[f"dev_bm25_{n}gram"] = {"f1": round(best[0], 4), "k1": k1, "b": b}
        lex[f"bm25_{n}"] = {"dev": bm25(Cd, Cdq, k1, b),
                            "test": bm25(ngram_matrix(te.pool, n), ngram_matrix(te.queries, n), k1, b)}

    cited = {"dev": cited_feature(dv, citation_counts([tr])), "test": cited_feature(te, citation_counts([tr, dv]))}
    rep["cited_coverage"] = {s: round(float((cited[s][0] > 0).mean()), 4) for s in cited}
    gold_rows = [(qi, dv.pool.index[c]) for qi, rel in enumerate(dv.rel) for c in (rel or []) if c in dv.pool.index]
    gc = np.array([cited["dev"][qi, j] for qi, j in gold_rows])
    rep["cited_dev_gold_vs_all"] = {"gold_mean": round(float(gc.mean()), 4), "all_mean": round(float(cited["dev"].mean()), 4),
                                    "dev_f1_alone": round(f1(cited["dev"], dv), 4)}

    later = {}
    for name, s in (("dev", dv), ("test", te)):
        qt, pt = raw_texts(data_root, name)
        later[name] = later_features(s, qt, pt)
    for k in later["dev"]:
        gl = np.array([later["dev"][k][qi, j] for qi, j in gold_rows])
        rep[f"{k}_dev"] = {"gold_flagged": int(gl.sum()), "gold_pairs": len(gl),
                           "pairs_flagged_share": round(float(later["dev"][k].mean()), 4),
                           "test_pairs_flagged_share": round(float(later["test"][k].mean()), 4)}

    feats = {s: {} for s in ("dev", "test")}
    for s in feats:
        if dense:
            feats[s]["dense"] = dense[s]
        for k, v in lex.items():
            feats[s][k] = zscore(v[s])
        feats[s]["cited"] = zscore(cited[s]) if cited[s].std() > 0 else cited[s] * 0
        feats[s].update(later[s])
    w, dev_f1 = tune(feats["dev"], dv)
    rep["weights"] = w
    rep["dev_f1_fused_tuned"] = round(dev_f1, 4)
    rep["dev_f1_fused_crossfit"] = round(cross_fit(feats["dev"], dv), 4)

    variants = {"fused": w}
    if dense:
        variants["dense"] = {"dense": 1.0}
    best_lex = max(lex, key=lambda k: rep[f"dev_{k}gram"]["f1"])
    variants["lexical"] = {best_lex: 1.0}
    rep["variants"] = {}
    for name, wv in variants.items():
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        write_submission(d, top_k(combine(feats["test"], wv), te.queries.ids, te.pool.ids), te.queries.ids, te.pool.ids)
        rep["variants"][name] = {"weights": wv, "dev_f1": round(f1(combine(feats["dev"], wv), dv), 4)}
        json.dump({"exp": f"fusion_{name}", "backbone": "fusion", "method": "fusion",
                   "dev_hybrid": {"f1": rep["variants"][name]["dev_f1"]}, "dev_dense": {}, "weights": wv},
                  open(os.path.join(d, "result.json"), "w"), indent=1)
    json.dump(rep, open(os.path.join(out, "fusion.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True, help="folder holding finished runs (scores_dev.npz / scores_test.npz)")
    ap.add_argument("--roots", nargs="+", default=["kaggle/out/cs781-p2-backbones"])
    ap.add_argument("--out", default="runs_p7/fusion")
    a = ap.parse_args(argv)
    root = D.find_root(a.data)
    if root is None:
        raise SystemExit(f"no Phase_7_PCR data under {a.data}")
    return run(root, a.runs, a.roots, a.out)


if __name__ == "__main__":
    main()
