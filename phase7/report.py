"""Phase-7 data report: raw format, query / pool counts, relevant candidates per query (and how many are missing
from the pool), document lengths, queries that sit in their own pool, overlap between splits, BM25 on dev for a
small (k1, b) grid, and parity of our micro-F1@7 with the official evaluate_phase7.py.

    python -m phase7.report --data DATA_ROOT --roots results --out runs_p7/data_report.json
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

from phase3.train import find_tokenizer
from . import data as D
from .score import BM25, at_k, diagnostics, top_k

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "course_files", "eval", "Evaluation Script", "evaluate_phase7.py")


def _pct(x):
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return None
    return {"mean": round(float(x.mean()), 2), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "max": float(x.max())}


def official(sp, pred, root):
    part, stem = D.SPLITS[sp.name]
    with tempfile.TemporaryDirectory() as tmp:
        json.dump(sp.pool.ids, open(os.path.join(tmp, "cands.json"), "w"))
        json.dump(pred, open(os.path.join(tmp, "pred.json"), "w"))
        r = subprocess.run([sys.executable, EVAL, "--truth", D._find(root, part, f"{stem}.json"), "--prediction",
                            os.path.join(tmp, "pred.json"), "--candidate-ids", os.path.join(tmp, "cands.json")],
                           capture_output=True, text=True)
        vals = {}
        for ln in r.stdout.splitlines():
            if ln.startswith("Micro-") and ":" in ln:
                k, v = ln.split(":", 1)
                vals[k.strip()] = float(v)
        return vals, (r.stdout + r.stderr)[-400:]


def build(root, roots, cache):
    print(D.describe(root), flush=True)
    t0 = time.time()
    data = D.load_data(root, find_tokenizer(roots), cache)
    rep = {"root": root, "tokenize_minutes": round((time.time() - t0) / 60, 2), "splits": {}}
    for s, sp in data.items():
        if sp is None:
            continue
        rec = {"queries": len(sp.queries), "pool": len(sp.pool), "query_tokens": _pct(sp.queries.n_tok()),
               "pool_tokens": _pct(sp.pool.n_tok()),
               "queries_in_own_pool": sum(q in sp.pool.index for q in sp.queries.ids),
               "duplicate_pool_texts": len(sp.h_p) - len(set(sp.h_p)), "query_id_examples": sp.queries.ids[:3]}
        if sp.labelled:
            n_rel = [len(r) for r in sp.rel]
            rec["relevant_per_query"] = _pct(n_rel)
            rec["queries_without_relevant"] = sum(n == 0 for n in n_rel)
            rec["relevant_missing_from_pool"] = sum(len(r) - len(sp.rel_rows(q)) for q, r in enumerate(sp.rel))
            rec["ceiling_f1_at7"] = round(at_k({q: (list(sp.rel[i]) + ["_"] * 7)[:7] for i, q in enumerate(sp.queries.ids)},
                                               {q: sp.rel[i] for i, q in enumerate(sp.queries.ids)})["f1"], 4)
        rep["splits"][s] = rec
    tr, dv, te = data["train"], data["dev"], data["test"]
    rep["overlap"] = {"test_queries_in_train_pool": len(set(te.queries.ids) & set(tr.pool.ids)),
                      "test_pool_in_train_pool": len(set(te.pool.ids) & set(tr.pool.ids)),
                      "test_queries_in_dev": len(set(te.queries.ids) & (set(dv.queries.ids) if dv else set())),
                      "test_query_texts_in_dev_queries": len(set(te.h_q) & (set(dv.h_q) if dv else set()))}
    if dv is not None:
        gold = {q: dv.rel[i] for i, q in enumerate(dv.queries.ids)}
        grid = {}
        best = None
        for k1 in (0.9, 1.2, 1.6):
            for b in (0.5, 0.75, 0.9):
                s = BM25(dv.pool, k1, b).scores(dv.queries)
                f = at_k(top_k(s, dv.queries.ids, dv.pool.ids), gold)["f1"]
                grid[f"k1={k1},b={b}"] = round(f, 4)
                if best is None or f > best[0]:
                    best = (f, k1, b, s)
        rep["bm25_dev_grid"] = grid
        rep["bm25_dev_best"] = {"f1": round(best[0], 4), "k1": best[1], "b": best[2], **diagnostics(best[3],
                                [dv.rel_rows(q) for q in range(len(dv.queries))])}
        pred = top_k(best[3], dv.queries.ids, dv.pool.ids)
        if os.path.exists(EVAL):
            vals, tail = official(dv, pred, root)
            ours = at_k(pred, gold)
            rep["official_parity"] = {"official": vals, "ours_f1": round(ours["f1"], 10),
                                      "match": abs(vals.get("Micro-F1@7", -1) - ours["f1"]) < 1e-8, "tail": tail[-200:]}
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--roots", nargs="+", default=["results"])
    ap.add_argument("--out", default="runs_p7/data_report.json")
    a = ap.parse_args(argv)
    root = D.find_root(a.data)
    if root is None:
        sys.exit(f"no Phase_7_PCR data under {a.data}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rep = build(root, a.roots, os.path.join(os.path.dirname(a.out) or ".", "_cache"))
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep, indent=1)[:8000])
    par = rep.get("official_parity")
    if par and not par["match"]:
        print("WARNING: our F1@7 and the official evaluator disagree:", par)
    return rep


if __name__ == "__main__":
    main()
