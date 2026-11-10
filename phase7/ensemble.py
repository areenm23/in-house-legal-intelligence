"""Fuse finished runs: mean of their z-scored dense scores (+ BM25 with a weight chosen on dev), top-7 -> zip.

    python -m phase7.ensemble --runs G_lorma A_lorma As_kd --data D --roots R --out runs_p7 [--name ens1]
Uses each run's scores_{dev,test}.npz (dense + BM25 matrices, same query / pool order) and the dev labels.
"""
import argparse
import json
import os

import numpy as np

from phase3.train import find_tokenizer
from . import data as D
from .score import at_k, top_k, write_submission, zscore
from .train import best_alpha, evaluate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--roots", nargs="+", default=["results"])
    ap.add_argument("--out", default="runs_p7")
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)
    name = a.name or "ens__" + "+".join(a.runs)
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    data = D.load_data(D.find_root(a.data), find_tokenizer(a.roots), os.path.join(a.out, "_cache"))
    dv, te = data["dev"], data["test"]
    z = {p: [np.load(os.path.join(a.out, r, f"scores_{p}.npz")) for r in a.runs] for p in ("dev", "test")}
    for p, sp in (("dev", dv), ("test", te)):
        for m in z[p]:
            if [str(x) for x in m["q_ids"]] != sp.queries.ids or [str(x) for x in m["p_ids"]] != sp.pool.ids:
                raise ValueError(f"{p}: member scores do not match the split order")
    dense = {p: np.mean([zscore(m["dense"].astype(np.float32)) for m in z[p]], 0) for p in z}
    bm = {p: z[p][0]["bm25"].astype(np.float32) for p in z}
    alpha, _ = best_alpha(dense["dev"], bm["dev"], dv)
    res = {"exp": name, "members": a.runs, "method": "ensemble", "backbone": "+".join(
        sorted({json.load(open(os.path.join(a.out, r, "result.json")))["backbone"] for r in a.runs})), "alpha": alpha}
    res["dev_dense"], _ = evaluate(dense["dev"], dv)
    res["dev_hybrid"], _ = evaluate(alpha * zscore(dense["dev"]) + (1 - alpha) * zscore(bm["dev"]), dv)
    res["dev_bm25"] = at_k(top_k(bm["dev"], dv.queries.ids, dv.pool.ids),
                           {q: dv.rel[i] for i, q in enumerate(dv.queries.ids)})
    hyb = alpha * zscore(dense["test"]) + (1 - alpha) * zscore(bm["test"])
    write_submission(out, top_k(hyb, te.queries.ids, te.pool.ids), te.queries.ids, te.pool.ids)
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(f"{name}: dev F1@7 hybrid {res['dev_hybrid']['f1']:.4f} (alpha {alpha}), dense {res['dev_dense']['f1']:.4f}")
    return res


if __name__ == "__main__":
    main()
