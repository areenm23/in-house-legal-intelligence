"""Run a queue of prior-case-retrieval experiments on one GPU, then summarise the output folder.

    python -m phase7.session --queue smoke_G G_lorma As_kd --data D --roots R --out runs_p7 --deadline EPOCH
    python -m phase7.session --summary --out runs_p7
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback

from . import data as D
from .experiments import get_experiment
from .train import parse_set, run


def parse_item(item):
    over = {}
    if "@" in item:
        item, kv = item.split("@", 1)
        over = parse_set(kv.split(","))
    exp, _, mode = item.partition(":")
    return exp, mode or "train", over


def summarize(out_root):
    rows = []
    for f in sorted(glob.glob(os.path.join(out_root, "*", "result.json"))):
        r = json.load(open(f))
        dd, dh, db = r.get("dev_dense") or {}, r.get("dev_hybrid") or {}, r.get("dev_bm25") or {}
        tm = r.get("params_trainable")
        rows.append({"run": os.path.basename(os.path.dirname(f)), "backbone": r.get("backbone"), "method": r.get("method"),
                     "kd": (r.get("kd") or {}).get("teacher"), "f1_hybrid": dh.get("f1"), "alpha": r.get("alpha"),
                     "f1_dense": dd.get("f1"), "p_dense": dd.get("precision"), "r_dense": dd.get("recall"),
                     "r@100_dense": dd.get("recall@100"), "mrr_dense": dd.get("mrr"), "f1_bm25": db.get("f1"),
                     "epochs": r.get("epochs_seen"), "stop": r.get("stop"),
                     "trainable_M": round(tm / 1e6, 3) if tm else None,
                     "total_M": round(r["params_total"] / 1e6, 1) if r.get("params_total") else None,
                     "train_min": r.get("train_minutes"), "embed_docs_s": r.get("embed_docs_s"),
                     "peak_gib": round(r["peak_mem_gib"], 2) if r.get("peak_mem_gib") else None})
    rows.sort(key=lambda x: -(x["f1_hybrid"] or x["f1_dense"] or -1))
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    cols = list(rows[0]) if rows else []
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and v < 1.5 else ("" if v is None else str(v))
    lines = ["Dev micro-F1@7 (official definition); hybrid = alpha * z(dense) + (1 - alpha) * z(BM25), alpha chosen on dev",
             "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    with open(os.path.join(out_root, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", nargs="*", default=[])
    ap.add_argument("--data")
    ap.add_argument("--roots", nargs="+", default=["results"])
    ap.add_argument("--out", default="runs_p7")
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--margin-minutes", type=float, default=10)
    ap.add_argument("--min-minutes", type=float, default=10)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--tag", default="q")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.summary:
        rows = summarize(a.out)
        print(open(os.path.join(a.out, "summary.md")).read())
        return rows
    root = D.find_root(a.data)
    if root is None:
        sys.exit(f"no Phase_7_PCR data under {a.data}")
    log = open(os.path.join(a.out, f"session_{a.tag}.jsonl"), "a")
    for item in a.queue:
        exp, mode, over = parse_item(item)
        left = None
        if a.deadline:
            left = (a.deadline - time.time()) / 60 - a.margin_minutes
            if left < a.min_minutes:
                log.write(json.dumps({"exp": exp, "status": "skipped (time)"}) + "\n")
                print(f"skip {exp}: {left:.0f} min left", flush=True)
                continue
        t0 = time.time()
        rec = {"exp": exp, "over": over}
        try:
            r = run(get_experiment(exp, over), root, a.roots, a.out, max_minutes=left)
            rec.update(status="ok", f1_hybrid=r["dev_hybrid"]["f1"], f1_dense=r["dev_dense"]["f1"])
        except Exception as e:
            traceback.print_exc()
            rec.update(status="failed", error=f"{type(e).__name__}: {e}"[:500])
        rec["minutes"] = round((time.time() - t0) / 60, 1)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print("SESSION", json.dumps(rec), flush=True)
        summarize(a.out)


if __name__ == "__main__":
    main()
