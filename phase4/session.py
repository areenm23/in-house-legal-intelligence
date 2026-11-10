"""Run a queue of court-judgment-prediction experiments on one GPU, then summarise the output folder.

    python -m phase4.session --queue smoke_G G_lorma As_kd G_lorma:hier --data D --roots R --out runs_p4 --deadline EPOCH
    python -m phase4.session --summary --out runs_p4

Queue items are EXP[:MODE][@key=value,key=value]. MODE is "train" (default) or "hier" (the two-stage
aggregators of phase4.hier on the chunk vectors a finished run EXP extracted). A failed item is logged
and the queue moves on. Each experiment gets the time left before the deadline (minus a margin) as its
limit; when less than --min-minutes remain the rest is skipped.
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
        dev = r.get("dev") or {}
        thr = r.get("dev_thr") or {}
        tm = r.get("params_trainable")
        rows.append({
            "run": os.path.basename(os.path.dirname(f)), "backbone": r.get("backbone"), "method": r.get("method"),
            "ctx": r.get("ctx"), "L": r.get("seq_len"), "chunks": r.get("n_chunks"), "agg": r.get("agg"),
            "kd": (r.get("kd") or {}).get("teacher"),
            "dev_f1": dev.get("macro_f1"), "dev_p": dev.get("precision"), "dev_r": dev.get("recall"),
            "dev_acc": dev.get("accuracy"), "thr": thr.get("threshold"), "thr_f1_opt": thr.get("macro_f1"),
            "eval_f1": r.get("best_eval_f1"), "epochs": r.get("epochs_seen"), "stop": r.get("stop"),
            "train_docs": r.get("train_docs"),
            "trainable_M": round(tm / 1e6, 3) if tm is not None else None,
            "total_M": round(r["params_total"] / 1e6, 1) if r.get("params_total") else None,
            "train_min": r.get("train_minutes"), "tok_s": r.get("train_tok_s"), "docs_s": r.get("infer_docs_s"),
            "peak_gib": round(r["peak_mem_gib"], 2) if r.get("peak_mem_gib") else None,
            "test_pos": r.get("test_pos_rate"),
            "by_length": [(b["tokens"], b["macro_f1"]) for b in (r.get("errors") or {}).get("by_length", [])],
        })
    rows.sort(key=lambda x: -(x["dev_f1"] or -1))
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    cols = ["run", "backbone", "method", "ctx", "L", "chunks", "agg", "kd", "dev_f1", "dev_p", "dev_r", "dev_acc",
            "thr", "thr_f1_opt", "eval_f1", "epochs", "stop", "train_docs", "trainable_M", "total_M", "train_min",
            "tok_s", "docs_s", "peak_gib", "test_pos"]
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and v < 1.5 else ("" if v is None else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    lines += ["", "Dev macro-F1 by document-length quartile (tokens)", ""]
    lines += [f"- {r['run']}: " + ", ".join(f"{a}-{b}: {f:.3f}" for (a, b), f in r["by_length"]) for r in rows
              if r["by_length"]]
    with open(os.path.join(out_root, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", nargs="*", default=[])
    ap.add_argument("--data")
    ap.add_argument("--roots", nargs="+", default=["results"])
    ap.add_argument("--out", default="runs_p4")
    ap.add_argument("--deadline", type=float, default=None, help="unix time by which the queue must finish")
    ap.add_argument("--margin-minutes", type=float, default=10)
    ap.add_argument("--min-minutes", type=float, default=8)
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
        sys.exit(f"no Phase_4_CJPE data under {a.data}")
    log = open(os.path.join(a.out, f"session_{a.tag}.jsonl"), "a")
    for item in a.queue:
        exp, mode, over = parse_item(item)
        left = None
        if a.deadline:
            left = (a.deadline - time.time()) / 60 - a.margin_minutes
            if left < a.min_minutes:
                log.write(json.dumps({"exp": exp, "mode": mode, "status": "skipped (time)"}) + "\n")
                print(f"skip {exp}:{mode}: {left:.0f} min left", flush=True)
                continue
        t0 = time.time()
        rec = {"exp": exp, "mode": mode, "over": over}
        try:
            if mode == "hier":
                from .hier import run_all
                res = run_all(os.path.join(a.out, exp), a.out, **over)
                rec.update(status="ok", best=max((r["dev"]["macro_f1"] for r in res.values()), default=None))
            else:
                r = run(get_experiment(exp, over), root, a.roots, a.out, max_minutes=left)
                rec.update(status="ok", dev_f1=(r.get("dev") or {}).get("macro_f1"))
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
