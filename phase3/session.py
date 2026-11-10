"""Run a queue of rhetorical-role experiments on one GPU, then summarise everything in the output folder.

    python -m phase3.session --queue G_lora A_lorma:dev As_kd:full --data D --roots R --out runs_p3 --deadline EPOCH
    python -m phase3.session --summary --out runs_p3

Queue items are EXP[:MODE][@key=value,key=value] (MODE dev by default). A failed experiment is logged
and the queue moves on. Each experiment gets the time left before the deadline (minus a margin) as
its limit; when less than `--min-minutes` remain the rest is skipped.
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
    return exp, mode or "dev", over


def summarize(out_root):
    rows = []
    for f in sorted(glob.glob(os.path.join(out_root, "*", "result.json"))):
        r = json.load(open(f))
        dev = r.get("dev") or {}
        doms = dev.get("domains", {})
        rows.append({
            "run": os.path.basename(os.path.dirname(f)), "backbone": r.get("backbone"), "method": r.get("method"),
            "mode": r.get("mode"), "ctx": r.get("ctx"), "head": r.get("head") + ("+crf" if r.get("crf") else ""),
            "dev_f1": dev.get("macro_f1"), "dev_f1_CL": doms.get("CL", {}).get("macro_f1"),
            "dev_f1_IT": doms.get("IT", {}).get("macro_f1"), "dev_p": dev.get("macro_p"), "dev_r": dev.get("macro_r"),
            "dev_acc": dev.get("accuracy"), "bias_f1_opt": r.get("dev_bias_f1_optimistic"),
            "best_epoch": r.get("best_epoch"),
            "trainable_M": round(r["params_trainable"] / 1e6, 3), "total_M": round(r["params_total"] / 1e6, 1),
            "train_min": r.get("train_minutes"), "sent_s": r.get("train_sent_s"), "infer_sent_s": r.get("infer_sent_s"),
            "peak_gib": round(r["peak_mem_gib"], 2) if r.get("peak_mem_gib") else None,
            "per_label_f1": {k: v["f1"] for k, v in dev.get("per_label", {}).items()},
            "top_confusions": dev.get("top_confusions", [])[:5],
        })
    rows.sort(key=lambda x: -(x["dev_f1"] or -1))
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    cols = ["run", "backbone", "method", "mode", "ctx", "head", "dev_f1", "dev_f1_CL", "dev_f1_IT", "dev_p", "dev_r",
            "dev_acc", "bias_f1_opt", "best_epoch", "trainable_M", "total_M", "train_min", "sent_s", "peak_gib"]
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and v < 1.5 else ("" if v is None else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    labels = list(next((r["per_label_f1"] for r in rows if r["per_label_f1"]), {}))
    if labels:
        lines += ["", "Per-label dev F1", "", "| run | " + " | ".join(labels) + " |", "|" + "---|" * (len(labels) + 1)]
        lines += ["| " + r["run"] + " | " + " | ".join(f"{r['per_label_f1'].get(l, 0):.2f}" for l in labels) + " |"
                  for r in rows if r["per_label_f1"]]
    with open(os.path.join(out_root, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", nargs="*", default=[])
    ap.add_argument("--data")
    ap.add_argument("--roots", nargs="+", default=["results"])
    ap.add_argument("--out", default="runs_p3")
    ap.add_argument("--deadline", type=float, default=None, help="unix time by which the queue must finish")
    ap.add_argument("--margin-minutes", type=float, default=10)
    ap.add_argument("--min-minutes", type=float, default=6)
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
        sys.exit(f"no Phase_3_RR data under {a.data}")
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
            r = run(get_experiment(exp, over), root, a.roots, a.out, mode, max_minutes=left)
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
