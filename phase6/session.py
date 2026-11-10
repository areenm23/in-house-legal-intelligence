"""Run a queue of summarisation experiments on one GPU, then summarise the output folder.

    python -m phase6.session --queue smoke_B B_lora "B_lora:decode@tag=r10,len_ratio=0.10" As_kd --data D --roots R --out runs_p6 --deadline EPOCH
    python -m phase6.session --summary --out runs_p6

Queue items are EXP[:MODE][@key=value,...]; MODE = train (default) or decode (re-generate the finished PEFT
run EXP with the given decoding / length overrides into EXP__dec_<tag>). Failed items are logged and the
queue moves on.
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
from .train import parse_set, redecode, run


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
        d = r.get("dev") or {}
        tm = r.get("params_trainable")
        rows.append({"run": os.path.basename(os.path.dirname(f)), "backbone": r.get("backbone"), "method": r.get("method"),
                     "ctx": r.get("ctx"), "budget": r.get("budget"), "kd": (r.get("kd") or {}).get("teacher"),
                     "bertscore": d.get("bertscore_f1"), "rouge1": d.get("rouge1"), "rougeL": d.get("rougeL"),
                     "words": round(d["words_mean"], 1) if d.get("words_mean") else None,
                     "len_ratio_vs_ref": d.get("length_ratio"), "rep4": d.get("repeated_4gram_share"),
                     "novel2": d.get("novel_bigrams"), "dev_loss": r.get("best_dev_loss"), "epochs": r.get("epochs_seen"),
                     "trainable_M": round(tm / 1e6, 3) if tm else None,
                     "total_M": round(r["params_total"] / 1e6, 1) if r.get("params_total") else None,
                     "train_min": r.get("train_minutes"), "tok_s": r.get("train_tok_s"),
                     "gen_tok_s": r.get("gen_tokens_per_s"),
                     "peak_gib": round(r["peak_mem_gib"], 2) if r.get("peak_mem_gib") else None})
    rows.sort(key=lambda x: -(x["bertscore"] or x["rouge1"] or -1))
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    cols = list(rows[0]) if rows else []
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and v < 1.5 else ("" if v is None else str(v))
    lines = ["Dev = held-out part of train.json (gen_dev_docs documents generated); official metric definitions", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    with open(os.path.join(out_root, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", nargs="*", default=[])
    ap.add_argument("--data")
    ap.add_argument("--roots", nargs="+", default=["results"])
    ap.add_argument("--out", default="runs_p6")
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--margin-minutes", type=float, default=10)
    ap.add_argument("--min-minutes", type=float, default=12)
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
        sys.exit(f"no Phase_6_Summ data under {a.data}")
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
            if mode == "decode":
                tag = over.pop("tag", "dec")
                r = redecode(os.path.join(a.out, exp), over, f"{exp}__dec_{tag}", root, a.roots, a.out)
            else:
                r = run(get_experiment(exp, over), root, a.roots, a.out, max_minutes=left)
            d = r.get("dev") or {}
            rec.update(status="ok", bertscore=d.get("bertscore_f1"), rouge1=d.get("rouge1"))
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
