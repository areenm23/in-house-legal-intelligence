"""Collect finished runs from the downloaded Kaggle outputs into a numbered Codabench upload folder.

    python -m phase6.upload --results results/p6 --out kaggle/p6_upload [--max 40]

Order: the required six first (each slot takes the best dev score among the runs of its backbone + technique, so a
decode sweep or a *_long run can fill it), then every other valid run by dev score. Every zip is checked by the
phase's own rule and must cover the same test ids as the other runs. phase7.upload reuses this with its own slots.
"""
import argparse
import glob
import json
import os
import shutil
import zipfile
from collections import Counter

from .score import validate

SIX = [("LoRMA", "A", "lorma", False), ("LoRA", "B", "lora", False), ("QLoRA", "C", "qlora", False),
       ("AdaLoRA", "D", "adalora", False), ("Houlsby", "B", "houlsby", False), ("KD", "As", None, True)]


def p6_metrics(r, run_dir):
    dev = r.get("dev") or {}
    if dev.get("bertscore_f1") is None:
        return None
    dp = os.path.join(run_dir, "dev_pred.json")
    return {"bertscore": dev["bertscore_f1"], "rouge1": dev.get("rouge1"), "rougeL": dev.get("rougeL"),
            "words": dev.get("words_mean"),
            "dev_docs": len(json.load(open(dp, encoding="utf-8"))) if os.path.exists(dp) else None}


def p6_ids(pj):
    ids = [r.get("id") for r in pj] if isinstance(pj, list) else []
    validate(pj, ids)
    return frozenset(ids)


def collect(results, metrics=p6_metrics, score="bertscore", zip_name="submission.zip"):
    """-> one record per run name (the best-scoring copy if a run exists in several notebook outputs)."""
    runs = {}
    for f in sorted(glob.glob(os.path.join(results, "**", "result.json"), recursive=True)):
        d = os.path.dirname(f)
        if os.path.basename(d).startswith("smoke_"):
            continue
        z = os.path.join(d, zip_name)
        r = json.load(open(f, encoding="utf-8"))
        m = metrics(r, d) if os.path.exists(z) else None
        if m is None or m.get(score) is None:
            continue
        rec = {"run": os.path.basename(d), "zip": z, "notebook": os.path.relpath(d, results).split(os.sep)[0],
               "backbone": r.get("backbone"), "method": r.get("method"), "kd": (r.get("kd") or {}).get("teacher"),
               "score": m[score], "metrics": m}
        if rec["run"] not in runs or rec["score"] > runs[rec["run"]]["score"]:
            runs[rec["run"]] = rec
    return list(runs.values())


def zip_ids(path, check=p6_ids):
    with zipfile.ZipFile(path) as z:
        if "prediction.json" not in z.namelist():
            raise ValueError("no prediction.json at the zip root")
        return check(json.loads(z.read("prediction.json")))


def plan(recs, max_n=40, six=SIX, check=p6_ids):
    """-> (ordered upload list, rejected [(run, reason)], unfilled slots)."""
    rejected, good = [], []
    for r in recs:
        try:
            r["ids"] = zip_ids(r["zip"], check)
            good.append(r)
        except (ValueError, zipfile.BadZipFile, json.JSONDecodeError) as e:
            rejected.append((r["run"], str(e)[:200]))
    if good:
        common = Counter(r["ids"] for r in good).most_common(1)[0][0]
        rejected += [(r["run"], f"test ids differ ({len(r['ids'])} ids)") for r in good if r["ids"] != common]
        good = [r for r in good if r["ids"] == common]
    order, missing = [], []
    for slot, bb, method, kd in six:
        cand = [r for r in good if r["run"] not in {o["run"] for o in order} and r["backbone"] == bb
                and (bool(r["kd"]) if kd else (r["method"] == method and not r["kd"]))]
        if cand:
            order.append({**max(cand, key=lambda r: r["score"]), "slot": slot})
        else:
            missing.append(slot)
    used = {r["run"] for r in order}
    order += [{**r, "slot": ""} for r in sorted(good, key=lambda r: -r["score"]) if r["run"] not in used]
    return order[:max_n], rejected, missing


def write(order, out):
    os.makedirs(out, exist_ok=True)
    for old in glob.glob(os.path.join(out, "[0-9][0-9]_*.zip")):
        os.remove(old)
    mcols = list(order[0]["metrics"]) if order else []
    cols = ["#", "file", "slot", "backbone", "method", "kd", *mcols, "notebook"]
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and v < 1.5 else ("" if v is None else
                                                                          f"{v:.0f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in enumerate(order, 1):
        name = f"{i:02d}_{r['run']}.zip"
        shutil.copy(r["zip"], os.path.join(out, name))
        vals = [i, name, r["slot"], r["backbone"], r["method"], r["kd"], *(r["metrics"].get(c) for c in mcols), r["notebook"]]
        lines.append("| " + " | ".join(fmt(v) for v in vals) + " |")
    open(os.path.join(out, "manifest.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump([{k: v for k, v in r.items() if k != "ids"} for r in order], open(os.path.join(out, "manifest.json"), "w"),
              indent=1)
    return lines


def run_cli(argv, results, out, collect_kw=None, six=SIX, check=p6_ids):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=results)
    ap.add_argument("--out", default=out)
    ap.add_argument("--max", type=int, default=40)
    a = ap.parse_args(argv)
    order, rejected, missing = plan(collect(a.results, **(collect_kw or {})), a.max, six, check)
    print("\n".join(write(order, a.out)))
    for run, why in rejected:
        print(f"REJECTED {run}: {why}")
    if missing:
        print("MISSING required slots:", ", ".join(missing))
    return order, rejected, missing


def main(argv=None):
    return run_cli(argv, "results/p6", "kaggle/p6_upload")


if __name__ == "__main__":
    main()
