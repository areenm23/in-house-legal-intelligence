"""Decoding sweep for one trained checkpoint: score every decoding config on the dev set, then write
test submissions for the best few.

  python -m phase1.sweep --run runs/run_A_muon_cont.py --ckpt <.../step_N.pt> --data <dir> \
      --decodes decode/a.json decode/b.json ... --top-k 2 --hard-stop <unix time> \
      --preds /kaggle/working/preds --subs /kaggle/working/submissions

<data> needs prep_manifest.json, tokenizer/, dev/{prompts,truth}.json and test.json (no shards).
Dev runs go in the given order (put cheap configs first); each step is skipped if it would pass
--hard-stop. Results: <subs>/<NAME>__sweep.json and <NAME>__<decode>__step<N>.zip.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

from .train.launch import load_run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--decodes", nargs="+", required=True)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--hard-stop", type=float, required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--subs", required=True)
    args = ap.parse_args()

    run = load_run(args.run)
    step = int(re.search(r"step_(\d+)\.pt$", args.ckpt).group(1))
    os.makedirs(args.subs, exist_ok=True)
    summary = {"run": run.NAME, "ckpt": args.ckpt, "step": step, "dev": [], "test": []}
    out_json = os.path.join(args.subs, f"{run.NAME}__sweep.json")

    def save():
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def predict(dec, split):
        tag = os.path.splitext(os.path.basename(dec))[0]
        left = args.hard_stop - time.time()
        if left < 300:
            print(f"[sweep {run.NAME}] skipping {tag} {split}: {left:.0f}s left", flush=True)
            return None
        outd = os.path.join(args.preds, f"{run.NAME}__{tag}_{split}")
        t = time.time()
        try:
            subprocess.run([sys.executable, "-u", "-m", "phase1.eval.predict", "--run", args.run, "--ckpt",
                            args.ckpt, "--data", args.data, "--decode", dec, "--split", split, "--out", outd],
                           check=True, timeout=left - 120)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            print(f"[sweep {run.NAME}] {tag} {split} failed: {type(e).__name__}", flush=True)
            return {"decode": tag, "file": dec, "error": type(e).__name__}
        with open(os.path.join(outd, "predict_record.json"), encoding="utf-8") as f:
            rec = json.load(f)
        res = {"decode": tag, "file": dec, "minutes": round((time.time() - t) / 60, 1),
               "short_outputs": rec["gen_stats"].get("short_outputs")}
        if split == "dev":
            res["scores"] = rec.get("scores")
        else:
            dst = os.path.join(args.subs, f"{run.NAME}__{tag}__step{step}.zip")
            shutil.copy(os.path.join(outd, "submission.zip"), dst)
            res["zip"] = dst
        print(f"[sweep {run.NAME}] {split} {json.dumps(res)}", flush=True)
        return res

    for dec in args.decodes:
        r = predict(dec, "dev")
        if r:
            summary["dev"].append(r)
            save()
    ranked = sorted((r for r in summary["dev"] if r.get("scores") and r["scores"].get("bertscore_f1") is not None),
                    key=lambda r: -r["scores"]["bertscore_f1"])
    summary["ranking"] = [(r["decode"], r["scores"]["bertscore_f1"]) for r in ranked]
    print(f"[sweep {run.NAME}] dev ranking: {summary['ranking']}", flush=True)
    for r in ranked[:args.top_k]:
        t = predict(r["file"], "test")
        if t:
            summary["test"].append(t)
            save()
    save()
    print(f"[sweep {run.NAME}] SUMMARY {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
