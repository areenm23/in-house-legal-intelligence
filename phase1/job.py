"""One GPU's work in a Kaggle session: train a run file until a deadline, then write test
submissions and dev scores for each decoding config.

  python -m phase1.job --run runs/run_A.py --data /tmp/cs781/p1data --out /kaggle/working/runs \
      --train-until <unix time> --hard-stop <unix time> \
      --decodes decode/greedy_nr4.json decode/nucleus_t07.json \
      --preds /kaggle/working/preds --subs /kaggle/working/submissions

Order after training, per decoding config: test (the submission) first, then dev (official scores).
Every step is skipped if it would run past --hard-stop, so the notebook ends before Kaggle's 12 h
limit (a session that hits the limit loses its whole output).
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

from .train.launch import load_run


def newest_ckpt(run_dir):
    cks = glob.glob(os.path.join(run_dir, "ckpt", "step_*.pt"))
    return max(cks, key=lambda p: int(re.search(r"step_(\d+)\.pt$", p).group(1))) if cks else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", default="/kaggle/input")
    ap.add_argument("--train-until", type=float, required=True)
    ap.add_argument("--hard-stop", type=float, required=True)
    ap.add_argument("--decodes", nargs="*", default=[])
    ap.add_argument("--preds", required=True)
    ap.add_argument("--subs", required=True)
    ap.add_argument("--skip-train", action="store_true")
    args, extra = ap.parse_known_args()

    run = load_run(args.run)
    run_dir = os.path.join(args.out, run.NAME)
    summary = {"run": run.NAME, "run_file": args.run}
    if not args.skip_train:
        t = time.time()
        r = subprocess.run([sys.executable, "-u", "-m", "phase1.train.launch", "--run", args.run, "--data", args.data,
                            "--out", args.out, "--inputs", args.inputs, "--deadline", str(args.train_until),
                            "--max-hours", "12", *extra])
        summary["train_exit"], summary["train_hours"] = r.returncode, round((time.time() - t) / 3600, 2)
        print(f"[job {run.NAME}] training exited with {r.returncode} after {summary['train_hours']} h", flush=True)

    ckpt = newest_ckpt(run_dir)
    if ckpt is None:
        raise SystemExit(f"[job {run.NAME}] no checkpoint in {run_dir}")
    step = int(re.search(r"step_(\d+)\.pt$", ckpt).group(1))
    summary["ckpt"], summary["results"] = ckpt, []
    os.makedirs(args.subs, exist_ok=True)
    decodes = list(getattr(run, "DECODES", None) or args.decodes)

    def predict_all(label, ck, ck_step, results):
        for dec in decodes:
            tag = os.path.splitext(os.path.basename(dec))[0]
            for split in ("test", "dev"):
                left = args.hard_stop - time.time()
                if left < 300:
                    print(f"[job {label}] skipping {tag} {split}: {left:.0f}s left", flush=True)
                    continue
                outd = os.path.join(args.preds, f"{label}__{tag}_{split}")
                cmd = [sys.executable, "-u", "-m", "phase1.eval.predict", "--run", args.run, "--ckpt", ck,
                       "--data", args.data, "--decode", dec, "--split", split, "--out", outd]
                try:
                    subprocess.run(cmd, check=True, timeout=left - 120)
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                    print(f"[job {label}] {tag} {split} failed: {type(e).__name__}", flush=True)
                    results.append({"decode": tag, "split": split, "error": type(e).__name__})
                    continue
                res = {"decode": tag, "split": split, "step": ck_step}
                if split == "test":
                    dst = os.path.join(args.subs, f"{label}__{tag}__step{ck_step}.zip")
                    shutil.copy(os.path.join(outd, "submission.zip"), dst)
                    res["zip"] = dst
                else:
                    with open(os.path.join(outd, "predict_record.json"), encoding="utf-8") as f:
                        res["scores"] = json.load(f).get("scores")
                results.append(res)
                print(f"[job {label}] {json.dumps(res)}", flush=True)

    predict_all(run.NAME, ckpt, step, summary["results"])
    init_info = os.path.join(run_dir, "init_from.json")
    if os.path.exists(init_info):
        with open(init_info, encoding="utf-8") as f:
            init = json.load(f)
        if os.path.exists(init["init_from"]):
            summary["init_results"] = []
            predict_all(f"{run.INIT_FROM}_regen", init["init_from"], init["step"], summary["init_results"])
    with open(os.path.join(args.subs, f"{run.NAME}__summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[job {run.NAME}] SUMMARY {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
