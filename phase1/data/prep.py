"""Whole Phase-1 data preparation in one command (made for a Kaggle CPU session).

  python -m phase1.data.prep --config phase1/configs/data.json --section full \
      --raw-dir /kaggle/tmp/raw --out /kaggle/working/p1data

Steps (each one is an existing module, run as a subprocess so its own checks apply):
  1. disk check, then download the needed train files + test prompts from Hugging Face
     (needs HF_TOKEN in the environment: the dataset is gated)
  2. inspect_data --profile --full                       -> <out>/stats.json
  3. train_tokenizer --impl a1                           -> <out>/tokenizer/a1bpe_<V>.json (+ .report.json)
  4. make_shards                                         -> <out>/shards/{train,val,dev_docs}, meta.json
  5. make_dev_set                                        -> <out>/dev/{prompts,truth,sources}.json
  6. copy the official test prompts                      -> <out>/test.json
  and <out>/prep_manifest.json with the config used, step timings and a code hash.

Every value comes from the config section; nothing is defaulted here.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

from .common import HF_DATASET_REPO, HF_TRAIN_DIR, SOURCES, source_file

TEST_PATH = "Phase_1_Pretraining/Test/test.json"
FILE_GB = {"acts": 0.73, "high_courts": 10.67, "supreme_court": 1.32, "tribunals": 6.48}
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_section(path, name):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)[name]

    def strip(d):
        return {k: strip(v) if isinstance(v, dict) else v for k, v in d.items() if not k.startswith("_")}

    cfg = strip(cfg)
    missing = []

    def walk(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{prefix}{k}.")
            elif v is None:
                missing.append(prefix + k)

    walk(cfg)
    if missing:
        raise SystemExit(f"config section {name!r} still has blank values: {', '.join(missing)}")
    budgets = {}
    for src, v in cfg["budgets"].items():
        if src not in SOURCES:
            raise SystemExit(f"unknown source {src!r} in budgets")
        n = 10 ** 13 if v == "all" else int(float(v))
        if n > 0:
            budgets[src] = n
    if not budgets:
        raise SystemExit("every budget is 0")
    cfg["budgets"] = budgets
    return cfg


def code_hash():
    h = hashlib.sha1()
    for d, _, files in sorted(os.walk(os.path.join(ROOT, "phase1"))):
        if "__pycache__" in d or os.sep + "tests" in d:
            continue
        for f in sorted(files):
            if f.endswith((".py", ".json")):
                with open(os.path.join(d, f), "rb") as fh:
                    h.update(f.encode() + fh.read())
    return h.hexdigest()[:12]


def disk_report(*paths):
    out = {}
    for p in paths:
        q = p
        while not os.path.exists(q):
            q = os.path.dirname(q) or "."
        u = shutil.disk_usage(q)
        out[p] = round(u.free / 1e9, 1)
        print(f"disk free at {p}: {u.free / 1e9:.1f} GB (of {u.total / 1e9:.1f} GB)")
    return out


def download(raw_dir, sources):
    from huggingface_hub import hf_hub_download
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set: the dataset is gated (Kaggle: Add-ons -> Secrets -> HF_TOKEN)")
    for path in [TEST_PATH] + [f"{HF_TRAIN_DIR}/train_{s}.json" for s in sources]:
        if os.path.exists(os.path.join(raw_dir, path)):
            print(f"already present: {path}")
            continue
        t = time.time()
        hf_hub_download(HF_DATASET_REPO, path, repo_type="dataset", local_dir=raw_dir)
        print(f"downloaded {path} ({time.time() - t:.0f}s)", flush=True)


def step(name, args, timings):
    print(f"\n===== {name}: python -m {' '.join(args)}", flush=True)
    t = time.time()
    subprocess.run([sys.executable, "-u", "-m", *args], cwd=ROOT, check=True)
    timings[name] = round(time.time() - t)
    print(f"===== {name} done in {timings[name]}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="phase1/configs/data.json")
    ap.add_argument("--section", default="full")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-download", action="store_true", help="raw files are already in --raw-dir")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--tokenizer-from", help="dir holding an already fitted a1bpe_<V>.json (+ .report.json): "
                    "copied instead of fitting again, so every session shares one tokenizer")
    ap.add_argument("--eval-mb", type=float, default=20, help="val text per source for the tokenizer report")
    args = ap.parse_args()

    cfg = load_section(os.path.join(ROOT, args.config) if not os.path.isabs(args.config) else args.config,
                       args.section)
    print(json.dumps(cfg, indent=1))
    sources = list(cfg["budgets"])
    os.makedirs(args.out, exist_ok=True)
    timings, t0 = {}, time.time()
    free = disk_report(args.raw_dir, args.out)
    need = sum(FILE_GB[s] for s in sources) + 1
    if not args.skip_download:
        if free[args.raw_dir] < need:
            raise SystemExit(f"need ~{need:.0f} GB free in {args.raw_dir} for the raw files, have {free[args.raw_dir]}")
        t = time.time()
        download(args.raw_dir, sources)
        timings["download"] = round(time.time() - t)

    test_file = os.path.join(args.raw_dir, TEST_PATH)
    profile = cfg.get("profile", True)
    if profile:
        step("profile", ["phase1.data.inspect_data", "--profile", "--full", "--raw-dir", args.raw_dir,
                         "--stats-out", os.path.join(args.out, "stats.json")], timings)

    tok = cfg["tokenizer"]
    tok_path = os.path.join(args.out, "tokenizer", f"a1bpe_{tok['vocab_size']}.json")
    if args.tokenizer_from:
        src = os.path.join(args.tokenizer_from, os.path.basename(tok_path))
        if not os.path.exists(src):
            raise SystemExit(f"--tokenizer-from: {src} not found (vocab_size in the config must match)")
        os.makedirs(os.path.dirname(tok_path), exist_ok=True)
        for f in (src, src.replace(".json", "") + ".report.json"):
            shutil.copy(f, os.path.join(os.path.dirname(tok_path), os.path.basename(f)))
        print(f"tokenizer copied from {args.tokenizer_from} (not refitted)")
    else:
        step("tokenizer", ["phase1.data.train_tokenizer", "--impl", "a1", "--raw-dir", args.raw_dir,
                           "--vocab-size", str(tok["vocab_size"]), "--mb-per-source", str(tok["mb_per_source"]),
                           "--val-frac", str(cfg["val_frac"]), "--eval-mb", str(args.eval_mb),
                           "--test-file", test_file, "--out", tok_path], timings)

    shards = os.path.join(args.out, "shards")
    dev_docs = max(4 * int(cfg["dev_examples_per_source"]), 100)
    keep_args = []
    if not profile:
        with open(tok_path.replace(".json", "") + ".report.json", encoding="utf-8") as f:
            rep = json.load(f)["sources"]
        keep_args = ["--keep-frac"]
        for s, n in cfg["budgets"].items():
            est = os.path.getsize(source_file(args.raw_dir, s)) / rep[s]["chars_per_token"]
            keep_args.append(f"{s}={min(1.0, 1.15 * n / est):.6f}")
            print(f"[{s}] ~{est:,.0f} tokens in file; keep {keep_args[-1]}")
    step("shards", ["phase1.data.make_shards", "--raw-dir", args.raw_dir, "--tokenizer", tok_path, *keep_args,
                    "--budget", *[f"{s}={n}" for s, n in cfg["budgets"].items()],
                    "--val-tokens-per-source", str(int(float(cfg["val_tokens_per_source"]))),
                    "--val-frac", str(cfg["val_frac"]), "--dev-docs-per-source", str(dev_docs),
                    "--batch-docs", "1024", "--workers", str(args.workers),
                    "--stats", os.path.join(args.out, "stats.json"), "--out", shards], timings)

    step("dev_set", ["phase1.eval.make_dev_set", "--dev-docs", os.path.join(shards, "dev_docs"),
                     "--per-source", str(int(cfg["dev_examples_per_source"])),
                     "--out", os.path.join(args.out, "dev")], timings)

    shutil.copy(test_file, os.path.join(args.out, "test.json"))
    manifest = {"section": args.section, "config": cfg, "timings_s": timings,
                "total_s": round(time.time() - t0), "code_hash": code_hash(), "tokenizer": os.path.relpath(tok_path, args.out),
                "disk_free_gb": free}
    with open(os.path.join(args.out, "prep_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(shards, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    print("\n===== summary")
    for s, m in meta["sources"].items():
        print(f"{s:14s} train {m['train']['tokens']:>14,} tok / {m['train']['docs']:>9,} docs   "
              f"val {m['val']['tokens']:>11,} tok   dev docs {m['dev_docs']}")
    print(json.dumps({k: manifest[k] for k in ("timings_s", "total_s", "code_hash")}, indent=1))


if __name__ == "__main__":
    main()
