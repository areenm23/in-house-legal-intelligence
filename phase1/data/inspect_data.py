"""Inspect the Phase-1 data: list files and sizes on the hub, optionally download, then profile.

  python -m phase1.data.inspect_data --list
  python -m phase1.data.inspect_data --download --raw-dir data/raw
  python -m phase1.data.inspect_data --profile --raw-dir data/raw [--full]

--profile streams each train file (never loads it whole), prints record keys and a
sample, and writes data/stats.json with per-source document/word counts. Without
--full only the first --max-docs records are profiled; --full reads every file end to
end (needed later to turn a token budget into a sampling fraction).
"""
import argparse
import json
import os
import re
import statistics
import time

from .common import HF_DATASET_REPO, SOURCES, iter_json_records, record_text, source_file

PHASE1_PREFIX = "Phase_1_Pretraining"
WORD = re.compile(r"\S+")


def list_files():
    from huggingface_hub import HfApi
    files = [f for f in HfApi().list_repo_tree(HF_DATASET_REPO, repo_type="dataset",
                                               path_in_repo=PHASE1_PREFIX, recursive=True)
             if getattr(f, "size", None) is not None]
    total = 0
    for f in files:
        total += f.size
        print(f"{f.size / 1e9:8.2f} GB  {f.path}")
    print(f"{total / 1e9:8.2f} GB  total")
    return [f.path for f in files]


def download(raw_dir):
    from huggingface_hub import hf_hub_download
    for path in list_files():
        t = time.time()
        local = hf_hub_download(HF_DATASET_REPO, path, repo_type="dataset", local_dir=raw_dir)
        print(f"downloaded {path} -> {local} ({time.time() - t:.0f}s)")


def describe(values):
    if not values:
        return "n/a"
    q = statistics.quantiles(values, n=20) if len(values) > 1 else values * 19
    return (f"n={len(values)} mean={statistics.fmean(values):.0f} p5={q[0]:.0f} "
            f"median={statistics.median(values):.0f} p95={q[-1]:.0f} max={max(values)}")


def profile_train(raw_dir, max_docs, full):
    stats = {}
    for src in SOURCES:
        path = source_file(raw_dir, src)
        if not os.path.exists(path):
            print(f"[{src}] missing: {path}")
            continue
        print(f"\n[{src}] {path}  ({os.path.getsize(path) / 1e9:.2f} GB)")
        t = time.time()
        n_docs = n_words = n_chars = 0
        word_counts, key_sets = [], set()
        devanagari_docs = empty_docs = 0
        for i, rec in enumerate(iter_json_records(path)):
            if not full and i >= max_docs:
                break
            if i == 0:
                print(f"  first record keys: {list(rec)}")
                print(f"  first record text[:400]: {record_text(rec)[:400]!r}")
            key_sets.add(tuple(sorted(rec)))
            text = record_text(rec)
            w = len(WORD.findall(text))
            n_docs += 1
            n_words += w
            n_chars += len(text)
            empty_docs += w == 0
            devanagari_docs += bool(re.search("[ऀ-ॿ]", text))
            if len(word_counts) < 200_000:
                word_counts.append(w)
        dt = time.time() - t
        print(f"  docs={n_docs:,} words={n_words:,} chars={n_chars:,} ({'full pass' if full else 'sample'}, {dt:.0f}s)")
        print(f"  words/doc: {describe(word_counts)}")
        print(f"  empty docs={empty_docs}  docs containing Devanagari={devanagari_docs}")
        if len(key_sets) > 1:
            print(f"  WARNING: records have differing key sets: {sorted(key_sets)[:5]}")
        stats[src] = {"docs": n_docs, "words": n_words, "chars": n_chars, "full_pass": full,
                      "file_bytes": os.path.getsize(path)}
    return stats


def profile_test(raw_dir):
    test_dir = os.path.join(raw_dir, PHASE1_PREFIX, "Test")
    if not os.path.isdir(test_dir):
        print(f"\n[test] missing dir {test_dir}")
        return None
    for name in os.listdir(test_dir):
        path = os.path.join(test_dir, name)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        ids = list(data)
        words = [len(WORD.findall(data[i]["text"])) for i in ids]
        print(f"\n[test] {name}: {len(ids)} prompts, ids {ids[:3]} ... {ids[-2:]}")
        print(f"  record keys: {list(data[ids[0]])}")
        print(f"  prompt words: {describe(words)}")
        print(f"  contains newlines: {sum(chr(10) in data[i]['text'] for i in ids)} / {len(ids)}")
        print(f"  first prompt start: {data[ids[0]]['text'][:300]!r}")
        print(f"  first prompt end:   {data[ids[0]]['text'][-200:]!r}")
        return {"file": name, "n": len(ids), "prompt_words_median": statistics.median(words)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--max-docs", type=int, default=2000)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--stats-out", default="data/stats.json")
    args = ap.parse_args()
    if args.list:
        list_files()
    if args.download:
        download(args.raw_dir)
    if args.profile:
        stats = {"train": profile_train(args.raw_dir, args.max_docs, args.full),
                 "test": profile_test(args.raw_dir)}
        os.makedirs(os.path.dirname(args.stats_out) or ".", exist_ok=True)
        with open(args.stats_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"\nwrote {args.stats_out}")


if __name__ == "__main__":
    main()
