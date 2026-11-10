"""Tokenize the Phase-1 corpus once into memory-mappable uint16 shards.

  python -m phase1.data.make_shards --raw-dir data/raw --tokenizer data/tokenizer/bpe_V.json \
      --budget acts=N1 high_courts=N2 supreme_court=N3 tribunals=N4 \
      --val-tokens-per-source M --out data/shards

Per source this writes
  <out>/train/<source>.bin        uint16 token ids, each document followed by the EOS id
                                  (A-1 BPE: </s> = 3; HF BPE: <|endoftext|> = 0; see meta.json)
  <out>/train/<source>.idx.npy    int64 start offset of every document in the .bin
  <out>/val/<source>.bin / .idx.npy   held-out documents (whole documents, never in train)
  <out>/dev_docs/<source>.jsonl   raw text of held-out documents, for building dev prompts
  <out>/meta.json                 everything needed to reproduce / sanity-check the shards

The data mix is set by --budget (training tokens per source). Documents are
subsampled uniformly across each file with a deterministic hash, so the budget is not
just "the first N documents". The keep fraction is estimated from data/stats.json (run
inspect_data --profile --full) and the tokenizer report; or pass --keep-frac directly.
Reading streams one record at a time, so RAM stays small regardless of file size.
"""
import argparse
import json
import os
import re
import time

import numpy as np
from .a1_bpe import A1Tokenizer, load_tokenizer, special_ids
from .common import SOURCES, is_val_doc, iter_json_records, keep_train_doc, record_text, source_file

WORD = re.compile(r"\S+")


class ShardWriter:
    def __init__(self, path_prefix, eos_id):
        self.eos = np.asarray([eos_id], dtype=np.uint16)
        self.bin_path = path_prefix + ".bin"
        self.idx_path = path_prefix + ".idx.npy"
        self.f = open(self.bin_path, "wb")
        self.starts, self.n_tokens = [], 0

    def add(self, encodings):
        chunks = []
        for enc in encodings:
            self.starts.append(self.n_tokens)
            ids = np.concatenate([np.asarray(enc.ids, dtype=np.uint16), self.eos])
            chunks.append(ids)
            self.n_tokens += len(ids)
        if chunks:
            np.concatenate(chunks).tofile(self.f)

    def close(self):
        self.f.close()
        np.save(self.idx_path, np.asarray(self.starts, dtype=np.int64))
        return {"tokens": self.n_tokens, "docs": len(self.starts)}


def parse_kv(items, cast):
    out = {}
    for item in items or []:
        k, v = item.split("=")
        if k not in SOURCES:
            raise SystemExit(f"unknown source {k!r}; expected one of {SOURCES}")
        out[k] = cast(v)
    return out


def estimate_keep_frac(src, budget, stats_path, tok_report_path):
    with open(stats_path, encoding="utf-8") as f:
        s = json.load(f)["train"][src]
    with open(tok_report_path, encoding="utf-8") as f:
        tpw = json.load(f)["sources"][src]["tokens_per_word"]
    if not s.get("full_pass"):
        print(f"  WARNING: stats for {src} come from a sample, not a full pass; keep fraction is unreliable")
    est_total = s["words"] * tpw
    return min(1.0, 1.1 * budget / est_total), est_total


def process_source(src, args, tok, eos_id, keep_frac, budget):
    path = source_file(args.raw_dir, src)
    os.makedirs(os.path.join(args.out, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "val"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "dev_docs"), exist_ok=True)
    train_w = ShardWriter(os.path.join(args.out, "train", src), eos_id)
    val_w = ShardWriter(os.path.join(args.out, "val", src), eos_id)
    dev_f = open(os.path.join(args.out, "dev_docs", f"{src}.jsonl"), "w", encoding="utf-8")
    train_batch, val_batch = [], []
    n_dev, seen, t0 = 0, 0, time.time()

    def flush(batch, writer):
        if isinstance(tok, A1Tokenizer):
            writer.add(tok.encode_batch(batch, workers=args.workers))
        else:
            writer.add(tok.encode_batch(batch))
        batch.clear()

    for i, rec in enumerate(iter_json_records(path)):
        seen = i + 1
        train_full = train_w.n_tokens >= budget
        val_full = val_w.n_tokens >= args.val_tokens_per_source
        if train_full and val_full and n_dev >= args.dev_docs_per_source:
            break
        text = record_text(rec)
        if not text.strip():
            continue
        if is_val_doc(src, i, args.val_frac):
            if not val_full:
                val_batch.append(text)
            if n_dev < args.dev_docs_per_source and len(WORD.findall(text)) >= args.dev_min_words:
                dev_f.write(json.dumps({"source": src, "index": i, "text": text}, ensure_ascii=False) + "\n")
                n_dev += 1
        elif not train_full and keep_train_doc(src, i, keep_frac):
            train_batch.append(text)
        if len(train_batch) >= args.batch_docs:
            flush(train_batch, train_w)
        if len(val_batch) >= args.batch_docs:
            flush(val_batch, val_w)
        if seen % 20000 == 0:
            print(f"  [{src}] docs read={seen:,} train tok={train_w.n_tokens:,} "
                  f"val tok={val_w.n_tokens:,} dev docs={n_dev} ({time.time() - t0:.0f}s)")
    flush(train_batch, train_w)
    flush(val_batch, val_w)
    dev_f.close()
    res = {"train": train_w.close(), "val": val_w.close(), "dev_docs": n_dev, "docs_read": seen,
           "keep_frac": keep_frac, "budget": budget, "seconds": round(time.time() - t0)}
    if res["train"]["tokens"] < budget:
        print(f"  NOTE: [{src}] train budget not reached ({res['train']['tokens']:,} < {budget:,}); "
              f"raise --keep-frac or the source has fewer tokens")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--budget", nargs="+", required=True, help="source=train_tokens ...")
    ap.add_argument("--keep-frac", nargs="*", help="source=fraction ... (overrides the estimate)")
    ap.add_argument("--val-tokens-per-source", type=int, required=True)
    ap.add_argument("--val-frac", type=float, default=0.01, help="fraction of documents held out; must match train_tokenizer")
    ap.add_argument("--dev-docs-per-source", type=int, default=500)
    ap.add_argument("--dev-min-words", type=int, default=700, help="dev docs need room for a 512+128-word window")
    ap.add_argument("--batch-docs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="A-1 BPE encoding processes")
    ap.add_argument("--stats", default="data/stats.json")
    ap.add_argument("--out", default="data/shards")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    assert tok.get_vocab_size() <= 65536, "uint16 shards need vocab <= 65536"
    ids = special_ids(tok)
    assert ids["eos"] is not None, "tokenizer has no EOS / document-separator token"
    assert ids["bos"] == 2, "phase1.train.data.BOS_ID assumes <s> = 2"
    budgets = parse_kv(args.budget, lambda v: int(float(v)))
    explicit = parse_kv(args.keep_frac, float)
    tok_report = args.tokenizer.replace(".json", "") + ".report.json"

    meta = {"tokenizer": os.path.abspath(args.tokenizer), "vocab_size": tok.get_vocab_size(),
            "tokenizer_impl": "a1" if isinstance(tok, A1Tokenizer) else "hf",
            "dtype": "uint16", "eos_id": ids["eos"], "bos_id": ids["bos"], "pad_id": ids["pad"],
            "val_frac": args.val_frac, "sources": {}}
    for src, budget in budgets.items():
        if src in explicit:
            keep = explicit[src]
        else:
            keep, est = estimate_keep_frac(src, budget, args.stats, tok_report)
            print(f"[{src}] estimated corpus tokens ~{est:,.0f}; keep fraction {keep:.4f}")
        print(f"[{src}] budget {budget:,} train tokens")
        meta["sources"][src] = process_source(src, args, tok, ids["eos"], keep, budget)
        print(f"[{src}] done: {meta['sources'][src]}")
    if isinstance(tok, A1Tokenizer):
        tok.close()
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {os.path.join(args.out, 'meta.json')}")


if __name__ == "__main__":
    main()
