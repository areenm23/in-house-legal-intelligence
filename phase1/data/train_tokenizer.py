"""Train a byte-level BPE (HF `tokenizers`, Rust) on a sample of the Phase-1 corpus and report on it.

  python -m phase1.data.train_tokenizer --raw-dir data/raw --vocab-size V \
      --mb-per-source 300 --out data/tokenizer/bpe_V.json

Only training-split documents are sampled (same split as make_shards), so the
validation documents are unseen by the tokenizer too. The report (written next to the
tokenizer as *.report.json) gives, per source, tokens per whitespace word measured on
validation documents, a lossless round-trip check, and the token lengths of the official
test prompts (to size the context window). The vocabulary
must stay <= 65536 so token ids fit in uint16 shards.
"""
import argparse
import json
import os
import re
import statistics
import time

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from .a1_bpe import A1Tokenizer, special_ids
from .common import SOURCES, is_val_doc, iter_json_records, record_text, source_file

SPECIAL_TOKENS = ["<|endoftext|>", "<pad>", "<s>", "<mask>"] + [f"<extra_{i}>" for i in range(12)]
WORD = re.compile(r"\S+")


def sample_texts(raw_dir, source, mb, val_frac, want_val):
    """Stream docs from one source; yield train docs (or val docs if want_val) up to `mb` MB of text."""
    budget, used = mb * 1e6, 0
    for i, rec in enumerate(iter_json_records(source_file(raw_dir, source))):
        if is_val_doc(source, i, val_frac) != want_val:
            continue
        text = record_text(rec)
        if not text.strip():
            continue
        yield text
        used += len(text)
        if used >= budget:
            return


def build_tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=True)
    return tok


def corpus(raw_dir, mb_per_source, val_frac):
    for src in SOURCES:
        if os.path.exists(source_file(raw_dir, src)):
            yield from sample_texts(raw_dir, src, mb_per_source, val_frac, want_val=False)


def train_a1(raw_dir, vocab_size, mb_per_source, val_frac, log_every):
    return A1Tokenizer.train(corpus(raw_dir, mb_per_source, val_frac), vocab_size, log_every=log_every)


def train(raw_dir, vocab_size, mb_per_source, val_frac, min_frequency):
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                                  special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=True)

    tok.train_from_iterator(corpus(raw_dir, mb_per_source, val_frac), trainer=trainer)
    return tok


def report(tok, raw_dir, val_frac, eval_mb, test_file):
    out = {"vocab_size": tok.get_vocab_size(), "special_ids": special_ids(tok), "sources": {}}
    for src in SOURCES:
        if not os.path.exists(source_file(raw_dir, src)):
            continue
        texts = list(sample_texts(raw_dir, src, eval_mb, val_frac, want_val=True))
        encs = tok.encode_batch(texts)
        n_tok = sum(len(e.ids) for e in encs)
        n_words = sum(len(WORD.findall(t)) for t in texts)
        lossy = isinstance(tok, A1Tokenizer)
        roundtrip_fail = sum(tok.decode(e.ids) != (" ".join(t.split()) if lossy else t)
                             for e, t in zip(encs, texts))
        unk = sum(i == tok.unk_id for e in encs for i in e.ids) if lossy else 0
        out["sources"][src] = {"val_docs": len(texts), "tokens_per_word": n_tok / max(n_words, 1),
                               "chars_per_token": sum(map(len, texts)) / max(n_tok, 1),
                               "roundtrip_failures": roundtrip_fail, "unk_tokens": unk}
        print(f"[{src}] tokens/word={n_tok / max(n_words, 1):.3f} "
              f"chars/token={sum(map(len, texts)) / max(n_tok, 1):.2f} "
              f"roundtrip failures={roundtrip_fail}/{len(texts)}")
    if test_file and os.path.exists(test_file):
        with open(test_file, encoding="utf-8") as f:
            test = json.load(f)
        lens = sorted(len(e.ids) for e in tok.encode_batch([r["text"] for r in test.values()]))
        out["test_prompt_tokens"] = {"median": statistics.median(lens), "p95": lens[int(0.95 * (len(lens) - 1))],
                                     "max": lens[-1]}
        print(f"[test prompts] tokens: {out['test_prompt_tokens']}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", choices=["a1", "hf"], required=True)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--vocab-size", type=int, required=True)
    ap.add_argument("--mb-per-source", type=float, required=True, help="MB of training text sampled per source")
    ap.add_argument("--min-frequency", type=int, default=2, help="hf only")
    ap.add_argument("--log-every", type=int, default=500, help="a1 only: print progress every N merges")
    ap.add_argument("--val-frac", type=float, default=0.01, help="must match make_shards")
    ap.add_argument("--eval-mb", type=float, default=20, help="MB of val text per source for the report")
    ap.add_argument("--test-file", default="data/raw/Phase_1_Pretraining/Test/test.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.vocab_size > 65536:
        ap.error("vocab must be <= 65536 for uint16 shards")
    t0 = time.time()
    if args.impl == "a1":
        tok = train_a1(args.raw_dir, args.vocab_size, args.mb_per_source, args.val_frac, args.log_every)
    else:
        tok = train(args.raw_dir, args.vocab_size, args.mb_per_source, args.val_frac, args.min_frequency)
    fit_seconds = time.time() - t0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok.save(args.out)
    print(f"saved {args.out} (vocab {tok.get_vocab_size()})")
    rep = report(tok, args.raw_dir, args.val_frac, args.eval_mb, args.test_file)
    rep["trained_on"] = {"impl": args.impl, "mb_per_source": args.mb_per_source, "val_frac": args.val_frac,
                         "min_frequency": args.min_frequency if args.impl == "hf" else None,
                         "fit_seconds": round(fit_seconds)}
    with open(args.out.replace(".json", "") + ".report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
