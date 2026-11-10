"""Build a local Phase-1 dev set in the official format from held-out (val) documents.

  python -m phase1.eval.make_dev_set --dev-docs data/shards/dev_docs --per-source 125 --out data/dev

Writes <out>/prompts.json  {id: {"text": 512-word prompt}}   (same shape as the test file)
       <out>/truth.json    {id: {"text": next 128 words}}    (what evaluate_phase1.py --truth expects)
       <out>/sources.json  {id: source}                      (for per-source breakdowns)

The window start is random within each document (seeded), and the prompt/reference are
cut from the original text so the document's own whitespace and newlines are kept.
Check the official test prompts' formatting (inspect_data --profile) and use
--normalize-whitespace if they turn out to be single-space joined.
"""
import argparse
import glob
import json
import os
import random
import re

WORD = re.compile(r"\S+")


def cut_window(text, start_word, n_prompt, n_ref, normalize):
    spans = [m.span() for m in WORD.finditer(text)]
    p = spans[start_word:start_word + n_prompt]
    r = spans[start_word + n_prompt:start_word + n_prompt + n_ref]
    if normalize:
        return (" ".join(text[a:b] for a, b in p), " ".join(text[a:b] for a, b in r))
    return text[p[0][0]:p[-1][1]], text[r[0][0]:r[-1][1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-docs", default="data/shards/dev_docs")
    ap.add_argument("--per-source", type=int, required=True)
    ap.add_argument("--prompt-words", type=int, default=512)
    ap.add_argument("--ref-words", type=int, default=128)
    ap.add_argument("--normalize-whitespace", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/dev")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    need = args.prompt_words + args.ref_words
    prompts, truth, sources = {}, {}, {}
    for path in sorted(glob.glob(os.path.join(args.dev_docs, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            docs = [json.loads(line) for line in f]
        docs = [d for d in docs if len(WORD.findall(d["text"])) >= need]
        rng.shuffle(docs)
        for d in docs[:args.per_source]:
            n = len(WORD.findall(d["text"]))
            start = rng.randint(0, n - need)
            p, r = cut_window(d["text"], start, args.prompt_words, args.ref_words, args.normalize_whitespace)
            key = f"{d['source']}-{d['index']}"
            prompts[key], truth[key], sources[key] = {"text": p}, {"text": r}, d["source"]
        print(f"{os.path.basename(path)}: {min(len(docs), args.per_source)} examples")
    os.makedirs(args.out, exist_ok=True)
    for name, obj in [("prompts", prompts), ("truth", truth), ("sources", sources)]:
        with open(os.path.join(args.out, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(prompts)} examples to {args.out}")


if __name__ == "__main__":
    main()
