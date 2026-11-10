"""Data report: corpus statistics, offset sanity, and the alignment oracle for each split mode.

    python -m phase2.report --data D --roots R --out runs_p2/data_report.json

The oracle converts gold spans to token BIO tags and back and scores the result with the strict
scorer: the best strict macro-F1 any token tagger can reach with that tokenisation.
"""
import argparse
import json
import os

from phase1.data.a1_bpe import load_tokenizer
from . import data as D
from .score import score
from .train import find_tokenizer


def oracle(texts, gold, labels, otok):
    ts = D.TagSet(labels)
    pred, st = {}, {}
    for did, t in texts.items():
        _ids, s, e = otok.encode(t)
        tags, x = D.tag_tokens(s, e, gold[did], ts)
        for k, v in x.items():
            st[k] = st.get(k, 0) + v
        pred[did] = D.spans_from_tags(tags, s, e, ts)
    truth = {k: {(a, b, l) for a, b, l, *_ in v} for k, v in gold.items()}
    sc = score(truth, pred, labels)
    return {"macro_f1": sc["macro_f1"], "per_label": {k: round(v["f1"], 4) for k, v in sc["per_label"].items()},
            "tagging": st}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--roots", nargs="+", default=["results"])
    ap.add_argument("--out", default="runs_p2/data_report.json")
    a = ap.parse_args(argv)
    data = D.load_split(D.find_root(a.data))
    tok = load_tokenizer(find_tokenizer(a.roots))
    labels = data["labels"]
    rep = {"labels": labels, "train_docs": len(data["texts"]), "test_docs": len(data["test"]),
           "fold_sizes": {k: sum(1 for v in data["fold"].values() if v == k) for k in (1, 2)},
           "offsets": D.offsets_check(data["texts"], data["gold"]),
           "train": D.corpus_stats(data["texts"], data["gold"], labels, D.OffsetTokenizer(tok, "edge")),
           "test_chars": {"docs": len(data["test"]), "total": sum(len(t) for t in data["test"].values())},
           "oracle": {}}
    sample_gold = next(iter(data["gold"].values()), [])[:3]
    rep["gold_example"] = [list(x) for x in sample_gold]
    for split in ("none", "edge", "punct"):
        rep["oracle"][split] = oracle(data["texts"], data["gold"], labels, D.OffsetTokenizer(tok, split))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: rep[k] for k in ("train_docs", "test_docs", "fold_sizes", "offsets")}, indent=1))
    print("tokens/doc:", rep["train"].get("tokens_per_doc"))
    print("labels:", {l: v["count"] for l, v in rep["train"]["per_label"].items()})
    print("overlaps:", rep["train"]["overlapping_pairs"], "nested:", rep["train"]["nested_pairs"])
    for s, o in rep["oracle"].items():
        print(f"oracle[{s}] strict macro-F1 {o['macro_f1']:.4f} {o['tagging']}")
    return rep


if __name__ == "__main__":
    main()
