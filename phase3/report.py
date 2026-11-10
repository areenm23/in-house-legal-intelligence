"""Phase-3 data report: file formats, counts, lengths, label distribution per domain/split, where each
role sits in a document, label transitions, train/test sentence overlap, trivial baselines, and a
parity check of our scorer against the official evaluate_phase3.py on the dev set.

    python -m phase3.report --data DATA_ROOT --roots results --out runs_p3/data_report.json
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter

import numpy as np

from . import data as D
from .score import score, write_truth
from .train import find_tokenizer, load_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "course_files", "eval", "Evaluation Script", "evaluate_phase3.py")


def _pct(x):
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return None
    return {"mean": round(float(x.mean()), 1), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "max": float(x.max()), "total": float(x.sum())}


def official_parity(dev_docs, pred, labels, labels_file):
    """Score `pred` with the official script and with ours -> (official stdout macro-F1, ours)."""
    with tempfile.TemporaryDirectory() as tmp:
        for dom in D.DOMAINS:
            write_truth([d for d in dev_docs if d.domain == dom], os.path.join(tmp, f"{dom}.json"))
        pj = {d.id: [labels[i] for i in pred[d.key]] for d in dev_docs}
        json.dump(pj, open(os.path.join(tmp, "pred.json"), "w"))
        r = subprocess.run([sys.executable, EVAL, "--truth-cl", os.path.join(tmp, "CL.json"), "--truth-it",
                            os.path.join(tmp, "IT.json"), "--prediction", os.path.join(tmp, "pred.json"),
                            "--labels", labels_file], capture_output=True, text=True)
        off = None
        for ln in r.stdout.splitlines():
            if ln.startswith("Macro-F1:"):
                off = float(ln.split()[-1])
        return off, score(dev_docs, pred, labels)["macro_f1"], (r.stderr or "").strip()[-300:]


def build(root, roots, max_sent_tokens=256, cache=None):
    print(D.describe(root))
    tok = find_tokenizer(roots)
    data = load_data(root, tok, max_sent_tokens, cache or os.path.join(tempfile.gettempdir(), "p3cache"))
    labels = data["labels"]
    n = len(labels)
    rep = {"root": root, "files": data["files"], "labels": labels, "splits": {}}
    for split in ("train", "dev", "test"):
        for dom, docs in data[split].items():
            if not docs:
                continue
            ns = [len(d) for d in docs]
            nt = [d.n_tok for d in docs]
            sl = [int(c) for d in docs for c in np.bincount(d.sent_of, minlength=len(d))] if docs else []
            rec = {"docs": len(docs), "empty_docs": sum(x == 0 for x in ns), "sentences_per_doc": _pct(ns),
                   "tokens_per_doc": _pct(nt), "tokens_per_sentence": _pct(sl)}
            rec["empty_sentences"] = sum(1 for d in docs for s in d.sents if not s.strip())
            rec["sentences_over_cap"] = sum(x >= max_sent_tokens for x in sl)
            if split != "test":
                cnt = D.label_counts(docs, n)
                rec["labels"] = {labels[i]: int(cnt[i]) for i in range(n)}
                pos = {l: [] for l in labels}
                trans = Counter()
                for d in docs:
                    for i, y in enumerate(d.y):
                        pos[labels[y]].append(i / max(1, len(d) - 1))
                        if i:
                            trans[(labels[d.y[i - 1]], labels[y])] += 1
                rec["median_rel_pos"] = {l: round(float(np.median(v)), 3) if v else None for l, v in pos.items()}
                same = sum(c for (a, b), c in trans.items() if a == b)
                rec["same_label_as_previous"] = round(same / max(1, sum(trans.values())), 3)
                rec["top_transitions"] = [f"{a} -> {b}: {c}" for (a, b), c in trans.most_common(12) if a != b]
            rep["splits"][f"{dom}_{split}"] = rec
    norm = lambda s: " ".join(s.lower().split())
    train_s = {norm(s) for d in D.all_docs(data["train"]) for s in d.sents if len(s.split()) >= 5}
    for split in ("dev", "test"):
        docs = D.all_docs(data[split])
        tot = sum(1 for d in docs for s in d.sents if len(s.split()) >= 5)
        hit = sum(1 for d in docs for s in d.sents if len(s.split()) >= 5 and norm(s) in train_s)
        rep[f"{split}_sentences_seen_in_train"] = {"count": hit, "of": tot, "frac": round(hit / max(1, tot), 4)}
    ids = [d.id for sp in ("train", "dev", "test") for d in D.all_docs(data[sp])]
    rep["duplicate_doc_ids_across_files"] = sorted(k for k, v in Counter(ids).items() if v > 1)[:20]
    dev = D.all_docs(data["dev"])
    if dev:
        cnt = D.label_counts(D.all_docs(data["train"]), n)
        maj = int(cnt.argmax())
        pred = {d.key: np.full(len(d), maj) for d in dev}
        rep["baseline_majority_dev_f1"] = round(score(dev, pred, labels)["macro_f1"], 4)
        tab = np.zeros((10, n))
        for d in D.all_docs(data["train"]):
            for i, y in enumerate(d.y):
                tab[min(9, int(10 * i / max(1, len(d)))), y] += 1
        ppos = {d.key: np.array([tab[min(9, int(10 * i / max(1, len(d))))].argmax() for i in range(len(d))],
                                dtype=np.int64) for d in dev}
        rep["baseline_position_dev_f1"] = round(score(dev, ppos, labels)["macro_f1"], 4)
        if os.path.exists(EVAL):
            off, ours, err = official_parity(dev, ppos, labels, data["files"]["labels"])
            rep["official_parity"] = {"official": off, "ours": round(ours, 6), "stderr": err,
                                      "match": off is not None and abs(off - ours) < 1e-5}
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--roots", nargs="+", default=["results"])
    ap.add_argument("--out", default="runs_p3/data_report.json")
    ap.add_argument("--max-sent-tokens", type=int, default=256)
    a = ap.parse_args(argv)
    root = D.find_root(a.data)
    if root is None:
        sys.exit(f"no Phase_3_RR data under {a.data}")
    rep = build(root, a.roots, a.max_sent_tokens, os.path.join(os.path.dirname(a.out) or ".", "_cache"))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep, indent=1)[:6000])
    par = rep.get("official_parity")
    if par and not par["match"]:
        print("WARNING: our scorer and the official evaluator disagree:", par)


if __name__ == "__main__":
    main()
