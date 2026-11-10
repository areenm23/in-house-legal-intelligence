"""Phase-6 data report: raw format, counts, document / summary lengths (words and A-1 tokens), compression
ratio, how much of a document each input budget covers, extractive baselines on the held-out dev split
(lead-k words, central sentences, reference-guided oracle sentences = an upper bound for selection), and
parity of our ROUGE with the official evaluate_phase6.py (+ BERTScore when bert_score is installed; this
also downloads its roberta-large model once, before the GPU queues start).

    python -m phase6.report --data DATA_ROOT --roots results --out runs_p6/data_report.json
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

from phase1.data.a1_bpe import load_tokenizer
from phase3.train import find_tokenizer
from . import data as D
from .score import evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "course_files", "eval", "Evaluation Script", "evaluate_phase6.py")


def _pct(x):
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return None
    return {"mean": round(float(x.mean()), 1), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)), "max": float(x.max())}


def extractive(sents_text, sents_tok, k_words, how, idf=None, ref_tok=None):
    """Pick sentences (kept in document order) until about k_words words."""
    if how == "lead":
        return " ".join(" ".join(sents_text).split()[:k_words])
    if how == "oracle":
        sc = np.array([D._unigram_f1(t, ref_tok) for t in sents_tok])
    else:
        x = D.select_input(sents_tok, 10 ** 9, "head")
        u, c = np.unique(x, return_counts=True)
        w2 = idf.idf ** 2
        dv = np.zeros(len(w2))
        dv[u] = c
        sc = []
        for t in sents_tok:
            us, cs = np.unique(t.astype(np.int64), return_counts=True)
            sc.append((cs * w2[us] * dv[us]).sum() / (np.sqrt((cs ** 2 * w2[us]).sum()) + 1e-9))
        sc = np.array(sc)
    keep, n = [], 0
    for j in np.argsort(-sc):
        keep.append(j)
        n += len(sents_text[j].split())
        if n >= k_words:
            break
    return " ".join(sents_text[j] for j in sorted(keep))


def official(preds, refs):
    with tempfile.TemporaryDirectory() as tmp:
        json.dump({f"d{i}": r for i, r in enumerate(refs)}, open(os.path.join(tmp, "t.json"), "w"))
        json.dump([{"id": f"d{i}", "summary": p} for i, p in enumerate(preds)], open(os.path.join(tmp, "p.json"), "w"))
        r = subprocess.run([sys.executable, EVAL, "--truth", os.path.join(tmp, "t.json"), "--prediction",
                            os.path.join(tmp, "p.json")], capture_output=True, text=True)
        vals = {}
        for ln in r.stdout.splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                vals[k.strip()] = v.strip()
        return vals, (r.stderr or "").strip()[-300:]


def build(root, roots, cache, n_base=150, bert=True):
    print(D.describe(root), flush=True)
    t0 = time.time()
    tok_path = find_tokenizer(roots)
    tok = load_tokenizer(tok_path)
    data = D.load_data(root, tok_path, cache)
    tr, dv, te = data["train"], data["dev"], data["test"]
    rep = {"root": root, "files": data["files"], "tokenize_minutes": round((time.time() - t0) / 60, 2), "splits": {}}
    for name, d in (("train", tr), ("dev", dv), ("test", te)):
        nt = np.array([sum(len(s) for s in d.sents[i]) for i in range(len(d))])
        rec = {"docs": len(d), "doc_words": _pct(d.n_words), "doc_tokens": _pct(nt),
               "sentences_per_doc": _pct([len(s) for s in d.sents]),
               "frac_doc_covered_by_budget": {str(b): round(float(np.minimum(1, b / np.maximum(nt, 1)).mean()), 4)
                                              for b in (1024, 1536, 2048, 4096)},
               "duplicate_ids": len(d.ids) - len(set(d.ids)), "id_examples": d.ids[:3]}
        if d.summ is not None:
            st = np.array([len(x) for x in d.summ])
            rec["summary_tokens"] = _pct(st)
            rec["summary_words"] = _pct([len(r.split()) for r in d.ref])
            rec["compression_tokens"] = _pct(st / np.maximum(nt, 1))
            rec["frac_summaries_le"] = {str(b): round(float((st <= b).mean()), 4) for b in (256, 512, 768, 1024)}
        rep["splits"][name] = rec
    rep["test_ids_in_train"] = len(set(te.ids) & set(tr.ids) | set(te.ids) & set(dv.ids))
    idf = D.IDF(tr)
    k = int(np.median([len(r.split()) for r in tr.ref]))
    n = min(n_base, len(dv))
    refs = [dv.ref[i] for i in range(n)]
    texts = [[" ".join(tok.decode(s.tolist()).split()) for s in dv.sents[i]] for i in range(n)]
    base = {}
    for how in ("lead", "central", "oracle"):
        preds = [extractive(texts[i], dv.sents[i], k, how, idf, dv.summ[i]) or "Judgment." for i in range(n)]
        try:
            base[how] = {m: (round(v, 4) if isinstance(v, float) else v) for m, v in evaluate(preds, refs, with_bert=bert).items()}
        except ImportError as e:
            base[how] = {"error": str(e)}
        if how == "lead" and os.path.exists(EVAL):
            vals, err = official(preds, refs)
            ours = base[how]
            rep["official_parity"] = {"official": vals, "stderr": err, "ours_rouge1": ours.get("rouge1"),
                                      "ours_rougeL": ours.get("rougeL"), "ours_bertscore": ours.get("bertscore_f1"),
                                      "match": "ROUGE-1" in vals and abs(float(vals["ROUGE-1"]) - (ours.get("rouge1") or -1)) < 1e-4
                                      and abs(float(vals["ROUGE-L"]) - (ours.get("rougeL") or -1)) < 1e-4}
    rep["extractive_baselines_dev"] = {"docs": n, "k_words": k, **base}
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--roots", nargs="+", default=["results"])
    ap.add_argument("--out", default="runs_p6/data_report.json")
    ap.add_argument("--no-bert", action="store_true")
    a = ap.parse_args(argv)
    root = D.find_root(a.data)
    if root is None:
        sys.exit(f"no Phase_6_Summ data under {a.data}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rep = build(root, a.roots, os.path.join(os.path.dirname(a.out) or ".", "_cache"), bert=not a.no_bert)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep, indent=1)[:8000])
    par = rep.get("official_parity")
    if par and not par["match"]:
        print("WARNING: our ROUGE and the official evaluator disagree:", par)
    return rep


if __name__ == "__main__":
    main()
