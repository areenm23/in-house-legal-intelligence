"""Phase-4 data report: raw format, counts, label balance, document lengths (per split and per label), how
much of a document each view covers, duplicated texts within / across splits, trivial baselines, and a
parity check of our scorer against the official evaluate_phase4.py. With --baselines, also TF-IDF +
logistic-regression baselines on different parts of the document (token-id n-grams, CPU) - a cheap first
look at which sections carry the outcome.

    python -m phase4.report --data DATA_ROOT --roots results --out runs_p4/data_report.json [--baselines]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

from . import data as D
from .score import prf, write_truth
from .train import find_tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "course_files", "eval", "Evaluation Script", "evaluate_phase4.py")


def _pct(x):
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return None
    return {"mean": round(float(x.mean()), 1), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)), "max": float(x.max()), "total": float(x.sum())}


def official_parity(ids, y, pred):
    """Score `pred` with the official script and with ours -> (official macro-F1, ours, stderr)."""
    with tempfile.TemporaryDirectory() as tmp:
        write_truth(ids, y, os.path.join(tmp, "truth.json"))
        json.dump([{"id": str(i), "pred": int(p)} for i, p in zip(ids, pred)], open(os.path.join(tmp, "pred.json"), "w"))
        r = subprocess.run([sys.executable, EVAL, "--truth", os.path.join(tmp, "truth.json"), "--prediction",
                            os.path.join(tmp, "pred.json")], capture_output=True, text=True)
        off = None
        for ln in r.stdout.splitlines():
            if ln.startswith("Macro-F1:"):
                off = float(ln.split()[-1])
        return off, prf(y, pred)["macro_f1"], (r.stderr or "").strip()[-300:]


def build(root, roots, cache=None, describe=True):
    if describe:
        print(D.describe(root), flush=True)
    t0 = time.time()
    data = D.load_data(root, find_tokenizer(roots), cache or os.path.join(tempfile.gettempdir(), "p4cache"))
    rep = {"root": root, "files": data["files"], "tokenize_minutes": round((time.time() - t0) / 60, 2), "splits": {}}
    for name in ("train", "dev", "test"):
        sp = data[name]
        if sp is None:
            continue
        nt = sp.n_tok()
        rec = {"docs": len(sp), "tokens_per_doc": _pct(nt), "chars_per_doc": _pct(sp.n_chars),
               "empty_docs": int((sp.n_chars == 0).sum()),
               "tokens_per_char": round(float(nt.sum() / max(1, sp.n_chars.sum())), 4),
               "frac_docs_le": {str(L): round(float((nt <= L).mean()), 4) for L in (512, 1024, 2048, 4096, 8192)},
               "mean_frac_seen_by_last": {str(L): round(float(np.minimum(1, L / np.maximum(nt, 1)).mean()), 4)
                                          for L in (512, 1024, 2048, 4096)},
               "duplicate_ids": len(sp.ids) - len(set(sp.ids)), "id_examples": sp.ids[:3],
               "duplicate_texts_within": len(sp.h) - len(set(sp.h))}
        if sp.labelled:
            rec["labels"] = {str(c): int((sp.y == c).sum()) for c in (0, 1)}
            rec["pos_rate"] = round(float(sp.y.mean()), 4)
            rec["tokens_by_label"] = {str(c): _pct(nt[sp.y == c]) for c in (0, 1)}
            q = np.quantile(nt, [0.25, 0.5, 0.75])
            b = np.digitize(nt, q)
            rec["pos_rate_by_length_quartile"] = [round(float(sp.y[b == k].mean()), 4) if (b == k).any() else None
                                                  for k in range(4)]
            groups = {}
            for i, hh in enumerate(sp.h):
                groups.setdefault(hh, set()).add(int(sp.y[i]))
            rec["duplicate_texts_with_conflicting_labels"] = sum(len(v) > 1 for v in groups.values())
        rep["splits"][name] = rec
    tr, dv, te = data["train"], data["dev"], data["test"]
    htr = set(tr.h)
    for name, sp in (("dev", dv), ("test", te)):
        if sp is not None:
            rep[f"{name}_texts_in_train"] = int(sum(h in htr for h in sp.h))
            rep[f"{name}_ids_in_train"] = int(len(set(sp.ids) & set(tr.ids)))
    if dv is not None and te is not None:
        rep["test_texts_in_dev"] = int(sum(h in set(dv.h) for h in te.h))
    if dv is not None and dv.labelled:
        maj = int(tr.y.mean() >= 0.5)
        rep["baseline_majority_dev_f1"] = round(prf(dv.y, np.full(len(dv), maj))["macro_f1"], 4)
        lt, lv = np.log1p(tr.n_tok()), np.log1p(dv.n_tok())
        best = (-1, 0, 1)
        for t in np.quantile(lt, np.linspace(0.05, 0.95, 37)):
            for sgn in (1, -1):
                f = prf(tr.y, (sgn * (lt - t) > 0).astype(int))["macro_f1"]
                if f > best[0]:
                    best = (f, t, sgn)
        pl = (best[2] * (lv - best[1]) > 0).astype(int)
        rep["baseline_length_dev_f1"] = round(prf(dv.y, pl)["macro_f1"], 4)
        rep["baseline_length_rule"] = {"log1p_tokens_threshold": round(float(best[1]), 3),
                                       "label1_if": "longer" if best[2] > 0 else "shorter"}
        if os.path.exists(EVAL):
            off, ours, err = official_parity(dv.ids, dv.y, pl)
            rep["official_parity"] = {"official": off, "ours": round(ours, 6), "stderr": err,
                                      "match": off is not None and abs(off - ours) < 1e-5}
    return rep, data


def tfidf_baselines(data, max_docs=16000, views=(("first512", "first", 512), ("last512", "last", 512),
                                                    ("middle512", "middle", 512), ("last2048", "last", 2048),
                                                    ("all8192", "headtail", 8192))):
    """Token-id 1-2-gram TF-IDF + logistic regression per view (train subset -> dev)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    tr, dv = data["train"], data["dev"]
    out = {}
    for name, ctx, L in views:
        t0 = time.time()
        idx = D.stratified_subset(tr.y, max_docs if L <= 2048 else max_docs // 2, seed=0)
        txt = lambda sp, i: " ".join(map(str, D.views(sp.doc(i), ctx, L, head_tokens=L // 4)[0][0].tolist()))
        vec = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=3, max_features=300000,
                              sublinear_tf=True, dtype=np.float32)
        Xtr = vec.fit_transform([txt(tr, i) for i in idx])
        Xdv = vec.transform([txt(dv, i) for i in range(len(dv))])
        clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000, solver="liblinear")
        clf.fit(Xtr, tr.y[idx])
        pred = clf.predict(Xdv)
        s = prf(dv.y, pred)
        out[name] = {"dev_f1": round(s["macro_f1"], 4), "dev_acc": round(s["accuracy"], 4), "train_docs": len(idx),
                     "minutes": round((time.time() - t0) / 60, 2)}
        print(f"TF-IDF+LR {name}: dev macro-F1 {s['macro_f1']:.4f}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--roots", nargs="+", default=["results"])
    ap.add_argument("--out", default="runs_p4/data_report.json")
    ap.add_argument("--baselines", action="store_true", help="also TF-IDF + LR per document view (CPU, minutes)")
    ap.add_argument("--baselines-only", action="store_true", help="only the TF-IDF baselines -> <out> (a separate file)")
    ap.add_argument("--max-docs", type=int, default=16000)
    a = ap.parse_args(argv)
    root = D.find_root(a.data)
    if root is None:
        sys.exit(f"no Phase_4_CJPE data under {a.data}")
    cache = os.path.join(os.path.dirname(a.out) or ".", "_cache")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.baselines_only:
        data = D.load_data(root, find_tokenizer(a.roots), cache)
        res = tfidf_baselines(data, a.max_docs)
        json.dump(res, open(a.out, "w"), indent=1)
        return res
    rep, data = build(root, a.roots, cache)
    if a.baselines:
        rep["tfidf_lr"] = tfidf_baselines(data, a.max_docs)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep, indent=1)[:8000])
    par = rep.get("official_parity")
    if par and not par["match"]:
        print("WARNING: our scorer and the official evaluator disagree:", par)
    return rep


if __name__ == "__main__":
    main()
