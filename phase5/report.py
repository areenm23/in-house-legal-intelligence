"""Phase-5 data report: raw format, counts, document lengths, label statistics (labels per document,
label frequency and its long tail, labels missing from dev, co-occurring pairs), dev/test overlap (the
clean-dev filter), trivial baselines, and parity of our scorer with the official evaluate_phase5.py on
clean dev. With --baselines-only: TF-IDF + one-vs-rest logistic regression per document view.

    python -m phase5.report --data DATA_ROOT --roots results --out runs_p5/data_report.json
    python -m phase5.report --data DATA_ROOT --roots results --out runs_p5/tfidf_baselines.json --baselines-only
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

from phase3.train import find_tokenizer
from . import data as D
from .score import apply, cross_fit, fit_global, metrics, write_truth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "course_files", "eval", "Evaluation Script", "evaluate_phase5.py")


def _pct(x):
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return None
    return {"mean": round(float(x.mean()), 2), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)), "max": float(x.max())}


def official_parity(ids, gold_lists, D_pred):
    with tempfile.TemporaryDirectory() as tmp:
        write_truth(ids, gold_lists, os.path.join(tmp, "truth.json"))
        json.dump([{"id": str(i), "pred": [int(x) for x in np.where(r)[0]]} for i, r in zip(ids, D_pred)],
                  open(os.path.join(tmp, "pred.json"), "w"))
        r = subprocess.run([sys.executable, EVAL, "--truth", os.path.join(tmp, "truth.json"), "--prediction",
                            os.path.join(tmp, "pred.json")], capture_output=True, text=True)
        vals = {}
        for ln in r.stdout.splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                vals[k.strip()] = v.strip()
        return vals, (r.stderr or "").strip()[-300:]


def build(root, roots, cache):
    print(D.describe(root), flush=True)
    t0 = time.time()
    data = D.load_data(root, find_tokenizer(roots), cache)
    NL = data["n_labels"]
    tr, dv, te, dc = data["train"], data["dev"], data["test"], data["dev_clean"]
    rep = {"root": root, "files": data["files"], "tokenize_minutes": round((time.time() - t0) / 60, 2), "n_labels": NL,
           "splits": {}}
    for name, sp in (("train", tr), ("dev", dv), ("test", te)):
        nt = sp.n_tok()
        rec = {"docs": len(sp), "tokens_per_doc": _pct(nt), "empty_docs": int((sp.n_chars == 0).sum()),
               "frac_docs_le": {str(L): round(float((nt <= L).mean()), 4) for L in (256, 512, 1024, 2048, 4096)},
               "duplicate_ids": len(sp.ids) - len(set(sp.ids)), "duplicate_texts": len(sp.h) - len(set(sp.h)),
               "id_examples": sp.ids[:3]}
        if sp.labelled:
            Y = sp.Y(NL)
            rec["labels_per_doc"] = _pct(Y.sum(1))
            rec["docs_without_labels"] = int((Y.sum(1) == 0).sum())
            cnt = Y.sum(0)
            rec["label_counts"] = cnt.tolist()
            rec["labels_present"] = int((cnt > 0).sum())
            rec["max_label_id"] = int(np.where(cnt > 0)[0].max())
        rep["splits"][name] = rec
    rep["dev_clean_docs"] = int(len(dc))
    rep["dev_rows_dropped"] = {"total": len(dv) - len(dc), "ids_in_test": int(sum(i in set(te.ids) for i in dv.ids)),
                               "texts_in_test": int(sum(h in set(te.h) for h in dv.h))}
    rep["test_texts_in_train"] = int(sum(h in set(tr.h) for h in te.h))
    rep["dev_texts_in_train"] = int(sum(h in set(tr.h) for h in dv.h))
    Ytr, Ydc = tr.Y(NL), dv.Y(NL, dc)
    cnt = Ytr.sum(0)
    order = np.argsort(-cnt)
    tot = max(1, cnt.sum())
    rep["label_share_top10"] = round(float(cnt[order[:10]].sum() / tot), 4)
    rep["labels_with_lt10_train_docs"] = int(((cnt > 0) & (cnt < 10)).sum())
    rep["labels_absent_from_clean_dev"] = [int(l) for l in np.where((Ydc.sum(0) == 0) & (cnt > 0))[0]]
    co = Ytr.T.astype(np.int64) @ Ytr.astype(np.int64)
    np.fill_diagonal(co, 0)
    iu = np.triu_indices(NL, 1)
    top = np.argsort(-co[iu])[:20]
    rep["top_cooccurring_pairs"] = [{"a": int(iu[0][k]), "b": int(iu[1][k]), "docs": int(co[iu][k]),
                                     "p_b_given_a": round(float(co[iu][k] / max(1, cnt[iu[0][k]])), 3)} for k in top]
    prior = np.tile(cnt / max(1, len(Ytr)), (len(Ydc), 1))
    k = max(1, int(round(Ytr.sum(1).mean())))
    topk = np.zeros_like(Ydc)
    topk[:, order[:k]] = 1
    rep["baseline_topk_frequent"] = {"k": k, **{m: round(v, 4) for m, v in metrics(Ydc, topk).items()
                                                if m in ("macro_f1", "micro_f1", "subset_acc")}}
    rep["baseline_prior_cv_global"] = round(cross_fit(Ydc, prior)["global"]["macro_f1"], 4)
    if os.path.exists(EVAL) and len(dc):
        vals, err = official_parity([dv.ids[i] for i in dc], [dv.labels(i).tolist() for i in dc], topk)
        ours = metrics(Ydc, topk)
        nl_eval = int(vals.get("Labels", -1)) if vals.get("Labels", "").isdigit() else None
        rep["official_parity"] = {"official": vals, "stderr": err, "ours_macro": round(ours["macro_f1"], 6),
                                  "ours_micro": round(ours["micro_f1"], 6), "official_label_dim": nl_eval,
                                  "label_dim_ours_full": NL,
                                  "match": "Macro-F1" in vals and abs(float(vals["Macro-F1"]) - ours["macro_f1"]) < 1e-5
                                           and abs(float(vals["Micro-F1"]) - ours["micro_f1"]) < 1e-5}
    return rep, data


def tfidf_baselines(data, max_docs=20000, views=(("first512", "first", 512), ("last512", "last", 512),
                                                    ("middle512", "middle", 512), ("first2048", "first", 2048),
                                                    ("all8192", "headtail", 8192))):
    """Token-id 1-2-gram TF-IDF + one-vs-rest logistic regression per view (train subset -> clean dev,
    threshold cross-fitted on clean dev)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    tr, dv, dc, NL = data["train"], data["dev"], data["dev_clean"], data["n_labels"]
    Ydc = dv.Y(NL, dc)
    out = {}
    for name, ctx, L in views:
        t0 = time.time()
        idx = D.random_subset(len(tr), max_docs if L <= 2048 else max_docs // 2, seed=0)
        Ytr = tr.Y(NL, idx)
        keep = np.where(Ytr.sum(0) > 0)[0]
        txt = lambda sp, i: " ".join(map(str, D.views(sp.doc(i), ctx, L, head_tokens=L // 4)[0][0].tolist()))
        vec = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=3, max_features=300000,
                              sublinear_tf=True, dtype=np.float32)
        Xtr = vec.fit_transform([txt(tr, i) for i in idx])
        Xdv = vec.transform([txt(dv, i) for i in dc])
        clf = OneVsRestClassifier(LogisticRegression(C=4.0, class_weight="balanced", max_iter=1000, solver="liblinear"),
                                  n_jobs=2)
        clf.fit(Xtr, Ytr[:, keep])
        P = np.zeros((len(dc), NL))
        P[:, keep] = clf.predict_proba(Xdv)
        cv = cross_fit(Ydc, P)
        t, f = fit_global(Ydc, P)
        out[name] = {"f1_cv_global": round(cv["global"]["macro_f1"], 4), "f1_cv_perlabel": round(cv["per_label"]["macro_f1"], 4),
                     "micro_cv": round(cv["global"]["micro_f1"], 4), "f1_fit_global": round(f, 4), "thr": t,
                     "f1_fixed05": round(metrics(Ydc, apply("fixed", 0.5, P))["macro_f1"], 4),
                     "train_docs": len(idx), "minutes": round((time.time() - t0) / 60, 2)}
        print(f"TF-IDF+LR {name}: clean-dev macro-F1 cv {out[name]['f1_cv_global']:.4f}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--roots", nargs="+", default=["results"])
    ap.add_argument("--out", default="runs_p5/data_report.json")
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--max-docs", type=int, default=20000)
    a = ap.parse_args(argv)
    root = D.find_root(a.data)
    if root is None:
        sys.exit(f"no Phase_5_LSI data under {a.data}")
    cache = os.path.join(os.path.dirname(a.out) or ".", "_cache")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.baselines_only:
        res = tfidf_baselines(D.load_data(root, find_tokenizer(a.roots), cache), a.max_docs)
        json.dump(res, open(a.out, "w"), indent=1)
        return res
    rep, _ = build(root, a.roots, cache)
    json.dump(rep, open(a.out, "w"), indent=1)
    show = {k: v for k, v in rep.items() if k != "splits"}
    show["splits"] = {s: {k: v for k, v in r.items() if k != "label_counts"} for s, r in rep["splits"].items()}
    print(json.dumps(show, indent=1)[:9000])
    par = rep.get("official_parity")
    if par and not par["match"]:
        print("WARNING: our scorer and the official evaluator disagree (or the label dimension differs):", par)
    return rep


if __name__ == "__main__":
    main()
