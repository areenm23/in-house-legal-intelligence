"""Phase-3 scoring (same numbers as the official evaluate_phase3.py: sklearn macro P/R/F1 over all 13
labels, zero_division=0, sentences of both domains pooled), plus per-domain / per-label scores, the
confusion matrix, prediction.json writing and validation.
"""
import json
import os
import zipfile

import numpy as np


def prf(y_true, y_pred, n):
    """Integer label arrays -> dict(macro_f1, macro_p, macro_r, per-label arrays, confusion)."""
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((n, n), dtype=np.int64)
    if len(y_true):
        np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    div = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)
    p, r = div(tp, tp + fp), div(tp, tp + fn)
    f = div(2 * tp, 2 * tp + fp + fn)
    return {"macro_f1": float(f.mean()), "macro_p": float(p.mean()), "macro_r": float(r.mean()),
            "accuracy": float(tp.sum() / max(1, len(y_true))), "p": p, "r": r, "f1": f,
            "support": cm.sum(1), "confusion": cm}


def score(docs, preds, labels):
    """docs: labelled RRDocs; preds: {doc.key: int array}. -> JSON-able dict (overall, per domain, per label,
    confusion matrix, most frequent confusions)."""
    n = len(labels)
    out = {}
    groups = {"all": docs}
    for d in docs:
        groups.setdefault(d.domain, []).append(d)
    for g, ds in groups.items():
        yt = np.concatenate([d.y for d in ds if len(d)] or [np.zeros(0, int)])
        yp = np.concatenate([np.asarray(preds[d.key]) for d in ds if len(d)] or [np.zeros(0, int)])
        s = prf(yt, yp, n)
        rec = {k: round(s[k], 6) for k in ("macro_f1", "macro_p", "macro_r", "accuracy")}
        rec["sentences"] = int(len(yt))
        rec["per_label"] = {labels[i]: {"p": round(float(s["p"][i]), 4), "r": round(float(s["r"][i]), 4),
                                        "f1": round(float(s["f1"][i]), 4), "support": int(s["support"][i])}
                            for i in range(n)}
        if g == "all":
            cm = s["confusion"]
            rec["confusion"] = cm.tolist()
            off = [(int(cm[i, j]), labels[i], labels[j]) for i in range(n) for j in range(n) if i != j and cm[i, j]]
            rec["top_confusions"] = [{"gold": a, "pred": b, "count": c} for c, a, b in sorted(off, reverse=True)[:15]]
        out[g] = rec
    res = dict(out.pop("all"))
    res["domains"] = out
    return res


def to_prediction_json(docs, preds, labels):
    return {d.id: [labels[int(i)] for i in preds[d.key]] for d in docs}


def validate(pj, test_docs, labels):
    """Same checks as the official evaluator: every test doc present once, one label per sentence, known labels."""
    allowed = set(labels)
    norm = lambda s: s[:-4] if s.endswith(".txt") else s
    want = {}
    for d in test_docs:
        k = norm(d.id)
        if k in want:
            raise ValueError(f"duplicate test document id {d.id} (after .txt normalisation)")
        want[k] = len(d)
    got = {norm(k): v for k, v in pj.items()}
    if len(got) != len(pj):
        raise ValueError("duplicate ids in prediction.json after .txt normalisation")
    if set(got) != set(want):
        raise ValueError(f"id mismatch: missing {sorted(set(want) - set(got))[:5]} extra {sorted(set(got) - set(want))[:5]}")
    for k, v in got.items():
        if not isinstance(v, list) or len(v) != want[k]:
            raise ValueError(f"{k}: {len(v)} predictions for {want[k]} sentences")
        bad = set(v) - allowed
        if bad:
            raise ValueError(f"{k}: unknown labels {sorted(bad)}")
    return True


def write_submission(out_dir, test_docs, preds, labels):
    pj = to_prediction_json(test_docs, preds, labels)
    validate(pj, test_docs, labels)
    path = os.path.join(out_dir, "prediction.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pj, f, ensure_ascii=False)
    with zipfile.ZipFile(os.path.join(out_dir, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, "prediction.json")
    return sum(len(v) for v in pj.values())


def write_truth(docs, path):
    """{id: [labels]} for one domain (input format of evaluate_phase3.py --truth-cl / --truth-it)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({d.id: list(d.labels) for d in docs}, f, ensure_ascii=False)


def fit_bias(logp, y, n, rounds=3, grid=np.linspace(-2, 2, 21)):
    """Per-class additive bias on log-probs that maximises macro-F1 on (logp, y) by coordinate ascent.
    logp (N, n) float, y (N,) int -> (n,) bias. Fitted on dev, so the dev score after it is optimistic."""
    b = np.zeros(n)
    best = prf(y, (logp + b).argmax(1), n)["macro_f1"]
    for _ in range(rounds):
        changed = False
        for c in range(n):
            for v in grid:
                bb = b.copy()
                bb[c] = v
                f = prf(y, (logp + bb).argmax(1), n)["macro_f1"]
                if f > best + 1e-9:
                    best, b, changed = f, bb, True
        if not changed:
            break
    return b, best
