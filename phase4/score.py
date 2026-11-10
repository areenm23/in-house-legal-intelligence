"""Phase-4 scoring (same numbers as the official evaluate_phase4.py: sklearn macro-F1 over labels {0, 1}
with zero_division=0, precision / recall of class 1), decision-threshold tuning, error breakdowns, and
prediction.json writing + validation.
"""
import json
import os
import zipfile

import numpy as np


def prf(y_true, y_pred):
    """-> dict(macro_f1, precision, recall (class 1), accuracy, f1 per class, confusion [[tn, fp], [fn, tp]])."""
    yt, yp = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((2, 2), dtype=np.int64)
    if len(yt):
        np.add.at(cm, (yt, yp), 1)
    f1 = []
    for c in (0, 1):
        tp, fp, fn = cm[c, c], cm[1 - c, c], cm[c, 1 - c]
        f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    tp, fp, fn = cm[1, 1], cm[0, 1], cm[1, 0]
    return {"macro_f1": float(np.mean(f1)), "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "accuracy": float(np.trace(cm) / max(1, len(yt))), "f1_0": float(f1[0]), "f1_1": float(f1[1]),
            "confusion": cm.tolist(), "n": int(len(yt)), "pred_pos_rate": float(yp.mean()) if len(yp) else 0.0}


def probs(logits):
    """(n, 2) logits -> P(label 1)."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(1)


def fit_threshold(p1, y, grid=None):
    """Threshold on P(1) that maximises macro-F1 on (p1, y) -> (thr, f1). Ties -> closest to 0.5."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    best = (-1.0, 0.5)
    for t in grid:
        f = prf(y, (p1 >= t).astype(np.int64))["macro_f1"]
        if f > best[0] + 1e-12 or (abs(f - best[0]) <= 1e-12 and abs(t - 0.5) < abs(best[1] - 0.5)):
            best = (f, float(t))
    return best[1], best[0]


def breakdown(y, pred, p1, n_tok, ids, top=40):
    """Error analysis: F1 by document-length quartile, and the most confident false positives/negatives."""
    y, pred, p1, n_tok = map(np.asarray, (y, pred, p1, n_tok))
    out = {"by_length": []}
    qs = np.quantile(n_tok, [0, 0.25, 0.5, 0.75, 1.0]) if len(n_tok) else []
    for i in range(len(qs) - 1):
        m = (n_tok >= qs[i]) & ((n_tok < qs[i + 1]) if i < 3 else (n_tok <= qs[i + 1]))
        if m.sum():
            s = prf(y[m], pred[m])
            out["by_length"].append({"tokens": [int(qs[i]), int(qs[i + 1])], "n": int(m.sum()),
                                     "pos_rate": round(float(y[m].mean()), 3), "macro_f1": round(s["macro_f1"], 4),
                                     "accuracy": round(s["accuracy"], 4)})
    fp = np.where((pred == 1) & (y == 0))[0]
    fn = np.where((pred == 0) & (y == 1))[0]
    rec = lambda i: {"id": ids[i], "p1": round(float(p1[i]), 4), "tokens": int(n_tok[i])}
    out["false_pos"] = [rec(i) for i in fp[np.argsort(-p1[fp])][:top]]
    out["false_neg"] = [rec(i) for i in fn[np.argsort(p1[fn])][:top]]
    for name, idx in (("fp", fp), ("fn", fn), ("tp", np.where((pred == 1) & (y == 1))[0]),
                      ("tn", np.where((pred == 0) & (y == 0))[0])):
        out[f"median_tokens_{name}"] = float(np.median(n_tok[idx])) if len(idx) else None
    return out


def to_prediction_json(ids, pred):
    return [{"id": str(i), "pred": int(p)} for i, p in zip(ids, pred)]


def validate(pj, ids):
    """Same checks as the official evaluator: a non-empty list, one record per test id, string ids,
    integer (not bool) predictions in {0, 1}."""
    if not isinstance(pj, list) or not pj:
        raise ValueError("prediction must be a non-empty JSON list")
    seen = set()
    for n, r in enumerate(pj):
        if not isinstance(r, dict) or "id" not in r or "pred" not in r:
            raise ValueError(f"item {n} must contain id and pred")
        if not isinstance(r["id"], str) or isinstance(r["pred"], bool) or r["pred"] not in (0, 1) \
                or not isinstance(r["pred"], int):
            raise ValueError(f"invalid item {n}: {r}")
        if r["id"] in seen:
            raise ValueError(f"duplicate id {r['id']}")
        seen.add(r["id"])
    want = {str(i) for i in ids}
    if seen != want:
        raise ValueError(f"id mismatch: missing {len(want - seen)}, extra {len(seen - want)}")
    return True


def write_submission(out_dir, ids, pred, name="submission.zip"):
    """-> number of predictions. submission.zip keeps prediction.json next to it, submission_X.zip keeps
    prediction_X.json (inside every zip the file is called prediction.json)."""
    pj = to_prediction_json(ids, pred)
    validate(pj, ids)
    path = os.path.join(out_dir, name.replace("submission", "prediction").replace(".zip", ".json"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pj, f)
    with zipfile.ZipFile(os.path.join(out_dir, name), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, "prediction.json")
    return len(pj)


def write_truth(ids, y, path):
    """[{"id", "gold"}] (input format of evaluate_phase4.py --truth)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"id": str(i), "gold": int(g)} for i, g in zip(ids, y)], f)
