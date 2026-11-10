"""Strict span scoring (same arithmetic as course evaluate_phase2.py) and error analysis.

score(truth, pred, labels) -> {"macro_f1", "macro_p", "macro_r", "per_label": {l: {p, r, f1, tp, fp, fn}}}
truth / pred: {doc_id: iterable of (start, end, label)}.
errors(truth, pred) classifies every false positive / negative: exact-span label confusion,
boundary error (same label, overlapping span; start / end / both off), or spurious / missed.
"""
from collections import Counter


def _sets(d):
    return {k: {(int(a), int(b), str(l)) for a, b, l, *_ in v} for k, v in d.items()}


def score(truth, pred, labels):
    truth, pred = _sets(truth), _sets(pred)
    stats = {l: [0, 0, 0] for l in labels}
    for did in truth:
        g, p = truth[did], pred.get(did, set())
        for l in labels:
            gs = {x for x in g if x[2] == l}
            ps = {x for x in p if x[2] == l}
            stats[l][0] += len(gs & ps)
            stats[l][1] += len(ps - gs)
            stats[l][2] += len(gs - ps)
    per = {}
    for l in labels:
        tp, fp, fn = stats[l]
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[l] = {"p": pr, "r": rc, "f1": f, "tp": tp, "fp": fp, "fn": fn}
    n = len(labels)
    return {"macro_f1": sum(v["f1"] for v in per.values()) / n, "macro_p": sum(v["p"] for v in per.values()) / n,
            "macro_r": sum(v["r"] for v in per.values()) / n, "per_label": per}


def errors(truth, pred):
    truth, pred = _sets(truth), _sets(pred)
    kinds = Counter()
    confusion = Counter()
    for did in truth:
        g, p = truth[did], pred.get(did, set())
        fp, fn = p - g, g - p
        g_span = {(a, b): l for a, b, l in g}
        used = set()
        for a, b, l in sorted(fp):
            if (a, b) in g_span:
                kinds["label_confusion"] += 1
                confusion[(g_span[(a, b)], l)] += 1
                used.add((a, b, g_span[(a, b)]))
                continue
            ov = [x for x in fn if x[0] < b and a < x[1] and x[2] == l]
            if ov:
                x = ov[0]
                kinds["boundary_" + ("both" if x[0] != a and x[1] != b else "start" if x[0] != a else "end")] += 1
                used.add(x)
            elif any(x[0] < b and a < x[1] for x in g):
                kinds["overlap_wrong_label"] += 1
            else:
                kinds["spurious"] += 1
        kinds["missed"] += len([x for x in fn if x not in used])
    return {"kinds": dict(kinds), "confusion": {f"{t}->{p}": c for (t, p), c in confusion.most_common(30)}}


def to_prediction_json(pred, texts=None):
    """{id: set of (s, e, l)} -> official format, sorted, optional entity text as a 4th field."""
    out = {}
    for did, spans in pred.items():
        rows = []
        for s, e, l in sorted(spans):
            row = [int(s), int(e), str(l)]
            if texts is not None:
                row.append(texts[did][s:e])
            rows.append(row)
        out[did] = rows
    return out


def validate(pred_json, ids, labels):
    """Same checks as the official evaluator (ids, types, 0 <= start < end, labels, duplicates)."""
    if set(pred_json) != set(ids):
        raise ValueError(f"id mismatch: missing {len(set(ids) - set(pred_json))}, extra {len(set(pred_json) - set(ids))}")
    allowed = set(labels)
    for did, rows in pred_json.items():
        if not isinstance(rows, list):
            raise ValueError(f"{did}: not a list")
        seen = set()
        for r in rows:
            if not isinstance(r, list) or len(r) not in (3, 4):
                raise ValueError(f"{did}: bad row {r!r}")
            a, b, l = r[:3]
            if isinstance(a, bool) or not isinstance(a, int) or isinstance(b, bool) or not isinstance(b, int):
                raise ValueError(f"{did}: non-int offsets {r!r}")
            if a < 0 or b <= a:
                raise ValueError(f"{did}: bad span {r!r}")
            if l not in allowed:
                raise ValueError(f"{did}: bad label {l!r}")
            if (a, b, l) in seen:
                raise ValueError(f"{did}: duplicate {r!r}")
            seen.add((a, b, l))
    return True
