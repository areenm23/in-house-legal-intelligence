"""Average P(label 1) of finished runs (train.py or hier.py folders), score on dev, write test zips (CPU, seconds).

    python -m phase4.ensemble --runs G_lorma A_lorma G_lorma__h_gru --out runs_p4 [--weights 1 1 2] [--name ens1]

Needs only each member's scores_{dev,test}.npz and the first member's dev_pred.json (dev gold labels), so
it also works on downloaded outputs. Writes <out>/<name>/result.json (+ ensemble.json), submission.zip
(P >= 0.5) and submission_thr.zip (threshold tuned on dev; its dev score is optimistic).
"""
import argparse
import json
import os

import numpy as np

from .score import fit_threshold, prf, probs, write_submission
from .train import load_scores


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float)
    ap.add_argument("--out", default="runs_p4")
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)
    w = np.array(a.weights or [1.0] * len(a.runs), dtype=np.float64)
    name = a.name or "ens__" + "+".join(a.runs)
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    gold = {r["id"]: r["gold"] for r in json.load(open(os.path.join(a.out, a.runs[0], "dev_pred.json")))}
    result = {"exp": name, "members": a.runs, "weights": w.tolist(), "backbone": "+".join(
        sorted({json.load(open(os.path.join(a.out, r, "result.json"))).get("backbone") or "?" for r in a.runs})),
        "method": "ensemble", "ctx": "ensemble"}
    thr = 0.5
    for part in ("dev", "test"):
        files = [os.path.join(a.out, r, f"scores_{part}.npz") for r in a.runs]
        if not all(os.path.exists(f) for f in files):
            print(f"{part}: missing {[f for f in files if not os.path.exists(f)]}")
            continue
        mem = [load_scores(f) for f in files]
        ids = mem[0]["ids"]
        p = np.zeros(len(ids))
        for wi, m in zip(w, mem):
            pos = {k: j for j, k in enumerate(m["ids"])}
            if set(pos) != set(ids):
                raise ValueError(f"{part}: members cover different documents")
            p += wi * probs(m["logits"])[[pos[k] for k in ids]]
        p /= w.sum()
        if part == "dev":
            y = np.array([gold[k] for k in ids])
            result["dev"] = prf(y, (p >= 0.5).astype(np.int64))
            thr, _ = fit_threshold(p, y)
            result["dev_thr"] = {"threshold": thr, **prf(y, (p >= thr).astype(np.int64))}
            print(f"{name}: dev macro-F1 {result['dev']['macro_f1']:.4f} (thr {thr:.2f} -> "
                  f"{result['dev_thr']['macro_f1']:.4f})")
        else:
            write_submission(out, ids, (p >= 0.5).astype(np.int64))
            write_submission(out, ids, (p >= thr).astype(np.int64), "submission_thr.zip")
            result["test_pos_rate"] = round(float((p >= 0.5).mean()), 4)
    for f in ("result.json", "ensemble.json"):
        json.dump(result, open(os.path.join(out, f), "w"), indent=1)
    return result


if __name__ == "__main__":
    main()
