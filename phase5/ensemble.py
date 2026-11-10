"""Average per-statute probabilities of finished runs (train.py or hier.py folders), fit thresholds on clean
dev, write test zips (CPU, seconds; works on downloaded outputs).

    python -m phase5.ensemble --runs G_lorma__h_gru A_lorma__h_mean As_kd --out runs_p5 [--weights 1 1 1] [--name ens1]

Members' scores_dev.npz cover the same clean-dev documents (train.py and hier.py only score clean dev);
gold labels come from the first member's dev_pred.json, n_labels / max_pred_id from its result.json.
"""
import argparse
import json
import os

import numpy as np

from phase4.train import load_scores
from .data import freq_buckets
from .score import apply, sigmoid, write_submission
from .train import final_scores


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float)
    ap.add_argument("--out", default="runs_p5")
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)
    w = np.array(a.weights or [1.0] * len(a.runs), dtype=np.float64)
    name = a.name or "ens__" + "+".join(a.runs)
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    r0 = json.load(open(os.path.join(a.out, a.runs[0], "result.json")))
    L, max_id = r0["n_labels"], r0["max_pred_id"]
    gold = {r["id"]: r["gold"] for r in json.load(open(os.path.join(a.out, a.runs[0], "dev_pred.json")))}

    def avg(part):
        mem = [load_scores(os.path.join(a.out, r, f"scores_{part}.npz")) for r in a.runs]
        ids = mem[0]["ids"]
        p = np.zeros((len(ids), L))
        for wi, m in zip(w, mem):
            pos = {k: j for j, k in enumerate(m["ids"])}
            if set(pos) != set(ids):
                raise ValueError(f"{part}: members cover different documents")
            p += wi * sigmoid(m["logits"][[pos[k] for k in ids]])
        return ids, p / w.sum()

    ids, P = avg("dev")
    Y = np.zeros((len(ids), L), np.uint8)
    for r, i in enumerate(ids):
        Y[r, [x for x in gold[i] if x < L]] = 1
    tc = np.ones(L) if "per_label" not in r0 else np.array([x["train"] for x in r0["per_label"]])
    Pc = np.clip(P, 1e-7, 1 - 1e-7)
    fs, th, _ = final_scores(Y, np.log(Pc / (1 - Pc)), L, tc, freq_buckets(tc))
    result = {"exp": name, "members": a.runs, "weights": w.tolist(), "method": "ensemble", "ctx": "ensemble",
              "backbone": "+".join(sorted({json.load(open(os.path.join(a.out, r, "result.json"))).get("backbone") or "?"
                                           for r in a.runs})), "n_labels": L, "max_pred_id": max_id, **fs}
    tids, Pt = avg("test")
    result["test"] = {"global": write_submission(out, tids, apply("global", th["global"], Pt), max_id),
                      "per_label": write_submission(out, tids, apply("per_label", th["per_label"], Pt), max_id,
                                                    "submission_pl.zip")}
    json.dump(result, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(f"{name}: clean-dev macro-F1 cv global {fs['dev_cv']['global']['macro_f1']:.4f}, "
          f"cv per-label {fs['dev_cv']['per_label']['macro_f1']:.4f}")
    return result


if __name__ == "__main__":
    main()
