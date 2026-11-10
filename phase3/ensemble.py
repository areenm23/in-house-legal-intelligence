"""Average the per-sentence log-probabilities of finished runs and take the argmax (CPU, seconds).

    python -m phase3.ensemble --runs G_lora A_lorma D_lora --data D --roots R --out runs_p3 [--weights 1 1 0.5]
           [--name ens1] [--bias]

Dev scores when every member has scores_dev.npz -> <out>/<name>/ensemble.json; test ->
<out>/<name>/submission.zip. --bias also fits per-class biases on the averaged dev log-probs and
writes submission_bias.zip (dev score after it is optimistic: fitted on the same dev set).
"""
import argparse
import json
import os

import numpy as np

from . import data as D
from .score import fit_bias, score, write_submission
from .train import find_tokenizer, load_data, load_scores


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float)
    ap.add_argument("--data", required=True)
    ap.add_argument("--roots", nargs="+", default=["results"])
    ap.add_argument("--out", default="runs_p3")
    ap.add_argument("--name", default=None)
    ap.add_argument("--bias", action="store_true")
    a = ap.parse_args(argv)
    w = a.weights or [1.0] * len(a.runs)
    name = a.name or "ens__" + "+".join(a.runs)
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    cfg0 = json.load(open(os.path.join(a.out, a.runs[0], "config.json")))
    data = load_data(D.find_root(a.data), cfg0.get("tokenizer") if os.path.exists(cfg0.get("tokenizer", ""))
                     else find_tokenizer(a.roots), cfg0["max_sent_tokens"], os.path.join(a.out, "_cache"))
    labels = data["labels"]
    n = len(labels)
    result = {"members": a.runs, "weights": w}
    bias = None
    for part, docs in (("dev", D.all_docs(data["dev"])), ("test", D.all_docs(data["test"]))):
        files = [os.path.join(a.out, r, f"scores_{part}.npz") for r in a.runs]
        if not all(os.path.exists(f) for f in files):
            print(f"{part}: missing scores in {[f for f in files if not os.path.exists(f)]}")
            continue
        members = [load_scores(f) for f in files]
        avg = {d.key: sum(wi * m[d.key] for wi, m in zip(w, members)) / sum(w) for d in docs}
        pred = {k: v.argmax(1) if len(v) else np.zeros(0, dtype=np.int64) for k, v in avg.items()}
        if part == "dev":
            result["dev"] = score(docs, pred, labels)
            print(f"{name}: dev macro-F1 {result['dev']['macro_f1']:.4f} "
                  + " ".join(f"{k} {v['macro_f1']:.4f}" for k, v in result["dev"]["domains"].items()))
            if a.bias:
                lp = np.concatenate([avg[d.key] for d in docs if len(d)])
                y = np.concatenate([d.y for d in docs if len(d)])
                bias, f_b = fit_bias(lp, y, n)
                result["bias"], result["dev_bias_f1_optimistic"] = bias.tolist(), f_b
        else:
            result["test_sentences"] = write_submission(out, docs, pred, labels)
            if bias is not None:
                bdir = os.path.join(out, "bias")
                os.makedirs(bdir, exist_ok=True)
                write_submission(bdir, docs, {k: (v + bias).argmax(1) if len(v) else np.zeros(0, dtype=np.int64)
                                              for k, v in avg.items()}, labels)
                os.replace(os.path.join(bdir, "submission.zip"), os.path.join(out, "submission_bias.zip"))
    json.dump(result, open(os.path.join(out, "ensemble.json"), "w"), indent=1)
    return result


if __name__ == "__main__":
    main()
