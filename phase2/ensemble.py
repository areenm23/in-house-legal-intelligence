"""Average the per-token log-probabilities of finished runs and decode once (CPU, minutes).

    python -m phase2.ensemble --runs A_full G_full B_full --data D --roots R --out runs_p2 [--weights 1 1 0.5] [--name ens1]

Runs must share the tokenisation (`split`); BIO-constrained Viterbi on the averaged scores,
then the same boundary trimming as the members. Dev scores (if every member has scores_dev.npz)
-> <out>/<name>/result.json; test -> <out>/<name>/submission.zip.
"""
import argparse
import json
import os

import numpy as np
import torch

from . import data as D
from .decode import apply_trim, decode_doc, fit_trim_rules
from .score import errors, score
from .train import find_tokenizer, load_docs, write_submission


def _load(path):
    z = np.load(path)
    return {k: torch.log_softmax(torch.from_numpy(z[k].astype(np.float32)), -1).numpy() for k in z.files}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float)
    ap.add_argument("--data", required=True)
    ap.add_argument("--roots", nargs="+", default=["results"])
    ap.add_argument("--out", default="runs_p2")
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)
    w = a.weights or [1.0] * len(a.runs)
    cfgs = [json.load(open(os.path.join(a.out, r, "config.json"))) for r in a.runs]
    splits = {c["split"] for c in cfgs}
    assert len(splits) == 1, f"members use different tokenisation: {splits}"
    name = a.name or "ens__" + "+".join(a.runs)
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    data = D.load_split(D.find_root(a.data))
    tagset = D.TagSet(data["labels"])
    lab, test = load_docs(data, cfgs[0].get("tokenizer") or find_tokenizer(a.roots), splits.pop(),
                          os.path.join(a.out, "_cache"), tagset)
    result = {"members": a.runs, "weights": w}
    for part, docs in (("dev", lab), ("test", test)):
        files = [os.path.join(a.out, r, f"scores_{part}.npz") for r in a.runs]
        if not all(os.path.exists(f) for f in files):
            continue
        members = [_load(f) for f in files]
        ids = set(members[0])
        pred = {}
        tr_ids = [d for d in lab if d not in ids] if part == "dev" else list(lab)
        rules = fit_trim_rules(data["texts"], {d: data["gold"][d] for d in tr_ids})
        for did in ids:
            avg = sum(wi * m[did] for wi, m in zip(w, members)) / sum(w)
            pred[did] = apply_trim(decode_doc(avg, docs[did], tagset, "viterbi"), docs[did].text, rules)
        if part == "dev":
            truth = {d: lab[d].gold for d in ids}
            result["dev"] = score(truth, pred, data["labels"])
            result["dev_errors"] = errors(truth, pred)
            print(f"{name}: dev strict macro-F1 {result['dev']['macro_f1']:.4f}")
        else:
            result["test_entities"] = write_submission(out, pred, list(test.values()), data["labels"])
    json.dump(result, open(os.path.join(out, "ensemble.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
