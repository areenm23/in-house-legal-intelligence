"""Synthetic court-judgment-prediction data in the official layout (tests and CPU smoke runs).

    python -m phase4.synth --out data/p4synth --docs 120
Writes <out>/Phase_4_CJPE/{Train/train.json, Train/dev.json, Test/test.json}: lists of
{"id", "text", "label"} (test without label). The outcome is mostly signalled near the end of a document
(the court's reasoning), weakly at the start (the kind of appeal), and not in the middle (filler), so the
first / last / hierarchical views differ. Lengths vary from a few words to several thousand; the test
file has one empty document.
"""
import argparse
import json
import os
import random

FILLER = ("the appellant the respondent learned counsel submitted that the high court the trial court section "
          "of the act the evidence on record the witness deposed the order dated the petition was filed the "
          "learned single judge the division bench the agreement the property the tribunal held that").split()
END_1 = ["the high court erred in law", "the impugned order cannot be sustained", "the findings are perverse",
         "we are unable to agree with the high court", "the conviction is not supported by evidence"]
END_0 = ["we find no infirmity in the impugned judgment", "the high court rightly held", "the concurrent findings "
         "of fact do not call for interference", "there is no merit in the contention", "the view taken is correct"]
START_1 = ["civil appeal against the order of the tribunal", "special leave petition by the assessee"]
START_0 = ["criminal appeal against conviction", "appeal by the accused against the judgment"]


def _doc(rng, label, n_words):
    w = [rng.choice(FILLER) for _ in range(max(0, n_words))]
    if rng.random() < 0.7:
        w = rng.choice(START_1 if label else START_0).split() + w
    k = max(1, min(4, n_words // 60))
    tail = []
    for _ in range(k):
        cue = END_1 if (label if rng.random() < 0.85 else 1 - label) else END_0
        tail += [rng.choice(FILLER) for _ in range(rng.randint(5, 20))] + rng.choice(cue).split()
    return " ".join(w + tail)


def write(out, docs=120, seed=0, pos_rate=0.4):
    rng = random.Random(seed)
    root = os.path.join(out, "Phase_4_CJPE")
    os.makedirs(os.path.join(root, "Train"), exist_ok=True)
    os.makedirs(os.path.join(root, "Test"), exist_ok=True)
    for split, k, part in (("train", docs, "Train"), ("dev", max(8, docs // 4), "Train"), ("test", max(8, docs // 4), "Test")):
        recs = []
        for i in range(k):
            y = int(rng.random() < pos_rate)
            n = rng.choice([rng.randint(3, 40), rng.randint(100, 600), rng.randint(600, 3000)])
            recs.append({"id": f"{1950 + i % 70}_{split}_{i}", "text": _doc(rng, y, n), "label": y})
        if split == "test":
            for r in recs:
                del r["label"]
            recs.append({"id": "test_empty", "text": ""})
        with open(os.path.join(root, part, f"{split}.json"), "w", encoding="utf-8") as f:
            json.dump(recs, f)
    return root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/p4synth")
    ap.add_argument("--docs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(write(a.out, a.docs, a.seed))


if __name__ == "__main__":
    main()
