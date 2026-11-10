"""Synthetic statute-identification data in the official layout (tests and CPU smoke runs).

    python -m phase5.synth --out data/p5synth --docs 160
Writes <out>/Phase_5_LSI/{Train/train.json, Train/dev.json, Test/test.json}: {"id", "text", "labels"}.
12 statute ids with a long-tailed frequency, 1-3 per document, two pairs that co-occur, the cue words of a
statute placed anywhere in the document (so long-document views matter). dev contains copies of a few
test documents and one duplicated row, like the Phase-4 data, to exercise the clean-dev logic.
"""
import argparse
import json
import os
import random

FILLER = ("the accused the complainant the police station the witness stated that on the night of the incident "
          "the prosecution case is that the investigating officer recovered the property the trial court the "
          "deceased the injured the vehicle the house the money the document").split()
CUES = ["murder with intent knife deceased", "culpable homicide not amounting", "hurt voluntarily caused injury",
        "grievous hurt fracture weapon", "theft movable property dishonestly", "robbery force snatched",
        "cheating deceived induced delivery", "criminal breach of trust entrusted", "forgery false document",
        "criminal intimidation threatened", "common intention several persons", "dowry demand cruelty husband"]
FREQ = [30, 22, 18, 14, 10, 8, 6, 5, 4, 3, 2, 1]
PAIRS = {0: 10, 7: 6}


def _doc(rng, n_words):
    labs = set(rng.choices(range(12), weights=FREQ, k=rng.choice([1, 1, 2, 3])))
    for a, b in PAIRS.items():
        if a in labs and rng.random() < 0.7:
            labs.add(b)
    words = [rng.choice(FILLER) for _ in range(n_words)]
    for l in labs:
        for _ in range(rng.randint(1, 2)):
            words.insert(rng.randint(0, len(words)), CUES[l])
    return " ".join(words), sorted(labs)


def write(out, docs=160, seed=0):
    rng = random.Random(seed)
    root = os.path.join(out, "Phase_5_LSI")
    os.makedirs(os.path.join(root, "Train"), exist_ok=True)
    os.makedirs(os.path.join(root, "Test"), exist_ok=True)
    split = {}
    for name, k in (("train", docs), ("dev", max(20, docs // 3)), ("test", max(12, docs // 5))):
        recs = []
        for i in range(k):
            text, labs = _doc(rng, rng.choice([rng.randint(5, 60), rng.randint(100, 500), rng.randint(500, 1500)]))
            recs.append({"id": f"{name}_{i}", "text": text, "labels": labs})
        split[name] = recs
    split["dev"] += [dict(split["test"][0]), dict(split["test"][1]), dict(split["dev"][0])]
    for r in split["test"]:
        del r["labels"]
    split["test"].append({"id": "test_empty", "text": ""})
    for name, part in (("train", "Train"), ("dev", "Train"), ("test", "Test")):
        with open(os.path.join(root, part, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(split[name], f)
    return root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/p5synth")
    ap.add_argument("--docs", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(write(a.out, a.docs, a.seed))


if __name__ == "__main__":
    main()
