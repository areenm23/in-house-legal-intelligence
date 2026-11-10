"""Synthetic prior-case-retrieval data in the official layout (tests and CPU smoke runs).

    python -m phase7.synth --out data/p7synth
Each split has its own pool; every case belongs to a topic (shared rare words), and a query's relevant
candidates are pool cases of the same topic, so both BM25 and a trained encoder can find them. One query id also
appears in its pool (it must never be returned for itself) and one relevant id is blank.
"""
import argparse
import json
import os
import random

COMMON = "the court held that the appellant respondent section act order judgment petition evidence facts".split()


def write(out, seed=0, topics=10, per_topic=6, queries=12):
    rng = random.Random(seed)
    root = os.path.join(out, "Phase_7_PCR")
    vocab = [[f"topic{t}word{k}" for k in range(8)] for t in range(topics * 3)]

    def case(t, n=40):
        sents = []
        for _ in range(rng.randint(3, 8)):
            w = [rng.choice(COMMON) for _ in range(rng.randint(5, 12))] + rng.sample(vocab[t], 3)
            rng.shuffle(w)
            sents.append(" ".join(w) + ".")
        return sents

    for s, (part, off) in {"train": ("Train", 0), "dev": ("Dev", topics), "test": ("Test", 2 * topics)}.items():
        os.makedirs(os.path.join(root, part), exist_ok=True)
        pool, by_topic = {}, {}
        for t in range(off, off + topics):
            for k in range(per_topic):
                cid = f"{s}{t:03d}{k:03d}"
                pool[cid] = {"text": case(t)}
                by_topic.setdefault(t, []).append(cid)
        qs = {}
        for i in range(queries):
            t = off + i % topics
            rel = rng.sample(by_topic[t], rng.randint(1, 4))
            qs[f"q{s}{i:04d}"] = {"text": case(t), "relevant_candidates": rel + ([""] if i == 0 else [])}
        first_q = next(iter(qs))
        pool[first_q] = {"text": qs[first_q]["text"]}
        if s == "test":
            for v in qs.values():
                del v["relevant_candidates"]
        json.dump(qs, open(os.path.join(root, part, f"{s}.json"), "w"))
        json.dump(pool, open(os.path.join(root, part, f"{s}_candidates.json"), "w"))
    return root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/p7synth")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(write(a.out, a.seed))


if __name__ == "__main__":
    main()
