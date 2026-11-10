"""Synthetic L-NER data in the official layout (for tests and CPU smoke runs before the real data).

    python -m phase2.synth --out data/p2synth --docs 24
Writes <out>/Phase_2_LNER/{Train/fold{1,2}-{text,gold}.json, Train/labels.txt, Test/fold3-text.json}.
Documents mimic judgment headers and bodies: punctuation glued to entities ("Singh,", "(Appellant)"),
newlines and double spaces, multi-byte characters, dates and case numbers.
"""
import argparse
import json
import os
import random

LABELS = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT", "AUTH", "COURT", "STAT", "PREC", "DATE", "CASENO"]
NAMES = ["Ram Singh", "Sita Devi", "M/s. Bharat Steel Pvt. Ltd.", "Mohd. Iqbal", "Rajesh Kumar Sharma", "Anjali Rao",
         "State of Uttar Pradesh", "Union of India", "K. Venkataraman", "Smt. Kamla Bai", "Śrī Gopal"]
COURTS = ["Supreme Court of India", "High Court of Delhi", "High Court of Judicature at Allahabad",
          "Madras High Court"]
STATS = ["Section 302 of the Indian Penal Code", "Article 21 of the Constitution", "Code of Criminal Procedure",
         "Section 13(1)(e) of the Prevention of Corruption Act, 1988"]
PRECS = ["Maneka Gandhi v. Union of India", "State of Punjab v. Baldev Singh", "K.S. Puttaswamy v. Union of India"]
AUTHS = ["Income Tax Officer", "Collector of Customs", "Central Bureau of Investigation"]
FILLER = ("the learned counsel submitted that the impugned order was passed without jurisdiction and in "
          "violation of the principles of natural justice ; it is well settled that the evidence on record "
          "must be appreciated as a whole , and the court below erred in law").split()


class _Doc:
    def __init__(self):
        self.parts, self.ents, self.n = [], [], 0

    def add(self, s):
        self.parts.append(s)
        self.n += len(s)

    def ent(self, s, lab):
        self.ents.append([self.n, self.n + len(s), lab, s])
        self.add(s)

    def text(self):
        return "".join(self.parts)


def make_doc(rng):
    d = _Doc()
    d.add("REPORTABLE\n\nIN THE ")
    d.ent(rng.choice(COURTS), "COURT")
    d.add("\n\nCRIMINAL APPEAL NO. ")
    d.ent(f"{rng.randint(100, 2000)} OF {rng.randint(1990, 2020)}", "CASENO")
    d.add("\n\n")
    d.ent(rng.choice(NAMES), "APP")
    d.add(rng.choice([" ...Appellant\n\nVersus\n\n", ",  (Appellant)\nv.\n"]))
    d.ent(rng.choice(NAMES), "RESP")
    d.add(rng.choice([" ...Respondent\n\n", " (Respondent).\n"]))
    d.add("For the Appellant : ")
    d.ent(rng.choice(NAMES), "A.COUNSEL")
    d.add(", Advocate.\nFor the Respondent: Mr. ")
    d.ent(rng.choice(NAMES), "R.COUNSEL")
    d.add(".\n\nJUDGMENT\n\n")
    d.ent(rng.choice(["R.F. Nariman", "D.Y. Chandrachud", "S. A. Bobde"]), "JUDGE")
    d.add(", J.\n\n")
    for _ in range(rng.randint(8, 30)):
        d.add(" ".join(rng.choice(FILLER) for _ in range(rng.randint(10, 40))) + " ")
        r = rng.random()
        if r < 0.2:
            d.add("under ")
            d.ent(rng.choice(STATS), "STAT")
            d.add(". ")
        elif r < 0.35:
            d.add("in (")
            d.ent(rng.choice(PRECS), "PREC")
            d.add("), ")
        elif r < 0.5:
            d.add("on ")
            d.ent(rng.choice(["12.03.2011", "5th March, 2004", "1 January 1999"]), "DATE")
            d.add(", ")
        elif r < 0.6:
            d.add("PW-1 ")
            d.ent(rng.choice(NAMES), "WIT")
            d.add(" deposed that ")
        elif r < 0.7:
            d.add("the ")
            d.ent(rng.choice(AUTHS), "AUTH")
            d.add(" issued notice. ")
    return d.text(), d.ents


def write(out, docs=24, test_docs=8, seed=0):
    rng = random.Random(seed)
    root = os.path.join(out, "Phase_2_LNER")
    os.makedirs(os.path.join(root, "Train"), exist_ok=True)
    os.makedirs(os.path.join(root, "Test"), exist_ok=True)
    k = 1000
    for fold, n in ((1, docs // 2), (2, docs - docs // 2)):
        texts, gold = {}, {}
        for _ in range(n):
            t, e = make_doc(rng)
            texts[str(k)], gold[str(k)] = t, e
            k += 1
        json.dump(texts, open(os.path.join(root, "Train", f"fold{fold}-text.json"), "w", encoding="utf-8"))
        json.dump(gold, open(os.path.join(root, "Train", f"fold{fold}-gold.json"), "w", encoding="utf-8"))
    test = {"empty_doc": "", "blank_doc": "  \n "}
    for _ in range(test_docs):
        test[str(k)] = make_doc(rng)[0]
        k += 1
    json.dump(test, open(os.path.join(root, "Test", "fold3-text.json"), "w", encoding="utf-8"))
    with open(os.path.join(root, "Train", "labels.txt"), "w") as f:
        f.write("\n".join(LABELS) + "\n")
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/p2synth")
    ap.add_argument("--docs", type=int, default=24)
    a = ap.parse_args()
    print(write(a.out, a.docs))
