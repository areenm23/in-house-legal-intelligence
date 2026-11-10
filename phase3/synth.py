"""Synthetic rhetorical-role data in the official layout (tests and CPU smoke runs before the real data).

    python -m phase3.synth --out data/p3synth --docs 12
Writes <out>/Phase_3_RR/{Train/{CL,IT}_{train,dev}.json, Train/label_vocab.json, Test/{CL,IT}_test.json}.
Documents follow a judgment's usual order (facts -> arguments -> precedents/statutes -> ratio ->
ruling) with role-typical wording, so a model can learn something; one empty sentence and one empty
test document are included to exercise the edge cases.
"""
import argparse
import json
import os
import random

LABELS = ["Fact", "Issue", "ArgumentPetitioner", "ArgumentRespondent", "Statute", "Dissent", "PrecedentReliedUpon",
          "PrecedentNotReliedUpon", "PrecedentOverruled", "RulingByLowerCourt", "RatioOfTheDecision",
          "RulingByPresentCourt", "None"]
WORDS = {
    "Fact": "the appellant company was incorporated in and entered into an agreement on dated with the respondent",
    "Issue": "the question that arises for consideration is whether the tribunal had jurisdiction under",
    "ArgumentPetitioner": "learned counsel for the appellant submitted that the impugned order is arbitrary",
    "ArgumentRespondent": "learned counsel for the respondent contended that the appeal is not maintainable",
    "Statute": "section of the companies act provides that every company shall maintain",
    "Dissent": "with respect i am unable to agree with the view taken by my learned brother",
    "PrecedentReliedUpon": "in the decision of this court it was held that the principle applies squarely",
    "PrecedentNotReliedUpon": "the judgment cited is distinguishable on facts and does not apply",
    "PrecedentOverruled": "the view taken in the earlier decision is not good law and stands overruled",
    "RulingByLowerCourt": "the high court by the impugned judgment dismissed the writ petition and held",
    "RatioOfTheDecision": "in our considered view the provision must be read harmoniously and therefore",
    "RulingByPresentCourt": "accordingly the appeal is allowed and the order is set aside no costs",
    "None": "heard the parties perused the record list the matter",
}
ORDER = ["None", "Fact", "RulingByLowerCourt", "Issue", "ArgumentPetitioner", "ArgumentRespondent", "Statute",
         "PrecedentReliedUpon", "PrecedentNotReliedUpon", "PrecedentOverruled", "RatioOfTheDecision", "Dissent",
         "RulingByPresentCourt"]


def _doc(rng, did, dom):
    sents, labs = [], []
    for lab in ORDER:
        if lab in ("Dissent", "PrecedentOverruled") and rng.random() < 0.6:
            continue
        for _ in range(rng.randint(1, 4)):
            w = WORDS[lab].split()
            s = " ".join(rng.sample(w, k=max(3, len(w) - 3)))
            if dom == "IT":
                s += " information technology act software"
            sents.append(s[0].upper() + s[1:] + ".")
            labs.append(lab)
    return {"id": did, "text": sents, "labels": labs}


def write(out, docs=12, seed=0):
    rng = random.Random(seed)
    root = os.path.join(out, "Phase_3_RR")
    os.makedirs(os.path.join(root, "Train"), exist_ok=True)
    os.makedirs(os.path.join(root, "Test"), exist_ok=True)
    json.dump(LABELS, open(os.path.join(root, "Train", "label_vocab.json"), "w"))
    for dom in ("CL", "IT"):
        for split, k in (("train", docs), ("dev", max(2, docs // 4)), ("test", max(2, docs // 4))):
            recs = [_doc(rng, f"{dom}_{split}_{i:03d}.txt", dom) for i in range(k)]
            if split == "train":
                recs[0]["text"][1] = "  "
            if split == "test":
                for r in recs:
                    del r["labels"]
                if dom == "IT":
                    recs.append({"id": f"{dom}_test_empty", "text": []})
            json.dump(recs, open(os.path.join(root, "Train" if split != "test" else "Test", f"{dom}_{split}.json"), "w"),
                      indent=1)
    return root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/p3synth")
    ap.add_argument("--docs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(write(a.out, a.docs, a.seed))


if __name__ == "__main__":
    main()
