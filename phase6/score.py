"""Phase-6 scoring with the official evaluate_phase6.py definitions: ROUGE-1 F (rouge_score, Porter stemmer),
ROUGE-L F from the evaluator's own LCS on the same stemmed tokens, BERTScore F1 (bert_score, lang="en",
i.e. roberta-large; it truncates long texts to the model's 512 tokens). Plus length / redundancy statistics,
and prediction.json (list format) writing + validation.
"""
import json
import os
import re
import zipfile

import numpy as np


def lcs(a, b):
    """Official evaluator's bit-parallel LCS length."""
    if len(a) > len(b):
        a, b = b, a
    masks = {}
    for i, t in enumerate(a):
        masks[t] = masks.get(t, 0) | (1 << i)
    s = 0
    for t in b:
        m = masks.get(t, 0)
        x = m | s
        y = (s << 1) | 1
        s = x & ~(x - y)
    return bin(s).count("1")


def rouge(preds, refs):
    """-> (mean ROUGE-1 F, mean ROUGE-L F) exactly as the official script (needs rouge_score)."""
    from rouge_score import rouge_scorer, tokenizers
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    tok = tokenizers.DefaultTokenizer(use_stemmer=True)
    r1 = rl = 0.0
    for pr, rf in zip(preds, refs):
        pr, rf = pr.strip(), rf.strip()
        r1 += scorer.score(rf, pr)["rouge1"].fmeasure
        pt, rt = tok.tokenize(pr), tok.tokenize(rf)
        if not pt and not rt:
            x = 1.0
        elif not pt or not rt:
            x = 0.0
        else:
            z = lcs(pt, rt)
            p, r = z / len(pt), z / len(rt)
            x = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        rl += x
    n = max(1, len(preds))
    return r1 / n, rl / n


def bertscore(preds, refs, device=None, batch_size=8):
    """Mean BERTScore F1 (None when bert_score is not installed)."""
    try:
        import bert_score
    except ImportError:
        return None
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, f = bert_score.score([p.strip() for p in preds], [r.strip() for r in refs], lang="en",
                               batch_size=batch_size, verbose=False, device=device)
    return float(f.mean().item())


def text_stats(preds, refs=None):
    """Length and redundancy: words, repeated 4-gram share, distinct-2; and length ratio to the reference."""
    def rep4(t):
        w = t.split()
        g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
        return 1 - len(set(g)) / len(g) if g else 0.0

    def dist2(t):
        w = t.split()
        g = list(zip(w, w[1:]))
        return len(set(g)) / len(g) if g else 0.0

    out = {"words_mean": float(np.mean([len(p.split()) for p in preds])) if preds else 0.0,
           "words_median": float(np.median([len(p.split()) for p in preds])) if preds else 0.0,
           "repeated_4gram_share": float(np.mean([rep4(p) for p in preds])) if preds else 0.0,
           "distinct_2": float(np.mean([dist2(p) for p in preds])) if preds else 0.0}
    if refs is not None:
        out["ref_words_mean"] = float(np.mean([len(r.split()) for r in refs]))
        out["length_ratio"] = out["words_mean"] / max(1e-9, out["ref_words_mean"])
    return out


def novel_bigrams(preds, docs):
    """Share of generated bigrams that do not occur in the source document (abstractiveness)."""
    vals = []
    for p, d in zip(preds, docs):
        pw, dw = p.lower().split(), set(zip(d.lower().split(), d.lower().split()[1:]))
        g = list(zip(pw, pw[1:]))
        if g:
            vals.append(sum(x not in dw for x in g) / len(g))
    return float(np.mean(vals)) if vals else 0.0


def evaluate(preds, refs, with_bert=True, device=None):
    r1, rl = rouge(preds, refs)
    out = {"rouge1": r1, "rougeL": rl, **text_stats(preds, refs)}
    if with_bert:
        out["bertscore_f1"] = bertscore(preds, refs, device)
    return out


def validate(pj, ids):
    """Official rules for the list format: id + non-empty string summary, no duplicates, every test id."""
    if not isinstance(pj, list) or not pj:
        raise ValueError("prediction must be a non-empty JSON list")
    seen = set()
    for n, r in enumerate(pj):
        if not isinstance(r, dict) or "id" not in r or "summary" not in r:
            raise ValueError(f"item {n} must contain id and summary")
        if not isinstance(r["id"], str) or not isinstance(r["summary"], str) or not r["summary"].strip():
            raise ValueError(f"invalid summary for id {r.get('id')!r}")
        if r["id"] in seen:
            raise ValueError(f"duplicate id {r['id']}")
        seen.add(r["id"])
    want = {str(i) for i in ids}
    if seen != want:
        raise ValueError(f"id mismatch: missing {len(want - seen)}, extra {len(seen - want)}")
    return True


def write_submission(out_dir, ids, summaries, name="submission.zip"):
    pj = [{"id": str(i), "summary": s} for i, s in zip(ids, summaries)]
    validate(pj, ids)
    path = os.path.join(out_dir, name.replace("submission", "prediction").replace(".zip", ".json"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pj, f, ensure_ascii=False)
    with zipfile.ZipFile(os.path.join(out_dir, name), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, "prediction.json")
    return len(pj)


_WS = re.compile(r"\s+")


def clean(text):
    return _WS.sub(" ", text or "").strip()
