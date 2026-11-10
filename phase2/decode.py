"""Token scores -> character spans.

doc_scores(): run the model over overlapping windows of a document and average the per-token
log-probabilities where windows overlap (each token gets a weight that falls off towards the
window edges, where context is one-sided).
decode_doc(): BIO-constrained Viterbi (or the CRF's learned transitions, or plain argmax) ->
token tags -> (char_start, char_end, label) spans -> optional boundary trimming.

Boundary trimming (learned from training gold, `fit_trim_rules`): if gold spans of a label
almost never end with a given punctuation character but predictions do (or the reverse), the
trailing character is dropped. Only characters at a piece edge are touched.
"""
from collections import Counter

import numpy as np
import torch

from ft.heads import constrained_decode
from .data import spans_from_tags, windows


def _edge_weights(L, ramp):
    w = np.ones(L, dtype=np.float32)
    if ramp and L > 2 * ramp:
        r = np.linspace(0.1, 1.0, ramp, dtype=np.float32)
        w[:ramp] = r
        w[-ramp:] = r[::-1]
    return w


@torch.no_grad()
def doc_scores(model, doc, size, stride, device, batch=8, amp=True, ramp=None):
    """-> (n_tokens, n_tags) float32 averaged log-probabilities (or CRF emissions)."""
    n = len(doc)
    if n == 0:
        return np.zeros((0, model.head_tags), dtype=np.float32)
    wins = windows(n, size, stride)
    ramp = ramp if ramp is not None else max(0, (size - stride) // 2)
    acc = None
    wsum = np.zeros(n, dtype=np.float32)
    ids = torch.from_numpy(doc.ids.astype(np.int64))
    for i in range(0, len(wins), batch):
        chunk = wins[i:i + batch]
        lens = {e - s for s, e in chunk}
        groups = [[w for w in chunk if w[1] - w[0] == L] for L in sorted(lens)]
        for grp in groups:
            x = torch.stack([ids[s:e] for s, e in grp]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                sc = model.scores(x)
            sc = sc.float().cpu().numpy()
            if acc is None:
                acc = np.zeros((n, sc.shape[-1]), dtype=np.float32)
            for (s, e), row in zip(grp, sc):
                w = _edge_weights(e - s, ramp if len(wins) > 1 else 0)
                if s == 0:
                    w[:ramp] = 1.0
                if e == n:
                    w[len(w) - ramp:] = 1.0
                acc[s:e] += row * w[:, None]
                wsum[s:e] += w
    return acc / np.maximum(wsum, 1e-6)[:, None]


def decode_doc(scores, doc, tagset, mode="viterbi", crf=None):
    """scores (n, n_tags) -> set of (start, end, label)."""
    if len(scores) == 0:
        return set()
    if mode == "crf":
        tags = crf.decode(torch.from_numpy(scores))
    elif mode == "viterbi":
        allowed, start = tagset.constraints()
        lp = torch.log_softmax(torch.from_numpy(scores), -1)
        tags = constrained_decode(lp, allowed, start)
    else:
        tags = scores.argmax(-1)
    return spans_from_tags(np.asarray(tags), doc.starts, doc.ends, tagset)


PUNCT = set(".,;:)(]['\"-/")


def fit_trim_rules(texts, gold, min_count=20, max_rate=0.02):
    """Labels whose gold spans almost never end with char c -> {label: set(c)} to strip."""
    ends = Counter()
    tot = Counter()
    for did, ents in gold.items():
        t = texts[did]
        for s, e, l, *_ in ents:
            tot[l] += 1
            if e > s and t[e - 1] in PUNCT:
                ends[(l, t[e - 1])] += 1
    rules = {}
    for l, n in tot.items():
        if n < min_count:
            continue
        for c in PUNCT:
            if ends[(l, c)] / n <= max_rate:
                rules.setdefault(l, set()).add(c)
    return rules


def apply_trim(spans, text, rules):
    out = set()
    for s, e, l in spans:
        strip = rules.get(l, ())
        while e - s > 1 and (text[e - 1] in strip or text[e - 1].isspace()):
            e -= 1
        while e - s > 1 and text[s].isspace():
            s += 1
        out.add((s, e, l))
    return out
