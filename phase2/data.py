"""L-NER data: loading, character-exact tokenisation, BIO tags, windows, and the alignment oracle.

Tokenisation keeps exact character offsets. The A-1 BPE tokenises whitespace-separated words
(BPE never crosses a word; the last piece carries the end-of-word marker). Inside a word it
often merges punctuation into the word ("Singh," and "(Appellant)" are single tokens), which
makes a gold span that stops before the comma impossible to express. `split` cuts words into
pieces first and runs the same merges inside each piece (only the last piece gets <EOW>):
  none  : the word as in Phase 1
  edge  : leading / trailing punctuation characters become their own pieces ("(", "Appellant", ")", ",")
  punct : every change between a letter/digit and punctuation is a cut ("I", ".", "P", ...)
Every token gets [char_start, char_end) in the original text; a token that holds only <EOW>
has zero width (start = end = word end).

Gold format is detected: {id: [[start, end, label(, text)], ...]} or lists of dicts with
start/end/label keys. Offsets are checked against the entity text when one is given.
"""
import hashlib
import json
import os
import re
import unicodedata

import numpy as np

from phase1.data.a1_bpe import ByteBPE

DATA_FILES = ["Phase_2_LNER/Train/fold1-text.json", "Phase_2_LNER/Train/fold1-gold.json",
              "Phase_2_LNER/Train/fold2-text.json", "Phase_2_LNER/Train/fold2-gold.json",
              "Phase_2_LNER/Train/labels.txt", "Phase_2_LNER/Test/fold3-text.json"]
HF_REPO = "Exploration-Lab/CS-781-Capstone"


def download(out_dir, token=None):
    from huggingface_hub import hf_hub_download
    for f in DATA_FILES:
        hf_hub_download(HF_REPO, f, repo_type="dataset", local_dir=out_dir, token=token)
    return os.path.join(out_dir, "Phase_2_LNER")


def find_root(*cands):
    for c in cands:
        for sub in ("", "Phase_2_LNER"):
            p = os.path.join(c, sub)
            if os.path.exists(os.path.join(p, "Train", "labels.txt")):
                return p
    return None


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _norm_ent(e):
    if isinstance(e, dict):
        s = e.get("start", e.get("start_char", e.get("begin")))
        t = e.get("end", e.get("end_char"))
        lab = e.get("label", e.get("type", e.get("tag")))
        txt = e.get("text", e.get("entity_text"))
        return int(s), int(t), str(lab), txt
    s, t, lab = e[:3]
    return int(s), int(t), str(lab), (e[3] if len(e) > 3 else None)


def normalize_gold(g):
    if isinstance(g, list):
        out = {}
        for rec in g:
            did = str(rec.get("id", rec.get("doc_id")))
            out[did] = [_norm_ent(e) for e in rec.get("entities", rec.get("spans", []))]
        return out
    return {str(k): [_norm_ent(e) for e in (v or [])] for k, v in g.items()}


def load_split(root):
    """-> dict(labels, texts {id: str}, gold {id: [(s, e, label, text)]}, fold {id: 1|2}, test {id: str})."""
    labels = [x.strip() for x in open(os.path.join(root, "Train", "labels.txt"), encoding="utf-8") if x.strip()]
    texts, gold, fold = {}, {}, {}
    for k in (1, 2):
        t = {str(a): b for a, b in _json(os.path.join(root, "Train", f"fold{k}-text.json")).items()}
        g = normalize_gold(_json(os.path.join(root, "Train", f"fold{k}-gold.json")))
        for did in t:
            texts[did] = t[did]
            gold[did] = g.get(did, [])
            fold[did] = k
    test = {str(a): b for a, b in _json(os.path.join(root, "Test", "fold3-text.json")).items()}
    return dict(labels=labels, texts=texts, gold=gold, fold=fold, test=test)


def dev_split(ids, frac=0.15, salt="p2dev"):
    """Deterministic doc-level split -> (train ids, dev ids)."""
    ids = sorted(ids)
    key = lambda d: hashlib.sha1(f"{salt}:{d}".encode()).hexdigest()
    ranked = sorted(ids, key=key)
    n_dev = max(1, int(round(frac * len(ids))))
    dev = set(ranked[:n_dev])
    return [d for d in ids if d not in dev], sorted(dev)


def _is_wordch(c):
    return c.isalnum()


def word_pieces(word, split):
    """-> list of (start, end) char ranges inside `word` covering it exactly."""
    n = len(word)
    if split == "none" or n == 1:
        return [(0, n)]
    if split == "edge":
        i = 0
        while i < n and not _is_wordch(word[i]):
            i += 1
        j = n
        while j > i and not _is_wordch(word[j - 1]):
            j -= 1
        if i == n:
            return [(0, n)]
        return [(k, k + 1) for k in range(i)] + [(i, j)] + [(k, k + 1) for k in range(j, n)]
    if split == "punct":
        out, s = [], 0
        for k in range(1, n):
            if _is_wordch(word[k]) != _is_wordch(word[k - 1]):
                out.append((s, k))
                s = k
        out.append((s, n))
        return out
    raise ValueError(split)


class OffsetTokenizer:
    """A-1 BPE with character offsets and optional sub-word pre-splitting."""

    def __init__(self, a1tok, split="edge"):
        self.tok = a1tok
        self.enc = a1tok._enc
        self.split = split
        self._cache = {}
        self.vocab_size = a1tok.get_vocab_size()

    def _bpe(self, piece, eow):
        key = (piece, eow)
        ids = self._cache.get(key)
        if ids is not None:
            return ids
        e = self.enc
        seq = [f"B{b:02X}" for b in piece.encode("utf-8")] + ([ByteBPE.EOW] if eow else [])
        while len(seq) > 1:
            best = None
            for i, p in enumerate(zip(seq, seq[1:])):
                r = e._ranks.get(p)
                if r is not None and (best is None or r < best):
                    best = r
            if best is None:
                break
            a, b, m = e.bpe.merges[best]
            seq = e.bpe._apply_merge(seq, (a, b), m)
        ids = [e.stoi.get(s, e.UNK_ID) for s in seq]
        if len(self._cache) > 2_000_000:
            self._cache.clear()
        self._cache[key] = ids
        return ids

    def _byte_len(self, tid):
        b = self.enc._bytes.get(self.enc.itos[tid], b"")
        return len(b) - (1 if b.endswith(b" ") else 0)

    def encode(self, text):
        """-> (ids int32 [n], starts int32 [n], ends int32 [n])."""
        ids, starts, ends = [], [], []
        for m in re.finditer(r"\S+", text):
            w0, word = m.start(), m.group()
            pieces = word_pieces(word, self.split)
            for pi, (a, b) in enumerate(pieces):
                piece = word[a:b]
                pid = self._bpe(piece, eow=(pi == len(pieces) - 1))
                ascii_ = piece.isascii()
                cb = None if ascii_ else np.cumsum([0] + [len(c.encode("utf-8")) for c in piece])
                pos = 0
                for t in pid:
                    L = self._byte_len(t)
                    if L == 0:
                        s_c = e_c = len(piece)
                    elif ascii_:
                        s_c, e_c = pos, pos + L
                    else:
                        s_c = int(np.searchsorted(cb, pos, side="left"))
                        e_c = int(np.searchsorted(cb, pos + L, side="left"))
                    ids.append(t)
                    starts.append(w0 + a + s_c)
                    ends.append(w0 + a + e_c)
                    pos += L
        return (np.asarray(ids, dtype=np.int32), np.asarray(starts, dtype=np.int32), np.asarray(ends, dtype=np.int32))


class TagSet:
    """BIO over the official labels: 0 = O, 1 + 2i = B-label_i, 2 + 2i = I-label_i."""

    def __init__(self, labels):
        self.labels = list(labels)
        self.names = ["O"] + [f"{p}-{l}" for l in self.labels for p in ("B", "I")]
        self.n = len(self.names)

    def b(self, li):
        return 1 + 2 * li

    def i(self, li):
        return 2 + 2 * li

    def label_of(self, tag):
        return None if tag == 0 else self.labels[(tag - 1) // 2]

    def is_b(self, tag):
        return tag > 0 and (tag - 1) % 2 == 0

    def constraints(self):
        import torch
        n = self.n
        allowed = torch.ones(n, n, dtype=torch.bool)
        start = torch.ones(n, dtype=torch.bool)
        for j in range(1, n):
            if not self.is_b(j):
                start[j] = False
                for i in range(n):
                    allowed[i, j] = i in (j - 1, j)
        return allowed, start


def tag_tokens(starts, ends, ents, tagset):
    """Gold char spans -> BIO tag per token. Returns (tags int64 [n], stats dict)."""
    n = len(starts)
    tags = np.zeros(n, dtype=np.int64)
    owner = np.full(n, -1)
    lab_idx = {l: k for k, l in enumerate(tagset.labels)}
    st = {"conflicts": 0, "unrepresentable": 0, "unknown_label": 0, "empty": 0}
    width = ends > starts
    for k, (s, e, lab, _t) in enumerate(sorted(ents, key=lambda x: (x[0], -x[1]))):
        if lab not in lab_idx:
            st["unknown_label"] += 1
            continue
        hit = np.nonzero(width & (starts < e) & (ends > s))[0]
        if len(hit) == 0:
            st["empty"] += 1
            continue
        if (owner[hit] >= 0).any():
            st["conflicts"] += 1
            continue
        if starts[hit[0]] != s or ends[hit[-1]] != e:
            st["unrepresentable"] += 1
        owner[hit] = k
        li = lab_idx[lab]
        tags[hit[0]] = tagset.b(li)
        tags[hit[1:]] = tagset.i(li)
        inner = np.nonzero(~width & (starts > starts[hit[0]]) & (starts < ends[hit[-1]]))[0]
        tags[inner] = tagset.i(li)
    return tags, st


def spans_from_tags(tags, starts, ends, tagset):
    """BIO tags -> set of (char_start, char_end, label). Zero-width tokens never open a span."""
    out = []
    cur = None
    for t, s, e in zip(tags, starts, ends):
        t = int(t)
        if e <= s:
            continue
        if t == 0:
            if cur:
                out.append(cur)
            cur = None
            continue
        lab = tagset.label_of(t)
        if tagset.is_b(t) or cur is None or cur[0] != lab:
            if cur:
                out.append(cur)
            cur = [lab, int(s), int(e)]
        else:
            cur[2] = int(e)
    if cur:
        out.append(cur)
    return {(s, e, lab) for lab, s, e in out}


class Doc:
    __slots__ = ("id", "text", "ids", "starts", "ends", "tags", "gold")

    def __init__(self, did, text, otok, tagset=None, ents=None):
        self.id, self.text = did, text
        self.ids, self.starts, self.ends = otok.encode(text)
        self.gold = None if ents is None else {(s, e, lab) for s, e, lab, _ in ents}
        self.tags = None
        if ents is not None and tagset is not None:
            self.tags, _ = tag_tokens(self.starts, self.ends, ents, tagset)

    def __len__(self):
        return len(self.ids)


def windows(n, size, stride, offset=0):
    """(start, end) token windows covering 0..n; the last one is aligned to the end."""
    if n <= size:
        return [(0, n)]
    out = []
    s = offset % stride if offset else 0
    if s:
        out.append((0, size))
    while s + size < n:
        out.append((s, s + size))
        s += stride
    out.append((n - size, n))
    return sorted(set(out))


def offsets_check(texts, gold):
    """How many gold spans have entity_text == text[start:end] (and alternative conventions)."""
    n = ok = inclusive = stripped = 0
    for did, ents in gold.items():
        t = texts.get(did, "")
        for s, e, _lab, txt in ents:
            if txt is None:
                continue
            n += 1
            ok += t[s:e] == txt
            inclusive += t[s:e + 1] == txt
            stripped += t[s:e].strip() == str(txt).strip()
    return {"with_text": n, "exact": ok, "end_inclusive": inclusive, "match_after_strip": stripped}


def corpus_stats(texts, gold, labels, otok=None):
    """Counts, lengths, label frequencies, boundary characters, overlaps (for the report + decisions)."""
    from collections import Counter
    lab = Counter()
    ln = Counter()
    first_ch, last_ch, before, after = Counter(), Counter(), Counter(), Counter()
    overlaps = nested = 0
    rel_pos = {l: [] for l in labels}
    ws_edge = 0
    for did, ents in gold.items():
        t = texts[did]
        es = sorted(ents, key=lambda x: (x[0], -x[1]))
        for k, (s, e, l, _x) in enumerate(es):
            lab[l] += 1
            ln[l] += e - s
            span = t[s:e]
            if span and (span[0].isspace() or span[-1].isspace()):
                ws_edge += 1
            if span:
                first_ch[unicodedata.category(span[0])] += 1
                last_ch[span[-1] if not span[-1].isalnum() else "alnum"] += 1
            before[t[s - 1] if s > 0 and not t[s - 1].isalnum() else ("alnum" if s > 0 else "BOS")] += 1
            after[t[e] if e < len(t) and not t[e].isalnum() else ("alnum" if e < len(t) else "EOS")] += 1
            if l in rel_pos:
                rel_pos[l].append(s / max(1, len(t)))
            if k + 1 < len(es) and es[k + 1][0] < e:
                overlaps += 1
                nested += es[k + 1][1] <= e
    chars = [len(t) for t in texts.values()]
    out = {
        "docs": len(texts), "entities": sum(lab.values()),
        "chars_per_doc": {"mean": float(np.mean(chars)), "median": float(np.median(chars)), "max": int(max(chars))},
        "per_label": {l: {"count": lab[l], "mean_chars": ln[l] / max(1, lab[l]),
                          "median_rel_pos": float(np.median(rel_pos[l])) if rel_pos[l] else None} for l in labels},
        "unknown_labels": sorted(set(lab) - set(labels)),
        "overlapping_pairs": overlaps, "nested_pairs": nested, "spans_with_edge_whitespace": ws_edge,
        "last_char": dict(last_ch.most_common(15)), "char_after": dict(after.most_common(15)),
        "char_before": dict(before.most_common(15)),
    }
    if otok is not None:
        lens = [len(otok.encode(t)[0]) for t in texts.values()]
        out["tokens_per_doc"] = {"mean": float(np.mean(lens)), "median": float(np.median(lens)),
                                 "p90": float(np.percentile(lens, 90)), "max": int(max(lens)), "total": int(sum(lens))}
    return out
