"""Prior-case-retrieval data: queries + per-split candidate pools, format auto-detected, A-1 token cache.

Official layout (HF `Exploration-Lab/CS-781-Capstone`, folder Phase_7_PCR):
  Train/train.json + Train/train_candidates.json, Dev/dev.json + Dev/dev_candidates.json,
  Test/test.json + Test/test_candidates.json
Queries: {query_id: {"text": [sentences], "relevant_candidates": [ids]}} (test: no relevant_candidates).
Candidates: {cand_id: {"text": [sentences]}} or {cand_id: [sentences] / "text"}. Lists of records with
id / text / relevant_candidates keys are accepted too. Blank relevant ids are dropped (as the evaluator does).
A query's own id is never a valid answer for it (excluded at ranking time if it is in the pool).
"""
import hashlib
import json
import os

import numpy as np

from phase4.data import _find, _first, _json, encode_texts

HF_REPO = "Exploration-Lab/CS-781-Capstone"
FOLDER = "Phase_7_PCR"
SPLITS = {"train": ("Train", "train"), "dev": ("Dev", "dev"), "test": ("Test", "test")}
TEXT_KEYS = ("text", "sentences", "document", "doc", "content", "body")
REL_KEYS = ("relevant_candidates", "relevant", "labels", "gold", "positives", "citations")
ID_KEYS = ("id", "query_id", "qid", "case_id", "doc_id", "candidate_id", "cid", "name")


def download(out_dir, token=None):
    from huggingface_hub import snapshot_download
    snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=[f"{FOLDER}/*"], local_dir=out_dir, token=token)
    return os.path.join(out_dir, FOLDER)


def find_root(*cands):
    for c in cands:
        if not c:
            continue
        for sub in ("", FOLDER):
            p = os.path.join(c, sub)
            if _find(p, "Train", "train.json") and _find(p, "Test", "test_candidates.json"):
                return p
    return None


def _text(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(_text(s) for s in x)
    if isinstance(x, dict):
        return _text(_first(x, TEXT_KEYS))
    return str(x)


def _rel(x):
    if x is None:
        return None
    if isinstance(x, str):
        x = [x]
    return [str(c).strip() for c in x if isinstance(c, (str, int)) and str(c).strip()]


def normalize(raw, with_rel=True):
    """-> list of (id, text, relevant ids or None)."""
    if isinstance(raw, dict):
        for k in ("data", "queries", "candidates", "documents"):
            if k in raw and isinstance(raw[k], (list, dict)) and len(raw) == 1:
                raw = raw[k]
                break
    out = []
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                out.append((str(k), _text(_first(v, TEXT_KEYS)), _rel(_first(v, REL_KEYS)) if with_rel else None))
            else:
                out.append((str(k), _text(v), None))
    else:
        for i, v in enumerate(raw):
            k = _first(v, ID_KEYS) if isinstance(v, dict) else None
            if k is None:
                raise ValueError(f"record {i} has no id field")
            out.append((str(k), _text(_first(v, TEXT_KEYS)), _rel(_first(v, REL_KEYS)) if with_rel else None))
    return out


class Store:
    """ids + tokens of a set of documents: doc(i), index (id -> row)."""

    def __init__(self, ids, off, tok):
        self.ids, self.off, self.tok = ids, off, tok
        self.index = {d: i for i, d in enumerate(ids)}

    def __len__(self):
        return len(self.ids)

    def doc(self, i):
        return self.tok[self.off[i]:self.off[i + 1]]

    def n_tok(self):
        return np.diff(self.off)


def _store(ids, texts, tok_path, workers):
    arrs = encode_texts(texts, tok_path, workers)
    arrs = [a if len(a) else np.array([3], dtype=np.uint16) for a in arrs]
    off = np.zeros(len(arrs) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(a) for a in arrs])
    return Store(ids, off, np.concatenate(arrs) if arrs else np.zeros(0, np.uint16))


class Split:
    """queries (Store), pool (Store), rel (list of relevant-id lists, None for test), h_q / h_p text hashes."""

    def __init__(self, name, queries, pool, rel, h_q, h_p):
        self.name, self.queries, self.pool, self.rel, self.h_q, self.h_p = name, queries, pool, rel, h_q, h_p

    @property
    def labelled(self):
        return self.rel is not None

    def rel_rows(self, q):
        """Pool rows of query q's relevant candidates (ids missing from the pool are skipped)."""
        return [self.pool.index[c] for c in (self.rel[q] or []) if c in self.pool.index]


def _hash(t):
    return hashlib.sha1(" ".join(t.lower().split()).encode()).hexdigest()[:16]


def cache_key(root, tok_path):
    h = hashlib.sha1(b"p7")
    for part, stem in SPLITS.values():
        for name in (f"{stem}.json", f"{stem}_candidates.json"):
            p = _find(root, part, name)
            if p:
                h.update(name.encode() + str(os.path.getsize(p)).encode())
    h.update(open(tok_path, "rb").read(4096))
    return h.hexdigest()[:16]


def load_split(root, split, tok_path, cache_dir, workers=None):
    part, stem = SPLITS[split]
    qp, cp = _find(root, part, f"{stem}.json"), _find(root, part, f"{stem}_candidates.json")
    if qp is None or cp is None:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"p7_{cache_key(root, tok_path)}_{split}.npz")
    if os.path.exists(path) and os.path.exists(path + ".json"):
        z, m = np.load(path), json.load(open(path + ".json", encoding="utf-8"))
        return Split(split, Store(m["q_ids"], z["q_off"], z["q_tok"]), Store(m["p_ids"], z["p_off"], z["p_tok"]),
                     m["rel"], m["h_q"], m["h_p"])
    workers = workers if workers is not None else max(1, min(8, os.cpu_count() or 1))
    q = normalize(_json(qp), with_rel=split != "test")
    c = normalize(_json(cp), with_rel=False)
    if len({r[0] for r in c}) != len(c):
        raise ValueError(f"{cp}: duplicate candidate ids")
    rel = None if split == "test" else [r[2] or [] for r in q]
    qs, ps = _store([r[0] for r in q], [r[1] for r in q], tok_path, workers), _store([r[0] for r in c], [r[1] for r in c],
                                                                                     tok_path, workers)
    sp = Split(split, qs, ps, rel, [_hash(r[1]) for r in q], [_hash(r[1]) for r in c])
    tmp = path[:-4] + f".{os.getpid()}.tmp.npz"
    np.savez(tmp, q_off=qs.off, q_tok=qs.tok, p_off=ps.off, p_tok=ps.tok)
    with open(tmp + ".json", "w", encoding="utf-8") as f:
        json.dump({"q_ids": qs.ids, "p_ids": ps.ids, "rel": rel, "h_q": sp.h_q, "h_p": sp.h_p}, f)
    os.replace(tmp + ".json", path + ".json")
    os.replace(tmp, path)
    return sp


def load_data(root, tok_path, cache_dir, workers=None):
    out = {s: load_split(root, s, tok_path, cache_dir, workers) for s in SPLITS}
    if out["train"] is None or out["test"] is None:
        raise FileNotFoundError(f"Phase-7 train / test files missing under {root}")
    return out


def describe(root, chars=200):
    lines = []
    for part, stem in SPLITS.values():
        for name in (f"{stem}.json", f"{stem}_candidates.json"):
            p = _find(root, part, name)
            if p is None:
                lines.append(f"{part}/{name}: not found")
                continue
            raw = _json(p)
            lines.append(f"{part}/{name}: {type(raw).__name__} of {len(raw)}")
            first = raw[list(raw)[0]] if isinstance(raw, dict) and raw else (raw[0] if raw else None)
            key = list(raw)[0] if isinstance(raw, dict) and raw else None
            s = json.dumps(first, ensure_ascii=False)
            lines.append(f"   first key {key!r}: {s[:chars]}" + (f" ... [{len(s)} chars]" if len(s) > chars else ""))
    return "\n".join(lines)
