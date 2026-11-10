"""Legal summarisation data: loading (format auto-detected), sentence splitting, A-1 token cache, a fixed
held-out dev split, and the input-selection strategies that fit a long judgment into a model's budget.

Official layout (HF `Exploration-Lab/CS-781-Capstone`, folder Phase_6_Summ):
  Train/train.json  [{"id", "document", "summary"}],  Test/test.json  [{"id", "document"}]
Also accepted: {id: {...}} mappings, a "data" wrapper, JSON lines, text / judgment / source for the
document, reference / headnote / target for the summary, and a list of sentences instead of a string.

No dev split is given: `dev_frac` of train (fixed seed) is held out as dev.

Input selection (`select_input`), all returning at most `budget` tokens:
  head      the first `budget` tokens
  headtail  first `head_tokens` + last (budget - head_tokens) tokens
  central   lead (first `lead_frac` of the budget) + the most central sentences (cosine of a sentence's
            TF-IDF vector with the document's), kept in document order
  oracle    sentences with the highest unigram overlap with the reference summary (training / dev
            analysis only: "how much does selection lose?"; documents without a summary use central)
Chunk-level ("dancer") examples (`dancer_pairs`): the document is cut into consecutive sentence chunks of
`budget` tokens, every summary sentence is assigned to the chunk it overlaps most, and each chunk is
trained to produce its sentences; at test time chunk summaries are concatenated.
"""
import hashlib
import json
import os
import re

import numpy as np

from phase4.data import _find, _first, _json, encode_texts

HF_REPO = "Exploration-Lab/CS-781-Capstone"
FOLDER = "Phase_6_Summ"
DOC_KEYS = ("document", "text", "judgment", "judgement", "source", "doc", "input", "case_text")
SUM_KEYS = ("summary", "reference", "headnote", "target", "abstract", "output", "gold")
ID_KEYS = ("id", "doc_id", "docid", "case_id", "uid", "name", "file", "filename")
_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'])|\n+")


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
            if _find(p, "Train", "train.json") and _find(p, "Test", "test.json"):
                return p
    return None


def _str(x):
    if x is None:
        return None
    if isinstance(x, list):
        return "\n".join(_str(s) or "" for s in x)
    return str(x)


def normalize(raw, summaries=True):
    """-> list of (id or None, document, summary or None)."""
    if isinstance(raw, dict):
        for k in ("data", "documents", "docs", "records", "examples"):
            if k in raw and isinstance(raw[k], (list, dict)):
                raw = raw[k]
                break
    recs = []
    items = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for key, v in items:
        if isinstance(v, dict):
            i = _first(v, ID_KEYS)
            i = i if i is not None else (key if isinstance(raw, dict) else None)
            recs.append((None if i is None else str(i), _str(_first(v, DOC_KEYS)) or "",
                         _str(_first(v, SUM_KEYS)) if summaries else None))
        else:
            recs.append((str(key) if isinstance(raw, dict) else None, _str(v) or "", None))
    return recs


def split_sentences(text):
    return [s.strip() for s in _SPLIT.split(text or "") if s and s.strip()]


class Docs:
    """ids, sents (list of lists of uint16 token arrays, one per sentence), summ (list of uint16 arrays or
    None), ref (reference summary strings or None), n_words (document words)."""

    def __init__(self, ids, sents, summ, ref, n_words):
        self.ids, self.sents, self.summ, self.ref, self.n_words = ids, sents, summ, ref, n_words

    def __len__(self):
        return len(self.ids)

    def doc_tokens(self, i):
        s = self.sents[i]
        return np.concatenate(s) if s else np.zeros(0, np.uint16)

    def take(self, idx):
        idx = list(idx)
        return Docs([self.ids[i] for i in idx], [self.sents[i] for i in idx],
                    None if self.summ is None else [self.summ[i] for i in idx],
                    None if self.ref is None else [self.ref[i] for i in idx], [self.n_words[i] for i in idx])


def _build(recs, tok_path, workers):
    flat, nsent = [], []
    for _i, doc, _s in recs:
        ss = split_sentences(doc) or [""]
        flat += ss
        nsent.append(len(ss))
    arrs = encode_texts(flat, tok_path, workers)
    sents, k = [], 0
    for n in nsent:
        sents.append([a for a in arrs[k:k + n] if len(a)])
        k += n
    summ = None
    if recs and recs[0][2] is not None:
        summ = encode_texts([r[2] or "" for r in recs], tok_path, workers)
    return sents, summ


def cache_key(root, tok_path, dev_frac, seed):
    h = hashlib.sha1(f"p6{dev_frac}{seed}".encode())
    for part, name in (("Train", "train.json"), ("Test", "test.json")):
        p = _find(root, part, name)
        h.update(os.path.basename(p).encode() + str(os.path.getsize(p)).encode())
    h.update(open(tok_path, "rb").read(4096))
    return h.hexdigest()[:16]


def _pack(d):
    lens = [len(s) for doc in d.sents for s in doc]
    out = {"tok": np.concatenate([s for doc in d.sents for s in doc]) if lens else np.zeros(0, np.uint16),
           "sent_len": np.array(lens, dtype=np.int32), "doc_nsent": np.array([len(x) for x in d.sents], dtype=np.int32),
           "n_words": np.array(d.n_words, dtype=np.int64)}
    if d.summ is not None:
        out["sum_tok"] = np.concatenate(d.summ) if d.summ else np.zeros(0, np.uint16)
        out["sum_len"] = np.array([len(x) for x in d.summ], dtype=np.int32)
    return out


def _unpack(z, meta):
    tok, sl, dn = z["tok"], z["sent_len"], z["doc_nsent"]
    so = np.concatenate([[0], np.cumsum(sl)])
    sents, k = [], 0
    for n in dn:
        sents.append([tok[so[j]:so[j + 1]] for j in range(k, k + n)])
        k += n
    summ = None
    if "sum_tok" in z.files:
        st = z["sum_tok"]
        o = np.concatenate([[0], np.cumsum(z["sum_len"])])
        summ = [st[o[j]:o[j + 1]] for j in range(len(dn))]
    return Docs(meta["ids"], sents, summ, meta.get("ref"), z["n_words"].tolist())


def load_data(root, tok_path, cache_dir, dev_frac=0.05, seed=0, min_dev=150, workers=None):
    """-> {"train", "dev", "test": Docs, "files"}; dev = fixed held-out part of train.json."""
    os.makedirs(cache_dir, exist_ok=True)
    key = cache_key(root, tok_path, dev_frac, seed)
    workers = workers if workers is not None else max(1, min(8, os.cpu_count() or 1))
    out = {"files": {}}
    paths = {s: os.path.join(cache_dir, f"p6_{key}_{s}.npz") for s in ("train", "dev", "test")}
    if all(os.path.exists(p) and os.path.exists(p + ".json") for p in paths.values()):
        for s, p in paths.items():
            out[s] = _unpack(np.load(p), json.load(open(p + ".json", encoding="utf-8")))
        return out
    for split, part, name in (("train", "Train", "train.json"), ("test", "Test", "test.json")):
        path = _find(root, part, name)
        if path is None:
            raise FileNotFoundError(f"no {name} under {root}")
        recs = normalize(_json(path), summaries=split == "train")
        if split == "test" and any(r[0] is None for r in recs):
            raise ValueError(f"{path}: test records without an id field")
        if split == "train" and any(not (r[2] or "").strip() for r in recs):
            print(f"WARNING: {sum(not (r[2] or '').strip() for r in recs)} training records without a summary (dropped)")
            recs = [r for r in recs if (r[2] or "").strip()]
        recs = [(r[0] if r[0] is not None else str(i), r[1], r[2]) for i, r in enumerate(recs)]
        out["files"][split] = path
        sents, summ = _build(recs, tok_path, workers)
        out[split] = Docs([r[0] for r in recs], sents, summ, [r[2] for r in recs] if split == "train" else None,
                          [len(r[1].split()) for r in recs])
    tr = out["train"]
    rng = np.random.default_rng(seed)
    n_dev = max(min_dev, int(round(dev_frac * len(tr)))) if len(tr) > 2 * min_dev else max(2, len(tr) // 5)
    perm = rng.permutation(len(tr))
    out["dev"], out["train"] = tr.take(np.sort(perm[:n_dev])), tr.take(np.sort(perm[n_dev:]))
    for s, p in paths.items():
        d = out[s]
        tmp = p[:-4] + f".{os.getpid()}.tmp.npz"
        np.savez(tmp, **_pack(d))
        with open(tmp + ".json", "w", encoding="utf-8") as f:
            json.dump({"ids": d.ids, "ref": d.ref}, f)
        os.replace(tmp + ".json", p + ".json")
        os.replace(tmp, p)
    return out


def describe(root, chars=300):
    lines = []
    for part, name in (("Train", "train.json"), ("Test", "test.json")):
        p = _find(root, part, name)
        if p is None:
            lines.append(f"{name}: not found")
            continue
        raw = _json(p)
        first = raw[list(raw)[0]] if isinstance(raw, dict) and raw else (raw[0] if raw else None)
        lines.append(f"{os.path.relpath(p, root)}: {type(raw).__name__} of {len(raw)}")
        if isinstance(first, dict):
            short = {k: (v[:chars] + f"... [{len(v)} chars]" if isinstance(v, str) and len(v) > chars else v)
                     for k, v in first.items()}
            lines.append(f"   keys {list(first)}; types {[type(v).__name__ for v in first.values()]}")
            lines.append("   first: " + json.dumps(short, ensure_ascii=False)[:chars * 3])
    return "\n".join(lines)


class IDF:
    """Document-frequency IDF over token ids (fitted on training documents)."""

    def __init__(self, docs, V=32768, max_docs=3000, seed=0):
        idx = np.random.default_rng(seed).permutation(len(docs))[:max_docs]
        df = np.zeros(V, np.float64)
        for i in idx:
            t = docs.doc_tokens(i)
            if len(t):
                df[np.unique(t)] += 1
        self.idf = np.log((1 + len(idx)) / (1 + df)) + 1.0


def _unigram_f1(a, b):
    if not len(a) or not len(b):
        return 0.0
    ca, cb = np.bincount(a.astype(np.int64)), np.bincount(b.astype(np.int64))
    n = min(len(ca), len(cb))
    ov = np.minimum(ca[:n], cb[:n]).sum()
    return 2 * ov / (len(a) + len(b))


def _fill(sents, order, budget):
    keep, used = [], 0
    for j in order:
        n = len(sents[j])
        if used + n <= budget:
            keep.append(j)
            used += n
        if used >= budget:
            break
    return sorted(keep)


def select_input(sents, budget, how="head", idf=None, head_tokens=256, lead_frac=0.25, summary=None):
    """sents: list of token arrays of one document -> int64 array of <= budget tokens."""
    full = np.concatenate(sents) if sents else np.zeros(0, np.uint16)
    if len(full) <= budget or how == "head":
        return full[:budget].astype(np.int64)
    if how == "headtail":
        h = min(head_tokens, budget - 1)
        return np.concatenate([full[:h], full[len(full) - (budget - h):]]).astype(np.int64)
    if how == "oracle" and summary is not None and len(summary):
        sc = np.array([_unigram_f1(s, summary) for s in sents])
        return np.concatenate([sents[j] for j in _fill(sents, np.argsort(-sc), budget)] or [full[:budget]]).astype(np.int64)
    if how in ("central", "oracle"):
        lead, used, chosen = int(lead_frac * budget), 0, []
        for j, s in enumerate(sents):
            if used + len(s) > lead:
                break
            chosen.append(j)
            used += len(s)
        w2 = idf.idf ** 2 if idf is not None else np.ones(int(full.max()) + 1)
        u, c = np.unique(full.astype(np.int64), return_counts=True)
        dvec = np.zeros(len(w2))
        dvec[u] = c
        dn = np.sqrt((dvec[u] ** 2 * w2[u]).sum()) + 1e-9
        rest = list(range(len(chosen), len(sents)))
        sc = []
        for j in rest:
            us, cs = np.unique(sents[j].astype(np.int64), return_counts=True)
            sc.append((cs * w2[us] * dvec[us]).sum() / (np.sqrt((cs ** 2 * w2[us]).sum()) + 1e-9) / dn)
        left = budget - used
        for k in np.argsort(-np.array(sc)):
            j = rest[k]
            if len(sents[j]) <= left:
                chosen.append(j)
                left -= len(sents[j])
        return np.concatenate([sents[j] for j in sorted(chosen)]).astype(np.int64)
    raise ValueError(f"unknown selection {how!r}")


def chunk_sentences(sents, budget):
    """Consecutive sentence ranges [(a, b)] with <= budget tokens each (a long sentence is its own chunk)."""
    out, a, used = [], 0, 0
    for j, s in enumerate(sents):
        if j > a and used + len(s) > budget:
            out.append((a, j))
            a, used = j, 0
        used += len(s)
    if sents:
        out.append((a, len(sents)))
    return out


def dancer_pairs(sents, summary_sents, budget):
    """-> list of (chunk tokens, target tokens): summary sentences aligned to the chunk they overlap most
    (chunks with no aligned sentence are skipped)."""
    ch = chunk_sentences(sents, budget)
    chunk_tok = [np.concatenate(sents[a:b])[:budget] for a, b in ch]
    tgt = [[] for _ in ch]
    for s in summary_sents:
        k = int(np.argmax([_unigram_f1(s, c) for c in chunk_tok]))
        tgt[k].append(s)
    return [(c.astype(np.int64), np.concatenate(t).astype(np.int64)) for c, t in zip(chunk_tok, tgt) if t]
