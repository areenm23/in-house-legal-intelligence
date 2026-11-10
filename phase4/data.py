"""Court-judgment-prediction data: loading (format auto-detected), whole-document A-1 tokenisation into a
compact cache, stratified subsets, and the document views a model reads.

Official layout (HF `Exploration-Lab/CS-781-Capstone`, folder Phase_4_CJPE):
  Train/train.json, Train/dev.json, Test/test.json
A record is {"id": ..., "text": "...", "label": 0|1} (test: no label). The loader also accepts an
{id: {...}} / {id: text} mapping, a "data"/"documents" wrapper, JSON lines, other key names
(case_id/name, case_text/document, gold/outcome), labels as strings/bools, and text given as a list of
paragraphs (joined with spaces). `describe()` prints what it found so the first minutes of a Kaggle run
show the real format.

Tokens: every document is encoded whole with the A-1 BPE (words from str.split(), so whitespace layout
is irrelevant). A split is stored as one uint16 array of all tokens + offsets (`Split`).

Views (`views(...)`): what the model reads from one document
  first / last / middle    one sequence of seq_len tokens from the start / end / centre
  headtail                 first `head_tokens` + last (seq_len - head_tokens) tokens, one sequence
  hier                     K chunks of seq_len tokens; chunks are aligned to the end of the document and
                           the first chunk starts at token 0 (so it overlaps the second): every chunk has
                           exactly seq_len tokens unless the whole document is shorter. pick = last |
                           spread (evenly over the document, first and last included) | random | all
Each view also gives the relative position (0..1) of every chunk's centre in the document.
"""
import glob
import hashlib
import json
import os
import re

import numpy as np

HF_REPO = "Exploration-Lab/CS-781-Capstone"
FOLDER = "Phase_4_CJPE"
PLACEHOLDER = 3
SPLITS = (("train", "Train", "train.json"), ("dev", "Train", "dev.json"), ("test", "Test", "test.json"))


def download(out_dir, token=None):
    from huggingface_hub import snapshot_download
    snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=[f"{FOLDER}/*"], local_dir=out_dir, token=token)
    return os.path.join(out_dir, FOLDER)


def _find(root, part, name):
    p = os.path.join(root, part, name)
    if os.path.exists(p):
        return p
    stem = os.path.splitext(name)[0]
    rx = re.compile(rf"^{stem}\.(json|jsonl)$", re.I)
    for d in (os.path.join(root, part), root):
        for q in sorted(glob.glob(os.path.join(d, "*"))):
            if rx.match(os.path.basename(q)):
                return q
    return None


def find_root(*cands):
    for c in cands:
        if not c:
            continue
        for sub in ("", FOLDER):
            p = os.path.join(c, sub)
            if _find(p, "Train", "train.json") and _find(p, "Test", "test.json"):
                return p
    return None


def _json(path):
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return [json.loads(ln) for ln in txt.splitlines() if ln.strip()]


TEXT_KEYS = ("text", "case_text", "document", "doc", "judgment", "judgement", "content", "body", "input")
LABEL_KEYS = ("label", "gold", "labels", "outcome", "decision", "y", "target", "pred")
ID_KEYS = ("id", "case_id", "doc_id", "docid", "uid", "name", "case_name", "file", "filename")


def _first(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join(_text(s) if not isinstance(s, dict) else _text(_first(s, TEXT_KEYS + ("sentence",)))
                        for s in x)
    return str(x)


def _label(x):
    if x is None:
        return None
    if isinstance(x, list) and len(x) == 1:
        x = x[0]
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        v = int(x)
    elif isinstance(x, float) and x in (0.0, 1.0):
        v = int(x)
    elif isinstance(x, str):
        s = x.strip().lower()
        m = {"0": 0, "1": 1, "rejected": 0, "dismissed": 0, "accepted": 1, "allowed": 1, "false": 0, "true": 1}
        if s not in m:
            raise ValueError(f"unknown label {x!r}")
        v = m[s]
    else:
        raise ValueError(f"unknown label {x!r}")
    if v not in (0, 1):
        raise ValueError(f"label {x!r} is not binary")
    return v


def normalize(raw, labels=True):
    """-> list of (id or None, text, label or None), in file order. labels=False ignores any label field
    (test files may carry placeholders)."""
    lab = _label if labels else (lambda x: None)
    if isinstance(raw, dict):
        for k in ("data", "documents", "docs", "records", "examples"):
            if k in raw and isinstance(raw[k], (list, dict)):
                raw = raw[k]
                break
    recs = []
    if isinstance(raw, dict):
        for did, v in raw.items():
            if isinstance(v, dict):
                i = _first(v, ID_KEYS)
                recs.append((str(i if i is not None else did), _text(_first(v, TEXT_KEYS)), lab(_first(v, LABEL_KEYS))))
            else:
                recs.append((str(did), _text(v), None))
    else:
        for v in raw:
            i = _first(v, ID_KEYS)
            recs.append((None if i is None else str(i), _text(_first(v, TEXT_KEYS)), lab(_first(v, LABEL_KEYS))))
    return recs


class Split:
    """One split as arrays: ids (list of str), y (int8, -1 = unknown), off (n+1 int64), tok (uint16),
    n_chars (int64), h (text hash, 16 hex chars) - documents are tok[off[i]:off[i+1]]."""

    def __init__(self, name, ids, y, off, tok, n_chars, h):
        self.name, self.ids, self.y, self.off, self.tok, self.n_chars, self.h = name, ids, y, off, tok, n_chars, h

    def __len__(self):
        return len(self.ids)

    def doc(self, i):
        return self.tok[self.off[i]:self.off[i + 1]]

    def n_tok(self):
        return np.diff(self.off)

    @property
    def labelled(self):
        return len(self.y) > 0 and bool((self.y >= 0).all())


def take(sp, idx, name=None):
    """A new Split with documents idx of sp (in that order)."""
    idx = np.asarray(idx, dtype=np.int64)
    parts = [sp.doc(i) for i in idx]
    off = np.zeros(len(idx) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(p) for p in parts])
    tok = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)
    return Split(name or sp.name, [sp.ids[i] for i in idx], sp.y[idx], off, tok, sp.n_chars[idx],
                 [sp.h[i] for i in idx])


def text_hash(text):
    return hashlib.sha1(" ".join(text.lower().split()).encode("utf-8")).hexdigest()[:16]


_TOK = None


def _init_worker(tok_path):
    global _TOK
    from phase1.data.a1_bpe import load_tokenizer
    _TOK = load_tokenizer(tok_path)


def _encode_texts(texts):
    return [np.asarray(_TOK.encode(t).ids, dtype=np.uint16) for t in texts]


def encode_texts(texts, tok_path, workers=1, part=64):
    """-> list of uint16 arrays (multiprocessing pool when workers > 1)."""
    if workers <= 1 or len(texts) < 4 * part:
        _init_worker(tok_path)
        return _encode_texts(texts)
    import multiprocessing as mp
    parts = [texts[i:i + part] for i in range(0, len(texts), part)]
    with mp.get_context().Pool(workers, initializer=_init_worker, initargs=(tok_path,)) as pool:
        return [a for chunk in pool.imap(_encode_texts, parts, chunksize=1) for a in chunk]


def build_split(name, recs, tok_path, workers=1):
    ids = [r[0] if r[0] is not None else str(i) for i, r in enumerate(recs)]
    texts = [r[1] for r in recs]
    y = np.array([-1 if r[2] is None else r[2] for r in recs], dtype=np.int8)
    arrs = encode_texts(texts, tok_path, workers)
    arrs = [a if len(a) else np.array([PLACEHOLDER], dtype=np.uint16) for a in arrs]
    off = np.zeros(len(arrs) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(a) for a in arrs])
    tok = np.concatenate(arrs) if arrs else np.zeros(0, dtype=np.uint16)
    n_chars = np.array([len(t) for t in texts], dtype=np.int64)
    h = [text_hash(t) for t in texts]
    return Split(name, ids, y, off, tok, n_chars, h)


def read_split(root, split):
    part, name = {s: (p, n) for s, p, n in SPLITS}[split]
    path = _find(root, part, name)
    if path is None:
        return None, None
    recs = normalize(_json(path), labels=split != "test")
    if split == "test":
        if any(r[0] is None for r in recs):
            raise ValueError(f"{path}: test records without an id field")
    elif any(r[2] is None for r in recs):
        raise ValueError(f"{path}: {sum(r[2] is None for r in recs)} records without a label")
    return recs, path


def cache_key(root, tok_path):
    h = hashlib.sha1()
    for _s, part, name in SPLITS:
        p = _find(root, part, name)
        if p:
            h.update(os.path.basename(p).encode())
            h.update(str(os.path.getsize(p)).encode())
    h.update(open(tok_path, "rb").read(4096))
    return h.hexdigest()[:16]


def _save(sp, path):
    np.savez(path, y=sp.y, off=sp.off, tok=sp.tok, n_chars=sp.n_chars)
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump({"ids": sp.ids, "h": sp.h}, f)


def _load(name, path):
    z = np.load(path)
    meta = json.load(open(path + ".json", encoding="utf-8"))
    return Split(name, meta["ids"], z["y"], z["off"], z["tok"], z["n_chars"], meta["h"])


def load_data(root, tok_path, cache_dir, workers=None):
    """-> {"train": Split, "dev": Split, "test": Split, "files": {...}} (dev may be None). Cached as
    <cache_dir>/p4_<key>_<split>.npz (+ .json with ids and text hashes)."""
    os.makedirs(cache_dir, exist_ok=True)
    key = cache_key(root, tok_path)
    workers = workers if workers is not None else max(1, min(8, (os.cpu_count() or 1)))
    out = {"files": {}}
    for split, _p, _n in SPLITS:
        path = os.path.join(cache_dir, f"p4_{key}_{split}.npz")
        if os.path.exists(path) and os.path.exists(path + ".json"):
            out[split] = _load(split, path)
            continue
        recs, src = read_split(root, split)
        if recs is None:
            if split == "train":
                raise FileNotFoundError(f"no train.json under {root}")
            out[split] = None
            continue
        out["files"][split] = src
        sp = build_split(split, recs, tok_path, workers)
        tmp = path[:-4] + f".{os.getpid()}.tmp.npz"
        _save(sp, tmp)
        os.replace(tmp + ".json", path + ".json")
        os.replace(tmp, path)
        out[split] = sp
    if out["test"] is None:
        raise FileNotFoundError(f"no test.json under {root}")
    return out


def describe(root, n=1, chars=300):
    """Raw structure of every file (type, count, keys, first record shortened) -> str."""
    lines = []
    for split, part, name in SPLITS:
        p = _find(root, part, name)
        if p is None:
            lines.append(f"{split}: not found")
            continue
        raw = _json(p)
        if isinstance(raw, dict):
            keys = list(raw)[:5]
            lines.append(f"{os.path.relpath(p, root)}: dict with {len(raw)} keys, first keys {keys}")
            first = [raw[k] for k in keys[:n]]
        else:
            lines.append(f"{os.path.relpath(p, root)}: {type(raw).__name__} of {len(raw)}")
            first = raw[:n]
        for r in first:
            if isinstance(r, dict):
                short = {k: (v[:chars] + f"... [{len(v)} chars]" if isinstance(v, str) and len(v) > chars else v)
                         for k, v in r.items()}
                lines.append(f"   keys {list(r)}; types {[type(v).__name__ for v in r.values()]}")
                lines.append("   first: " + json.dumps(short, ensure_ascii=False)[:chars * 2])
            else:
                lines.append("   first: " + json.dumps(r, ensure_ascii=False)[:chars])
    return "\n".join(lines)


def stratified_subset(y, n, seed=0):
    """Indices of a class-stratified random subset of size ~n (all indices if n is None or >= len)."""
    idx = np.arange(len(y))
    if n is None or n <= 0 or n >= len(y):
        return idx
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(y):
        ci = idx[y == c]
        k = int(round(n * len(ci) / len(y)))
        out.append(rng.choice(ci, size=min(k, len(ci)), replace=False))
    return np.sort(np.concatenate(out))


def chunk_starts(n, L, stride=None):
    """Start positions of the chunks covering a document of n tokens, in document order. Chunks are
    aligned to the end (the last chunk is [n-L, n)); the first starts at 0. n <= L -> [0]."""
    stride = stride or L
    if n <= L:
        return np.array([0], dtype=np.int64)
    s = np.arange(n - L, 0, -stride)[::-1]
    return np.concatenate([[0], s]).astype(np.int64) if s[0] > 0 else s.astype(np.int64)


def pick_chunks(m, k, how, rng=None):
    """Indices (sorted) of k chunks out of m."""
    if how == "all" or m <= k:
        return np.arange(m)
    if how == "last":
        return np.arange(m - k, m)
    if how == "first":
        return np.arange(k)
    if how == "spread":
        return np.unique(np.round(np.linspace(0, m - 1, k)).astype(np.int64))
    if how == "random":
        rng = rng or np.random.default_rng()
        return np.sort(rng.choice(m, size=k, replace=False))
    raise ValueError(how)


def views(t, ctx, L, head_tokens=128, k=4, pick="last", stride=None, rng=None):
    """t: token array of one document -> (list of int64 arrays, rel_pos float array): the sequences to
    encode and the relative position of each one's centre (0 = start, 1 = end)."""
    n = len(t)
    if n == 0:
        return [np.array([PLACEHOLDER], dtype=np.int64)], np.array([0.5])
    t = t.astype(np.int64)
    if ctx == "hier":
        st = chunk_starts(n, L, stride)
        sel = pick_chunks(len(st), k, pick, rng)
        seqs = [t[s:s + L] for s in st[sel]]
        pos = np.array([(s + len(q) / 2) / n for s, q in zip(st[sel], seqs)])
        return seqs, pos
    if n <= L:
        return [t], np.array([0.5])
    if ctx == "first":
        return [t[:L]], np.array([L / 2 / n])
    if ctx == "last":
        return [t[n - L:]], np.array([1 - L / 2 / n])
    if ctx == "middle":
        s = (n - L) // 2
        return [t[s:s + L]], np.array([0.5])
    if ctx == "headtail":
        h = min(head_tokens, L - 1)
        return [np.concatenate([t[:h], t[n - (L - h):]])], np.array([0.5])
    raise ValueError(f"unknown ctx {ctx!r}")
