"""Legal-statute-identification data: loading (format auto-detected), whole-document A-1 token cache,
multi-hot labels, a *clean* dev split, label-frequency buckets.

Official layout (HF `Exploration-Lab/CS-781-Capstone`, folder Phase_5_LSI):
  Train/train.json, Train/dev.json, Test/test.json
A record is {"id": ..., "text": "...", "labels": [1, 7, 23]} (test: no labels). The loader also accepts
an {id: {...}} mapping, a "data"/"documents" wrapper, JSON lines, other key names (case_id, statutes,
statute_ids, gold, label), a single int, and ids given as digit strings.

Clean dev: Phase 4's dev.json contained every test document (with gold labels) and duplicate rows. Here
`clean_dev()` keeps one row per dev id and drops rows whose id or text also occurs in test, so model
selection and threshold fitting never see test documents. Documents, views and chunking are Phase 4's.
"""
import hashlib
import json
import os

import numpy as np

from phase4.data import _find, _first, _json, _text, chunk_starts, encode_texts, pick_chunks, text_hash, views

HF_REPO = "Exploration-Lab/CS-781-Capstone"
FOLDER = "Phase_5_LSI"
PLACEHOLDER = 3
SPLITS = (("train", "Train", "train.json"), ("dev", "Train", "dev.json"), ("test", "Test", "test.json"))
TEXT_KEYS = ("text", "case_text", "document", "doc", "facts", "content", "body", "input")
LABEL_KEYS = ("labels", "label", "statutes", "statute_ids", "statute", "gold", "targets", "y")
ID_KEYS = ("id", "case_id", "doc_id", "docid", "uid", "name", "file", "filename")


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


def _labels(x):
    """-> sorted unique list of non-negative ints."""
    if x is None:
        return None
    if isinstance(x, (int, np.integer, str)) and not isinstance(x, bool):
        x = [x]
    out = set()
    for v in x:
        if isinstance(v, dict):
            v = _first(v, ("id", "label", "statute"))
        if isinstance(v, bool):
            raise ValueError(f"boolean label {v!r}")
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        if isinstance(v, str) and v.strip().lstrip("-").isdigit():
            v = int(v.strip())
        if not isinstance(v, (int, np.integer)) or v < 0:
            raise ValueError(f"label {v!r} is not a non-negative integer statute id")
        out.add(int(v))
    return sorted(out)


def normalize(raw, labels=True):
    """-> list of (id or None, text, label list or None), in file order."""
    lab = _labels if labels else (lambda x: None)
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
    """ids (list of str), off (n+1), tok (uint16), n_chars, h (text hashes), lab_off (n+1) + lab (int16)
    = label lists, labelled (bool). doc(i) -> tokens, labels(i) -> label ids, Y(L) -> (n, L) uint8."""

    def __init__(self, name, ids, off, tok, n_chars, h, lab_off, lab, labelled):
        self.name, self.ids, self.off, self.tok, self.n_chars, self.h = name, ids, off, tok, n_chars, h
        self.lab_off, self.lab, self.labelled = lab_off, lab, labelled

    def __len__(self):
        return len(self.ids)

    def doc(self, i):
        return self.tok[self.off[i]:self.off[i + 1]]

    def labels(self, i):
        return self.lab[self.lab_off[i]:self.lab_off[i + 1]]

    def n_tok(self):
        return np.diff(self.off)

    def Y(self, L, idx=None):
        idx = np.arange(len(self)) if idx is None else np.asarray(idx)
        out = np.zeros((len(idx), L), dtype=np.uint8)
        for r, i in enumerate(idx):
            lb = self.labels(i)
            out[r, lb[lb < L]] = 1
        return out


def build_split(name, recs, tok_path, workers=1):
    ids = [r[0] if r[0] is not None else str(i) for i, r in enumerate(recs)]
    texts = [r[1] for r in recs]
    arrs = encode_texts(texts, tok_path, workers)
    arrs = [a if len(a) else np.array([PLACEHOLDER], dtype=np.uint16) for a in arrs]
    off = np.zeros(len(arrs) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(a) for a in arrs])
    labelled = all(r[2] is not None for r in recs)
    labs = [np.asarray(r[2] or [], dtype=np.int16) for r in recs]
    lab_off = np.zeros(len(labs) + 1, dtype=np.int64)
    lab_off[1:] = np.cumsum([len(x) for x in labs])
    return Split(name, ids, off, np.concatenate(arrs) if arrs else np.zeros(0, np.uint16),
                 np.array([len(t) for t in texts], dtype=np.int64), [text_hash(t) for t in texts], lab_off,
                 np.concatenate(labs) if labs else np.zeros(0, np.int16), labelled)


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
        raise ValueError(f"{path}: {sum(r[2] is None for r in recs)} records without labels")
    return recs, path


def cache_key(root, tok_path):
    h = hashlib.sha1(b"p5")
    for _s, part, name in SPLITS:
        p = _find(root, part, name)
        if p:
            h.update(os.path.basename(p).encode())
            h.update(str(os.path.getsize(p)).encode())
    h.update(open(tok_path, "rb").read(4096))
    return h.hexdigest()[:16]


def _save(sp, path):
    np.savez(path, off=sp.off, tok=sp.tok, n_chars=sp.n_chars, lab_off=sp.lab_off, lab=sp.lab)
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump({"ids": sp.ids, "h": sp.h, "labelled": sp.labelled}, f)


def _load(name, path):
    z = np.load(path)
    m = json.load(open(path + ".json", encoding="utf-8"))
    return Split(name, m["ids"], z["off"], z["tok"], z["n_chars"], m["h"], z["lab_off"], z["lab"], m["labelled"])


def load_data(root, tok_path, cache_dir, workers=None):
    """-> {"train", "dev", "test": Split, "n_labels", "dev_clean": indices, "files"} (cached .npz + .json)."""
    os.makedirs(cache_dir, exist_ok=True)
    key = cache_key(root, tok_path)
    workers = workers if workers is not None else max(1, min(8, (os.cpu_count() or 1)))
    out = {"files": {}}
    for split, _p, _n in SPLITS:
        path = os.path.join(cache_dir, f"p5_{key}_{split}.npz")
        if os.path.exists(path) and os.path.exists(path + ".json"):
            out[split] = _load(split, path)
            continue
        recs, src = read_split(root, split)
        if recs is None:
            if split == "dev":
                out[split] = None
                continue
            raise FileNotFoundError(f"no {split}.json under {root}")
        out["files"][split] = src
        sp = build_split(split, recs, tok_path, workers)
        tmp = path[:-4] + f".{os.getpid()}.tmp.npz"
        _save(sp, tmp)
        os.replace(tmp + ".json", path + ".json")
        os.replace(tmp, path)
        out[split] = sp
    labs = [out[s].lab for s in ("train", "dev") if out.get(s) is not None and len(out[s].lab)]
    out["n_labels"] = int(max(int(x.max()) for x in labs)) + 1 if labs else 0
    out["dev_clean"] = clean_dev(out["dev"], out["test"]) if out["dev"] is not None else None
    return out


def clean_dev(dev, test):
    """Indices of dev rows to use: first row of each id, and neither id nor text in the test split."""
    tid, th = set(test.ids), set(test.h)
    seen, keep = set(), []
    for i, (d, h) in enumerate(zip(dev.ids, dev.h)):
        if d in seen or d in tid or h in th:
            continue
        seen.add(d)
        keep.append(i)
    return np.array(keep, dtype=np.int64)


def label_counts(sp, L, idx=None):
    return sp.Y(L, idx).sum(0)


def freq_buckets(counts):
    """Train-frequency buckets per label: 'frequent' (top third of labels by count), 'medium', 'rare'
    (bottom third), 'unseen' (0 training documents)."""
    counts = np.asarray(counts)
    order = np.argsort(-counts, kind="stable")
    seen = order[counts[order] > 0]
    out = np.array(["unseen"] * len(counts), dtype=object)
    k = len(seen)
    for r, l in enumerate(seen):
        out[l] = "frequent" if r < k / 3 else ("medium" if r < 2 * k / 3 else "rare")
    return out.tolist()


def random_subset(n, k, seed=0):
    if k is None or k <= 0 or k >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))


def describe(root, chars=300):
    lines = []
    for split, part, name in SPLITS:
        p = _find(root, part, name)
        if p is None:
            lines.append(f"{split}: not found")
            continue
        raw = _json(p)
        first = (raw[list(raw)[0]] if isinstance(raw, dict) and raw else (raw[0] if raw else None))
        lines.append(f"{os.path.relpath(p, root)}: {type(raw).__name__} of {len(raw)}")
        if isinstance(first, dict):
            short = {k: (v[:chars] + f"... [{len(v)} chars]" if isinstance(v, str) and len(v) > chars else v)
                     for k, v in first.items()}
            lines.append(f"   keys {list(first)}; types {[type(v).__name__ for v in first.values()]}")
            lines.append("   first: " + json.dumps(short, ensure_ascii=False)[:chars * 2])
    return "\n".join(lines)
