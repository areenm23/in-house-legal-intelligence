"""Rhetorical-role data: loading (format auto-detected), tokenisation per sentence, training chunks.

Official layout (HF `Exploration-Lab/CS-781-Capstone`, folder Phase_3_RR):
  Train/{CL,IT}_{train,dev}.json, Train/label_vocab.json, Test/{CL,IT}_test.json
A document is {"id": ..., "text": [sentence, ...], "labels": [label, ...]} (test: no labels).
The loader also accepts {id: {...}} / {id: [sentences]} mappings, a "data"/"documents" wrapper,
other key names (sentences/sents, label/roles), integer label ids (mapped through label_vocab)
and a single text string (split on newlines). `describe()` prints what it found so the first
minutes of a Kaggle run show the real format.

Tokenisation: every sentence is encoded on its own with the A-1 BPE (BPE never crosses a word, so
this equals encoding the document and cutting at sentence boundaries). A document is the
concatenation of its sentences; `sent_of[t]` is the sentence of token t. Sentences are capped at
`max_sent_tokens`; an empty sentence gets one placeholder token (</s>) so it still has a vector.
"""
import glob
import hashlib
import json
import os
import re

import numpy as np

HF_REPO = "Exploration-Lab/CS-781-Capstone"
FOLDER = "Phase_3_RR"
DOMAINS = ("CL", "IT")
PLACEHOLDER = 3


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
            if glob.glob(os.path.join(p, "Train", "*label*")) or glob.glob(os.path.join(p, "Train", "*_train.json")):
                return p
    return None


def _find(root, part, pattern):
    hits = sorted(glob.glob(os.path.join(root, part, pattern)))
    if not hits:
        rx = re.compile(pattern.replace(".", r"\.").replace("*", ".*") + "$", re.I)
        hits = sorted(p for p in glob.glob(os.path.join(root, part, "*")) if rx.match(os.path.basename(p)))
    return hits[0] if hits else None


def _json(path):
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return [json.loads(ln) for ln in txt.splitlines() if ln.strip()]


TEXT_KEYS = ("text", "sentences", "sents", "sentence", "document", "doc")
LABEL_KEYS = ("labels", "label", "roles", "rhetorical_roles", "tags", "annotations")
ID_KEYS = ("id", "doc_id", "document_id", "docid", "name", "file")


def load_labels(path):
    v = _json(path)
    if isinstance(v, dict):
        for k in ("labels", "label_vocab", "vocab"):
            if k in v and isinstance(v[k], (list, dict)):
                v = v[k]
                break
    if isinstance(v, list):
        return [str(x) for x in v]
    vals = list(v.values())
    if all(isinstance(x, int) and not isinstance(x, bool) for x in vals):
        return [k for k, _ in sorted(v.items(), key=lambda kv: kv[1])]
    if all(isinstance(x, str) for x in vals):
        return [v[k] for k in sorted(v, key=lambda k: int(k) if str(k).isdigit() else k)]
    raise ValueError(f"unsupported label_vocab format in {path}")


def _first(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _sentences(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [s for s in x.split("\n")]
    out = []
    for s in x:
        if isinstance(s, dict):
            s = _first(s, ("text", "sentence", "sent")) or ""
        out.append("" if s is None else str(s))
    return out


def _labels(x, vocab):
    if x is None:
        return None
    out = []
    for l in x:
        if isinstance(l, dict):
            l = _first(l, ("label", "role", "tag"))
        if isinstance(l, list) and len(l) == 1:
            l = l[0]
        if isinstance(l, (int, np.integer)) and not isinstance(l, bool):
            l = vocab[int(l)]
        out.append(str(l))
    return out


def normalize_docs(raw, vocab):
    """-> list of (id, sentences, labels or None), in file order."""
    if isinstance(raw, dict):
        for k in ("data", "documents", "docs"):
            if k in raw and isinstance(raw[k], (list, dict)):
                raw = raw[k]
                break
    recs = []
    if isinstance(raw, dict):
        for did, v in raw.items():
            if isinstance(v, dict):
                recs.append((str(_first(v, ID_KEYS) or did), _sentences(_first(v, TEXT_KEYS)),
                             _labels(_first(v, LABEL_KEYS), vocab)))
            else:
                recs.append((str(did), _sentences(v), None))
    else:
        for i, v in enumerate(raw):
            did = _first(v, ID_KEYS)
            recs.append((str(did if did is not None else i), _sentences(_first(v, TEXT_KEYS)),
                         _labels(_first(v, LABEL_KEYS), vocab)))
    for did, sents, labs in recs:
        if labs is not None and len(labs) != len(sents):
            raise ValueError(f"doc {did}: {len(sents)} sentences but {len(labs)} labels")
    return recs


class RRDoc:
    __slots__ = ("id", "domain", "sents", "labels", "y", "ids", "sent_of", "n_tok")

    def __init__(self, did, domain, sents, labels=None, lab2id=None):
        self.id, self.domain, self.sents, self.labels = did, domain, sents, labels
        self.y = None if labels is None or lab2id is None else np.array([lab2id[l] for l in labels], dtype=np.int64)
        self.ids = self.sent_of = None
        self.n_tok = 0

    @property
    def key(self):
        return f"{self.domain}/{self.id}"

    def __len__(self):
        return len(self.sents)

    def encode(self, tok, max_sent_tokens=256):
        ids, sof = [], []
        for i, s in enumerate(self.sents):
            t = tok.encode(s).ids[:max_sent_tokens] if s.strip() else []
            if not t:
                t = [PLACEHOLDER]
            ids += t
            sof += [i] * len(t)
        self.ids = np.array(ids, dtype=np.int32)
        self.sent_of = np.array(sof, dtype=np.int32)
        self.n_tok = len(ids)
        return self


def load_split(root):
    """-> dict(labels, train/dev/test: {domain: [RRDoc]}, files: {...})."""
    lv = _find(root, "Train", "label_vocab.json") or _find(root, "Train", "*label*.json")
    if lv is None:
        raise FileNotFoundError(f"label_vocab.json not found under {root}/Train")
    labels = load_labels(lv)
    lab2id = {l: i for i, l in enumerate(labels)}
    out = {"labels": labels, "train": {}, "dev": {}, "test": {}, "files": {"labels": lv}}
    for dom in DOMAINS:
        for split, part in (("train", "Train"), ("dev", "Train"), ("test", "Test")):
            p = _find(root, part, f"{dom}_{split}.json")
            if p is None:
                out[split][dom] = []
                continue
            out["files"][f"{dom}_{split}"] = p
            docs = []
            for did, sents, labs in normalize_docs(_json(p), labels):
                if split == "test":
                    labs = None
                elif labs is None:
                    raise ValueError(f"{p}: document {did} has no labels")
                unknown = set(labs or []) - set(labels)
                if unknown:
                    raise ValueError(f"{p}: labels not in label_vocab: {sorted(unknown)}")
                docs.append(RRDoc(did, dom, sents, labs, lab2id))
            out[split][dom] = docs
    if not any(out["train"].values()):
        raise FileNotFoundError(f"no *_train.json under {root}/Train")
    return out


def all_docs(split_map, domains=DOMAINS):
    return [d for dom in domains for d in split_map.get(dom, [])]


def describe(root, n=2):
    """Print the raw structure of every file (first records) -> str (for the notebook log)."""
    lines = []
    for p in sorted(glob.glob(os.path.join(root, "*", "*.json"))):
        raw = _json(p)
        kind = type(raw).__name__
        if isinstance(raw, dict):
            keys = list(raw)[:5]
            first = raw[keys[0]] if keys else None
            lines.append(f"{os.path.relpath(p, root)}: dict with {len(raw)} keys, first keys {keys}")
        else:
            first = raw[0] if raw else None
            lines.append(f"{os.path.relpath(p, root)}: {kind} of {len(raw)}")
        s = json.dumps(first, ensure_ascii=False)
        lines.append("   first: " + (s[:400] + (" ..." if len(s) > 400 else "")))
    return "\n".join(lines)


def encode_all(data, tok, max_sent_tokens=256):
    for split in ("train", "dev", "test"):
        for docs in data[split].values():
            for d in docs:
                d.encode(tok, max_sent_tokens)
    return data


def cache_key(root, tok_path, max_sent_tokens):
    h = hashlib.sha1()
    for p in sorted(glob.glob(os.path.join(root, "*", "*.json"))):
        h.update(os.path.basename(p).encode())
        h.update(str(os.path.getsize(p)).encode())
    h.update(open(tok_path, "rb").read(4096))
    h.update(str(max_sent_tokens).encode())
    return h.hexdigest()[:16]


def windows(n, size, stride):
    """(start, end) token windows covering 0..n; all have length `size` (last one aligned to the end)
    unless n <= size. size <= 0 = one window over everything."""
    if size <= 0 or n <= size:
        return [(0, n)]
    out, s = [], 0
    while s + size < n:
        out.append((s, s + size))
        s += stride
    out.append((n - size, n))
    return sorted(set(out))


def sentence_token_ranges(doc):
    """(n_sent, 2) [start, end) token range of every sentence."""
    n = len(doc)
    starts = np.searchsorted(doc.sent_of, np.arange(n), side="left")
    ends = np.searchsorted(doc.sent_of, np.arange(n), side="right")
    return np.stack([starts, ends], 1)


def chunks(doc, max_tokens, rng=None):
    """Split a document into runs of consecutive sentences with <= max_tokens tokens each
    (a sentence longer than that is a chunk of its own). A random first cut (rng) varies the
    boundaries between epochs. -> list of (s0, s1) sentence ranges."""
    n = len(doc)
    if n == 0:
        return []
    rng_ = sentence_token_ranges(doc)
    lens = rng_[:, 1] - rng_[:, 0]
    out, s0, tot = [], 0, 0
    first_cap = int(rng.integers(max(1, max_tokens // 4), max_tokens + 1)) if rng is not None else max_tokens
    cap = first_cap
    for i in range(n):
        if i > s0 and tot + lens[i] > cap:
            out.append((s0, i))
            s0, tot, cap = i, 0, max_tokens
        tot += lens[i]
    out.append((s0, n))
    return out


def label_counts(docs, n):
    return np.bincount(np.concatenate([d.y for d in docs if d.y is not None and len(d)] or [np.zeros(0, int)]),
                       minlength=n)
