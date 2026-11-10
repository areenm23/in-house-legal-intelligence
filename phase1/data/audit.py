"""Phase-1 corpus audit for the report: size, document lengths, text quality, exact and near
duplicates, word vocabulary, tokenizer fertility, and how much of the dev set and the official
test prompts already appears in the documents our models were trained on.

  python -m phase1.data.audit --raw-dir /tmp/cs781/raw --ref round1 --out /kaggle/working/data_audit \
      [--download --delete-raw] [--workers N] [--max-docs N]

--ref holds the round-1 artefacts the models were trained with: tokenizer/a1bpe_*.json,
dev/{prompts,truth,sources}.json and shards/meta.json. The training subset is rebuilt from
meta.json the way make_shards chose it (same val split, keep hash, and the budget cut-off with
its --batch-docs flush granularity) and checked against the document and token counts it logged.

One streaming pass per source (worker processes do the per-document work, the main process
keeps the order-dependent state), then near-duplicate clustering over all sources.
Nothing is removed: the quality flags are diagnostics with fixed thresholds (QUALITY_FLAGS)
that the report can quote. Deciding what to clean is a separate step.

Outputs in --out: audit.json (every number), audit.md (tables), examples.json (flagged
documents, repeated lines, duplicate clusters), per_example_overlap.json, figures/*.png.
"""
import argparse
import glob
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import re
import shutil
import string
import threading
import time
from collections import Counter

import numpy as np

from .a1_bpe import A1Tokenizer
from .common import HF_DATASET_REPO, HF_TRAIN_DIR, SOURCES, _unit_hash, is_val_doc, iter_json_records, \
    keep_train_doc, record_text, source_file

TEST_PATH = "Phase_1_Pretraining/Test/test.json"
SHINGLE = 5
PERMS, BANDS = 64, 16
NEAR_J = 0.8
NGRAM = 13
SAMPLE_WORDS = 30e6
LINE_SHARE = 0.25
SNIPPET = 160
LONG_WORD = 30

EMPTY, TRAIN, VAL, CUT, UNUSED = 0, 1, 2, 3, 4
FLAG = {EMPTY: "empty", TRAIN: "train", VAL: "val", CUT: "cut_by_budget", UNUSED: "not_sampled"}

QF = ("letter_frac", "digit_frac", "upper_frac", "nonascii_frac", "devanagari_frac", "mean_word_len",
      "long_word_frac", "dup_line_frac", "bad_chars_per_1k")
QUALITY_FLAGS = {
    "short": ("fewer than 50 words", lambda q, w: w < 50),
    "very_long": ("more than 50,000 words", lambda q, w: w > 50_000),
    "low_letters": ("ASCII letters are under 50% of non-space characters (tables, numbers, OCR noise)",
                    lambda q, w: q[:, 0] < 0.5),
    "non_english": ("more than 20% of non-space characters are non-ASCII", lambda q, w: q[:, 3] > 0.2),
    "devanagari": ("more than 5% Devanagari characters", lambda q, w: q[:, 4] > 0.05),
    "all_caps": ("more than 80% of letters upper case (50+ words)", lambda q, w: (q[:, 2] > 0.8) & (w >= 50)),
    "repetitive": ("more than 50% of lines are repeats of another line in the same document",
                   lambda q, w: q[:, 7] > 0.5),
    "garbled": (f"more than 5% of words longer than {LONG_WORD} characters, or more than 1 replacement/"
                "mojibake character per 1,000", lambda q, w: (q[:, 6] > 0.05) | (q[:, 8] > 1)),
}

_WS = b" \t\n\r\x0b\x0c"
_LETTERS, _UPPER, _DIGITS = (s.encode() for s in (string.ascii_letters, string.ascii_uppercase, string.digits))
_HIGH = bytes(range(128, 256))
_DEVA = re.compile("[ऀ-ॿ]")
_MULT = np.uint64(0x9E3779B97F4A7C15)


def _count(b, chars):
    return len(b) - len(b.translate(None, chars))


def _h64(s):
    """Deterministic 64-bit hash (same in every process and on every machine, unlike hash())."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "little")


class _WordHash(dict):
    """word -> 64-bit hash, cached (bounded)."""
    def __missing__(self, w):
        if len(self) > 3_000_000:
            self.clear()
        v = self[w] = _h64(w)
        return v


class _TokLen(dict):
    """word -> number of tokens, cached (bounded); much faster than re-entering the encoder per word."""
    def __init__(self, encw):
        super().__init__()
        self.encw = encw

    def __missing__(self, w):
        if len(self) > 3_000_000:
            self.clear()
        v = self[w] = len(self.encw(w))
        return v


def ngram_hashes(wh, n):
    """uint64 word hashes -> one uint64 hash per window of n consecutive words."""
    m = len(wh) - n + 1
    if m <= 0 or n <= 0:
        return np.empty(0, np.uint64)
    out = wh[:m].copy()
    with np.errstate(over="ignore"):
        for k in range(1, n):
            out = out * _MULT + wh[k:k + m]
    return out


_rng = np.random.default_rng(781)
_A = _rng.integers(1, 2 ** 62, PERMS, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
_B = _rng.integers(1, 2 ** 62, PERMS, dtype=np.uint64)


def minhash(shingles):
    sig = np.full(PERMS, np.iinfo(np.uint64).max, np.uint64)
    sh = np.unique(shingles)
    with np.errstate(over="ignore"):
        for s in range(0, len(sh), 4096):
            np.minimum(sig, (_A[:, None] * sh[None, s:s + 4096] + _B[:, None]).min(axis=1), out=sig)
    return (sig >> np.uint64(32)).astype(np.uint32)


def quality(text, words, wlens):
    b = text.encode("utf-8", "surrogatepass")
    ws = _count(b, _WS)
    nonws = max(1, len(text) - ws)
    letters = _count(b, _LETTERS)
    nonascii = len(text) - (len(b) - _count(b, _HIGH))
    deva = len(_DEVA.findall(text)) if _DEVA.search(text) else 0
    lines = [" ".join(ln.split()) for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    dup = 1 - len(set(lines)) / len(lines) if len(lines) >= 5 else 0.0
    bad = text.count("�") + text.count("Ã") + text.count("â€")
    n = len(words)
    return (letters / nonws, _count(b, _DIGITS) / nonws, _count(b, _UPPER) / max(1, letters), nonascii / nonws,
            deva / nonws, nonws / n, int((wlens > LONG_WORD).sum()) / n, dup, 1000 * bad / len(text))


_G = {}


def _init(state):
    _G.update(state)
    _G["encw"] = A1Tokenizer._from_state(state["tok_state"])._enc._encode_word
    _G["wh"] = _WordHash()
    _G["toklen"] = _TokLen(_G["encw"])
    _G["rep_any"], _G["rep_cand"] = set(), set()


def _work(job):
    src, items, sample_frac, val_frac, keep_frac = job
    encw, toklen, wh, targets, twords = _G["encw"], _G["toklen"], _G["wh"], _G["targets"], _G["twords"]
    rep_any, rep_cand = _G["rep_any"], _G["rep_cand"]
    n = len(items)
    r = {"i": np.array([i for i, _ in items], np.int64), "words": np.zeros(n, np.int64),
         "chars": np.zeros(n, np.int64), "tokens": np.zeros(n, np.int64), "cand": np.zeros(n, np.int8),
         "exact": np.zeros(n, np.uint64), "q": np.zeros((n, len(QF)), np.float32),
         "sig": np.full((n, PERMS), 0xFFFFFFFF, np.uint32), "snip": [], "matches": [], "first_any": [],
         "first_cand": [], "vocab": Counter(), "lines": Counter(), "tok_ids": Counter(), "tok_hist": Counter(),
         "sample": np.zeros(3, np.int64)}
    for j, (i, text) in enumerate(items):
        words = text.split()
        nw = len(words)
        r["snip"].append(" ".join(words[:40])[:SNIPPET])
        if nw == 0:
            continue
        if is_val_doc(src, i, val_frac):
            cand = 2
        elif keep_frac is not None and keep_train_doc(src, i, keep_frac):
            cand = 1
        else:
            cand = 4
        ntok = sum(map(toklen.__getitem__, words))
        wlens = np.fromiter(map(len, words), np.int32, nw)
        r["words"][j], r["chars"][j], r["tokens"][j], r["cand"][j] = nw, len(text), ntok, cand
        r["exact"][j] = _h64(" ".join(words))
        r["q"][j] = quality(text, words, wlens)
        wa = np.fromiter(map(wh.__getitem__, words), np.uint64, nw)
        r["sig"][j] = minhash(ngram_hashes(wa, min(SHINGLE, nw)))
        if targets:
            hit = targets.intersection(ngram_hashes(wa, NGRAM).tolist())
            if hit:
                r["matches"].append((i, list(hit)))
        if twords:
            for w in twords.intersection(words):
                if (src, w) not in rep_any:
                    rep_any.add((src, w))
                    r["first_any"].append(w)
                if cand == 1 and (src, w) not in rep_cand:
                    rep_cand.add((src, w))
                    r["first_cand"].append((w, i))
        if _unit_hash("audit", src, i) < sample_frac:
            toks = list(map(encw, words))
            r["vocab"].update(words)
            r["tok_hist"].update(map(len, toks))
            r["tok_ids"].update(itertools.chain.from_iterable(toks))
            r["sample"] += (nw, ntok, int(wlens.sum()))
            if _unit_hash("audit-lines", src, i) < LINE_SHARE:
                r["lines"].update({" ".join(ws) for ws in (ln.split() for ln in text.split("\n")) if len(ws) >= 6})
    return r


def _batches(path, job_args, max_docs, sem, size=256):
    items, chars = [], 0
    for i, rec in enumerate(iter_json_records(path)):
        if max_docs and i >= max_docs:
            break
        t = record_text(rec)
        items.append((i, t))
        chars += len(t)
        if len(items) >= size or chars >= 8_000_000:
            sem.acquire()
            yield (job_args[0], items, *job_args[1:])
            items, chars = [], 0
    if items:
        sem.acquire()
        yield (job_args[0], items, *job_args[1:])


def _words_of(d):
    return {k: v["text"].split() for k, v in d.items()}


def load_ref(ref):
    toks = [p for p in glob.glob(os.path.join(ref, "tokenizer", "a1bpe_*.json")) if not p.endswith(".report.json")]
    if len(toks) != 1:
        raise SystemExit(f"{ref}/tokenizer must hold exactly one a1bpe_*.json, found {toks}")
    out = {"tokenizer": toks[0], "tokenizer_sha": hashlib.sha1(open(toks[0], "rb").read()).hexdigest()[:12]}
    for name in ("prompts", "truth", "sources"):
        with open(os.path.join(ref, "dev", f"{name}.json"), encoding="utf-8") as f:
            out[name] = json.load(f)
    out["dev_sha"] = hashlib.sha1(json.dumps([out["prompts"], out["truth"]], sort_keys=True).encode()).hexdigest()[:12]
    with open(os.path.join(ref, "shards", "meta.json"), encoding="utf-8") as f:
        out["meta"] = json.load(f)
    return out


def build_targets(dev_prompts, dev_truth, test):
    """Slots = dev prompt, dev continuation, test prompt. Returns slot list, hash -> slots, word types."""
    wh = _WordHash()
    slots, inv = [], {}
    parts = [("dev_prompt", k, v["text"]) for k, v in dev_prompts.items()] + \
            [("dev_truth", k, v["text"]) for k, v in dev_truth.items()] + \
            [("test_prompt", k, v["text"]) for k, v in test.items()]
    for kind, key, text in parts:
        words = text.split()
        hs = set(ngram_hashes(np.fromiter(map(wh.__getitem__, words), np.uint64, len(words)), NGRAM).tolist())
        self_doc = None
        if kind != "test_prompt":
            s, i = key.rsplit("-", 1)
            self_doc = (s, int(i))
        slots.append({"kind": kind, "id": key, "n": len(hs), "self": self_doc})
        for h in hs:
            inv.setdefault(h, []).append(len(slots) - 1)
    twords = set()
    for d in (dev_prompts, dev_truth, test):
        for v in d.values():
            twords.update(v["text"].split())
    return slots, inv, twords


def pct(a, qs=(1, 5, 25, 50, 75, 95, 99)):
    if len(a) == 0:
        return None
    return {**{f"p{q}": float(np.percentile(a, q)) for q in qs}, "mean": float(np.mean(a)), "max": float(np.max(a))}


def ensure_file(raw_dir, rel, download):
    path = os.path.join(raw_dir, rel)
    if not os.path.exists(path):
        if not download:
            raise SystemExit(f"missing {path} (pass --download)")
        from huggingface_hub import hf_hub_download
        t = time.time()
        hf_hub_download(HF_DATASET_REPO, rel, repo_type="dataset", local_dir=raw_dir)
        print(f"downloaded {rel} ({time.time() - t:.0f}s)", flush=True)
    return path


class SourceState:
    """Order-dependent bookkeeping for one source: make_shards' budget cut-off and first sightings."""

    def __init__(self, src, meta_src, batch_docs):
        self.src, self.batch_docs = src, batch_docs
        self.budget = meta_src["budget"] if meta_src else 0
        self.written = self.pend_docs = self.pend_tok = 0
        self.cols = {k: [] for k in ("i", "words", "chars", "tokens", "flag", "exact", "q", "sig")}
        self.snip = []
        self.vocab, self.lines, self.tok_ids, self.tok_hist = Counter(), Counter(), Counter(), Counter()
        self.sample = np.zeros(3, np.int64)
        self.heaps = []

    def flags(self, r):
        out = np.empty(len(r["i"]), np.int8)
        for j, (c, ntok) in enumerate(zip(r["cand"].tolist(), r["tokens"].tolist())):
            if c != 1:
                out[j] = {0: EMPTY, 2: VAL, 4: UNUSED}[c]
            elif self.written >= self.budget:
                out[j] = CUT
            else:
                out[j] = TRAIN
                self.pend_docs += 1
                self.pend_tok += ntok + 1
                if self.pend_docs >= self.batch_docs:
                    self.written += self.pend_tok
                    self.pend_docs = self.pend_tok = 0
        return out


def run_audit(args):
    t_start = time.time()
    ref = load_ref(args.ref)
    meta = ref["meta"]
    tok = A1Tokenizer.from_file(ref["tokenizer"])
    print(f"tokenizer {os.path.basename(ref['tokenizer'])} sha {ref['tokenizer_sha']}, dev set sha {ref['dev_sha']}")
    with open(ensure_file(args.raw_dir, TEST_PATH, args.download), encoding="utf-8") as f:
        test = json.load(f)
    slots, inv, twords = build_targets(ref["prompts"], ref["truth"], test)
    print(f"overlap targets: {len(slots)} slots, {len(inv):,} distinct {NGRAM}-grams, {len(twords):,} word types")
    targets = set(inv)
    best_train = [(0.0, None, None)] * len(slots)
    best_any = [(0.0, None, None, None)] * len(slots)
    union_train = [set() for _ in slots]
    seen_train, seen_any = set(), set()
    states = {}

    state = {"tok_state": tok._state(), "targets": targets, "twords": twords}
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    pool = ctx.Pool(args.workers, initializer=_init, initargs=(state,))
    sources = [s for s in args.sources if s in SOURCES]
    for src in sources:
        path = ensure_file(args.raw_dir, f"{HF_TRAIN_DIR}/train_{src}.json", args.download)
        size = os.path.getsize(path)
        sample_frac = min(1.0, SAMPLE_WORDS * 6.5 / size)
        m = meta["sources"].get(src)
        st = states[src] = SourceState(src, m, args.batch_docs)
        keep = m["keep_frac"] if m else None
        sem = threading.BoundedSemaphore(4 * args.workers)
        t0, nb, nd = time.time(), 0, 0
        print(f"\n[{src}] {size / 1e9:.2f} GB, sample fraction {sample_frac:.4f}, keep_frac {keep}", flush=True)
        jobs = _batches(path, (src, sample_frac, meta["val_frac"], keep), args.max_docs, sem)
        for r in pool.imap(_work, jobs):
            sem.release()
            fl = st.flags(r)
            for k in ("i", "words", "chars", "tokens", "exact", "q", "sig"):
                st.cols[k].append(r[k])
            st.cols["flag"].append(fl)
            st.snip.extend(r["snip"])
            fmap = dict(zip(r["i"].tolist(), fl.tolist()))
            for i, hits in r["matches"]:
                f = fmap[i]
                cnt = Counter()
                for h in hits:
                    for s in inv[h]:
                        if slots[s]["self"] == (src, i):
                            continue
                        cnt[s] += 1
                        if f == TRAIN:
                            union_train[s].add(h)
                for s, c in cnt.items():
                    cov = c / slots[s]["n"]
                    if cov > best_any[s][0]:
                        best_any[s] = (cov, src, i, FLAG[f])
                    if f == TRAIN and cov > best_train[s][0]:
                        best_train[s] = (cov, src, i)
            seen_any.update(r["first_any"])
            seen_train.update(w for w, i in r["first_cand"] if fmap[i] == TRAIN)
            st.vocab.update(r["vocab"])
            st.lines.update(r["lines"])
            st.tok_ids.update(r["tok_ids"])
            st.tok_hist.update(r["tok_hist"])
            st.sample += r["sample"]
            if r["sample"][0]:
                st.heaps.append((int(st.sample[0]), len(st.vocab)))
            nb += 1
            nd += len(r["i"])
            if nb % 500 == 0:
                print(f"  [{src}] {nd:,} docs, {nd / (time.time() - t0):,.0f} docs/s", flush=True)
        st.written += st.pend_tok
        for k in st.cols:
            st.cols[k] = np.concatenate(st.cols[k]) if st.cols[k] else np.empty(0)
        print(f"  [{src}] done: {nd:,} docs in {time.time() - t0:.0f}s", flush=True)
        if args.delete_raw:
            os.remove(path)
    pool.close()
    pool.join()
    print(f"\npass finished in {time.time() - t_start:.0f}s; clustering near-duplicates", flush=True)
    res = analyse(args, ref, tok, test, sources, states, slots, best_train, best_any, union_train, seen_train,
                  seen_any)
    res["meta"]["seconds"] = round(time.time() - t_start)
    return res


def lsh_components(sig, thr=NEAR_J):
    """Connected components of 'estimated Jaccard >= thr' over LSH candidate pairs (each candidate is
    compared with the first member of its bucket)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n, rows = len(sig), PERMS // BANDS
    a_all, b_all = [], []
    for band in range(BANDS):
        part = sig[:, band * rows:(band + 1) * rows].astype(np.uint64)
        key = part[:, 0].copy()
        with np.errstate(over="ignore"):
            for c in range(1, rows):
                key = key * _MULT + part[:, c]
        order = np.argsort(key, kind="stable")
        ks = key[order]
        start = np.r_[True, ks[1:] != ks[:-1]]
        first = order[np.maximum.accumulate(np.where(start, np.arange(n), 0))]
        a, b = order[~start], first[~start]
        for s in range(0, len(a), 1 << 20):
            aa, bb = a[s:s + (1 << 20)], b[s:s + (1 << 20)]
            ok = (sig[aa] == sig[bb]).mean(axis=1) >= thr
            a_all.append(aa[ok])
            b_all.append(bb[ok])
    a = np.concatenate(a_all) if a_all else np.empty(0, np.int64)
    b = np.concatenate(b_all) if b_all else np.empty(0, np.int64)
    g = coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def removable(labels_or_hashes, mask=None):
    """Documents that are a repeat of an earlier one in the same group (= what dedup would drop)."""
    x = labels_or_hashes if mask is None else labels_or_hashes[mask]
    if len(x) == 0:
        return np.zeros(0, bool)
    _, first = np.unique(x, return_index=True)
    rep = np.ones(len(x), bool)
    rep[first] = False
    return rep


def zipf_heaps(counter, heaps):
    f = np.sort(np.fromiter(counter.values(), np.float64))[::-1]
    r = np.arange(1, len(f) + 1)
    sel = (r >= 10) & (r <= 10_000)
    zipf = float(-np.polyfit(np.log(r[sel]), np.log(f[sel]), 1)[0]) if sel.sum() > 10 else None
    if len(heaps) > 5:
        x, y = np.log([h[0] for h in heaps]), np.log([h[1] for h in heaps])
        beta, logk = np.polyfit(x, y, 1)
        return zipf, float(beta), float(np.exp(logk)), f
    return zipf, None, None, f


def vocab_stats(c, heaps):
    tot = sum(c.values())
    if not tot:
        return None
    f = np.sort(np.fromiter(c.values(), np.int64))[::-1]
    cum = np.cumsum(f) / tot
    zipf, beta, k, _ = zipf_heaps(c, heaps)
    low = Counter()
    for w, n in c.items():
        low[w.lower()] += n
    return {"sample_words": int(tot), "types": len(c), "types_lowercased": len(low),
            "type_token_ratio": len(c) / tot, "hapax_share_of_types": float((f == 1).sum() / len(c)),
            "types_for_90pct": int(np.searchsorted(cum, 0.90) + 1), "types_for_95pct": int(np.searchsorted(cum, 0.95) + 1),
            "types_for_99pct": int(np.searchsorted(cum, 0.99) + 1),
            "numeric_types_share": sum(any(ch.isdigit() for ch in w) for w in c) / len(c),
            "nonascii_types_share": sum(not w.isascii() for w in c) / len(c),
            "zipf_slope_rank10_10k": zipf, "heaps_beta": beta, "heaps_K": k,
            "top_words": c.most_common(30)}


def tok_stats(tok, texts):
    words = [w for t in texts for w in t.split()]
    lens = [len(tok._enc._encode_word(w)) for w in words]
    per_text = [len(tok.encode(t).ids) for t in texts]
    h = Counter(min(n, 5) for n in lens)
    return {"words": len(words), "tokens": sum(lens), "fertility": sum(lens) / max(1, len(words)),
            "chars_per_token": sum(map(len, words)) / max(1, sum(lens)),
            "single_token_word_pct": 100 * h[1] / max(1, len(words)),
            "tokens_per_word_hist_pct": {str(k) if k < 5 else "5+": 100 * h[k] / max(1, len(words)) for k in range(1, 6)},
            "tokens_per_text": pct(per_text), "texts_over_768_tokens": int(sum(n > 768 for n in per_text))}


def analyse(args, ref, tok, test, sources, states, slots, best_train, best_any, union_train, seen_train, seen_any):
    meta = ref["meta"]
    out = {"meta": {"tokenizer": os.path.basename(ref["tokenizer"]), "tokenizer_sha": ref["tokenizer_sha"],
                    "dev_set_sha": ref["dev_sha"], "sources": sources, "max_docs": args.max_docs,
                    "batch_docs": args.batch_docs, "val_frac": meta["val_frac"],
                    "settings": {"shingle_words": SHINGLE, "minhash_perms": PERMS, "lsh_bands": BANDS,
                                 "near_dup_jaccard": NEAR_J, "overlap_ngram_words": NGRAM,
                                 "vocab_sample_words_per_source": SAMPLE_WORDS},
                    "quality_flags": {k: v[0] for k, v in QUALITY_FLAGS.items()}}}
    ex = {"flagged": {}, "repeated_lines": {}, "exact_dup_groups": [], "near_dup_clusters": []}

    out["membership_check"], out["size"] = {}, {}
    for src in sources:
        c, m = states[src].cols, meta["sources"].get(src)
        tr = c["flag"] == TRAIN
        rebuilt = {"train_docs": int(tr.sum()), "train_tokens_incl_eos": int((c["tokens"][tr] + 1).sum())}
        if m:
            rebuilt.update(meta_train_docs=m["train"]["docs"], meta_train_tokens=m["train"]["tokens"],
                           exact=rebuilt["train_docs"] == m["train"]["docs"]
                           and rebuilt["train_tokens_incl_eos"] == m["train"]["tokens"])
        out["membership_check"][src] = rebuilt
        ne = c["flag"] != EMPTY
        out["size"][src] = {
            "docs": int(len(c["i"])), "empty_docs": int((~ne).sum()), "words": int(c["words"].sum()),
            "chars": int(c["chars"].sum()), "tokens": int(c["tokens"].sum()),
            "by_membership": {FLAG[f]: {"docs": int((c["flag"] == f).sum()), "words": int(c["words"][c["flag"] == f].sum()),
                                        "tokens": int(c["tokens"][c["flag"] == f].sum())} for f in FLAG},
            "doc_words": pct(c["words"][ne]), "train_doc_words": pct(c["words"][tr]),
            "fertility_full": float(c["tokens"].sum() / max(1, c["words"].sum()))}

    out["quality"] = {}
    for src in sources:
        c = states[src].cols
        ne = c["flag"] != EMPTY
        q, w, tr = c["q"][ne], c["words"][ne], (c["flag"] == TRAIN)[ne]
        snips = [s for s, keep in zip(states[src].snip, ne) if keep]
        res = {"feature_medians": {k: float(np.median(q[:, j])) for j, k in enumerate(QF)} if len(q) else {}}
        for name, (_, test_fn) in QUALITY_FLAGS.items():
            hit = test_fn(q, w) if len(q) else np.zeros(0, bool)
            res[name] = {"docs": int(hit.sum()), "docs_pct": 100 * float(hit.mean()) if len(hit) else 0.0,
                         "words_pct": 100 * float(w[hit].sum() / max(1, w.sum())),
                         "train_docs": int((hit & tr).sum()),
                         "train_words_pct": 100 * float(w[hit & tr].sum() / max(1, w[tr].sum()))}
            ex["flagged"].setdefault(name, {})[src] = [snips[j] for j in np.flatnonzero(hit)[:5]]
        out["quality"][src] = res
        ex["repeated_lines"][src] = states[src].lines.most_common(25)

    src_of = np.concatenate([np.full(len(states[s].cols["i"]), k, np.int8) for k, s in enumerate(sources)])
    cat = {k: np.concatenate([states[s].cols[k] for s in sources]) for k in ("i", "words", "flag", "exact", "sig")}
    snip = [x for s in sources for x in states[s].snip]
    ne = cat["flag"] != EMPTY
    idx = np.flatnonzero(ne)
    labels = np.full(len(ne), -1, np.int64)
    labels[idx] = lsh_components(cat["sig"][idx])
    exact_rep, near_rep = np.zeros(len(ne), bool), np.zeros(len(ne), bool)
    exact_rep[idx], near_rep[idx] = removable(cat["exact"][idx]), removable(labels[idx])
    tr_all = cat["flag"] == TRAIN
    exact_rep_tr, near_rep_tr = np.zeros(len(ne), bool), np.zeros(len(ne), bool)
    t_idx = np.flatnonzero(tr_all)
    exact_rep_tr[t_idx], near_rep_tr[t_idx] = removable(cat["exact"][t_idx]), removable(labels[t_idx])
    dup = {}
    for k, s in enumerate(sources):
        sm = (src_of == k) & ne
        st = (src_of == k) & tr_all
        w = cat["words"]
        dup[s] = {"docs": int(sm.sum()),
                  "exact_dup_docs": int(exact_rep[sm].sum()), "exact_dup_pct": 100 * float(exact_rep[sm].mean()) if sm.any() else 0,
                  "near_dup_docs": int(near_rep[sm].sum()), "near_dup_pct": 100 * float(near_rep[sm].mean()) if sm.any() else 0,
                  "near_dup_words_pct": 100 * float(w[sm & near_rep].sum() / max(1, w[sm].sum())),
                  "train_docs": int(st.sum()),
                  "train_exact_dup_pct": 100 * float(exact_rep_tr[st].mean()) if st.any() else 0,
                  "train_near_dup_pct": 100 * float(near_rep_tr[st].mean()) if st.any() else 0,
                  "train_near_dup_words_pct": 100 * float(w[st & near_rep_tr].sum() / max(1, w[st].sum()))}
    lab = labels[idx]
    comp_src = {}
    for l, s in zip(lab.tolist(), src_of[idx].tolist()):
        comp_src.setdefault(l, set()).add(s)
    sizes = Counter(lab.tolist())
    multi = {l for l, n in sizes.items() if n > 1}
    out["duplicates"] = {"per_source": dup,
                         "clusters_with_2plus_docs": len(multi),
                         "cross_source_clusters": sum(len(comp_src[l]) > 1 for l in multi),
                         "corpus_exact_dup_pct": 100 * float(exact_rep[ne].mean()) if ne.any() else 0,
                         "corpus_near_dup_pct": 100 * float(near_rep[ne].mean()) if ne.any() else 0,
                         "cluster_size_hist": {str(k): v for k, v in sorted(Counter(min(n, 100) for n in sizes.values() if n > 1).items())}}
    pos = {(sources[s], int(i)): j for j, (s, i) in enumerate(zip(src_of.tolist(), cat["i"].tolist()))}
    train_labels = set(labels[t_idx].tolist())
    train_hashes = set(cat["exact"][t_idx].tolist())
    dev_hits = []
    for key, s in ref["sources"].items():
        j = pos.get((s, int(key.rsplit("-", 1)[1])))
        if j is not None:
            dev_hits.append((key, labels[j] in train_labels, int(cat["exact"][j]) in train_hashes))
    out["duplicates"]["dev_docs_checked"] = len(dev_hits)
    out["duplicates"]["dev_docs_with_near_dup_in_train"] = sum(h[1] for h in dev_hits)
    out["duplicates"]["dev_docs_with_exact_copy_in_train"] = sum(h[2] for h in dev_hits)
    for l, n in sizes.most_common(10):
        if n < 2:
            break
        mem = idx[lab == l]
        ex["near_dup_clusters"].append({"size": n, "sources": dict(Counter(sources[s] for s in src_of[mem].tolist())),
                                        "train_members": int(tr_all[mem].sum()),
                                        "examples": [snip[j] for j in mem[:3]]})
    eh = Counter(cat["exact"][idx].tolist())
    for h, n in eh.most_common(10):
        if n < 2:
            break
        j = idx[np.flatnonzero(cat["exact"][idx] == np.uint64(h))[0]]
        ex["exact_dup_groups"].append({"copies": n, "example": snip[j]})
    out["dev_dup_per_doc"] = {k: {"near_dup_in_train": a, "exact_copy_in_train": b} for k, a, b in dev_hits}

    allc, allh = Counter(), []
    out["vocab"] = {}
    for s in sources:
        out["vocab"][s] = vocab_stats(states[s].vocab, states[s].heaps)
        allc.update(states[s].vocab)
    out["vocab"]["all"] = vocab_stats(allc, [])
    top = {s: {w for w, _ in states[s].vocab.most_common(10_000)} for s in sources}
    out["vocab"]["top10k_jaccard"] = {f"{a}|{b}": len(top[a] & top[b]) / max(1, len(top[a] | top[b]))
                                      for a, b in itertools.combinations(sources, 2)}
    oov = {}
    for name, d in (("test_prompts", test), ("dev_prompts", ref["prompts"]), ("dev_truth", ref["truth"])):
        occ = Counter(w for v in d.values() for w in v["text"].split())
        for ref_name, seen in (("train", seen_train), ("corpus", seen_any)):
            miss = [w for w in occ if w not in seen]
            oov[f"{name}_vs_{ref_name}"] = {"types": len(occ), "oov_types": len(miss),
                                            "oov_type_pct": 100 * len(miss) / max(1, len(occ)),
                                            "oov_token_pct": 100 * sum(occ[w] for w in miss) / max(1, sum(occ.values())),
                                            "examples": sorted(miss, key=lambda w: -occ[w])[:20]}
    out["vocab"]["oov"] = oov

    V = tok.get_vocab_size()
    ids = Counter()
    per = {}
    for s in sources:
        st = states[s]
        ids.update(st.tok_ids)
        nw, nt, nc = (int(x) for x in st.sample)
        h = st.tok_hist
        per[s] = {"fertility_sample": nt / max(1, nw), "fertility_full_pass": out["size"][s]["fertility_full"],
                  "chars_per_token": nc / max(1, nt), "single_token_word_pct": 100 * h[1] / max(1, nw),
                  "tokens_per_word_hist_pct": {str(k): 100 * h[k] / max(1, nw) for k in range(1, 5)} |
                  {"5+": 100 * sum(v for k, v in h.items() if k >= 5) / max(1, nw)}}
    itos = tok.itos
    byte_like = {i for i, s in enumerate(itos) if (s.startswith("B") and len(s) == 3) or s == "<EOW>"}
    tot = sum(ids.values())
    counts = np.zeros(V, np.int64)
    for i, n in ids.items():
        counts[i] = n
    n_enc = len(tok._enc.itos)
    piece = lambda i: tok._enc.tokens([i])[0] if i < n_enc else tok.itos[i]
    lens = np.array([len(piece(i).encode("utf-8")) for i in range(V)])
    out["tokenizer"] = {
        "vocab_size": V, "per_source": per,
        "dev_prompts": tok_stats(tok, [v["text"] for v in ref["prompts"].values()]),
        "dev_truth": tok_stats(tok, [v["text"] for v in ref["truth"].values()]),
        "test_prompts": tok_stats(tok, [v["text"] for v in test.values()]),
        "ids_used_in_sample": int((counts > 0).sum()), "ids_used_under_10_times": int(((counts > 0) & (counts < 10)).sum()),
        "ids_never_used": int((counts == 0).sum()),
        "byte_level_token_pct": 100 * sum(ids[i] for i in byte_like) / max(1, tot),
        "top_tokens": [(piece(i), int(n)) for i, n in ids.most_common(30)],
        "longest_tokens": [piece(int(i)) for i in np.argsort(-lens)[:15]],
        "mean_token_bytes_in_vocab": float(lens.mean()),
        "corpus_tokens_total": int(sum(out["size"][s]["tokens"] for s in sources))}
    out["tokenizer"]["_counts"] = counts

    ov, per_ex = {}, {}
    for kind in ("dev_prompt", "dev_truth", "test_prompt"):
        sl = [j for j, s in enumerate(slots) if s["kind"] == kind]
        union = np.array([len(union_train[j]) / max(1, slots[j]["n"]) for j in sl])
        bt = np.array([best_train[j][0] for j in sl])
        ba = np.array([best_any[j][0] for j in sl])
        ov[kind] = {"examples": len(sl), "ngrams_per_example": float(np.mean([slots[j]["n"] for j in sl])),
                    "train_ngram_coverage_mean": float(union.mean()), "train_ngram_coverage_median": float(np.median(union)),
                    "examples_with_any_train_ngram": int((union > 0).sum()),
                    "best_train_doc_coverage": pct(bt),
                    **{f"best_train_doc_ge_{t}": int((bt >= t).sum()) for t in (0.5, 0.8, 0.95)},
                    **{f"best_any_doc_ge_{t}": int((ba >= t).sum()) for t in (0.5, 0.8, 0.95)},
                    "best_any_doc_ge_0.8_by_membership": dict(Counter(best_any[j][3] for j in sl if best_any[j][0] >= 0.8))}
        for j in sl:
            per_ex[f"{kind}:{slots[j]['id']}"] = {"ngrams": slots[j]["n"], "train_coverage": len(union_train[j]) / max(1, slots[j]["n"]),
                                                   "best_train_doc": best_train[j], "best_any_doc": best_any[j]}
    out["overlap"] = ov
    out["_per_example"], out["_examples"] = per_ex, ex
    out["_curves"] = {s: {"heaps": states[s].heaps[::max(1, len(states[s].heaps) // 200)],
                          "zipf": zipf_heaps(states[s].vocab, [])[3] if states[s].vocab else np.zeros(0),
                          "doc_words": states[s].cols["words"][states[s].cols["flag"] != EMPTY]} for s in sources}
    return out


SRC_COLOR = {"acts": "#2a78d6", "high_courts": "#eb6834", "supreme_court": "#1baf7a", "tribunals": "#eda100"}
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]


def _style(plt):
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                         "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})


def figures(res, outdir, sources):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: no figures")
        return []
    _style(plt)
    fd = os.path.join(outdir, "figures")
    os.makedirs(fd, exist_ok=True)
    made = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(fd, name), dpi=150)
        plt.close(fig)
        made.append(name)

    cur = res["_curves"]
    fig, axes = plt.subplots(1, len(sources), figsize=(3 * len(sources), 2.6), sharey=False, squeeze=False)
    for ax, s in zip(axes[0], sources):
        w = cur[s]["doc_words"]
        if len(w):
            ax.hist(np.log10(np.maximum(w, 1)), bins=50, color=SRC_COLOR.get(s, RAMP[2]), edgecolor=SURF, linewidth=0.4)
            ax.axvline(np.log10(np.median(w)), color=INK2, linewidth=1, linestyle="--")
            ax.text(np.log10(np.median(w)), ax.get_ylim()[1] * 0.92, f" median {np.median(w):,.0f}", color=INK2, fontsize=8)
        ax.set_title(s)
        ax.set_xlabel("words per document (log10)")
    axes[0][0].set_ylabel("documents")
    save(fig, "doc_lengths.png")

    tz = res["tokenizer"]
    names = sources + ["dev_prompts", "dev_truth", "test_prompts"]
    vals = [tz["per_source"][s]["fertility_full_pass"] for s in sources] + [tz[k]["fertility"] for k in names[len(sources):]]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    bars = ax.barh(names[::-1], vals[::-1], color=RAMP[2], edgecolor=SURF, linewidth=2, height=0.6)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", color=INK2, fontsize=8)
    ax.set_xlabel("tokens per whitespace word (fertility)")
    ax.set_title("Tokenizer fertility (A-1 BPE, 32,768 ids)")
    ax.grid(axis="y", visible=False)
    save(fig, "fertility.png")

    fig, ax = plt.subplots(figsize=(6, 2.6))
    keys = ["1", "2", "3", "4", "5+"]
    left = np.zeros(len(sources))
    for k, col in zip(keys, RAMP):
        v = np.array([tz["per_source"][s]["tokens_per_word_hist_pct"][k] for s in sources])
        ax.barh(sources[::-1], v[::-1], left=left[::-1], color=col, edgecolor=SURF, linewidth=1.5, height=0.6,
                label=f"{k} token{'s' if k != '1' else ''}")
        left += v
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of words")
    ax.set_title("How many tokens each word is split into")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=8)
    ax.grid(axis="y", visible=False)
    save(fig, "tokens_per_word.png")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    for s in sources:
        z, h = cur[s]["zipf"], cur[s]["heaps"]
        col = SRC_COLOR.get(s, RAMP[2])
        if len(z):
            r = np.unique(np.logspace(0, np.log10(len(z)), 300).astype(int)) - 1
            a1.loglog(r + 1, z[r], color=col, linewidth=2, label=s)
        if h:
            x, y = zip(*h)
            a2.plot(np.array(x) / 1e6, np.array(y) / 1e6, color=col, linewidth=2, label=s)
            a2.text(x[-1] / 1e6, y[-1] / 1e6, f" {s}", color=INK2, fontsize=8, va="center")
    a1.set_xlabel("word rank")
    a1.set_ylabel("frequency")
    a1.set_title("Zipf: word frequency by rank")
    a1.legend(fontsize=8)
    a2.set_xlabel("words read (millions)")
    a2.set_ylabel("distinct words (millions)")
    a2.set_title("Heaps: vocabulary growth")
    save(fig, "zipf_heaps.png")

    d = res["duplicates"]["per_source"]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    x = np.arange(len(sources))
    series = [("exact_dup_pct", "exact copies, corpus", RAMP[0]), ("near_dup_pct", "near-duplicates, corpus", RAMP[2]),
              ("train_near_dup_pct", "near-duplicates, our training subset", RAMP[4])]
    wdt = 0.26
    for k, (key, lab, col) in enumerate(series):
        v = [d[s][key] for s in sources]
        bars = ax.bar(x + (k - 1) * wdt, v, wdt, color=col, edgecolor=SURF, linewidth=2, label=lab)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, sources)
    ax.set_ylabel("% of documents that are repeats")
    ax.set_title(f"Duplicate documents (near = MinHash Jaccard >= {NEAR_J} on {SHINGLE}-word shingles)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, "duplicates.png")

    qn = list(QUALITY_FLAGS)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(qn))
    h = 0.8 / len(sources)
    for k, s in enumerate(sources):
        v = [res["quality"][s][f]["docs_pct"] for f in qn]
        ax.barh(y + k * h - 0.4 + h / 2, v, h, color=SRC_COLOR.get(s, RAMP[2]), edgecolor=SURF, linewidth=1, label=s)
    ax.set_yticks(y, qn)
    ax.invert_yaxis()
    ax.set_xlabel("% of documents flagged")
    ax.set_title("Quality flags (diagnostic thresholds, nothing removed)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="y", visible=False)
    save(fig, "quality_flags.png")

    per = res["_per_example"]
    kinds = [("dev_truth", "dev continuations"), ("dev_prompt", "dev prompts"), ("test_prompt", "test prompts")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, (k, lab) in zip(axes, kinds):
        v = [e["best_train_doc"][0] for key, e in per.items() if key.startswith(k + ":")]
        ax.hist(v, bins=np.linspace(0, 1, 21), color=RAMP[2], edgecolor=SURF, linewidth=1)
        ax.set_title(lab)
        ax.set_xlabel(f"share of its {NGRAM}-grams found in\nthe single most similar training doc")
    axes[0].set_ylabel("examples")
    save(fig, "overlap.png")

    counts = np.sort(res["tokenizer"]["_counts"])[::-1]
    fig, ax = plt.subplots(figsize=(6, 2.6))
    nz = counts[counts > 0]
    ax.loglog(np.arange(1, len(nz) + 1), nz, color=RAMP[2], linewidth=2)
    ax.set_xlabel("token id rank")
    ax.set_ylabel("count in sample")
    ax.set_title(f"Token usage: {len(nz):,} of {len(counts):,} ids seen in the sample")
    save(fig, "token_usage.png")
    return made


def _f(x, nd=2):
    return "n/a" if x is None else (f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}")


def report_md(res, sources, figs):
    L = ["# Phase-1 corpus audit", "",
         f"Tokenizer `{res['meta']['tokenizer']}` (sha {res['meta']['tokenizer_sha']}), dev set sha "
         f"{res['meta']['dev_set_sha']}, runtime {res['meta'].get('seconds', 0) / 60:.0f} min."
         + (f" **Partial run: first {res['meta']['max_docs']:,} docs per source.**" if res["meta"]["max_docs"] else ""), ""]
    L += ["## Training subset rebuilt from round-1 meta.json", "",
          "| source | rebuilt docs | logged docs | rebuilt tokens | logged tokens | exact |", "|---|---|---|---|---|---|"]
    for s in sources:
        m = res["membership_check"][s]
        L.append(f"| {s} | {m['train_docs']:,} | {_f(m.get('meta_train_docs'))} | {m['train_tokens_incl_eos']:,} | "
                 f"{_f(m.get('meta_train_tokens'))} | {m.get('exact', 'n/a')} |")
    L += ["", "## Size", "", "| source | docs | empty | words | tokens | fertility | train docs | train tokens | median words/doc | p99 words/doc |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in sources:
        z = res["size"][s]
        tr = z["by_membership"]["train"]
        L.append(f"| {s} | {z['docs']:,} | {z['empty_docs']:,} | {z['words']:,} | {z['tokens']:,} | {z['fertility_full']:.3f} | "
                 f"{tr['docs']:,} | {tr['tokens']:,} | {_f(z['doc_words'] and z['doc_words']['p50'], 0)} | "
                 f"{_f(z['doc_words'] and z['doc_words']['p99'], 0)} |")
    L += ["", "## Tokenizer", "", "| text | fertility (tokens/word) | chars/token | 1-token words % |", "|---|---|---|---|"]
    tz = res["tokenizer"]
    for s in sources:
        p = tz["per_source"][s]
        L.append(f"| {s} (sample) | {p['fertility_sample']:.3f} | {p['chars_per_token']:.2f} | {p['single_token_word_pct']:.1f} |")
    for k in ("dev_prompts", "dev_truth", "test_prompts"):
        p = tz[k]
        L.append(f"| {k} | {p['fertility']:.3f} | {p['chars_per_token']:.2f} | {p['single_token_word_pct']:.1f} |")
    L += ["", f"- Vocabulary {tz['vocab_size']:,} ids; {tz['ids_used_in_sample']:,} seen in the sample, "
          f"{tz['ids_used_under_10_times']:,} seen fewer than 10 times, {tz['ids_never_used']:,} never.",
          f"- Byte-level (single-byte or bare `<EOW>`) tokens: {tz['byte_level_token_pct']:.2f}% of sampled tokens.",
          f"- Test prompts: {_f(tz['test_prompts']['tokens_per_text']['p50'], 0)} tokens median, "
          f"{tz['test_prompts']['texts_over_768_tokens']} of {res['overlap']['test_prompt']['examples']} over 768 tokens "
          "(the decode configs keep the last 768).",
          f"- Whole corpus: {tz['corpus_tokens_total']:,} tokens.", ""]
    L += ["## Vocabulary (whitespace words, equal-size sample per source)", "",
          "| source | sample words | types | hapax % of types | types for 95% of text | Zipf slope | Heaps beta | numeric types % | non-ASCII types % |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in sources + ["all"]:
        v = res["vocab"].get(s)
        if v:
            L.append(f"| {s} | {v['sample_words']:,} | {v['types']:,} | {100 * v['hapax_share_of_types']:.1f} | {v['types_for_95pct']:,} | "
                     f"{_f(v['zipf_slope_rank10_10k'])} | {_f(v['heaps_beta'])} | {100 * v['numeric_types_share']:.1f} | "
                     f"{100 * v['nonascii_types_share']:.1f} |")
    L += ["", "| words of | OOV types % vs training docs | OOV tokens % vs training docs | OOV tokens % vs whole corpus |", "|---|---|---|---|"]
    for k in ("test_prompts", "dev_prompts", "dev_truth"):
        a, b = res["vocab"]["oov"][f"{k}_vs_train"], res["vocab"]["oov"][f"{k}_vs_corpus"]
        L.append(f"| {k} | {a['oov_type_pct']:.2f} | {a['oov_token_pct']:.3f} | {b['oov_token_pct']:.3f} |")
    L += ["", "## Quality flags (% of documents; nothing was removed)", "",
          "| flag | " + " | ".join(sources) + " | meaning |", "|---|" + "---|" * len(sources) + "---|"]
    for f, (desc, _) in QUALITY_FLAGS.items():
        L.append(f"| {f} | " + " | ".join(f"{res['quality'][s][f]['docs_pct']:.2f}" for s in sources) + f" | {desc} |")
    d = res["duplicates"]
    L += ["", "## Duplicates", "", f"Near-duplicate = MinHash-estimated Jaccard >= {NEAR_J} on {SHINGLE}-word shingles "
          f"({PERMS} permutations, {BANDS} LSH bands). A 'repeat' is every member of a group except the first, i.e. what "
          "deduplication would remove.", "",
          "| source | docs | exact repeats % | near repeats % | near repeats % of words | train: exact % | train: near % |",
          "|---|---|---|---|---|---|---|"]
    for s in sources:
        x = d["per_source"][s]
        L.append(f"| {s} | {x['docs']:,} | {x['exact_dup_pct']:.2f} | {x['near_dup_pct']:.2f} | {x['near_dup_words_pct']:.2f} | "
                 f"{x['train_exact_dup_pct']:.2f} | {x['train_near_dup_pct']:.2f} |")
    L += ["", f"- Whole corpus: {d['corpus_exact_dup_pct']:.2f}% exact repeats, {d['corpus_near_dup_pct']:.2f}% near repeats; "
          f"{d['clusters_with_2plus_docs']:,} groups, {d['cross_source_clusters']:,} spanning more than one source.",
          f"- Dev documents with a near-duplicate among the training documents: {d['dev_docs_with_near_dup_in_train']} of "
          f"{d['dev_docs_checked']} (exact copies: {d['dev_docs_with_exact_copy_in_train']}).", ""]
    L += [f"## Dev / test overlap with the training documents ({NGRAM}-word n-grams)", "",
          "| text | examples | mean share of n-grams seen in training | best single training doc >= 50% | >= 80% | >= 95% | any corpus doc >= 95% |",
          "|---|---|---|---|---|---|---|"]
    for k, o in res["overlap"].items():
        L.append(f"| {k} | {o['examples']} | {o['train_ngram_coverage_mean']:.3f} | {o['best_train_doc_ge_0.5']} | "
                 f"{o['best_train_doc_ge_0.8']} | {o['best_train_doc_ge_0.95']} | {o['best_any_doc_ge_0.95']} |")
    L += ["", "Where the best match (>= 80%) of each kind lives: " + "; ".join(
        f"{k}: {o['best_any_doc_ge_0.8_by_membership']}" for k, o in res["overlap"].items()), ""]
    L += ["## Figures", ""] + [f"![{f}](figures/{f})" for f in figs]
    return "\n".join(L) + "\n"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--ref", required=True, help="round-1 tokenizer/, dev/ and shards/meta.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", nargs="+", default=["acts", "supreme_court", "tribunals", "high_courts"],
                    help="processing order (smallest first, so a failure shows early)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--batch-docs", type=int, default=1024, help="make_shards --batch-docs used for the shards (prep: 1024)")
    ap.add_argument("--max-docs", type=int, default=0, help="first N documents per source only (smoke test)")
    ap.add_argument("--download", action="store_true", help="download missing files from Hugging Face (HF_TOKEN)")
    ap.add_argument("--delete-raw", action="store_true", help="delete each train file after its pass (saves disk)")
    args = ap.parse_args()

    res = run_audit(args)
    os.makedirs(args.out, exist_ok=True)
    srcs = res["meta"]["sources"]
    figs = figures(res, args.out, srcs)
    with open(os.path.join(args.out, "audit.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(res), f, indent=1, ensure_ascii=False)
    with open(os.path.join(args.out, "examples.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(res["_examples"]), f, indent=1, ensure_ascii=False)
    with open(os.path.join(args.out, "per_example_overlap.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(res["_per_example"]), f, indent=1, ensure_ascii=False)
    md = report_md(res, srcs, figs)
    with open(os.path.join(args.out, "audit.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"wrote {args.out}: audit.json, audit.md, examples.json, per_example_overlap.json, figures/ ({len(figs)})")


if __name__ == "__main__":
    main()
