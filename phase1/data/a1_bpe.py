"""Assignment-1 byte-level BPE, used as the shared tokenizer for all Phase-1 models.

`ByteBPE` (A-1, unchanged), `_fast_fit` (same merges as `ByteBPE.fit`, incremental counts)
and `A1BPETokenizer` are copied from CS781 Assm-3/a1_tokenizer.py. The only addition to
`_fast_fit` is optional progress logging. `A1Tokenizer` at the bottom wraps them in the small
part of the HF `tokenizers.Tokenizer` interface the Phase-1 pipeline uses:

    tok = A1Tokenizer.train(texts, vocab_size)        # texts may be a generator
    tok.save("bpe.json"); tok = A1Tokenizer.from_file("bpe.json")
    tok.encode(text).ids, tok.encode_batch(texts), tok.decode(ids)
    tok.get_vocab_size(), tok.token_to_id(s), tok.id_to_token(i)

Differences from the HF byte-level BPE:
  * Words come from str.split(), so whitespace layout (newlines, double spaces) is not kept:
    decode(encode(t)) == " ".join(t.split()). The Phase-1 scorer splits on whitespace too.
  * Special ids: <pad>=0, <unk>=1, <s>=2 (BOS / decoder start), </s>=3 (document separator, EOS).
    <mask> and <extra_0..11> are appended after the merges for MLM and later phases.
  * All 256 byte symbols are placed in the vocabulary before fitting, so a byte that is
    missing from the fitting sample is still encodable (never <unk>). Symbols with zero
    count never take part in a merge, so the learned merges are the same as plain A-1.
"""
import heapq
import json
import multiprocessing as mp
import time
from collections import Counter, defaultdict

class ByteBPE:
    """
    Minimal byte-level BPE tokenizer implemented from scratch (Assignment 1).

    Each word starts as UTF-8 byte symbols plus an explicit <EOW> marker.
    Merges are learned only inside a word, so BPE can never cross a word boundary.
    """

    EOW = "<EOW>"

    def __init__(self, vocab_size=5000):
        self.target_vocab_size = vocab_size
        self.specials = ["<pad>", "<unk>", "<s>", "</s>"]
        self.itos = list(self.specials)
        self.merges = []
        self._merge_map = {}

    @staticmethod
    def _byte_symbols(word):
        return [f"B{b:02X}" for b in word.encode("utf-8")] + [ByteBPE.EOW]

    @staticmethod
    def _merged_symbol(a, b):
        return f"M[{a}|{b}]"

    def _apply_merge(self, seq, pair, merged):
        out = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and (seq[i], seq[i+1]) == pair:
                out.append(merged)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    def fit(self, texts):
        word_counts = Counter()
        for text in texts:
            word_counts.update(str(text).split())

        sequences = [self._byte_symbols(word) for word in word_counts]
        counts = list(word_counts.values())

        symbols = set()
        for seq in sequences:
            symbols.update(seq)
        for s in sorted(symbols):
            if s not in self.itos:
                self.itos.append(s)

        while len(self.itos) < self.target_vocab_size:
            pair_counts = Counter()
            for seq, count in zip(sequences, counts):
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i+1])] += count

            if not pair_counts:
                break

            best_pair, best_count = min(
                pair_counts.items(),
                key=lambda kv: (-kv[1], kv[0])
            )
            if best_count <= 1:
                break

            merged = self._merged_symbol(*best_pair)
            self.merges.append([best_pair[0], best_pair[1], merged])
            self.itos.append(merged)
            self._merge_map[tuple(best_pair)] = merged

            sequences = [
                self._apply_merge(seq, best_pair, merged)
                for seq in sequences
            ]

        self.vocab_size = len(self.itos)
        return self

    def encode_word(self, word):
        seq = self._byte_symbols(word)
        for a, b, merged in self.merges:
            seq = self._apply_merge(seq, (a, b), merged)
        return seq

    def encode_sentence(self, sentence):
        return [tok for word in str(sentence).split() for tok in self.encode_word(word)]

    def tokenize_corpus(self, texts):
        return [self.encode_sentence(text) for text in texts]

    def to_json_dict(self):
        return {
            "merges": self.merges,
            "itos": self.itos,
            "vocab_size": self.vocab_size,
        }


def _fast_fit(bpe: ByteBPE, texts, log_every=0) -> ByteBPE:
    """Same result as `bpe.fit(texts)`, with incremental pair counting."""
    t0 = time.time()
    word_counts = Counter()
    for text in texts:
        word_counts.update(str(text).split())

    words = [bpe._byte_symbols(word) for word in word_counts]
    counts = list(word_counts.values())

    symbols = set()
    for seq in words:
        symbols.update(seq)
    for s in sorted(symbols):
        if s not in bpe.itos:
            bpe.itos.append(s)

    pair_counts = defaultdict(int)
    where = defaultdict(set)
    for wi, (seq, c) in enumerate(zip(words, counts)):
        for p in zip(seq, seq[1:]):
            pair_counts[p] += c
            where[p].add(wi)
    heap = [(-c, a, b) for (a, b), c in pair_counts.items()]
    heapq.heapify(heap)

    while len(bpe.itos) < bpe.target_vocab_size:
        best = None
        while heap:
            negc, a, b = heap[0]
            cur = pair_counts.get((a, b), 0)
            if cur > 0 and cur == -negc:
                best = (a, b)
                break
            heapq.heappop(heap)
        if best is None:
            break
        best_count = pair_counts[best]
        if best_count <= 1:
            break
        heapq.heappop(heap)
        if log_every and len(bpe.merges) % log_every == 0:
            print(f"  merges={len(bpe.merges):,} vocab={len(bpe.itos):,} best_count={best_count:,} "
                  f"({time.time() - t0:.0f}s)", flush=True)

        merged = bpe._merged_symbol(*best)
        bpe.merges.append([best[0], best[1], merged])
        bpe.itos.append(merged)
        bpe._merge_map[best] = merged

        touched = set()
        for wi in where.pop(best, ()):
            seq, c = words[wi], counts[wi]
            new = bpe._apply_merge(seq, best, merged)
            if len(new) == len(seq):
                continue
            for p in zip(seq, seq[1:]):
                pair_counts[p] -= c
                touched.add(p)
            for p in zip(new, new[1:]):
                pair_counts[p] += c
                where[p].add(wi)
                touched.add(p)
            words[wi] = new
        for p in touched:
            c = pair_counts[p]
            if c > 0:
                heapq.heappush(heap, (-c, p[0], p[1]))
            else:
                pair_counts.pop(p, None)

    bpe.vocab_size = len(bpe.itos)
    return bpe


class A1BPETokenizer:
    """encode/decode/vocab_size interface required by A-3 around a fitted ByteBPE."""

    UNK_ID = 1

    def __init__(self, bpe: ByteBPE):
        self.bpe = bpe
        self.itos = list(bpe.itos)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
        self._ranks = {(a, b): r for r, (a, b, _m) in enumerate(bpe.merges)}
        self._merged = {(a, b): m for a, b, m in bpe.merges}
        self._bytes = self._build_symbol_bytes()
        self._cache = {}

    def _build_symbol_bytes(self):
        table = {}
        for s in self.itos:
            if s == ByteBPE.EOW:
                table[s] = b" "
            elif len(s) == 3 and s[0] == "B":
                table[s] = bytes([int(s[1:], 16)])
            elif s in self.bpe.specials:
                table[s] = b""
        for a, b, m in self.bpe.merges:
            table[m] = table[a] + table[b]
        return table

    def get_vocab_size(self):
        return self.vocab_size

    def _encode_word(self, word):
        ids = self._cache.get(word)
        if ids is not None:
            return ids
        seq = ByteBPE._byte_symbols(word)
        while len(seq) > 1:
            ranked = [(self._ranks.get(p, None), i) for i, p in enumerate(zip(seq, seq[1:]))]
            ranked = [(r, i) for r, i in ranked if r is not None]
            if not ranked:
                break
            r = min(ranked)[0]
            pair = (self.bpe.merges[r][0], self.bpe.merges[r][1])
            seq = self.bpe._apply_merge(seq, pair, self._merged[pair])
        ids = [self.stoi.get(s, self.UNK_ID) for s in seq]
        self._cache[word] = ids
        return ids

    def encode(self, text):
        out = []
        for word in str(text).split():
            out.extend(self._encode_word(word))
        return out

    def decode(self, ids):
        buf = b"".join(self._bytes.get(self.itos[int(i)], b"") for i in ids
                       if 0 <= int(i) < self.vocab_size)
        text = buf.decode("utf-8", errors="replace")
        return text[:-1] if text.endswith(" ") else text

    def tokens(self, ids):
        """Human-readable pieces (for the highest-loss-token and embedding analyses)."""
        return [self._bytes.get(self.itos[int(i)], b"").decode("utf-8", errors="replace")
                .replace(" ", "_") or self.itos[int(i)] for i in ids]


EXTRA_SPECIALS = ["<mask>"] + [f"<extra_{i}>" for i in range(12)]
FORMAT = "a1-bytebpe"


class Encoding:
    __slots__ = ("ids",)

    def __init__(self, ids):
        self.ids = ids


class _Encoder(A1BPETokenizer):
    """A1BPETokenizer with a bounded word cache (the corpus has tens of millions of word types)."""

    MAX_CACHE = 2_000_000

    def _encode_word(self, word):
        if len(self._cache) >= self.MAX_CACHE:
            self._cache.clear()
        return super()._encode_word(word)


def _seeded_bpe(vocab_size):
    bpe = ByteBPE(vocab_size=vocab_size)
    all_bytes = [f"B{b:02X}" for b in range(256)] + [ByteBPE.EOW]
    bpe.itos.extend(sorted(all_bytes))
    return bpe


_worker = None


def _worker_init(state):
    global _worker
    _worker = A1Tokenizer._from_state(state)


def _worker_encode(texts):
    return [_worker._enc.encode(t) for t in texts]


class A1Tokenizer:
    def __init__(self, bpe):
        self.bpe = bpe
        self._enc = _Encoder(bpe)
        self.itos = list(bpe.itos) + EXTRA_SPECIALS
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad_id, self.unk_id = self.stoi["<pad>"], self.stoi["<unk>"]
        self.bos_id, self.eos_id = self.stoi["<s>"], self.stoi["</s>"]
        self.mask_id = self.stoi["<mask>"]
        self._pool, self._pool_size = None, 0

    @classmethod
    def train(cls, texts, vocab_size, log_every=0):
        return cls(_fast_fit(_seeded_bpe(vocab_size), texts, log_every=log_every))

    def _state(self):
        return {"format": FORMAT, "merges": self.bpe.merges, "itos": self.bpe.itos,
                "vocab_size": self.bpe.vocab_size, "target_vocab_size": self.bpe.target_vocab_size,
                "extra_specials": EXTRA_SPECIALS}

    @classmethod
    def _from_state(cls, d):
        if d.get("format") != FORMAT:
            raise ValueError(f"not an A-1 BPE file (format={d.get('format')!r})")
        if d.get("extra_specials", EXTRA_SPECIALS) != EXTRA_SPECIALS:
            raise ValueError("extra special tokens differ from this version of a1_bpe.py")
        bpe = ByteBPE(vocab_size=d["target_vocab_size"])
        bpe.itos = list(d["itos"])
        bpe.merges = [list(m) for m in d["merges"]]
        bpe._merge_map = {(a, b): m for a, b, m in bpe.merges}
        bpe.vocab_size = d["vocab_size"]
        return cls(bpe)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._state(), f)

    @classmethod
    def from_file(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls._from_state(json.load(f))

    def get_vocab_size(self):
        return len(self.itos)

    def token_to_id(self, token):
        return self.stoi.get(token)

    def id_to_token(self, i):
        return self.itos[i] if 0 <= i < len(self.itos) else None

    def encode(self, text):
        return Encoding(self._enc.encode(text))

    def encode_batch(self, texts, workers=1, chunk=64):
        """workers > 1 encodes in a process pool (kept open between calls; call close() when done)."""
        if workers <= 1 or len(texts) < 2 * chunk:
            return [Encoding(self._enc.encode(t)) for t in texts]
        if self._pool is None or self._pool_size != workers:
            self.close()
            self._pool = mp.get_context().Pool(workers, initializer=_worker_init, initargs=(self._state(),))
            self._pool_size = workers
        parts = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
        return [Encoding(ids) for part in self._pool.map(_worker_encode, parts) for ids in part]

    def decode(self, ids):
        ids = [int(i) for i in ids if int(i) < self._enc.vocab_size]
        return self._enc.decode(ids)

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_pool"], d["_pool_size"] = None, 0
        return d


def load_tokenizer(path):
    """A-1 BPE file -> A1Tokenizer; anything else -> HF tokenizers.Tokenizer."""
    with open(path, encoding="utf-8") as f:
        head = f.read(64)
    if f'"format": "{FORMAT}"' in head:
        return A1Tokenizer.from_file(path)
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def special_ids(tok):
    """{eos, bos, pad} ids for either tokenizer family."""
    if isinstance(tok, A1Tokenizer):
        return {"eos": tok.eos_id, "bos": tok.bos_id, "pad": tok.pad_id}
    return {"eos": tok.token_to_id("<|endoftext|>"), "bos": tok.token_to_id("<s>"), "pad": tok.token_to_id("<pad>")}
