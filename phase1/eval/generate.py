"""Generate 128-word continuations for {id: {"text": prompt}} with any Phase-1 model.

The driver talks to a model through a two-method "decoder API":

    logits[B,V], state = api.prefill(prompt_ids[B,T])     # read the whole prompt
    logits[B,V], state = api.step(state, next_ids[B])     # feed one new token per row

Write a fast prefill/step for each architecture (KV cache for attention, recurrent state for
RetNet/Mamba, cached encoder output for the encoder-decoder). Until then, the no-cache
adapters below work with any model by re-running the full forward pass every step (slow,
but useful to check that the cached version gives the same text).

    from phase1.eval.generate import CausalNoCache, DecodeConfig, generate
    preds = generate(prompts, tokenizer, CausalNoCache(model), DecodeConfig(...), max_prompt_tokens=...)

The DecodeConfig defaults are plain sampling.
Prompts are sorted by length and each batch is cut (from the left, keeping the end of the
prompt) to the shortest prompt in the batch, so no padding is needed; the number of
dropped prompt tokens is reported.
"""
import dataclasses
import re
import time
from collections import Counter

import torch

from ..data.a1_bpe import special_ids
from .submission import to_prediction_text

BOS_ID = 2


@dataclasses.dataclass
class DecodeConfig:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram: int = 0
    suppress_eos: bool = False
    max_new_tokens: int = 384
    target_words: int = 128
    seed: int = 0
    mbr_samples: int = 0
    mbr_utility: str = "unigram_f1"


class CausalNoCache:
    """Any causal model with forward(ids[B,T]) -> logits[B,T,V]."""
    def __init__(self, forward):
        self.forward = forward

    def prefill(self, ids):
        return self.forward(ids)[:, -1], ids

    def step(self, ids, nxt):
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        return self.forward(ids)[:, -1], ids


class EncDecNoCache:
    """Any encoder-decoder with forward(src[B,S], tgt[B,T]) -> logits[B,T,V]; the prompt is the source."""
    def __init__(self, forward):
        self.forward = forward

    def prefill(self, src):
        tgt = torch.full((src.shape[0], 1), BOS_ID, dtype=src.dtype, device=src.device)
        return self.forward(src, tgt)[:, -1], (src, tgt)

    def step(self, state, nxt):
        src, tgt = state
        tgt = torch.cat([tgt, nxt[:, None]], dim=1)
        return self.forward(src, tgt)[:, -1], (src, tgt)


def _banned_ngram_tokens(gen, n):
    """For each row, tokens that would complete an n-gram already present in gen[row]."""
    out = []
    for row in gen:
        banned = set()
        if len(row) >= n:
            prefix = tuple(row[-(n - 1):]) if n > 1 else ()
            for i in range(len(row) - n + 1):
                if tuple(row[i:i + n - 1]) == prefix:
                    banned.add(row[i + n - 1])
        out.append(banned)
    return out


def sample_next(logits, gen, cfg, generator, eos_id=None):
    """logits[B,V] -> next ids[B], applying the configured penalties and filters."""
    logits = logits.float().clone()
    if cfg.suppress_eos and eos_id is not None:
        logits[:, eos_id] = float("-inf")
    if cfg.repetition_penalty != 1.0:
        for b, row in enumerate(gen):
            if row:
                idx = torch.tensor(sorted(set(row)), device=logits.device)
                vals = logits[b, idx]
                logits[b, idx] = torch.where(vals > 0, vals / cfg.repetition_penalty, vals * cfg.repetition_penalty)
    if cfg.no_repeat_ngram > 0:
        for b, banned in enumerate(_banned_ngram_tokens(gen, cfg.no_repeat_ngram)):
            if banned:
                logits[b, list(banned)] = float("-inf")
    if cfg.temperature == 0:
        return logits.argmax(-1)
    logits = logits / cfg.temperature
    if cfg.top_k > 0:
        kth = torch.topk(logits, min(cfg.top_k, logits.size(-1)), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if cfg.top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(-1)
        drop = probs.cumsum(-1) - probs > cfg.top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    return torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)


_WORD = re.compile(r"\S+")


def _unigram_f1(a, b):
    """Clipped bag-of-words F1 between two word lists."""
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    common = sum((ca & cb).values())
    if common == 0:
        return 0.0
    p, r = common / len(a), common / len(b)
    return 2 * p * r / (p + r)


def _rouge_l(a, b):
    """ROUGE-L F1 (longest common subsequence) between two word lists."""
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def mbr_select(candidates, utility="unigram_f1"):
    """Minimum Bayes risk: the candidate with the highest mean agreement with the others."""
    f = {"unigram_f1": _unigram_f1, "rougeL": _rouge_l}[utility]
    words = [c.split() for c in candidates]
    n = len(words)
    score = [0.0] * n
    for i in range(n):
        for j in range(i + 1, n):
            u = f(words[i], words[j])
            score[i] += u
            score[j] += u
    best = max(range(n), key=lambda i: (score[i], len(words[i])))
    return candidates[best], score[best] / max(1, n - 1)


def generate(prompts, tokenizer, api, cfg, max_prompt_tokens, batch_size=16, device=None, amp=True,
             check_every=16, log_every=5, eos_id=None):
    """Like _generate; with cfg.mbr_samples > 1, every prompt is sampled mbr_samples times (the
    copies share a batch, batch_size counts rows) and the most central candidate is kept (MBR)."""
    n = cfg.mbr_samples
    if n <= 1:
        return _generate(prompts, tokenizer, api, cfg, max_prompt_tokens, batch_size, device, amp,
                         check_every, log_every, eos_id)
    if cfg.temperature == 0:
        raise ValueError("MBR needs sampling (temperature > 0)")
    expanded = {f"{i}#mbr{k}": p for i, p in prompts.items() for k in range(n)}
    outs, stats = _generate(expanded, tokenizer, api, cfg, max_prompt_tokens, batch_size, device, amp,
                            check_every, log_every * n, eos_id)
    t0, agree, chosen = time.time(), [], {}
    for i in prompts:
        text, a = mbr_select([outs[f"{i}#mbr{k}"]["text"] for k in range(n)], cfg.mbr_utility)
        chosen[i] = {"text": text}
        agree.append(a)
    stats = dict(stats, mbr_samples=n, mbr_mean_agreement=sum(agree) / len(agree),
                 mbr_select_seconds=time.time() - t0,
                 short_outputs=sum(len(v["text"].split()) < cfg.target_words for v in chosen.values()))
    print({k: stats[k] for k in ("mbr_samples", "mbr_mean_agreement", "mbr_select_seconds", "short_outputs")})
    return chosen, stats


@torch.no_grad()
def _generate(prompts, tokenizer, api, cfg, max_prompt_tokens, batch_size=16, device=None, amp=True,
              check_every=16, log_every=5, eos_id=None):
    """Return {id: {"text": continuation of exactly <= target_words words}} plus generation stats.

    A row stops at the tokenizer's EOS / document separator (A-1 BPE: </s> = 3) unless eos_id
    is given.
    """
    if eos_id is None:
        eos_id = special_ids(tokenizer)["eos"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    ids = list(prompts)
    encs = {i: tokenizer.encode(prompts[i]["text"]).ids for i in ids}
    ids.sort(key=lambda i: len(encs[i]))
    out, dropped, short, t0 = {}, 0, 0, time.time()
    for bi in range(0, len(ids), batch_size):
        batch_ids = ids[bi:bi + batch_size]
        keep = min(min(len(encs[i]) for i in batch_ids), max_prompt_tokens)
        dropped += sum(len(encs[i]) - keep for i in batch_ids)
        x = torch.tensor([encs[i][-keep:] for i in batch_ids], device=device)
        gen = [[] for _ in batch_ids]
        done = [False] * len(batch_ids)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
            logits, state = api.prefill(x)
            for t in range(cfg.max_new_tokens):
                nxt = sample_next(logits, gen, cfg, generator, eos_id)
                for b, tok in enumerate(nxt.tolist()):
                    if not done[b]:
                        if tok == eos_id:
                            done[b] = True
                        else:
                            gen[b].append(tok)
                if (t + 1) % check_every == 0:
                    for b in range(len(gen)):
                        if not done[b] and len(_WORD.findall(tokenizer.decode(gen[b]))) > cfg.target_words:
                            done[b] = True
                if all(done):
                    break
                logits, state = api.step(state, nxt)
        for b, i in enumerate(batch_ids):
            text = to_prediction_text(tokenizer.decode(gen[b]), cfg.target_words)
            short += len(text.split()) < cfg.target_words
            out[i] = {"text": text}
        if (bi // batch_size) % log_every == 0:
            print(f"generated {min(bi + batch_size, len(ids))}/{len(ids)} ({time.time() - t0:.0f}s)")
    stats = {"prompt_tokens_dropped": dropped, "short_outputs": short, "seconds": time.time() - t0}
    print(stats)
    return out, stats
