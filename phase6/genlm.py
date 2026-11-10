"""Phase-1 language models as summarisers: training logits on summary positions only, and generation.

    g = load_genlm("B", roots)            # full LM (B keeps its decoder), RoPE caches extended
    apply_method(g, "lora", ...)          # ft.peft works on it (layers() = encoder + decoder for B)
    logits = g.target_logits(batch)       # (N_target_tokens, V) for the loss
    texts  = generate(g, inputs, ...)     # greedy / sampling with length control

Input formats (all share the A-1 BPE):
  decoder-only (A, As, C RetNet, D Mamba-2):  [document tokens] [SEP] [summary] [EOS]; loss on summary+EOS
  encoder-decoder (B):                        encoder <- document tokens; decoder <- [BOS] summary -> summary [EOS]
SEP = <extra_0>, EOS = </s> (3), BOS = <s> (2). Sequences in a batch are right-padded (causal models do not
look right); B's encoder inputs are grouped by length so the bidirectional encoder never sees padding.

Generation uses each family's cached prefill/step (KV cache, recurrent state, encoder memory computed once),
rows with equal prompt length share a batch, and per-row min/max new-token limits set the length.
Adapters (Houlsby / Pfeiffer) are only allowed on A / As / B: the recurrent prefill/step paths of RetNet and
Mamba call sublayer methods that an adapter wrapper does not see.
"""
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ft.backbones import build_lm, family_of, find_checkpoint, load_state, tiny_lm
from phase1.eval.generate import DecodeConfig, sample_next
from phase1.models.layers import build_rope_cache

EOS, BOS, PAD = 3, 2, 0
SEP = 32756
BANNED = list(range(32755, 32768)) + [PAD, BOS, 1]


class GenLM(nn.Module):
    def __init__(self, lm, key="?", name="?", max_len=4096):
        super().__init__()
        self.lm, self.key, self.name = lm, key, name
        self.family = family_of(lm)
        if self.family not in ("decoder", "encdec", "retnet", "mamba"):
            raise ValueError(f"{self.family} cannot generate summaries here (no decoder)")
        self.d_model = lm.cfg.d_model
        self.prompt = self.deep_prompt = None
        if hasattr(lm, "rope_cos") and lm.rope_cos.shape[0] < max_len:
            cos, sin = build_rope_cache(max_len, lm.rope_cos.shape[1], getattr(lm.cfg, "rope_base", 10000.0))
            lm.rope_cos, lm.rope_sin = cos, sin

    @property
    def is_encdec(self):
        return self.family == "encdec"

    def layers(self):
        return list(self.lm.encoder) + list(self.lm.decoder) if self.is_encdec else list(self.lm.blocks)

    def final_norm(self):
        return self.lm.dec_norm if self.is_encdec else self.lm.norm

    def embedding(self):
        return self.lm.tok_emb

    def _hidden(self, ids):
        lm = self.lm
        x = lm.tok_emb(ids)
        cs = (lm.rope_cos, lm.rope_sin) if hasattr(lm, "rope_cos") else None
        for blk in lm.blocks:
            x = blk(x) if self.family == "mamba" else blk(x, cs)
        return lm.norm(x)

    def _vocab(self, h):
        return self.lm.lm_head(h)[..., :self.lm.cfg.vocab_size]

    def target_logits(self, batch):
        """batch from make_batch -> (logits (N, V) float, targets (N,)) over all target positions."""
        if self.is_encdec:
            lm = self.lm
            memory = lm.encode(batch["src"])
            x = lm.tok_emb(batch["tgt_in"])
            cs = (lm.rope_cos, lm.rope_sin)
            for blk in lm.decoder:
                x = blk(x, memory, cs)
            h = lm.dec_norm(x)
        else:
            h = self._hidden(batch["ids"])
        m = batch["mask"]
        return self._vocab(h[m]).float(), batch["targets"][m]

    def api(self):
        from phase1.models.decoder import DecoderAPI
        from phase1.models.encdec import EncDecAPI
        from phase1.models.mamba2 import Mamba2API
        from phase1.models.retnet import RetNetAPI
        return {"decoder": DecoderAPI, "encdec": EncDecAPI, "retnet": RetNetAPI, "mamba": Mamba2API}[self.family](self.lm)


def load_genlm(key, roots=(), random_init=False, tiny=False, max_len=4096):
    lm, name = tiny_lm(key) if tiny else build_lm(key)
    src = None
    if not (random_init or tiny):
        src = find_checkpoint(name, list(roots))
        if src is None:
            raise FileNotFoundError(f"no checkpoint for {key} ({name}) under {list(roots)}")
        missing, unexpected = lm.load_state_dict(load_state(src), strict=False)
        missing = [k for k in missing if not k.endswith(("rope_cos", "rope_sin"))]
        if missing or unexpected:
            raise RuntimeError(f"{name}: missing {missing[:5]} unexpected {unexpected[:5]}")
    g = GenLM(lm, key, name, max_len)
    g.ckpt_path = src
    return g


def make_batch(family, pairs, device, max_target):
    """pairs: list of (input tokens int64, target tokens int64) -> tensors for target_logits.
    Targets are cut to max_target tokens (+ EOS only when the summary was not cut)."""
    rows = []
    for x, y in pairs:
        cut = len(y) > max_target
        y = y[:max_target]
        rows.append((x, np.concatenate([y, [] if cut else [EOS]]).astype(np.int64)))
    if family == "encdec":
        S, T = max(len(x) for x, _ in rows), max(len(y) for _, y in rows)
        if len({len(x) for x, _ in rows}) > 1:
            raise ValueError("encoder inputs in one batch must have equal length (group them first)")
        src = torch.tensor(np.stack([x for x, _ in rows]), device=device)
        tin = torch.full((len(rows), T), PAD, dtype=torch.long)
        tgt = torch.full((len(rows), T), PAD, dtype=torch.long)
        mask = torch.zeros((len(rows), T), dtype=torch.bool)
        for r, (_x, y) in enumerate(rows):
            tin[r, 0] = BOS
            tin[r, 1:len(y)] = torch.from_numpy(y[:-1])
            tgt[r, :len(y)] = torch.from_numpy(y)
            mask[r, :len(y)] = True
        return {"src": src, "tgt_in": tin.to(device), "targets": tgt.to(device), "mask": mask.to(device), "S": S}
    T = max(len(x) + 1 + len(y) for x, y in rows)
    ids = torch.full((len(rows), T), PAD, dtype=torch.long)
    tgt = torch.full((len(rows), T), PAD, dtype=torch.long)
    mask = torch.zeros((len(rows), T), dtype=torch.bool)
    for r, (x, y) in enumerate(rows):
        seq = np.concatenate([x, [SEP], y])
        ids[r, :len(seq)] = torch.from_numpy(seq)
        tgt[r, len(x):len(x) + len(y)] = torch.from_numpy(y)
        mask[r, len(x):len(x) + len(y)] = True
    return {"ids": ids.to(device), "targets": tgt.to(device), "mask": mask.to(device)}


def group_pairs(family, pairs, max_rows, max_tokens):
    """Indices of pairs grouped into batches: B groups equal encoder lengths; decoder-only models sort by
    length and fill up to max_tokens padded tokens per batch."""
    order = sorted(range(len(pairs)), key=lambda i: (len(pairs[i][0]), len(pairs[i][1])))
    out, cur, key, longest = [], [], None, 0
    for i in order:
        k = len(pairs[i][0]) if family == "encdec" else None
        L = len(pairs[i][0]) + 1 + len(pairs[i][1])
        if cur and (len(cur) >= max_rows or k != key or max(longest, L) * (len(cur) + 1) > max_tokens):
            out.append(cur)
            cur, longest = [], 0
        cur.append(i)
        key, longest = k, max(longest, L)
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def generate(g, inputs, lengths, device, decode=None, batch_size=16, amp=True, log=print):
    """inputs: list of int64 token arrays (document part); lengths: list of (min_new, max_new) per row.
    -> list of generated token lists (EOS excluded). Rows with equal input length share a batch."""
    decode = decode or DecodeConfig(temperature=0.0, no_repeat_ngram=4)
    was = g.training
    g.eval()
    api = g.api()
    gen_rng = torch.Generator(device=device).manual_seed(decode.seed)
    prompts = [np.asarray(x, dtype=np.int64) if g.is_encdec else np.concatenate([x, [SEP]]).astype(np.int64)
               for x in inputs]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    batches, cur = [], []
    for i in order:
        if cur and (len(prompts[i]) != len(prompts[cur[0]]) or len(cur) >= batch_size):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)

    def run_batch(bt):
        x = torch.tensor(np.stack([prompts[i] for i in bt]), device=device)
        lo = [lengths[i][0] for i in bt]
        hi = [lengths[i][1] for i in bt]
        gen = [[] for _ in bt]
        done = [False] * len(bt)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits, state = api.prefill(x)
            for t in range(max(hi)):
                logits = logits.float()
                logits[:, BANNED] = -1e4
                for r in range(len(bt)):
                    if t < lo[r]:
                        logits[r, EOS] = -1e4
                nxt = sample_next(logits, gen, decode, gen_rng)
                for r, tok in enumerate(nxt.tolist()):
                    if done[r]:
                        continue
                    if tok == EOS:
                        done[r] = True
                    else:
                        gen[r].append(tok)
                        if len(gen[r]) >= hi[r]:
                            done[r] = True
                if all(done):
                    break
                logits, state = api.step(state, nxt)
        return gen

    out = [None] * len(prompts)
    pending = deque(batches)
    finished = 0
    while pending:
        bt = pending.popleft()
        gen = None
        try:
            gen = run_batch(bt)
        except torch.cuda.OutOfMemoryError:
            if len(bt) == 1:
                raise
        if gen is None:
            torch.cuda.empty_cache()
            m = len(bt) // 2
            pending.extendleft([bt[m:], bt[:m]])
            if log:
                log(f"generation OOM: batch of {len(bt)} -> {m} + {len(bt) - m}")
            continue
        for r, i in enumerate(bt):
            out[i] = gen[r]
        finished += 1
        if log and (finished % 5 == 1 or not pending):
            log(f"generated {sum(o is not None for o in out)}/{len(out)}")
    g.train(was)
    return out


def kd_terms(s_logits, t_logits, T):
    """Token-level T^2 KL(softmax(t/T) || softmax(s/T)), mean over target tokens."""
    pt = F.softmax(t_logits / T, -1)
    return (pt * (F.log_softmax(t_logits / T, -1) - F.log_softmax(s_logits / T, -1))).sum(-1).mean() * T * T
