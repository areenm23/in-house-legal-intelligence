"""Batches from the uint16 token shards written by phase1.data.make_shards.

Every batch is a pure function of (seed, step, micro-step): there is no loader state, so
resuming after a Kaggle session ends needs only the step counter, and two architectures
trained with the same seed see exactly the same token windows (a matched-token comparison).

  shards = TokenShards("data/shards", "train")
  x, y = causal_batch(shards, batch_size=B, seq_len=T, seed=0, index=k)
  src, tgt_in, tgt_out = continuation_batch(shards, B, src_len=S, tgt_len=T, seed=0, index=k)

`continuation_batch` cuts the same windows as `causal_batch` (window length S+T) and splits
them into an encoder prefix and a decoder target, so the encoder-decoder is trained on the
same tokens as the causal models.
"""
import json
import os

import numpy as np
import torch

BOS_ID = 2


class TokenShards:
    def __init__(self, shard_dir, split, sources=None):
        with open(os.path.join(shard_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        names = sources or [s for s in self.meta["sources"] if self.meta["sources"][s][split]["tokens"] > 0]
        self.arrays = {s: np.memmap(os.path.join(shard_dir, split, f"{s}.bin"), dtype=np.uint16, mode="r")
                       for s in names}
        self.names = list(self.arrays)
        sizes = np.array([len(self.arrays[s]) for s in self.names], dtype=np.float64)
        self.proportional = sizes / sizes.sum()

    def weights(self, mix=None):
        """mix: {source: weight} or None for proportional-to-size."""
        if mix is None:
            return self.proportional
        w = np.array([float(mix.get(s, 0.0)) for s in self.names])
        return w / w.sum()


def _windows(shards, batch_size, length, seed, index, mix):
    rng = np.random.default_rng([seed, index])
    srcs = rng.choice(len(shards.names), size=batch_size, p=shards.weights(mix))
    rows = []
    for k in srcs:
        arr = shards.arrays[shards.names[k]]
        i = int(rng.integers(0, len(arr) - length))
        rows.append(np.asarray(arr[i:i + length], dtype=np.int64))
    return torch.from_numpy(np.stack(rows))


def causal_batch(shards, batch_size, seq_len, seed, index, mix=None):
    w = _windows(shards, batch_size, seq_len + 1, seed, index, mix)
    return w[:, :-1], w[:, 1:]


def continuation_batch(shards, batch_size, src_len, tgt_len, seed, index, mix=None):
    w = _windows(shards, batch_size, src_len + tgt_len, seed, index, mix)
    src, tgt = w[:, :src_len], w[:, src_len:]
    tgt_in = torch.cat([torch.full((batch_size, 1), BOS_ID, dtype=tgt.dtype), tgt[:, :-1]], dim=1)
    return src, tgt_in, tgt


def fixed_eval_batches(shards, make_batch, n_batches, seed=1234):
    """Fixed per-source evaluation batches (same every time, same for every model).

    make_batch(shards, seed, index, mix) -> batch, e.g.
        lambda sh, seed, i, mix: causal_batch(sh, 8, 1024, seed, i, mix)
    Returns {source: [batch, ...]}.
    """
    return {s: [make_batch(shards, seed, i, {s: 1.0}) for i in range(n_batches)] for s in shards.names}
