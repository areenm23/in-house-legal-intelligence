"""Document classifier (court judgment prediction) on top of a Phase-1 backbone.

  document -> views (data.views: first / last / middle / headtail = one sequence, hier = K chunks)
    -> backbone over every sequence (sequences of equal length share a batch, so there is no padding
       and no attention mask is needed; only documents shorter than seq_len give shorter sequences)
    -> chunk vector = pool(token vectors)                 mean | max | last | attn | mean_last (concat)
       + chunk-position embedding (relative position of the chunk in the document, zero at init)
    -> document vector = aggregate(chunk vectors)         mean | max | attn | gru (BiGRU + attention)
    -> [+ log-length embedding] -> dropout -> head (linear | mlp) -> 2 logits
"""
import numpy as np
import torch
import torch.nn as nn

N_POS = 16
N_LEN = 12


class CJPEModel(nn.Module):
    def __init__(self, bb, pool="mean", agg="attn", head="linear", dropout=0.1, hidden=512, pos_feat=True,
                 len_feat=False, echo=False, n_classes=2):
        super().__init__()
        self.bb, self.pool, self.agg, self.echo = bb, pool, agg, echo
        d = bb.d_model
        self.d_chunk = 2 * d if pool == "mean_last" else d
        self.tok_attn = nn.Linear(d, 1) if pool == "attn" else None
        self.pos = nn.Embedding(N_POS, self.d_chunk) if pos_feat else None
        if self.pos is not None:
            nn.init.zeros_(self.pos.weight)
        if agg == "gru":
            self.gru = nn.GRU(self.d_chunk, hidden // 2, batch_first=True, bidirectional=True)
            self.d_doc = hidden
        elif agg in ("mean", "max", "attn"):
            self.d_doc = self.d_chunk
        else:
            raise ValueError(f"unknown agg {agg!r}")
        self.agg_attn = nn.Linear(self.d_doc, 1) if agg in ("attn", "gru") else None
        self.len_emb = nn.Embedding(N_LEN, 16) if len_feat else None
        d_in = self.d_doc + (16 if len_feat else 0)
        self.drop = nn.Dropout(dropout)
        if head == "linear":
            self.out = nn.Linear(d_in, n_classes)
        elif head == "mlp":
            self.out = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))
        else:
            raise ValueError(f"unknown head {head!r}")

    def _pool(self, h):
        """(B, T, d) float -> (B, d_chunk)."""
        if self.pool == "mean":
            return h.mean(1)
        if self.pool == "max":
            return h.max(1).values
        if self.pool == "last":
            return h[:, -1]
        if self.pool == "mean_last":
            return torch.cat([h.mean(1), h[:, -1]], -1)
        if self.pool == "attn":
            a = torch.softmax(self.tok_attn(h).squeeze(-1), dim=1)
            return (a.unsqueeze(-1) * h).sum(1)
        raise ValueError(f"unknown pool {self.pool!r}")

    def encode(self, seqs, device, amp, seq_batch):
        """list of int64 token arrays -> (n, d_chunk) float chunk vectors."""
        by_len = {}
        for i, s in enumerate(seqs):
            by_len.setdefault(len(s), []).append(i)
        rows = [None] * len(seqs)
        for _L, idx in sorted(by_len.items()):
            for j in range(0, len(idx), max(1, seq_batch)):
                grp = idx[j:j + max(1, seq_batch)]
                x = torch.from_numpy(np.stack([seqs[i] for i in grp])).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    h = self.bb(x, echo=self.echo)
                v = self._pool(h.float())
                for i, r in zip(grp, v):
                    rows[i] = r
        return torch.stack(rows)

    def aggregate(self, v, pos):
        """v (k, d_chunk) chunk vectors of one document, pos (k,) relative positions -> (d_doc,)."""
        if self.pos is not None:
            v = v + self.pos((pos * (N_POS - 1)).round().long().clamp(0, N_POS - 1))
        if self.agg == "mean":
            return v.mean(0)
        if self.agg == "max":
            return v.max(0).values
        if self.agg == "gru":
            with torch.autocast(device_type=v.device.type, enabled=False):
                v, _ = self.gru(v.float()[None])
            v = v[0]
        a = torch.softmax(self.agg_attn(v).squeeze(-1), 0)
        return (a[:, None] * v).sum(0)

    def head(self, D, n_tok=None):
        x = D
        if self.len_emb is not None:
            b = np.clip(np.floor(np.log2(np.maximum(np.asarray(n_tok, dtype=np.float64), 1))) - 6, 0, N_LEN - 1)
            x = torch.cat([x, self.len_emb(torch.as_tensor(b.astype(np.int64), device=D.device))], -1)
        return self.out(self.drop(x)).float()

    def forward(self, docs, device, amp, seq_batch, n_tok=None):
        """docs: list of (seqs, rel_pos) per document (data.views) -> (logits (n, 2), doc vectors (n, d_doc))."""
        flat = [s for seqs, _ in docs for s in seqs]
        V = self.encode(flat, device, amp, seq_batch)
        out, k0 = [], 0
        for seqs, pos in docs:
            k = len(seqs)
            out.append(self.aggregate(V[k0:k0 + k], torch.as_tensor(np.asarray(pos, dtype=np.float32), device=device)))
            k0 += k
        D = torch.stack(out)
        return self.head(D, n_tok), D
