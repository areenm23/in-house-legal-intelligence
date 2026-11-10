"""Sentence-level rhetorical-role model on top of a Phase-1 backbone.

  tokens of a run of consecutive sentences
    -> backbone over windows (ctx="win": windows of `window` tokens across sentence boundaries, so a
       sentence sees its neighbours; ctx="sent": every sentence encoded on its own, no context)
    -> token vectors averaged where windows overlap
    -> sentence vector = pool(token vectors of the sentence)      (mean | max | attn)
       + optional relative-position embedding (where the sentence sits in the document)
       + optional domain embedding (CL / IT)
    -> head over the sentence sequence: linear | mlp | bilstm (a BiLSTM across sentences gives
       document-level context to every backbone, causal or not)
    -> optional linear-chain CRF over the role sequence
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ft.heads import CRF
from .data import windows

N_POS = 20


def _ramp(L, ramp):
    w = torch.ones(L)
    if ramp and L > 2 * ramp:
        r = torch.linspace(0.1, 1.0, ramp)
        w[:ramp] = r
        w[-ramp:] = r.flip(0)
    return w


class RRModel(nn.Module):
    def __init__(self, bb, n_labels, ctx="win", pool="mean", head="bilstm", crf=False, echo=False, dropout=0.1,
                 hidden=512, pos_feat=True, dom_feat=True, n_dom=2):
        super().__init__()
        self.bb, self.n, self.ctx, self.pool, self.echo = bb, n_labels, ctx, pool, echo
        d = bb.d_model
        self.attn_w = nn.Linear(d, 1) if pool == "attn" else None
        self.pos = nn.Embedding(N_POS, 32) if pos_feat else None
        self.dom = nn.Embedding(n_dom, 16) if dom_feat else None
        d_in = d + (32 if pos_feat else 0) + (16 if dom_feat else 0)
        self.drop = nn.Dropout(dropout)
        self.kind = head
        if head == "linear":
            self.out = nn.Linear(d_in, n_labels)
        elif head == "mlp":
            self.out = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_labels))
        elif head == "bilstm":
            self.rnn = nn.LSTM(d_in, hidden // 2, batch_first=True, bidirectional=True)
            self.out = nn.Linear(hidden, n_labels)
        else:
            raise ValueError(head)
        self.crf = CRF(n_labels) if crf else None

    def _encode(self, ids, size, stride, amp, batch):
        """ids (L,) long on device -> (L, d) float token vectors."""
        L = ids.shape[0]
        dev = ids.device
        wins = windows(L, size, stride)
        ramp = max(0, (size - stride) // 2) if len(wins) > 1 else 0
        acc, wsum = None, torch.zeros(L, device=dev)
        for i in range(0, len(wins), batch):
            grp = wins[i:i + batch]
            x = torch.stack([ids[s:e] for s, e in grp])
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                h = self.bb(x, echo=self.echo)
            h = h.float()
            if acc is None:
                acc = torch.zeros(L, h.shape[-1], device=dev)
            for (s, e), row in zip(grp, h):
                w = _ramp(e - s, ramp).to(dev)
                if s == 0:
                    w[:ramp] = 1.0
                if e == L:
                    w[len(w) - ramp:] = 1.0
                acc = acc.index_add(0, torch.arange(s, e, device=dev), row * w[:, None])
                wsum = wsum.index_add(0, torch.arange(s, e, device=dev), w)
        return acc / wsum.clamp(min=1e-6)[:, None]

    def _encode_sentences(self, ids, sof, n_sent, amp, batch, max_len):
        """ctx="sent": every sentence alone (grouped by length so no padding is needed)."""
        dev = ids.device
        bounds = torch.searchsorted(sof, torch.arange(n_sent + 1, device=dev))
        by_len = {}
        for i in range(n_sent):
            s, e = int(bounds[i]), int(bounds[i + 1])
            by_len.setdefault(e - s, []).append(s)
        out = None
        for L, starts in sorted(by_len.items()):
            for j in range(0, len(starts), batch * 4):
                st = starts[j:j + batch * 4]
                x = torch.stack([ids[s:s + L] for s in st])
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                    h = self.bb(x, echo=self.echo)
                h = h.float()
                if out is None:
                    out = torch.zeros(ids.shape[0], h.shape[-1], device=dev)
                idx = (torch.tensor(st, device=dev)[:, None] + torch.arange(L, device=dev)[None]).reshape(-1)
                out = out.index_copy(0, idx, h.reshape(-1, h.shape[-1]))
        return out

    def _pool(self, H, sof, n_sent):
        d = H.shape[1]
        if self.pool == "max":
            out = torch.full((n_sent, d), -1e4, device=H.device, dtype=H.dtype)
            return out.scatter_reduce(0, sof[:, None].expand(-1, d), H, reduce="amax", include_self=True)
        if self.pool == "attn":
            a = self.attn_w(H).squeeze(-1)
            a = (a - a.max()).exp()
            den = torch.zeros(n_sent, device=H.device).index_add(0, sof, a)
            num = torch.zeros(n_sent, d, device=H.device).index_add(0, sof, H * a[:, None])
            return num / den.clamp(min=1e-9)[:, None]
        cnt = torch.zeros(n_sent, device=H.device).index_add(0, sof, torch.ones_like(sof, dtype=H.dtype))
        return torch.zeros(n_sent, d, device=H.device).index_add(0, sof, H) / cnt.clamp(min=1)[:, None]

    def forward(self, doc, s0, s1, dom_id, cfg, device, amp, grad_window=True):
        """Scores (s1 - s0, n_labels) for sentences s0..s1-1 of an encoded RRDoc."""
        t0 = int(np.searchsorted(doc.sent_of, s0, side="left"))
        t1 = int(np.searchsorted(doc.sent_of, s1 - 1, side="right"))
        ids = torch.from_numpy(doc.ids[t0:t1].astype(np.int64)).to(device)
        sof = torch.from_numpy((doc.sent_of[t0:t1] - s0).astype(np.int64)).to(device)
        n_sent = s1 - s0
        if self.ctx == "sent":
            H = self._encode_sentences(ids, sof, n_sent, amp, cfg["eval_batch"], cfg["window"])
        else:
            stride = cfg["window"] if (self.training and grad_window) else cfg["stride"]
            H = self._encode(ids, cfg["window"], stride, amp, cfg["eval_batch"])
        v = self._pool(H, sof, n_sent)
        feats = [v]
        if self.pos is not None:
            rel = (torch.arange(s0, s1, device=device).float() / max(1, len(doc) - 1))
            feats.append(self.pos((rel * (N_POS - 1)).round().long().clamp(0, N_POS - 1)))
        if self.dom is not None:
            feats.append(self.dom(torch.full((n_sent,), dom_id, device=device, dtype=torch.long)))
        x = self.drop(torch.cat(feats, -1))
        if self.kind == "bilstm":
            x, _ = self.rnn(x[None])
            x = self.drop(x[0])
        return self.out(x).float()

    def loss(self, sc, y, weight=None, focal=0.0, smoothing=0.0):
        if self.crf is not None:
            return self.crf.nll(sc[None], y[None], torch.ones_like(y, dtype=torch.bool)[None])
        if focal:
            lp = F.log_softmax(sc, -1)
            pt = lp.gather(1, y[:, None]).squeeze(1).exp()
            ce = F.nll_loss(lp, y, weight=weight, reduction="none")
            return ((1 - pt) ** focal * ce).sum() / (weight[y].sum() if weight is not None else len(y))
        return F.cross_entropy(sc, y, weight=weight, label_smoothing=smoothing)

    @torch.no_grad()
    def decode(self, sc):
        if self.crf is not None:
            return np.asarray(self.crf.decode(sc), dtype=np.int64)
        return sc.argmax(-1).cpu().numpy()
