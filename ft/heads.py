"""Task heads and structured decoding.

TokenHead: per-token tag scores from backbone vectors ("linear", "mlp", or "bilstm", which adds
a bidirectional LSTM so even a causal backbone's tags see right context).
CRF: linear-chain CRF over tag scores (negative log-likelihood + Viterbi), with optional
hard constraints (e.g. BIO: I-X may only follow B-X or I-X).
viterbi(): constrained best path for plain per-token log-probabilities (no learned transitions).
"""
import numpy as np
import torch
import torch.nn as nn

NEG = -1e4


class TokenHead(nn.Module):
    def __init__(self, d, n_tags, kind="linear", dropout=0.1, hidden=512):
        super().__init__()
        self.kind = kind
        self.drop = nn.Dropout(dropout)
        if kind == "linear":
            self.out = nn.Linear(d, n_tags)
        elif kind == "mlp":
            self.out = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_tags))
        elif kind == "bilstm":
            self.rnn = nn.LSTM(d, hidden // 2, batch_first=True, bidirectional=True)
            self.out = nn.Linear(hidden, n_tags)
        else:
            raise ValueError(kind)

    def forward(self, h):
        h = self.drop(h)
        if self.kind == "bilstm":
            with torch.autocast(device_type=h.device.type, enabled=False):
                h, _ = self.rnn(h.float())
            h = self.drop(h)
        return self.out(h).float()


class CRF(nn.Module):
    def __init__(self, n_tags, allowed=None, start_allowed=None):
        """allowed: (n, n) bool, allowed[i, j] = tag j may follow tag i. start_allowed: (n,) bool."""
        super().__init__()
        self.n = n_tags
        self.trans = nn.Parameter(torch.zeros(n_tags, n_tags))
        self.start = nn.Parameter(torch.zeros(n_tags))
        self.end = nn.Parameter(torch.zeros(n_tags))
        self.register_buffer("allowed", allowed if allowed is not None else torch.ones(n_tags, n_tags, dtype=torch.bool))
        self.register_buffer("start_ok", start_allowed if start_allowed is not None else torch.ones(n_tags, dtype=torch.bool))

    def _T(self):
        return self.trans.masked_fill(~self.allowed, NEG)

    def _S(self):
        return self.start.masked_fill(~self.start_ok, NEG)

    def nll(self, em, tags, mask):
        """em (B, T, n) float, tags (B, T) long, mask (B, T) bool, contiguous from position 0."""
        em = em.float()
        B, T, n = em.shape
        tr, st = self._T(), self._S()
        tags = tags.clamp(min=0)
        score = st[tags[:, 0]] + em[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, T):
            s = tr[tags[:, t - 1], tags[:, t]] + em[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + s * mask[:, t]
        last = tags.gather(1, (mask.sum(1, keepdim=True) - 1).clamp(min=0)).squeeze(1)
        score = score + self.end[last]
        alpha = st[None] + em[:, 0]
        for t in range(1, T):
            nxt = torch.logsumexp(alpha[:, :, None] + tr[None] + em[:, t, None, :], dim=1)
            alpha = torch.where(mask[:, t, None], nxt, alpha)
        logZ = torch.logsumexp(alpha + self.end[None], dim=1)
        return (logZ - score).sum() / mask.sum().clamp(min=1)

    @torch.no_grad()
    def decode(self, em):
        """em (T, n) for one sequence -> best tag list."""
        f = lambda t: t.detach().float().cpu().numpy()
        return viterbi(f(em), f(self._T()), f(self._S()), f(self.end))


def viterbi(em, trans, start, end=None):
    """em (T, n) scores, trans (n, n) (NEG where forbidden), start (n,) -> best path (list of ints). numpy."""
    em = np.asarray(em, dtype=np.float64)
    T, n = em.shape
    score = start + em[0]
    back = np.zeros((max(T - 1, 0), n), dtype=np.int32)
    for t in range(1, T):
        cand = score[:, None] + trans
        idx = cand.argmax(0)
        back[t - 1] = idx
        score = cand[idx, np.arange(n)] + em[t]
    if end is not None:
        score = score + end
    path = [int(score.argmax())]
    for t in range(T - 2, -1, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


def constrained_decode(logp, allowed, start_ok):
    """Per-token log-probs (T, n) + hard BIO constraints -> best valid tag path."""
    allowed = np.asarray(allowed.cpu() if hasattr(allowed, "cpu") else allowed, dtype=bool)
    start_ok = np.asarray(start_ok.cpu() if hasattr(start_ok, "cpu") else start_ok, dtype=bool)
    trans = np.where(allowed, 0.0, NEG)
    start = np.where(start_ok, 0.0, NEG)
    lp = logp.detach().float().cpu().numpy() if hasattr(logp, "detach") else np.asarray(logp)
    return viterbi(lp, trans, start)
