"""CTC decoding: greedy and prefix beam search. No external language model."""

import math

import numpy as np
import torch

NEG = -1e30


def _logaddexp(a, b):
    if a < b:
        a, b = b, a
    if b <= NEG:
        return a
    return a + math.log1p(math.exp(b - a))


def greedy_decode(log_probs, lengths=None):
    """log_probs: (B, T, V). Returns list of token-id lists."""
    ids = log_probs.detach().argmax(dim=-1).cpu().numpy()
    n = len(ids) if lengths is None else None
    lengths = [ids.shape[1]] * len(ids) if lengths is None else lengths.tolist()
    out = []
    for b in range(len(ids) if n is None else n):
        prev, tokens = -1, []
        for t in ids[b, : lengths[b]].tolist():
            if t != prev and t != 0:
                tokens.append(t)
            prev = t
        out.append(tokens)
    return out


def prefix_beam_decode(frame_log_probs, beam_width=16, top_k=8):
    """frame_log_probs: (T, V) numpy array of log-probabilities. Returns token ids."""
    if isinstance(frame_log_probs, torch.Tensor):
        frame_log_probs = frame_log_probs.detach().cpu().numpy()
    beams = {(): (0.0, NEG)}  # prefix -> (log p ending in blank, log p ending in non-blank)

    for t in range(frame_log_probs.shape[0]):
        row = frame_log_probs[t]
        k = min(top_k, row.shape[0] - 1)
        top = np.argpartition(row, -k)[-k:] if k > 0 else np.array([0])
        candidates = set(int(c) for c in top if c != 0) | {0}

        nxt = {}
        for prefix, (p_blank, p_nonblank) in beams.items():
            p_total = _logaddexp(p_blank, p_nonblank)

            # Emit blank: prefix stays, mass moves into the "ends with blank" slot.
            nb = nxt.get(prefix, (NEG, NEG))
            nxt[prefix] = (_logaddexp(nb[0], p_total + row[0]), nb[1])

            last = prefix[-1] if prefix else None
            for c in candidates:
                if c == 0:
                    continue
                if c == last:
                    # Repeat without a blank in between collapses: it cannot extend the prefix.
                    nb = nxt.get(prefix, (NEG, NEG))
                    nxt[prefix] = (nb[0], _logaddexp(nb[1], p_nonblank + row[c]))
                    # Only a prefix that ended in blank can extend with the same character.
                    ext = prefix + (c,)
                    eb = nxt.get(ext, (NEG, NEG))
                    nxt[ext] = (eb[0], _logaddexp(eb[1], p_blank + row[c]))
                else:
                    ext = prefix + (c,)
                    eb = nxt.get(ext, (NEG, NEG))
                    nxt[ext] = (eb[0], _logaddexp(eb[1], p_total + row[c]))

        beams = sorted(nxt.items(), key=lambda kv: -_logaddexp(*kv[1]))[:beam_width]
        beams = dict(beams)

    best = max(beams.items(), key=lambda kv: _logaddexp(*kv[1]))
    return list(best[0])
