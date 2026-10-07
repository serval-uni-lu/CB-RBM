"""Natural drift as a random walk in feature space, with the neighborhood
context recomputed from the drifted features at every step."""

import numpy as np

from .context import build_context


def random_walk(det, x, edge_index, T=20, sigma=0.2, rng=None, idx=None):
    """Returns the list [(x_t, c_t)] for t = 0..T restricted to idx, in the
    detector's normalized feature space. Each step adds N(0, sigma^2 I) to
    every node's normalized features and recomputes all contexts."""
    rng = np.random.RandomState(0) if rng is None else rng
    xn = det.normalize(x)
    n = xn.shape[0]
    idx = np.arange(n) if idx is None else np.asarray(idx)
    out = [(xn[idx].copy(), build_context(xn, edge_index, n, det.sign_vec)[idx])]
    cur = xn.copy()
    for _ in range(T):
        cur = cur + sigma * rng.standard_normal(cur.shape).astype(np.float32)
        ctx = build_context(cur, edge_index, n, det.sign_vec)
        out.append((cur[idx].copy(), ctx[idx]))
    return out


def drifted(det, x, edge_index, T=20, sigma=0.2, rng=None, idx=None):
    """Features and contexts after T steps in one draw, N(0, T sigma^2 I)."""
    rng = np.random.RandomState(0) if rng is None else rng
    xn = det.normalize(x)
    n = xn.shape[0]
    idx = np.arange(n) if idx is None else np.asarray(idx)
    cur = xn + sigma * np.sqrt(T) * rng.standard_normal(xn.shape).astype(np.float32)
    ctx = build_context(cur, edge_index, n, det.sign_vec)
    return cur[idx], ctx[idx]
