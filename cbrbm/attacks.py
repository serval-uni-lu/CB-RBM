"""Injected-adversary attacks on a trained detector.

Each function returns (x_adv, ctx_adv) for the attacked nodes in the
detector's normalized feature space, so that detector.score(x_adv, ctx_adv,
normalized=True) gives their detection scores. Graph edits are applied to a
copy of edge_index; the original graph is not modified.
"""

import numpy as np
import torch

from .context import build_context


def isolated_injection(det, x_adv, rng):
    """Adversarial features with context drawn independently from N(0, I)."""
    xn = det.normalize(x_adv)
    ctx = rng.standard_normal(xn.shape).astype(np.float32)
    return xn, ctx


def feature_camouflage(det, x_legit, rng):
    """Copies of legitimate features with context drawn from N(0, I)."""
    return isolated_injection(det, x_legit, rng)


def sybil(det, x, edge_index, adv_idx, legit_idx, rng, clique_size=50, noise=0.05):
    """Attacked nodes connect to a fully connected clique of decoys whose
    features are legitimate features plus Gaussian noise. Contexts are
    recomputed on the expanded graph."""
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    decoy_src = rng.choice(legit_idx, size=clique_size, replace=True)
    decoys = x[decoy_src] + noise * rng.standard_normal((clique_size, x.shape[1])).astype(np.float32)
    x_ext = np.vstack([x, decoys])
    decoy_idx = np.arange(n, n + clique_size)
    src, dst = [], []
    for i in decoy_idx:
        for j in decoy_idx:
            if i != j:
                src.append(i); dst.append(j)
    for a in adv_idx:
        for s in decoy_idx:
            src += [a, s]; dst += [s, a]
    new_edges = torch.tensor([src, dst], dtype=torch.long)
    ei = torch.cat([torch.as_tensor(edge_index, dtype=torch.long), new_edges], dim=1)
    xn = det.normalize(x_ext)
    ctx = build_context(xn, ei, n + clique_size, det.sign_vec)
    return xn[adv_idx], ctx[adv_idx]


def relation_camouflage(det, x, edge_index, adv_idx, legit_idx, k=10):
    """Attacked nodes connect to the k highest-degree legitimate nodes.
    Contexts are recomputed on the expanded graph."""
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    ei0 = torch.as_tensor(edge_index, dtype=torch.long)
    deg = torch.zeros(n, dtype=torch.float32)
    deg.index_add_(0, ei0[1], torch.ones(ei0.shape[1]))
    legit_idx = np.asarray(legit_idx)
    hubs = legit_idx[torch.topk(deg[legit_idx], min(k, len(legit_idx))).indices.numpy()]
    src, dst = [], []
    for a in adv_idx:
        for h in hubs:
            src += [a, h]; dst += [h, a]
    new_edges = torch.tensor([src, dst], dtype=torch.long)
    ei = torch.cat([ei0, new_edges], dim=1)
    xn = det.normalize(x)
    ctx = build_context(xn, ei, n, det.sign_vec)
    return xn[adv_idx], ctx[adv_idx]
