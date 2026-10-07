"""Neighborhood context: sign-corrected mean pooling of neighbor features."""

import numpy as np
import torch


def mean_pool(x, edge_index, n_nodes):
    """Mean of neighbor features for every node.

    x: (n_nodes, d) array or tensor. edge_index: (2, E) long tensor of
    directed edges (src, dst); the context of dst averages the features of
    its sources. Nodes without in-edges get a zero context.
    """
    feat = torch.as_tensor(x, dtype=torch.float32)
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    src, dst = edge_index[0], edge_index[1]
    agg = torch.zeros(n_nodes, feat.shape[1], dtype=torch.float32)
    deg = torch.zeros(n_nodes, dtype=torch.float32)
    agg.index_add_(0, dst, feat[src])
    deg.index_add_(0, dst, torch.ones(len(src)))
    return (agg / deg.clamp(min=1).unsqueeze(1)).numpy()


def estimate_sign_vector(x, raw_context, idx):
    """Per-dimension sign of the Pearson correlation between a node's feature
    and the mean feature of its neighbors, estimated on the nodes in idx.
    Dimensions with zero correlation get +1. Uses features only, no labels."""
    xs = np.asarray(x, dtype=np.float64)[idx]
    cs = np.asarray(raw_context, dtype=np.float64)[idx]
    xs = xs - xs.mean(0)
    cs = cs - cs.mean(0)
    corr = (xs * cs).mean(0) / ((xs.std(0) + 1e-8) * (cs.std(0) + 1e-8))
    s = np.sign(np.nan_to_num(corr)).astype(np.float32)
    s[s == 0] = 1.0
    return s


def build_context(x, edge_index, n_nodes, sign_vec):
    """Sign-corrected mean-pool context c_i = s * mean_{j in N(i)} x_j."""
    raw = mean_pool(x, edge_index, n_nodes)
    return (raw * sign_vec[None, :]).astype(np.float32)
