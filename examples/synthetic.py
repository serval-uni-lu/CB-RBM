"""End-to-end run on a synthetic homophilic graph.

Builds a graph whose node features correlate with their neighbors' features,
trains a CB-RBM on the nodes assumed legitimate, calibrates the threshold,
and reports detection on the four attacks of the paper and the false-positive
rate under natural drift at the frozen threshold.
"""

import argparse

import numpy as np
import torch

from cbrbm import CBRBMDetector, attacks, drift, metrics


def synthetic_graph(n_nodes=4000, dim=16, n_groups=8, m=5, p_cross=0.05, seed=0):
    rng = np.random.RandomState(seed)
    groups = rng.randint(0, n_groups, size=n_nodes)
    centers = rng.standard_normal((n_groups, dim)) * 2.0
    x = centers[groups] + rng.standard_normal((n_nodes, dim))
    src, dst = [], []
    for i in range(n_nodes):
        same = np.flatnonzero(groups == groups[i])
        for j in rng.choice(same, size=m, replace=False):
            if j != i:
                src += [i, j]; dst += [j, i]
        if rng.rand() < p_cross:
            j = rng.randint(n_nodes)
            src += [i, j]; dst += [j, i]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return x.astype(np.float32), edge_index, rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    x, edge_index, rng = synthetic_graph(seed=args.seed)
    n = x.shape[0]
    perm = rng.permutation(n)
    adv_idx = np.sort(perm[:200])          # hosts for injected adversaries
    pool_idx = np.sort(perm[200:])          # nodes assumed legitimate
    # adversarial features: drawn from a different distribution than any group
    x_adv = (rng.standard_normal((len(adv_idx), x.shape[1])) * 1.5 + 3.0).astype(np.float32)
    x_attacked = x.copy()
    x_attacked[adv_idx] = x_adv

    det = CBRBMDetector(n_hidden=args.hidden, epochs=args.epochs, seed=args.seed)
    det.fit(x, edge_index, pool_idx, verbose=True)

    ctx_all = det.context(x, edge_index)
    s_pool = det.score(x[pool_idx], ctx_all[pool_idx])
    print(f"\nrho_0 on the pool: {metrics.cosine_correlation(det.normalize(x)[pool_idx], ctx_all[pool_idx]):.3f}")
    print(f"threshold: {det.threshold:.3f}   FPR on the pool: {metrics.fpr_at_threshold(s_pool, det.threshold):.3f}")

    runs = {
        "isolated injection": attacks.isolated_injection(det, x_adv, rng),
        "feature camouflage": attacks.feature_camouflage(det, x[rng.choice(pool_idx, 200, replace=False)], rng),
        "sybil": attacks.sybil(det, x_attacked, edge_index, adv_idx, pool_idx, rng),
        "relation camouflage": attacks.relation_camouflage(det, x_attacked, edge_index, adv_idx, pool_idx),
    }
    print("\nattack                 AUROC   TPR@threshold")
    for name, (xa, ca) in runs.items():
        s_adv = det.score(xa, ca, normalized=True)
        print(f"{name:22s} {metrics.auroc(s_pool, s_adv):.3f}   {metrics.tpr_at_threshold(s_adv, det.threshold):.3f}")

    xd, cd = drift.drifted(det, x, edge_index, T=20, sigma=0.2, rng=rng, idx=pool_idx)
    s_drift = det.score(xd, cd, normalized=True)
    print(f"\nFPR on the pool after 20 drift steps, frozen threshold: {metrics.fpr_at_threshold(s_drift, det.threshold):.3f}")


if __name__ == "__main__":
    main()
