"""Train on your own graph and write detection scores.

Input: an .npz file with three arrays.
  x          float array of shape (n_nodes, n_features), raw node features
  edge_index int array of shape (2, n_edges); row 0 = source, row 1 = target.
             The context of a node averages the features of its sources, so
             for an undirected graph include both directions of every edge.
  pool_idx   int array of node indices assumed legitimate, for example
             accounts with no abuse report after an aging window. The model
             is trained on 90% of them and the threshold is calibrated on the
             remaining 10%.

Output: a .npz with the detection score of every node (s = -CB), a boolean
flag (score >= threshold) and the threshold, plus the saved detector.

Usage:
  python examples/run_npz.py graph.npz --out scores.npz --model detector.pt
"""

import argparse

import numpy as np

from cbrbm import CBRBMDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help=".npz with x, edge_index, pool_idx")
    ap.add_argument("--out", default="scores.npz")
    ap.add_argument("--model", default="detector.pt")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    d = np.load(args.data)
    x, edge_index, pool_idx = d["x"], d["edge_index"], d["pool_idx"]

    det = CBRBMDetector(n_hidden=args.hidden, epochs=args.epochs, seed=args.seed,
                        device=args.device)
    det.fit(x, edge_index, pool_idx, verbose=True)
    det.save(args.model)

    scores = det.score_graph(x, edge_index)
    np.savez(args.out, scores=scores, flagged=det.predict(scores), threshold=det.threshold)
    print(f"threshold {det.threshold:.4f}; flagged {int(det.predict(scores).sum())} of {len(scores)} nodes")
    print(f"wrote {args.out} and {args.model}")


if __name__ == "__main__":
    main()
