"""Label-free detector built on the Context Boost score."""

import numpy as np
import torch

from .context import mean_pool, estimate_sign_vector, build_context
from .model import ContextBoostedRBM

MIN_POOL = 20


def _check_inputs(x, edge_index):
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("x must be a 2-D array of shape (n_nodes, n_features)")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN or infinite values")
    ei = torch.as_tensor(edge_index, dtype=torch.long)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, n_edges)")
    if ei.numel() and (ei.min() < 0 or ei.max() >= x.shape[0]):
        raise ValueError("edge_index contains node ids outside [0, n_nodes)")
    return ei


class CBRBMDetector:
    """Trains a CB-RBM on a pool of nodes assumed legitimate, calibrates a
    threshold on a held-out slice of that pool, and flags nodes whose score
    s = -CB(x, c) is at or above the threshold.

    Parameters
    ----------
    n_hidden : hidden units.
    target_fpr : calibration target; the threshold is the (1 - target_fpr)
        quantile of the scores on the calibration slice.
    calib_fraction : share of the pool held out for calibration.
    epochs, batch_size, lr : CD-1 training settings.
    """

    def __init__(self, n_hidden=256, target_fpr=0.05, calib_fraction=0.1,
                 epochs=100, batch_size=512, lr=1e-3, device="cpu", seed=0):
        self.n_hidden = n_hidden
        self.target_fpr = target_fpr
        self.calib_fraction = calib_fraction
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.seed = seed
        self.model = None
        self.mean_ = None
        self.std_ = None
        self.sign_vec = None
        self.threshold = None
        self.train_idx = None
        self.calib_idx = None

    # fitting

    def fit(self, x, edge_index, pool_idx, verbose=False):
        """x: (n_nodes, d) raw features. edge_index: (2, E). pool_idx: indices
        of the nodes assumed legitimate. Returns self."""
        x = np.asarray(x, dtype=np.float32)
        edge_index = _check_inputs(x, edge_index)
        n_nodes = x.shape[0]
        pool = np.unique(np.asarray(pool_idx, dtype=np.int64))
        if pool.size < MIN_POOL:
            raise ValueError(f"pool_idx has {pool.size} distinct nodes; at least {MIN_POOL} are needed")
        if pool.min() < 0 or pool.max() >= n_nodes:
            raise ValueError("pool_idx contains indices outside [0, n_nodes)")
        rng = np.random.RandomState(self.seed)
        pool = rng.permutation(pool)
        n_cal = max(1, int(round(self.calib_fraction * len(pool))))
        self.calib_idx = np.sort(pool[:n_cal])
        self.train_idx = np.sort(pool[n_cal:])

        self.mean_ = x[self.train_idx].mean(0)
        self.std_ = x[self.train_idx].std(0) + 1e-8
        xn = self.normalize(x)

        raw = mean_pool(xn, edge_index, n_nodes)
        self.sign_vec = estimate_sign_vector(xn, raw, self.train_idx)
        ctx = (raw * self.sign_vec[None, :]).astype(np.float32)

        torch.manual_seed(self.seed)
        self.model = ContextBoostedRBM(x.shape[1], self.n_hidden, device=self.device)
        self.model.fit(xn[self.train_idx], ctx[self.train_idx], epochs=self.epochs,
                       batch_size=self.batch_size, lr=self.lr, seed=self.seed,
                       verbose=verbose)
        self.threshold = self._quantile(self.score(xn[self.calib_idx], ctx[self.calib_idx],
                                                   normalized=True))
        return self

    def _quantile(self, scores):
        return float(np.percentile(scores, 100 * (1 - self.target_fpr)))

    # scoring

    def normalize(self, x):
        return ((np.asarray(x, dtype=np.float32) - self.mean_) / self.std_).astype(np.float32)

    def context(self, x, edge_index, normalized=False):
        xn = x if normalized else self.normalize(x)
        return build_context(xn, edge_index, xn.shape[0], self.sign_vec)

    def score(self, x, ctx, normalized=False):
        """Detection score s = -CB(x, c); higher means more suspicious."""
        xn = x if normalized else self.normalize(x)
        with torch.no_grad():
            v = torch.as_tensor(xn, dtype=torch.float32)
            c = torch.as_tensor(ctx, dtype=torch.float32)
            return -self.model.context_boost(v, c).cpu().numpy()

    def score_graph(self, x, edge_index, idx=None):
        """Scores for the nodes idx (all nodes if None) on the given graph."""
        xn = self.normalize(x)
        ctx = build_context(xn, edge_index, xn.shape[0], self.sign_vec)
        if idx is None:
            idx = np.arange(xn.shape[0])
        return self.score(xn[idx], ctx[idx], normalized=True)

    def predict(self, scores):
        return scores >= self.threshold

    # label-free refresh

    def refresh(self, x, edge_index, window_idx):
        """Re-estimate the sign vector and the threshold on a current window
        of unlabeled nodes, keeping the model weights fixed."""
        x = np.asarray(x, dtype=np.float32)
        edge_index = _check_inputs(x, edge_index)
        window_idx = np.unique(np.asarray(window_idx, dtype=np.int64))
        if window_idx.size < MIN_POOL:
            raise ValueError(f"window_idx has {window_idx.size} distinct nodes; at least {MIN_POOL} are needed")
        xn = self.normalize(x)
        raw = mean_pool(xn, edge_index, xn.shape[0])
        self.sign_vec = estimate_sign_vector(xn, raw, window_idx)
        ctx = (raw * self.sign_vec[None, :]).astype(np.float32)
        self.threshold = self._quantile(self.score(xn[window_idx], ctx[window_idx],
                                                   normalized=True))
        return self

    # persistence

    def save(self, path):
        """Store weights, normalization, sign vector, threshold and settings."""
        if self.model is None:
            raise RuntimeError("fit the detector before saving")
        torch.save({
            "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "n_visible": self.model.n_v,
            "n_hidden": self.n_hidden,
            "mean": self.mean_,
            "std": self.std_,
            "sign_vec": self.sign_vec,
            "threshold": self.threshold,
            "settings": dict(target_fpr=self.target_fpr, calib_fraction=self.calib_fraction,
                             epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
                             seed=self.seed),
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        det = cls(n_hidden=blob["n_hidden"], device=device, **blob["settings"])
        det.model = ContextBoostedRBM(blob["n_visible"], blob["n_hidden"], device=device)
        det.model.load_state_dict(blob["state_dict"])
        det.model.eval()
        det.mean_ = blob["mean"]
        det.std_ = blob["std"]
        det.sign_vec = blob["sign_vec"]
        det.threshold = blob["threshold"]
        return det
