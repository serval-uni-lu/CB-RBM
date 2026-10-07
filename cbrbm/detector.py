"""Label-free detector built on the Context Boost score."""

import numpy as np
import torch

from .context import mean_pool, estimate_sign_vector, build_context
from .model import ContextBoostedRBM


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
        n_nodes = x.shape[0]
        rng = np.random.RandomState(self.seed)
        pool = rng.permutation(np.asarray(pool_idx))
        n_cal = max(1, int(round(self.calib_fraction * len(pool))))
        self.calib_idx = np.sort(pool[:n_cal])
        self.train_idx = np.sort(pool[n_cal:])

        self.mean_ = x[self.train_idx].mean(0)
        self.std_ = x[self.train_idx].std(0) + 1e-8
        xn = self.normalize(x)

        raw = mean_pool(xn, edge_index, n_nodes)
        self.sign_vec = estimate_sign_vector(xn, raw, self.train_idx)
        ctx = (raw * self.sign_vec[None, :]).astype(np.float32)

        self.model = ContextBoostedRBM(x.shape[1], self.n_hidden, device=self.device)
        torch.manual_seed(self.seed)
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
        xn = self.normalize(x)
        raw = mean_pool(xn, edge_index, xn.shape[0])
        self.sign_vec = estimate_sign_vector(xn, raw, window_idx)
        ctx = (raw * self.sign_vec[None, :]).astype(np.float32)
        self.threshold = self._quantile(self.score(xn[window_idx], ctx[window_idx],
                                                   normalized=True))
        return self
