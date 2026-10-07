import numpy as np
import torch

from cbrbm import ContextBoostedRBM, CBRBMDetector, mean_pool, estimate_sign_vector


def ring_graph(n, d, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    src = np.arange(n)
    dst = (src + 1) % n
    edge_index = torch.tensor(np.stack([np.r_[src, dst], np.r_[dst, src]]), dtype=torch.long)
    return x, edge_index


def test_context_boost_is_zero_without_context():
    rbm = ContextBoostedRBM(5, 7)
    v = torch.randn(10, 5)
    c = torch.zeros(10, 5)
    assert torch.allclose(rbm.context_boost(v, c), torch.zeros(10), atol=1e-6)


def test_mean_pool_and_sign_vector():
    x, ei = ring_graph(6, 3)
    ctx = mean_pool(x, ei, 6)
    assert ctx.shape == (6, 3)
    np.testing.assert_allclose(ctx[0], (x[1] + x[5]) / 2, rtol=1e-5)
    s = estimate_sign_vector(x, ctx, np.arange(6))
    assert set(np.unique(s)) <= {-1.0, 1.0}


def test_detector_calibration():
    x, ei = ring_graph(400, 4)
    det = CBRBMDetector(n_hidden=8, epochs=2, calib_fraction=0.25, seed=0)
    det.fit(x, ei, np.arange(400))
    s = det.score_graph(x, ei, det.calib_idx)
    fpr = (s >= det.threshold).mean()
    assert abs(fpr - 0.05) < 0.02
    det.refresh(x, ei, np.arange(400))
    assert np.isfinite(det.threshold)
