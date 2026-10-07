import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(scores_legit, scores_adv):
    y = np.r_[np.zeros(len(scores_legit)), np.ones(len(scores_adv))]
    return float(roc_auc_score(y, np.r_[scores_legit, scores_adv]))


def tpr_at_threshold(scores_adv, threshold):
    return float((np.asarray(scores_adv) >= threshold).mean())


def fpr_at_threshold(scores_legit, threshold):
    return float((np.asarray(scores_legit) >= threshold).mean())


def cosine_correlation(x, ctx):
    """Mean per-node cosine between features and context (rho_0 of Assumption 1)."""
    x = np.asarray(x, dtype=np.float64)
    c = np.asarray(ctx, dtype=np.float64)
    nx = np.linalg.norm(x, axis=1).clip(min=1e-8)
    nc = np.linalg.norm(c, axis=1).clip(min=1e-8)
    return float(((x * c).sum(1) / (nx * nc)).mean())
