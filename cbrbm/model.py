"""Context-Boosted Restricted Boltzmann Machine.

Gaussian-Bernoulli RBM whose hidden pre-activations receive a gated context
term. For visible v, context c and hidden h:

    E(v, h, c) = 0.5 ||v - b_v||^2 - sum_j h_j (w_j^T v + chat_j + b_j)
    chat_j     = (U_j^T c) * sigmoid(Gate_j^T c)

Marginalizing h gives the free energy

    F(v, c) = 0.5 ||v - b_v||^2 - sum_j softplus(w_j^T v + chat_j + b_j)

and the Context Boost score is the free-energy reduction due to context,

    CB(v, c) = sum_j [softplus(w_j^T v + chat_j + b_j) - softplus(w_j^T v + b_j)].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextBoostedRBM(nn.Module):
    def __init__(self, n_visible, n_hidden, n_context=None, device="cpu"):
        super().__init__()
        n_context = n_visible if n_context is None else n_context
        self.n_v = n_visible
        self.n_h = n_hidden
        self.n_c = n_context
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.U = nn.Parameter(torch.randn(n_context, n_hidden) * 0.01)
        self.Gate = nn.Parameter(torch.randn(n_context, n_hidden) * 0.01)
        self.b_v = nn.Parameter(torch.zeros(n_visible))
        self.b_h = nn.Parameter(torch.zeros(n_hidden))
        self.device = torch.device(device)
        self.to(self.device)

    def gated_context(self, c):
        return (c @ self.U) * torch.sigmoid(c @ self.Gate)

    def free_energy(self, v, c=None):
        v = v.to(self.device)
        pre = v @ self.W + self.b_h
        if c is not None:
            pre = pre + self.gated_context(c.to(self.device))
        quad = 0.5 * ((v - self.b_v) ** 2).sum(dim=1)
        return quad - F.softplus(pre).sum(dim=1)

    def context_boost(self, v, c):
        v = v.to(self.device)
        c = c.to(self.device)
        pre_feat = v @ self.W + self.b_h
        pre_full = pre_feat + self.gated_context(c)
        return (F.softplus(pre_full) - F.softplus(pre_feat)).sum(dim=1)

    def fit(self, v, c, epochs=100, batch_size=512, lr=1e-3, neg_std=0.1,
            seed=0, verbose=False):
        """One-step contrastive divergence on (v, c) pairs.

        The negative phase reconstructs v from a Bernoulli hidden sample while
        the context is held fixed; neg_std is the Gaussian reconstruction noise
        in normalized feature space.
        """
        v = torch.as_tensor(v, dtype=torch.float32).to(self.device)
        c = torch.as_tensor(c, dtype=torch.float32).to(self.device)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        n = v.shape[0]
        self.train()
        for epoch in range(epochs):
            perm = torch.randperm(n, generator=gen).to(self.device)
            total = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                v_pos, c_pos = v[idx], c[idx]
                pre = v_pos @ self.W + self.b_h + self.gated_context(c_pos)
                h = torch.bernoulli(torch.sigmoid(pre))
                v_neg = h @ self.W.t() + self.b_v
                v_neg = v_neg + neg_std * torch.randn_like(v_neg)
                loss = (self.free_energy(v_pos, c_pos) - self.free_energy(v_neg, c_pos)).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(idx)
            if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                print(f"epoch {epoch + 1:4d}  cd loss {total / n:.4f}")
        self.eval()
        return self
