"""
robust_likelihoods.py  -  impulsive-noise-aware Bayesian estimator (Q1 add-on)
=============================================================================
SELF-CONTAINED and ISOLATED. This module imports NOTHING from bayce_model.py
and writes checkpoints with unique names (robust_*.pt). Deleting this file and
train_robust.py removes the whole Q1 experiment with ZERO effect on the
original pipeline or its results. Keep your original bayce_model.py untouched.

The novelty core of the Q1 version: the baseline trains a Gaussian negative
log-likelihood (NLL), which is MISMATCHED to the Bernoulli-Gaussian impulsive
noise of the UAC channel. We replace it with likelihoods matched to the
physics, so robustness and calibrated uncertainty emerge from ONE model:

  1. gaussian   (baseline)  - light-tailed reference.
  2. studentt   (robust)    - heavy-tailed; large residuals down-weighted
                              logarithmically; degrees of freedom nu learned.
  3. bgmix      (matched)    - the EXACT generative noise model,
                              (1-eps)N(0,s^2)+eps N(0,k s^2); eps,k learned.

The network below is architecturally IDENTICAL to the original BayCE-Net (so
comparisons are fair), but defined here independently so the experiment stays
self-contained.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ======================================================================
# Dataset (identical split logic to the original; kept here for isolation)
# ======================================================================
class CEDataset(Dataset):
    def __init__(self, npz_path, split="train", train_frac=.7, val_frac=.1, seed=0):
        d = np.load(npz_path, allow_pickle=True)
        cid = d["chan_id"]; uniq = np.unique(cid)
        rng = np.random.default_rng(seed); rng.shuffle(uniq)
        n = len(uniq); ntr, nva = int(train_frac*n), int(val_frac*n)
        sets = {"train": uniq[:ntr], "val": uniq[ntr:ntr+nva], "test": uniq[ntr+nva:]}
        m = np.isin(cid, sets[split])
        Hls, s2, H = d["H_ls"][m], d["sigma2"][m], d["H_true"][m]
        self.x = np.stack([Hls.real, Hls.imag, np.log(s2+1e-8)], 1).astype(np.float32)
        self.y = np.stack([H.real, H.imag], 1).astype(np.float32)
        self.scale = np.clip(np.sqrt(np.mean(np.abs(Hls)**2, 1, keepdims=True)),
                             1e-6, None).astype(np.float32)
        print(f"[{split}] {m.sum()} samples from {len(sets[split])} channels")

    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        s = self.scale[i]; x = self.x[i].copy(); x[:2] /= s
        return torch.from_numpy(x), torch.from_numpy(self.y[i]/s)


# ======================================================================
# Model (identical architecture to the original BayCE-Net)
# ======================================================================
class ImpulseGate(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, ch, 5, padding=2), nn.ReLU(),
            nn.Conv1d(ch, 1, 5, padding=2), nn.Sigmoid())
    def forward(self, x):
        w = self.net(x)
        gated = x[:, :2] * w; rest = x[:, 2:]
        return torch.cat([gated, rest], 1), w


class SparseBackbone(nn.Module):
    def __init__(self, ch=64, n_blocks=5, p_drop=0.1):
        super().__init__()
        self.inp = nn.Conv1d(3, ch, 3, padding=1)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Conv1d(ch, ch, 3, padding=1)))
        self.act = nn.ReLU()
    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = self.act(h + b(h))
        return h


class BayCENet(nn.Module):
    def __init__(self, ch=64, n_blocks=5, p_drop=0.1):
        super().__init__()
        self.gate = ImpulseGate()
        self.backbone = SparseBackbone(ch, n_blocks, p_drop)
        self.head_mu = nn.Conv1d(ch, 2, 3, padding=1)
        self.head_lv = nn.Conv1d(ch, 1, 3, padding=1)
    def forward(self, x):
        x, w = self.gate(x)
        feat = self.backbone(x)
        mu = self.head_mu(feat)
        log_var = torch.clamp(self.head_lv(feat), -10, 10)
        return mu, log_var, w


@torch.no_grad()
def mc_predict(model, x, T=20):
    model.train()
    mus = torch.stack([model(x)[0] for _ in range(T)])
    return mus.mean(0), mus.var(0).sum(1, keepdim=True)


# ======================================================================
# likelihoods  (mu:(B,2,N)  logvar:(B,1,N)  target:(B,2,N)) -> scalar
# ======================================================================
def _err2_L(mu, logvar, target):
    err2 = ((mu - target) ** 2).sum(1, keepdim=True)
    L = torch.clamp(logvar, -10, 10)
    return err2, L


def gaussian_nll(mu, logvar, target, **kw):
    err2, L = _err2_L(mu, logvar, target)
    return (0.5 * err2 * torch.exp(-L) + L).mean()


def studentt_nll(mu, logvar, target, log_nu=None, **kw):
    err2, L = _err2_L(mu, logvar, target)
    nu = 2.0 + torch.nn.functional.softplus(log_nu)
    z = err2 * torch.exp(-L) / nu
    data = 0.5 * (nu + 2.0) * torch.log1p(z)
    norm = L + (torch.log(nu) + torch.lgamma(nu/2.0) - torch.lgamma((nu+2.0)/2.0))
    return (data + norm).mean()


def bgmix_nll(mu, logvar, target, logit_eps=None, log_km1=None, **kw):
    err2, L = _err2_L(mu, logvar, target)
    eps = torch.sigmoid(logit_eps)
    k = 1.0 + torch.nn.functional.softplus(log_km1)
    e1 = 0.5 * err2 * torch.exp(-L); e2 = e1 / k
    a1 = torch.log(1.0 - eps) - e1
    a2 = torch.log(eps) - torch.log(k) - e2
    lse = torch.logsumexp(torch.stack([a1, a2], 0), dim=0)
    return (L - lse).mean()


LOSSES = {"gaussian": gaussian_nll, "studentt": studentt_nll, "bgmix": bgmix_nll}


# ======================================================================
# wrapper: BayCENet + learnable likelihood parameters
# ======================================================================
class RobustBayCE(nn.Module):
    def __init__(self, likelihood="studentt", ch=64, n_blocks=5, p_drop=0.1):
        super().__init__()
        assert likelihood in LOSSES
        self.likelihood = likelihood
        self.net = BayCENet(ch, n_blocks, p_drop)
        if likelihood == "studentt":
            self.log_nu = nn.Parameter(torch.tensor(1.0))
        elif likelihood == "bgmix":
            self.logit_eps = nn.Parameter(torch.tensor(math.log(0.05/0.95)))
            self.log_km1 = nn.Parameter(torch.tensor(math.log(math.expm1(100.0))))

    def forward(self, x): return self.net(x)

    def loss(self, mu, logvar, target):
        if self.likelihood == "studentt":
            return studentt_nll(mu, logvar, target, log_nu=self.log_nu)
        if self.likelihood == "bgmix":
            return bgmix_nll(mu, logvar, target,
                             logit_eps=self.logit_eps, log_km1=self.log_km1)
        return gaussian_nll(mu, logvar, target)

    def var_scale(self):
        """Factor to turn the head's sigma^2 into the TRUE predictive variance
        of this likelihood (needed for a fair calibration comparison):
          gaussian: 1 ;  studentt: nu/(nu-2) ;  bgmix: 1-eps+eps*k ."""
        with torch.no_grad():
            if self.likelihood == "studentt":
                nu = 2.0 + torch.nn.functional.softplus(self.log_nu)
                return (nu / (nu - 2.0)).item()
            if self.likelihood == "bgmix":
                eps = torch.sigmoid(self.logit_eps)
                k = 1.0 + torch.nn.functional.softplus(self.log_km1)
                return (1.0 - eps + eps * k).item()
            return 1.0

    def params_report(self):
        with torch.no_grad():
            if self.likelihood == "studentt":
                nu = 2.0 + torch.nn.functional.softplus(self.log_nu)
                return f"nu={nu.item():.2f}"
            if self.likelihood == "bgmix":
                eps = torch.sigmoid(self.logit_eps)
                k = 1.0 + torch.nn.functional.softplus(self.log_km1)
                return f"eps={eps.item():.3f} k={k.item():.1f}"
            return "-"


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(8, 3, 1024).to(dev); y = torch.randn(8, 2, 1024).to(dev)
    for lk in LOSSES:
        m = RobustBayCE(lk).to(dev)
        mu, lv, _ = m(x); loss = m.loss(mu, lv, y); loss.backward()
        assert torch.isfinite(loss), lk
        print(f"{lk:9s} loss={loss.item():8.4f}  params[{m.params_report()}]  OK")
    print("\nself-contained; imports nothing from bayce_model.py")
