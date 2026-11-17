"""
bayce_model.py  —  BayCE-Net dataset + model + loss
===================================================
Contents
  1. CEDataset          : loads .npz, splits BY channel id (no leakage)
  2. BayCENet           : 3 stages + dual Bayesian head (mu, log-var)
  3. gaussian_nll_loss  : the Bayesian training objective
  4. mc_predict         : MC-Dropout inference -> epistemic uncertainty
  5. __main__ self-test : forward + backward on a dummy/real batch

Design choice: complex numbers are carried as TWO REAL CHANNELS
(real, imag) — the standard in ChannelNet / DeSA-DNN — because it is
numerically stable and every layer supports it.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ======================================================================
# 1. Dataset  — split by channel id so train/test never share a channel
# ======================================================================
class CEDataset(Dataset):
    def __init__(self, npz_path, split="train",
                 train_frac=0.7, val_frac=0.1, seed=0):
        d = np.load(npz_path, allow_pickle=True)
        chan_id = d["chan_id"]
        uniq = np.unique(chan_id)

        rng = np.random.default_rng(seed)
        rng.shuffle(uniq)
        n = len(uniq)
        n_tr, n_va = int(train_frac * n), int(val_frac * n)
        sets = {"train": uniq[:n_tr],
                "val":   uniq[n_tr:n_tr + n_va],
                "test":  uniq[n_tr + n_va:]}
        mask = np.isin(chan_id, sets[split])

        # input  : [Re(H_ls), Im(H_ls), log(sigma2)]   -> (3, N)
        Hls = d["H_ls"][mask]
        s2 = d["sigma2"][mask]
        self.x = np.stack([Hls.real, Hls.imag,
                           np.log(s2 + 1e-8)], axis=1).astype(np.float32)
        # target : [Re(H), Im(H)]                       -> (2, N)
        H = d["H_true"][mask]
        self.y = np.stack([H.real, H.imag], axis=1).astype(np.float32)

        # per-sample normalisation factor (by LS energy) — store to invert
        self.scale = np.sqrt(np.mean(np.abs(Hls) ** 2, axis=1,
                                     keepdims=True)).astype(np.float32)
        self.scale = np.clip(self.scale, 1e-6, None)

        print(f"[{split}] {mask.sum()} samples "
              f"from {len(sets[split])} channels")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        s = self.scale[i]                     # (1,)
        x = self.x[i].copy()
        x[:2] /= s                            # scale Re/Im, keep log-sigma
        y = self.y[i] / s
        return torch.from_numpy(x), torch.from_numpy(y)


# ======================================================================
# 2. Model
# ======================================================================
class ImpulseGate(nn.Module):
    """Stage 1: learns to down-weight impulse-corrupted subcarriers."""
    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, ch, 5, padding=2), nn.ReLU(),
            nn.Conv1d(ch, 1, 5, padding=2), nn.Sigmoid())

    def forward(self, x):
        w = self.net(x)                       # (B,1,N) in (0,1)
        gated = x[:, :2] * w                  # gate Re/Im only (no in-place op)
        rest = x[:, 2:]
        return torch.cat([gated, rest], 1), w


class SparseBackbone(nn.Module):
    """Stage 2: residual conv backbone (light, ~sparse-friendly)."""
    def __init__(self, ch=64, n_blocks=5, p_drop=0.1):
        super().__init__()
        self.inp = nn.Conv1d(3, ch, 3, padding=1)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU(),
                nn.Dropout(p_drop),                    # <- MC-Dropout
                nn.Conv1d(ch, ch, 3, padding=1)))
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = self.act(h + b(h))            # residual
        return h                              # (B, ch, N)


class BayCENet(nn.Module):
    """Stage 3: dual Bayesian head -> (mu, log_var) per subcarrier."""
    def __init__(self, ch=64, n_blocks=5, p_drop=0.1):
        super().__init__()
        self.gate = ImpulseGate()
        self.backbone = SparseBackbone(ch, n_blocks, p_drop)
        self.head_mu = nn.Conv1d(ch, 2, 3, padding=1)      # Re/Im estimate
        self.head_lv = nn.Conv1d(ch, 1, 3, padding=1)      # log aleatoric var

    def forward(self, x):
        x, w = self.gate(x)
        feat = self.backbone(x)
        mu = self.head_mu(feat)                             # (B,2,N)
        log_var = self.head_lv(feat)                        # (B,1,N)
        log_var = torch.clamp(log_var, -10, 10)            # stability
        return mu, log_var, w


# ======================================================================
# 3. Loss  — Gaussian negative log-likelihood (aleatoric) + L1 sparsity
# ======================================================================
def gaussian_nll_loss(mu, log_var, target, l1_weight=1e-4):
    # complex error magnitude^2 = (Re err)^2 + (Im err)^2
    err2 = ((mu - target) ** 2).sum(dim=1, keepdim=True)   # (B,1,N)
    inv_var = torch.exp(-log_var)
    nll = 0.5 * (inv_var * err2 + log_var)                 # (B,1,N)
    l1 = mu.abs().mean()                                   # crude sparsity
    return nll.mean() + l1_weight * l1


# ======================================================================
# 4. MC-Dropout inference  -> mean estimate + epistemic uncertainty
# ======================================================================
@torch.no_grad()
def mc_predict(model, x, T=20):
    model.train()                             # keep dropout ON
    mus = []
    for _ in range(T):
        mu, _, _ = model(x)
        mus.append(mu)
    mus = torch.stack(mus)                     # (T,B,2,N)
    mu_mean = mus.mean(0)
    epistemic = mus.var(0).sum(1, keepdim=True)   # variance across passes
    return mu_mean, epistemic


# ======================================================================
# 5. Self-test
# ======================================================================
if __name__ == "__main__":
    import os
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    model = BayCENet().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BayCE-Net parameters: {n_params:,}")

    # use the real dataset if present, else a dummy batch
    if os.path.exists("bayce_dataset.npz"):
        ds = CEDataset("bayce_dataset.npz", "train")
        dl = DataLoader(ds, batch_size=32, shuffle=True)
        x, y = next(iter(dl))
    else:
        x = torch.randn(32, 3, 1024)
        y = torch.randn(32, 2, 1024)
    x, y = x.to(dev), y.to(dev)

    # forward + backward
    mu, log_var, w = model(x)
    loss = gaussian_nll_loss(mu, log_var, y)
    loss.backward()
    print(f"forward OK: mu{tuple(mu.shape)}  log_var{tuple(log_var.shape)}")
    print(f"loss = {loss.item():.4f}  (backward OK)")

    # MC-Dropout
    mu_m, epi = mc_predict(model, x, T=10)
    print(f"MC-Dropout OK: epistemic mean = {epi.mean().item():.4e}")
    print("\nAll checks passed ✅")
