"""
bayce_model_v2.py  —  Improved BayCE-Net (dilated, higher capacity)
==================================================================
Why v2: the dense WATERMARK channel has WIDE frequency correlations.
The v1 backbone (kernel 3, no dilation) saw only ~21 subcarriers of
1024. v2 uses DILATED convolutions -> receptive field ~305 subcarriers,
so it can actually model the dense channel structure.

Drop-in replacement: same class names, same dataset, same loss.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------- dataset
class CEDataset(Dataset):
    def __init__(self, npz_path, split="train",
                 train_frac=.7, val_frac=.1, seed=0):
        d = np.load(npz_path, allow_pickle=True)
        cid = d["chan_id"]; uniq = np.unique(cid)
        rng = np.random.default_rng(seed); rng.shuffle(uniq)
        n = len(uniq); ntr, nva = int(train_frac*n), int(val_frac*n)
        sets = {"train": uniq[:ntr], "val": uniq[ntr:ntr+nva],
                "test": uniq[ntr+nva:]}
        m = np.isin(cid, sets[split])
        Hls, s2, H = d["H_ls"][m], d["sigma2"][m], d["H_true"][m]
        self.x = np.stack([Hls.real, Hls.imag,
                           np.log(s2+1e-8)], 1).astype(np.float32)
        self.y = np.stack([H.real, H.imag], 1).astype(np.float32)
        self.scale = np.clip(np.sqrt(np.mean(np.abs(Hls)**2, 1,
                             keepdims=True)), 1e-6, None).astype(np.float32)
        print(f"[{split}] {m.sum()} samples from {len(sets[split])} channels")

    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        s = self.scale[i]; x = self.x[i].copy(); x[:2] /= s
        return torch.from_numpy(x), torch.from_numpy(self.y[i]/s)


# ---------------------------------------------------------------- model v2
class ImpulseGate(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, ch, 7, padding=3), nn.ReLU(),
            nn.Conv1d(ch, 1, 7, padding=3), nn.Sigmoid())
    def forward(self, x):
        w = self.net(x)
        gated = x[:, :2] * w; rest = x[:, 2:]
        return torch.cat([gated, rest], 1), w


class DilatedBlock(nn.Module):
    """Residual block with dilated conv -> wide receptive field."""
    def __init__(self, ch, k=5, dilation=1, p=0.1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation),
            nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation),
            nn.BatchNorm1d(ch))
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(x + self.net(x))


class BayCENet(nn.Module):
    def __init__(self, ch=128, k=5, dilations=(1, 2, 4, 8, 16, 1, 2, 4), p=0.1):
        super().__init__()
        self.gate = ImpulseGate()
        self.inp = nn.Conv1d(3, ch, k, padding=k//2)
        self.blocks = nn.ModuleList(
            [DilatedBlock(ch, k, d, p) for d in dilations])
        self.head_mu = nn.Conv1d(ch, 2, 3, padding=1)
        self.head_lv = nn.Conv1d(ch, 1, 3, padding=1)
    def forward(self, x):
        x, w = self.gate(x)
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        mu = self.head_mu(h)
        lv = torch.clamp(self.head_lv(h), -10, 10)
        return mu, lv, w


# ---------------------------------------------------------------- loss
def gaussian_nll(mu, log_var, y, l1=0.0):
    err2 = ((mu - y) ** 2).sum(1, keepdim=True)
    nll = 0.5 * (torch.exp(-log_var) * err2 + log_var)
    return nll.mean() + (l1 * mu.abs().mean() if l1 else 0.0)


@torch.no_grad()
def mc_predict(model, x, T=20):
    model.train()
    mus = torch.stack([model(x)[0] for _ in range(T)])
    return mus.mean(0), mus.var(0).sum(1, keepdim=True)


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    import os
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = BayCENet().to(dev)
    print("device:", dev)
    print(f"BayCE-Net v2 parameters: {sum(p.numel() for p in m.parameters()):,}")
    if os.path.exists("bayce_dataset.npz"):
        ds = CEDataset("bayce_dataset.npz", "train")
        x, y = next(iter(DataLoader(ds, batch_size=16, shuffle=True)))
    else:
        x, y = torch.randn(16, 3, 1024), torch.randn(16, 2, 1024)
    x, y = x.to(dev), y.to(dev)
    mu, lv, w = m(x)
    loss = gaussian_nll(mu, lv, y)
    loss.backward()
    print(f"forward+backward OK  loss={loss.item():.4f}  mu{tuple(mu.shape)}")
    print("v2 ready")
