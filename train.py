"""
train.py  —  BayCE-Net training loop
====================================
Key features that protect you from known failure modes:
  * VARIANCE-HEAD WARM-UP: first train mu with plain MSE, THEN switch on
    the Gaussian NLL. Training mu and log-var together from scratch is the
    classic cause of Bayesian-regression instability.
  * NMSE tracked on the validation set every epoch (the metric reviewers
    read), best checkpoint saved by val-NMSE.
  * gradient clipping + LR scheduler for stability.

Run:
    conda activate paper2
    python train.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from bayce_model import CEDataset, BayCENet, gaussian_nll_loss

# ----------------------------------------------------------------------
CFG = dict(
    data      = "bayce_dataset.npz",
    epochs    = 60,
    warmup    = 8,          # epochs of MSE-only before enabling NLL
    batch     = 64,
    lr        = 1e-3,
    l1        = 1e-4,
    clip      = 1.0,
    ckpt      = "bayce_best.pt",
    seed      = 0,
)
torch.manual_seed(CFG["seed"])
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
def nmse_db(mu, y):
    num = ((mu - y) ** 2).sum().item()
    den = (y ** 2).sum().item()
    return 10 * np.log10(num / den + 1e-12)


def mse_loss(mu, log_var, y, l1):
    """Warm-up objective: plain MSE on mu, ignore variance head."""
    return ((mu - y) ** 2).mean() + l1 * mu.abs().mean()


# ----------------------------------------------------------------------
def run():
    tr = DataLoader(CEDataset(CFG["data"], "train"),
                    batch_size=CFG["batch"], shuffle=True, drop_last=True)
    va = DataLoader(CEDataset(CFG["data"], "val"),
                    batch_size=CFG["batch"])

    model = BayCENet().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG["epochs"])

    best = 1e9
    hist = []
    for ep in range(1, CFG["epochs"] + 1):
        use_nll = ep > CFG["warmup"]
        model.train()
        tl = 0.0
        for x, y in tr:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            mu, log_var, _ = model(x)
            if use_nll:
                loss = gaussian_nll_loss(mu, log_var, y, CFG["l1"])
            else:
                loss = mse_loss(mu, log_var, y, CFG["l1"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
            opt.step()
            tl += loss.item()
        sched.step()

        # ---- validation NMSE ----
        model.eval()
        num = den = 0.0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(DEV), y.to(DEV)
                mu, _, _ = model(x)
                num += ((mu - y) ** 2).sum().item()
                den += (y ** 2).sum().item()
        vnmse = 10 * np.log10(num / den + 1e-12)
        hist.append(vnmse)

        tag = "NLL" if use_nll else "MSE"
        star = ""
        if vnmse < best:
            best = vnmse
            torch.save(model.state_dict(), CFG["ckpt"])
            star = "  <- best, saved"
        print(f"ep {ep:3d} [{tag}] train {tl/len(tr):.4f} | "
              f"val NMSE {vnmse:6.2f} dB{star}")

    print(f"\nBest val NMSE = {best:.2f} dB  ->  {CFG['ckpt']}")
    return hist


if __name__ == "__main__":
    run()
