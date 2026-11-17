"""
train_baselines.py  -  train the DL baselines and build the master table
========================================================================
Trains DetTwin / ChannelNet / CRNet / DeSA-DNN on the SAME NOF1 dataset
and split as BayCE-Net, then prints ONE per-SNR NMSE table comparing every
estimator (classical + all DL baselines + BayCE) on the held-out test
channels, and saves the figure. This is the master comparison of Paper 2.

Fairness: identical input/output, identical train/val/test split (seed 0,
70/10/20), identical optimizer budget (all point estimators use MSE; BayCE
was trained with its NLL objective). Use the SAME epoch budget you trained
BayCE with, via --epochs, so no model is under/over-trained relative to it.

Run (from  E:\PhDpapers\PAPER 2\code , env = paper2):
    python train_baselines.py                 # all four, 40 epochs
    python train_baselines.py --epochs 60
    python train_baselines.py --models crnet desadnn
    python train_baselines.py --eval-only     # skip training, use saved ckpts

Needs: bayce_dataset.npz, bayce_best.pt, bayce_model.py, baselines.py.
Produces: base_<name>.pt per model, baselines_nmse.png, baselines_nmse.csv
"""

import os, sys, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from bayce_model import CEDataset, BayCENet
from baselines import MODELS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = "bayce_dataset.npz"


def nmse_db(est, ref):
    return 10*np.log10(np.mean(np.abs(est-ref)**2)/np.mean(np.abs(ref)**2)+1e-12)


# ---------------------------------------------------------------- training
def train_one(name, epochs, lr, bs):
    torch.manual_seed(0)
    model = MODELS[name]().to(DEV)
    n = sum(p.numel() for p in model.parameters())
    print(f"\n=== train {name}  ({n:,} params) ===")
    tr = DataLoader(CEDataset(DATA, "train"), batch_size=bs, shuffle=True)
    va = DataLoader(CEDataset(DATA, "val"),   batch_size=256)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()

    best, best_state = 1e9, None
    for ep in range(1, epochs+1):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in tr:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward(); opt.step()
            tot += loss.item()*len(x)
        tr_loss = tot/len(tr.dataset)

        model.eval(); vtot = 0.0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(DEV), y.to(DEV)
                vtot += lossf(model(x), y).item()*len(x)
        v = vtot/len(va.dataset)
        flag = ""
        if v < best:
            best, best_state = v, {k: t.cpu().clone() for k, t in model.state_dict().items()}
            flag = " *"
        print(f"  ep {ep:3d}/{epochs}  train {tr_loss:.4e}  val {v:.4e}"
              f"  ({time.time()-t0:.0f}s){flag}")
    torch.save(best_state, f"base_{name}.pt")
    print(f"  saved base_{name}.pt  (best val {best:.4e})")


# ---------------------------------------------------------------- inference
def infer(model, x_np, wrap_bayce=False):
    mu = np.zeros((len(x_np), 2, x_np.shape[-1]), np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x_np), 256):
            xb = torch.from_numpy(x_np[i:i+256]).to(DEV)
            out = model(xb)
            if wrap_bayce: out = out[0]
            mu[i:i+256] = out.cpu().numpy()
    return mu


# ---------------------------------------------------------------- master eval
def evaluate(names):
    d = np.load(DATA, allow_pickle=True)
    u = np.unique(d["chan_id"]); rng = np.random.default_rng(0); rng.shuffle(u)
    n = len(u); a, b = int(.7*n), int(.1*n)
    m = np.isin(d["chan_id"], u[a+b:])
    H, Hls = d["H_true"][m], d["H_ls"][m]
    s2, snr = d["sigma2"][m], d["snr_db"][m]
    N = H.shape[1]; L = d["h_true"].shape[1]; pidx = d["pilot_idx"]
    print(f"\ntest: {m.sum()} samples")

    Ffull = np.exp(-2j*np.pi*np.outer(np.arange(N), np.arange(L))/N)
    Fp_pinv = np.linalg.pinv(Ffull[pidx, :])

    scale = np.clip(np.sqrt(np.mean(np.abs(Hls)**2, 1, keepdims=True)),
                    1e-6, None).astype(np.float32)
    x = np.stack([Hls.real/scale, Hls.imag/scale, np.log(s2+1e-8)],
                 1).astype(np.float32)

    # gather each estimator's full-band complex estimate
    est = {}
    est["LS-lin"] = Hls
    est["DFT-LS"] = (Hls[:, pidx] @ Fp_pinv.T) @ Ffull.T
    Rhh = np.mean(np.abs(H)**2); nz = np.mean(np.abs(Hls-H)**2)
    est["LMMSE"] = (Rhh/(Rhh+nz))*Hls

    for name in names:
        model = MODELS[name]().to(DEV)
        model.load_state_dict(torch.load(f"base_{name}.pt", map_location=DEV))
        mu = infer(model, x)
        est[name] = (mu[:, 0]+1j*mu[:, 1])*scale

    if os.path.exists("bayce_best.pt"):
        bm = BayCENet().to(DEV)
        bm.load_state_dict(torch.load("bayce_best.pt", map_location=DEV))
        mu = infer(bm, x, wrap_bayce=True)
        est["BayCE"] = (mu[:, 0]+1j*mu[:, 1])*scale

    # per-SNR table
    cols = list(est.keys())
    snrs = sorted(np.unique(snr))
    print("\n" + "="*(8+9*len(cols)))
    print("MASTER NMSE (dB) vs SNR  -  NOF1 test channels")
    print("="*(8+9*len(cols)))
    print(f"{'SNR':>4} | " + " ".join(f"{c:>8}" for c in cols))
    table = []
    for s in snrs:
        j = snr == s
        row = [nmse_db(est[c][j], H[j]) for c in cols]
        table.append([s]+row)
        print(f"{s:4.0f} | " + " ".join(f"{v:8.2f}" for v in row))

    table = np.array(table)
    # csv
    with open("baselines_nmse.csv", "w") as f:
        f.write("SNR," + ",".join(cols) + "\n")
        for r in table:
            f.write(",".join(f"{v:.3f}" for v in r) + "\n")
    print("\nsaved baselines_nmse.csv")

    # figure
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 6))
        styles = {"LS-lin": "x:", "DFT-LS": "s--", "LMMSE": "v--",
                  "dettwin": "o-", "channelnet": "P-", "crnet": "*-",
                  "desadnn": "D-", "BayCE": "^-"}
        for i, c in enumerate(cols):
            plt.plot(table[:, 0], table[:, i+1], styles.get(c, "o-"),
                     lw=2 if c in ("BayCE", "dettwin") else 1.3, label=c)
        plt.xlabel("SNR (dB)"); plt.ylabel("NMSE (dB)")
        plt.title("Master comparison: classical + DL baselines vs BayCE (NOF1 test)")
        plt.grid(alpha=.3); plt.legend(fontsize=8, ncol=2)
        plt.tight_layout(); plt.savefig("baselines_nmse.png", dpi=150)
        print("saved baselines_nmse.png")
    except Exception as e:
        print(f"(figure skipped: {e})")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                    choices=list(MODELS.keys()))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    print(f"device={DEV}  models={args.models}  epochs={args.epochs}")
    if not args.eval_only:
        for name in args.models:
            train_one(name, args.epochs, args.lr, args.bs)
    evaluate(args.models)
    print("\nDONE.")


if __name__ == "__main__":
    main()
