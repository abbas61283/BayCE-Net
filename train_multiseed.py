"""
train_multiseed.py  -  multi-seed training for statistical significance (Reviewer #6)
=====================================================================================
Trains BayCE-Net under several random seeds (model init + batch order + dropout),
keeping the SAME 70/10/20 channel split, and reports test NMSE as mean +/- std per
SNR and overall. The saved per-seed checkpoints are REUSED as a Deep Ensemble by
uncertainty_eval.py (Reviewer #4) -> one training run serves both points.

Run (same env as training):
    python train_multiseed.py
Needs: bayce_dataset.npz, bayce_model.py (BayCENet).
Outputs: bayce_seed0.pt ... bayce_seed{K-1}.pt , multiseed_nmse.csv , printed table.
"""
import numpy as np, torch, csv
import torch.nn as nn
from bayce_model import BayCENet

SEEDS   = [0, 1, 2, 3, 4]      # 3 is acceptable if time-limited; 5 is better for the ensemble
EPOCHS  = 60
WARMUP  = 8
LR      = 1e-3
BATCH   = 64
CLIP    = 1.0
DEV     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- data (fixed split, seed 0) ----------
d = np.load("bayce_dataset.npz", allow_pickle=True)
H, Hls, s2, snr = d["H_true"], d["H_ls"], d["sigma2"], d["snr_db"].astype(int)
chan = d["chan_id"].astype(int)
u = np.unique(chan); rng = np.random.default_rng(0); rng.shuffle(u)
n = len(u); a, b = int(.7*n), int(.1*n)
tr = np.isin(chan, u[:a]); va = np.isin(chan, u[a:a+b]); te = np.isin(chan, u[a+b:])

scale = np.clip(np.sqrt(np.mean(np.abs(Hls)**2, 1, keepdims=True)), 1e-6, None).astype(np.float32)
X = np.stack([Hls.real/scale, Hls.imag/scale, np.log(s2+1e-8)], 1).astype(np.float32)   # (M,3,N)
Y = np.stack([H.real/scale,  H.imag/scale], 1).astype(np.float32)                        # (M,2,N) scaled

def to(t, m): return torch.from_numpy(t[m]).to(DEV)
Xtr, Ytr = to(X, tr), to(Y, tr)
Xte, Yte = to(X, te), to(Y, te)
scale_te = torch.from_numpy(scale[te]).to(DEV)        # (Mte,1) for un-scaling
Hte      = torch.from_numpy(H[te]).to(DEV)            # complex true CFR (test)
snr_te   = snr[te]
print(f"train {tr.sum()} | val {va.sum()} | test {te.sum()} frames | device {DEV}")

def nll(mu, lv, y):
    err2 = ((mu - y)**2).sum(1, keepdim=True)          # (B,1,N)
    return (err2/(2*torch.exp(lv)) + 0.5*lv).mean()

@torch.no_grad()
def test_nmse(model):
    model.eval()
    mu = torch.zeros_like(Yte)
    for i in range(0, len(Xte), 256):
        out = model(Xte[i:i+256]); m = out[0] if isinstance(out,(tuple,list)) else out
        mu[i:i+256] = m
    Hhat = (mu[:,0] + 1j*mu[:,1]) * scale_te            # un-scale -> complex (Mte,N)
    num = torch.mean(torch.abs(Hhat - Hte)**2, 1)
    den = torch.mean(torch.abs(Hte)**2, 1)
    nmse_f = 10*torch.log10(num/den).cpu().numpy()      # per-frame NMSE (dB)
    overall = float(10*np.log10((num.sum()/den.sum()).item()))
    per = {s: float(np.mean(nmse_f[snr_te == s])) for s in sorted(set(snr_te))}
    return overall, per

def train_one(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = BayCENet().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    idx = np.arange(len(Xtr))
    for ep in range(EPOCHS):
        model.train(); np.random.shuffle(idx)
        for i in range(0, len(idx), BATCH):
            j = idx[i:i+BATCH]; xb, yb = Xtr[j], Ytr[j]
            out = model(xb); mu = out[0]; lv = out[1]
            loss = ((mu-yb)**2).mean() if ep < WARMUP else nll(mu, lv, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step()
        sch.step()
    torch.save(model.state_dict(), f"bayce_seed{seed}.pt")
    return test_nmse(model)

# ---------- run all seeds ----------
overalls, pers = [], []
snr_vals = sorted(set(snr_te))
for s in SEEDS:
    ov, per = train_one(s); overalls.append(ov); pers.append(per)
    print(f"seed {s}: overall NMSE = {ov:6.2f} dB  | 30dB = {per[max(snr_vals)]:.2f} dB  -> saved bayce_seed{s}.pt")

ov = np.array(overalls)
print(f"\n==== OVERALL NMSE over {len(SEEDS)} seeds: {ov.mean():.2f} +/- {ov.std():.2f} dB ====")
print(f"{'SNR':>5s} {'mean(dB)':>10s} {'std(dB)':>9s}")
rows=[]
for s in snr_vals:
    col = np.array([p[s] for p in pers])
    print(f"{s:5d} {col.mean():10.2f} {col.std():9.2f}")
    rows.append({"SNR": s, "nmse_mean_dB": round(col.mean(),3), "nmse_std_dB": round(col.std(),3)})
rows.append({"SNR":"overall","nmse_mean_dB":round(ov.mean(),3),"nmse_std_dB":round(ov.std(),3)})
with open("multiseed_nmse.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["SNR","nmse_mean_dB","nmse_std_dB"]); w.writeheader(); w.writerows(rows)
print("\nsaved multiseed_nmse.csv + bayce_seed*.pt  (reused as Deep Ensemble by uncertainty_eval.py)")
