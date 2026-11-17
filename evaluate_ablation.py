"""
evaluate_ablation.py  —  impulsive-noise ablation (the key Paper-2 figure)
=========================================================================
Evaluates LS, LMMSE, BayCE-Net per-SNR on BOTH datasets:
  - bayce_dataset.npz   (WITH impulsive noise)
  - bayce_noimp.npz     (WITHOUT impulsive noise)
and overlays them, to prove that impulsive noise is what caps the
classical estimators while BayCE-Net stays robust.

Run (no arguments needed):
    python evaluate_ablation.py
"""
import os, numpy as np, torch, matplotlib.pyplot as plt
from bayce_model import BayCENet

CKPT = "bayce_best.pt"
DEV  = "cuda" if torch.cuda.is_available() else "cpu"
FILES = {"with impulsive": "bayce_dataset.npz",
         "no impulsive":   "bayce_noimp.npz"}

def nmse_db(e, r):
    return 10*np.log10(np.mean(np.abs(e-r)**2)/np.mean(np.abs(r)**2)+1e-12)

def test_split(cid):
    uniq = np.unique(cid); rng = np.random.default_rng(0); rng.shuffle(uniq)
    n = len(uniq); ntr, nva = int(.7*n), int(.1*n)
    return np.isin(cid, uniq[ntr+nva:])

model = BayCENet().to(DEV)
model.load_state_dict(torch.load(CKPT, map_location=DEV)); model.eval()

curves = {}
for label, path in FILES.items():
    if not os.path.exists(path):
        print(f"!! missing {path} — skipping '{label}'"); continue
    d = np.load(path, allow_pickle=True)
    m = test_split(d["chan_id"])
    H, Hls, s2, snr = d["H_true"][m], d["H_ls"][m], d["sigma2"][m], d["snr_db"][m]
    scale = np.clip(np.sqrt(np.mean(np.abs(Hls)**2,1,keepdims=True)),1e-6,None).astype(np.float32)
    x = np.stack([Hls.real/scale, Hls.imag/scale, np.log(s2+1e-8)],1).astype(np.float32)
    mu = np.zeros((len(x),2,H.shape[1]),np.float32)
    with torch.no_grad():
        for i in range(0,len(x),256):
            mu[i:i+256] = model(torch.from_numpy(x[i:i+256]).to(DEV))[0].cpu().numpy()
    Hbc = (mu[:,0]+1j*mu[:,1])*scale

    snrs = sorted(np.unique(snr)); rows=[]
    print(f"\n=== {label} ({path}) ===")
    print(f"{'SNR':>4} | {'LS':>7} {'LMMSE':>7} {'BayCE':>7}")
    for s in snrs:
        j = snr==s
        ls = nmse_db(Hls[j],H[j])
        Rhh=np.mean(np.abs(H[j])**2); nz=np.mean(np.abs(Hls[j]-H[j])**2)
        lm = nmse_db((Rhh/(Rhh+nz))*Hls[j],H[j])
        bc = nmse_db(Hbc[j],H[j])
        rows.append((s,ls,lm,bc)); print(f"{s:4.0f} | {ls:7.2f} {lm:7.2f} {bc:7.2f}")
    curves[label]=np.array(rows)

# ---- overlay plot: solid = with impulsive, dashed = no impulsive ----
fig,ax=plt.subplots(1,2,figsize=(13,5),sharey=True)
for k,(label,r) in enumerate(curves.items()):
    ax[k].plot(r[:,0],r[:,1],'o-',label='LS')
    ax[k].plot(r[:,0],r[:,2],'s-',label='LMMSE')
    ax[k].plot(r[:,0],r[:,3],'^-',label='BayCE-Net',lw=2)
    ax[k].set_title(label); ax[k].set_xlabel('SNR (dB)'); ax[k].grid(alpha=.3); ax[k].legend()
ax[0].set_ylabel('NMSE (dB)')
plt.suptitle('Impulsive-noise ablation: classical estimators saturate, BayCE-Net stays robust')
plt.tight_layout(); plt.savefig('ablation_impulsive.png',dpi=150)
print("\nsaved -> ablation_impulsive.png")
