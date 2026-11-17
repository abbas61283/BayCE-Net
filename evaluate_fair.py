"""
evaluate_fair.py  —  fair-baseline comparison (adds DFT-based LS)
================================================================
Adds the STRONG classical baseline (DFT-based LS: solve the channel in
the delay domain instead of naive linear interpolation) and compares
LS-linear, LS-DFT, LMMSE, BayCE-Net per-SNR on both datasets.

This tells the TRUE NMSE standing of BayCE-Net vs a fair baseline.

Run (no arguments):
    python evaluate_fair.py
"""
import os, numpy as np, torch
from bayce_model import BayCENet

CKPT="bayce_best.pt"; DEV="cuda" if torch.cuda.is_available() else "cpu"
FILES={"with impulsive":"bayce_dataset.npz","no impulsive":"bayce_noimp.npz"}

def nmse_db(e,r): return 10*np.log10(np.mean(np.abs(e-r)**2)/np.mean(np.abs(r)**2)+1e-12)
def test_split(cid):
    u=np.unique(cid); rng=np.random.default_rng(0); rng.shuffle(u)
    n=len(u); a,b=int(.7*n),int(.1*n); return np.isin(cid,u[a+b:])

model=BayCENet().to(DEV); model.load_state_dict(torch.load(CKPT,map_location=DEV)); model.eval()

for label,path in FILES.items():
    if not os.path.exists(path):
        print(f"!! missing {path}"); continue
    d=np.load(path,allow_pickle=True); m=test_split(d["chan_id"])
    H,Hls,s2,snr=d["H_true"][m],d["H_ls"][m],d["sigma2"][m],d["snr_db"][m]
    N=H.shape[1]; L=d["h_true"].shape[1]; pilot_idx=d["pilot_idx"]
    # DFT submatrix for LS-DFT (N x L), pilots rows
    Ffull=np.exp(-2j*np.pi*np.outer(np.arange(N),np.arange(L))/N)
    Fp=Ffull[pilot_idx,:]; Fp_pinv=np.linalg.pinv(Fp)   # precompute

    # BayCE inputs
    scale=np.clip(np.sqrt(np.mean(np.abs(Hls)**2,1,keepdims=True)),1e-6,None).astype(np.float32)
    x=np.stack([Hls.real/scale,Hls.imag/scale,np.log(s2+1e-8)],1).astype(np.float32)
    mu=np.zeros((len(x),2,N),np.float32)
    with torch.no_grad():
        for i in range(0,len(x),256):
            mu[i:i+256]=model(torch.from_numpy(x[i:i+256]).to(DEV))[0].cpu().numpy()
    Hbc=(mu[:,0]+1j*mu[:,1])*scale

    print(f"\n=== {label} ({path}) ===")
    print(f"{'SNR':>4} | {'LS-lin':>7} {'LS-DFT':>7} {'LMMSE':>7} {'BayCE':>7}")
    for s in sorted(np.unique(snr)):
        j=np.where(snr==s)[0]
        ls=nmse_db(Hls[j],H[j])
        # LS-DFT: recover raw pilot LS values, solve to L taps, reconstruct
        Hp=Hls[j][:,pilot_idx]                       # (nj, P) raw LS at pilots
        hhat=Hp@Fp_pinv.T                            # (nj, L)
        Hdft=hhat@Ffull.T                            # (nj, N)
        dft=nmse_db(Hdft,H[j])
        Rhh=np.mean(np.abs(H[j])**2); nz=np.mean(np.abs(Hls[j]-H[j])**2)
        lm=nmse_db((Rhh/(Rhh+nz))*Hls[j],H[j])
        bc=nmse_db(Hbc[j],H[j])
        print(f"{s:4.0f} | {ls:7.2f} {dft:7.2f} {lm:7.2f} {bc:7.2f}")
