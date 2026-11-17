"""
pilot_sweep.py  -  effect of pilot overhead on accuracy (Reviewer #8)
=====================================================================
Regenerates the LS input + reliability map at several pilot counts, RETRAINS
BayCE-Net at each, and reports test NMSE (BayCE, LS, DFT-LS) vs pilot overhead.
Answers the reviewer's request to study reducing the 342/1024 (33%) overhead.

Run:  python pilot_sweep.py
Needs: bayce_dataset.npz, bayce_model.py
Output: pilot_sweep.csv , pilot_sweep.png
"""
import numpy as np, torch, csv
import torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from bayce_model import BayCENet

PILOTS   = [171, 256, 342, 512]       # overhead 17 / 25 / 33 / 50 %  (342 = paper)
EPOCHS   = 40                         # shorter than 60 for the sweep (trend is what matters)
WARMUP, LR, BATCH, CLIP = 6, 1e-3, 64, 1.0
IMPULSIVE, IMP_P, IMP_DB = True, 0.05, 20
L = 256
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d = np.load("bayce_dataset.npz", allow_pickle=True)
Htrue, snr = d["H_true"], d["snr_db"].astype(int)
N = Htrue.shape[1]; M = Htrue.shape[0]
chan = d["chan_id"].astype(int); u=np.unique(chan)
rng=np.random.default_rng(0); rng.shuffle(u); n=len(u); a,b=int(.7*n),int(.1*n)
tr=np.isin(chan,u[:a]); te=np.isin(chan,u[a+b:])
gen=np.random.default_rng(1)

def build_inputs(P):
    """Regenerate H_ls (interp) and sigma2 for all frames with P comb pilots."""
    pidx = np.linspace(0, N-1, P, dtype=int)
    Xp   = np.exp(1j*np.random.default_rng(P).uniform(0,2*np.pi,P))
    Hls  = np.zeros((M,N),np.complex64); sig=np.zeros((M,N),np.float32)
    for k in range(M):
        H = Htrue[k]; sig_p = np.mean(np.abs(H)**2); n0 = sig_p/(10**(snr[k]/10))
        w = np.sqrt(n0/2)*(gen.standard_normal(P)+1j*gen.standard_normal(P))
        Yp = H[pidx]*Xp + w
        if IMPULSIVE:
            sI2=n0*(10**(IMP_DB/10)); mask=gen.random(P)<IMP_P
            Yp += mask*np.sqrt(sI2/2)*(gen.standard_normal(P)+1j*gen.standard_normal(P))
        Hp = Yp/Xp
        Hls[k] = np.interp(np.arange(N),pidx,Hp.real)+1j*np.interp(np.arange(N),pidx,Hp.imag)
        sig[k] = n0
    return Hls, sig

def nmse(Hhat, mask):
    num=np.mean(np.abs(Hhat[mask]-Htrue[mask])**2); den=np.mean(np.abs(Htrue[mask])**2)
    return 10*np.log10(num/den)

def dft_ls(Hls):
    h=np.fft.ifft(Hls,axis=1); h[:,L:]=0; return np.fft.fft(h,axis=1)

def train_eval(Hls, sig):
    scale=np.clip(np.sqrt(np.mean(np.abs(Hls)**2,1,keepdims=True)),1e-6,None).astype(np.float32)
    X=np.stack([Hls.real/scale,Hls.imag/scale,np.log(sig+1e-8)],1).astype(np.float32)
    Y=np.stack([Htrue.real/scale,Htrue.imag/scale],1).astype(np.float32)
    Xtr=torch.from_numpy(X[tr]).to(DEV); Ytr=torch.from_numpy(Y[tr]).to(DEV)
    Xte=torch.from_numpy(X[te]).to(DEV); sc_te=torch.from_numpy(scale[te]).to(DEV)
    torch.manual_seed(0)
    m=BayCENet().to(DEV); opt=torch.optim.Adam(m.parameters(),lr=LR)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS); idx=np.arange(len(Xtr))
    for ep in range(EPOCHS):
        m.train(); np.random.shuffle(idx)
        for i in range(0,len(idx),BATCH):
            j=idx[i:i+BATCH]; o=m(Xtr[j]); mu,lv=o[0],o[1]
            if ep<WARMUP: loss=((mu-Ytr[j])**2).mean()
            else:
                e2=((mu-Ytr[j])**2).sum(1,keepdim=True); loss=(e2/(2*torch.exp(lv))+0.5*lv).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),CLIP); opt.step()
        sch.step()
    m.eval()
    with torch.no_grad():
        mu=torch.zeros(len(Xte),2,N,device=DEV)
        for i in range(0,len(Xte),256):
            o=m(Xte[i:i+256]); mu[i:i+256]=o[0]
        Hhat=((mu[:,0]+1j*mu[:,1])*sc_te).cpu().numpy()
    num=np.mean(np.abs(Hhat-Htrue[te])**2); den=np.mean(np.abs(Htrue[te])**2)
    return 10*np.log10(num/den)

rows=[]
for P in PILOTS:
    Hls,sig=build_inputs(P)
    ber_ls = nmse(Hls, te)
    ber_dft= nmse(dft_ls(Hls), te)
    ber_bay= train_eval(Hls, sig)
    ov = 100.0*P/N
    print(f"P={P:4d} ({ov:4.1f}%)  LS={ber_ls:6.2f}  DFT-LS={ber_dft:6.2f}  BayCE={ber_bay:6.2f} dB")
    rows.append({"pilots":P,"overhead_%":round(ov,1),
                 "LS_NMSE":round(ber_ls,2),"DFTLS_NMSE":round(ber_dft,2),"BayCE_NMSE":round(ber_bay,2)})

with open("pilot_sweep.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("saved pilot_sweep.csv")

ov=[r["overhead_%"] for r in rows]
fig,ax=plt.subplots(figsize=(6.2,4.6))
ax.plot(ov,[r["BayCE_NMSE"] for r in rows],"o-",color="#0072B2",lw=2.6,ms=9,label="BayCE-Net")
ax.plot(ov,[r["DFTLS_NMSE"] for r in rows],"s--",color="#E69F00",lw=2,ms=7,label="DFT-LS")
ax.plot(ov,[r["LS_NMSE"] for r in rows],"x:",color="#56B4E9",lw=1.8,ms=8,label="LS")
ax.axvline(33.4,color="#aaa",ls=":"); ax.text(33.8,ax.get_ylim()[0]+1,"proposed (342 pilots)",fontsize=8,color="#777")
ax.set_xlabel("pilot overhead (%)"); ax.set_ylabel("NMSE (dB)")
ax.set_title("Accuracy vs pilot overhead"); ax.grid(True,ls=":",color="#ddd"); ax.legend()
fig.tight_layout(); fig.savefig("pilot_sweep.png",dpi=300); print("saved pilot_sweep.png")
