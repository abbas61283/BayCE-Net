"""
student_t.py  -  Gaussian vs Student-t likelihood (Reviewer #7)
===============================================================
Trains BayCE-Net with (a) the Gaussian NLL used in the paper and (b) a heavy-tailed
Student-t NLL (learnable dof nu), and compares test NMSE, calibration (ECE), and the
uncertainty-error correlation. This justifies -- or revises -- the Gaussian choice
under impulsive noise.

Run:  python student_t.py
Needs: bayce_dataset.npz, bayce_model.py
Output: student_t.csv + printed table
"""
import numpy as np, torch, csv, math
import torch.nn as nn
from bayce_model import BayCENet

EPOCHS, WARMUP, LR, BATCH, CLIP = 60, 8, 1e-3, 64, 1.0
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d=np.load("bayce_dataset.npz",allow_pickle=True)
H,Hls,s2=d["H_true"],d["H_ls"],d["sigma2"]; N=H.shape[1]
chan=d["chan_id"].astype(int); u=np.unique(chan); rng=np.random.default_rng(0); rng.shuffle(u)
n=len(u); a,b=int(.7*n),int(.1*n); tr=np.isin(chan,u[:a]); te=np.isin(chan,u[a+b:])
scale=np.clip(np.sqrt(np.mean(np.abs(Hls)**2,1,keepdims=True)),1e-6,None).astype(np.float32)
X=np.stack([Hls.real/scale,Hls.imag/scale,np.log(s2+1e-8)],1).astype(np.float32)
Y=np.stack([H.real/scale,H.imag/scale],1).astype(np.float32)
Xtr,Ytr=torch.from_numpy(X[tr]).to(DEV),torch.from_numpy(Y[tr]).to(DEV)
Xte,Yte=torch.from_numpy(X[te]).to(DEV),torch.from_numpy(Y[te]).to(DEV)
sc_te=torch.from_numpy(scale[te]).to(DEV); Hte=H[te]

def gauss_nll(mu,lv,y):
    e2=((mu-y)**2).sum(1,keepdim=True); return (e2/(2*torch.exp(lv))+0.5*lv).mean()

def studt_nll(mu,lv,y,nu):
    # per-component Student-t; per-comp variance-scale^2 = exp(lv)/2
    s2c=torch.exp(lv)/2.0; e=(mu-y)                      # (B,2,N), s2c (B,1,N)
    logc=torch.lgamma((nu+1)/2)-torch.lgamma(nu/2)-0.5*torch.log(nu*math.pi)
    ll=logc-0.5*torch.log(s2c)-((nu+1)/2)*torch.log1p((e**2)/(nu*s2c))
    return -(ll.sum(1,keepdim=True)).mean()

def train(kind):
    torch.manual_seed(0)
    m=BayCENet().to(DEV); nu_raw=torch.zeros(1,device=DEV,requires_grad=(kind=="t"))
    params=list(m.parameters())+([nu_raw] if kind=="t" else [])
    opt=torch.optim.Adam(params,lr=LR); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    idx=np.arange(len(Xtr))
    for ep in range(EPOCHS):
        m.train(); np.random.shuffle(idx)
        for i in range(0,len(idx),BATCH):
            nu=torch.nn.functional.softplus(nu_raw)+2.0     # recompute per batch (fresh graph)
            j=idx[i:i+BATCH]; o=m(Xtr[j]); mu,lv=o[0],o[1]
            if ep<WARMUP: loss=((mu-Ytr[j])**2).mean()
            elif kind=="gauss": loss=gauss_nll(mu,lv,Ytr[j])
            else: loss=studt_nll(mu,lv,Ytr[j],nu)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),CLIP); opt.step()
        sch.step()
    nu_val=float(torch.nn.functional.softplus(nu_raw)+2.0) if kind=="t" else float("inf")
    # ---- eval ----
    m.eval()
    with torch.no_grad():
        mu=torch.zeros(len(Xte),2,N,device=DEV); lv=torch.zeros(len(Xte),1,N,device=DEV)
        for i in range(0,len(Xte),256):
            o=m(Xte[i:i+256]); mu[i:i+256]=o[0]; lv[i:i+256]=o[1]
    mu=mu.cpu().numpy(); lv=lv.cpu().numpy()
    Hhat=(mu[:,0]+1j*mu[:,1])*scale[te]
    nmse=10*np.log10(np.mean(np.abs(Hhat-Hte)**2)/np.mean(np.abs(Hte)**2))
    # predictive variance: Gaussian=exp(lv); t = exp(lv)*nu/(nu-2)
    vtot=np.exp(lv) if kind=="gauss" else np.exp(lv)*nu_val/(nu_val-2)
    pred_std=np.sqrt(vtot).ravel(); act=np.sqrt(((mu-Y[te])**2).sum(1,keepdims=True)).ravel()
    qs=np.quantile(pred_std,np.linspace(0,1,13)); qs[-1]+=1e-9; bp,ba,bw=[],[],[]
    for i in range(12):
        sel=(pred_std>=qs[i])&(pred_std<qs[i+1])
        if sel.sum()<10: continue
        bp.append(pred_std[sel].mean()); ba.append(np.sqrt((act[sel]**2).mean())); bw.append(sel.mean())
    bp,ba,bw=map(np.array,(bp,ba,bw)); ece=float(np.sum(bw*np.abs(bp-ba))/(ba.mean()+1e-12))
    corr=float(np.corrcoef(pred_std,act)[0,1])
    return dict(likelihood=kind,nu=("%.2f"%nu_val if kind=="t" else "inf"),
                NMSE_dB=round(float(nmse),2),ECE=round(ece,4),corr=round(corr,3))

rows=[train("gauss"),train("t")]
for r in rows: print(r)
with open("student_t.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("saved student_t.csv")
print("Read: if Student-t gives lower ECE / similar NMSE -> heavy tails help calibration under impulses;")
print("      if nu is large and metrics ~equal -> Gaussian is justified (tails not needed at the error level).")
