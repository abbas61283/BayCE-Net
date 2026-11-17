"""
master_eval.py  -  ONE consistent NMSE comparison for ALL methods (resolves the condition ambiguity)
===================================================================================================
Evaluates every estimator on the SAME held-out frames under TWO clearly-labelled noise
conditions, so Table II / Fig. 3 are internally consistent and you know exactly what the
numbers mean:

    clean      : additive Gaussian noise only
    impulsive  : Gaussian + Bernoulli-Gaussian impulses (p = 0.05, 20 dB)

Methods: LS, DFT-LS, LMMSE, DeSA-DNN, CRNet, ChannelNet, DetTwin, BayCE-Net, robust-ADMM
All methods see the IDENTICAL noisy pilot observation per frame -> a fair, consistent table.

Run:  python master_eval.py
Needs: bayce_dataset.npz, bayce_model.py, baselines.py,
       bayce_best.pt, base_dettwin.pt, base_channelnet.pt, base_crnet.pt, base_desadnn.pt
Outputs: master_clean.csv , master_impulsive.csv , master_clean.pdf , master_impulsive.pdf
"""
import numpy as np, torch, csv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from bayce_model import BayCENet
from baselines import MODELS

N_EVAL = 250                 # frames per SNR (same subset for every method; ADMM is iterative)
L = 256; IMP_P, IMP_DB = 0.05, 20
LAM_H, LAM_E, OUTER, INNER = 0.03, 0.08, 30, 5
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d=np.load("bayce_dataset.npz",allow_pickle=True)
Htrue=d["H_true"]; snr=d["snr_db"].astype(int); N=Htrue.shape[1]
pilot_idx=d["pilot_idx"].astype(int); Xp=d["X_pilot"]
chan=d["chan_id"].astype(int); u=np.unique(chan); rng=np.random.default_rng(0); rng.shuffle(u)
nn_=len(u); a,b=int(.7*nn_),int(.1*nn_); te=np.where(np.isin(chan,u[a+b:]))[0]
snr_list=sorted(set(snr)); gen=np.random.default_rng(7)

# robust-ADMM operator
l=np.arange(L)[None,:]; Fp=np.exp(-2j*np.pi*pilot_idx[:,None]*l/N); mu=1/(np.linalg.norm(Fp,2)**2+1e-9); FpH=Fp.conj().T
def soft(x,t): m=np.abs(x); return np.where(m>t,(m-t)/np.maximum(m,1e-12)*x,0.0)
def robust_admm(z):
    h=np.zeros(L,complex); e=np.zeros(len(z),complex)
    for _ in range(OUTER):
        r=z-e
        for _ in range(INNER): h=soft(h-mu*(FpH@(Fp@h-r)),mu*LAM_H)
        e=soft(z-Fp@h,LAM_E)
    return np.fft.fft(np.concatenate([h,np.zeros(N-L,complex)]))
Fp_pinv=np.linalg.pinv(Fp)                 # (L,P) proper delay-domain LS fit
def dft_ls(z):                             # z = LS at pilots -> least-squares channel taps -> full CFR
    h=Fp_pinv@z; return np.fft.fft(np.concatenate([h,np.zeros(N-L,complex)]))
def lmmse(H,n0): R=np.mean(np.abs(H)**2); return (R/(R+n0+1e-12))*H

# DL models
def load(name,ckpt):
    m=(BayCENet() if name=="bayce" else MODELS[name]()).to(DEV)
    try: m.load_state_dict(torch.load(ckpt,map_location=DEV)); m.eval(); return m
    except Exception as ex: print(f"skip {name}: {ex}"); return None
dls={}
for nm,ck in [("bayce","bayce_best.pt"),("desadnn","base_desadnn.pt"),("crnet","base_crnet.pt"),
              ("channelnet","base_channelnet.pt"),("dettwin","base_dettwin.pt")]:
    m=load(nm,ck);
    if m: dls[nm]=m
@torch.no_grad()
def dl_batch(m,Hls,sig):
    sc=np.clip(np.sqrt(np.mean(np.abs(Hls)**2,1,keepdims=True)),1e-6,None).astype(np.float32)
    x=np.stack([Hls.real/sc,Hls.imag/sc,np.log(sig+1e-8)],1).astype(np.float32)
    o=m(torch.from_numpy(x).to(DEV)); mm=(o[0] if isinstance(o,(tuple,list)) else o).cpu().numpy()
    return (mm[:,0]+1j*mm[:,1])*sc

NAMES=["LS","DFT-LS","LMMSE","DeSA-DNN","CRNet","ChannelNet","DetTwin","BayCE-Net","robust-ADMM"]
key={"DeSA-DNN":"desadnn","CRNet":"crnet","ChannelNet":"channelnet","DetTwin":"dettwin","BayCE-Net":"bayce"}

def run(impulsive):
    out={n:{s:[] for s in snr_list} for n in NAMES}
    for s in snr_list:
        idx=te[snr[te]==s]
        if len(idx)>N_EVAL: idx=gen.choice(idx,N_EVAL,replace=False)
        Hls_b=np.zeros((len(idx),N),complex); sig_b=np.zeros((len(idx),N))
        for ii,k in enumerate(idx):
            H=Htrue[k]; fp=np.mean(np.abs(H)**2); n0=fp/(10**(s/10))
            w=np.sqrt(n0/2)*(gen.standard_normal(len(pilot_idx))+1j*gen.standard_normal(len(pilot_idx)))
            y=H[pilot_idx]*Xp+w
            if impulsive:
                sI2=n0*(10**(IMP_DB/10)); mask=gen.random(len(pilot_idx))<IMP_P
                y+=mask*np.sqrt(sI2/2)*(gen.standard_normal(len(pilot_idx))+1j*gen.standard_normal(len(pilot_idx)))
            Hp=y/Xp
            Hls=np.interp(np.arange(N),pilot_idx,Hp.real)+1j*np.interp(np.arange(N),pilot_idx,Hp.imag)
            Hls_b[ii]=Hls; sig_b[ii]=n0
            def nm(Hh): return np.mean(np.abs(Hh-H)**2)/np.mean(np.abs(H)**2)
            out["LS"][s].append(nm(Hls)); out["DFT-LS"][s].append(nm(dft_ls(Hp)))
            out["LMMSE"][s].append(nm(lmmse(Hls,n0))); out["robust-ADMM"][s].append(nm(robust_admm(Hp)))
        for disp,k2 in key.items():
            if k2 in dls:
                Hh=dl_batch(dls[k2],Hls_b,sig_b)
                for ii,kk in enumerate(idx): out[disp][s].append(np.mean(np.abs(Hh[ii]-Htrue[kk])**2)/np.mean(np.abs(Htrue[kk])**2))
        print(f"[{'imp' if impulsive else 'cln'}] SNR {s:2d}  BayCE={10*np.log10(np.mean(out['BayCE-Net'][s])):6.2f}  DFT-LS={10*np.log10(np.mean(out['DFT-LS'][s])):6.2f}  ADMM={10*np.log10(np.mean(out['robust-ADMM'][s])):6.2f}")
    return out

def save(out,tag):
    rows=[{"SNR":s,**{n:round(float(10*np.log10(np.mean(out[n][s]))),2) for n in NAMES if out[n][s]}} for s in snr_list]
    cols=["SNR"]+[n for n in NAMES if out[n][snr_list[0]]]
    with open(f"master_{tag}.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
    sty={"LS":("x","#56B4E9",":"),"DFT-LS":("s","#E69F00","--"),"LMMSE":("v","#999999",":"),
         "DeSA-DNN":("P","#555555","-"),"CRNet":("*","#D55E00","-"),"ChannelNet":("D","#CC79A7","-"),
         "DetTwin":("^","#009E73","-"),"robust-ADMM":("h","#8B4513","-"),"BayCE-Net":("o","#0072B2","-")}
    fig,ax=plt.subplots(figsize=(7.0,5.2))
    for n in NAMES:
        if not out[n][snr_list[0]]: continue
        mk,c,ls=sty[n]; y=[10*np.log10(np.mean(out[n][s])) for s in snr_list]
        lw=3.2 if n=="BayCE-Net" else 1.5; ms=10 if n=="BayCE-Net" else 6; z=20 if n=="BayCE-Net" else 3
        ax.plot(snr_list,y,marker=mk,color=c,ls=ls,lw=lw,ms=ms,label=n,zorder=z)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("NMSE (dB)")
    ax.set_title(f"NMSE vs SNR ({'impulsive, p=0.05' if tag=='impulsive' else 'Gaussian noise'})")
    ax.grid(True,ls=":",color="#ddd"); ax.legend(fontsize=8,ncol=2)
    fig.tight_layout(); fig.savefig(f"master_{tag}.pdf"); fig.savefig(f"master_{tag}.png",dpi=300)
    print(f"saved master_{tag}.csv / .pdf / .png")

save(run(False),"clean")
save(run(True),"impulsive")
print("\nDONE. Use master_clean for the accuracy table (Table II / Fig 3);")
print("     use master_impulsive for the robustness comparison (includes robust-ADMM).")
