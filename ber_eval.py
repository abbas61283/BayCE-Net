"""
ber_eval.py  -  coded-free (uncoded) BER vs SNR for every estimator (Reviewer point #3)
=======================================================================================
The journal's own papers (Zhang 2022, Applied Acoustics; Wang 2024, JMSE) report BER
vs SNR as the PRIMARY metric. This script adds it: for each channel estimator it
equalises the received QPSK symbols with the estimated channel, demodulates, and
counts bit errors, producing BER(SNR) for:

    LS, DFT-LS, LMMSE, BayCE-Net, DetTwin, ChannelNet, CRNet, DeSA-DNN, Perfect-CSI

It reuses the SAME channels (WATERMARK H_true), pilot pattern, and noise model as
training, so BER and NMSE are consistent. Impulsive noise (p=0.05) is ON by default
to match the paper's operating point; set IMPULSIVE=False for the clean case.

Run (same env as training):
    python ber_eval.py
Needs: bayce_dataset.npz, bayce_model.py, baselines.py,
       bayce_best.pt, base_dettwin.pt, base_channelnet.pt, base_crnet.pt, base_desadnn.pt
Output: ber_vs_snr.csv  +  ber_vs_snr.png
"""
import numpy as np, torch, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bayce_model import BayCENet
from baselines import MODELS

# ---------------- config ----------------
IMPULSIVE   = True            # True: p=0.05 impulsive + Gaussian (paper's condition)
IMP_PROB    = 0.05
IMP_DB      = 20              # impulse power dB above Gaussian level
N_TRIALS    = 60             # random data frames per (channel, SNR)  -> more = lower reachable BER
SEED        = 0
DEV         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng         = np.random.default_rng(SEED)
BLUE="#0072B2"; ORA="#E69F00"; GRN="#009E73"; RED="#D55E00"; PUR="#CC79A7"; GRY="#555555"; SKY="#56B4E9"; YEL="#F0E442"

# ---------------- data ----------------
d = np.load("bayce_dataset.npz", allow_pickle=True)
N          = int(d["H_true"].shape[1])
pilot_idx  = d["pilot_idx"].astype(int)
X_pilot    = d["X_pilot"]
snr_list   = sorted(set(int(s) for s in d["snr_db"]))
L          = 256                                   # channel length for DFT-LS truncation
data_idx   = np.setdiff1d(np.arange(N), pilot_idx)
prof       = d["noise_profile"].astype(float) if "noise_profile" in d else np.ones(N)

# one H_true (CFR) per unique test channel  (H_true is constant per chan_id)
chan = d["chan_id"].astype(int); uch = np.unique(chan)
rng2 = np.random.default_rng(0); order = uch.copy(); rng2.shuffle(order)
n = len(order); a, b = int(.7*n), int(.1*n); test_ch = order[a+b:]
H_by_ch = {c: d["H_true"][np.argmax(chan == c)] for c in test_ch}   # first frame of each test channel
print(f"N={N}, pilots={len(pilot_idx)}, test channels={len(test_ch)}, SNRs={snr_list}")

# ---------------- helpers ----------------
def qpsk(bits):                                    # bits:(M,2)->(M,) Gray QPSK
    return ((1 - 2*bits[:, 0]) + 1j*(1 - 2*bits[:, 1])) / np.sqrt(2)

def qpsk_demod(sym):                               # (M,)->bits (M,2)
    return np.stack([(sym.real < 0).astype(int), (sym.imag < 0).astype(int)], 1)

def add_noise(y, snr_db):
    sig_p = np.mean(np.abs(y)**2)
    n0 = sig_p / (10**(snr_db/10))
    nk = n0 * prof
    w = np.sqrt(nk/2)*(rng.standard_normal(y.shape)+1j*rng.standard_normal(y.shape))
    out = y + w
    if IMPULSIVE:
        sI2 = n0*(10**(IMP_DB/10)); mask = rng.random(y.shape) < IMP_PROB
        out += mask*np.sqrt(sI2/2)*(rng.standard_normal(y.shape)+1j*rng.standard_normal(y.shape))
    return out, nk

def ls_interp(Yp):                                 # LS at pilots -> full-band interp
    Hp = Yp / X_pilot
    return (np.interp(np.arange(N), pilot_idx, Hp.real)
            + 1j*np.interp(np.arange(N), pilot_idx, Hp.imag))

def dft_ls(H_ls):                                  # DFT denoise: keep first L taps
    h = np.fft.ifft(H_ls); h[L:] = 0
    return np.fft.fft(h)

def lmmse(H_ls, nk):                               # simple diagonal Wiener
    Rhh = np.mean(np.abs(H_ls)**2); noise = np.mean(nk/ (np.abs(X_pilot)**2).mean())
    return (Rhh/(Rhh+noise+1e-12))*H_ls

# ---------------- load models ----------------
def load(name, ckpt):
    m = (BayCENet() if name=="bayce" else MODELS[name]()).to(DEV)
    try:
        m.load_state_dict(torch.load(ckpt, map_location=DEV)); print(f"loaded {ckpt}")
    except Exception as e:
        print(f"!! could not load {ckpt}: {e}  (skipping {name})"); return None
    m.eval(); return m

models = {}
for name, ckpt in [("bayce","bayce_best.pt"),("dettwin","base_dettwin.pt"),
                   ("channelnet","base_channelnet.pt"),("crnet","base_crnet.pt"),
                   ("desadnn","base_desadnn.pt")]:
    mm = load(name, ckpt)
    if mm is not None: models[name] = mm

@torch.no_grad()
def dl_estimate(model, H_ls, sigma2):              # H_ls,sigma2: (B,N) -> Ĥ (B,N)
    scale = np.clip(np.sqrt(np.mean(np.abs(H_ls)**2,1,keepdims=True)),1e-6,None).astype(np.float32)
    x = np.stack([H_ls.real/scale, H_ls.imag/scale, np.log(sigma2+1e-8)],1).astype(np.float32)
    out = model(torch.from_numpy(x).to(DEV))
    mu = (out[0] if isinstance(out,(tuple,list)) else out).cpu().numpy()   # (B,2,N)
    return (mu[:,0]+1j*mu[:,1])*scale                                       # unscale

# ---------------- BER loop ----------------
EST = ["LS","DFT-LS"] + [n.upper() for n in models] + ["Perfect-CSI"]   # LMMSE removed (see NMSE table)
err = {e:{s:0 for s in snr_list} for e in EST}
tot = {e:{s:0 for s in snr_list} for e in EST}

for s in snr_list:
    for c in test_ch:
        H = H_by_ch[c]
        # build N_TRIALS frames for this channel/SNR
        Hls_b=np.zeros((N_TRIALS,N),complex); sig_b=np.zeros((N_TRIALS,N))
        Yb=np.zeros((N_TRIALS,N),complex); bits_b=np.zeros((N_TRIALS,len(data_idx),2),int)
        for t in range(N_TRIALS):
            bits = rng.integers(0,2,size=(len(data_idx),2)); bits_b[t]=bits
            X=np.zeros(N,complex); X[data_idx]=qpsk(bits); X[pilot_idx]=X_pilot
            Y,nk = add_noise(H*X, s); Yb[t]=Y
            Hls_b[t]=ls_interp(Y[pilot_idx])
            var_p = nk[pilot_idx]/(np.abs(X_pilot)**2)
            sig_b[t]=np.interp(np.arange(N),pilot_idx,var_p)
        # ---- estimates per method ----
        est_H={}
        est_H["LS"]=Hls_b
        est_H["DFT-LS"]=np.stack([dft_ls(Hls_b[t]) for t in range(N_TRIALS)])
        for name,mm in models.items():
            est_H[name.upper()]=dl_estimate(mm,Hls_b,sig_b)
        est_H["Perfect-CSI"]=np.tile(H,(N_TRIALS,1))
        # ---- equalise data subcarriers, demod, count ----
        for e in EST:
            Hd=est_H[e][:,data_idx]; Yd=Yb[:,data_idx]
            Xhat=Yd/np.where(np.abs(Hd)<1e-6,1e-6,Hd)
            bhat=np.stack([(Xhat.real<0).astype(int),(Xhat.imag<0).astype(int)],-1)
            err[e][s]+=int(np.sum(bhat!=bits_b)); tot[e][s]+=bits_b.size
    print(f"SNR {s:2d} dB done")

# ---------------- save + plot ----------------
ber={e:[max(err[e][s]/tot[e][s],1e-6) for s in snr_list] for e in EST}
with open("ber_vs_snr.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["SNR"]+EST)
    for i,s in enumerate(snr_list): w.writerow([s]+[f"{ber[e][i]:.3e}" for e in EST])
print("saved ber_vs_snr.csv")

BLK="#000000"
# style: marker, color, linestyle, linewidth, markersize ; BayCE drawn last & bold so it sits on top
sty={"LS":("x",SKY,":",1.8,6),"DFT-LS":("s",ORA,"--",2.0,7),
     "DETTWIN":("^",GRN,"-",1.4,6),"CHANNELNET":("D",PUR,"-",1.4,6),
     "CRNET":("*",RED,"-",1.6,8),"DESADNN":("P",GRY,"-",1.4,6),
     "Perfect-CSI":("x",BLK,"--",1.8,6),
     "BAYCE":("o",BLUE,"-",3.4,10)}            # proposed model: thick, large, on top
label={"BAYCE":"BayCE-Net (proposed)","DETTWIN":"DetTwin","CHANNELNET":"ChannelNet",
       "CRNET":"CRNet","DESADNN":"DeSA-DNN","Perfect-CSI":"Perfect CSI (bound)"}
order=[e for e in EST if e!="BAYCE"]+["BAYCE"]     # BayCE last -> highest zorder
fig,ax=plt.subplots(figsize=(6.8,5.0))
for e in order:
    if e not in ber: continue
    mk,co,ls,lw,ms=sty.get(e,("o","#333","-",2,6))
    z=20 if e=="BAYCE" else 3
    ax.semilogy(snr_list,ber[e],marker=mk,color=co,ls=ls,lw=lw,ms=ms,
                label=label.get(e,e),zorder=z,
                markeredgecolor="white" if e=="BAYCE" else co,
                markeredgewidth=1.1 if e=="BAYCE" else 0.6)
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("uncoded BER")
ax.set_title(f"BER vs SNR ({'impulsive, p = 0.05' if IMPULSIVE else 'Gaussian noise'})")
ax.grid(True,which="both",ls=":",color="#ddd")
# legend ordered with BayCE first for emphasis
h,l=ax.get_legend_handles_labels(); idx=l.index("BayCE-Net (proposed)")
h=[h[idx]]+h[:idx]+h[idx+1:]; l=[l[idx]]+l[:idx]+l[idx+1:]
ax.legend(h,l,fontsize=8,ncol=2,loc="lower left")
fig.tight_layout(); fig.savefig("ber_vs_snr.png",dpi=300)
print("saved ber_vs_snr.png")
