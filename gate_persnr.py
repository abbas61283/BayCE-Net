"""
gate_persnr.py  —  does the impulse gate help at LOW SNR specifically?
=====================================================================
Trains Full and no_gate, then compares NMSE PER SNR (not averaged),
because the averaged ablation hides regime-specific effects. The gate
was hypothesised to matter under impulsive noise at low-mid SNR.

Run: python gate_persnr.py
"""
import numpy as np, torch
from torch.utils.data import DataLoader
from bayce_model import CEDataset, BayCENet, gaussian_nll_loss

DATA="bayce_dataset.npz"; DEV="cuda" if torch.cuda.is_available() else "cpu"
EPOCHS=40; WARMUP=6; LR=1e-3; BATCH=64
torch.manual_seed(0); np.random.seed(0)

def loaders():
    tr=CEDataset(DATA,"train"); va=CEDataset(DATA,"val")
    return DataLoader(tr,BATCH,shuffle=True,drop_last=True), DataLoader(va,BATCH)

def build(no_gate):
    m=BayCENet().to(DEV)
    if no_gate:
        def fwd(x,_m=m):
            feat=_m.backbone(x); mu=_m.head_mu(feat)
            lv=torch.clamp(_m.head_lv(feat),-10,10)
            return mu,lv,torch.ones_like(x[:,:1])
        m.forward=fwd
    return m

def train(m,tr,va):
    opt=torch.optim.Adam(m.parameters(),lr=LR)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
    best=1e9; state=None
    for ep in range(1,EPOCHS+1):
        m.train()
        for x,y in tr:
            x,y=x.to(DEV),y.to(DEV); opt.zero_grad()
            mu,lv,_=m(x)
            loss=gaussian_nll_loss(mu,lv,y,0.0) if ep>WARMUP else ((mu-y)**2).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        sch.step()
        m.eval(); num=den=0.0
        with torch.no_grad():
            for x,y in va:
                x,y=x.to(DEV),y.to(DEV); mu,_,_=m(x)
                num+=((mu-y)**2).sum().item(); den+=(y**2).sum().item()
        vn=10*np.log10(num/den+1e-12)
        if vn<best: best=vn; state={k:v.cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(state); return m

# per-SNR test evaluation
d=np.load(DATA,allow_pickle=True)
cid=d["chan_id"]; u=np.unique(cid); rng=np.random.default_rng(0); rng.shuffle(u)
n=len(u); a,b=int(.7*n),int(.1*n); mte=np.isin(cid,u[a+b:])
H=d["H_true"][mte]; Hls=d["H_ls"][mte]; s2=d["sigma2"][mte]; snr=d["snr_db"][mte]
scale=np.clip(np.sqrt(np.mean(np.abs(Hls)**2,1,keepdims=True)),1e-6,None).astype(np.float32)
x=np.stack([Hls.real/scale,Hls.imag/scale,np.log(s2+1e-8)],1).astype(np.float32)
def nmse(e,r): return 10*np.log10(np.mean(np.abs(e-r)**2)/np.mean(np.abs(r)**2)+1e-12)

results={}
for label,ng in [("Full",False),("no_gate",True)]:
    print(f"\n=== training {label} ===")
    tr,va=loaders(); m=train(build(ng),tr,va); m.eval()
    mu=np.zeros((len(x),2,H.shape[1]),np.float32)
    with torch.no_grad():
        for i in range(0,len(x),256):
            mu[i:i+256]=m(torch.from_numpy(x[i:i+256]).to(DEV))[0].cpu().numpy()
    Hh=(mu[:,0]+1j*mu[:,1])*scale
    results[label]={s:nmse(Hh[snr==s],H[snr==s]) for s in sorted(np.unique(snr))}

print("\n==== NMSE per SNR: Full vs no_gate ====")
print(f"{'SNR':>4} {'Full':>8} {'no_gate':>8} {'gate helps':>11}")
for s in sorted(results["Full"]):
    f=results["Full"][s]; g=results["no_gate"][s]
    print(f"{s:4.0f} {f:8.2f} {g:8.2f} {g-f:+8.2f} dB")
