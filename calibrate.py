"""
calibrate.py  —  Bayesian uncertainty calibration (the core novelty test)
========================================================================
Checks whether BayCE-Net's predicted per-subcarrier uncertainty is
HONEST: where it reports high uncertainty, is the actual error high?

Outputs:
  - reliability diagram (predicted variance vs actual squared error)
  - ECE-like calibration error + correlation
  - recalibration scale factor (for optional temperature scaling)

Run (no arguments):
    python calibrate.py
"""
import numpy as np, torch, matplotlib.pyplot as plt
from bayce_model import BayCENet

DATA="bayce_dataset.npz"; CKPT="bayce_best.pt"
DEV="cuda" if torch.cuda.is_available() else "cpu"

d=np.load(DATA,allow_pickle=True)
cid=d["chan_id"]; u=np.unique(cid); rng=np.random.default_rng(0); rng.shuffle(u)
n=len(u); a,b=int(.7*n),int(.1*n); m=np.isin(cid,u[a+b:])
Hls,s2,H=d["H_ls"][m],d["sigma2"][m],d["H_true"][m]
scale=np.clip(np.sqrt(np.mean(np.abs(Hls)**2,1,keepdims=True)),1e-6,None).astype(np.float32)
x=np.stack([Hls.real/scale,Hls.imag/scale,np.log(s2+1e-8)],1).astype(np.float32)
y=np.stack([H.real/scale,H.imag/scale],1).astype(np.float32)  # normalised target

model=BayCENet().to(DEV); model.load_state_dict(torch.load(CKPT,map_location=DEV)); model.eval()
MU=np.zeros((len(x),2,H.shape[1]),np.float32); LV=np.zeros((len(x),1,H.shape[1]),np.float32)
with torch.no_grad():
    for i in range(0,len(x),256):
        mu,lv,_=model(torch.from_numpy(x[i:i+256]).to(DEV))
        MU[i:i+256]=mu.cpu().numpy(); LV[i:i+256]=lv.cpu().numpy()

v_pred=np.exp(LV[:,0]).ravel()                       # predicted variance per subcarrier
err2=((MU-y)**2).sum(1).ravel()                      # actual squared error per subcarrier

# ---- reliability (bin by predicted variance) ----
order=np.argsort(v_pred); p=v_pred[order]; e=err2[order]
bins=10; idx=np.array_split(np.arange(len(p)),bins)
mp=np.array([p[i].mean() for i in idx]); me=np.array([e[i].mean() for i in idx])
ece=np.mean(np.abs(me-mp))/np.mean(err2)
corr=np.corrcoef(v_pred,err2)[0,1]
scale_factor=np.mean(err2)/np.mean(v_pred)

print(f"samples x subcarriers: {len(v_pred):,}")
print(f"correlation(pred var, actual err^2): {corr:.3f}   (higher = uncertainty tracks error)")
print(f"ECE-like calibration error:          {ece:.3f}   (lower = better calibrated)")
print(f"recalibration scale factor:          {scale_factor:.3f}   (1.0 = perfectly scaled)")
print("\n bin |  pred var  actual err^2")
for i in range(bins):
    print(f" {i+1:3d} | {mp[i]:9.4f}  {me[i]:9.4f}")

plt.figure(figsize=(6,6))
lim=max(mp.max(),me.max())*1.05
plt.plot([0,lim],[0,lim],'k--',label='ideal (calibrated)')
plt.plot(mp,me,'o-',label='BayCE-Net')
plt.xlabel('Predicted variance'); plt.ylabel('Actual mean squared error')
plt.title(f'Uncertainty calibration (corr={corr:.2f}, ECE={ece:.3f})')
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig('calibration.png',dpi=150)
print("\nsaved -> calibration.png")
