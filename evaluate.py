"""
evaluate.py  —  BayCE-Net per-SNR evaluation (the real Paper-2 figure)
=====================================================================
Produces NMSE-vs-SNR curves for LS, LMMSE, and the trained BayCE-Net
on the TEST split (channels the model never saw). This is the honest
picture that a single averaged NMSE hides.

Run:
    conda activate paper2
    python evaluate.py
Needs: bayce_dataset.npz, bayce_best.pt, bayce_model.py in the folder.
"""
import numpy as np, torch, matplotlib.pyplot as plt
from bayce_model import BayCENet          # v1 architecture (matches checkpoint)

DATA = "bayce_dataset.npz"
CKPT = "bayce_best.pt"
DEV  = "cuda" if torch.cuda.is_available() else "cpu"

def nmse_db(est, ref):
    return 10*np.log10(np.mean(np.abs(est-ref)**2)/np.mean(np.abs(ref)**2)+1e-12)

# ---- reproduce the SAME test split CEDataset uses (seed 0, 70/10/20) ----
d = np.load(DATA, allow_pickle=True)
cid = d["chan_id"]; uniq = np.unique(cid)
rng = np.random.default_rng(0); rng.shuffle(uniq)
n = len(uniq); ntr, nva = int(.7*n), int(.1*n)
test_ch = uniq[ntr+nva:]
m = np.isin(cid, test_ch)
print(f"test: {m.sum()} samples from {len(test_ch)} channels")

H   = d["H_true"][m]; Hls = d["H_ls"][m]
s2  = d["sigma2"][m]; snr = d["snr_db"][m]

# ---- model inputs, normalised exactly like CEDataset ----
scale = np.clip(np.sqrt(np.mean(np.abs(Hls)**2,1,keepdims=True)),1e-6,None).astype(np.float32)
x = np.stack([Hls.real/scale, Hls.imag/scale, np.log(s2+1e-8)],1).astype(np.float32)

model = BayCENet().to(DEV)
model.load_state_dict(torch.load(CKPT, map_location=DEV)); model.eval()

mu_all = np.zeros((len(x),2,H.shape[1]), np.float32)
with torch.no_grad():
    for i in range(0, len(x), 256):
        xb = torch.from_numpy(x[i:i+256]).to(DEV)
        mu,_,_ = model(xb)
        mu_all[i:i+256] = mu.cpu().numpy()
H_model = (mu_all[:,0]+1j*mu_all[:,1]) * scale     # undo normalisation

# ---- per-SNR curves ----
snrs = sorted(np.unique(snr))
rows = []
print(f"\n{'SNR':>4} | {'LS':>7} {'LMMSE':>7} {'BayCE':>7}   (NMSE dB)")
print("-"*40)
for s in snrs:
    j = snr == s
    ls  = nmse_db(Hls[j], H[j])
    # crude per-SNR LMMSE (Wiener)
    Rhh = np.mean(np.abs(H[j])**2); noise = np.mean(np.abs(Hls[j]-H[j])**2)
    Hlm = (Rhh/(Rhh+noise))*Hls[j]
    lm  = nmse_db(Hlm, H[j])
    bc  = nmse_db(H_model[j], H[j])
    rows.append((s, ls, lm, bc))
    print(f"{s:4.0f} | {ls:7.2f} {lm:7.2f} {bc:7.2f}")

rows = np.array(rows)
plt.figure(figsize=(8,5))
plt.plot(rows[:,0], rows[:,1], 'o--', label='LS')
plt.plot(rows[:,0], rows[:,2], 's--', label='LMMSE')
plt.plot(rows[:,0], rows[:,3], '^-',  label='BayCE-Net', lw=2)
plt.xlabel('SNR (dB)'); plt.ylabel('NMSE (dB)')
plt.title('Channel estimation NMSE vs SNR (WATERMARK NOF1, test channels)')
plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
plt.savefig('nmse_vs_snr.png', dpi=150)
print("\nsaved figure -> nmse_vs_snr.png")
print(f"\nBayCE-Net at highest SNR ({snrs[-1]:.0f} dB): {rows[-1,3]:.2f} dB")
print(f"gain over LMMSE at {snrs[-1]:.0f} dB: {rows[-1,2]-rows[-1,3]:.2f} dB")
