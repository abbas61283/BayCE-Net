"""
calibrate_fig.py  -  publication reliability diagram (Fig. 6) for BayCE-Net
==========================================================================
Produces the calibration figure that validates the paper's core novelty:
the predicted per-subcarrier uncertainty is honest. It bins subcarriers by
predicted error std and plots, per bin, predicted vs. actually-observed
RMS error; a well-calibrated model lies on the diagonal. It also prints and
annotates an ECE-like score.

Run (from  E:\PhDpapers\PAPER 2\code , env = paper2):
    python calibrate_fig.py
Needs: bayce_dataset.npz, bayce_best.pt, bayce_model.py.
Output: fig6_calibration.png  (matches the fig2-fig5 style)
"""
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from bayce_model import BayCENet

rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"})
BLUE = "#0072B2"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---- test split (same seed-0 70/10/20) ----
d = np.load("bayce_dataset.npz", allow_pickle=True)
u = np.unique(d["chan_id"]); rng = np.random.default_rng(0); rng.shuffle(u)
n = len(u); a, b = int(.7*n), int(.1*n)
m = np.isin(d["chan_id"], u[a+b:])
H, Hls, s2 = d["H_true"][m], d["H_ls"][m], d["sigma2"][m]
N = H.shape[1]

scale = np.clip(np.sqrt(np.mean(np.abs(Hls)**2, 1, keepdims=True)), 1e-6, None).astype(np.float32)
x = np.stack([Hls.real/scale, Hls.imag/scale, np.log(s2+1e-8)], 1).astype(np.float32)
y = np.stack([H.real/scale, H.imag/scale], 1).astype(np.float32)  # scaled target (scale is (N,1))

model = BayCENet().to(DEV)
model.load_state_dict(torch.load("bayce_best.pt", map_location=DEV)); model.eval()

mu = np.zeros((len(x), 2, N), np.float32); lv = np.zeros((len(x), N), np.float32)
with torch.no_grad():
    for i in range(0, len(x), 256):
        m_, l_, _ = model(torch.from_numpy(x[i:i+256]).to(DEV))
        mu[i:i+256] = m_.cpu().numpy(); lv[i:i+256] = l_.squeeze(1).cpu().numpy()

pred_std = np.sqrt(np.exp(lv)).ravel()                       # predicted RMS error
act_err  = np.sqrt(((mu - y)**2).sum(1)).ravel()             # actual complex error magnitude

# ---- bin by predicted std, compute per-bin RMS ----
nb = 12
qs = np.quantile(pred_std, np.linspace(0, 1, nb+1))
qs[-1] += 1e-9
bp, ba, bw = [], [], []
for i in range(nb):
    sel = (pred_std >= qs[i]) & (pred_std < qs[i+1])
    if sel.sum() < 10:
        continue
    bp.append(pred_std[sel].mean())
    ba.append(np.sqrt((act_err[sel]**2).mean()))
    bw.append(sel.mean())
bp, ba, bw = np.array(bp), np.array(ba), np.array(bw)
ece = float(np.sum(bw * np.abs(bp - ba)) / (ba.mean() + 1e-12))
print(f"bins={len(bp)}  ECE-like={ece:.4f}  corr={np.corrcoef(pred_std, act_err)[0,1]:.3f}")

# ---- figure ----
fig, ax = plt.subplots(figsize=(5.8, 5.4))
lim = max(bp.max(), ba.max()) * 1.08
ax.plot([0, lim], [0, lim], ls="--", color="#999999", lw=1.6, label="perfect calibration")
ax.plot(bp, ba, color=BLUE, lw=2.2, marker="o", ms=8, label="BayCE-Net")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("predicted error std (per bin)")
ax.set_ylabel("observed RMS error (per bin)")
ax.set_title("Reliability diagram: uncertainty is well calibrated", fontsize=12)
ax.text(0.05*lim, 0.90*lim, f"ECE-like = {ece:.3f}", fontsize=12, fontweight="bold", color=BLUE)
ax.legend(frameon=False, loc="lower right", fontsize=10)
ax.set_aspect("equal")
fig.savefig("fig6_calibration.png")
print("saved fig6_calibration.png")
