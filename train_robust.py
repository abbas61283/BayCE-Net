"""
train_robust.py  -  train & evaluate the impulsive-noise-aware estimator (Q1)
=============================================================================
ISOLATED experiment. Imports only from robust_likelihoods.py (self-contained)
and generate_dataset.py (read-only). Writes ONLY these files:
    robust_<likelihood>.pt        (model checkpoints)
    robust_nof1_test.npz          (cached test CIRs)
Nothing here touches bayce_model.py, bayce_best.pt, base_*.pt, or any prior
result. Delete this file + robust_likelihoods.py + robust_*.pt to fully undo.

Reports on the NOF1 test channels:
  (1) NMSE vs SNR,
  (2) uncertainty CALIBRATION (ECE-like + corr(pred var, actual err^2)),
  (3) ROBUSTNESS: NMSE and uncertainty as impulse rate p grows (SNR 15).

Run (env paper2, terminal):
    python train_robust.py --likelihood gaussian --epochs 60
    python train_robust.py --likelihood studentt --epochs 60
    python train_robust.py --likelihood bgmix    --epochs 60
    python train_robust.py --likelihood studentt --eval-only

Needs (read-only): bayce_dataset.npz, watermark_nof1.npz, generate_dataset.py.
"""

import os, argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader

from robust_likelihoods import CEDataset, RobustBayCE, mc_predict
import generate_dataset as G

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = "bayce_dataset.npz"
WM = "watermark_nof1.npz"
TESTCIR = "robust_nof1_test.npz"     # unique name (does NOT clash with ood_stress)
TMP = "_robust_tmp.npz"


def nmse_db(e, r):
    return 10*np.log10(np.mean(np.abs(e-r)**2)/np.mean(np.abs(r)**2)+1e-12)


# ------------------------------------------------------------------ train
def train(lk, epochs, warmup, lr):
    torch.manual_seed(0)
    model = RobustBayCE(lk).to(DEV)
    tr = DataLoader(CEDataset(DATA, "train"), batch_size=64, shuffle=True)
    va = DataLoader(CEDataset(DATA, "val"), batch_size=256)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    print(f"\n=== train likelihood={lk}  (warmup {warmup}) ===")
    best, best_state = 1e18, None
    for ep in range(1, epochs+1):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in tr:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            mu, lv, _ = model(x)
            loss = mse(mu, y) if ep <= warmup else model.loss(mu, lv, y)
            loss.backward(); opt.step()
            tot += loss.item()*len(x)
        model.eval(); vtot = 0.0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(DEV), y.to(DEV)
                mu, lv, _ = model(x)
                vtot += (mse(mu, y) if ep <= warmup else model.loss(mu, lv, y)).item()*len(x)
        v = vtot/len(va.dataset)
        flag = ""
        if ep > warmup and v < best:
            best, best_state = v, {k: t.cpu().clone() for k, t in model.state_dict().items()}
            flag = " *"
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(f"  ep {ep:3d}/{epochs}  {'MSE' if ep<=warmup else 'NLL'} "
                  f"train {tot/len(tr.dataset):.4e} val {v:.4e}  "
                  f"[{model.params_report()}] ({time.time()-t0:.0f}s){flag}")
    if best_state is None:
        best_state = {k: t.cpu().clone() for k, t in model.state_dict().items()}
    torch.save(best_state, f"robust_{lk}.pt")
    print(f"  saved robust_{lk}.pt  (best val {best:.4e}, {model.params_report()})")


# ------------------------------------------------------------------ helpers
def load(lk):
    m = RobustBayCE(lk).to(DEV)
    m.load_state_dict(torch.load(f"robust_{lk}.pt", map_location=DEV)); m.eval()
    print(f"loaded robust_{lk}.pt  [{m.params_report()}]")
    return m


def forward_all(model, x_np):
    mu = np.zeros((len(x_np), 2, x_np.shape[-1]), np.float32)
    lv = np.zeros((len(x_np), 1, x_np.shape[-1]), np.float32)
    with torch.no_grad():
        for i in range(0, len(x_np), 256):
            m_, l_, _ = model(torch.from_numpy(x_np[i:i+256]).to(DEV))
            mu[i:i+256] = m_.cpu().numpy(); lv[i:i+256] = l_.cpu().numpy()
    return mu, lv


def test_arrays():
    d = np.load(DATA, allow_pickle=True)
    u = np.unique(d["chan_id"]); rng = np.random.default_rng(0); rng.shuffle(u)
    n = len(u); a, b = int(.7*n), int(.1*n)
    m = np.isin(d["chan_id"], u[a+b:])
    return d["H_true"][m], d["H_ls"][m], d["sigma2"][m], d["snr_db"][m]


def scale_input(Hls, s2):
    sc = np.clip(np.sqrt(np.mean(np.abs(Hls)**2, 1, keepdims=True)), 1e-6, None).astype(np.float32)
    x = np.stack([Hls.real/sc, Hls.imag/sc, np.log(s2+1e-8)], 1).astype(np.float32)
    return x, sc


# ------------------------------------------------------------------ (1)+(2)
def eval_nmse_calib(model):
    H, Hls, s2, snr = test_arrays()
    x, sc = scale_input(Hls, s2)
    mu, lv = forward_all(model, x)
    Hhat = (mu[:, 0]+1j*mu[:, 1])*sc
    print(f"\n{'SNR':>4} | {'NMSE':>7}")
    for s in sorted(np.unique(snr)):
        j = snr == s
        print(f"{s:4.0f} | {nmse_db(Hhat[j], H[j]):7.2f}")
    y_sc = np.stack([H.real/sc, H.imag/sc], 1)
    err2 = ((mu - y_sc)**2).sum(1)
    vscale = model.var_scale()                 # true predictive variance factor
    var = np.exp(lv[:, 0]) * vscale
    corr = np.corrcoef(var.ravel(), err2.ravel())[0, 1]
    print(f"  (predictive-variance scale for {model.likelihood} = {vscale:.3f})")
    ps = np.sqrt(var).ravel(); ae = np.sqrt(err2).ravel()
    q = np.quantile(ps, np.linspace(0, 1, 11)); ece = 0.0
    print(f"\n  calibration: corr(pred var, err^2) = {corr:.3f}")
    print(f"  {'bin':>3} {'pred_std':>9} {'rms_err':>9}")
    for i in range(10):
        mk = (ps >= q[i]) & (ps <= q[i+1] if i == 9 else ps < q[i+1])
        if mk.sum() == 0: continue
        pv, av = ps[mk].mean(), ae[mk].mean(); ece += mk.mean()*abs(pv-av)
        print(f"  {i:3d} {pv:9.4f} {av:9.4f}")
    print(f"  ECE-like = {ece:.4f}  (lower = better calibrated)")


# ------------------------------------------------------------------ (3)
def build_test_cirs():
    if os.path.exists(TESTCIR): return TESTCIR
    d = np.load(DATA, allow_pickle=True)
    u = np.unique(d["chan_id"]); rng = np.random.default_rng(0); rng.shuffle(u)
    n = len(u); a, b = int(.7*n), int(.1*n); test_ids = set(u[a+b:].tolist())
    w = np.load(WM, allow_pickle=True); cir, fid = w["cir"], w["file_id"]
    mk = np.isin(fid, list(test_ids))
    np.savez_compressed(TESTCIR, cir=cir[mk].astype(np.complex64),
                        file_id=fid[mk].astype(np.int32),
                        src_id=np.zeros(int(mk.sum()), np.int32))
    return TESTCIR


def eval_robust(model, snr=15, plist=(0.0, 0.05, 0.10, 0.20, 0.30)):
    cf = build_test_cirs()
    print(f"\nROBUSTNESS (SNR={snr}) : impulse sweep")
    print(f"{'p':>5} | {'NMSE':>7} | {'aleat':>8} {'epist':>10}")
    base_a = base_e = None
    vscale = model.var_scale()
    for pv in plist:
        G.CFG.update(cir_file=cf, use_watermark=True, out_file=TMP,
                     snr_list=[snr], add_impulsive=True, imp_prob=pv,
                     max_realizations=1200)
        G.build()
        with np.load(TMP, allow_pickle=True) as d:   # close handle before remove (Windows)
            Hls, s2, Ht = d["H_ls"], d["sigma2"], d["H_true"]
        x, sc = scale_input(Hls, s2)
        mu, lv = forward_all(model, x)
        Hhat = (mu[:, 0]+1j*mu[:, 1])*sc
        nm = nmse_db(Hhat, Ht)
        al = np.sqrt(np.exp(lv[:, 0]).mean() * vscale)     # true predictive std
        idx = np.random.default_rng(0).choice(len(x), size=min(400, len(x)), replace=False)
        ep = []
        for i in range(0, len(idx), 128):
            _, e = mc_predict(model, torch.from_numpy(x[idx[i:i+128]]).to(DEV), T=15)
            ep.append(e.mean(dim=(1, 2)).cpu().numpy())
        model.eval(); ep = np.concatenate(ep).mean()
        if base_a is None: base_a, base_e = al, ep
        print(f"{pv:5.2f} | {nm:7.2f} | {al:8.4f} {ep:10.3e}   (x{al/base_a:.2f}, x{ep/base_e:.2f})")
        try: os.remove(TMP)
        except OSError: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--likelihood", default="studentt", choices=["gaussian", "studentt", "bgmix"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-robust", action="store_true")
    a = ap.parse_args()
    print(f"device={DEV}  likelihood={a.likelihood}  (isolated experiment)")
    if not a.eval_only:
        train(a.likelihood, a.epochs, a.warmup, a.lr)
    model = load(a.likelihood)
    eval_nmse_calib(model)
    if not a.no_robust:
        eval_robust(model)
    print("\nDONE.")


if __name__ == "__main__":
    main()
