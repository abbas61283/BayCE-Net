"""
ood_stress.py  -  Controlled distribution-shift stress test for BayCE-Net
========================================================================
WHY: the dense-site generalization tests (NOF1->NCS1/KAU1) showed almost
no NMSE gap and a flat epistemic ratio (~1.17x). That is because every
WATERMARK site we have is a DENSE channel -> not a real distribution
shift. To actually prove the paper's core claim ("the estimator knows
when it is out of its element"), we shift along a MEANINGFUL axis the
model was trained on only mildly: IMPULSIVE-NOISE SEVERITY.

The model was trained at impulse probability p = 0.05. Here we sweep
    p in {0.00, 0.05, 0.10, 0.20, 0.30}
on the SAME NOF1 TEST channels (never trained on), at a FIXED SNR, and
measure, as p grows:
    - NMSE of BayCE and of DFT-LS      (does BayCE degrade more gracefully?)
    - mean ALEATORIC std  (predicted)  (does the data-uncertainty head react?)
    - mean EPISTEMIC var  (MC-Dropout) (does model-uncertainty rise?)

The paper figure: uncertainty RISES together with actual error under a
real, interpretable shift -> calibrated confidence, the Paper-3 signal.

Run (from  E:\PhDpapers\PAPER 2\code , env = paper2):
    python ood_stress.py
    python ood_stress.py 10      # fix a different SNR (default 15 dB)

Needs: bayce_best.pt, bayce_dataset.npz, watermark_nof1.npz, and the
project modules (generate_dataset.py, bayce_model.py) in the folder.
"""

import os, sys
import numpy as np
import torch

import generate_dataset as G
from bayce_model import BayCENet, mc_predict

SNR    = int(sys.argv[1]) if len(sys.argv) > 1 else 15
P_LIST = [0.00, 0.05, 0.10, 0.20, 0.30]
CKPT   = "bayce_best.pt"
NOF1   = "bayce_dataset.npz"
WM     = "watermark_nof1.npz"
WM_TE  = "watermark_nof1_test.npz"      # test-only CIRs (built here)
MAXR   = 1500                            # CIR realizations per p (fast, plenty)
DEV    = "cuda" if torch.cuda.is_available() else "cpu"


def nmse_db(est, ref):
    return 10*np.log10(np.mean(np.abs(est-ref)**2)/np.mean(np.abs(ref)**2)+1e-12)


# ---- 1. isolate NOF1 TEST channels (same seed-0 70/10/20 split) ----------
def build_test_cirs():
    d = np.load(NOF1, allow_pickle=True)
    u = np.unique(d["chan_id"]); rng = np.random.default_rng(0); rng.shuffle(u)
    n = len(u); a, b = int(.7*n), int(.1*n); test_ids = set(u[a+b:].tolist())
    w = np.load(WM, allow_pickle=True)
    cir, fid = w["cir"], w["file_id"]
    m = np.isin(fid, list(test_ids))
    np.savez_compressed(WM_TE, cir=cir[m].astype(np.complex64),
                        file_id=fid[m].astype(np.int32),
                        src_id=np.zeros(int(m.sum()), np.int32))
    print(f"test CIRs: {int(m.sum())} from {len(test_ids)} unseen files -> {WM_TE}")


# ---- 2. build an OFDM set at a given impulse prob p, fixed SNR ------------
def build_at_p(p, out):
    G.CFG["cir_file"]         = WM_TE
    G.CFG["use_watermark"]    = True
    G.CFG["out_file"]         = out
    G.CFG["snr_list"]         = [SNR]
    G.CFG["add_impulsive"]    = True
    G.CFG["imp_prob"]         = p
    G.CFG["max_realizations"] = MAXR
    G.build()


# ---- 3. model outputs: point est, aleatoric, epistemic -------------------
def load_model():
    m = BayCENet().to(DEV)
    ck = torch.load(CKPT, map_location=DEV)
    if isinstance(ck, dict) and "state_dict" in ck: ck = ck["state_dict"]
    m.load_state_dict(ck); m.eval()
    return m


def analyse(model, path):
    d = np.load(path, allow_pickle=True)
    H, Hls, s2 = d["H_true"], d["H_ls"], d["sigma2"]
    N = H.shape[1]; L = d["h_true"].shape[1]; pidx = d["pilot_idx"]
    Ffull = np.exp(-2j*np.pi*np.outer(np.arange(N), np.arange(L))/N)
    Fp_pinv = np.linalg.pinv(Ffull[pidx, :])

    scale = np.clip(np.sqrt(np.mean(np.abs(Hls)**2, 1, keepdims=True)),
                    1e-6, None).astype(np.float32)
    x = np.stack([Hls.real/scale, Hls.imag/scale, np.log(s2+1e-8)],
                 1).astype(np.float32)

    mu = np.zeros((len(x), 2, N), np.float32)
    ale = np.zeros((len(x), N), np.float32)          # aleatoric var (scaled domain)
    with torch.no_grad():
        for i in range(0, len(x), 256):
            xb = torch.from_numpy(x[i:i+256]).to(DEV)
            m_, lv_, _ = model(xb)
            mu[i:i+256]  = m_.cpu().numpy()
            ale[i:i+256] = torch.exp(lv_).squeeze(1).cpu().numpy()
    Hbc = (mu[:, 0] + 1j*mu[:, 1]) * scale
    Hdft = (Hls[:, pidx] @ Fp_pinv.T) @ Ffull.T

    # epistemic on a subset
    idx = np.random.default_rng(0).choice(len(x), size=min(512, len(x)), replace=False)
    epi = []
    for i in range(0, len(idx), 128):
        xb = torch.from_numpy(x[idx[i:i+128]]).to(DEV)
        _, e = mc_predict(model, xb, T=20)
        epi.append(e.mean(dim=(1, 2)).cpu().numpy())
    model.eval()
    epi = np.concatenate(epi).mean()

    return dict(bayce=nmse_db(Hbc, H), dft=nmse_db(Hdft, H),
                aleat=np.sqrt(ale.mean()), epist=float(epi))


# ---- main ----------------------------------------------------------------
def main():
    print("="*64)
    print(f"CONTROLLED OOD STRESS  (NOF1 test channels, SNR = {SNR} dB)")
    print("sweep impulse prob p ; model was trained at p = 0.05")
    print("="*64)
    build_test_cirs()
    model = load_model()

    rows = []
    for p in P_LIST:
        out = f"_ood_p{int(p*100):02d}.npz"
        build_at_p(p, out)
        r = analyse(model, out)
        rows.append((p, r["bayce"], r["dft"], r["aleat"], r["epist"]))
        os.remove(out)

    print("\n" + "="*64)
    print("RESULTS  (aleatoric/epistemic normalised to p=0 in the last cols)")
    print("="*64)
    base_a, base_e = rows[0][3], rows[0][4]
    print(f"{'p':>5} | {'BayCE':>7} {'DFT-LS':>7} | {'aleat':>8} {'epist':>10} "
          f"| {'aleat/0':>7} {'epist/0':>7}")
    for p, bc, df, al, ep in rows:
        print(f"{p:5.2f} | {bc:7.2f} {df:7.2f} | {al:8.4f} {ep:10.3e} "
              f"| {al/base_a:7.2f} {ep/base_e:7.2f}")

    # ---- figure: error and uncertainty vs shift ----
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        R = np.array(rows)
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(R[:, 0], R[:, 1], "^-", color="#C44E52", lw=2, label="BayCE NMSE")
        ax1.plot(R[:, 0], R[:, 2], "s--", color="#8172B3", lw=1.5, label="DFT-LS NMSE")
        ax1.set_xlabel("impulse probability p (trained at 0.05)")
        ax1.set_ylabel("NMSE (dB)"); ax1.grid(alpha=.3)
        ax2 = ax1.twinx()
        ax2.plot(R[:, 0], R[:, 3]/base_a, "o-",  color="#4C72B0", lw=2,
                 label="aleatoric (norm.)")
        ax2.plot(R[:, 0], R[:, 4]/base_e, "d-",  color="#55A868", lw=2,
                 label="epistemic (norm.)")
        ax2.set_ylabel("uncertainty, relative to p=0")
        l1, la = ax1.get_legend_handles_labels()
        l2, lb = ax2.get_legend_handles_labels()
        ax1.legend(l1+l2, la+lb, loc="upper left", fontsize=8)
        plt.title(f"Error and uncertainty rise together under shift (SNR={SNR} dB)")
        plt.tight_layout(); plt.savefig("ood_stress.png", dpi=150)
        print("\nsaved figure -> ood_stress.png")
    except Exception as e:
        print(f"(figure skipped: {e})")

    if os.path.exists(WM_TE): os.remove(WM_TE)
    print("\nDONE.")


if __name__ == "__main__":
    main()
