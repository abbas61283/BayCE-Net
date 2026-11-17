"""
generalize.py  -  Cross-environment generalization test for BayCE-Net
=====================================================================
THE CORE PAPER-2 EXPERIMENT.

Train site = NOF1 (what the model saw). Test site = a DIFFERENT WATERMARK
environment (default NCS1). We measure two things:

  1. GENERALIZATION GAP
     NMSE-vs-SNR of the SAME trained model on the unseen site, next to
     the fair DFT-LS baseline and the in-domain (NOF1-test) reference.
     Question: does a data-driven estimator collapse off-distribution,
     or does it degrade gracefully like the model-free baseline?

  2. EPISTEMIC HONESTY  (the golden thread to Paper 3)
     MC-Dropout epistemic variance on NOF1-test vs on the unseen site.
     Claim: a well-built Bayesian estimator KNOWS it is out of its
     training distribution -> epistemic uncertainty RISES on the new
     environment. This is what a plain regressor cannot give you and is
     exactly the signal Paper 3's controller will act on.

It reuses the existing pipeline end to end:
    convert_watermark.py  (load_h, to_realizations)
    generate_dataset.py   (build, same CFG -> same pilots/noise)
    bayce_model.py         (BayCENet, mc_predict  -> matches bayce_best.pt)

Run (from  E:\PhDpapers\PAPER 2\code , env = paper2):
    python generalize.py            # default unseen site = NCS1
    python generalize.py KAU1       # or any other WATERMARK site
    python generalize.py NCS1 force # rebuild the .npz even if present

Needs: bayce_best.pt, bayce_dataset.npz (NOF1) already in the folder,
and the WATERMARK package at convert_watermark.WM_ROOT.
"""

import os, sys, glob
import numpy as np
import torch

import convert_watermark as C          # load_h, to_realizations, WM_ROOT, L, TIME_STRIDE
import generate_dataset as G           # build(), CFG (same pilots/noise as training)
from bayce_model import BayCENet, mc_predict

# ---------------------------------------------------------------- config
SITE   = sys.argv[1] if len(sys.argv) > 1 else "NCS1"
FORCE  = (len(sys.argv) > 2 and sys.argv[2].lower() == "force")
CKPT   = "bayce_best.pt"
NOF1   = "bayce_dataset.npz"                 # in-domain reference (already built)
WM_OUT = f"watermark_{SITE.lower()}.npz"     # converted CIRs for the unseen site
DS_OUT = f"bayce_dataset_{SITE.lower()}.npz" # OFDM test set for the unseen site
MAXR   = 3000                                # CIR realizations for the test site (plenty)
DEV    = "cuda" if torch.cuda.is_available() else "cpu"


def nmse_db(est, ref):
    return 10*np.log10(np.mean(np.abs(est-ref)**2)/np.mean(np.abs(ref)**2)+1e-12)


# ======================================================================
# 1. Convert the unseen WATERMARK site  ->  CIR .npz
# ======================================================================
def convert_site(site, out):
    if os.path.exists(out) and not FORCE:
        print(f"[1] {out} exists -> skip conversion (use 'force' to rebuild)")
        return
    pat = os.path.join(C.WM_ROOT, "input", "channels", site, "mat", f"{site}_*.mat")
    mats = sorted(glob.glob(pat))
    if not mats:
        sys.exit(f"[1] no .mat files for {site} at:\n    {pat}\n"
                 f"    check the site name / WM_ROOT in convert_watermark.py")
    print(f"[1] {site}: {len(mats)} files -> converting")
    cirs, fids = [], []
    for fi, p in enumerate(mats):
        h, _ = C.load_h(p)
        r = C.to_realizations(h, C.L, C.TIME_STRIDE, C.MAX_PER_FILE)
        cirs.append(r); fids.append(np.full(len(r), fi, np.int32))
    cir = np.concatenate(cirs).astype(np.complex64)
    fid = np.concatenate(fids)
    np.savez_compressed(out, cir=cir, file_id=fid,
                        src_id=np.zeros(len(cir), np.int32))
    pdp = np.mean(np.abs(cir)**2, 0); pdp_db = 10*np.log10(pdp/pdp.max()+1e-30)
    sig = int((pdp_db > -20).sum())
    print(f"    saved {out}: {cir.shape}  "
          f"({sig}/{C.L} taps > -20 dB, "
          f"{'sparse' if sig < 0.3*C.L else 'dense'})")


# ======================================================================
# 2. Build the OFDM test set for the unseen site (SAME pilots / noise)
# ======================================================================
def build_site_dataset(cir_file, out):
    if os.path.exists(out) and not FORCE:
        print(f"[2] {out} exists -> skip OFDM build (use 'force' to rebuild)")
        return
    print(f"[2] building OFDM test set -> {out}")
    G.CFG["cir_file"]        = cir_file
    G.CFG["out_file"]        = out
    G.CFG["use_watermark"]   = True
    G.CFG["max_realizations"] = MAXR
    G.build()      # uses the SAME N/CP/L/N_pilots/impulsive settings as training


# ======================================================================
# 3. Load the trained model  (matches bayce_best.pt = v1 architecture)
# ======================================================================
def load_model():
    model = BayCENet().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model" in ckpt \
       and all(not k.startswith(("gate", "backbone", "head")) for k in ckpt):
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[3] loaded {CKPT}  ({n:,} params, device={DEV})")
    return model


# ======================================================================
# 4. Evaluate one dataset  ->  per-SNR NMSE (LS / DFT-LS / LMMSE / BayCE)
#    + arrays needed for the epistemic comparison
# ======================================================================
def evaluate(model, path, in_domain):
    d = np.load(path, allow_pickle=True)
    if in_domain:
        # reproduce CEDataset's 70/10/20 test split (seed 0) -> unseen NOF1 channels
        u = np.unique(d["chan_id"]); rng = np.random.default_rng(0); rng.shuffle(u)
        n = len(u); a, b = int(.7*n), int(.1*n)
        m = np.isin(d["chan_id"], u[a+b:])
        tag = "NOF1 (in-domain test)"
    else:
        m = np.ones(len(d["chan_id"]), bool)          # whole unseen site
        tag = f"{SITE} (cross-domain, unseen)"

    H, Hls = d["H_true"][m], d["H_ls"][m]
    s2, snr = d["sigma2"][m], d["snr_db"][m]
    N = H.shape[1]; L = d["h_true"].shape[1]; pilot_idx = d["pilot_idx"]

    # DFT-LS operators (fair strong baseline)
    Ffull = np.exp(-2j*np.pi*np.outer(np.arange(N), np.arange(L))/N)
    Fp_pinv = np.linalg.pinv(Ffull[pilot_idx, :])

    # BayCE inputs, normalised EXACTLY like CEDataset
    scale = np.clip(np.sqrt(np.mean(np.abs(Hls)**2, 1, keepdims=True)),
                    1e-6, None).astype(np.float32)
    x = np.stack([Hls.real/scale, Hls.imag/scale,
                  np.log(s2+1e-8)], 1).astype(np.float32)

    mu = np.zeros((len(x), 2, N), np.float32)
    with torch.no_grad():
        for i in range(0, len(x), 256):
            mu[i:i+256] = model(torch.from_numpy(x[i:i+256]).to(DEV))[0].cpu().numpy()
    Hbc = (mu[:, 0] + 1j*mu[:, 1]) * scale

    print(f"\n--- {tag} : {m.sum()} samples ---")
    print(f"{'SNR':>4} | {'LS-lin':>7} {'DFT-LS':>7} {'LMMSE':>7} {'BayCE':>7}")
    rows = []
    for s in sorted(np.unique(snr)):
        j = np.where(snr == s)[0]
        ls = nmse_db(Hls[j], H[j])
        Hdft = (Hls[j][:, pilot_idx] @ Fp_pinv.T) @ Ffull.T
        dft = nmse_db(Hdft, H[j])
        Rhh = np.mean(np.abs(H[j])**2); nz = np.mean(np.abs(Hls[j]-H[j])**2)
        lm = nmse_db((Rhh/(Rhh+nz))*Hls[j], H[j])
        bc = nmse_db(Hbc[j], H[j])
        rows.append((s, ls, dft, lm, bc))
        print(f"{s:4.0f} | {ls:7.2f} {dft:7.2f} {lm:7.2f} {bc:7.2f}")
    return np.array(rows), x, scale


# ======================================================================
# 5. Epistemic honesty  -  MC-Dropout variance in vs out of distribution
# ======================================================================
def epistemic_mean(model, x, n=512, T=20):
    idx = np.random.default_rng(0).choice(len(x), size=min(n, len(x)), replace=False)
    vals = []
    for i in range(0, len(idx), 128):
        xb = torch.from_numpy(x[idx[i:i+128]]).to(DEV)
        _, epi = mc_predict(model, xb, T=T)             # (B,1,N)
        vals.append(epi.mean(dim=(1, 2)).cpu().numpy())  # per-sample mean
    return np.concatenate(vals)


# ======================================================================
# main
# ======================================================================
def main():
    print("="*64)
    print(f"CROSS-ENVIRONMENT GENERALIZATION  NOF1 -> {SITE}")
    print("="*64)

    convert_site(SITE, WM_OUT)
    build_site_dataset(WM_OUT, DS_OUT)
    model = load_model()

    if not os.path.exists(NOF1):
        sys.exit(f"missing in-domain reference {NOF1} (the NOF1 training set)")
    rows_in,  x_in,  _ = evaluate(model, NOF1,  in_domain=True)
    rows_out, x_out, _ = evaluate(model, DS_OUT, in_domain=False)

    # ---- generalization gap (BayCE and DFT-LS), per SNR ----
    print("\n" + "="*64)
    print("GENERALIZATION GAP  (positive dB = worse on unseen site)")
    print("="*64)
    print(f"{'SNR':>4} | {'dBayCE':>7} {'dDFT-LS':>8}")
    for r_in, r_out in zip(rows_in, rows_out):
        s = r_in[0]
        print(f"{s:4.0f} | {r_out[4]-r_in[4]:7.2f} {r_out[2]-r_in[2]:8.2f}")

    # ---- epistemic honesty ----
    print("\n" + "="*64)
    print("EPISTEMIC UNCERTAINTY  (MC-Dropout, mean variance per sample)")
    print("="*64)
    epi_in  = epistemic_mean(model, x_in)
    epi_out = epistemic_mean(model, x_out)
    mi, mo = epi_in.mean(), epi_out.mean()
    print(f"  NOF1 in-domain   : {mi:.4e}")
    print(f"  {SITE} unseen    : {mo:.4e}")
    print(f"  ratio (out/in)   : {mo/mi:6.2f}x  "
          f"{'-> RISES on unseen site (good)' if mo > mi else '-> does NOT rise (check)'}")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(rows_in[:, 0],  rows_in[:, 4],  "^-",  lw=2, label="BayCE (in-domain NOF1)")
        ax[0].plot(rows_out[:, 0], rows_out[:, 4], "^--", lw=2, label=f"BayCE (unseen {SITE})")
        ax[0].plot(rows_in[:, 0],  rows_in[:, 2],  "s-",  alpha=.6, label="DFT-LS (in-domain)")
        ax[0].plot(rows_out[:, 0], rows_out[:, 2], "s--", alpha=.6, label=f"DFT-LS (unseen {SITE})")
        ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("NMSE (dB)")
        ax[0].set_title(f"Generalization NOF1 -> {SITE}")
        ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)

        ax[1].bar(["NOF1\n(in-domain)", f"{SITE}\n(unseen)"], [mi, mo],
                  color=["#4C72B0", "#C44E52"])
        ax[1].set_ylabel("mean epistemic variance")
        ax[1].set_title("Model knows it is out of distribution")
        for i, v in enumerate([mi, mo]):
            ax[1].text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(f"generalize_{SITE.lower()}.png", dpi=150)
        print(f"\nsaved figure -> generalize_{SITE.lower()}.png")
    except Exception as e:
        print(f"(figure skipped: {e})")

    print("\nDONE.")


if __name__ == "__main__":
    main()
