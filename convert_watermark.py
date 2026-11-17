"""
convert_watermark.py  —  WATERMARK .mat  ->  CIR array for BayCE-Net
===================================================================
Reads WATERMARK channel files (each holds h = (258 delays, 2048 time
snapshots)), time-subsamples the snapshots into independent CIR
realizations, trims/pads to L taps, normalises each, and saves an .npz.

Output keys:
    cir      (n_realizations, L)  complex64   -> the channels
    src_id   (n_realizations,)    int32       -> which site (0=NOF1,1=NCS1,..)
    file_id  (n_realizations,)    int32       -> which .mat file
    fs_tau, fs_t, fc, bandwidth              -> metadata (from first file)

Usage:
    edit WM_ROOT and CHANNELS below, then:
    python convert_watermark.py
"""

import os, glob
import numpy as np
from scipy.io import loadmat

# ---------------------------------------------------------------- config
WM_ROOT = r"E:\PhDpapers\PAPER 2\WatermarkV1\Watermark"
CHANNELS = ["NOF1"]          # add "NCS1","KAU1" later for generalization tests
L = 256                      # model taps (258 -> trim to 256)
TIME_STRIDE = 8              # take every Nth snapshot (decorrelate in time)
MAX_PER_FILE = None          # or an int cap per file
OUT = "watermark_nof1.npz"


def load_h(path):
    """Load the complex TVIR 'h' from one WATERMARK .mat file."""
    try:
        d = loadmat(path)
        h = d["h"]
        meta = dict(fs_tau=int(d["fs_tau"].ravel()[0]),
                    fs_t=float(d["fs_t"].ravel()[0]),
                    fc=int(d["fc"].ravel()[0]))
        return np.asarray(h), meta
    except NotImplementedError:
        import h5py
        with h5py.File(path, "r") as f:
            h = np.array(f["h"])
            # h5py may store complex as compound or split; handle compound
            if h.dtype.names and "real" in h.dtype.names:
                h = h["real"] + 1j * h["imag"]
            meta = dict(fs_tau=int(np.array(f["fs_tau"]).ravel()[0]),
                        fs_t=float(np.array(f["fs_t"]).ravel()[0]),
                        fc=int(np.array(f["fc"]).ravel()[0]))
        return h, meta


def to_realizations(h, L, stride, cap):
    """h: (delays, time) -> (n, L) time-subsampled, trimmed, normalised."""
    n_delays, n_time = h.shape
    cols = list(range(0, n_time, stride))
    if cap:
        cols = cols[:cap]
    out = np.zeros((len(cols), L), dtype=np.complex64)
    t = min(L, n_delays)
    for i, c in enumerate(cols):
        v = h[:t, c]
        out[i, :t] = v
    # per-realization energy normalisation
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    out /= (norm + 1e-12)
    return out


def main():
    all_cir, all_src, all_file = [], [], []
    meta0 = None

    for src_idx, ch in enumerate(CHANNELS):
        mats = sorted(glob.glob(os.path.join(
            WM_ROOT, "input", "channels", ch, "mat", f"{ch}_*.mat")))
        if not mats:
            print(f"  !! no files for {ch} — check path")
            continue
        print(f"{ch}: {len(mats)} files")

        for fi, path in enumerate(mats):
            h, meta = load_h(path)
            if meta0 is None:
                meta0 = meta
            reals = to_realizations(h, L, TIME_STRIDE, MAX_PER_FILE)
            all_cir.append(reals)
            all_src.append(np.full(len(reals), src_idx, np.int32))
            all_file.append(np.full(len(reals), fi, np.int32))

    cir = np.concatenate(all_cir).astype(np.complex64)
    src = np.concatenate(all_src)
    fid = np.concatenate(all_file)

    print(f"\nTotal realizations: {len(cir)}")
    print(f"CIR shape: {cir.shape}  dtype: {cir.dtype}")
    print(f"metadata: {meta0}")

    np.savez_compressed(OUT, cir=cir, src_id=src, file_id=fid,
                        fs_tau=meta0["fs_tau"], fs_t=meta0["fs_t"],
                        fc=meta0["fc"])
    print(f"saved -> {OUT}  ({os.path.getsize(OUT)/1e6:.0f} MB)")

    # quick sanity: average power-delay profile
    pdp = np.mean(np.abs(cir) ** 2, axis=0)
    pdp_db = 10 * np.log10(pdp / pdp.max() + 1e-30)
    sig = int((pdp_db > -20).sum())
    print(f"avg PDP: {sig}/{L} taps > -20 dB "
          f"({'sparse' if sig < 0.3 * L else 'dense'})")


if __name__ == "__main__":
    main()
