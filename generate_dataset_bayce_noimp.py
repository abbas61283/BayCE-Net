"""
generate_dataset.py  -  BayCE-Net data generator (WATERMARK-ready)
==================================================================
Builds a UAC-OFDM dataset from real WATERMARK CIRs (or a statistical
fallback), injects AWGN + Bernoulli-Gaussian impulsive noise, computes
the LS estimate + per-subcarrier uncertainty, and saves an .npz.

KEY: with WATERMARK, chan_id = file_id, so the later train/test split
is BY MEASUREMENT FILE (no time-correlated leakage between splits).
"""

import numpy as np

CFG = dict(
    N=1024, CP=256, L=256, N_pilots=342, mod="QPSK",
    use_watermark=True,
    cir_file="watermark_nof1.npz",
    n_paths=15, max_delay=200, n_realizations_fallback=200,
    snr_list=[0, 5, 10, 15, 20, 25, 30],
    imp_prob=0.05, imp_power_dB=20, add_impulsive=False,
    frames_per_realization=1, max_realizations=8000,
    out_file="bayce_noimp.npz", seed=42,
)
rng = np.random.default_rng(CFG["seed"])


def qpsk(bits):
    b = bits.reshape(-1, 2)
    return ((1 - 2*b[:, 0]) + 1j*(1 - 2*b[:, 1])) / np.sqrt(2)

def modulate(n_syms):
    return qpsk(rng.integers(0, 2, size=n_syms*2))


def statistical_cir(L, n_paths, max_delay):
    h = np.zeros(L, dtype=np.complex128)
    delays = rng.choice(max_delay, size=n_paths, replace=False)
    power = np.exp(-delays / (max_delay/3.0))
    g = rng.standard_normal(n_paths) + 1j*rng.standard_normal(n_paths)
    h[delays] = g * np.sqrt(power/2.0)
    return h / np.linalg.norm(h)

def load_watermark(path, L, cap):
    d = np.load(path)
    cir = np.asarray(d["cir"])
    file_id = np.asarray(d["file_id"]) if "file_id" in d else np.arange(cir.shape[0])
    out = np.zeros((cir.shape[0], L), dtype=np.complex128)
    t = min(L, cir.shape[1]); out[:, :t] = cir[:, :t]
    out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
    if cap and out.shape[0] > cap:
        idx = np.sort(rng.choice(out.shape[0], size=cap, replace=False))
        out, file_id = out[idx], file_id[idx]
    return out, file_id.astype(np.int32)


def add_noise(y, snr_db):
    sig_p = np.mean(np.abs(y)**2)
    n0 = sig_p / (10**(snr_db/10))
    w = np.sqrt(n0/2)*(rng.standard_normal(y.shape)+1j*rng.standard_normal(y.shape))
    out = y + w
    if CFG["add_impulsive"]:
        sI2 = n0 * (10**(CFG["imp_power_dB"]/10))
        mask = rng.random(y.shape) < CFG["imp_prob"]
        imp = np.sqrt(sI2/2)*(rng.standard_normal(y.shape)+1j*rng.standard_normal(y.shape))
        out += mask*imp
    return out, n0


def make_frame(h_cir, snr_db, pilot_idx, X_pilot):
    N, L = CFG["N"], CFG["L"]
    H_true = np.fft.fft(h_cir, n=N)
    X = np.zeros(N, dtype=np.complex128)
    data_idx = np.setdiff1d(np.arange(N), pilot_idx)
    X[data_idx] = modulate(len(data_idx))
    X[pilot_idx] = X_pilot
    Y, n0 = add_noise(H_true*X, snr_db)
    H_ls_p = Y[pilot_idx] / X_pilot
    H_ls = np.interp(np.arange(N), pilot_idx, H_ls_p.real) \
         + 1j*np.interp(np.arange(N), pilot_idx, H_ls_p.imag)
    var_p = n0 / (np.abs(X_pilot)**2)
    sigma2 = np.interp(np.arange(N), pilot_idx, var_p)
    return (H_true.astype(np.complex64), h_cir[:L].astype(np.complex64),
            H_ls.astype(np.complex64), sigma2.astype(np.float32))


def build():
    N = CFG["N"]
    pilot_idx = np.linspace(0, N-1, CFG["N_pilots"], dtype=int)
    X_pilot = np.exp(1j*rng.uniform(0, 2*np.pi, CFG["N_pilots"]))

    if CFG["use_watermark"]:
        cirs, file_id = load_watermark(CFG["cir_file"], CFG["L"], CFG["max_realizations"])
        print(f"Loaded {cirs.shape[0]} WATERMARK CIRs from {len(np.unique(file_id))} files.")
    else:
        n = CFG["n_realizations_fallback"]
        cirs = np.stack([statistical_cir(CFG["L"], CFG["n_paths"], CFG["max_delay"]) for _ in range(n)])
        file_id = np.arange(n, dtype=np.int32)
        print(f"Generated {n} statistical realizations.")

    H_true, h_true, H_ls, sigma2, snr_db, chan_id = [], [], [], [], [], []
    for r in range(cirs.shape[0]):
        for snr in CFG["snr_list"]:
            for _ in range(CFG["frames_per_realization"]):
                Ht, ht, Hl, s2 = make_frame(cirs[r], snr, pilot_idx, X_pilot)
                H_true.append(Ht); h_true.append(ht); H_ls.append(Hl)
                sigma2.append(s2); snr_db.append(snr); chan_id.append(int(file_id[r]))

    ds = dict(H_true=np.array(H_true), h_true=np.array(h_true),
              H_ls=np.array(H_ls), sigma2=np.array(sigma2),
              snr_db=np.array(snr_db, np.float32),
              chan_id=np.array(chan_id, np.int32),
              pilot_idx=pilot_idx.astype(np.int32),
              X_pilot=X_pilot.astype(np.complex64))
    np.savez_compressed(CFG["out_file"], **ds)
    print(f"Saved {len(H_true)} samples -> {CFG['out_file']}")
    return ds


def nmse(est, ref):
    return 10*np.log10(np.mean(np.abs(est-ref)**2)/np.mean(np.abs(ref)**2))

def sanity_check(ds):
    H, Hls = ds["H_true"], ds["H_ls"]
    ls = nmse(Hls, H)
    Rhh = np.mean(np.abs(H)**2); noise = np.mean(np.abs(Hls-H)**2)
    Hlmmse = (Rhh/(Rhh+noise))*Hls
    lm = nmse(Hlmmse, H)
    print("\n--- SANITY CHECK ---")
    print(f"LS    NMSE = {ls:6.2f} dB")
    print(f"LMMSE NMSE = {lm:6.2f} dB")
    g = ls - lm
    print(f"LMMSE gain over LS = {g:.2f} dB", "OK" if g > 1.0 else "CHECK pilots/normalisation")


if __name__ == "__main__":
    sanity_check(build())
