"""
ber_utility.py  -  does the calibrated uncertainty BUY anything? (Q1 utility)
============================================================================
ISOLATED. Imports only robust_likelihoods.py (self-contained) and reads
bayce_dataset.npz / watermark_nof1.npz read-only. Writes only ber_utility.png
and ber_utility.csv. Delete this file to undo.

We build a CODED OFDM link (rate-1/2 K=3 convolutional code, soft Viterbi)
and feed the estimator's per-subcarrier uncertainty into the soft-decision
LLRs. A subcarrier the estimator is UNSURE about (deep fade, corrupted
pilots) should count for LESS in the decoder. We compare four LLR settings:

  agnostic : sigma_eff^2 = N0 / |Hhat|^2            (ignores estimation error)
  aware    : sigma_eff^2 = (N0 + v_pred) / |Hhat|^2 (our calibrated variance)
  genie    : sigma_eff^2 = (N0 + |Hhat-H|^2)/|Hhat|^2  (perfect uncertainty)
  perfectCSI: equalize with the TRUE channel                (lower bound)

If BER(agnostic) > BER(aware) and aware approaches genie, the calibrated
uncertainty carries actionable information -> a system-level benefit no
baseline provides. This is the Q1 headline.

Run (env paper2):
    python ber_utility.py                    # model = robust_gaussian.pt
    python ber_utility.py --model studentt   # or robust_studentt.pt
    python ber_utility.py --impulse 0.1      # impulsive-noise case
"""

import os, argparse
import numpy as np
import torch
from robust_likelihoods import RobustBayCE

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = "bayce_dataset.npz"; WM = "watermark_nof1.npz"


# ======================================================================
# rate-1/2 K=3 convolutional code  (generators 7,5 octal)  + soft Viterbi
# ======================================================================
G = [0o7, 0o5]; K = 3; NS = 1 << (K-1)     # 4 states

def _out_bits(state, u):
    reg = (u << (K-1)) | state             # u,b1,b2
    o = []
    for g in G:
        o.append(bin(reg & g).count("1") & 1)
    return o[0], o[1]

# precompute trellis
NEXT = np.zeros((NS, 2), np.int64); OUT = np.zeros((NS, 2, 2), np.int64)
for s in range(NS):
    for u in (0, 1):
        NEXT[s, u] = ((s << 1) | u) & (NS-1)
        OUT[s, u] = _out_bits(s, u)

def conv_encode(bits):
    s = 0; out = []
    for u in bits:
        o0, o1 = OUT[s, u]; out += [o0, o1]; s = NEXT[s, u]
    return np.array(out, np.int8)

def viterbi(llr):
    """llr: (2*Nsteps,) log P(bit=0)/P(bit=1). Returns decoded info bits.
    Backtrace uses the fact that next_state = ((s<<1)|u), so the input bit
    that entered a state is that state's LSB."""
    n = len(llr)//2
    pm = np.full(NS, -1e18); pm[0] = 0.0
    bp = np.zeros((n, NS), np.int8)
    for t in range(n):
        l0, l1 = llr[2*t], llr[2*t+1]
        npm = np.full(NS, -1e18)
        for s in range(NS):
            if pm[s] <= -1e17: continue
            for u in (0, 1):
                c0, c1 = OUT[s, u]
                bm = (l0 if c0 == 0 else -l0) + (l1 if c1 == 0 else -l1)
                ns = NEXT[s, u]; cand = pm[s] + bm
                if cand > npm[ns]:
                    npm[ns] = cand; bp[t, ns] = s
        pm = npm
    s = int(np.argmax(pm)); bits = np.zeros(n, np.int8)
    for t in range(n-1, -1, -1):
        bits[t] = s & 1          # input bit = LSB of state AFTER step t
        s = bp[t, s]
    return bits


# ======================================================================
# QPSK (Gray): bits (b0,b1) -> ((1-2 b0)+j(1-2 b1))/sqrt2
# ======================================================================
def qpsk_mod(bits):
    b = bits.reshape(-1, 2)
    return ((1-2*b[:, 0]) + 1j*(1-2*b[:, 1]))/np.sqrt(2)


# ======================================================================
# link-level simulation
# ======================================================================
def load_model(tag):
    m = RobustBayCE(tag).to(DEV)
    m.load_state_dict(torch.load(f"robust_{tag}.pt", map_location=DEV)); m.eval()
    return m

def estimate(model, Hls, s2):
    sc = np.clip(np.sqrt(np.mean(np.abs(Hls)**2)), 1e-6, None)
    x = np.stack([Hls.real/sc, Hls.imag/sc, np.log(s2+1e-8)])[None].astype(np.float32)
    with torch.no_grad():
        mu, lv, _ = model(torch.from_numpy(x).to(DEV))
    mu = mu[0].cpu().numpy(); lv = lv[0, 0].cpu().numpy()
    Hhat = (mu[0]+1j*mu[1])*sc
    v = np.exp(lv) * sc*sc * model.var_scale()      # predictive var in TRUE units
    return Hhat, v

def run(model, cirs, pilot_idx, X_pilot, snr_list, imp_p, seed=0):
    rng = np.random.default_rng(seed)
    N = 1024; L = 256
    data_idx = np.setdiff1d(np.arange(N), pilot_idx)
    Nd = len(data_idx)                       # QPSK symbols per frame
    ninfo = Nd - (K-1)                        # zero-terminated
    variants = ["agnostic", "aware", "genie", "perfectCSI"]
    res = {v: {s: [0, 0] for s in snr_list} for v in variants}

    for h in cirs:
        H = np.fft.fft(h, N)
        for snr in snr_list:
            info = rng.integers(0, 2, ninfo).astype(np.int8)
            info_t = np.concatenate([info, np.zeros(K-1, np.int8)])   # tail
            coded = conv_encode(info_t)                               # 2*Nd bits
            Xd = qpsk_mod(coded)                                      # Nd symbols
            X = np.zeros(N, complex); X[data_idx] = Xd; X[pilot_idx] = X_pilot
            sig = H*X; n0 = np.mean(np.abs(sig)**2)/(10**(snr/10))
            w = np.sqrt(n0/2)*(rng.standard_normal(N)+1j*rng.standard_normal(N))
            Y = sig + w
            if imp_p > 0:
                sI2 = n0*(10**(20/10)); mask = rng.random(N) < imp_p
                Y += mask*np.sqrt(sI2/2)*(rng.standard_normal(N)+1j*rng.standard_normal(N))
            # channel estimation
            Hls_p = Y[pilot_idx]/X_pilot
            Hls = np.interp(np.arange(N), pilot_idx, Hls_p.real) \
                + 1j*np.interp(np.arange(N), pilot_idx, Hls_p.imag)
            s2 = np.interp(np.arange(N), pilot_idx, n0/np.abs(X_pilot)**2)
            Hhat, v = estimate(model, Hls, s2)

            Hd, vd, Yd, Htrue_d = Hhat[data_idx], v[data_idx], Y[data_idx], H[data_idx]
            for var in variants:
                if var == "perfectCSI":
                    u = Yd/Htrue_d; seff = n0/np.abs(Htrue_d)**2
                else:
                    u = Yd/Hd
                    if var == "agnostic":  seff = n0/np.abs(Hd)**2
                    elif var == "aware":   seff = (n0 + vd)/np.abs(Hd)**2
                    else:                  seff = (n0 + np.abs(Hd-Htrue_d)**2)/np.abs(Hd)**2
                seff = np.maximum(seff, 1e-9)
                llr = np.empty(2*Nd)
                llr[0::2] = 2*np.sqrt(2)*u.real/seff      # b0 from Re
                llr[1::2] = 2*np.sqrt(2)*u.imag/seff      # b1 from Im
                dec = viterbi(llr)[:ninfo]
                res[var][snr][0] += int((dec != info).sum())
                res[var][snr][1] += ninfo
    return res, variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gaussian",
                    choices=["gaussian", "studentt", "bgmix"])
    ap.add_argument("--impulse", type=float, default=0.0)
    ap.add_argument("--snr", type=int, nargs="+", default=[0, 3, 6, 9, 12, 15])
    ap.add_argument("--nchan", type=int, default=60, help="test CIRs to average")
    a = ap.parse_args()

    d = np.load(DATA, allow_pickle=True)
    pilot_idx = d["pilot_idx"]; X_pilot = d["X_pilot"]
    # unseen test channels (same seed-0 split) -> take their CIRs from WATERMARK
    u = np.unique(d["chan_id"]); rng = np.random.default_rng(0); rng.shuffle(u)
    n = len(u); aa, bb = int(.7*n), int(.1*n); test_ids = set(u[aa+bb:].tolist())
    w = np.load(WM, allow_pickle=True); cir, fid = w["cir"], w["file_id"]
    cirs = cir[np.isin(fid, list(test_ids))]
    idx = np.random.default_rng(1).choice(len(cirs), size=min(a.nchan, len(cirs)), replace=False)
    cirs = cirs[idx]
    print(f"model=robust_{a.model}.pt  impulse p={a.impulse}  channels={len(cirs)}")

    model = load_model(a.model)
    res, variants = run(model, cirs, pilot_idx, X_pilot, a.snr, a.impulse)

    print(f"\n{'SNR':>4} | " + " ".join(f"{v:>10}" for v in variants))
    table = []
    for s in a.snr:
        row = [max(res[v][s][0]/max(res[v][s][1], 1), 1e-6) for v in variants]
        table.append([s]+row)
        print(f"{s:4d} | " + " ".join(f"{r:10.2e}" for r in row))

    with open("ber_utility.csv", "w") as f:
        f.write("SNR," + ",".join(variants) + "\n")
        for r in table: f.write(",".join(f"{x:.3e}" for x in r) + "\n")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        T = np.array(table); plt.figure(figsize=(8, 5))
        mk = {"agnostic": "o--", "aware": "^-", "genie": "s:", "perfectCSI": "x-"}
        for i, v in enumerate(variants):
            plt.semilogy(T[:, 0], T[:, i+1], mk[v], lw=2 if v == "aware" else 1.3, label=v)
        plt.xlabel("SNR (dB)"); plt.ylabel("coded BER")
        plt.title(f"Uncertainty-aware decoding (robust_{a.model}, impulse p={a.impulse})")
        plt.grid(alpha=.3, which="both"); plt.legend()
        plt.tight_layout(); plt.savefig("ber_utility.png", dpi=150)
        print("\nsaved ber_utility.png / ber_utility.csv")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
