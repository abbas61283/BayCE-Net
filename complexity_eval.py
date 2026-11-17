"""
complexity_eval.py  -  model complexity & inference-time table (Reviewer point #5)
==================================================================================
Reports, for BayCE-Net and every baseline:
  * parameter count
  * single-pass inference time (ms per frame)  on CPU and (if available) GPU
  * BayCE-Net ALSO: full predictive time = T=20 MC-dropout passes

Produces a table like the ones in Zhang 2022 (Applied Acoustics) Table 5 and
Wang 2024 (JMSE) Table 4, which the journal expects.

Run (same env as training, from your code folder):
    python complexity_eval.py
Needs: bayce_model.py (BayCENet), baselines.py (MODELS registry).
Output: complexity.csv  + printed table.
"""
import time, numpy as np, torch
from bayce_model import BayCENet
from baselines import MODELS            # {"dettwin":..., "channelnet":..., "crnet":..., "desadnn":...}

N        = 1024          # subcarriers (model input length)
T_MC     = 20            # MC-dropout passes used at inference
WARMUP   = 10           # timing warm-up iterations
ITERS    = 100          # timing iterations (averaged)
BATCH    = 1            # per-frame latency (set >1 for throughput)

def count_params(m):
    return sum(p.numel() for p in m.parameters())

@torch.no_grad()
def time_forward(m, x, iters=ITERS, warmup=WARMUP, mc_passes=1, train_mode=False):
    """Average wall-clock ms for `mc_passes` forward passes of model m on input x."""
    if train_mode:
        m.train()          # keep dropout active (epistemic / MC)
    else:
        m.eval()
    dev = x.device
    for _ in range(warmup):
        for _ in range(mc_passes):
            _ = m(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for _ in range(mc_passes):
            _ = m(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3     # ms per frame

def build(name):
    if name == "bayce":
        return BayCENet()
    return MODELS[name]()

def run_on(device):
    dev = torch.device(device)
    x = torch.randn(BATCH, 3, N, device=dev)
    rows = []
    # ---- BayCE-Net: single pass + full T=20 MC predictive ----
    m = build("bayce").to(dev)
    p = count_params(m)
    t1  = time_forward(m, x, mc_passes=1,   train_mode=False)   # point estimate (dropout off)
    tmc = time_forward(m, x, mc_passes=T_MC, train_mode=True)   # T MC passes (dropout on)
    rows.append(("BayCE-Net", p, t1, tmc))
    # ---- baselines: single pass only (no uncertainty) ----
    for name in MODELS:
        m = build(name).to(dev)
        p = count_params(m)
        t1 = time_forward(m, x, mc_passes=1, train_mode=False)
        rows.append((name, p, t1, float("nan")))
    return rows

def main():
    import csv
    out = []
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for d in devices:
        print(f"\n==== device = {d} ====")
        print(f"{'model':12s} {'params':>10s} {'1-pass(ms)':>12s} {'T=20 MC(ms)':>12s}")
        for name, p, t1, tmc in run_on(d):
            mc = f"{tmc:10.2f}" if not np.isnan(tmc) else "       -  "
            print(f"{name:12s} {p:10,d} {t1:12.3f} {mc:>12s}")
            out.append({"device": d, "model": name, "params": p,
                        "ms_single": round(t1, 4),
                        "ms_T20_MC": (round(tmc, 4) if not np.isnan(tmc) else "")})
    with open("complexity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["device","model","params","ms_single","ms_T20_MC"])
        w.writeheader(); w.writerows(out)
    print("\nsaved complexity.csv")
    print("NOTE: report the CPU row in the paper (matches Wang 2024 / Zhang 2022 practice);")
    print("      state your CPU model, as they do (e.g. 'Intel Core i7 @ 2.5 GHz').")

if __name__ == "__main__":
    main()
