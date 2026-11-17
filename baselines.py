"""
baselines.py  -  DL channel-estimation baselines for the Paper-2 comparison
===========================================================================
Four baselines, all trained on the SAME NOF1 dataset, the SAME 3-channel
input  [Re(H_ls), Im(H_ls), log sigma^2]  and the SAME 2-channel output
[Re(H), Im(H)] as BayCE-Net, so the ONLY differences are architecture and
(for BayCE) the uncertainty head. This isolates each design choice.

  1. DetTwin     - deterministic twin of OUR backbone (same conv backbone,
                   single point-estimate head, MSE loss, NO uncertainty).
                   The fairest control: same capacity as BayCE, minus the
                   Bayesian head -> shows the head's NMSE cost is ~0 while
                   it alone provides calibrated uncertainty.

  2. ChannelNet  - Soltani et al., "Deep Learning-Based Channel Estimation",
                   IEEE COMML 2019. SRCNN super-resolution + DnCNN denoiser
                   cascade. The canonical DL-CE baseline. (1D adaptation.)

  3. CRNet       - Convolution-Recurrent Network, Frontiers Mar. Sci. 2025
                   (UWA-OFDM CE). Conv1d(64)-Conv1d(64)-LSTM(64)-LSTM(64)-
                   Dense(128)-out. MSE, Adam, dropout 0.2.
                   NOTE: the paper's LMD/EMD signal-decomposition front-end
                   is OMITTED so the comparison is architecture-only on the
                   identical input; stated as such in the paper.

  4. DeSA_DNN    - Denoising Sparsity-Aware DNN, Liu et al., ICN 2023.
                   Deep-unfolded ISTA sparse recovery in the delay domain
                   (learnable step + soft-threshold per layer) + a DnCNN
                   denoiser. Faithful-representative re-implementation of the
                   published design. Its sparsity prior is MISMATCHED to the
                   dense NOF1 channel -> expected to underperform (a fair,
                   informative result).

All models expose forward(x)->mu with mu shape (B,2,N), so the training and
evaluation code treats them uniformly. Sources:
  ChannelNet : https://ieeexplore.ieee.org/document/8640815
  CRNet      : https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1671853/full
  DeSA-DNN   : https://www.sciopen.com/article/10.23919/ICN.2023.0001
"""

import numpy as np
import torch
import torch.nn as nn

from bayce_model import SparseBackbone     # reuse OUR backbone for the twin


# ======================================================================
# 1. Deterministic twin of BayCE-Net (same backbone, point estimate only)
# ======================================================================
class DetTwin(nn.Module):
    def __init__(self, ch=64, n_blocks=5, p_drop=0.1):
        super().__init__()
        self.backbone = SparseBackbone(ch, n_blocks, p_drop)
        self.head_mu  = nn.Conv1d(ch, 2, 3, padding=1)

    def forward(self, x):
        return self.head_mu(self.backbone(x))          # (B,2,N)


# ======================================================================
# 2. ChannelNet  (SRCNN + DnCNN cascade, 1D)
# ======================================================================
class _SRCNN(nn.Module):
    def __init__(self, cin=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, 64, 9, padding=4), nn.ReLU(),
            nn.Conv1d(64, 32, 1),             nn.ReLU(),
            nn.Conv1d(32, 2, 5, padding=2))
    def forward(self, x): return self.net(x)


class _DnCNN(nn.Module):
    """Residual denoiser: predicts the noise, subtracts it (Zhang 2017)."""
    def __init__(self, ch=64, depth=15):
        super().__init__()
        layers = [nn.Conv1d(2, ch, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Conv1d(ch, ch, 3, padding=1),
                       nn.BatchNorm1d(ch), nn.ReLU()]
        layers += [nn.Conv1d(ch, 2, 3, padding=1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return x - self.net(x)       # residual


class ChannelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.srcnn = _SRCNN(3)
        self.dncnn = _DnCNN()
    def forward(self, x):
        return self.dncnn(self.srcnn(x))               # (B,2,N)


# ======================================================================
# 3. CRNet  (Conv-Conv-LSTM-LSTM-Dense, per-subcarrier output)
# ======================================================================
class CRNet(nn.Module):
    def __init__(self, ch=64, hidden=64, p_drop=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, ch, 3, padding=1), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU())
        self.lstm1 = nn.LSTM(ch, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.drop  = nn.Dropout(p_drop)
        self.dense = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, 2))                          # per subcarrier -> Re/Im
    def forward(self, x):
        h = self.conv(x)                    # (B,ch,N)
        h = h.transpose(1, 2)               # (B,N,ch)  sequence over subcarriers
        h, _ = self.lstm1(h)
        h, _ = self.lstm2(h)
        h = self.drop(h)
        h = self.dense(h)                   # (B,N,2)
        return h.transpose(1, 2)            # (B,2,N)


# ======================================================================
# 4. DeSA-DNN  (deep-unfolded ISTA in delay domain + DnCNN denoiser)
# ======================================================================
class DeSA_DNN(nn.Module):
    """
    Model-driven sparse recovery unfolded for K layers:
        h <- soft_theta( h + mu * A^H (H_ls - A h) )
    with learnable step mu_k and threshold theta_k per layer (sparsity-aware),
    then a DnCNN cleans the reconstructed CFR (denoising).
    A is the fixed (N x L) delay->frequency DFT operator.
    """
    def __init__(self, N=1024, L=256, K=6):
        super().__init__()
        self.N, self.L, self.K = N, L, K
        n = np.arange(N)[:, None]; l = np.arange(L)[None, :]
        # normalise so A^H A ~ I  ->  ISTA step is stable (spectral radius ~1)
        A = (np.exp(-2j*np.pi*n*l/N) / np.sqrt(N)).astype(np.complex64)  # (N,L)
        self.register_buffer("A_r", torch.from_numpy(A.real))
        self.register_buffer("A_i", torch.from_numpy(A.imag))
        self.mu    = nn.Parameter(torch.full((K,), 0.5))     # ~gradient-optimal
        self.theta = nn.Parameter(torch.full((K,), 1e-3))
        self.dncnn = _DnCNN(ch=32, depth=6)

    def _Ah(self, hr, hi):                 # A @ h   (complex), h:(B,L)
        Hr = hr @ self.A_r.T - hi @ self.A_i.T
        Hi = hr @ self.A_i.T + hi @ self.A_r.T
        return Hr, Hi                      # (B,N)

    def _AH(self, Hr, Hi):                 # A^H @ H  (complex), H:(B,N)
        hr =  Hr @ self.A_r + Hi @ self.A_i
        hi = -Hr @ self.A_i + Hi @ self.A_r
        return hr, hi                      # (B,L)

    @staticmethod
    def _soft(hr, hi, th):                 # complex soft-threshold on magnitude
        mag = torch.sqrt(hr*hr + hi*hi + 1e-12)
        scale = torch.clamp(mag - th, min=0.0) / mag
        return hr*scale, hi*scale

    def forward(self, x):
        Hr, Hi = x[:, 0], x[:, 1]                       # (B,N) LS estimate (scaled)
        hr = torch.zeros(x.size(0), self.L, device=x.device)
        hi = torch.zeros_like(hr)
        for k in range(self.K):
            Ar, Ai = self._Ah(hr, hi)                   # current CFR
            rr, ri = Hr - Ar, Hi - Ai                   # residual
            gr, gi = self._AH(rr, ri)                    # gradient step
            hr = hr + self.mu[k]*gr
            hi = hi + self.mu[k]*gi
            hr, hi = self._soft(hr, hi, torch.clamp(self.theta[k], min=0.0))
        Hr, Hi = self._Ah(hr, hi)                        # reconstruct CFR
        cfr = torch.stack([Hr, Hi], 1)                   # (B,2,N)
        return self.dncnn(cfr)                           # denoise -> (B,2,N)


# ======================================================================
# registry + self-test
# ======================================================================
MODELS = {"dettwin": DetTwin, "channelnet": ChannelNet,
          "crnet": CRNet, "desadnn": DeSA_DNN}

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(4, 3, 1024).to(dev)
    for name, cls in MODELS.items():
        m = cls().to(dev)
        y = m(x)
        n = sum(p.numel() for p in m.parameters())
        assert y.shape == (4, 2, 1024), (name, y.shape)
        print(f"{name:11s} OK  out={tuple(y.shape)}  params={n:,}")
    print("\nall baselines forward OK")
