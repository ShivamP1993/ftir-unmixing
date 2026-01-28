# Generated from: unmix_cnn_wsad.ipynb
# Converted at: 2026-01-28T14:02:03.186Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import os
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional
import kaleido
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import NMF
from scipy.optimize import nnls
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch.nn.functional as F

# read_dpt_view.py
# pip install numpy plotly

import numpy as np
from pathlib import Path
import plotly.graph_objects as go

# ---- file paths (edit if yours differ) ----
REF1 = "/home/shpande/Downloads/painting_unmix/francisco-ftir/CaOx_refNaCl.dpt"
REF2 = "/home/shpande/Downloads/painting_unmix/francisco-ftir/ref_colleesturgeon_MCTATR.dpt"
REF3 = "/home/shpande/Downloads/painting_unmix/francisco-ftir/ref_PbPalmitate_MCTATR.dpt"
SAMPLE = "/home/shpande/Downloads/painting_unmix/francisco-ftir/17_38_P213.079_C092.002_08_140717_01_ATRFPA_NGL14.0.dpt"

# ---- readers ----
def ref_parser(filename):
    """Two columns: wavenumber, value."""
    arr = np.loadtxt(filename, delimiter=",", ndmin=2)
    x, y = arr[:, 0], arr[:, 1]
    print(f"{Path(filename).name}: {arr.shape[0]} wavelengths")
    return x, y

def sample_parser(filename):
    """First col: wavenumber; remaining cols: pixels (each column is one pixel)."""
    arr = np.loadtxt(filename, delimiter=",", ndmin=2)
    wavelengths = arr[:, 0]
    sample = arr[:, 1:]           # shape (L, Npix)
    print(f"sample: {len(wavelengths)} wavelengths, {sample.shape[1]} pixels")
    return wavelengths, sample

# ---- plotting helpers ----
def plot_refs(refs, title="Reference spectra"):
    fig = go.Figure()
    for name, (x, y) in refs.items():
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Wavenumber (cm⁻¹)", autorange="reversed"),
        yaxis=dict(title="Value"),
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2)
    )
    return fig

def plot_sample_pixels(wavenumbers, sample, pixels=(0, 1, 2, 100, 500)):
    """Plot a few pixel spectra from the sample (by column index)."""
    fig = go.Figure()
    n_pix = sample.shape[1]
    for p in pixels:
        if 0 <= p < n_pix:
            fig.add_trace(go.Scatter(x=wavenumbers, y=sample[:, p], mode="lines", name=f"pixel {p}"))
    fig.update_layout(
        title=f"Sample spectra (pixels: {', '.join(str(p) for p in pixels if 0 <= p < n_pix)})",
        xaxis=dict(title="Wavenumber (cm⁻¹)", autorange="reversed"),
        yaxis=dict(title="Value"),
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2)
    )
    return fig

def main():
    # read files
    wl1, ref1 = ref_parser(REF1)
    wl2, ref2 = ref_parser(REF2)
    wl3, ref3 = ref_parser(REF3)
    wavelengths, sample = sample_parser(SAMPLE)

    # show references
    fig_refs = plot_refs({
        "CaOx_refNaCl": (wl1, ref1),
        "SturgeonGlue": (wl2, ref2),
        "PbPalmitate":  (wl3, ref3),
    })
    fig_refs.show()  # opens in a browser tab

    # show a few sample pixels (edit indices as you like)
    fig_sample = plot_sample_pixels(wavelengths, sample, pixels=(0, 1, 2, 142, 1000))
    fig_sample.show()

for p in (REF1, REF2, REF3, SAMPLE):
    if not Path(p).exists():
        print(f"Warning: file not found: {p}")
main()


wn1, ref1 = ref_parser(REF1)
wn2, ref2 = ref_parser(REF2)
wn3, ref3 = ref_parser(REF3)
wavenumbers, sample = sample_parser(SAMPLE)

def resample_spectrum_to_target(
    wn_src: np.ndarray,
    y_src: np.ndarray,
    wn_target: np.ndarray,
    *,
    kind: str = "linear",
    fill: str = "nan",
) -> np.ndarray:
    """
    Resample a spectrum y_src(wn_src) onto wn_target.

    Parameters
    ----------
    wn_src : (Ns,) array
        Source wavenumbers (descending or ascending).
    y_src : (Ns,) array
        Source spectral intensities corresponding to wn_src.
    wn_target : (Nt,) array
        Target wavenumbers (descending or ascending).
    kind : str
        Interpolation kind: "linear" (recommended), "nearest" etc.
        (Only "linear" and "nearest" are implemented here for robustness.)
    fill : str
        How to handle target points outside source range:
        - "nan": fill with np.nan
        - "edge": clamp to edge values (constant extension)

    Returns
    -------
    y_tgt : (Nt,) array
        Resampled spectrum on wn_target grid.
    """
    wn_src = np.asarray(wn_src).astype(float)
    y_src = np.asarray(y_src).astype(float)
    wn_target = np.asarray(wn_target).astype(float)

    if wn_src.ndim != 1 or y_src.ndim != 1 or wn_target.ndim != 1:
        raise ValueError("wn_src, y_src, wn_target must be 1D arrays.")
    if wn_src.shape[0] != y_src.shape[0]:
        raise ValueError("wn_src and y_src must have the same length.")

    # Ensure strictly increasing order for interpolation
    if wn_src[0] > wn_src[-1]:
        wn_src_i = wn_src[::-1]
        y_src_i = y_src[::-1]
    else:
        wn_src_i = wn_src
        y_src_i = y_src

    # Sort target for interpolation, then unsort back
    tgt_desc = wn_target[0] > wn_target[-1]
    wn_tgt_i = wn_target[::-1] if tgt_desc else wn_target

    # Remove any duplicate wavenumbers in source (interp requires monotonic)
    # If duplicates exist, average them.
    uniq_wn, inv = np.unique(wn_src_i, return_inverse=True)
    if uniq_wn.size != wn_src_i.size:
        y_acc = np.zeros_like(uniq_wn, dtype=float)
        counts = np.zeros_like(uniq_wn, dtype=float)
        for j, k in enumerate(inv):
            y_acc[k] += y_src_i[j]
            counts[k] += 1.0
        y_src_i = y_acc / np.maximum(counts, 1.0)
        wn_src_i = uniq_wn

    # Interpolate (linear or nearest) without scipy dependency
    if kind not in ("linear", "nearest"):
        raise ValueError("Only kind='linear' or kind='nearest' is supported in this function.")

    # Determine fill behavior
    src_min, src_max = wn_src_i[0], wn_src_i[-1]
    if fill == "nan":
        left = np.nan
        right = np.nan
    elif fill == "edge":
        left = y_src_i[0]
        right = y_src_i[-1]
    else:
        raise ValueError("fill must be 'nan' or 'edge'.")

    if kind == "linear":
        y_tgt_i = np.interp(wn_tgt_i, wn_src_i, y_src_i, left=left, right=right)
    else:  # nearest
        # nearest-neighbor via searchsorted
        idx = np.searchsorted(wn_src_i, wn_tgt_i, side="left")
        idx = np.clip(idx, 0, wn_src_i.size - 1)
        # fix cases where previous is closer
        prev = np.clip(idx - 1, 0, wn_src_i.size - 1)
        choose_prev = np.abs(wn_tgt_i - wn_src_i[prev]) < np.abs(wn_tgt_i - wn_src_i[idx])
        idx[choose_prev] = prev[choose_prev]
        y_tgt_i = y_src_i[idx]

        # apply fill outside range
        outside = (wn_tgt_i < src_min) | (wn_tgt_i > src_max)
        if fill == "nan":
            y_tgt_i[outside] = np.nan
        elif fill == "edge":
            y_tgt_i[wn_tgt_i < src_min] = y_src_i[0]
            y_tgt_i[wn_tgt_i > src_max] = y_src_i[-1]

    # Restore original target ordering
    y_tgt = y_tgt_i[::-1] if tgt_desc else y_tgt_i
    return y_tgt


def resample_all_refs_to_sample_grid(
    wn_refs: list,
    y_refs: list,
    wn_sample: np.ndarray,
    *,
    kind: str = "linear",
    fill: str = "nan",
    drop_nans: bool = True,
):
    """
    Resample multiple reference spectra (each with its own wn grid) onto wn_sample.

    Returns:
    - wn_common: wavenumbers after optional NaN dropping
    - Y: (K, B) stacked resampled refs aligned to wn_common
    """
    if len(wn_refs) != len(y_refs):
        raise ValueError("wn_refs and y_refs must have the same length.")

    Y = []
    for wn_r, y_r in zip(wn_refs, y_refs):
        y_on_sample = resample_spectrum_to_target(wn_r, y_r, wn_sample, kind=kind, fill=fill)
        Y.append(y_on_sample)
    Y = np.vstack(Y)  # (K, B)

    wn_common = np.asarray(wn_sample, dtype=float)

    if drop_nans:
        good = np.all(np.isfinite(Y), axis=0)
        wn_common = wn_common[good]
        Y = Y[:, good]

    return wn_common, Y


wn_common, refs_on_sample = resample_all_refs_to_sample_grid(
    wn_refs=[wn1, wn2, wn3],
    y_refs=[ref1, ref2, ref3],
    wn_sample=wavenumbers,   # your 1504-band grid
    kind="linear",
    fill="nan",              # avoids extrapolation
    drop_nans=True           # crops to overlap shared by all 3 refs
)

# refs_on_sample shape: (3, len(wn_common)) and matches your cube after applying the same band selection


'''sample_sub = np.concatenate([sample[:720, :], sample[821:, :]], axis=0)
wavenumbers_sub = np.concatenate([wavenumbers[:720], wavenumbers[821:]], axis=0)'''

refs_on_sample_sub = np.concatenate([refs_on_sample[:,:720], refs_on_sample[:,821:]], axis=1)

sample_sub_cube = np.reshape(sample, [1504, 320, 64])
sample_sub_cube = np.transpose(sample_sub_cube, (1,2,0))
cube_norm = np.zeros(np.shape(sample_sub_cube))

for i in range(np.shape(sample_sub_cube)[2]):

    cube_norm[:,:,i] = sample_sub_cube[:,:,i] - np.min(sample_sub_cube[:,:,i])
    cube_norm[:,:,i] = cube_norm[:,:,i]/np.max(cube_norm[:,:,i])

class SumToOne(nn.Module):
    """Abundance Sum-to-One via scaled softmax."""
    def __init__(self, alpha=3.5):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):  # x: (B, R, H, W)
        return F.softmax(self.alpha * x, dim=1)

def spectral_angle_distance(x, x_hat):
    """
    SAD loss over all pixels.
    x, x_hat: (B, bands, H, W)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, -1)
    x_hat = x_hat.view(B, C, -1)
    x = F.normalize(x, p=2, dim=1)
    x_hat = F.normalize(x_hat, p=2, dim=1)
    cos = (x * x_hat).sum(dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    sad = torch.acos(cos)
    return sad.mean()

def SADLoss(x, x_hat):
    dot = (x * x_hat).sum(dim=1)
    norm = torch.norm(x, dim=1) * torch.norm(x_hat, dim=1) + 1e-7
    cos = torch.clamp(dot / norm, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.mean(torch.acos(cos))

class PositiveDecoder(nn.Module):
    def __init__(self, num_bands, num_endmembers, patch_size):
        super().__init__()
        self.raw_weight = nn.Parameter(
            torch.empty(num_bands, num_endmembers, patch_size, patch_size)
        )
        nn.init.xavier_uniform_(self.raw_weight)   # Glorot init
        self.patch_size = patch_size

    def forward(self, a):
        # enforce nonnegativity
        weight = F.softplus(self.raw_weight)
        return F.conv2d(a, weight, bias=None, padding=self.patch_size // 2)

    def endmember_matrix(self):
        weight = F.softplus(self.raw_weight)
        return weight.sum(dim=(2,3))   # (bands, R)

def compute_band_weights(
    cube_hwb: np.ndarray,
    wavenumbers: np.ndarray | None = None,
    bad_windows: list[tuple[float, float]] | None = None,
    *,
    w_min: float = 0.05,
    z_thresh: float = 3.0,
    sharpness: float = 2.0,
    gamma_rough: float = 0.75,
    eps: float = 1e-8,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Estimate band reliability weights w in [w_min, 1].

    Inputs:
      cube_hwb: (H, W, B) float array (your cube_norm).
      wavenumbers: (B,) optional. If provided with bad_windows, those ranges get weight=0.
      bad_windows: list of (lo, hi) wavenumber ranges to force weight=0 (e.g. CO2).
    Heuristic signals (per band):
      - neighbor correlation deficit across pixels (bands that don't agree with neighbors)
      - roughness of the median spectrum (bands that are "spiky" in the median)
    Output:
      w: torch tensor shaped (1, B, 1, 1) on `device`.
    """
    assert cube_hwb.ndim == 3, f"Expected (H,W,B), got {cube_hwb.shape}"
    H, W, B = cube_hwb.shape
    X = cube_hwb.reshape(-1, B).astype(np.float32)  # (N, B)
    N = X.shape[0]

    # --- robust standardize each band across pixels ---
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) + eps
    Z = (X - med) / mad

    # normalize to ~unit variance for correlation computation
    Z = Z - Z.mean(axis=0, keepdims=True)
    std = Z.std(axis=0, keepdims=True) + eps
    Z = Z / std

    # --- neighbor correlation (adjacent bands) ---
    # corr_adj[b] = corr(Z[:,b], Z[:,b+1]) for b=0..B-2
    corr_adj = np.mean(Z[:, :-1] * Z[:, 1:], axis=0)  # (B-1,)
    corr_adj = np.clip(corr_adj, -1.0, 1.0)

    # For each band, take min correlation with neighbors as "reliability"
    neigh_corr = np.empty(B, dtype=np.float32)
    neigh_corr[0] = corr_adj[0]
    neigh_corr[-1] = corr_adj[-1]
    if B > 2:
        neigh_corr[1:-1] = np.minimum(corr_adj[:-1], corr_adj[1:])

    corr_def = (1.0 - neigh_corr).astype(np.float32)  # higher = worse

    # --- roughness of median spectrum (systematic spikiness) ---
    med_spec = np.median(X, axis=0).astype(np.float32)
    d2 = np.zeros(B, dtype=np.float32)
    if B >= 3:
        d2_mid = np.abs(med_spec[2:] - 2 * med_spec[1:-1] + med_spec[:-2])  # (B-2,)
        d2[1:-1] = d2_mid
        d2[0] = d2[1]
        d2[-1] = d2[-2]

    # --- robust z-scores for corr_def and roughness ---
    def robust_z(v: np.ndarray) -> np.ndarray:
        m = np.median(v)
        s = np.median(np.abs(v - m)) + eps
        return (v - m) / s

    z_corr = robust_z(corr_def)
    z_rough = robust_z(d2)

    # Combine (only penalize positive outliers)
    score = np.maximum(z_corr, 0.0) + gamma_rough * np.maximum(z_rough, 0.0)

    # Map score -> weight in (w_min, 1] with a smooth step around z_thresh
    # Large score => weight -> w_min
    t = sharpness * (score - z_thresh)  # (B,)

    sig = np.empty_like(t, dtype=np.float32)
    pos = t > 0
    # for t>0: 1/(1+exp(t)) = exp(-t)/(1+exp(-t)) avoids exp(large)
    sig[pos] = np.exp(-t[pos]) / (1.0 + np.exp(-t[pos]))
    # for t<=0: exp(t) is safe
    sig[~pos] = 1.0 / (1.0 + np.exp(t[~pos]))

    w = w_min + (1.0 - w_min) * sig
    w = w.astype(np.float32)

    # Force specified windows to 0 if wavenumbers given
    if (wavenumbers is not None) and (bad_windows is not None):
        wn = np.asarray(wavenumbers).astype(np.float32)
        for lo, hi in bad_windows:
            # handle descending or ascending wn
            w[(wn >= min(lo, hi)) & (wn <= max(lo, hi))] = 0.0

    w_t = torch.from_numpy(w).view(1, B, 1, 1)
    if device is not None:
        w_t = w_t.to(device)
    return w_t


def weighted_sad(x: torch.Tensor, x_hat: torch.Tensor, w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Weighted SAD loss.
    x, x_hat: (B, bands, H, W)
    w: (1, bands, 1, 1) or (bands,)
    """
    if w.ndim == 1:
        w = w.view(1, -1, 1, 1)
    # Apply weights
    xw = x * w
    xhw = x_hat * w

    B, C, H, W_ = xw.shape
    xw = xw.view(B, C, -1)
    xhw = xhw.view(B, C, -1)

    # Normalize
    xw = xw / (xw.norm(p=2, dim=1, keepdim=True) + eps)
    xhw = xhw / (xhw.norm(p=2, dim=1, keepdim=True) + eps)

    cos = (xw * xhw).sum(dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    sad = torch.acos(cos)  # (B, HW)

    return sad.mean()

q = 1
class CNNAEU(nn.Module):
    """
    Convolutional Autoencoder for Spectral–Spatial HU.
    """
    def __init__(self, num_bands, num_endmembers, patch_size=5, alpha=3.5, dropout=0.2):
        super().__init__()
        self.num_bands = num_bands
        self.num_endmembers = num_endmembers
        self.patch_size = patch_size

        # Encoder
        self.conv1 = nn.Conv2d(num_bands, 48*q, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(48*q)
        self.drop1 = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(48*q, num_endmembers, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(num_endmembers)
        self.drop2 = nn.Dropout2d(p=dropout)

        self.asc = SumToOne(alpha=alpha)

        # Decoder with positivity constraint
        self.decoder = PositiveDecoder(num_bands, num_endmembers, patch_size)

    def forward(self, x):  # (B, BANDS, H, W)
        z = F.leaky_relu(self.bn1(self.conv1(x)))
        z = self.drop1(z)
        a = F.leaky_relu(self.bn2(self.conv2(z)))
        a = self.drop2(a)
        a = self.asc(a)  # abundances (sum-to-one)
        x_hat = self.decoder(a)
        return x_hat, a

    def endmember_matrix(self):
        return self.decoder.endmember_matrix()

from sklearn.feature_extraction.image import extract_patches_2d
from torch.utils.data import Dataset, DataLoader

class HSIPatchDataset(Dataset):
    def __init__(self, cube, patch_size=5, max_patches=20000):
        """
        cube: (rows, cols, bands), float32, normalized [0,1]
        """
        self.patch_size = patch_size
        patches = extract_patches_2d(cube, (patch_size, patch_size), max_patches=max_patches)
        # to (N, bands, H, W)
        self.patches = np.transpose(patches, (0, 3, 1, 2)).astype(np.float32)

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.patches[idx])


def load_hsi_mat(path, cube_key="Y", gt_endmembers_key="GT", gt_abundances_key="S"):
    """
    Loads .mat with HSI cube and optional GT.
    - cube 'Y' can be (rows, cols, bands), or (bands, pixels) + 'lines'/'cols'
    - GT endmembers 'GT': (R, bands) or (bands, R)
    - GT abundances 'S': (rows, cols, R) or (R, rows*cols) etc.
    """
    data = sio.loadmat(path)
    Y = data[cube_key]

    if Y.ndim == 2:  # (bands, pixels)
        rows = int(np.array(data['lines']).squeeze())
        cols = int(np.array(data['cols']).squeeze())
        bands = Y.shape[0]
        cube = Y.T.reshape(rows, cols, bands)
    else:
        cube = Y

    cube = cube.astype(np.float32)
    mx = cube.max()
    if mx > 0: cube = cube / mx

    GT_M = None
    if gt_endmembers_key in data:
        GT_M = np.array(data[gt_endmembers_key]).astype(np.float32).squeeze()
        # shape to (bands, R)
        if GT_M.ndim == 1:
            GT_M = GT_M[:, None]
        if GT_M.shape[0] < GT_M.shape[1]:  # (R, bands) -> (bands, R)
            GT_M = GT_M.T

        # normalize each spectrum to unit max like in the TF utils
        GT_M = GT_M / (np.max(GT_M, axis=0, keepdims=True) + 1e-8)

    GT_S = None
    if gt_abundances_key in data:
        S = np.array(data[gt_abundances_key]).astype(np.float32)
        # reshape attempts to (rows, cols, R)
        if S.ndim == 2:
            # maybe (R, rows*cols)
            if S.shape[1] == cube.shape[0]*cube.shape[1]:
                R = S.shape[0]
                GT_S = S.reshape(R, cube.shape[0], cube.shape[1]).transpose(1,2,0)
            else:
                # maybe (rows*cols, R)
                if S.shape[0] == cube.shape[0]*cube.shape[1]:
                    GT_S = S.reshape(cube.shape[0], cube.shape[1], -1)
        elif S.ndim == 3:
            # could be (R, rows, cols)
            if S.shape[0] < S.shape[1] and S.shape[0] < S.shape[2]:
                GT_S = S.transpose(1,2,0)
            else:
                GT_S = S  # assume (rows, cols, R)

        # clip & renorm per pixel to sum 1
        if GT_S is not None:
            GT_S = np.clip(GT_S, 0, None)
            ssum = GT_S.sum(axis=2, keepdims=True) + 1e-8
            GT_S = GT_S / ssum

    return cube, GT_M, GT_S

patch_size=5
batch_size=64
epochs=500
lr=1e-3
alpha=3.5
dropout=0.2
device = "cuda" if torch.cuda.is_available() else "cpu"
num_endmembers = 10
num_bands = 1504

ds = HSIPatchDataset(cube_norm, patch_size=patch_size, max_patches=20000)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

# Model
model = CNNAEU(num_bands=num_bands,
                num_endmembers=num_endmembers,
                patch_size=patch_size,
                alpha=alpha,
                dropout=dropout).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

def count_trainable_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    return total_params

count_trainable_params(model)

# Training
path = "/home/shpande/Downloads/painting_unmix/models/cnnaeu_best_sub_wsad2.pth"
best_loss = np.inf

w_band = compute_band_weights(
    cube_norm,
    wavenumbers=wavenumbers,
    bad_windows=None,   # CO2 already dropped in your case; set if you keep it
    device=device
)

np.shape(np.unique(w_band.cpu().numpy()))

np.shape(w_band.cpu().numpy().ravel())

plt.figure(figsize = (30,10))
plt.scatter(wn_common, w_band.cpu().numpy().ravel())
plt.gca().invert_xaxis()

for epoch in range(1, epochs+1):
    model.train()
    total = 0.0
    nitems = 0

    for batch in dl:
        batch = batch.to(device)   # (B, bands, H, W)
        optimizer.zero_grad()
        x_hat, _ = model(batch)
        loss = weighted_sad(batch, x_hat, w_band)
        #loss = spectral_angle_distance(batch, x_hat)
        loss.backward()
        optimizer.step()

        bs = batch.size(0)
        total += loss.item() * bs
        nitems += bs

    avg_loss = total / max(nitems, 1)

    if avg_loss<best_loss:
      best_loss = avg_loss
      torch.save(model.state_dict(), path)

    print(f"[{epoch:03d}/{epochs}] loss={avg_loss:.6f}")

# Training
path = "/home/shpande/Downloads/painting_unmix/models/cnnaeu_best_sub_wsad2.pth"                             
best_loss = np.inf

# Full cube tensor for periodic evaluation & final abundances
cube_t = torch.from_numpy(np.transpose(cube_norm.astype('float32'), (2,0,1))[None, ...]).to(device)  # (1, bands, rows, cols)

model.load_state_dict(torch.load(path, weights_only=False))
model.eval()
with torch.no_grad():
    xhat_full, abund_full = model(cube_t)
    A_est = abund_full[0].detach().cpu().numpy().transpose(1,2,0)  # (rows, cols, R)
    M_est = model.endmember_matrix().detach().cpu().numpy()  # (bands, R)

np.shape(M_est)

plt.figure(figsize=(20, 16))
plt.imshow(A_est[:,:,4])

A_est2 = np.flip(np.rot90(np.rot90(A_est)),1)
np.shape(A_est2)

em_wt = np.empty(np.shape(M_est))
em_wt = M_est-np.min(M_est)
em_wt = em_wt/np.max(em_wt)

a, b, c = A_est.shape

rows, cols = 1, 10
fig, axes = plt.subplots(rows, cols, figsize=(cols*2, 10))  # tweak figsize as you like

axes = axes.ravel()

for k in range(c):
    ax = axes[k]
    im = ax.imshow(A_est2[:, :, k], aspect='auto')  # let it adapt to axes size
    ax.set_title(f"Map {k+1}", fontsize=20)
    ax.axis('off')

# If there are unused axes (in case c < rows*cols)
for k in range(c, rows * cols):
    axes[k].axis('off')

plt.tight_layout()
plt.show()


# X has shape (1504, 10)
n_samples, n_signals = em_wt.shape   # 1504, 10

x = wavenumbers         # x-axis: sample index (0..1503)

rows, cols = 5, 2
fig, axes = plt.subplots(rows, cols, figsize=(20, 10), sharex=True)
axes = axes.ravel()

for k in range(n_signals):
    ax = axes[k]
    ax.plot(x, em_wt[:, k])
    ax.set_title(f"Endmember {k+1}", fontsize=15)
    ax.set_xlabel("Reflectance/Intensity", fontsize=15)
    ax.set_ylabel("Bands", fontsize=15)

# In case n_signals < rows*cols, hide the extras (not needed here, but safe)
for k in range(n_signals, rows * cols):
    axes[k].axis('off')
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# em_wt: (B_sub, K) where B_sub = 1403 after dropping
# wavenumbers_sub: (B_sub,) matching em_wt rows (descending)
# choose split using the actual gap in wavenumber_sub rather than hardcoding indices

def find_gap_split(x, gap_factor=10.0):
    """
    Find the index where x jumps (because a band window was removed).
    x is assumed 1D and monotonic (descending OK).
    Returns split index s such that x[:s] and x[s:] are the two segments.
    """
    x = np.asarray(x)
    dx = np.abs(np.diff(x))
    med = np.median(dx)
    # gap where step is much larger than typical spacing
    gap_idx = np.argmax(dx)  # biggest jump
    if dx[gap_idx] < gap_factor * med:
        # fallback: no obvious gap found
        return None
    return gap_idx + 1  # split happens after gap_idx

split = find_gap_split(wavenumbers_sub)

rows, cols = 5, 2
fig, axes = plt.subplots(rows, cols, figsize=(20, 10), sharex=True)
axes = axes.ravel()

for k in range(em_wt.shape[1]):
    ax = axes[k]
    if split is None:
        ax.plot(wavenumbers_sub, em_wt[:, k])
    else:
        line1, = ax.plot(wavenumbers_sub[:split], em_wt[:split, k])
        ax.plot(wavenumbers_sub[split:], em_wt[split:, k], color=line1.get_color())
    ax.set_title(f"Endmember {k+1}", fontsize=15)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=15)
    ax.set_ylabel("Intensity", fontsize=15)
    #ax.invert_xaxis()
fig.suptitle('Endmembers with min-max scaling and without baseline correction')
for k in range(em_wt.shape[1], rows * cols):
    axes[k].axis('off')
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# X has shape (1504, 10)
n_samples, n_signals = em_wt.shape   # 1504, 10

x = np.arange(n_samples)         # x-axis: sample index (0..1503)

rows, cols = 5, 2
fig, axes = plt.subplots(rows, cols, figsize=(20, 10), sharex=True)
axes = axes.ravel()

for k in range(n_signals):
    ax = axes[k]
    ax.plot(x, em_wt[:, k])
    ax.set_title(f"Endmember {k+1}", fontsize=15)
    ax.set_xlabel("Reflectance/Intensity", fontsize=15)
    ax.set_ylabel("Bands", fontsize=15)

# In case n_signals < rows*cols, hide the extras (not needed here, but safe)
for k in range(n_signals, rows * cols):
    axes[k].axis('off')

plt.tight_layout()
plt.show()

res = sio.loadmat("Downloads/Unmixing/AA_unmix/unmix_results.mat")