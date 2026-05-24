"""
utils/preprocessing.py  —  Preprocessing pipeline for MSSS-Net
Steps:
  1. normalize()     — scale Y to [0,1] per band
  2. make_patches()  — k×k spatial patches for SSR
  3. sd_partition()  — multiview spectral split for MSSR
  4. vca()           — VCA endmember initialisation for decoder
"""
import numpy as np


# -- 1. Normalisation --
def normalize(Y):
    """Y: (L,N) -> (L,N) in [0,1] per band"""
    Y_min  = Y.min(axis=1, keepdims=True)
    Y_max  = Y.max(axis=1, keepdims=True)
    Y_norm = (Y - Y_min) / (Y_max - Y_min + 1e-8)
    print(f'Normalized  Y:{Y_norm.shape}  range:[{Y_norm.min():.4f},{Y_norm.max():.4f}]')
    return Y_norm


# -- 2. Spatial Patch Construction (SSR) --
def make_patches(Y, H, W, patch_size=3):
    """
    Build k×k spatial patches for every pixel.
    Paper Section III-A, best patch_size=3 (Fig.5)
    Y: (L,N) -> patches: (N, k*k, L)
    """
    L, N   = Y.shape
    img    = Y.T.reshape(H, W, L)
    pad    = patch_size // 2
    padded = np.pad(img, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
    patches = []
    for i in range(H):
        for j in range(W):
            p = padded[i:i+patch_size, j:j+patch_size, :]
            patches.append(p.reshape(-1, L))
    patches = np.array(patches, dtype=np.float32)   # (N, k*k, L)
    print(f'Patches     shape:{patches.shape}  (N, k*k={patch_size**2}, L)')
    return patches


# -- 3. Multiview Spectral Partitioning (MSSR) --
def sd_partition(Y, num_views=2):
    """
    Spectrometer-Driven (SD) interleaved band partitioning.
    Paper Section III-B, best num_views=2 (Fig.6)

    FIX: When L is not divisible by num_views, truncate bands
         so every view has exactly Lv = L // num_views bands.
         This prevents 'all input arrays must have the same shape' error.

    Y: (L,N) -> mv_data: (N, s, Lv)
    """
    L, N = Y.shape
    Lv   = L // num_views          # bands per view (floor division)

    views = []
    for v in range(num_views):
        # Interleaved: view v gets bands v, v+s, v+2s, ...
        # Take exactly Lv bands - truncates leftover bands cleanly
        band_idx = list(range(v, L, num_views))[:Lv]
        views.append(Y[band_idx, :].T)             # (N, Lv)

    mv_data = np.stack(views, axis=1).astype(np.float32)  # (N, s, Lv)
    print(f'MV Spectral shape:{mv_data.shape}  (N, s={num_views}, Lv={Lv})')
    return mv_data


# -- 4. VCA Endmember Initialisation --
def vca(Y, p):
    """
    Improved VCA — Random pixel selection with highest-magnitude basis vectors.
    Better than original for small p.
    Y: (L,N), p: number of endmembers -> A_init: (L,p)
    """
    L, N = Y.shape
    
    # Center data
    u = Y.mean(axis=1, keepdims=True)
    Y_c = Y - u
    
    # PCA projection onto top p components
    _, _, Vt = np.linalg.svd(Y_c, full_matrices=False)
    Yp = Vt[:p, :].T  # Project to PCA space (N, p)
    
    # Select p pixels with highest magnitudes in PCA space
    # This is more stable than random direction selection
    A = np.zeros((L, p))
    
    # Select pixel 1: maximum norm in PCA space
    idx = [int(np.argmax(np.linalg.norm(Yp, axis=1)))]
    A[:, 0] = Y[:, idx[0]]
    
    # Select remaining p-1 pixels using iterative maximum distance
    for i in range(1, p):
        # Find pixel farthest from all selected pixels
        selected = Yp[idx, :]  # (i, p)
        
        # Compute min distance from each pixel to any selected pixel
        distances = []
        for j in range(N):
            if j not in idx:
                min_dist = np.min(np.linalg.norm(Yp[j, :] - selected, axis=1))
                distances.append(min_dist)
            else:
                distances.append(-np.inf)
        
        next_idx = int(np.argmax(distances))
        idx.append(next_idx)
        A[:, i] = Y[:, next_idx]
    
    # Normalize endmembers (optional, can help convergence)
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-8)
    
    print(f'VCA init    A_init:{A.shape}  (improved initialization)')
    return A   # (L, p)


# -- Full pipeline --
def preprocess(Y, H, W, patch_size=3, num_views=2, num_endmembers=4):
    """Run all preprocessing steps."""
    print('\n-- Preprocessing --')
    Y_norm  = normalize(Y)
    patches = make_patches(Y_norm, H, W, patch_size)
    mv_data = sd_partition(Y_norm, num_views)
    A_init  = vca(Y_norm, num_endmembers)
    print('-- Done --\n')
    return Y_norm, patches, mv_data, A_init