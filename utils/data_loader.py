"""
utils/data_loader.py  —  Load all 4 datasets
Returns: Y (L,N), M_gt (L,p), A_gt (p,N) or None
"""
import scipy.io as sio
import numpy as np
import os


def load_urban(data_dir='./data'):
    img  = sio.loadmat(os.path.join(data_dir, 'Urban', 'Urban_R162.mat'))
    gt   = sio.loadmat(os.path.join(data_dir, 'Urban', 'end4_groundTruth.mat'))
    # Try common variable names for image cube
    for key in ['Y', 'data', 'X', 'img']:
        if key in img:
            Y = img[key].astype(np.float32)
            break
    if Y.ndim == 3:
        H, W, L = Y.shape; Y = Y.reshape(-1, L).T
    M_gt = gt['M'].astype(np.float32)   # (162, 4)
    A_gt = gt['A'].astype(np.float32)   # (4, 94249)
    print(f'Urban loaded       Y:{Y.shape}  M:{M_gt.shape}  A:{A_gt.shape}')
    return Y, M_gt, A_gt


def load_jasperridge(data_dir='./data'):
    img  = sio.loadmat(os.path.join(data_dir, 'JasperRidge', 'JasperRidge2_R198.mat'))
    gt   = sio.loadmat(os.path.join(data_dir, 'JasperRidge', 'end4.mat'))
    Y        = img['Y'].astype(np.float32)                    # (198, 10000)
    max_val  = float(img['maxValue'].flatten()[0])            # 5000
    Y        = Y / max_val                                    # → [0, 1]
    M_gt = gt['M'].astype(np.float32)   # (198, 4)
    A_gt = gt['A'].astype(np.float32)   # (4, 10000)
    print(f'JasperRidge loaded Y:{Y.shape}  M:{M_gt.shape}  A:{A_gt.shape}')
    return Y, M_gt, A_gt


def load_cuprite(data_dir='./data'):
    img  = sio.loadmat(os.path.join(data_dir, 'Cuprite', 'Cuprite_S1_R188.mat'))
    gt   = sio.loadmat(os.path.join(data_dir, 'Cuprite', 'cuprite.mat'))
    for key in ['Y', 'data', 'X', 'img']:
        if key in img:
            Y = img[key].astype(np.float32)
            break
    if Y.ndim == 3:
        H, W, L = Y.shape; Y = Y.reshape(-1, L).T
    # W1 shape is (12, 188) → transpose to (188, 12)
    M_gt = gt['W1'].astype(np.float32).T
    print(f'Cuprite loaded     Y:{Y.shape}  M:{M_gt.shape}  A:None')
    return Y, M_gt, None


def load_synthetic(data_dir='./data', snr=30):
    fname = os.path.join(data_dir, 'synthetic_data', f'synthetic_snr{snr}dB.mat')
    d     = sio.loadmat(fname)
    Y    = d['Y'].astype(np.float32)      # (224, 4096)
    M_gt = d['M_gt'].astype(np.float32)  # (224, 5)
    A_gt = d['A_gt'].astype(np.float32)  # (5,   4096)
    print(f'Synthetic loaded   Y:{Y.shape}  M:{M_gt.shape}  A:{A_gt.shape}  SNR={snr}dB')
    return Y, M_gt, A_gt


def load_dataset(name, data_dir='./data', snr=30):
    """Main loader. name: 'Urban'|'JasperRidge'|'Cuprite'|'Synthetic'"""
    loaders = {
        'Urban'      : lambda d: load_urban(d),
        'JasperRidge': lambda d: load_jasperridge(d),
        'Cuprite'    : lambda d: load_cuprite(d),
        'Synthetic'  : lambda d: load_synthetic(d, snr=snr),
    }
    assert name in loaders, f"Unknown dataset '{name}'"
    print(f'\nLoading {name}...')
    return loaders[name](data_dir)
