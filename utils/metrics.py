"""
utils/metrics.py  —  SAD and RMSE evaluation metrics
Paper Equations 29 and 30.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def sad(a, b):
    """Spectral Angle Distance between two spectra. Returns degrees."""
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return np.arccos(np.clip(cos, -1.0, 1.0)) * 180.0 / np.pi


def mean_sad(M_gt, M_est):
    """
    Mean SAD with Hungarian matching.
    Includes NaN/Inf guard to catch invalid model outputs.
    """
    # ── Guard: check for invalid values ──────────────────────
    if np.isnan(M_est).any() or np.isinf(M_est).any():
        print('  WARNING: M_est contains NaN/Inf — model may not be trained yet')
        print('  Run python train.py first, then evaluate.py')
        return np.nan, np.array([np.nan])

    p    = M_gt.shape[1]
    cost = np.array([[sad(M_gt[:, i], M_est[:, j])
                      for j in range(p)] for i in range(p)])

    # ── Guard: check cost matrix ──────────────────────────────
    if np.isnan(cost).any() or np.isinf(cost).any():
        print('  WARNING: SAD cost matrix has invalid entries')
        return np.nan, np.array([np.nan])

    row, col = linear_sum_assignment(cost)
    per_em   = cost[row, col]
    return per_em.mean(), per_em


def rmse(A_gt, A_est):
    """RMSE between GT and estimated abundances. A_gt, A_est: (p, N)."""
    return np.sqrt(np.mean((A_gt - A_est) ** 2))


def evaluate(M_gt, M_est, A_gt, A_est, dataset_name=''):
    """Print full evaluation report."""
    print(f'\n{"="*52}')
    print(f'  Results  —  {dataset_name}')
    print(f'{"="*52}')
    if M_gt is not None:
        avg, per = mean_sad(M_gt, M_est)
        print(f'  Mean SAD  : {avg:.4f} deg  (x10^-2 = {avg*100:.2f})')
        for i, v in enumerate(per):
            print(f'    Endmember #{i+1}: {v:.4f} deg')
    if A_gt is not None:
        r = rmse(A_gt, A_est)
        print(f'  RMSE      : {r:.4f}       (x10^-2 = {r*100:.2f})')
    print(f'{"="*52}\n')
