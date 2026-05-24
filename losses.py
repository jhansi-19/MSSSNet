"""
losses.py  —  Numerically stable loss functions for MSSS-Net
Paper Equations 18, 27, 28
"""
import torch
import torch.nn.functional as F


def sad_loss(y_true, y_pred):
    """SAD loss — fully stable via F.normalize."""
    y_true_n = F.normalize(y_true, p=2, dim=1, eps=1e-8)
    y_pred_n = F.normalize(y_pred, p=2, dim=1, eps=1e-8)
    cos  = (y_true_n * y_pred_n).sum(dim=1)
    cos  = torch.clamp(cos, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos).mean()


def l_half(x):
    """L1/2 sparsity norm — stable, no negative sqrt."""
    return x.clamp(min=0).sqrt().sum(dim=1).mean()


def total_loss(recon_ssr, recon_mssr, Y_batch, ab_ssr, ab_views, cfg):
    """Combined loss. Paper Eq.28."""
    loss_ssr  = sad_loss(Y_batch, recon_ssr) + cfg.lam * l_half(ab_ssr)
    sp_mssr   = sum(l_half(ab_views[:, v, :]) for v in range(cfg.num_views))
    loss_mssr = sad_loss(Y_batch, recon_mssr) + cfg.lam * sp_mssr / cfg.num_views
    
    # Add penalty for small/zero abundances (encourages meaningful reconstruction)
    # This helps both streams learn non-trivial solutions
    eps_penalty_ssr = 0.001 * torch.mean(torch.clamp(-ab_ssr, min=0))  # Penalty for negative values
    eps_penalty_mssr = 0.001 * torch.mean(torch.clamp(-ab_views, min=0))
    
    return cfg.mu * (loss_ssr + eps_penalty_ssr) + (1.0 - cfg.mu) * (loss_mssr + eps_penalty_mssr)