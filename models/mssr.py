"""
models/mssr.py  —  Multiview Spectral Stream RNN
Paper Section III-C: spectral complementary recurrent fusion
Input : (B, s, Lv) → Output: (B,p) fused + (B,s,p) per-view abundances
"""
import torch.nn as nn
from models.encoder import CascadedRNNEncoder


class MSSR(nn.Module):
    def __init__(self, band_per_view, hidden, num_endmembers, num_views):
        super().__init__()
        self.s   = num_views
        self.enc = CascadedRNNEncoder(band_per_view, hidden, num_endmembers, 'spectral')

    def forward(self, mv):                 # (B, s, Lv)
        out = self.enc(mv)                # (B, s, p)
        out = out / (out.norm(p=1, dim=2, keepdim=True) + 1e-6)  # ASC Eq.24
        ab  = out.mean(dim=1)            # fuse views Eq.25 → (B, p)
        return ab, out
