"""
models/ssr.py  —  Spatial Stream RNN
Paper Section III-A: patch-based recurrent perception
Input : (B, k*k, L) → Output: (B, p) abundance of centre pixel
"""
import torch.nn as nn
from models.encoder import CascadedRNNEncoder


class SSR(nn.Module):
    def __init__(self, num_bands, hidden, num_endmembers):
        super().__init__()
        self.enc = CascadedRNNEncoder(num_bands, hidden, num_endmembers, 'spatial')

    def forward(self, patch):              # (B, k*k, L)
        ab = self.enc(patch)              # (B, p)
        ab = ab / (ab.norm(p=1, dim=1, keepdim=True) + 1e-6)  # ASC Eq.16
        return ab
