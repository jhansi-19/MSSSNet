"""
models/msss_net.py  —  Full MSSS-Net: Two-Stream + Shared Decoder
Paper Fig.1: SSR + MSSR share one linear decoder (weights = endmembers)
"""
import torch
import torch.nn as nn
from models.ssr  import SSR
from models.mssr import MSSR


class MSSsNet(nn.Module):
    def __init__(self, L, hidden, p, s, Lv):
        super().__init__()
        self.ssr     = SSR(L, hidden, p)
        self.mssr    = MSSR(Lv, hidden, p, s)
        self.decoder = nn.Linear(p, L, bias=False)  # weights = endmembers (L,p)

    def forward(self, patch, mv):
        ab_ssr            = self.ssr(patch)          # (B, p)
        ab_mssr, ab_views = self.mssr(mv)            # (B,p), (B,s,p)
        recon_ssr         = self.decoder(ab_ssr)     # (B, L)
        recon_mssr        = self.decoder(ab_mssr)    # (B, L)
        return ab_ssr, recon_ssr, ab_mssr, recon_mssr, ab_views

    def endmembers(self):
        """Return estimated endmember matrix (L, p)"""
        return self.decoder.weight.data              # (L, p)

    def init_decoder(self, A_init):
        """Initialise decoder with VCA endmembers. A_init: (L,p) numpy"""
        import torch
        with torch.no_grad():
            self.decoder.weight.copy_(
                torch.tensor(A_init, dtype=torch.float32))

    def enforce_anc(self):
        """Clamp decoder weights >=0 (endmember reflectance non-negative)"""
        with torch.no_grad():
            self.decoder.weight.clamp_(min=0)
