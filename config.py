"""
config.py  —  All hyperparameters for MSSS-Net
Paper: Qi et al., IEEE TGRS 2023, Section IV-B-1
"""

class Config:
    # ── Dataset ─────────────────────────────────────────────────────────
    # Choose one: 'Urban' | 'JasperRidge' | 'Cuprite' | 'Synthetic'
    dataset        = 'Urban'

    # ── Model ───────────────────────────────────────────────────────────
    num_endmembers = 4       # p  (Urban=4, JasperRidge=4, Cuprite=12, Synthetic=5)
    hidden_size    = 128     # LSTM hidden units
    patch_size     = 3       # k×k spatial window  (paper Fig.5: best = 3x3)
    num_views      = 2       # s spectral partitions (paper Fig.6: best = 2)

    # ── Training ────────────────────────────────────────────────────────
    epochs         = 100     # paper Section IV-B-1
    lr             = 1e-4    # Adam learning rate
    lam            = 1e-5    # lambda sparsity weight   (REDUCED from 5e-5 for better endmember learning)
    mu             = 0.5     # mu  SSR/MSSR balance  (paper Eq.28, Fig.14)

    # ── Paths ────────────────────────────────────────────────────────────
    data_dir       = './data'
    save_dir       = './checkpoints'
    result_dir     = './results'


# Spatial + spectral dimensions per dataset: (H, W, L, p)
DIMS = {
    'Urban'      : (307, 307, 162,  4),
    'JasperRidge': (100, 100, 198,  4),
    'Cuprite'    : (250, 190, 188, 12),
    'Synthetic'  : ( 64,  64, 224,  5),
}

# Endmember class names per dataset
LABELS = {
    'Urban'      : ['Asphalt', 'Grass', 'Tree', 'Roof'],
    'JasperRidge': ['Tree', 'Water', 'Soil', 'Road'],
    'Cuprite'    : ['Alunite','Andradite','Buddingtonite','Dumortierite',
                    'Kaolinite1','Kaolinite2','Muscovite','Montmorillonite',
                    'Nontronite','Pyrope','Sphene','Chalcedony'],
    'Synthetic'  : ['Endmember1','Endmember2','Endmember3',
                    'Endmember4','Endmember5'],
}
