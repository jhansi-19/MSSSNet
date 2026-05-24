# MSSS-Net: Multi-Stream Spatial-Spectral Network for Hyperspectral Image Unmixing

A PyTorch implementation of MSSS-Net for hyperspectral image unmixing, combining single-stream and multi-view spectral processing with spatial-spectral features.

## 📋 Overview

MSSS-Net is a deep learning architecture for **hyperspectral unmixing** that estimates endmembers and abundances from hyperspectral images. The model employs a two-stream approach:

- **SSR (Single Spectral Stream)**: Processes the full spectral signature using an RNN
- **MSSR (Multi-view Spectral Stream RNN)**: Partitions the spectrum into multiple views and processes each with cascaded RNNs, capturing spectral diversity
- **Shared Decoder**: Both streams share a linear decoder whose weights represent the endmembers

The architecture leverages spatial-spectral patches to improve unmixing accuracy by capturing local spatial context.

**Reference**: Qi et al., IEEE TGRS 2023, "MSSS-Net: Multi-Stream Spatial-Spectral Network for Hyperspectral Unmixing"

## 🌟 Features

- **Two-Stream Architecture**: Combines full-spectrum and multi-view spectral processing
- **Spatial-Spectral Learning**: Uses patches to capture spatial context
- **Numerically Stable**: Implements stable loss functions with proper normalization
- **Multi-Dataset Support**: Pre-configured for Urban, JasperRidge, Cuprite, and Synthetic datasets
- **Flexible Configuration**: Easy hyperparameter tuning through `config.py`
- **Comprehensive Metrics**: Evaluates using SAD (Spectral Angle Distance) and RMSE

## 📁 Project Structure

```
├── config.py                  # Configuration file with all hyperparameters
├── losses.py                  # Loss functions (SAD, L1/2 sparsity, combined loss)
├── requirements.txt           # Python dependencies
├── models/
│   ├── __init__.py
│   ├── encoder.py            # CascadedRNNEncoder for spectral processing
│   ├── ssr.py                # Single Spectral Stream
│   ├── mssr.py               # Multi-view Spectral Stream RNN
│   └── msss_net.py           # Full MSSS-Net architecture
├── utils/
│   ├── __init__.py
│   ├── data_loader.py        # Data loading utilities
│   ├── metrics.py            # Evaluation metrics (SAD, RMSE, accuracy)
│   └── preprocessing.py      # Data preprocessing and normalization
├── data/
│   ├── cuprite/              # Cuprite mine dataset (12 endmembers)
│   ├── JasperRidge/          # Jasper Ridge dataset (4 endmembers)
│   ├── Urban/                # Urban dataset (4 endmembers)
│   └── synthetic_data/       # Synthetic datasets with known ground truth
├── checkpoints/              # Saved model checkpoints
├── results/                  # Evaluation results
└── README.md
```

## 🔧 Installation

### Requirements
- Python 3.7+
- PyTorch >= 1.8.0
- NumPy, SciPy, scikit-learn
- Matplotlib, tqdm, pandas

### Setup

1. **Clone or download the repository**
   ```bash
   cd additional_project2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare datasets** (optional)
   - Place hyperspectral datasets in the `data/` directory
   - Supported formats: `.mat` (MATLAB files)
   - Pre-configured datasets: Urban, JasperRidge, Cuprite, Synthetic

## ⚙️ Configuration

Edit `config.py` to customize hyperparameters:

```python
class Config:
    # Dataset selection
    dataset = 'Urban'  # Choose: 'Urban', 'JasperRidge', 'Cuprite', 'Synthetic'
    
    # Model parameters
    num_endmembers = 4      # Number of endmembers (p)
    hidden_size = 128       # LSTM hidden units
    patch_size = 3          # Spatial window size (k×k)
    num_views = 2           # Number of spectral partitions (s)
    
    # Training parameters
    epochs = 100
    lr = 1e-4               # Learning rate
    lam = 1e-5              # Sparsity weight (L1/2 regularization)
    mu = 0.5                # SSR/MSSR balance parameter
    
    # Paths
    data_dir = './data'
    save_dir = './checkpoints'
    result_dir = './results'
```

### Dataset Specifications

| Dataset | Spatial Dims (H×W) | Spectral Bands (L) | Endmembers (p) | Ground Truth |
|---------|-------------------|------------------|----------------|--------------|
| Urban | 307×307 | 162 | 4 | ✓ |
| JasperRidge | 100×100 | 198 | 4 | ✓ |
| Cuprite | 250×190 | 188 | 12 | ✓ |
| Synthetic | 64×64 | 224 | 5 | ✓ |

## 🚀 Usage

### Training

```python
from config import Config
from models.msss_net import MSSsNet
from utils.data_loader import load_dataset
import torch

# Load configuration
cfg = Config()

# Load data
X, A_true, E_true = load_dataset(cfg.dataset, cfg.data_dir)

# Initialize model
model = MSSsNet(
    L=cfg.DIMS[cfg.dataset][2],      # Spectral bands
    hidden=cfg.hidden_size,
    p=cfg.num_endmembers,
    s=cfg.num_views,
    Lv=cfg.DIMS[cfg.dataset][2] // cfg.num_views
)

# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop (implementation in your training script)
```

### Evaluation

```python
from utils.metrics import compute_sad, compute_rmse

# Get estimated endmembers
E_estimated = model.endmembers()

# Compute metrics
sad_error = compute_sad(E_true, E_estimated)
rmse_error = compute_rmse(Y_true, Y_estimated)

print(f"Mean SAD: {sad_error:.4f} degrees")
print(f"RMSE: {rmse_error:.4f}")
```

## 📊 Loss Functions

The model uses numerically stable loss functions:

### Spectral Angle Distance (SAD)
```python
def sad_loss(y_true, y_pred):
    """Measures angle between true and predicted spectral signatures"""
    y_true_n = F.normalize(y_true, p=2, dim=1, eps=1e-8)
    y_pred_n = F.normalize(y_pred, p=2, dim=1, eps=1e-8)
    cos = (y_true_n * y_pred_n).sum(dim=1)
    cos = torch.clamp(cos, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos).mean()
```

### Combined Loss (Eq. 28)
```
L_total = μ·(L_ssr + λ·Ω(A_ssr)) + (1-μ)·(L_mssr + λ·Ω(A_mssr))
```
Where:
- μ: balances SSR and MSSR contributions
- λ: controls sparsity regularization strength
- Ω: L1/2 sparsity norm

## 📈 Results

Results are saved in `results/EVALUATION_RESULTS.txt` with per-dataset metrics:

**Urban Dataset**
- Mean SAD: 15.57° (error < 20°)
- RMSE: 0.3077
- Per-endmember SAD: 7.83° - 19.78°

**JasperRidge Dataset**
- Mean SAD: 17.33°
- RMSE: 0.3510
- Per-endmember SAD: 7.57° - 36.32°

## 📚 Key Components

### Models

| Module | Purpose |
|--------|---------|
| `encoder.py` | CascadedRNNEncoder for spectral processing |
| `ssr.py` | Single Spectral Stream (full spectrum RNN) |
| `mssr.py` | Multi-view Spectral Stream with multi-view RNNs |
| `msss_net.py` | Complete model combining SSR, MSSR + shared decoder |

### Utilities

| Module | Purpose |
|--------|---------|
| `data_loader.py` | Load MATLAB datasets and prepare tensors |
| `preprocessing.py` | Normalization and data augmentation |
| `metrics.py` | SAD, RMSE, abundance accuracy evaluation |

## 🔍 Architecture Details

### Two-Stream Processing
1. **SSR Path**: Full spectrum → RNN → Abundance vector (B, p)
2. **MSSR Path**: Partitioned spectrum (s views) → Cascaded RNNs → Per-view abundances (B, s, p) → Fused abundances (B, p)
3. **Shared Decoder**: Both abundance vectors → Linear layer (weights = endmembers) → Reconstructed spectra

### Spatial-Spectral Processing
- Extracts k×k spatial patches (default: 3×3)
- Each patch processed independently
- Enables learning of spatial-spectral context

## 💡 Tips & Best Practices

1. **Hyperparameter Tuning**:
   - Adjust `mu` (0-1) to balance SSR/MSSR contributions
   - Increase `lam` for sparser solutions
   - Reduce `lr` if training is unstable

2. **Dataset Selection**:
   - Use Synthetic data for quick testing
   - Urban/JasperRidge for standard benchmarking
   - Cuprite for challenging multi-endmember unmixing

3. **Training**:
   - Monitor SAD loss on validation set
   - Use checkpointing to save best models
   - Normalize input data to [0, 1] range

4. **Evaluation**:
   - Compare SAD and RMSE with baseline methods
   - Visualize abundance maps for quality assessment
   - Check endmember extraction accuracy

## 📄 License

[Add appropriate license information here]

## 🙏 Citation

If you use this code, please cite the original paper:

```bibtex
@article{qi2023msss,
  title={MSSS-Net: Multi-Stream Spatial-Spectral Network for Hyperspectral Unmixing},
  author={Qi, et al.},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2023}
}
```

## 📧 Contact & Support

For issues, questions, or contributions, please open an issue or contact the project maintainers.

---

**Last Updated**: May 2024  
**Status**: Active Development
