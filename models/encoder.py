"""
models/encoder.py  —  Cascaded RNN Encoder  (STABLE VERSION)
Fixes: input projection, LayerNorm, orthogonal init, mini-batch safe
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CascadedRNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, mode='spatial'):
        super().__init__()
        self.mode = mode

        # Input projection: reduce dimensionality before LSTM
        proj_size = min(64, input_size)
        self.proj = nn.Sequential(
            nn.Linear(input_size, proj_size, bias=False),
            nn.LayerNorm(proj_size),
            nn.Tanh()
        )

        # Bidirectional LSTM (Layer 1)
        self.bi_lstm = nn.LSTM(
            input_size    = proj_size,
            hidden_size   = hidden_size,
            batch_first   = True,
            bidirectional = True
        )
        self.bi_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(0.1)

        # Unidirectional LSTM (Layer 2)
        self.uni_lstm = nn.LSTM(
            input_size    = hidden_size * 2,
            hidden_size   = out_size,
            batch_first   = True,
            bidirectional = False
        )

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.data.dim() < 2:
                # 1D tensors (biases) — just zero them
                param.data.zero_()
                # Set forget gate bias to 1 for better gradient flow
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)
            elif 'weight_hh' in name:
                # Recurrent weights — orthogonal init
                nn.init.orthogonal_(param.data)
            elif 'weight_ih' in name:
                # Input weights — xavier init
                nn.init.xavier_uniform_(param.data)
            else:
                # Projection and other 2D weights
                nn.init.xavier_uniform_(param.data)

    def forward(self, x):
        """x: (B, seq_len, input_size)"""
        B, T, _ = x.shape

        # Project each timestep
        x_flat = x.reshape(B * T, -1)
        x_proj = self.proj(x_flat).reshape(B, T, -1)  # (B, T, proj_size)

        # BiLSTM
        bi_out, _ = self.bi_lstm(x_proj)   # (B, T, H*2)
        bi_out    = self.bi_norm(bi_out)
        bi_out    = torch.tanh(bi_out)
        bi_out    = self.dropout(bi_out)

        # UniLSTM
        uni_out, _ = self.uni_lstm(bi_out)  # (B, T, out_size)

        if self.mode == 'spatial':
            center = T // 2
            # Use softplus instead of relu to maintain gradients
            # softplus(x) = log(1 + exp(x)) — always positive, smooth
            out = torch.nn.functional.softplus(uni_out[:, center, :], beta=1.0)
            return out  # (B, p)
        else:
            return torch.sigmoid(uni_out)          # (B, s, p)