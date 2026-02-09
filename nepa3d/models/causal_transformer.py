import torch
import torch.nn as nn


class CausalTransformer(nn.Module):
    def __init__(self, d_model=384, nhead=6, num_layers=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        t = x.size(1)
        attn_mask = torch.triu(
            torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1
        )
        return self.enc(x, mask=attn_mask)
