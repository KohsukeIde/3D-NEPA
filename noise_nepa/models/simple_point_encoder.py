from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePointEncoder(nn.Module):
    """Small PointNet-style encoder for smoke tests.

    This is intentionally lightweight. Use PointGPT adapter for stronger experiments.
    """
    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, 256), nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(256 * 2, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B N 3
        h = self.mlp(x)
        h_max = h.max(dim=1).values
        h_mean = h.mean(dim=1)
        z = self.proj(torch.cat([h_max, h_mean], dim=-1))
        return F.normalize(z, dim=-1)
