from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePointEncoder(nn.Module):
    def __init__(self, dim: int = 256, hidden: int = 128):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.out = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.point_mlp(x)
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        z = self.out(pooled)
        return F.normalize(z, dim=-1)
