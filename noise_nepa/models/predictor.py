from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentTransitionPredictor(nn.Module):
    def __init__(self, dim: int = 384, hidden: int = 768, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        dz = self.net(torch.cat([z, time_emb], dim=-1))
        out = z + dz if self.residual else dz
        return F.normalize(out, dim=-1)
