from __future__ import annotations

import torch
import torch.nn as nn


class SimplePointEncoder(nn.Module):
    """Small PointNet-style encoder for Phase-0/1 smoke tests.

    This is not intended as the final paper backbone. It lets the ViewAction
    pipeline be tested before wiring PointGPT.
    """
    def __init__(self, dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, 256), nn.GELU(),
        )
        self.proj = nn.Sequential(nn.Linear(256, dim), nn.LayerNorm(dim))
        self.output_dim = dim

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: B N 3
        h = self.net(points.float())
        h = h.max(dim=1).values
        return self.proj(h)
