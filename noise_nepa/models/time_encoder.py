from __future__ import annotations

import math
import torch
import torch.nn as nn


class SinusoidalTimeEncoder(nn.Module):
    def __init__(self, num_steps: int = 1000, dim: int = 128, out_dim: int = 384):
        super().__init__()
        self.num_steps = num_steps
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def _emb(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float() / float(max(1, self.num_steps - 1))
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0, math.log(10000.0), half, device=t.device) * -1.0)
        args = t[:, None] * freqs[None, :] * 1000.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([self._emb(t), self._emb(s), self._emb(t - s)], dim=-1))
