from __future__ import annotations
import torch
import torch.nn as nn


class LevelEncoder(nn.Module):
    def __init__(self, max_levels: int = 12, dim: int = 256):
        super().__init__()
        self.emb = nn.Embedding(max_levels, dim)
        self.delta = nn.Embedding(max_levels, dim)
        self.mlp = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, k: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        d = (m - k).clamp_min(0).clamp_max(self.delta.num_embeddings - 1)
        return self.mlp(torch.cat([self.emb(k), self.emb(m), self.delta(d)], dim=-1))
