from __future__ import annotations

import torch
import torch.nn as nn


class ActionEncoder(nn.Module):
    def __init__(self, num_actions: int, action_vec_dim: int, dim: int, use_vec: bool = True):
        super().__init__()
        self.use_vec = use_vec
        self.embedding = nn.Embedding(num_actions, dim)
        in_dim = dim + (action_vec_dim if use_vec else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, action_id: torch.Tensor, action_vec: torch.Tensor | None = None) -> torch.Tensor:
        e = self.embedding(action_id)
        if self.use_vec:
            if action_vec is None:
                raise ValueError("action_vec required when use_vec=True")
            x = torch.cat([e, action_vec.float()], dim=-1)
        else:
            x = e
        return self.mlp(x)
