from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1.0 - (pred * target).sum(dim=-1).mean()


def variance_regularizer(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    # VICReg-style variance floor. Use only if collapse appears.
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


def batch_stats(z: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        return {
            "z_norm_mean": float(z.norm(dim=-1).mean().detach().cpu()),
            "z_norm_std": float(z.norm(dim=-1).std().detach().cpu()),
            "z_var_mean": float(z.var(dim=0).mean().detach().cpu()),
        }
