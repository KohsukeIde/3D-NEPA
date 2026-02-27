"""Serialization-based patch embedding (PTv3-style).

Patch grouping strategy:
- serialize points with Morton order (or random/identity),
- optionally use transposed Morton order (XY-swapped, PTv3 z-trans style),
- chunk contiguous points into fixed-size groups,
- mini-PointNet over each chunk.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from nepa3d.models.point_patch_embed import PatchEmbedOutput


def _part1by2_10bits(v: torch.Tensor) -> torch.Tensor:
    """Dilate 10 bits into 30 bits with two zeros between bits."""
    v = v & 0x000003FF
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v


def morton3d_codes(xyz: torch.Tensor, bits: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """Compute Morton (Z-order) codes for xyz (B,N,3)."""
    if bits != 10:
        raise ValueError("serial Morton currently supports only bits=10")

    mins = xyz.amin(dim=1, keepdim=True)
    maxs = xyz.amax(dim=1, keepdim=True)
    scale = (maxs - mins).clamp_min(eps)
    u = (xyz - mins) / scale

    qmax = (1 << bits) - 1
    q = torch.clamp((u * qmax).round(), 0, qmax).to(torch.int64)
    x = _part1by2_10bits(q[..., 0])
    y = _part1by2_10bits(q[..., 1])
    z = _part1by2_10bits(q[..., 2])
    return x | (y << 1) | (z << 2)


def serialize_indices(
    xyz: torch.Tensor,
    order: Literal["morton", "morton_trans", "z", "z-trans", "random", "identity"] = "morton",
    bits: int = 10,
) -> torch.Tensor:
    """Return serialized indices (B,N)."""
    b, n, _ = xyz.shape
    device = xyz.device
    if order in {"morton", "z"}:
        return torch.argsort(morton3d_codes(xyz, bits=bits), dim=1)
    if order in {"morton_trans", "z-trans"}:
        # PTv3-style "trans" variant: swap X/Y before serialization.
        xyz_t = xyz[..., [1, 0, 2]]
        return torch.argsort(morton3d_codes(xyz_t, bits=bits), dim=1)
    if order == "random":
        return torch.argsort(torch.rand((b, n), device=device), dim=1)
    if order == "identity":
        return torch.arange(n, device=device).view(1, n).expand(b, n)
    raise ValueError(f"unknown serial order: {order}")


class SerialPatchEmbed(nn.Module):
    """Patch embedding via serialization + contiguous chunking."""

    def __init__(
        self,
        embed_dim: int,
        group_size: int = 16,
        order: Literal["morton", "morton_trans", "z", "z-trans", "random", "identity"] = "morton",
        bits: int = 10,
        shuffle_within_patch: bool = False,
        use_normals: bool = False,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.group_size = int(group_size)
        self.order = order
        self.bits = int(bits)
        self.shuffle_within_patch = bool(shuffle_within_patch)
        self.use_normals = bool(use_normals)

        in_dim = 3 + (3 if self.use_normals else 0)
        hid = int(self.embed_dim * float(mlp_ratio))
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Linear(hid, self.embed_dim),
            nn.GELU(),
        )
        self.center_proj = nn.Linear(3, self.embed_dim)

    def forward(self, xyz: torch.Tensor, normals: torch.Tensor | None = None) -> PatchEmbedOutput:
        if xyz.ndim != 3 or xyz.size(-1) != 3:
            raise ValueError(f"xyz must be (B,N,3), got {tuple(xyz.shape)}")
        if self.use_normals:
            if normals is None:
                raise ValueError("use_normals=True but normals is None")
            if normals.shape != xyz.shape:
                raise ValueError(f"normals must match xyz, got {tuple(normals.shape)} vs {tuple(xyz.shape)}")

        b, n, _ = xyz.shape
        if n < self.group_size:
            raise ValueError(f"need n>=group_size, got n={n}, group_size={self.group_size}")

        idx = serialize_indices(xyz, order=self.order, bits=self.bits)
        idx3 = idx.unsqueeze(-1).expand(-1, -1, 3)
        xyz_s = torch.gather(xyz, 1, idx3)
        normals_s = torch.gather(normals, 1, idx3) if (self.use_normals and normals is not None) else None

        g = n // self.group_size
        n2 = g * self.group_size
        xyz_s = xyz_s[:, :n2]
        if normals_s is not None:
            normals_s = normals_s[:, :n2]

        xyz_g = xyz_s.view(b, g, self.group_size, 3)
        centers_xyz = xyz_g.mean(dim=2)  # (B,G,3)
        rel = xyz_g - centers_xyz.unsqueeze(2)

        feat = rel
        if self.use_normals and normals_s is not None:
            nrm_g = normals_s.view(b, g, self.group_size, 3)
            feat = torch.cat([feat, nrm_g], dim=-1)

        if self.shuffle_within_patch:
            perm = torch.argsort(torch.rand((b, g, self.group_size), device=xyz.device), dim=-1)
            pidx = perm.unsqueeze(-1).expand(-1, -1, -1, feat.size(-1))
            feat = torch.gather(feat, 2, pidx)

        x = self.point_mlp(feat).max(dim=2).values  # (B,G,D)
        tokens = x + self.center_proj(centers_xyz)
        group_idx = idx[:, :n2].view(b, g, self.group_size)
        return PatchEmbedOutput(tokens=tokens, centers_xyz=centers_xyz, group_idx=group_idx)
