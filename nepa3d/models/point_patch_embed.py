"""Point patch embedding (FPS + kNN + mini-PointNet).

This is intentionally minimal and self-contained so we can use it for:

- classification scratch baseline ("Transformer (rand)" style)
- later NEPA pretrain on patch tokens (without changing the objective)

Design goals:
- Works on torch tensors (GPU-friendly).
- No external CUDA ops required.
- Deterministic given inputs (unless user randomizes sampling upstream).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Index points.

    Args:
        points: (B, N, C)
        idx: (B, S) or (B, S, K)
    Returns:
        new_points: (B, S, C) or (B, S, K, C)
    """
    assert points.dim() == 3, f"points must be (B,N,C), got {points.shape}"
    B, N, C = points.shape
    if idx.dim() == 2:
        # (B,S)
        batch = torch.arange(B, device=points.device).unsqueeze(1)
        return points[batch, idx, :]
    if idx.dim() == 3:
        # (B,S,K)
        batch = torch.arange(B, device=points.device).view(B, 1, 1)
        return points[batch, idx, :]
    raise ValueError(f"idx must be (B,S) or (B,S,K), got {idx.shape}")


def farthest_point_sample(xyz: torch.Tensor, npoint: int, random_start: bool = False) -> torch.Tensor:
    """Farthest point sampling (batched).

    Args:
        xyz: (B, N, 3)
        npoint: number of samples
    Returns:
        centroids_idx: (B, npoint)

    Notes:
        - O(B*N*npoint) iterative FPS. For typical settings (N=1024, npoint=64)
          this is usually acceptable.
    """
    assert xyz.dim() == 3 and xyz.size(-1) == 3, f"xyz must be (B,N,3), got {xyz.shape}"
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)

    if bool(random_start):
        # Point-MAE-style randomized FPS seed.
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    else:
        # Deterministic seed: farthest-from-mean.
        mean_xyz = xyz.mean(dim=1, keepdim=True)  # (B,1,3)
        dist0 = ((xyz - mean_xyz) ** 2).sum(dim=-1)  # (B,N)
        farthest = dist0.max(dim=1)[1]  # (B,)

    batch_indices = torch.arange(B, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = distance.max(dim=1)[1]
    return centroids


def knn_indices(xyz: torch.Tensor, centers: torch.Tensor, k: int) -> torch.Tensor:
    """kNN indices for each center.

    Args:
        xyz: (B, N, 3)
        centers: (B, S, 3)
        k: number of neighbors
    Returns:
        idx: (B, S, k)
    """
    assert xyz.dim() == 3 and centers.dim() == 3
    B, N, _ = xyz.shape
    _, S, _ = centers.shape
    # (B,S,N)
    dist2 = ((centers.unsqueeze(2) - xyz.unsqueeze(1)) ** 2).sum(dim=-1)
    # smallest k
    idx = dist2.topk(k=k, dim=-1, largest=False)[1]
    return idx


class MiniPointNet(nn.Module):
    """A tiny PointNet-like encoder for a local group."""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers = []
        d = in_dim
        for i in range(max(1, n_layers - 1)):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, embed_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a group.

        Args:
            x: (B, S, K, C)
        Returns:
            feat: (B, S, D)
        """
        assert x.dim() == 4, f"x must be (B,S,K,C), got {x.shape}"
        B, S, K, C = x.shape
        y = self.mlp(x.reshape(B * S * K, C)).reshape(B, S, K, -1)
        # max-pool over K
        y = y.max(dim=2)[0]
        return y


class PointMAELocalEncoder(nn.Module):
    """Point-MAE style local encoder (Conv1d + BN + max pooling)."""

    def __init__(self, in_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(int(in_dim), 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, int(embed_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,S,K,C)
        assert x.dim() == 4, f"x must be (B,S,K,C), got {x.shape}"
        B, S, K, C = x.shape
        feat = x.reshape(B * S, K, C).transpose(1, 2).contiguous()  # (BS,C,K)
        feat = self.first_conv(feat)  # (BS,256,K)
        feat_global = feat.max(dim=2, keepdim=True)[0]  # (BS,256,1)
        feat = torch.cat([feat_global.expand(-1, -1, K), feat], dim=1)  # (BS,512,K)
        feat = self.second_conv(feat)  # (BS,D,K)
        feat = feat.max(dim=2, keepdim=False)[0]  # (BS,D)
        return feat.view(B, S, -1)


@dataclass
class PatchEmbedOutput:
    tokens: torch.Tensor  # (B, G, D)
    centers_xyz: torch.Tensor  # (B, G, 3)
    group_idx: torch.Tensor  # (B, G, K)


class PointPatchEmbed(nn.Module):
    """FPS + kNN grouping + MiniPointNet -> patch tokens."""

    def __init__(
        self,
        num_groups: int = 64,
        group_size: int = 32,
        embed_dim: int = 384,
        use_normals: bool = False,
        center_mode: str = "fps",  # fps | first
        fps_random_start: bool = False,
        local_encoder: str = "mlp",  # mlp | pointmae_conv
        mlp_hidden_dim: int = 128,
        mlp_layers: int = 2,
        mlp_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert center_mode in {"fps", "first"}
        assert str(local_encoder) in {"mlp", "pointmae_conv"}
        self.num_groups = int(num_groups)
        self.group_size = int(group_size)
        self.embed_dim = int(embed_dim)
        self.use_normals = bool(use_normals)
        self.center_mode = center_mode
        self.fps_random_start = bool(fps_random_start)
        self.local_encoder = str(local_encoder)

        in_dim = 3  # relative xyz
        if self.use_normals:
            in_dim += 3
        if self.local_encoder == "pointmae_conv":
            self.group_encoder = PointMAELocalEncoder(
                in_dim=in_dim,
                embed_dim=self.embed_dim,
            )
        else:
            self.group_encoder = MiniPointNet(
                in_dim=in_dim,
                embed_dim=self.embed_dim,
                hidden_dim=mlp_hidden_dim,
                n_layers=mlp_layers,
                dropout=mlp_dropout,
            )

    @torch.no_grad()
    def _select_centers(self, xyz: torch.Tensor) -> torch.Tensor:
        """Return center indices (B, G)."""
        B, N, _ = xyz.shape
        G = min(self.num_groups, N)
        if self.center_mode == "first":
            idx = torch.arange(G, device=xyz.device).view(1, G).repeat(B, 1)
            return idx
        # fps
        return farthest_point_sample(xyz, G, random_start=self.fps_random_start)

    def forward(
        self, xyz: torch.Tensor, normals: Optional[torch.Tensor] = None
    ) -> PatchEmbedOutput:
        """Compute patch tokens.

        Args:
            xyz: (B, N, 3)
            normals: (B, N, 3) or None
        Returns:
            PatchEmbedOutput
        """
        assert xyz.dim() == 3 and xyz.size(-1) == 3, f"xyz must be (B,N,3), got {xyz.shape}"
        if self.use_normals:
            if normals is None:
                raise ValueError("use_normals=True but normals is None")
            assert normals.shape[:2] == xyz.shape[:2] and normals.size(-1) == 3

        center_idx = self._select_centers(xyz)  # (B,G)
        centers_xyz = _index_points(xyz, center_idx)  # (B,G,3)
        group_idx = knn_indices(xyz, centers_xyz, k=min(self.group_size, xyz.size(1)))  # (B,G,K)
        group_xyz = _index_points(xyz, group_idx)  # (B,G,K,3)
        rel_xyz = group_xyz - centers_xyz.unsqueeze(2)  # (B,G,K,3)

        if self.use_normals:
            group_n = _index_points(normals, group_idx)  # (B,G,K,3)
            group_feat = torch.cat([rel_xyz, group_n], dim=-1)
        else:
            group_feat = rel_xyz

        tokens = self.group_encoder(group_feat)  # (B,G,D)
        return PatchEmbedOutput(tokens=tokens, centers_xyz=centers_xyz, group_idx=group_idx)
