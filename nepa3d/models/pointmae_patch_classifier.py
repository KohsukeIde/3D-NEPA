from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

try:
    from pointnet2_ops import pointnet2_utils
except Exception:
    pointnet2_utils = None

try:
    from knn_cuda import KNN as _KNN
except Exception:
    _KNN = None


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, mean=0.0, std=float(std), a=-2 * float(std), b=2 * float(std))


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def fps(points: torch.Tensor, number: int) -> torch.Tensor:
    """Point-MAE-like FPS helper: returns sampled points, not indices."""
    if number >= points.size(1):
        return points
    if pointnet2_utils is not None:
        fps_idx = pointnet2_utils.furthest_point_sample(points, int(number))
        fps_points = pointnet2_utils.gather_operation(
            points.transpose(1, 2).contiguous(), fps_idx
        ).transpose(1, 2).contiguous()
        return fps_points

    # CUDA extension fallback.
    bsz, n_pts, _ = points.shape
    centroids = torch.zeros((bsz, int(number)), dtype=torch.long, device=points.device)
    distance = torch.full((bsz, n_pts), float("inf"), device=points.device)
    farthest = torch.randint(0, n_pts, (bsz,), dtype=torch.long, device=points.device)
    batch_indices = torch.arange(bsz, dtype=torch.long, device=points.device)
    for i in range(int(number)):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(bsz, 1, 3)
        dist = torch.sum((points - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return points[batch_indices[:, None], centroids, :].contiguous()


class Group(nn.Module):
    """Point-MAE Group: FPS centers + KNN neighborhoods."""

    def __init__(self, num_group: int, group_size: int):
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)
        self.knn = _KNN(k=self.group_size, transpose_mode=True) if _KNN is not None else None

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_points, _ = xyz.shape
        center = fps(xyz, self.num_group)  # (B, G, 3)
        if self.knn is not None:
            _, idx = self.knn(xyz, center)  # (B, G, K)
        else:
            dist = torch.cdist(center, xyz)
            idx = dist.topk(k=self.group_size, dim=-1, largest=False).indices

        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size

        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Encoder(nn.Module):
    """Point-MAE local patch encoder (Conv1d+BN)."""

    def __init__(self, encoder_channel: int):
        super().__init__()
        self.encoder_channel = int(encoder_channel)
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        # point_groups: (B, G, K, 3)
        bs, g, k, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, k, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))  # (BG, 256, K)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (BG, 256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, k), feature], dim=1)  # (BG, 512, K)
        feature = self.second_conv(feature)  # (BG, C, K)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # (BG, C)
        return feature_global.reshape(bs, g, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(float(drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        head_dim = int(dim) // self.num_heads
        self.scale = float(qk_scale) if qk_scale is not None else head_dim ** -0.5
        self.qkv = nn.Linear(int(dim), int(dim) * 3, bias=bool(qkv_bias))
        self.attn_drop = nn.Dropout(float(attn_drop))
        self.proj = nn.Linear(int(dim), int(dim))
        self.proj_drop = nn.Dropout(float(proj_drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(int(dim))
        self.drop_path = DropPath(float(drop_path)) if float(drop_path) > 0.0 else nn.Identity()
        self.norm2 = norm_layer(int(dim))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=int(dim),
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=float(drop),
        )
        self.attn = Attention(
            int(dim),
            num_heads=int(num_heads),
            qkv_bias=bool(qkv_bias),
            qk_scale=qk_scale,
            attn_drop=float(attn_drop),
            proj_drop=float(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float | list[float] = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=int(embed_dim),
                    num_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    qkv_bias=bool(qkv_bias),
                    qk_scale=qk_scale,
                    drop=float(drop_rate),
                    attn_drop=float(attn_drop_rate),
                    drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else float(drop_path_rate),
                )
                for i in range(int(depth))
            ]
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x + pos)
        return x


class PointMAEPatchClassifier(nn.Module):
    """Point-MAE PointTransformer classifier backbone as a selectable patchcls mode."""

    def __init__(
        self,
        num_classes: int,
        *,
        trans_dim: int = 384,
        depth: int = 12,
        drop_path_rate: float = 0.1,
        num_heads: int = 6,
        group_size: int = 32,
        num_groups: int = 64,
        encoder_dims: int = 384,
        init_mode: str = "pointmae",  # default | pointmae
    ) -> None:
        super().__init__()
        assert init_mode in {"default", "pointmae"}
        self.init_mode = str(init_mode)

        self.trans_dim = int(trans_dim)
        self.depth = int(depth)
        self.drop_path_rate = float(drop_path_rate)
        self.cls_dim = int(num_classes)
        self.num_heads = int(num_heads)
        self.group_size = int(group_size)
        self.num_group = int(num_groups)
        self.encoder_dims = int(encoder_dims)

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            qkv_bias=False,  # Point-MAE default
            drop_rate=0.0,
            attn_drop_rate=0.0,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim),
        )

        _trunc_normal_(self.cls_token, std=0.02)
        _trunc_normal_(self.cls_pos, std=0.02)

        if self.init_mode == "pointmae":
            self.apply(self._init_weights)
            _trunc_normal_(self.cls_token, std=0.02)
            _trunc_normal_(self.cls_pos, std=0.02)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts: torch.Tensor, normals: Optional[torch.Tensor] = None) -> torch.Tensor:
        # normals are intentionally ignored in Point-MAE classification path.
        del normals
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # (B, G, C)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        return self.cls_head_finetune(concat_f)

