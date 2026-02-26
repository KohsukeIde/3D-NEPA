"""Patchified Transformer classifier.

This is intended as the *baseline* for answering:

- Can a patch-token transformer trained from scratch reach the typical
  ScanObjectNN / ModelNet40 baselines (~Point-MAE "Transformer" lines)?

We keep it minimal:
- PointPatchEmbed (FPS + kNN + mini-PointNet)
- (Bi)directional transformer backbone (we reuse CausalTransformer with is_causal toggle)
- mean pooling (default) or CLS token pooling

No decoder, no masked reconstruction objective, no NEPA objective here.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from nepa3d.models.causal_transformer import CausalTransformer
from nepa3d.models.point_patch_embed import PointPatchEmbed


class PatchTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        # patch embed
        num_groups: int = 64,
        group_size: int = 32,
        use_normals: bool = False,
        center_mode: str = "fps",
        # transformer
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        # pooling
        pooling: str = "cls",  # mean | cls
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        assert pooling in {"mean", "cls"}
        self.pooling = pooling
        self.is_causal = bool(is_causal)

        self.patch_embed = PointPatchEmbed(
            num_groups=num_groups,
            group_size=group_size,
            embed_dim=d_model,
            use_normals=use_normals,
            center_mode=center_mode,
        )

        self.use_cls = pooling == "cls"
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        # positional embedding for patch tokens (+ optional CLS)
        max_len = num_groups + (1 if self.use_cls else 0)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.backbone = CausalTransformer(
            d_model=d_model,
            nhead=n_heads,
            num_layers=n_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path=drop_path_rate,
            backbone_impl="nepa2d",
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def reset_head(self, num_classes: int) -> None:
        self.head = nn.Linear(self.head.in_features, num_classes)

    def _resolved_pooling(self) -> str:
        return self.pooling

    def forward_features(self, xyz: torch.Tensor, normals: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return pooled features (B, D)."""
        out = self.patch_embed(xyz, normals)
        x = out.tokens  # (B,G,D)
        B, G, D = x.shape

        if self.use_cls:
            cls = self.cls_token.expand(B, 1, D)
            if self.is_causal:
                # Put CLS at the end so it can attend to all tokens under causal mask.
                x = torch.cat([x, cls], dim=1)
            else:
                # Standard: CLS at front.
                x = torch.cat([cls, x], dim=1)

        # Add positional embedding (slice to length)
        x = x + self.pos_emb[:, : x.size(1), :]
        x = self.backbone(x, is_causal=self.is_causal)
        x = self.norm(x)

        if self.use_cls:
            feat = x[:, -1, :] if self.is_causal else x[:, 0, :]
        else:
            feat = x.mean(dim=1)
        return feat

    def forward(self, xyz: torch.Tensor, normals: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.forward_features(xyz, normals)
        return self.head(feat)
