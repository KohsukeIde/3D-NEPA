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
from nepa3d.models.serial_patch_embed import SerialPatchEmbed


class PatchTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        # patch embed
        patch_embed: str = "fps_knn",  # fps_knn | serial
        num_groups: int = 64,
        group_size: int = 32,
        use_normals: bool = False,
        center_mode: str = "fps",
        serial_order: str = "morton",  # morton | morton_trans | z | z-trans | random | identity
        serial_bits: int = 10,
        serial_shuffle_within_patch: int = 0,
        # transformer
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        backbone_mode: str = "nepa2d",  # nepa2d | vanilla
        qk_norm: bool = True,
        qk_norm_affine: bool = False,
        qk_norm_bias: bool = False,
        layerscale_value: float = 1e-5,
        rope_theta: float = 100.0,
        rope_prefix_tokens: int = 1,
        use_gated_mlp: bool = False,
        hidden_act: str = "gelu",
        # pooling
        pooling: str = "cls_max",  # mean | cls | cls_max
        # positional encoding
        pos_mode: str = "center_mlp",  # learned | center_mlp
        # optional ray-patch fusion (Option-A style)
        use_ray_patch: bool = False,
        ray_pool_mode: str = "max",  # max | mean
        ray_fuse_mode: str = "concat",  # concat | add
        ray_hidden_dim: int = 128,
        ray_miss_t: float = 4.0,
        ray_hit_threshold: float = 0.5,
        # classifier head
        head_mode: str = "auto",  # auto | linear | pointmae_mlp
        head_hidden_dim: int = 256,
        head_dropout: float = 0.5,
        init_mode: str = "default",  # default | pointmae
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        assert pooling in {"mean", "cls", "cls_max"}
        assert pos_mode in {"learned", "center_mlp"}
        assert ray_pool_mode in {"max", "mean"}
        assert ray_fuse_mode in {"concat", "add"}
        assert head_mode in {"auto", "linear", "pointmae_mlp"}
        assert init_mode in {"default", "pointmae"}
        assert backbone_mode in {"nepa2d", "vanilla"}
        assert patch_embed in {"fps_knn", "serial"}
        self.pooling = pooling
        self.pos_mode = pos_mode
        self.head_mode = head_mode
        self.init_mode = init_mode
        self.backbone_mode = backbone_mode
        self.patch_embed_mode = patch_embed
        self.head_hidden_dim = int(head_hidden_dim)
        self.head_dropout = float(head_dropout)
        self.is_causal = bool(is_causal)
        self.d_model = int(d_model)
        self.use_ray_patch = bool(use_ray_patch)
        self.ray_pool_mode = str(ray_pool_mode)
        self.ray_fuse_mode = str(ray_fuse_mode)
        self.ray_miss_t = float(ray_miss_t)
        self.ray_hit_threshold = float(ray_hit_threshold)

        if patch_embed == "fps_knn":
            self.patch_embed = PointPatchEmbed(
                num_groups=num_groups,
                group_size=group_size,
                embed_dim=d_model,
                use_normals=use_normals,
                center_mode=center_mode,
            )
        else:
            self.patch_embed = SerialPatchEmbed(
                embed_dim=d_model,
                group_size=group_size,
                order=serial_order,
                bits=serial_bits,
                shuffle_within_patch=bool(serial_shuffle_within_patch),
                use_normals=use_normals,
            )

        self.use_cls = pooling in {"cls", "cls_max"}
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        # Positional branch is only used on vanilla backbone.
        # On nepa2d path, positional handling is internal (RoPE), so we avoid
        # registering unused pos parameters to keep DDP stable.
        if self.backbone_mode == "vanilla":
            if self.pos_mode == "learned":
                max_len = num_groups + (1 if self.use_cls else 0)
                self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
                self.pos_mlp = None
                self.cls_pos = None
                nn.init.trunc_normal_(self.pos_emb, std=0.02)
            else:
                self.pos_emb = None
                self.pos_mlp = nn.Sequential(
                    nn.Linear(3, 128),
                    nn.GELU(),
                    nn.Linear(128, d_model),
                )
                self.cls_pos = nn.Parameter(torch.zeros(1, 1, d_model)) if self.use_cls else None
        else:
            self.pos_emb = None
            self.pos_mlp = None
            self.cls_pos = None

        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.cls_pos is not None:
            nn.init.trunc_normal_(self.cls_pos, std=0.02)

        self.backbone = CausalTransformer(
            d_model=d_model,
            nhead=n_heads,
            num_layers=n_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path=drop_path_rate,
            qk_norm=bool(qk_norm),
            qk_norm_affine=bool(qk_norm_affine),
            qk_norm_bias=bool(qk_norm_bias),
            layerscale_value=float(layerscale_value),
            rope_theta=float(rope_theta),
            rope_prefix_tokens=int(rope_prefix_tokens),
            use_gated_mlp=bool(use_gated_mlp),
            hidden_act=str(hidden_act),
            # vanilla -> Point-MAE-like TransformerEncoder path (no RoPE/QK-Norm/LayerScale)
            backbone_impl="legacy" if self.backbone_mode == "vanilla" else "nepa2d",
        )
        self.norm = nn.LayerNorm(d_model)

        if self.use_ray_patch:
            # Per-ray encoder over local geometric relation to assigned point patch center.
            # Query-only input (no answer-side ray targets):
            # [x_proxy-center (3), ray_d (3)] = 6 dims.
            self.ray_encoder = nn.Sequential(
                nn.Linear(6, int(ray_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(ray_hidden_dim), d_model),
            )
            if self.ray_fuse_mode == "concat":
                self.ray_fuse_proj = nn.Linear(2 * d_model, d_model)
            else:
                self.ray_fuse_proj = None
        else:
            self.ray_encoder = None
            self.ray_fuse_proj = None

        if self.pooling == "cls_max":
            self._head_in_dim = 2 * d_model
        else:
            self._head_in_dim = d_model
        self.head = self._build_head(num_classes)

        if self.init_mode == "pointmae":
            self._init_pointmae_weights()
            # Keep token/pos params aligned with Point-MAE convention.
            if self.cls_token is not None:
                nn.init.trunc_normal_(self.cls_token, std=0.02)
            if self.cls_pos is not None:
                nn.init.trunc_normal_(self.cls_pos, std=0.02)
            if self.pos_emb is not None:
                nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def _init_pointmae_weights(self) -> None:
        # Align with Point-MAE scratch init policy for fair baseline comparison.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def reset_head(self, num_classes: int) -> None:
        self.head = self._build_head(num_classes)

    def _resolved_head_mode(self) -> str:
        if self.head_mode != "auto":
            return self.head_mode
        # Point-MAE-style cls+max representation benefits from MLP head.
        if self.pooling == "cls_max":
            return "pointmae_mlp"
        return "linear"

    def _build_head(self, num_classes: int) -> nn.Module:
        mode = self._resolved_head_mode()
        if mode == "linear":
            return nn.Linear(self._head_in_dim, num_classes)
        return nn.Sequential(
            nn.Linear(self._head_in_dim, self.head_hidden_dim),
            nn.BatchNorm1d(self.head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.head_dropout),
            nn.Linear(self.head_hidden_dim, self.head_hidden_dim),
            nn.BatchNorm1d(self.head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.head_dropout),
            nn.Linear(self.head_hidden_dim, num_classes),
        )

    def _resolved_pooling(self) -> str:
        return self.pooling

    def _build_ray_anchor(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
    ) -> torch.Tensor:
        # Query-only proxy anchor: do not use answer-side hit/t fields.
        return ray_o + float(self.ray_miss_t) * ray_d

    def _pool_rays_to_patches(
        self,
        centers_xyz: torch.Tensor,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
    ) -> torch.Tensor:
        """Pool encoded ray features into point patches by nearest center assignment."""
        B, G, _ = centers_xyz.shape
        D = int(self.d_model)
        pooled = centers_xyz.new_zeros((B, G, D))
        if ray_o is None or ray_o.numel() == 0:
            return pooled

        ray_anchor = self._build_ray_anchor(ray_o, ray_d)  # (B,R,3)
        for b in range(B):
            if ray_anchor[b].numel() == 0:
                continue
            # (R,G)
            dist = torch.cdist(ray_anchor[b].unsqueeze(0), centers_xyz[b].unsqueeze(0)).squeeze(0)
            assign = dist.argmin(dim=-1)  # (R,)

            ctr = centers_xyz[b][assign]  # (R,3)
            feat_in = torch.cat(
                [
                    ray_anchor[b] - ctr,
                    ray_d[b],
                ],
                dim=-1,
            )
            feat = self.ray_encoder(feat_in)  # (R,D)

            for g in range(G):
                m = assign == g
                if not bool(m.any()):
                    continue
                if self.ray_pool_mode == "mean":
                    pooled[b, g] = feat[m].mean(dim=0)
                else:
                    pooled[b, g] = feat[m].max(dim=0).values
        return pooled

    def forward_features(
        self,
        xyz: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        ray_o: Optional[torch.Tensor] = None,
        ray_d: Optional[torch.Tensor] = None,
        ray_t: Optional[torch.Tensor] = None,
        ray_hit: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return pooled features (B, D)."""
        out = self.patch_embed(xyz, normals)
        x = out.tokens  # (B,G,D)
        centers_xyz = out.centers_xyz  # (B,G,3)
        B, G, D = x.shape

        if self.use_ray_patch:
            if (ray_o is None) or (ray_d is None):
                raise ValueError("use_ray_patch=True requires ray_o/ray_d inputs")
            ray_tok = self._pool_rays_to_patches(centers_xyz, ray_o, ray_d)  # (B,G,D)
            if self.ray_fuse_mode == "add":
                x = x + ray_tok
            else:
                x = self.ray_fuse_proj(torch.cat([x, ray_tok], dim=-1))

        if self.use_cls:
            cls = self.cls_token.expand(B, 1, D)
            if self.is_causal:
                # Put CLS at the end so it can attend to all tokens under causal mask.
                x = torch.cat([x, cls], dim=1)
            else:
                # Standard: CLS at front.
                x = torch.cat([cls, x], dim=1)

        # Positional injection mode:
        # - vanilla: explicit pos tensor is added at each block input.
        # - nepa2d: positional handling is internal (RoPE in attention), so no x+pos add here.
        if self.backbone_mode == "vanilla":
            if self.pos_mode == "learned":
                # Learned positional embedding (slice to sequence length).
                pos = self.pos_emb[:, : x.size(1), :]
            else:
                # Center-MLP positional embedding (+ learned CLS pos if used).
                pos = self.pos_mlp(centers_xyz)  # (B,G,D)
                if self.use_cls:
                    cls_pos = self.cls_pos.expand(B, 1, D)
                    if self.is_causal:
                        pos = torch.cat([pos, cls_pos], dim=1)
                    else:
                        pos = torch.cat([cls_pos, pos], dim=1)
            x = self.backbone(x, is_causal=self.is_causal, pos=pos)
        else:
            x = self.backbone(x, is_causal=self.is_causal)
        x = self.norm(x)

        if self.pooling == "cls":
            feat = x[:, -1, :] if self.is_causal else x[:, 0, :]
        elif self.pooling == "cls_max":
            if self.is_causal:
                cls_feat = x[:, -1, :]
                tok = x[:, :-1, :]
            else:
                cls_feat = x[:, 0, :]
                tok = x[:, 1:, :]
            max_feat = tok.max(dim=1)[0]
            feat = torch.cat([cls_feat, max_feat], dim=-1)
        else:
            feat = x.mean(dim=1)
        return feat

    def forward(
        self,
        xyz: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        ray_o: Optional[torch.Tensor] = None,
        ray_d: Optional[torch.Tensor] = None,
        ray_t: Optional[torch.Tensor] = None,
        ray_hit: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = self.forward_features(xyz, normals, ray_o=ray_o, ray_d=ray_d, ray_t=ray_t, ray_hit=ray_hit)
        return self.head(feat)
