"""Patch-NEPA classifier that reuses PatchTransformerNepa end-to-end.

Unlike PatchCls adaptation, this keeps the PatchNEPA architecture intact for
pretrain -> finetune transfer and only adds a classification head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .patch_nepa import PatchTransformerNepa
from ..token.tokenizer import (
    TYPE_A_POINT,
    TYPE_A_RAY,
    TYPE_BOS,
    TYPE_EOS,
    TYPE_MISSING_RAY,
    TYPE_POINT,
    TYPE_Q_POINT,
    TYPE_Q_RAY,
    TYPE_RAY,
    TYPE_SEP,
)


class PatchTransformerNepaClassifier(PatchTransformerNepa):
    def __init__(
        self,
        num_classes: int,
        *,
        pooling: str = "cls_max",  # mean | cls | cls_max
        head_mode: str = "pointmae_mlp",  # linear | pointmae_mlp
        head_hidden_dim: int = 256,
        head_dropout: float = 0.5,
        is_causal: bool = False,
        **nepa_kwargs,
    ) -> None:
        super().__init__(**nepa_kwargs)
        assert pooling in {"mean", "cls", "cls_max"}
        assert head_mode in {"linear", "pointmae_mlp"}
        self.pooling = str(pooling)
        self.head_mode = str(head_mode)
        self.head_hidden_dim = int(head_hidden_dim)
        self.head_dropout = float(head_dropout)
        self.is_causal = bool(is_causal)

        if self.pooling == "cls_max":
            head_in_dim = 2 * int(self.d_model)
        else:
            head_in_dim = int(self.d_model)

        if self.head_mode == "linear":
            self.cls_head = nn.Linear(head_in_dim, int(num_classes))
        else:
            self.cls_head = nn.Sequential(
                nn.Linear(head_in_dim, self.head_hidden_dim),
                nn.BatchNorm1d(self.head_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.head_dropout),
                nn.Linear(self.head_hidden_dim, self.head_hidden_dim),
                nn.BatchNorm1d(self.head_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.head_dropout),
                nn.Linear(self.head_hidden_dim, int(num_classes)),
            )

    @staticmethod
    def _query_token_mask(type_id: torch.Tensor) -> torch.Tensor:
        # Primary query-token mask for both qa_tokens=1 and qa_tokens=0 cases.
        mask = (
            (type_id == int(TYPE_Q_POINT))
            | (type_id == int(TYPE_Q_RAY))
            | (type_id == int(TYPE_POINT))
            | (type_id == int(TYPE_RAY))
        )
        has_any = mask.any(dim=1, keepdim=True)
        # Fallback: remove specials and explicit answer-only/missing tokens.
        fallback = (
            (type_id != int(TYPE_BOS))
            & (type_id != int(TYPE_EOS))
            & (type_id != int(TYPE_SEP))
            & (type_id != int(TYPE_A_POINT))
            & (type_id != int(TYPE_A_RAY))
            & (type_id != int(TYPE_MISSING_RAY))
        )
        return torch.where(has_any, mask, fallback)

    def _pool_features(self, h: torch.Tensor, type_id: torch.Tensor) -> torch.Tensor:
        # BOS is always sequence index 0 in PatchNEPA builders.
        cls_feat = h[:, 0, :]
        q_mask = self._query_token_mask(type_id)

        if self.pooling == "cls":
            return cls_feat

        q_count = q_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=h.dtype)
        q_sum = (h * q_mask.unsqueeze(-1).to(dtype=h.dtype)).sum(dim=1)
        q_mean = q_sum / q_count

        if self.pooling == "mean":
            return q_mean

        neg_inf = torch.finfo(h.dtype).min
        q_vals = h.masked_fill(~q_mask.unsqueeze(-1), neg_inf)
        q_max = q_vals.max(dim=1).values
        # If a row has no valid query token, fallback to cls token.
        valid = q_mask.any(dim=1, keepdim=True)
        q_max = torch.where(valid, q_max, cls_feat)
        return torch.cat([cls_feat, q_max], dim=-1)

    def forward_features(
        self,
        xyz: torch.Tensor,
        normals: torch.Tensor | None = None,
        ray_o: torch.Tensor | None = None,
        ray_d: torch.Tensor | None = None,
        ray_t: torch.Tensor | None = None,
        ray_hit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Query-only classification protocol:
        # - no point-distance supervision at finetune (pt_dist=0)
        # - no ray-answer values (ray_t/ray_hit are intentionally ignored)
        _ = ray_t
        _ = ray_hit
        pt_dist = torch.zeros((xyz.shape[0], xyz.shape[1], 1), dtype=xyz.dtype, device=xyz.device)
        out = super().forward(
            pt_xyz=xyz,
            pt_n=normals,
            pt_dist=pt_dist,
            ray_o=ray_o,
            ray_d=ray_d,
            ray_t=None,
            ray_hit=None,
            is_causal=self.is_causal,
            dual_mask_near=0.0,
            dual_mask_far=0.0,
            dual_mask_window=0,
            dual_mask_type_aware=0,
        )
        return self._pool_features(out.h, out.type_id)

    def forward(
        self,
        xyz: torch.Tensor,
        normals: torch.Tensor | None = None,
        ray_o: torch.Tensor | None = None,
        ray_d: torch.Tensor | None = None,
        ray_t: torch.Tensor | None = None,
        ray_hit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feat = self.forward_features(
            xyz,
            normals,
            ray_o=ray_o,
            ray_d=ray_d,
            ray_t=ray_t,
            ray_hit=ray_hit,
        )
        return self.cls_head(feat)
