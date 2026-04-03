"""PatchNEPA direct part-segmentation head for ShapeNetPart."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from nepa3d.core.models.encdec_transformer import EncoderDecoderTransformer
from nepa3d.token.tokenizer import TYPE_BOS, TYPE_EOS, TYPE_MISSING_RAY, TYPE_Q_POINT, TYPE_Q_RAY
from nepa3d.tracks.patch_nepa.mainline.models.patch_nepa import PatchTransformerNepa


class PatchTransformerNepaPartSeg(nn.Module):
    def __init__(
        self,
        *,
        num_parts: int = 50,
        num_shape_classes: int = 16,
        head_dropout: float = 0.5,
        label_dim: int = 64,
        ft_sequence_mode: str = "q_only",
        **nepa_kwargs,
    ) -> None:
        super().__init__()
        if str(ft_sequence_mode) != "q_only":
            raise ValueError("PatchTransformerNepaPartSeg currently supports only ft_sequence_mode='q_only'")
        self.core = PatchTransformerNepa(**nepa_kwargs)
        self.d_model = int(self.core.d_model)
        self.use_normals = bool(getattr(self.core, "use_normals", False))
        self.num_parts = int(num_parts)
        self.num_shape_classes = int(num_shape_classes)
        self.ft_sequence_mode = str(ft_sequence_mode)

        self.label_mlp = nn.Sequential(
            nn.Linear(self.num_shape_classes, int(label_dim)),
            nn.ReLU(inplace=True),
        )

        in_dim = int(self.d_model + self.d_model + label_dim + 3 + (3 if self.use_normals else 0))
        self.seg_head = nn.Sequential(
            nn.Conv1d(in_dim, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(float(head_dropout)),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(float(head_dropout)),
            nn.Conv1d(256, self.num_parts, 1),
        )

    def _build_q_only_sequence(
        self,
        xyz: torch.Tensor,
        normals: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_out = self.core.patch_embed(xyz, normals if self.use_normals else None)
        patch_out = self.core._maybe_reorder_patch_embed_output(patch_out)
        q_tok = patch_out.tokens
        centers_xyz = patch_out.centers_xyz
        group_idx = patch_out.group_idx
        bsz, n_patch, d_model = q_tok.shape
        dev = q_tok.device

        parts = [self.core.bos_token.expand(bsz, 1, d_model), q_tok, self.core.eos_token.expand(bsz, 1, d_model)]
        type_parts = [
            torch.full((bsz, 1), int(TYPE_BOS), device=dev, dtype=torch.long),
            torch.full((bsz, n_patch), int(TYPE_Q_POINT), device=dev, dtype=torch.long),
            torch.full((bsz, 1), int(TYPE_EOS), device=dev, dtype=torch.long),
        ]
        centers_parts = [
            torch.zeros((bsz, 1, 3), device=dev, dtype=centers_xyz.dtype),
            centers_xyz,
            torch.zeros((bsz, 1, 3), device=dev, dtype=centers_xyz.dtype),
        ]

        if bool(getattr(self.core, "use_ray_patch", False)):
            raise ValueError("ray-patch part-seg is not implemented in the current local package")

        tokens = torch.cat(parts, dim=1)
        type_id = torch.cat(type_parts, dim=1)
        centers_seq = torch.cat(centers_parts, dim=1)
        return tokens, type_id, centers_seq, centers_xyz, group_idx

    def forward_patch_features(
        self,
        xyz: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, type_id, centers_seq, centers_xyz, _group_idx = self._build_q_only_sequence(xyz, normals)
        backbone_in, backbone_pos, _z = self.core._prepare_backbone_inputs(tokens, type_id, centers_seq)
        if isinstance(self.core.backbone, EncoderDecoderTransformer):
            enc = backbone_in[:, :-1, :]
            dec = backbone_in[:, -1:, :]
            enc_out, _ = self.core.backbone(enc, dec, enc_xyz=None)
            h = enc_out
            patch_feat = h[:, : centers_xyz.size(1), :]
        else:
            h = self.core.backbone(
                backbone_in,
                is_causal=False,
                type_id=type_id,
                pos=backbone_pos,
                dual_mask_near=0.0,
                dual_mask_far=0.0,
                dual_mask_window=0,
                dual_mask_type_aware=0,
            )
            h = self.core.pred_head[0](h)
            patch_feat = h[:, 1 : 1 + centers_xyz.size(1), :]
        return patch_feat, centers_xyz

    @staticmethod
    def _nearest_patch_unpool(
        xyz: torch.Tensor,
        centers_xyz: torch.Tensor,
        patch_feat: torch.Tensor,
    ) -> torch.Tensor:
        dist = torch.cdist(xyz, centers_xyz)
        assign = dist.argmin(dim=-1)
        feat_idx = assign.unsqueeze(-1).expand(-1, -1, patch_feat.size(-1))
        return torch.gather(patch_feat, 1, feat_idx)

    def forward(
        self,
        xyz: torch.Tensor,
        cls_label: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        patch_feat, centers_xyz = self.forward_patch_features(xyz, normals)
        point_feat = self._nearest_patch_unpool(xyz, centers_xyz, patch_feat)
        global_feat = patch_feat.max(dim=1, keepdim=True).values.expand(-1, xyz.size(1), -1)

        cls_onehot = torch.zeros(
            (xyz.size(0), self.num_shape_classes),
            dtype=xyz.dtype,
            device=xyz.device,
        )
        cls_onehot.scatter_(1, cls_label.view(-1, 1), 1.0)
        cls_feat = self.label_mlp(cls_onehot).unsqueeze(1).expand(-1, xyz.size(1), -1)

        pieces = [point_feat, global_feat, cls_feat, xyz]
        if self.use_normals and normals is not None:
            pieces.append(normals)
        feat = torch.cat(pieces, dim=-1).transpose(1, 2).contiguous()
        return self.seg_head(feat).transpose(1, 2).contiguous()
