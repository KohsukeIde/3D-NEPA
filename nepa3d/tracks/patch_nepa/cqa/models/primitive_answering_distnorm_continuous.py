from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from nepa3d.core.models.causal_transformer import CausalTransformer
from nepa3d.core.models.point_patch_embed import PointPatchEmbed
from nepa3d.tracks.patch_nepa.cqa.data.cqa_codec import ASK_DISTANCE, ASK_NORMAL, ASK_VISIBILITY


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = int(in_dim)
        for _ in range(max(1, int(n_layers) - 1)):
            layers.append(nn.Linear(d, int(hidden_dim)))
            layers.append(nn.GELU())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            d = int(hidden_dim)
        layers.append(nn.Linear(d, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DistNormContinuousOutput:
    pred_answer: torch.Tensor
    raw_answer: torch.Tensor
    hidden: torch.Tensor
    ctx_tokens: torch.Tensor
    ctx_centers: torch.Tensor
    query_tokens: torch.Tensor
    answer_hidden: torch.Tensor
    sequence: torch.Tensor
    attn_mask: torch.Tensor


class PrimitiveAnsweringDistNormContinuousModel(nn.Module):
    """Shared continuous typed-answer model for scalar/norm CQA tasks.

    This keeps the current strongest CQA skeleton:
      - shared context encoder
      - typed query tokens in the prompt
      - independent answer slots
    but replaces the discrete answer vocab with a shared 3D regression head.

    Current typed decoding:
      - `ASK_DISTANCE`   -> positive scalar via `softplus`
      - `ASK_VISIBILITY` -> `[0,1]` scalar via `sigmoid` (used by `mesh_ao`)
      - `ASK_NORMAL`     -> normalized 3D vector
    """

    def __init__(
        self,
        *,
        d_model: int = 384,
        n_layers: int = 12,
        n_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        backbone_impl: str = "nepa2d",
        num_groups: int = 64,
        group_size: int = 32,
        patch_center_mode: str = "fps",
        patch_fps_random_start: bool = True,
        local_encoder: str = "pointmae_conv",
        query_type_vocab: int = 6,
        generator_depth: int = 2,
        distance_floor: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.query_type_vocab = int(query_type_vocab)
        self.distance_floor = float(distance_floor)

        self.ctx_patch = PointPatchEmbed(
            num_groups=int(num_groups),
            group_size=int(group_size),
            embed_dim=int(d_model),
            use_normals=False,
            center_mode=str(patch_center_mode),
            fps_random_start=bool(patch_fps_random_start),
            local_encoder=str(local_encoder),
        )
        self.center_pos = _MLP(3, int(d_model), hidden_dim=int(d_model), n_layers=2, dropout=float(dropout))
        self.query_embed = _MLP(3, int(d_model), hidden_dim=int(d_model), n_layers=2, dropout=float(dropout))
        self.query_type_embed = nn.Embedding(int(query_type_vocab), int(d_model))

        self.bos = nn.Parameter(torch.randn(1, 1, int(d_model)) * 0.02)
        self.sep_cq = nn.Parameter(torch.randn(1, 1, int(d_model)) * 0.02)
        self.sep_a = nn.Parameter(torch.randn(1, 1, int(d_model)) * 0.02)
        self.ans_bos = nn.Parameter(torch.randn(1, 1, int(d_model)) * 0.02)

        self.backbone = CausalTransformer(
            d_model=int(d_model),
            nhead=int(n_heads),
            num_layers=int(n_layers),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
            drop_path=float(drop_path),
            backbone_impl=str(backbone_impl),
        )
        self.generator = None
        if int(generator_depth) > 0:
            self.generator = CausalTransformer(
                d_model=int(d_model),
                nhead=int(n_heads),
                num_layers=int(generator_depth),
                mlp_ratio=float(mlp_ratio),
                dropout=float(dropout),
                drop_path=0.0,
                backbone_impl=str(backbone_impl),
            )
        self.answer_head = _MLP(int(d_model), 3, hidden_dim=int(d_model), n_layers=2, dropout=float(dropout))

    @staticmethod
    def _build_prompt_answer_mask(prompt_len: int, n_answer: int, device: torch.device) -> torch.Tensor:
        total = int(prompt_len) + int(n_answer)
        mask = torch.zeros((total, total), device=device, dtype=torch.bool)
        if int(n_answer) > 0:
            offdiag = torch.ones((n_answer, n_answer), device=device, dtype=torch.bool)
            offdiag.fill_diagonal_(False)
            mask[int(prompt_len):, int(prompt_len):] = offdiag
            mask[:int(prompt_len), int(prompt_len):] = True
        return mask

    @staticmethod
    def _decode_typed(raw_answer: torch.Tensor, qry_type: torch.Tensor, distance_floor: float) -> torch.Tensor:
        pred = torch.zeros_like(raw_answer)
        dist_mask = qry_type == int(ASK_DISTANCE)
        vis_mask = qry_type == int(ASK_VISIBILITY)
        norm_mask = qry_type == int(ASK_NORMAL)
        if bool(dist_mask.any()):
            pred[..., 0] = torch.where(
                dist_mask,
                F.softplus(raw_answer[..., 0]) + float(distance_floor),
                pred[..., 0],
            )
        if bool(vis_mask.any()):
            pred[..., 0] = torch.where(
                vis_mask,
                torch.sigmoid(raw_answer[..., 0]),
                pred[..., 0],
            )
        if bool(norm_mask.any()):
            norm_pred = F.normalize(raw_answer, dim=-1, eps=1e-8)
            pred = torch.where(norm_mask.unsqueeze(-1), norm_pred, pred)
        return pred

    def encode_context(self, ctx_xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patch_out = self.ctx_patch(ctx_xyz)
        tok = patch_out.tokens + self.center_pos(patch_out.centers_xyz)
        return tok, patch_out.centers_xyz

    def encode_queries(self, qry_xyz: torch.Tensor, qry_type: torch.Tensor) -> torch.Tensor:
        q = self.query_embed(qry_xyz)
        if qry_type.dim() == 1:
            qry_type = qry_type.unsqueeze(0).expand(qry_xyz.shape[0], -1)
        return q + self.query_type_embed(qry_type)

    def forward(self, ctx_xyz: torch.Tensor, qry_xyz: torch.Tensor, qry_type: torch.Tensor) -> DistNormContinuousOutput:
        b = int(ctx_xyz.shape[0])
        ctx_tok, ctx_centers = self.encode_context(ctx_xyz)
        q_tok = self.encode_queries(qry_xyz, qry_type)
        ans_in = self.ans_bos.expand(b, int(qry_xyz.shape[1]), -1) + self.query_embed(qry_xyz)

        if qry_type.dim() == 2:
            type_scalar = qry_type[:, 0]
        else:
            type_scalar = qry_type
        type_tok = self.query_type_embed(type_scalar).unsqueeze(1)
        seq = torch.cat(
            [
                self.bos.expand(b, 1, -1),
                ctx_tok,
                self.sep_cq.expand(b, 1, -1),
                type_tok,
                q_tok,
                self.sep_a.expand(b, 1, -1),
                ans_in,
            ],
            dim=1,
        )
        prompt_len = 1 + ctx_tok.shape[1] + 1 + 1 + q_tok.shape[1] + 1
        mask = self._build_prompt_answer_mask(prompt_len, ans_in.shape[1], seq.device)
        h = self.backbone(seq, is_causal=False, attn_mask_override=mask)
        h_ans = h[:, prompt_len:, :]
        if self.generator is not None:
            gen_mask = torch.ones((h_ans.shape[1], h_ans.shape[1]), device=h_ans.device, dtype=torch.bool)
            gen_mask.fill_diagonal_(False)
            h_ans = self.generator(h_ans, is_causal=False, attn_mask_override=gen_mask)
        raw_answer = self.answer_head(h_ans)
        pred_answer = self._decode_typed(raw_answer, qry_type, float(self.distance_floor))
        return DistNormContinuousOutput(
            pred_answer=pred_answer,
            raw_answer=raw_answer,
            hidden=h,
            ctx_tokens=ctx_tok,
            ctx_centers=ctx_centers,
            query_tokens=q_tok,
            answer_hidden=h_ans,
            sequence=seq,
            attn_mask=mask,
        )

    @torch.no_grad()
    def predict(self, ctx_xyz: torch.Tensor, qry_xyz: torch.Tensor, qry_type: torch.Tensor) -> torch.Tensor:
        return self.forward(ctx_xyz, qry_xyz, qry_type).pred_answer


def build_distnorm_continuous_model_from_args(args: Dict[str, Any]) -> PrimitiveAnsweringDistNormContinuousModel:
    return PrimitiveAnsweringDistNormContinuousModel(
        d_model=int(args.get("d_model", 384)),
        n_layers=int(args.get("n_layers", 12)),
        n_heads=int(args.get("n_heads", 6)),
        mlp_ratio=float(args.get("mlp_ratio", 4.0)),
        dropout=float(args.get("dropout", 0.0)),
        drop_path=float(args.get("drop_path", 0.0)),
        backbone_impl=str(args.get("backbone_impl", "nepa2d")),
        num_groups=int(args.get("num_groups", 64)),
        group_size=int(args.get("group_size", 32)),
        patch_center_mode=str(args.get("patch_center_mode", "fps")),
        patch_fps_random_start=bool(args.get("patch_fps_random_start", 1)),
        local_encoder=str(args.get("local_encoder", "pointmae_conv")),
        query_type_vocab=int(args.get("query_type_vocab", 6)),
        generator_depth=int(args.get("generator_depth", 2)),
        distance_floor=float(args.get("distance_floor", 0.0)),
    )


def load_distnorm_continuous_model(
    ckpt_path: str,
    device: torch.device,
) -> tuple[PrimitiveAnsweringDistNormContinuousModel, Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = dict(ckpt.get("args", {}))
    model = build_distnorm_continuous_model_from_args(args)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, ckpt, args
