from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from nepa3d.core.models.encdec_transformer import EncoderDecoderTransformer
from nepa3d.core.models.point_patch_embed import PointPatchEmbed
from nepa3d.tracks.patch_nepa.cqa.data.cqa_codec import CQA_VOCAB_VERSION, mask_logits_for_query_type
from nepa3d.tracks.patch_nepa.cqa.models.primitive_answering import (
    PrimitiveAnsweringOutput,
    _MLP,
)


class PrimitiveAnsweringEncDecModel(nn.Module):
    """Discrete encoder-decoder CQA model.

    Enc-dec v1 keeps the current typed-query discrete CE interface while
    separating context understanding from query-conditioned answering:

      encoder input: [BOS, C_1..C_P]
      decoder input: query_embed(Q_i.xyz) + type_embed(TYPE_i)

    `answer_code` is accepted for API compatibility but is only used as a loss
    target by the trainer. It is not consumed by the decoder because enc-dec v1
    is strictly independent over query answers.
    """

    def __init__(
        self,
        *,
        d_model: int = 384,
        n_layers: int = 12,
        decoder_layers: int = 4,
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
        answer_vocab: int = 640,
        generator_depth: int = 2,
        codec_version: str = CQA_VOCAB_VERSION,
        answer_factorization: str = "independent",
        query_interface_mode: str = "no_q",
    ) -> None:
        super().__init__()
        del generator_depth  # enc-dec v1 has no answer-side generator stack
        self.model_arch = "encdec"
        self.d_model = int(d_model)
        self.answer_vocab = int(answer_vocab)
        self.query_type_vocab = int(query_type_vocab)
        self.codec_version = str(codec_version or CQA_VOCAB_VERSION)
        self.num_groups = int(num_groups)
        self.group_size = int(group_size)
        self.decoder_layers = int(decoder_layers)
        self.backbone_impl = str(backbone_impl)

        self.answer_factorization = str(answer_factorization).strip().lower()
        if self.answer_factorization != "independent":
            raise ValueError(
                "PrimitiveAnsweringEncDecModel currently supports only "
                f"answer_factorization='independent', got {answer_factorization!r}"
            )

        self.requested_query_interface_mode = str(query_interface_mode).strip().lower()
        self.query_interface_mode = "no_q"

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
        self.backbone = EncoderDecoderTransformer(
            d_model=int(d_model),
            nhead=int(n_heads),
            num_encoder_layers=int(n_layers),
            num_decoder_layers=int(decoder_layers),
            dim_feedforward=int(round(float(d_model) * float(mlp_ratio))),
            dropout=float(dropout),
            drop_path=float(drop_path),
            decoder_self_attn="independent",
        )
        self.answer_head = _MLP(int(d_model), int(answer_vocab), hidden_dim=int(d_model), n_layers=2, dropout=float(dropout))

    def encode_context(self, ctx_xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_out = self.ctx_patch(ctx_xyz)
        ctx_tok = patch_out.tokens + self.center_pos(patch_out.centers_xyz)
        enc_in = torch.cat([self.bos.expand(ctx_tok.shape[0], 1, -1), ctx_tok], dim=1)
        enc_out = self.backbone.encode(enc_in)
        return enc_out, ctx_tok, patch_out.centers_xyz

    def encode_queries(self, qry_xyz: torch.Tensor, qry_type: torch.Tensor) -> torch.Tensor:
        q = self.query_embed(qry_xyz)
        if qry_type.dim() == 1:
            qry_type = qry_type.unsqueeze(0).expand(qry_xyz.shape[0], -1)
        return q + self.query_type_embed(qry_type)

    def forward(
        self,
        ctx_xyz: torch.Tensor,
        qry_xyz: torch.Tensor,
        qry_type: torch.Tensor,
        answer_code: torch.Tensor,
    ) -> PrimitiveAnsweringOutput:
        del answer_code  # target-only in enc-dec v1; decoder input is query-only
        enc_out, ctx_tok, ctx_centers = self.encode_context(ctx_xyz)
        dec_in = self.encode_queries(qry_xyz, qry_type)
        dec_out = self.backbone.decode(enc_out, dec_in)
        logits = self.answer_head(dec_out)

        n_query = int(dec_in.shape[1])
        attn_mask = torch.ones((n_query, n_query), device=dec_in.device, dtype=torch.bool)
        if n_query > 0:
            attn_mask.fill_diagonal_(False)
        return PrimitiveAnsweringOutput(
            logits=logits,
            hidden=enc_out,
            ctx_tokens=ctx_tok,
            ctx_centers=ctx_centers,
            query_tokens=dec_in,
            answer_hidden=dec_out,
            sequence=dec_in,
            attn_mask=attn_mask,
        )

    @torch.no_grad()
    def generate(self, ctx_xyz: torch.Tensor, qry_xyz: torch.Tensor, qry_type: torch.Tensor) -> torch.Tensor:
        b = int(ctx_xyz.shape[0])
        n = int(qry_xyz.shape[1])
        dummy = torch.zeros((b, n), device=ctx_xyz.device, dtype=torch.long)
        out = self.forward(ctx_xyz=ctx_xyz, qry_xyz=qry_xyz, qry_type=qry_type, answer_code=dummy)
        step_logits = mask_logits_for_query_type(
            out.logits,
            qry_type,
            codec_version=self.codec_version,
            vocab_size=int(self.answer_vocab),
        )
        return step_logits.argmax(dim=-1)


class PrimitiveAnsweringEncDecClassifier(nn.Module):
    def __init__(
        self,
        pretrained: PrimitiveAnsweringEncDecModel,
        n_cls: int,
        pool: Literal["mean", "bos"] = "mean",
    ) -> None:
        super().__init__()
        self.ctx_patch = pretrained.ctx_patch
        self.center_pos = pretrained.center_pos
        self.bos = pretrained.bos
        self.backbone = pretrained.backbone
        self.pool = str(pool)
        self.head = _MLP(
            int(pretrained.d_model),
            int(n_cls),
            hidden_dim=int(pretrained.d_model),
            n_layers=2,
            dropout=0.0,
        )

    def forward(self, ctx_xyz: torch.Tensor) -> torch.Tensor:
        patch_out = self.ctx_patch(ctx_xyz)
        ctx_tok = patch_out.tokens + self.center_pos(patch_out.centers_xyz)
        enc_in = torch.cat([self.bos.expand(ctx_tok.shape[0], 1, -1), ctx_tok], dim=1)
        h = self.backbone.encode(enc_in)
        pooled = h[:, 0, :] if self.pool == "bos" else h[:, 1:, :].mean(dim=1)
        return self.head(pooled)
