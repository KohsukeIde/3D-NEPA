from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from nepa3d.tracks.patch_nepa.cqa.models.primitive_answering import (
    PrimitiveAnsweringClassifier,
    PrimitiveAnsweringModel,
    _MLP,
)
from nepa3d.tracks.patch_nepa.mainline.models.pointmae_patch_classifier import (
    Encoder as PointMAEEncoder,
    Group as PointMAEGroup,
    TransformerEncoder as PointMAETransformerEncoder,
)


def _extract_pointmae_encoder_state(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    raw_state: Mapping[str, Any]
    if isinstance(payload.get("base_model", None), Mapping):
        raw_state = payload["base_model"]  # type: ignore[assignment]
    elif isinstance(payload.get("model", None), Mapping):
        raw_state = payload["model"]  # type: ignore[assignment]
    elif isinstance(payload.get("state_dict", None), Mapping):
        raw_state = payload["state_dict"]  # type: ignore[assignment]
    else:
        raw_state = payload

    mapped: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if not isinstance(value, torch.Tensor):
            continue
        k = str(key)
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("MAE_encoder."):
            k = k[len("MAE_encoder.") :]
        if k == "mask_token":
            continue
        if k.startswith(("encoder.", "pos_embed.", "blocks.", "norm.")):
            mapped[k] = value
    return mapped


class FrozenPointMAEContextEncoder(nn.Module):
    """Point-MAE encoder surface used as a frozen point-only external baseline."""

    def __init__(
        self,
        *,
        trans_dim: int = 384,
        depth: int = 12,
        drop_path_rate: float = 0.1,
        num_heads: int = 6,
        group_size: int = 32,
        num_groups: int = 64,
        encoder_dims: int = 384,
        ckpt_path: str = "",
    ) -> None:
        super().__init__()
        self.trans_dim = int(trans_dim)
        self.depth = int(depth)
        self.drop_path_rate = float(drop_path_rate)
        self.num_heads = int(num_heads)
        self.group_size = int(group_size)
        self.num_group = int(num_groups)
        self.encoder_dims = int(encoder_dims)
        self.ckpt_path = str(ckpt_path).strip()

        self.group_divider = PointMAEGroup(num_group=self.num_group, group_size=self.group_size)
        self.encoder = PointMAEEncoder(encoder_channel=self.encoder_dims)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = PointMAETransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            qkv_bias=False,
            drop_rate=0.0,
            attn_drop_rate=0.0,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        if self.ckpt_path:
            self.load_pretrained(self.ckpt_path)

    def load_pretrained(self, ckpt_path: str) -> None:
        path = Path(str(ckpt_path))
        if not path.is_file():
            raise FileNotFoundError(f"external Point-MAE checkpoint not found: {path}")
        payload = torch.load(str(path), map_location="cpu")
        if not isinstance(payload, Mapping):
            raise TypeError(f"unsupported Point-MAE checkpoint payload type: {type(payload)!r}")
        state = _extract_pointmae_encoder_state(payload)
        if not state:
            raise RuntimeError(f"no encoder weights found in Point-MAE checkpoint: {path}")
        self.load_state_dict(state, strict=False)

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        neighborhood, center = self.group_divider(xyz)
        tokens = self.encoder(neighborhood)
        pos = self.pos_embed(center)
        h = self.blocks(tokens, pos)
        h = self.norm(h)
        return h, center


class ExternalPointMAEPrimitiveAnsweringModel(PrimitiveAnsweringModel):
    """Shared typed answerer with a frozen external Point-MAE context encoder."""

    def __init__(
        self,
        *,
        external_backbone_ckpt: str,
        freeze_external_encoder: bool = True,
        external_backbone_depth: int = 12,
        external_backbone_heads: int = 6,
        external_backbone_drop_path: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_arch = "external_pointmae"
        self.external_backbone_ckpt = str(external_backbone_ckpt).strip()
        self.freeze_external_encoder = bool(freeze_external_encoder)
        if not self.external_backbone_ckpt:
            raise ValueError("model_arch=external_pointmae requires external_backbone_ckpt")

        del self.ctx_patch
        del self.center_pos
        self.external_ctx = FrozenPointMAEContextEncoder(
            trans_dim=int(self.d_model),
            depth=int(external_backbone_depth),
            drop_path_rate=float(external_backbone_drop_path),
            num_heads=int(external_backbone_heads),
            group_size=int(kwargs.get("group_size", 32)),
            num_groups=int(kwargs.get("num_groups", 64)),
            encoder_dims=int(self.d_model),
            ckpt_path=self.external_backbone_ckpt,
        )
        if self.freeze_external_encoder:
            for p in self.external_ctx.parameters():
                p.requires_grad = False
            self.external_ctx.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_external_encoder:
            self.external_ctx.eval()
        return self

    def encode_context(self, ctx_xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        ctx = contextlib.nullcontext()
        if self.freeze_external_encoder:
            ctx = torch.no_grad()
        with ctx:
            tok, centers = self.external_ctx(ctx_xyz)
        return tok, centers, None


class ExternalPointMAEPrimitiveAnsweringClassifier(nn.Module):
    def __init__(self, pretrained: ExternalPointMAEPrimitiveAnsweringModel, n_cls: int, pool: str = "mean") -> None:
        super().__init__()
        self.external_ctx = pretrained.external_ctx
        self.backbone = pretrained.backbone
        self.bos = pretrained.bos
        self.pool = str(pool)
        self.freeze_external_encoder = bool(pretrained.freeze_external_encoder)
        self.head = _MLP(int(pretrained.d_model), int(n_cls), hidden_dim=int(pretrained.d_model), n_layers=2, dropout=0.0)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_external_encoder:
            self.external_ctx.eval()
        return self

    def forward(self, ctx_xyz: torch.Tensor) -> torch.Tensor:
        ctx = contextlib.nullcontext()
        if self.freeze_external_encoder:
            ctx = torch.no_grad()
        with ctx:
            ctx_tok, _ctx_centers = self.external_ctx(ctx_xyz)
        seq = torch.cat([self.bos.expand(ctx_tok.shape[0], 1, -1), ctx_tok], dim=1)
        h = self.backbone(seq, is_causal=False)
        pooled = h[:, 0, :] if self.pool == "bos" else h[:, 1:, :].mean(dim=1)
        return self.head(pooled)

