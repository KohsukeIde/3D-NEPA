"""Encoder-Decoder transformer for long Q/A sequences.

This is a pragmatic scaling option for QA *split* layout:
  [BOS][Q...][A...][EOS]

- Encoder runs on (BOS+Q) tokens (bidirectional self-attn).
- Decoder runs on A tokens (causal self-attn) with cross-attn to encoder memory.

Optionally supports a simple kNN topology mask on the encoder side (dense mask).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def _causal_mask(t: int, device: torch.device) -> torch.Tensor:
    # True means masked (disallowed). Works for nn.MultiheadAttention / TransformerDecoderLayer.
    return torch.triu(torch.ones((t, t), device=device, dtype=torch.bool), diagonal=1)


def _knn_topology_mask(
    xyz: torch.Tensor,
    k: int,
    include_bos: bool = True,
) -> torch.Tensor:
    """Build an encoder self-attention mask based on kNN in xyz.

    Args:
        xyz: (B, L, 3) positions. For BOS, use zeros.
        k: number of neighbors (excluding self) to attend to.
        include_bos: if True, allow all tokens to attend to BOS and BOS to attend to all.

    Returns:
        attn_mask: (B, L, L) bool mask where True means "mask out" (disallow).
    """
    b, l, _ = xyz.shape
    if k <= 0 or l <= 1:
        return torch.zeros((b, l, l), device=xyz.device, dtype=torch.bool)

    # pairwise distances
    d = torch.cdist(xyz, xyz, p=2)  # (B, L, L)
    eye = torch.eye(l, device=xyz.device, dtype=torch.bool)
    d = d.masked_fill(eye[None, ...], float("inf"))  # exclude self

    # pick k nearest for each query
    kk = min(int(k), max(l - 1, 0))
    nn_idx = torch.topk(d, k=kk, dim=-1, largest=False).indices  # (B, L, kk)

    allow = torch.zeros((b, l, l), device=xyz.device, dtype=torch.bool)
    allow.scatter_(dim=-1, index=nn_idx, value=True)
    allow = allow | eye[None, ...]  # always allow self

    if include_bos and l > 0:
        allow[:, :, 0] = True  # everyone can attend to BOS
        allow[:, 0, :] = True  # BOS can attend to all

    return ~allow


class EncoderDecoderTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        topo_k: int = 0,
        topo_include_bos: bool = True,
        src_causal: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.topo_k = int(topo_k)
        self.topo_include_bos = bool(topo_include_bos)
        self.src_causal = bool(src_causal)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_encoder_layers))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(num_decoder_layers))

        self.enc_ln = nn.LayerNorm(d_model)
        self.dec_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        enc_in: torch.Tensor,
        dec_in: torch.Tensor,
        enc_xyz: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            enc_in: (B, L_enc, D)
            dec_in: (B, L_dec, D)
            enc_xyz: (B, L_enc, 3) optional positions for topology mask.

        Returns:
            enc_out: (B, L_enc, D)
            dec_out: (B, L_dec, D)
        """
        b, l_enc, _ = enc_in.shape
        b2, l_dec, _ = dec_in.shape
        assert b == b2

        enc_mask = None
        mask_bll: Optional[torch.Tensor] = None
        if self.topo_k > 0 and enc_xyz is not None and l_enc > 1:
            mask_bll = _knn_topology_mask(enc_xyz, k=self.topo_k, include_bos=self.topo_include_bos)

        if self.src_causal and l_enc > 1:
            causal = _causal_mask(l_enc, device=enc_in.device)  # (L, L)
            causal_bll = causal[None, :, :].expand(b, l_enc, l_enc)
            mask_bll = causal_bll if mask_bll is None else (mask_bll | causal_bll)

        if mask_bll is not None:
            # nn.TransformerEncoder expects (L, L) or (B*nhead, L, L).
            enc_mask = mask_bll[:, None, :, :].expand(b, self.nhead, l_enc, l_enc).reshape(
                b * self.nhead, l_enc, l_enc
            )

        enc_out = self.encoder(enc_in, mask=enc_mask)
        enc_out = self.enc_ln(enc_out)

        dec_mask = _causal_mask(l_dec, device=dec_in.device) if l_dec > 0 else None
        dec_out = self.decoder(dec_in, memory=enc_out, tgt_mask=dec_mask)
        dec_out = self.dec_ln(dec_out)
        return enc_out, dec_out
