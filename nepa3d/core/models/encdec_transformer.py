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


def _independent_mask(t: int, device: torch.device) -> torch.Tensor:
    mask = torch.ones((t, t), device=device, dtype=torch.bool)
    mask.fill_diagonal_(False)
    return mask


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
        drop_path: float = 0.0,
        topo_k: int = 0,
        topo_include_bos: bool = True,
        src_causal: bool = False,
        decoder_self_attn: str = "causal",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.topo_k = int(topo_k)
        self.topo_include_bos = bool(topo_include_bos)
        self.src_causal = bool(src_causal)
        self.decoder_self_attn = str(decoder_self_attn).strip().lower()
        if self.decoder_self_attn not in {"causal", "independent"}:
            raise ValueError(
                "decoder_self_attn must be 'causal' or 'independent', "
                f"got {decoder_self_attn!r}"
            )

        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(int(num_encoder_layers))
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(int(num_decoder_layers))
            ]
        )

        drop_path = max(0.0, min(float(drop_path), 1.0))
        if int(num_encoder_layers) <= 1:
            enc_rates = [drop_path]
        else:
            enc_rates = torch.linspace(0.0, drop_path, int(num_encoder_layers)).tolist()
        if int(num_decoder_layers) <= 1:
            dec_rates = [drop_path]
        else:
            dec_rates = torch.linspace(0.0, drop_path, int(num_decoder_layers)).tolist()
        self.encoder_drop_path_rates = [float(r) for r in enc_rates]
        self.decoder_drop_path_rates = [float(r) for r in dec_rates]

        self.enc_ln = nn.LayerNorm(d_model)
        self.dec_ln = nn.LayerNorm(d_model)

    @staticmethod
    def _apply_drop_path(x_prev: torch.Tensor, x_next: torch.Tensor, drop_prob: float) -> torch.Tensor:
        p = float(drop_prob)
        if p <= 0.0 or (not x_prev.requires_grad) or (not x_prev.is_floating_point()):
            return x_next
        if p >= 1.0:
            return x_prev
        keep_prob = 1.0 - p
        shape = (x_prev.shape[0],) + (1,) * (x_prev.ndim - 1)
        rnd = torch.rand(shape, device=x_prev.device, dtype=x_prev.dtype)
        mask = (rnd < keep_prob).to(x_prev.dtype)
        residual = (x_next - x_prev) * mask / keep_prob
        return x_prev + residual

    def encode(
        self,
        enc_in: torch.Tensor,
        enc_xyz: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, l_enc, _ = enc_in.shape

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

        enc_out = enc_in
        for li, enc_layer in enumerate(self.encoder_layers):
            enc_next = enc_layer(enc_out, src_mask=enc_mask)
            if self.training:
                enc_out = self._apply_drop_path(
                    enc_out,
                    enc_next,
                    self.encoder_drop_path_rates[li],
                )
            else:
                enc_out = enc_next
        return self.enc_ln(enc_out)

    def decode(
        self,
        memory: torch.Tensor,
        dec_in: torch.Tensor,
    ) -> torch.Tensor:
        _b, l_dec, _ = dec_in.shape
        if l_dec <= 0:
            return self.dec_ln(dec_in)

        if self.decoder_self_attn == "causal":
            dec_mask = _causal_mask(l_dec, device=dec_in.device)
        elif self.decoder_self_attn == "independent":
            dec_mask = _independent_mask(l_dec, device=dec_in.device)
        else:
            raise AssertionError("unreachable")
        dec_out = dec_in
        for li, dec_layer in enumerate(self.decoder_layers):
            dec_next = dec_layer(dec_out, memory=memory, tgt_mask=dec_mask)
            if self.training:
                dec_out = self._apply_drop_path(
                    dec_out,
                    dec_next,
                    self.decoder_drop_path_rates[li],
                )
            else:
                dec_out = dec_next
        return self.dec_ln(dec_out)

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
        b, _l_enc, _ = enc_in.shape
        b2, _l_dec, _ = dec_in.shape
        assert b == b2

        enc_out = self.encode(enc_in, enc_xyz=enc_xyz)
        dec_out = self.decode(enc_out, dec_in)
        return enc_out, dec_out
