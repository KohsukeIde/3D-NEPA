from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from nepa3d.core.models.causal_transformer import CausalTransformer
from nepa3d.core.models.point_patch_embed import PointPatchEmbed
from nepa3d.tracks.patch_nepa.cqa.data.cqa_codec import CQA_VOCAB_VERSION, mask_logits_for_query_type


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
class PrimitiveAnsweringOutput:
    logits: torch.Tensor
    hidden: torch.Tensor
    ctx_tokens: torch.Tensor
    ctx_centers: torch.Tensor
    query_tokens: torch.Tensor
    answer_hidden: torch.Tensor
    sequence: torch.Tensor
    attn_mask: torch.Tensor


class PrimitiveAnsweringModel(nn.Module):
    """Explicit-query promptable primitive answering model.

    Sequence:
      [BOS, C_1..C_P, SEP_CQ, TYPE, Q_1..Q_N, SEP_A, A_1..A_N]
      or, when ``query_interface_mode='no_q'``:
      [BOS, C_1..C_P, SEP_CQ, TYPE, SEP_A, A_1..A_N]

    Prompt block (BOS + C + SEP_CQ + TYPE + Q + SEP_A) is bidirectional.

    Answer block factorization:
      - ``ar``: causal with teacher-forced answer-prefix inputs.
      - ``parallel``: joint non-AR answer block; slots see the prompt and other
        answer slots, but never previous answer codes.
      - ``independent``: strictly slot-wise non-AR answer block; each slot sees
        only the prompt and itself.

    ``parallel`` and ``independent`` separate two questions:
      - does the gain come from the Q/A interface rather than AR decoding?
      - do answer slots need to interact with each other at all?

    ``query_interface_mode`` separates a third question:
      - ``full_q``: answer slots can read the full query block
      - ``self_q``: answer slot ``A_i`` can read only ``Q_i``
      - ``no_q``  : there is no explicit query block, only the per-slot query anchor
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
        answer_vocab: int = 640,
        generator_depth: int = 2,
        codec_version: str = CQA_VOCAB_VERSION,
        answer_factorization: str = "ar",
        query_interface_mode: str = "full_q",
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.answer_vocab = int(answer_vocab)
        self.query_type_vocab = int(query_type_vocab)
        self.codec_version = str(codec_version or CQA_VOCAB_VERSION)
        self.num_groups = int(num_groups)
        self.group_size = int(group_size)
        self.answer_factorization = str(answer_factorization).strip().lower()
        if self.answer_factorization not in {"ar", "parallel", "independent"}:
            raise ValueError(
                "answer_factorization must be 'ar', 'parallel', or "
                f"'independent', got {answer_factorization!r}"
            )
        self.query_interface_mode = str(query_interface_mode).strip().lower()
        if self.query_interface_mode not in {"full_q", "self_q", "no_q"}:
            raise ValueError(
                "query_interface_mode must be 'full_q', 'self_q', or 'no_q', "
                f"got {query_interface_mode!r}"
            )

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
        self.answer_input_embed = nn.Embedding(int(answer_vocab), int(d_model))

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
        self.answer_head = _MLP(int(d_model), int(answer_vocab), hidden_dim=int(d_model), n_layers=2, dropout=float(dropout))

    @staticmethod
    def _build_prompt_answer_mask(
        *,
        ctx_len: int,
        n_query: int,
        n_answer: int,
        device: torch.device,
        answer_factorization: str = "ar",
        query_interface_mode: str = "full_q",
    ) -> torch.Tensor:
        prompt_len = 1 + int(ctx_len) + 1 + 1 + int(n_query) + 1
        total = int(prompt_len) + int(n_answer)
        mask = torch.zeros((total, total), device=device, dtype=torch.bool)
        query_start = 1 + int(ctx_len) + 1 + 1
        query_end = query_start + int(n_query)
        if int(n_answer) > 0:
            if str(answer_factorization) == "ar":
                tri = torch.triu(torch.ones((n_answer, n_answer), device=device, dtype=torch.bool), diagonal=1)
                mask[int(prompt_len):, int(prompt_len):] = tri
            elif str(answer_factorization) == "parallel":
                pass
            elif str(answer_factorization) == "independent":
                offdiag = torch.ones((n_answer, n_answer), device=device, dtype=torch.bool)
                offdiag.fill_diagonal_(False)
                mask[int(prompt_len):, int(prompt_len):] = offdiag
            else:
                raise KeyError(f"unknown answer_factorization={answer_factorization}")
            mask[:int(prompt_len), int(prompt_len):] = True
            if int(n_query) > 0:
                if str(query_interface_mode) == "full_q":
                    pass
                elif str(query_interface_mode) == "self_q":
                    qmask = torch.ones((n_answer, n_query), device=device, dtype=torch.bool)
                    diag = min(int(n_answer), int(n_query))
                    if diag > 0:
                        idx = torch.arange(diag, device=device)
                        qmask[idx, idx] = False
                    mask[int(prompt_len):, int(query_start):int(query_end)] = qmask
                elif str(query_interface_mode) == "no_q":
                    pass
                else:
                    raise KeyError(f"unknown query_interface_mode={query_interface_mode}")
        return mask

    def encode_context(self, ctx_xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_out = self.ctx_patch(ctx_xyz)
        tok = patch_out.tokens + self.center_pos(patch_out.centers_xyz)
        return tok, patch_out.centers_xyz, patch_out.group_idx

    def encode_queries(self, qry_xyz: torch.Tensor, qry_type: torch.Tensor) -> torch.Tensor:
        q = self.query_embed(qry_xyz)
        if qry_type.dim() == 1:
            qry_type = qry_type.unsqueeze(0).expand(qry_xyz.shape[0], -1)
        return q + self.query_type_embed(qry_type)

    def _build_answer_inputs(self, qry_xyz: torch.Tensor, answer_code: torch.Tensor) -> torch.Tensor:
        b, n, _ = qry_xyz.shape
        pos = self.query_embed(qry_xyz)
        if n <= 0:
            return pos.new_zeros((b, 0, self.d_model))
        if self.answer_factorization == "ar":
            prev = self.answer_input_embed(answer_code[:, :-1]) if n > 1 else pos.new_zeros((b, 0, self.d_model))
            first = self.ans_bos.expand(b, 1, -1)
            ans_tok = torch.cat([first, prev], dim=1)
        else:
            # Non-AR baselines keep the same query anchors but remove any
            # previous-answer conditioning from the input side.
            ans_tok = self.ans_bos.expand(b, n, -1)
        return ans_tok + pos

    def forward(
        self,
        ctx_xyz: torch.Tensor,
        qry_xyz: torch.Tensor,
        qry_type: torch.Tensor,
        answer_code: torch.Tensor,
    ) -> PrimitiveAnsweringOutput:
        b = int(ctx_xyz.shape[0])
        ctx_tok, ctx_centers, _ctx_group_idx = self.encode_context(ctx_xyz)
        q_tok = self.encode_queries(qry_xyz, qry_type)
        ans_in = self._build_answer_inputs(qry_xyz, answer_code)

        if qry_type.dim() == 2:
            type_scalar = qry_type[:, 0]
        else:
            type_scalar = qry_type
        type_tok = self.query_type_embed(type_scalar).unsqueeze(1)

        if self.query_interface_mode == "no_q":
            q_tok_seq = q_tok[:, :0, :]
        else:
            q_tok_seq = q_tok
        seq = torch.cat(
            [
                self.bos.expand(b, 1, -1),
                ctx_tok,
                self.sep_cq.expand(b, 1, -1),
                type_tok,
                q_tok_seq,
                self.sep_a.expand(b, 1, -1),
                ans_in,
            ],
            dim=1,
        )
        prompt_len = 1 + ctx_tok.shape[1] + 1 + 1 + q_tok_seq.shape[1] + 1
        mask = self._build_prompt_answer_mask(
            ctx_len=int(ctx_tok.shape[1]),
            n_query=int(q_tok_seq.shape[1]),
            n_answer=int(ans_in.shape[1]),
            device=seq.device,
            answer_factorization=self.answer_factorization,
            query_interface_mode=self.query_interface_mode,
        )
        h = self.backbone(seq, is_causal=False, attn_mask_override=mask)
        h_ans = h[:, prompt_len:, :]
        if self.generator is not None:
            if self.answer_factorization == "ar":
                gen_mask = torch.triu(
                    torch.ones((h_ans.shape[1], h_ans.shape[1]), device=h_ans.device, dtype=torch.bool),
                    diagonal=1,
                )
            elif self.answer_factorization == "parallel":
                gen_mask = torch.zeros((h_ans.shape[1], h_ans.shape[1]), device=h_ans.device, dtype=torch.bool)
            else:
                gen_mask = torch.ones((h_ans.shape[1], h_ans.shape[1]), device=h_ans.device, dtype=torch.bool)
                gen_mask.fill_diagonal_(False)
            h_ans = self.generator(h_ans, is_causal=False, attn_mask_override=gen_mask)
        logits = self.answer_head(h_ans)
        return PrimitiveAnsweringOutput(
            logits=logits,
            hidden=h,
            ctx_tokens=ctx_tok,
            ctx_centers=ctx_centers,
            query_tokens=q_tok_seq,
            answer_hidden=h_ans,
            sequence=seq,
            attn_mask=mask,
        )

    @torch.no_grad()
    def generate(self, ctx_xyz: torch.Tensor, qry_xyz: torch.Tensor, qry_type: torch.Tensor) -> torch.Tensor:
        b = int(ctx_xyz.shape[0])
        n = int(qry_xyz.shape[1])
        out_codes = torch.zeros((b, n), device=ctx_xyz.device, dtype=torch.long)
        if self.answer_factorization in {"parallel", "independent"}:
            cur = self.forward(ctx_xyz, qry_xyz, qry_type, out_codes)
            step_logits = mask_logits_for_query_type(
                cur.logits,
                qry_type,
                codec_version=self.codec_version,
                vocab_size=int(self.answer_vocab),
            )
            return step_logits.argmax(dim=-1)
        for i in range(n):
            cur = self.forward(ctx_xyz, qry_xyz[:, : i + 1, :], qry_type[:, : i + 1], out_codes[:, : i + 1])
            step_logits = mask_logits_for_query_type(
                cur.logits[:, i : i + 1, :],
                qry_type[:, i : i + 1],
                codec_version=self.codec_version,
                vocab_size=int(self.answer_vocab),
            )
            out_codes[:, i] = step_logits[:, 0, :].argmax(dim=-1)
        return out_codes


class PrimitiveAnsweringClassifier(nn.Module):
    def __init__(self, pretrained: PrimitiveAnsweringModel, n_cls: int, pool: str = "mean") -> None:
        super().__init__()
        self.ctx_patch = pretrained.ctx_patch
        self.center_pos = pretrained.center_pos
        self.backbone = pretrained.backbone
        self.bos = pretrained.bos
        self.pool = str(pool)
        self.head = _MLP(int(pretrained.d_model), int(n_cls), hidden_dim=int(pretrained.d_model), n_layers=2, dropout=0.0)

    def forward(self, ctx_xyz: torch.Tensor) -> torch.Tensor:
        patch_out = self.ctx_patch(ctx_xyz)
        ctx_tok = patch_out.tokens + self.center_pos(patch_out.centers_xyz)
        seq = torch.cat([self.bos.expand(ctx_tok.shape[0], 1, -1), ctx_tok], dim=1)
        h = self.backbone(seq, is_causal=False)
        pooled = h[:, 0, :] if self.pool == "bos" else h[:, 1:, :].mean(dim=1)
        return self.head(pooled)
