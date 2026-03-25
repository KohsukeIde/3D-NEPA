from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from nepa3d.tracks.patch_nepa.cqa.data.cqa_codec import (
    CQA_VOCAB_VERSION,
    answer_vocab_size,
    query_type_vocab_size,
)
from nepa3d.tracks.patch_nepa.cqa.models.primitive_answering import (
    PrimitiveAnsweringClassifier,
    PrimitiveAnsweringModel,
)
from nepa3d.tracks.patch_nepa.cqa.models.primitive_answering_encdec import (
    PrimitiveAnsweringEncDecClassifier,
    PrimitiveAnsweringEncDecModel,
)


def normalize_cqa_model_args(args: Mapping[str, Any], *, ckpt_vocab_version: str | None = None) -> dict[str, Any]:
    cfg = dict(args)
    codec_version = str(ckpt_vocab_version or cfg.get("codec_version", CQA_VOCAB_VERSION) or CQA_VOCAB_VERSION)
    if int(cfg.get("answer_vocab", 0)) <= 0:
        cfg["answer_vocab"] = int(answer_vocab_size(codec_version))
    if int(cfg.get("query_type_vocab", 0)) <= 0:
        cfg["query_type_vocab"] = int(query_type_vocab_size(codec_version))
    cfg["codec_version"] = codec_version

    model_arch = str(cfg.get("model_arch", "prefixlm") or "prefixlm").strip().lower()
    if model_arch not in {"prefixlm", "encdec"}:
        raise ValueError(f"unknown model_arch={model_arch!r}")
    cfg["model_arch"] = model_arch

    if model_arch == "encdec":
        factorization = str(cfg.get("answer_factorization", "independent") or "independent").strip().lower()
        if factorization != "independent":
            raise ValueError(
                "encdec CQA currently supports only answer_factorization='independent', "
                f"got {factorization!r}"
            )
        cfg["answer_factorization"] = "independent"
        cfg["query_interface_mode"] = "no_q"
        cfg["decoder_layers"] = int(cfg.get("decoder_layers", 4))

    return cfg


def build_cqa_model_from_args(args: Mapping[str, Any], *, ckpt_vocab_version: str | None = None) -> nn.Module:
    cfg = normalize_cqa_model_args(args, ckpt_vocab_version=ckpt_vocab_version)
    common = dict(
        d_model=int(cfg.get("d_model", 384)),
        n_layers=int(cfg.get("n_layers", 12)),
        n_heads=int(cfg.get("n_heads", 6)),
        mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        dropout=float(cfg.get("dropout", 0.0)),
        drop_path=float(cfg.get("drop_path", 0.0)),
        backbone_impl=str(cfg.get("backbone_impl", "nepa2d")),
        num_groups=int(cfg.get("num_groups", 64)),
        group_size=int(cfg.get("group_size", 32)),
        patch_center_mode=str(cfg.get("patch_center_mode", "fps")),
        patch_fps_random_start=bool(cfg.get("patch_fps_random_start", 1)),
        local_encoder=str(cfg.get("local_encoder", "pointmae_conv")),
        query_type_vocab=int(cfg.get("query_type_vocab")),
        answer_vocab=int(cfg.get("answer_vocab")),
        codec_version=str(cfg.get("codec_version")),
        answer_factorization=str(cfg.get("answer_factorization", "ar")),
        query_interface_mode=str(cfg.get("query_interface_mode", "full_q")),
    )
    if str(cfg["model_arch"]) == "encdec":
        return PrimitiveAnsweringEncDecModel(
            **common,
            decoder_layers=int(cfg.get("decoder_layers", 4)),
            generator_depth=int(cfg.get("generator_depth", 0)),
        )
    return PrimitiveAnsweringModel(
        **common,
        generator_depth=int(cfg.get("generator_depth", 2)),
    )


def build_cqa_classifier(pretrained: nn.Module, *, n_cls: int, pool: str = "mean") -> nn.Module:
    if isinstance(pretrained, PrimitiveAnsweringEncDecModel):
        return PrimitiveAnsweringEncDecClassifier(pretrained, n_cls=int(n_cls), pool=str(pool))
    if isinstance(pretrained, PrimitiveAnsweringModel):
        return PrimitiveAnsweringClassifier(pretrained, n_cls=int(n_cls), pool=str(pool))
    raise TypeError(f"unsupported pretrained CQA model type: {type(pretrained)!r}")


def load_cqa_model_from_ckpt(
    ckpt_path: str,
    device: torch.device | str,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = normalize_cqa_model_args(
        ckpt.get("args", {}),
        ckpt_vocab_version=str(ckpt.get("vocab_version", "")) or None,
    )
    model = build_cqa_model_from_args(args)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, ckpt, args
