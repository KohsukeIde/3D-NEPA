"""Patchified Transformer classification training (scratch or finetune).

This script is intentionally *separate* from `finetune_cls.py` so we don't
accidentally break existing NEPA-token pipelines.

Primary purpose (Step 1 baseline):
- Train a patch-token transformer from random init ("Transformer (rand)" style)
- Verify ScanObjectNN / ModelNet40 reach typical ~80%+ ranges under matched settings

We reuse the cached datasets produced by the existing preprocess scripts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader

from nepa3d.data.cls_patch_dataset import (
    PatchClsArrayDataset,
    PatchClsPointDataset,
    PointAugConfig,
    load_scanobjectnn_h5_arrays,
)
from nepa3d.data.modelnet40_index import (
    build_label_map,
    list_npz,
    scanobjectnn_group_key,
    stratified_train_val_split,
)
from nepa3d.models.patch_classifier import PatchTransformerClassifier
from nepa3d.models.patch_nepa_classifier import PatchTransformerNepaClassifier
from nepa3d.models.pointmae_patch_classifier import PointMAEPatchClassifier


def add_args(p: argparse.ArgumentParser) -> None:
    # IO
    p.add_argument("--ckpt", type=str, default="", help="Optional init checkpoint (can be empty for scratch).")
    p.add_argument("--ckpt_use_ema", type=int, default=0, choices=[0, 1], help="Load model_ema from pretrain ckpt when available.")
    p.add_argument("--save_dir", type=str, default="runs/patchcls", help="Directory to save outputs.")
    p.add_argument("--run_name", type=str, default="", help="Optional run name subfolder.")
    p.add_argument("--use_wandb", type=int, default=0, choices=[0, 1])
    p.add_argument("--wandb_project", type=str, default="patchnepa-finetune")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_group", type=str, default="")
    p.add_argument("--wandb_tags", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    # Dataset
    p.add_argument("--cache_root", type=str, default="", help="Cache root (preprocessed npz tree).")
    p.add_argument(
        "--data_format",
        type=str,
        default="npz",
        choices=["npz", "scan_h5"],
        help="Input backend: npz cache or ScanObjectNN h5 direct.",
    )
    p.add_argument("--scan_h5_root", type=str, default="", help="ScanObjectNN h5 root for --data_format=scan_h5.")
    p.add_argument(
        "--scan_variant",
        type=str,
        default="auto",
        choices=["auto", "obj_bg", "obj_only", "pb_t50_rs"],
        help="Variant selector for scan_h5 route.",
    )
    p.add_argument("--split_train", type=str, default="train")
    p.add_argument("--split_test", type=str, default="test")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--val_seed", type=int, default=0)
    p.add_argument(
        "--val_split_mode",
        type=str,
        default="file",
        choices=["file", "group_auto", "group_scanobjectnn", "pointmae"],
        help=(
            "Validation split mode from TRAIN. "
            "group_auto resolves to ScanObjectNN group split when cache_root contains 'scanobjectnn'. "
            "pointmae uses official train for training and official test for validation (Point-MAE legacy test-as-val). "
            "Default=file to match current Point-MAE strict policy (train->train/val split)."
        ),
    )

    p.add_argument("--n_point", type=int, default=1024)
    p.add_argument("--pt_sample_mode_train", type=str, default="random", choices=["random", "fps"])
    p.add_argument("--pt_sample_mode_eval", type=str, default="fps", choices=["random", "fps"])
    p.add_argument("--use_normals", type=int, default=0)

    # Aug
    p.add_argument("--aug_preset", type=str, default="pointmae", choices=["none", "default", "strong", "pointmae"])
    p.add_argument("--aug_prob", type=float, default=0.5)
    p.add_argument("--aug_scale_min", type=float, default=0.9)
    p.add_argument("--aug_scale_max", type=float, default=1.1)
    p.add_argument("--aug_shift_std", type=float, default=0.02)
    p.add_argument("--aug_jitter_std", type=float, default=0.005)
    p.add_argument("--aug_jitter_clip", type=float, default=0.02)
    p.add_argument("--aug_rot_axis", type=str, default="y", choices=["x", "y", "z"])
    p.add_argument("--aug_rot_deg", type=float, default=180.0)
    p.add_argument("--aug_dropout_ratio", type=float, default=0.0)
    p.add_argument("--aug_dropout_prob", type=float, default=0.0)
    p.add_argument(
        "--pointmae_exact_aug",
        type=int,
        default=1,
        choices=[0, 1],
        help="Use Point-MAE exact ScaleAndTranslate (per-axis scale) when aug_preset=pointmae.",
    )
    p.add_argument(
        "--aug_eval",
        type=int,
        default=1,
        help="Apply augmentation during eval (TTA). Default on by policy.",
    )
    p.add_argument(
        "--mc_eval_k_test",
        type=int,
        default=10,
        help="MC crops at test time (K>1 averages logits). Default 10 by policy.",
    )

    # Model
    p.add_argument(
        "--patch_embed",
        type=str,
        default="fps_knn",
        choices=["fps_knn", "serial"],
        help="Patch grouping backend: fps_knn (default) or serial (Morton/chunk).",
    )
    p.add_argument(
        "--patch_local_encoder",
        type=str,
        default="pointmae_conv",
        choices=["mlp", "pointmae_conv"],
        help="Local patch encoder for fps_knn patch embed.",
    )
    p.add_argument(
        "--patch_fps_random_start",
        type=int,
        default=1,
        choices=[0, 1],
        help="Use random FPS start for fps_knn patch embed.",
    )
    p.add_argument("--num_groups", type=int, default=64)
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--center_mode", type=str, default="fps", choices=["fps", "first"])
    p.add_argument(
        "--serial_order",
        type=str,
        default="morton",
    )
    p.add_argument("--serial_bits", type=int, default=10)
    p.add_argument("--serial_shuffle_within_patch", type=int, default=0, choices=[0, 1])
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--drop_path_rate", type=float, default=0.1)
    p.add_argument(
        "--model_source",
        type=str,
        default="patchcls",
        choices=["patchcls", "pointmae", "patchnepa"],
        help="Backbone source: patchcls(custom), pointmae(backbone-parity), or patchnepa(direct pretrain->ft).",
    )
    p.add_argument("--backbone_mode", type=str, default="nepa2d", choices=["nepa2d", "vanilla", "pointmae"])
    p.add_argument("--qk_norm", type=int, default=1, choices=[0, 1])
    p.add_argument("--qk_norm_affine", type=int, default=0, choices=[0, 1])
    p.add_argument("--qk_norm_bias", type=int, default=0, choices=[0, 1])
    p.add_argument("--layerscale_value", type=float, default=1e-5)
    p.add_argument("--rope_theta", type=float, default=100.0, help="<=0 disables RoPE in nepa2d blocks.")
    p.add_argument("--rope_prefix_tokens", type=int, default=1)
    p.add_argument("--use_gated_mlp", type=int, default=0, choices=[0, 1], help="Enable gated MLP path.")
    p.add_argument("--hidden_act", type=str, default="gelu", choices=["gelu", "silu"], help="MLP activation.")
    p.add_argument("--pooling", type=str, default="cls_max", choices=["mean", "mean_q", "cls", "cls_max", "sep"])
    p.add_argument(
        "--pool_mode",
        type=str,
        default="",
        choices=["", "mean", "mean_q", "cls", "cls_max", "sep"],
        help="Alias of --pooling for PatchNEPA compatibility.",
    )
    p.add_argument("--pos_mode", type=str, default="center_mlp", choices=["learned", "center_mlp"])
    p.add_argument("--use_ray_patch", type=int, default=0, choices=[0, 1])
    p.add_argument("--n_ray", type=int, default=256)
    p.add_argument("--ray_sample_mode_train", type=str, default="random", choices=["random", "first"])
    p.add_argument("--ray_sample_mode_eval", type=str, default="first", choices=["random", "first"])
    p.add_argument("--ray_pool_mode", type=str, default="max", choices=["max", "mean"])
    p.add_argument("--ray_fuse_mode", type=str, default="concat", choices=["concat", "add"])
    p.add_argument("--ray_hidden_dim", type=int, default=128)
    p.add_argument("--ray_miss_t", type=float, default=4.0)
    p.add_argument("--ray_hit_threshold", type=float, default=0.5)
    p.add_argument("--head_mode", type=str, default="auto", choices=["auto", "linear", "pointmae_mlp"])
    p.add_argument("--head_hidden_dim", type=int, default=256)
    p.add_argument("--head_dropout", type=float, default=0.5)
    p.add_argument("--init_mode", type=str, default="default", choices=["default", "pointmae"])
    p.add_argument("--is_causal", type=int, default=0)
    p.add_argument("--patchnepa_ft_mode", type=str, default="qa_zeroa", choices=["qa_zeroa", "q_only"])
    p.add_argument(
        "--patchnepa_cls_token_source",
        type=str,
        default="last_q",
        choices=["bos", "last_q", "eos"],
        help="Classification anchor token source for PatchNEPA direct FT.",
    )
    p.add_argument(
        "--patchnepa_freeze_patch_embed",
        type=int,
        default=1,
        choices=[0, 1],
        help="Freeze patch embedding during PatchNEPA direct FT.",
    )
    p.add_argument(
        "--patch_order_mode",
        type=str,
        default="",
        help=(
            "Patch order mode(s) for PatchNEPA. Comma-separated list allowed. "
            "Empty keeps checkpoint/default setting."
        ),
    )
    p.add_argument(
        "--patch_order_schedule",
        type=str,
        default="fixed",
        choices=["fixed", "epoch", "batch", "batch_random", "sample"],
        help="How to switch patch_order_mode when multiple modes are provided.",
    )

    # Optim
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument(
        "--batch_mode",
        type=str,
        default="global",
        choices=["global", "per_proc"],
        help=(
            "Batch interpretation. "
            "'global' keeps total batch fixed across DDP world-size (per-proc=batch/world_size). "
            "'per_proc' uses --batch as per-process batch."
        ),
    )
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument(
        "--llrd_scheduler",
        type=str,
        default="static",
        choices=["static", "llrd_cosine", "llrd_cosine_warmup"],
        help=(
            "LLRD scheduling mode for model_source=patchnepa. "
            "'static' keeps per-layer LR fixed (current behavior). "
            "'llrd_cosine*' uses step-wise LayerLambdaLR (2D-NEPA style)."
        ),
    )
    p.add_argument(
        "--llrd_mode",
        type=str,
        default="linear",
        choices=["linear", "exp"],
        help="Layer-wise scale mode when llrd_scheduler=llrd_cosine*: linear(shallow->deep) or exp(legacy llrd**depth).",
    )
    p.add_argument("--warmup_epochs", type=float, default=10.0)
    p.add_argument("--warmup_start_factor", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--llrd_start", type=float, default=1.0, help="Layer-wise LR decay start scale (PatchNEPA).")
    p.add_argument("--llrd_end", type=float, default=1.0, help="Layer-wise LR decay end scale (PatchNEPA).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--allow_scan_uniscale_v2",
        type=int,
        default=0,
        choices=[0, 1],
        help="Safety guard: disallow scanobjectnn_*_v2 caches unless explicitly set to 1.",
    )


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _init_wandb(args: argparse.Namespace, save_dir: Path, accelerator: Accelerator) -> Any:
    if int(args.use_wandb) != 1:
        return None
    if not accelerator.is_main_process:
        return None
    try:
        import wandb  # type: ignore
    except Exception as e:
        print(f"[wandb] disabled: import failed ({e})")
        return None

    tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]
    run_name = str(args.wandb_run_name).strip() or str(args.run_name) or save_dir.name
    group = str(args.wandb_group).strip() or None
    entity = str(args.wandb_entity).strip() or None
    mode = str(args.wandb_mode).strip()

    cfg = dict(vars(args))
    cfg["save_dir_resolved"] = str(save_dir)
    try:
        run = wandb.init(
            project=str(args.wandb_project),
            entity=entity,
            name=run_name,
            group=group,
            tags=tags if tags else None,
            mode=mode,
            config=cfg,
        )
        print(
            f"[wandb] enabled project={args.wandb_project} run={run_name} "
            f"group={group or '-'} mode={mode}"
        )
        return run
    except Exception as e:
        print(f"[wandb] disabled: init failed ({e})")
        return None


def _stratified_split_indices(labels: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    labels = labels.astype(np.int64, copy=False)
    train_idx = []
    val_idx = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        idx = idx.copy()
        rng.shuffle(idx)
        n = len(idx)
        n_val = int(n * float(val_ratio))
        # Match modelnet40_index.stratified_train_val_split behavior.
        if n_val <= 0 and n >= 2:
            n_val = 1
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    train_idx = np.concatenate(train_idx, axis=0) if train_idx else np.zeros((0,), dtype=np.int64)
    val_idx = np.concatenate(val_idx, axis=0) if val_idx else np.zeros((0,), dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _adapt_legacy_querynepa_to_patchcls_nepa2d(
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Map legacy QueryNepa backbone weights to patchcls-nepa2d naming.

    Supported source prefixes:
    - `backbone.layers.{i}.*`
    - `backbone.enc.layers.{i}.*`
    """
    out: Dict[str, torch.Tensor] = {}
    direct = 0
    mapped = 0

    # Direct copy for any exact key/shape match.
    for k, v in source_state.items():
        tv = target_state.get(k, None)
        if tv is not None and tuple(tv.shape) == tuple(v.shape):
            out[k] = v
            direct += 1

    def _assign(dst_k: str, src_v: torch.Tensor) -> None:
        nonlocal mapped
        tv = target_state.get(dst_k, None)
        if tv is not None and tuple(tv.shape) == tuple(src_v.shape):
            out[dst_k] = src_v
            mapped += 1

    for src_prefix in ("backbone.layers", "backbone.enc.layers"):
        for i in range(96):
            src_base = f"{src_prefix}.{i}"
            dst_base = f"backbone.blocks.{i}"
            w_qkv_k = f"{src_base}.self_attn.in_proj_weight"
            b_qkv_k = f"{src_base}.self_attn.in_proj_bias"
            if w_qkv_k not in source_state:
                continue

            w_qkv = source_state[w_qkv_k]
            b_qkv = source_state.get(b_qkv_k, None)

            if w_qkv.ndim == 2 and w_qkv.shape[0] % 3 == 0:
                wq, wk, wv = w_qkv.chunk(3, dim=0)
                _assign(f"{dst_base}.attn.query.weight", wq)
                _assign(f"{dst_base}.attn.key.weight", wk)
                _assign(f"{dst_base}.attn.value.weight", wv)
            if b_qkv is not None and b_qkv.ndim == 1 and b_qkv.shape[0] % 3 == 0:
                bq, bk, bv = b_qkv.chunk(3, dim=0)
                _assign(f"{dst_base}.attn.query.bias", bq)
                _assign(f"{dst_base}.attn.key.bias", bk)
                _assign(f"{dst_base}.attn.value.bias", bv)

            _assign(f"{dst_base}.attn.proj.weight", source_state[f"{src_base}.self_attn.out_proj.weight"])
            _assign(f"{dst_base}.attn.proj.bias", source_state[f"{src_base}.self_attn.out_proj.bias"])

            _assign(f"{dst_base}.mlp.up_proj.weight", source_state[f"{src_base}.linear1.weight"])
            _assign(f"{dst_base}.mlp.up_proj.bias", source_state[f"{src_base}.linear1.bias"])
            _assign(f"{dst_base}.mlp.fc2.weight", source_state[f"{src_base}.linear2.weight"])
            _assign(f"{dst_base}.mlp.fc2.bias", source_state[f"{src_base}.linear2.bias"])

            _assign(f"{dst_base}.layernorm_before.weight", source_state[f"{src_base}.norm1.weight"])
            _assign(f"{dst_base}.layernorm_before.bias", source_state[f"{src_base}.norm1.bias"])
            _assign(f"{dst_base}.layernorm_after.weight", source_state[f"{src_base}.norm2.weight"])
            _assign(f"{dst_base}.layernorm_after.bias", source_state[f"{src_base}.norm2.bias"])

    stats = {
        "direct": int(direct),
        "mapped": int(mapped),
        "total_to_load": int(len(out)),
        "src_total": int(len(source_state)),
        "dst_total": int(len(target_state)),
    }
    return out, stats


def _adapt_patchnepa_pretrain_to_patchcls(
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Map PatchNEPA-pretrain checkpoint keys into PatchCls classifier keys.

    Key intent:
    - carry over shared backbone/patch-embed weights,
    - map `bos_token` (pretrain) -> `cls_token` (classifier),
    - ignore pretrain-only heads/tokens safely.
    """
    out: Dict[str, torch.Tensor] = {}
    direct = 0
    mapped = 0

    # direct key/shape matches first
    for k, v in source_state.items():
        tv = target_state.get(k, None)
        if tv is not None and tuple(tv.shape) == tuple(v.shape):
            out[k] = v
            direct += 1

    # explicit remaps
    bos = source_state.get("bos_token", None)
    cls = target_state.get("cls_token", None)
    if bos is not None and cls is not None and tuple(bos.shape) == tuple(cls.shape):
        out["cls_token"] = bos
        mapped += 1

    stats = {
        "direct": int(direct),
        "mapped": int(mapped),
        "total_to_load": int(len(out)),
        "src_total": int(len(source_state)),
        "dst_total": int(len(target_state)),
    }
    return out, stats


def _adapt_patchnepa_pretrain_to_patchnepa_classifier(
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Map PatchNEPA pretrain keys to composition classifier (`core.*`)."""
    out: Dict[str, torch.Tensor] = {}
    direct = 0
    mapped = 0

    # direct key/shape matches first
    for k, v in source_state.items():
        tv = target_state.get(k, None)
        if tv is not None and tuple(tv.shape) == tuple(v.shape):
            out[k] = v
            direct += 1

    # pretrain key -> classifier core key
    for k, v in source_state.items():
        kk = f"core.{k}"
        tv = target_state.get(kk, None)
        if tv is not None and tuple(tv.shape) == tuple(v.shape):
            out[kk] = v
            mapped += 1

    stats = {
        "direct": int(direct),
        "mapped": int(mapped),
        "total_to_load": int(len(out)),
        "src_total": int(len(source_state)),
        "dst_total": int(len(target_state)),
    }
    return out, stats


def _as_bool01(v: object, default: bool = False) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return bool(v)


def _parse_patch_order_modes(raw: str) -> list[str]:
    txt = str(raw or "").strip()
    if not txt:
        return []
    if txt.lower().replace("-", "_").startswith("sample:"):
        return [txt]
    return [m.strip() for m in txt.split(",") if m.strip()]


def _select_patch_order_mode(
    modes: list[str],
    schedule: str,
    *,
    epoch: int,
    global_step: int,
    seed: int,
) -> str:
    sched = str(schedule).strip().lower()
    if len(modes) == 1 or sched == "fixed":
        return modes[0]
    if sched == "sample":
        if len(modes) == 1 and str(modes[0]).lower().replace("-", "_").startswith("sample:"):
            return modes[0]
        return "sample:" + ",".join(modes)
    if sched == "epoch":
        return modes[int(epoch) % len(modes)]
    if sched == "batch":
        return modes[int(global_step) % len(modes)]
    if sched == "batch_random":
        rng = np.random.RandomState(int(seed) + int(global_step))
        return modes[int(rng.randint(0, len(modes)))]
    raise ValueError(f"unknown patch_order_schedule={schedule}")


def _infer_answer_in_dim_from_state(state: Optional[Dict[str, torch.Tensor]]) -> Optional[int]:
    if not isinstance(state, dict):
        return None
    first_layers: list[tuple[int, int]] = []
    for k, v in state.items():
        if (not torch.is_tensor(v)) or v.ndim != 2 or (not k.endswith(".weight")):
            continue
        if "answer_embed.mlp." not in k:
            continue
        tail = k.split("answer_embed.mlp.", 1)[1]
        layer_tok = tail.split(".", 1)[0]
        try:
            layer_idx = int(layer_tok)
        except Exception:
            continue
        # Linear weight shape is [out_dim, in_dim].
        first_layers.append((layer_idx, int(v.shape[1])))
    if not first_layers:
        return None
    first_layers.sort(key=lambda x: x[0])
    in_dim = int(first_layers[0][1])
    return in_dim if in_dim > 0 else None


def _patchnepa_kwargs_from_ckpt(
    ckpt_args: Dict[str, object],
    args: argparse.Namespace,
    pretrain_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, object]:
    """Build PatchTransformerNepa kwargs from pretrain ckpt args, with safe fallbacks."""
    c = ckpt_args or {}
    kw: Dict[str, object] = {
        # patchify
        "patch_embed": str(c.get("patch_embed", args.patch_embed)),
        "patch_local_encoder": str(c.get("patch_local_encoder", "pointmae_conv")),
        "patch_fps_random_start": _as_bool01(c.get("patch_fps_random_start", 1)),
        "n_point": int(c.get("n_point", args.n_point)),
        "group_size": int(c.get("group_size", args.group_size)),
        "num_groups": int(c.get("num_groups", args.num_groups)),
        "serial_order": str(c.get("serial_order", args.serial_order)),
        "serial_bits": int(c.get("serial_bits", args.serial_bits)),
        "serial_shuffle_within_patch": int(c.get("serial_shuffle_within_patch", args.serial_shuffle_within_patch)),
        "patch_order_mode": str(c.get("patch_order_mode", "none")),
        "use_normals": _as_bool01(c.get("use_normals", args.use_normals)),
        # transformer
        "d_model": int(c.get("d_model", args.d_model)),
        "n_layers": int(c.get("n_layers", args.n_layers)),
        "n_heads": int(c.get("n_heads", args.n_heads)),
        "mlp_ratio": float(c.get("mlp_ratio", args.mlp_ratio)),
        "dropout": float(c.get("dropout", args.dropout)),
        "drop_path_rate": float(c.get("drop_path_rate", args.drop_path_rate)),
        "qk_norm": int(_as_bool01(c.get("qk_norm", args.qk_norm))),
        "qk_norm_affine": int(_as_bool01(c.get("qk_norm_affine", args.qk_norm_affine))),
        "qk_norm_bias": int(_as_bool01(c.get("qk_norm_bias", args.qk_norm_bias))),
        "layerscale_value": float(c.get("layerscale_value", args.layerscale_value)),
        "rope_theta": float(c.get("rope_theta", args.rope_theta)),
        "use_gated_mlp": int(_as_bool01(c.get("use_gated_mlp", args.use_gated_mlp))),
        "hidden_act": str(c.get("hidden_act", args.hidden_act)),
        "backbone_mode": str(c.get("backbone_mode", args.backbone_mode)),
        # Q/A
        "qa_tokens": int(c.get("qa_tokens", 1)),
        "qa_layout": str(c.get("qa_layout", "split_sep")),
        "qa_sep_token": _as_bool01(c.get("qa_sep_token", 1)),
        "qa_fuse": str(c.get("qa_fuse", "add")),
        "use_pt_dist": _as_bool01(c.get("use_pt_dist", 1)),
        "use_pt_grad": _as_bool01(c.get("use_pt_grad", 0)),
        "answer_mlp_layers": int(c.get("answer_mlp_layers", 2)),
        "answer_pool": str(c.get("answer_pool", "max")),
        # embeddings / arch
        "max_len": int(c.get("max_len", 4096)),
        "nepa2d_pos": _as_bool01(c.get("nepa2d_pos", 1)),
        "type_specific_pos": _as_bool01(c.get("type_specific_pos", 0)),
        "type_pos_max_len": int(c.get("type_pos_max_len", 4096)),
        "pos_mode": str(c.get("pos_mode", "center_mlp")),
        "encdec_arch": _as_bool01(c.get("encdec_arch", 0)),
        # ray binding
        "use_ray_patch": _as_bool01(c.get("use_ray_patch", args.use_ray_patch)),
        "include_ray_normal": _as_bool01(c.get("include_ray_normal", 1)),
        "include_ray_unc": _as_bool01(c.get("include_ray_unc", 0)),
        "use_ray_origin": _as_bool01(c.get("ray_use_origin", 0)),
        "ray_assign_mode": str(c.get("ray_assign_mode", "proxy_sphere")),
        "ray_proxy_radius_scale": float(c.get("ray_proxy_radius_scale", 1.05)),
        "ray_pool_mode": str(c.get("ray_pool_mode", "amax")),
    }
    # Keep answer feature dimensionality consistent with pretrain.
    try:
        ckpt_answer_in_dim = int(c.get("answer_in_dim", 0) or 0)
    except Exception:
        ckpt_answer_in_dim = 0
    if ckpt_answer_in_dim > 0:
        kw["answer_in_dim"] = ckpt_answer_in_dim
    else:
        inferred_answer_in_dim = _infer_answer_in_dim_from_state(pretrain_state)
        if inferred_answer_in_dim is not None:
            kw["answer_in_dim"] = int(inferred_answer_in_dim)
    # FT-time override for ray usage is allowed.
    kw["use_ray_patch"] = bool(args.use_ray_patch)
    # Optional FT-time override for patch order mode.
    ft_patch_modes = _parse_patch_order_modes(str(getattr(args, "patch_order_mode", "")))
    if ft_patch_modes:
        kw["patch_order_mode"] = str(
            _select_patch_order_mode(
                ft_patch_modes,
                str(getattr(args, "patch_order_schedule", "fixed")),
                epoch=0,
                global_step=0,
                seed=int(getattr(args, "seed", 0)),
            )
        )
    return kw



def _patchnepa_backbone_depth_info(model: nn.Module) -> Tuple[int, int]:
    core = getattr(model, "core", None)
    bb = getattr(core, "backbone", None)
    if bb is None:
        return 0, 1
    n_enc = len(getattr(bb, "encoder_layers", []))
    n_dec = len(getattr(bb, "decoder_layers", []))
    n_blocks = len(getattr(bb, "blocks", []))
    n_layers = len(getattr(bb, "layers", []))
    depth = max(int(n_blocks), int(n_layers), int(n_enc + n_dec), 1)
    return int(n_enc), int(depth)


def _patchnepa_param_depth(name: str, *, n_enc: int, backbone_depth: int) -> int:
    # Embedding/front-end parameters are depth 0.
    if name.startswith("cls_head."):
        return int(backbone_depth + 1)
    if name.startswith("core.backbone.blocks."):
        try:
            return int(name.split(".")[3]) + 1
        except Exception:
            return int(backbone_depth)
    if name.startswith("core.backbone.layers."):
        try:
            return int(name.split(".")[3]) + 1
        except Exception:
            return int(backbone_depth)
    if name.startswith("core.backbone.encoder_layers."):
        try:
            return int(name.split(".")[3]) + 1
        except Exception:
            return int(backbone_depth)
    if name.startswith("core.backbone.decoder_layers."):
        try:
            return int(n_enc) + int(name.split(".")[3]) + 1
        except Exception:
            return int(backbone_depth)
    if (
        name.startswith("core.backbone.final_ln")
        or name.startswith("core.backbone.enc_ln")
        or name.startswith("core.backbone.dec_ln")
    ):
        return int(backbone_depth)
    return 0


def _is_no_decay_param(name: str, p: torch.Tensor) -> bool:
    lname = name.lower()
    if name.endswith(".bias") or p.ndim <= 1:
        return True
    if ("norm" in lname) or ("bn" in lname) or ("emb" in lname):
        return True
    return False


def _build_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
) -> Tuple[optim.Optimizer, Dict[str, float]]:
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named_params:
        raise RuntimeError("No trainable parameters found for optimizer.")

    # Keep non-PatchNEPA behavior simple.
    if str(args.model_source) != "patchnepa":
        optimizer = optim.AdamW([p for _, p in named_params], lr=args.lr, weight_decay=args.weight_decay)
        stats = {"n_groups": 1.0, "lr_min": float(args.lr), "lr_max": float(args.lr)}
        return optimizer, stats

    llrd_start = float(args.llrd_start)
    llrd_end = float(args.llrd_end)
    if llrd_start <= 0.0 or llrd_end <= 0.0:
        llrd_start, llrd_end = 1.0, 1.0
    if llrd_end < llrd_start:
        llrd_start, llrd_end = llrd_end, llrd_start

    n_enc, backbone_depth = _patchnepa_backbone_depth_info(model)
    max_depth = int(backbone_depth + 1)
    groups: Dict[Tuple[float, float], Dict[str, object]] = {}
    llrd_sched_mode = str(getattr(args, "llrd_scheduler", "static"))
    use_dynamic_llrd = llrd_sched_mode in {"llrd_cosine", "llrd_cosine_warmup"}

    for name, p in named_params:
        depth = _patchnepa_param_depth(name, n_enc=n_enc, backbone_depth=backbone_depth)
        depth = max(0, min(int(depth), max_depth))
        if use_dynamic_llrd:
            llrd_scale = float(max_depth - depth)
            scale = 1.0
        else:
            llrd_scale = 0.0
            scale = float(llrd_start + (llrd_end - llrd_start) * (float(depth) / float(max_depth)))
        wd = 0.0 if _is_no_decay_param(name, p) else float(args.weight_decay)
        key = (round(scale, 8), wd) if not use_dynamic_llrd else (round(llrd_scale, 8), wd)
        g = groups.get(key)
        if g is None:
            g = {"params": [], "weight_decay": wd}
            if use_dynamic_llrd:
                g["lr"] = float(args.lr)
                g["llrd"] = float(llrd_start)
                g["llrd_scale"] = float(llrd_scale)
                g["llrd_mode"] = str(getattr(args, "llrd_mode", "linear"))
                g["llrd_scale_max"] = float(max_depth)
            else:
                g["lr"] = float(args.lr) * scale
            groups[key] = g
        if use_dynamic_llrd:
            g["llrd"] = float(llrd_start)
            g["llrd_scale"] = float(llrd_scale)
            g["llrd_mode"] = str(getattr(args, "llrd_mode", "linear"))
            g["llrd_scale_max"] = float(max_depth)
        g["params"].append(p)

    param_groups = list(groups.values())
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    lrs = [float(g["lr"]) for g in param_groups]
    stats = {
        "n_groups": float(len(param_groups)),
        "lr_min": float(min(lrs) if lrs else float(args.lr)),
        "lr_max": float(max(lrs) if lrs else float(args.lr)),
    }
    return optimizer, stats


def _resolve_scan_h5_files(scan_h5_root: str, scan_variant: str) -> Tuple[Path, Path]:
    root = Path(scan_h5_root)
    if not root.exists():
        raise FileNotFoundError(f"scan_h5_root not found: {scan_h5_root}")

    variant = scan_variant
    if variant == "auto":
        name_l = str(root).lower()
        if "nobg" in name_l or "obj_only" in name_l:
            variant = "obj_only"
        elif "pb_t50_rs" in name_l or "scale75" in name_l:
            variant = "pb_t50_rs"
        else:
            variant = "obj_bg"

    if variant == "pb_t50_rs":
        tr = root / "training_objectdataset_augmentedrot_scale75.h5"
        te = root / "test_objectdataset_augmentedrot_scale75.h5"
    else:
        tr = root / "training_objectdataset.h5"
        te = root / "test_objectdataset.h5"

    if not tr.exists() or not te.exists():
        raise FileNotFoundError(f"missing h5 files: train={tr} test={te}")
    return tr, te


@torch.no_grad()
def evaluate_local(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_normals: bool,
    use_ray_patch: bool,
    model_source: str = "patchcls",
) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    for batch in loader:
        xyz = batch["xyz"].to(device)
        label = batch["label"].to(device)
        normal = batch.get("normal", None)
        if use_normals and normal is not None:
            normal = normal.to(device)
        ray_o = batch.get("ray_o", None)
        ray_d = batch.get("ray_d", None)
        ray_t = batch.get("ray_t", None)
        ray_hit = batch.get("ray_hit", None)
        if use_ray_patch:
            if (ray_o is None) or (ray_d is None):
                raise ValueError("use_ray_patch=True but batch does not contain ray_o/ray_d tensors")
            ray_o = ray_o.to(device)
            ray_d = ray_d.to(device)
            if model_source != "patchnepa":
                if ray_t is not None:
                    ray_t = ray_t.to(device)
                if ray_hit is not None:
                    ray_hit = ray_hit.to(device)
            else:
                # Query-only ray protocol for PatchNEPA classification:
                # never pass answer-side ray signals.
                ray_t = None
                ray_hit = None

        # MC eval: xyz is (B,K,N,3)
        if xyz.dim() == 4:
            B, K, N, C = xyz.shape
            xyz2 = xyz.reshape(B * K, N, C)
            if use_normals and normal is not None and normal.dim() == 4:
                normal2 = normal.reshape(B * K, N, 3)
            else:
                normal2 = None
            if use_ray_patch:
                # Rays are sampled once per shape and shared across MC crops.
                ray_o2 = ray_o.repeat_interleave(K, dim=0)
                ray_d2 = ray_d.repeat_interleave(K, dim=0)
                if model_source != "patchnepa":
                    ray_t2 = ray_t.repeat_interleave(K, dim=0) if ray_t is not None else None
                    ray_hit2 = ray_hit.repeat_interleave(K, dim=0) if ray_hit is not None else None
                else:
                    ray_t2 = None
                    ray_hit2 = None
            else:
                ray_o2 = ray_d2 = ray_t2 = ray_hit2 = None
            logits = model(xyz2, normal2, ray_o=ray_o2, ray_d=ray_d2, ray_t=ray_t2, ray_hit=ray_hit2)
            logits = logits.reshape(B, K, -1).mean(dim=1)
        else:
            logits = model(xyz, normal, ray_o=ray_o, ray_d=ray_d, ray_t=ray_t, ray_hit=ray_hit)

        loss = F.cross_entropy(logits, label)
        preds = logits.argmax(dim=-1)
        correct += (preds == label).sum().item()
        total += label.numel()
        loss_sum += loss.item() * label.size(0)

    acc = float(correct) / float(max(1, total))
    loss_avg = float(loss_sum) / float(max(1, total))
    return {"acc": acc, "loss": loss_avg}


def main() -> None:
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    # Guard against ineffective MC evaluation setup:
    # fps sampling is deterministic; with aug_eval=0, K>1 repeats identical crops.
    if int(args.mc_eval_k_test) > 1 and str(args.pt_sample_mode_eval) == "fps" and (not bool(args.aug_eval)):
        raise ValueError(
            "Invalid eval config: mc_eval_k_test>1 with pt_sample_mode_eval=fps and aug_eval=0. "
            "This produces identical crops (no real voting). "
            "Use aug_eval=1, or pt_sample_mode_eval=random, or set mc_eval_k_test=1."
        )

    _set_seed(args.seed)
    patch_order_modes = _parse_patch_order_modes(str(getattr(args, "patch_order_mode", "")))
    patch_order_schedule = str(getattr(args, "patch_order_schedule", "fixed")).strip().lower()
    active_patch_order_mode: Optional[str]
    if patch_order_modes:
        active_patch_order_mode = _select_patch_order_mode(
            patch_order_modes,
            patch_order_schedule,
            epoch=0,
            global_step=0,
            seed=int(args.seed),
        )
    else:
        active_patch_order_mode = None
    ddp_find_unused = bool(args.model_source == "patchnepa" and str(args.patchnepa_ft_mode) == "q_only")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=ddp_find_unused)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # Resolve save path
    save_dir = Path(args.save_dir)
    if args.run_name:
        save_dir = save_dir / args.run_name
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "checkpoints").mkdir(exist_ok=True)
        with open(save_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    accelerator.wait_for_everyone()
    wandb_run = _init_wandb(args, save_dir, accelerator)

    if args.data_format == "npz":
        if not args.cache_root:
            raise ValueError("--cache_root is required when --data_format=npz")
        cache_root = args.cache_root
        cache_root_abs_l = os.path.abspath(cache_root).lower()
        if ("scanobjectnn_" in cache_root_abs_l) and ("_v2" in cache_root_abs_l) and (int(args.allow_scan_uniscale_v2) != 1):
            raise ValueError(
                f"cache_root={cache_root} is a uniscale v2 cache and is disallowed by policy. "
                "Use scanobjectnn_*_v3_nonorm, or set --allow_scan_uniscale_v2 1 for intentional legacy reruns."
            )
        if "scanobjectnn_main_split_v2" in cache_root_abs_l:
            raise ValueError(
                f"cache_root={cache_root} is disallowed for benchmark runs (main_split deprecated). "
                "Use variant cache roots: scanobjectnn_obj_bg_v3_nonorm | scanobjectnn_obj_only_v3_nonorm | scanobjectnn_pb_t50_rs_v3_nonorm."
            )
        train_paths_full = list_npz(cache_root, split=args.split_train)
        test_paths = list_npz(cache_root, split=args.split_test)

        # Build label map from the available class folders.
        label_map = build_label_map(train_paths_full + test_paths)
        num_classes = max(label_map.values()) + 1

        resolved_val_split_mode = str(args.val_split_mode)
        if args.val_split_mode == "pointmae":
            # Point-MAE style: train on official train split, select/check by official test split.
            train_paths = train_paths_full
            val_paths = test_paths
            resolved_val_split_mode = "pointmae(test-as-val)"
        else:
            val_group_key_fn = None
            if args.val_split_mode == "group_scanobjectnn":
                val_group_key_fn = scanobjectnn_group_key
            elif args.val_split_mode == "group_auto":
                if "scanobjectnn" in cache_root_abs_l:
                    val_group_key_fn = scanobjectnn_group_key
                    resolved_val_split_mode = "group_scanobjectnn(auto)"
                else:
                    resolved_val_split_mode = "file(auto-fallback)"

            train_paths, val_paths = stratified_train_val_split(
                train_paths_full,
                val_ratio=args.val_ratio,
                seed=args.val_seed,
                group_key_fn=val_group_key_fn,
            )
    else:
        if not args.scan_h5_root:
            raise ValueError("--scan_h5_root is required when --data_format=scan_h5")
        if bool(args.use_ray_patch):
            raise ValueError("--use_ray_patch is not supported with --data_format=scan_h5")
        h5_train, h5_test = _resolve_scan_h5_files(args.scan_h5_root, args.scan_variant)
        tr_points, tr_labels = load_scanobjectnn_h5_arrays(str(h5_train))
        te_points, te_labels = load_scanobjectnn_h5_arrays(str(h5_test))
        if args.val_split_mode == "pointmae":
            tr_idx = np.arange(tr_labels.shape[0], dtype=np.int64)
            va_points, va_labels = te_points, te_labels
            resolved_val_split_mode = "pointmae(test-as-val)"
        else:
            tr_idx, va_idx = _stratified_split_indices(tr_labels, args.val_ratio, args.val_seed)
            va_points, va_labels = tr_points[va_idx], tr_labels[va_idx]
            resolved_val_split_mode = "stratified_label(h5)"
        num_classes = int(max(tr_labels.max(initial=0), te_labels.max(initial=0))) + 1

    # Aug preset
    if args.aug_preset == "none":
        aug_cfg = PointAugConfig(prob=0.0)
        aug_train = False
    elif args.aug_preset == "pointmae":
        # Point-MAE ScanObjectNN transform:
        # PointcloudScaleAndTranslate(scale=[2/3, 3/2], translate=0.2)
        # No jitter/rotation/dropout in this preset.
        aug_cfg = PointAugConfig(
            prob=1.0,
            scale_min=2.0 / 3.0,
            scale_max=3.0 / 2.0,
            shift_std=0.2,
            jitter_std=0.0,
            jitter_clip=0.0,
            rot_axis="y",
            rot_deg=0.0,
            dropout_ratio=0.0,
            dropout_prob=0.0,
            pointmae_exact=bool(args.pointmae_exact_aug),
        )
        aug_train = True
    else:
        # default / strong
        prob = args.aug_prob if args.aug_preset == "default" else max(0.8, args.aug_prob)
        aug_cfg = PointAugConfig(
            prob=prob,
            scale_min=args.aug_scale_min,
            scale_max=args.aug_scale_max,
            shift_std=args.aug_shift_std,
            jitter_std=args.aug_jitter_std,
            jitter_clip=args.aug_jitter_clip,
            rot_axis=args.aug_rot_axis,
            rot_deg=args.aug_rot_deg,
            dropout_ratio=args.aug_dropout_ratio,
            dropout_prob=args.aug_dropout_prob,
        )
        aug_train = True

    use_normals = bool(args.use_normals)
    if args.data_format == "npz":
        train_set = PatchClsPointDataset(
            train_paths,
            cache_root=cache_root,
            label_map=label_map,
            n_point=args.n_point,
            sample_mode=args.pt_sample_mode_train,
            use_normals=use_normals,
            aug=aug_train,
            aug_cfg=aug_cfg,
            rng_seed=args.seed,
            use_ray_patch=bool(args.use_ray_patch),
            n_ray=args.n_ray,
            ray_sample_mode=args.ray_sample_mode_train,
            mc_eval_k=1,
            aug_eval=False,
            deterministic_eval_sampling=False,
        )
        val_set = PatchClsPointDataset(
            val_paths,
            cache_root=cache_root,
            label_map=label_map,
            n_point=args.n_point,
            sample_mode=args.pt_sample_mode_eval,
            use_normals=use_normals,
            aug=False,
            aug_cfg=aug_cfg,
            rng_seed=args.seed + 123,
            use_ray_patch=bool(args.use_ray_patch),
            n_ray=args.n_ray,
            ray_sample_mode=args.ray_sample_mode_eval,
            mc_eval_k=1,
            aug_eval=False,
            deterministic_eval_sampling=True,
        )
        test_set = PatchClsPointDataset(
            test_paths,
            cache_root=cache_root,
            label_map=label_map,
            n_point=args.n_point,
            sample_mode=args.pt_sample_mode_eval,
            use_normals=use_normals,
            aug=bool(args.aug_eval),
            aug_cfg=aug_cfg,
            rng_seed=args.seed + 456,
            use_ray_patch=bool(args.use_ray_patch),
            n_ray=args.n_ray,
            ray_sample_mode=args.ray_sample_mode_eval,
            mc_eval_k=max(1, args.mc_eval_k_test),
            aug_eval=bool(args.aug_eval),
            deterministic_eval_sampling=True,
        )
    else:
        train_set = PatchClsArrayDataset(
            tr_points[tr_idx],
            tr_labels[tr_idx],
            n_point=args.n_point,
            sample_mode=args.pt_sample_mode_train,
            aug=aug_train,
            aug_cfg=aug_cfg,
            rng_seed=args.seed,
            mc_eval_k=1,
            aug_eval=False,
            deterministic_eval_sampling=False,
        )
        val_set = PatchClsArrayDataset(
            va_points,
            va_labels,
            n_point=args.n_point,
            sample_mode=args.pt_sample_mode_eval,
            aug=False,
            aug_cfg=aug_cfg,
            rng_seed=args.seed + 123,
            mc_eval_k=1,
            aug_eval=False,
            deterministic_eval_sampling=True,
        )
        test_set = PatchClsArrayDataset(
            te_points,
            te_labels,
            n_point=args.n_point,
            sample_mode=args.pt_sample_mode_eval,
            aug=bool(args.aug_eval),
            aug_cfg=aug_cfg,
            rng_seed=args.seed + 456,
            mc_eval_k=max(1, args.mc_eval_k_test),
            aug_eval=bool(args.aug_eval),
            deterministic_eval_sampling=True,
        )

    world_size = max(1, int(accelerator.num_processes))
    eff_batch = int(args.batch)
    if args.batch_mode == "global" and world_size > 1:
        if eff_batch % world_size != 0:
            raise ValueError(
                f"--batch {eff_batch} must be divisible by world_size={world_size} when --batch_mode=global"
            )
        eff_batch = eff_batch // world_size

    train_loader = DataLoader(
        train_set,
        batch_size=eff_batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=eff_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=eff_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Optional checkpoint preload (used by multiple model-source branches).
    ckpt_obj: Optional[Dict[str, object]] = None
    pretrain_state: Optional[Dict[str, torch.Tensor]] = None
    pretrain_args: Dict[str, object] = {}
    if args.ckpt:
        ckpt_obj = torch.load(args.ckpt, map_location="cpu")
        state_source = "model"
        if isinstance(ckpt_obj, dict):
            use_ema = bool(int(args.ckpt_use_ema))
            if use_ema and isinstance(ckpt_obj.get("model_ema", None), dict):
                pretrain_state = ckpt_obj["model_ema"]
                state_source = "model_ema"
            else:
                pretrain_state = ckpt_obj.get("model", ckpt_obj)
            if isinstance(ckpt_obj.get("args", None), dict):
                pretrain_args = ckpt_obj["args"]  # type: ignore[index]
        else:
            pretrain_state = ckpt_obj
        if accelerator.is_main_process:
            print(f"[ckpt] source={state_source} ckpt_use_ema={int(args.ckpt_use_ema)}")

    # Model
    effective_pooling = str(args.pool_mode).strip() or str(args.pooling)
    if args.model_source == "pointmae":
        if use_normals:
            raise ValueError("--use_normals is not supported with --model_source=pointmae")
        if bool(args.use_ray_patch):
            raise ValueError("--use_ray_patch is not supported with --model_source=pointmae")
        model = PointMAEPatchClassifier(
            num_classes=num_classes,
            trans_dim=args.d_model,
            depth=args.n_layers,
            drop_path_rate=args.drop_path_rate,
            num_heads=args.n_heads,
            group_size=args.group_size,
            num_groups=args.num_groups,
            encoder_dims=args.d_model,
            init_mode=args.init_mode,
        )
    elif args.model_source == "patchnepa":
        # Allow strict A/B with identical FT recipe by supporting patchnepa scratch init
        # when --ckpt is omitted.
        nepa_kwargs = _patchnepa_kwargs_from_ckpt(pretrain_args, args, pretrain_state)
        head_mode = args.head_mode if args.head_mode != "auto" else ("pointmae_mlp" if effective_pooling == "cls_max" else "linear")
        model = PatchTransformerNepaClassifier(
            num_classes=num_classes,
            pooling=effective_pooling,
            pool_mode=(str(args.pool_mode).strip() or None),
            head_mode=head_mode,
            head_hidden_dim=args.head_hidden_dim,
            head_dropout=args.head_dropout,
            is_causal=bool(args.is_causal),
            ft_sequence_mode=str(args.patchnepa_ft_mode),
            cls_token_source=str(args.patchnepa_cls_token_source),
            **nepa_kwargs,
        )
        if active_patch_order_mode is None and hasattr(model, "core"):
            active_patch_order_mode = str(getattr(model.core, "patch_order_mode", "none"))
        if active_patch_order_mode is not None and hasattr(model, "core"):
            if hasattr(model.core, "set_patch_order_mode"):
                model.core.set_patch_order_mode(active_patch_order_mode)
            else:
                setattr(model.core, "patch_order_mode", active_patch_order_mode)
    else:
        model = PatchTransformerClassifier(
            num_classes=num_classes,
            patch_embed=args.patch_embed,
            patch_local_encoder=str(args.patch_local_encoder),
            patch_fps_random_start=bool(int(args.patch_fps_random_start)),
            num_groups=args.num_groups,
            group_size=args.group_size,
            use_normals=use_normals,
            center_mode=args.center_mode,
            serial_order=args.serial_order,
            serial_bits=args.serial_bits,
            serial_shuffle_within_patch=int(args.serial_shuffle_within_patch),
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
            drop_path_rate=args.drop_path_rate,
            backbone_mode=args.backbone_mode,
            qk_norm=bool(args.qk_norm),
            qk_norm_affine=bool(args.qk_norm_affine),
            qk_norm_bias=bool(args.qk_norm_bias),
            layerscale_value=float(args.layerscale_value),
            rope_theta=float(args.rope_theta),
            rope_prefix_tokens=int(args.rope_prefix_tokens),
            use_gated_mlp=bool(args.use_gated_mlp),
            hidden_act=str(args.hidden_act),
            pooling=effective_pooling,
            pos_mode=args.pos_mode,
            use_ray_patch=bool(args.use_ray_patch),
            ray_pool_mode=args.ray_pool_mode,
            ray_fuse_mode=args.ray_fuse_mode,
            ray_hidden_dim=args.ray_hidden_dim,
            ray_miss_t=args.ray_miss_t,
            ray_hit_threshold=args.ray_hit_threshold,
            head_mode=args.head_mode,
            head_hidden_dim=args.head_hidden_dim,
            head_dropout=args.head_dropout,
            init_mode=args.init_mode,
            is_causal=bool(args.is_causal),
        )
    if args.model_source == "patchnepa" and pretrain_state is None:
        if accelerator.is_main_process:
            print("[ckpt] source=scratch ckpt_use_ema=0 (no --ckpt provided)")

    if pretrain_state is not None:
        state = pretrain_state
        state_to_load = pretrain_state
        if args.model_source == "patchnepa":
            adapted_patchnepa_direct, patchnepa_direct_stats = _adapt_patchnepa_pretrain_to_patchnepa_classifier(
                state,
                model.state_dict(),
            )
            state_to_load = adapted_patchnepa_direct
            if accelerator.is_main_process:
                print(
                    "[ckpt-adapt] patchnepa-pretrain->patchnepa-classifier "
                    f"mapped={patchnepa_direct_stats['mapped']} direct={patchnepa_direct_stats['direct']} "
                    f"load={patchnepa_direct_stats['total_to_load']} src={patchnepa_direct_stats['src_total']} "
                    f"dst={patchnepa_direct_stats['dst_total']}"
                )
        # Prefer PatchNEPA->PatchCls adaptation when the pretrain checkpoint uses
        # BOS token naming (PatchNEPA) and classifier expects CLS token.
        if args.model_source == "patchcls":
            adapted_patchnepa, patchnepa_stats = _adapt_patchnepa_pretrain_to_patchcls(state, model.state_dict())
            if patchnepa_stats["mapped"] > 0:
                state_to_load = adapted_patchnepa
                if accelerator.is_main_process:
                    print(
                        "[ckpt-adapt] patchnepa->patchcls "
                        f"mapped={patchnepa_stats['mapped']} direct={patchnepa_stats['direct']} "
                        f"load={patchnepa_stats['total_to_load']} src={patchnepa_stats['src_total']} dst={patchnepa_stats['dst_total']}"
                    )
        if args.model_source == "patchcls" and args.backbone_mode == "nepa2d":
            adapted, stats = _adapt_legacy_querynepa_to_patchcls_nepa2d(state, model.state_dict())
            if stats["mapped"] > 0:
                state_to_load = adapted
                if accelerator.is_main_process:
                    print(
                        "[ckpt-adapt] legacy->patchcls-nepa2d "
                        f"mapped={stats['mapped']} direct={stats['direct']} "
                        f"load={stats['total_to_load']} src={stats['src_total']} dst={stats['dst_total']}"
                    )
        # allow head mismatch
        missing, unexpected = model.load_state_dict(state_to_load, strict=False)
        if accelerator.is_main_process:
            print(f"Loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}")
            if args.model_source == "patchnepa":
                bad_missing = [k for k in missing if not k.startswith("cls_head.")]
                # PatchNEPA direct transfer should only miss cls_head in normal cases.
                if bad_missing or unexpected:
                    print(
                        "[warn] patchnepa direct-load key mismatch: "
                        f"bad_missing={len(bad_missing)} unexpected={len(unexpected)}"
                    )
                    if bad_missing:
                        print(f"  bad_missing(sample): {bad_missing[:20]}")
                    if unexpected:
                        print(f"  unexpected(sample): {unexpected[:20]}")

    if args.model_source == "patchnepa":
        pred_head = None
        if hasattr(model, "core") and hasattr(model.core, "pred_head"):
            pred_head = model.core.pred_head
        elif hasattr(model, "pred_head"):
            pred_head = model.pred_head
        if pred_head is not None:
            # Prediction head is pretrain-only; keep it out of finetune optimization.
            for p in pred_head.parameters():
                p.requires_grad_(False)
            if accelerator.is_main_process:
                print("[finetune] patchnepa direct: pred_head frozen/excluded from optimizer")

        if int(args.patchnepa_freeze_patch_embed) == 1 and hasattr(model, "core") and hasattr(model.core, "patch_embed"):
            for p in model.core.patch_embed.parameters():
                p.requires_grad_(False)
            if accelerator.is_main_process:
                print("[finetune] patchnepa direct: patch_embed frozen")

    optimizer, opt_stats = _build_optimizer(model, args)

    lr_scheduler = None
    scheduler_step_per_batch = False
    if args.lr_scheduler == "cosine":
        llrd_sched_mode = str(args.llrd_scheduler)
        if str(args.model_source) == "patchnepa" and llrd_sched_mode in {"llrd_cosine", "llrd_cosine_warmup"}:
            # 2D-NEPA style step-wise LLRD scheduler.
            from schedulers import get_llrd_cosine_schedule, get_llrd_cosine_schedule_with_warmup

            steps_per_epoch = max(1, len(train_loader))
            num_training_steps = max(1, int(args.epochs) * int(steps_per_epoch))
            num_warmup_steps = int(round(float(args.warmup_epochs) * float(steps_per_epoch)))
            num_warmup_steps = max(0, min(num_warmup_steps, num_training_steps - 1))

            if llrd_sched_mode == "llrd_cosine":
                lr_scheduler = get_llrd_cosine_schedule(
                    optimizer=optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                    llrd_end=float(args.llrd_end),
                    llrd_mode=str(args.llrd_mode),
                )
            else:
                lr_scheduler = get_llrd_cosine_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                    llrd_mode=str(args.llrd_mode),
                )
            scheduler_step_per_batch = True
        else:
            # warmup+cosine in epoch units (legacy static mode).
            warmup_epochs = max(0.0, float(args.warmup_epochs))
            if warmup_epochs > 0:
                schedulers = [
                    optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=float(args.warmup_start_factor),
                        total_iters=int(round(warmup_epochs)),
                    ),
                    optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=max(1, args.epochs - int(round(warmup_epochs))),
                        eta_min=0.0,
                    ),
                ]
                lr_scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=schedulers,
                    milestones=[int(round(warmup_epochs))],
                )
            else:
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=0.0)

    # Only train loader is distributed. Val/test are evaluated on main process only
    # to keep single-GPU and DDP metrics strictly comparable.
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    model_raw = accelerator.unwrap_model(model)

    best_val = -1.0
    best_path = save_dir / "checkpoints" / "best.pt"

    if accelerator.is_main_process:
        if args.model_source == "pointmae":
            print(
                f"PatchCls: classes={num_classes} train={len(train_set)} val={len(val_set)} test={len(test_set)}\n"
                f"  model_source=pointmae n_point={args.n_point} groups={args.num_groups} group_size={args.group_size} "
                f"d_model={args.d_model} layers={args.n_layers} heads={args.n_heads} "
                f"drop_path_rate={args.drop_path_rate} init_mode={args.init_mode}\n"
                f"  world_size={world_size} batch_mode={args.batch_mode} batch_arg={args.batch} batch_effective={eff_batch}\n"
                f"  data_format={args.data_format} input_root={args.cache_root if args.data_format == 'npz' else args.scan_h5_root}\n"
                f"  val_split_mode={resolved_val_split_mode}\n"
                f"  train_sample={args.pt_sample_mode_train} eval_sample={args.pt_sample_mode_eval} "
                f"aug_train={aug_train} aug_preset={args.aug_preset} aug_eval={bool(args.aug_eval)} mc_test={args.mc_eval_k_test}"
            )
        elif args.model_source == "patchnepa":
            print(
                f"PatchCls: classes={num_classes} train={len(train_set)} val={len(val_set)} test={len(test_set)}\n"
                f"  model_source=patchnepa(direct) n_point={args.n_point} groups={args.num_groups} group_size={args.group_size} "
                f"use_ray_patch={int(args.use_ray_patch)} n_ray={args.n_ray} "
                f"pooling={effective_pooling} patchnepa_ft_mode={args.patchnepa_ft_mode} cls_token={args.patchnepa_cls_token_source} "
                f"patch_order_mode={','.join(patch_order_modes) if patch_order_modes else (active_patch_order_mode or 'ckpt/default')} "
                f"patch_order_schedule={patch_order_schedule} "
                f"head_mode={args.head_mode} backbone_mode={args.backbone_mode} "
                f"freeze_patch_embed={int(args.patchnepa_freeze_patch_embed)} llrd=({float(args.llrd_start):.2f}->{float(args.llrd_end):.2f}) "
                f"llrd_scheduler={args.llrd_scheduler} llrd_mode={args.llrd_mode} "
                f"lr_groups={int(opt_stats.get('n_groups', 1))} lr_range=[{opt_stats.get('lr_min', args.lr):.2e},{opt_stats.get('lr_max', args.lr):.2e}] "
                f"ray_query_only=1 is_causal={bool(args.is_causal)} ddp_find_unused={ddp_find_unused}\n"
                f"  world_size={world_size} batch_mode={args.batch_mode} batch_arg={args.batch} batch_effective={eff_batch}\n"
                f"  data_format={args.data_format} input_root={args.cache_root if args.data_format == 'npz' else args.scan_h5_root}\n"
                f"  val_split_mode={resolved_val_split_mode}\n"
                f"  train_sample={args.pt_sample_mode_train} eval_sample={args.pt_sample_mode_eval} "
                f"aug_train={aug_train} aug_preset={args.aug_preset} aug_eval={bool(args.aug_eval)} mc_test={args.mc_eval_k_test}"
            )
        else:
            pos_inject_note = (
                "per_layer_explicit" if args.backbone_mode == "vanilla" else "internal_rope_only"
            )
            print(
                f"PatchCls: classes={num_classes} train={len(train_set)} val={len(val_set)} test={len(test_set)}\n"
                f"  model_source=patchcls patch_embed={args.patch_embed} n_point={args.n_point} groups={args.num_groups} group_size={args.group_size} "
                f"use_ray_patch={int(args.use_ray_patch)} n_ray={args.n_ray} ray_pool_mode={args.ray_pool_mode} ray_fuse_mode={args.ray_fuse_mode} "
                f"ray_query_only=1 "
                f"ray_sample_train={args.ray_sample_mode_train} ray_sample_eval={args.ray_sample_mode_eval} "
                f"serial_order={args.serial_order} serial_bits={args.serial_bits} serial_shuffle={int(args.serial_shuffle_within_patch)} "
                f"d_model={args.d_model} layers={args.n_layers} heads={args.n_heads} "
                f"backbone_mode={args.backbone_mode} qk_norm={int(args.qk_norm)} qk_norm_affine={int(args.qk_norm_affine)} "
                f"qk_norm_bias={int(args.qk_norm_bias)} layerscale_value={float(args.layerscale_value):g} "
                f"rope_theta={float(args.rope_theta):g} rope_prefix_tokens={int(args.rope_prefix_tokens)} "
                f"use_gated_mlp={int(args.use_gated_mlp)} hidden_act={str(args.hidden_act)} "
                f"pooling={effective_pooling} pos_mode={args.pos_mode} pos_inject={pos_inject_note} head_mode={args.head_mode} "
                f"init_mode={args.init_mode} "
                f"is_causal={bool(args.is_causal)}\n"
                f"  world_size={world_size} batch_mode={args.batch_mode} batch_arg={args.batch} batch_effective={eff_batch}\n"
                f"  data_format={args.data_format} input_root={args.cache_root if args.data_format == 'npz' else args.scan_h5_root}\n"
                f"  val_split_mode={resolved_val_split_mode}\n"
                f"  train_sample={args.pt_sample_mode_train} eval_sample={args.pt_sample_mode_eval} "
                f"aug_train={aug_train} aug_preset={args.aug_preset} aug_eval={bool(args.aug_eval)} mc_test={args.mc_eval_k_test}"
            )

    try:
        global_step = 0
        for epoch in range(args.epochs):
            model.train()
            train_loss_sum = 0.0
            train_loss_count = 0
            if (
                args.model_source == "patchnepa"
                and patch_order_modes
                and patch_order_schedule == "epoch"
            ):
                next_mode = _select_patch_order_mode(
                    patch_order_modes,
                    patch_order_schedule,
                    epoch=epoch,
                    global_step=global_step,
                    seed=int(args.seed),
                )
                if next_mode != active_patch_order_mode:
                    if hasattr(model_raw, "core"):
                        core = model_raw.core
                        if hasattr(core, "set_patch_order_mode"):
                            core.set_patch_order_mode(next_mode)
                        else:
                            setattr(core, "patch_order_mode", next_mode)
                    active_patch_order_mode = next_mode
            for batch in train_loader:
                if (
                    args.model_source == "patchnepa"
                    and patch_order_modes
                    and patch_order_schedule in {"batch", "batch_random"}
                ):
                    next_mode = _select_patch_order_mode(
                        patch_order_modes,
                        patch_order_schedule,
                        epoch=epoch,
                        global_step=global_step,
                        seed=int(args.seed),
                    )
                    if next_mode != active_patch_order_mode:
                        if hasattr(model_raw, "core"):
                            core = model_raw.core
                            if hasattr(core, "set_patch_order_mode"):
                                core.set_patch_order_mode(next_mode)
                            else:
                                setattr(core, "patch_order_mode", next_mode)
                        active_patch_order_mode = next_mode
                xyz = batch["xyz"].to(accelerator.device)
                label = batch["label"].to(accelerator.device)
                normal = batch.get("normal", None)
                if use_normals and normal is not None:
                    normal = normal.to(accelerator.device)
                ray_o = batch.get("ray_o", None)
                ray_d = batch.get("ray_d", None)
                ray_t = batch.get("ray_t", None)
                ray_hit = batch.get("ray_hit", None)
                if bool(args.use_ray_patch):
                    if (ray_o is None) or (ray_d is None):
                        raise ValueError("use_ray_patch=True but train batch does not contain ray_o/ray_d tensors")
                    ray_o = ray_o.to(accelerator.device)
                    ray_d = ray_d.to(accelerator.device)
                    if args.model_source != "patchnepa":
                        if ray_t is not None:
                            ray_t = ray_t.to(accelerator.device)
                        if ray_hit is not None:
                            ray_hit = ray_hit.to(accelerator.device)
                    else:
                        # Query-only ray protocol in PatchNEPA direct FT.
                        ray_t = None
                        ray_hit = None

                logits = model(xyz, normal, ray_o=ray_o, ray_d=ray_d, ray_t=ray_t, ray_hit=ray_hit)
                loss = F.cross_entropy(logits, label)

                accelerator.backward(loss)
                if args.grad_clip and args.grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if lr_scheduler is not None and scheduler_step_per_batch:
                    lr_scheduler.step()

                loss_mean = accelerator.reduce(loss.detach(), reduction="mean")
                if accelerator.is_main_process:
                    train_loss_sum += float(loss_mean.item())
                    train_loss_count += 1
                global_step += 1

            if lr_scheduler is not None and (not scheduler_step_per_batch):
                lr_scheduler.step()

            # val (main process only, full dataset)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # Rank-0-only eval must run on the unwrapped module. Running forward on
                # DDP-wrapped model only on rank0 causes collective mismatch.
                val_metrics = evaluate_local(
                    accelerator.unwrap_model(model),
                    val_loader,
                    accelerator.device,
                    use_normals=use_normals,
                    use_ray_patch=bool(args.use_ray_patch),
                    model_source=str(args.model_source),
                )
                lr_now = optimizer.param_groups[0]["lr"]
                train_loss_avg = float(train_loss_sum / max(1, train_loss_count))
                print(
                    f"[ep {epoch+1:03d}/{args.epochs}] lr={lr_now:.2e} "
                    f"train_loss={train_loss_avg:.4f} "
                    f"{f'po={active_patch_order_mode} ' if active_patch_order_mode is not None else ''}"
                    f"val_acc={val_metrics['acc']:.4f} val_loss={val_metrics['loss']:.4f}"
                )

                if wandb_run is not None:
                    wb = {
                        "train/epoch": float(epoch + 1),
                        "train/lr": float(lr_now),
                        "train/loss": float(train_loss_avg),
                        "val/acc": float(val_metrics["acc"]),
                        "val/loss": float(val_metrics["loss"]),
                        "val/best_acc": float(max(best_val, val_metrics["acc"])),
                    }
                    if patch_order_modes and active_patch_order_mode is not None:
                        if active_patch_order_mode in patch_order_modes:
                            wb["train/patch_order_mode_idx"] = float(patch_order_modes.index(active_patch_order_mode))
                        else:
                            wb["train/patch_order_mode_idx"] = -1.0
                    wandb_run.log(wb, step=int(epoch + 1))

                if val_metrics["acc"] > best_val:
                    best_val = val_metrics["acc"]
                    torch.save({"model": accelerator.unwrap_model(model).state_dict(), "args": vars(args)}, best_path)
                    print(f"  saved best -> {best_path} (val_acc={best_val:.4f})")
            accelerator.wait_for_everyone()
    finally:
        pass

    # Load best on all processes for a consistent final test.
    accelerator.wait_for_everyone()
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(ckpt["model"], strict=True)

    accelerator.wait_for_everyone()
    test_metrics = None
    if accelerator.is_main_process:
        test_metrics = evaluate_local(
            accelerator.unwrap_model(model),
            test_loader,
            accelerator.device,
            use_normals=use_normals,
            use_ray_patch=bool(args.use_ray_patch),
            model_source=str(args.model_source),
        )
    if accelerator.is_main_process:
        print(f"TEST acc={test_metrics['acc']:.4f} loss={test_metrics['loss']:.4f}")
        if wandb_run is not None:
            wandb_run.log({"test/acc": float(test_metrics["acc"]), "test/loss": float(test_metrics["loss"])}, step=int(args.epochs + 1))
            wandb_run.finish()
        with open(save_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
    accelerator.wait_for_everyone()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    if hasattr(accelerator, "end_training"):
        accelerator.end_training()


if __name__ == "__main__":
    main()
