#!/usr/bin/env python
"""Checkpoint diagnostic: copy-baseline vs prediction quality.

Measures:
  - cos(pred, target) where pred = z_hat[t], target = z[t+k]
  - cos(prev, target) where prev = z[t]
  - lift = cos(pred, target) - cos(prev, target)
  - win rate = P(cos(pred, target) > cos(prev, target))

This directly tests whether prediction is better than a trivial copy baseline.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nepa3d.data.mixed_pretrain import build_mixed_pretrain
from nepa3d.models.patch_nepa import PatchTransformerNepa
from nepa3d.token.tokenizer import TYPE_A_POINT, TYPE_A_RAY, TYPE_BOS, TYPE_EOS, TYPE_MISSING_RAY, TYPE_SEP


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def _target_mask(type_id: torch.Tensor | None, k: int) -> torch.Tensor | None:
    if type_id is None:
        return None
    tgt_ty = type_id[:, k:]
    has_answer = bool((tgt_ty == int(TYPE_A_POINT)).any() or (tgt_ty == int(TYPE_A_RAY)).any())
    if has_answer:
        return (
            ((tgt_ty == int(TYPE_A_POINT)) | (tgt_ty == int(TYPE_A_RAY)))
            & (tgt_ty != int(TYPE_MISSING_RAY))
        )
    return (
        (tgt_ty != int(TYPE_BOS))
        & (tgt_ty != int(TYPE_SEP))
        & (tgt_ty != int(TYPE_EOS))
        & (tgt_ty != int(TYPE_MISSING_RAY))
    )


def _model_from_ckpt_args(a: Dict[str, Any]) -> PatchTransformerNepa:
    serial_bits = int(a.get("serial_bits", a.get("morton_bits", 10)))
    return PatchTransformerNepa(
        patch_embed=str(a.get("patch_embed", "fps_knn")),
        patch_local_encoder=str(a.get("patch_local_encoder", "mlp")),
        patch_fps_random_start=bool(int(a.get("patch_fps_random_start", 0))),
        n_point=int(a.get("n_point", 1024)),
        group_size=int(a.get("group_size", 32)),
        num_groups=int(a.get("num_groups", 64)),
        serial_order=str(a.get("serial_order", "morton")),
        serial_bits=serial_bits,
        serial_shuffle_within_patch=int(a.get("serial_shuffle_within_patch", 0)),
        use_normals=bool(int(a.get("use_normals", 0))),
        d_model=int(a.get("d_model", 384)),
        n_layers=int(a.get("n_layers", 12)),
        n_heads=int(a.get("n_heads", 6)),
        mlp_ratio=float(a.get("mlp_ratio", 4.0)),
        dropout=float(a.get("dropout", 0.0)),
        drop_path_rate=float(a.get("drop_path_rate", 0.0)),
        qk_norm=int(a.get("qk_norm", 1)),
        qk_norm_affine=int(a.get("qk_norm_affine", 0)),
        qk_norm_bias=int(a.get("qk_norm_bias", 0)),
        layerscale_value=float(a.get("layerscale_value", 1e-5)),
        rope_theta=float(a.get("rope_theta", 100.0)),
        use_gated_mlp=int(a.get("use_gated_mlp", 0)),
        hidden_act=str(a.get("hidden_act", "gelu")),
        backbone_mode=str(a.get("backbone_mode", "nepa2d")),
        qa_tokens=int(a.get("qa_tokens", 1)),
        qa_layout=str(a.get("qa_layout", "split_sep")),
        qa_sep_token=bool(int(a.get("qa_sep_token", 1))),
        qa_fuse=str(a.get("qa_fuse", "add")),
        use_pt_dist=bool(int(a.get("use_pt_dist", 1))),
        use_pt_grad=bool(int(a.get("use_pt_grad", 0))),
        answer_mlp_layers=int(a.get("answer_mlp_layers", 2)),
        answer_pool=str(a.get("answer_pool", "max")),
        max_len=int(a.get("max_len", 4096)),
        nepa2d_pos=bool(int(a.get("nepa2d_pos", 1))),
        type_specific_pos=bool(int(a.get("type_specific_pos", 0))),
        type_pos_max_len=int(a.get("type_pos_max_len", 4096)),
        pos_mode=str(a.get("pos_mode", "center_mlp")),
        encdec_arch=bool(int(a.get("encdec_arch", 0))),
        use_ray_patch=bool(int(a.get("use_ray_patch", 0))),
        include_ray_unc=bool(int(a.get("include_ray_unc", 0))),
        ray_assign_mode=str(a.get("ray_assign_mode", "proxy_sphere")),
        use_ray_origin=bool(int(a.get("ray_use_origin", 0))),
        ray_proxy_radius_scale=float(a.get("ray_proxy_radius_scale", 1.05)),
        ray_pool_mode=str(a.get("ray_pool_mode", "amax")),
        ray_num_groups=int(a.get("ray_num_groups", 32)),
        ray_group_size=int(a.get("ray_group_size", 32)),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("diag_ckpt_copy_probe")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--use_ema", type=int, default=1, choices=[0, 1])
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--max_batches", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    payload = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(payload, dict) or "args" not in payload:
        raise RuntimeError("checkpoint must be a dict with 'args'")

    ck_args: Dict[str, Any] = payload["args"]
    model = _model_from_ckpt_args(ck_args)
    state_key = "model_ema" if (int(args.use_ema) == 1 and isinstance(payload.get("model_ema", None), dict)) else "model"
    missing, unexpected = model.load_state_dict(payload[state_key], strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")
    model.to(device).eval()

    mix_config = str(ck_args["mix_config_path"])
    ds, sampler, _ = build_mixed_pretrain(
        mix_config_path=mix_config,
        n_point=int(ck_args.get("n_point", 1024)),
        n_ray=int(ck_args.get("n_ray", 0)),
        num_workers=int(args.num_workers),
        mode="train",
        return_raw=True,
        pt_xyz_key=str(ck_args.get("pt_xyz_key", "pt_xyz_pool")),
        pt_dist_key=str(ck_args.get("pt_dist_key", "pt_dist_pool")),
        ablate_point_dist=_bool(ck_args.get("ablate_point_dist", 0)),
        pt_sample_mode=str(ck_args.get("pt_sample_mode", "random")),
        pt_fps_key=str(ck_args.get("pt_fps_key", "auto")),
        pt_rfps_key=str(ck_args.get("pt_rfps_key", "auto")),
        pt_rfps_m=int(ck_args.get("pt_rfps_m", 4096)),
        point_order_mode=str(ck_args.get("point_order_mode", "morton")),
        include_pt_grad=_bool(ck_args.get("use_pt_grad", 0)),
        include_ray_unc=_bool(ck_args.get("include_ray_unc", 0)),
        aug_rotate_z=False,
        aug_scale_min=1.0,
        aug_scale_max=1.0,
        aug_translate=0.0,
        aug_jitter_sigma=0.0,
        aug_jitter_clip=0.0,
        aug_recompute_dist=False,
    )
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
    )

    k = max(1, int(args.k))
    dm_near = float(ck_args.get("dual_mask_near", 0.0))
    dm_far = float(ck_args.get("dual_mask_far", 0.0))
    dm_win = int(ck_args.get("dual_mask_window", 32))
    dm_type_aware = int(ck_args.get("dual_mask_type_aware", 0))

    all_pred_tgt = []
    all_prev_tgt = []
    all_pred_prev = []
    all_lift = []
    all_win = []
    used = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= int(args.max_batches):
                break
            b = {kk: (vv.to(device, non_blocking=True) if torch.is_tensor(vv) else vv) for kk, vv in batch.items()}
            out = model(
                pt_xyz=b["pt_xyz"],
                pt_n=None,
                pt_dist=b.get("pt_dist", None),
                pt_grad=b.get("pt_grad", None),
                ray_o=b.get("ray_o", None),
                ray_d=b.get("ray_d", None),
                ray_t=b.get("ray_t", None),
                ray_hit=b.get("ray_hit", None),
                ray_n=b.get("ray_n", None),
                ray_unc=b.get("ray_unc", None),
                ray_available=b.get("ray_available", None),
                is_causal=True,
                dual_mask_near=dm_near,
                dual_mask_far=dm_far,
                dual_mask_window=dm_win,
                dual_mask_type_aware=dm_type_aware,
            )
            if out.z.size(1) <= k or out.z_hat.size(1) <= k:
                continue

            pred = out.z_hat[:, :-k, :]
            tgt = out.z[:, k:, :]
            prev = out.z[:, :-k, :]
            mask = _target_mask(out.type_id, k)

            cos_pred_tgt = F.cosine_similarity(pred, tgt, dim=-1)
            cos_prev_tgt = F.cosine_similarity(prev, tgt, dim=-1)
            cos_pred_prev = F.cosine_similarity(pred, prev, dim=-1)
            if mask is not None:
                cos_pred_tgt = cos_pred_tgt[mask]
                cos_prev_tgt = cos_prev_tgt[mask]
                cos_pred_prev = cos_pred_prev[mask]
            if cos_pred_tgt.numel() == 0:
                continue

            lift = cos_pred_tgt - cos_prev_tgt
            win = (cos_pred_tgt > cos_prev_tgt).float().mean()

            all_pred_tgt.append(cos_pred_tgt.detach().cpu())
            all_prev_tgt.append(cos_prev_tgt.detach().cpu())
            all_pred_prev.append(cos_pred_prev.detach().cpu())
            all_lift.append(lift.detach().cpu())
            all_win.append(float(win.item()))
            used += 1

    if used == 0:
        print("[diag-copy-ckpt] no valid tokens processed")
        return

    pred_tgt = torch.cat(all_pred_tgt, dim=0)
    prev_tgt = torch.cat(all_prev_tgt, dim=0)
    pred_prev = torch.cat(all_pred_prev, dim=0)
    lift = torch.cat(all_lift, dim=0)
    win = torch.tensor(all_win, dtype=torch.float32)

    def _m(x: torch.Tensor) -> float:
        return float(x.mean().item())

    def _s(x: torch.Tensor) -> float:
        return float(x.std(unbiased=False).item())

    print("[diag-copy-ckpt] summary")
    print(f"  ckpt={ckpt_path}")
    print(f"  state={state_key} k={k} batches_used={used} tokens={int(pred_tgt.numel())}")
    print(f"  cos(pred,target): mean={_m(pred_tgt):.6f} std={_s(pred_tgt):.6f}")
    print(f"  cos(prev,target): mean={_m(prev_tgt):.6f} std={_s(prev_tgt):.6f}")
    print(f"  cos(pred,prev)  : mean={_m(pred_prev):.6f} std={_s(pred_prev):.6f}")
    print(f"  lift(pred-target minus prev-target): mean={_m(lift):.6f} std={_s(lift):.6f}")
    print(f"  win_rate(pred better than prev baseline): mean={_m(win):.6f}")
    if math.isfinite(_m(lift)):
        if _m(lift) > 0:
            print("  verdict=prediction beats copy-baseline on average")
        elif _m(lift) < 0:
            print("  verdict=copy-baseline stronger than prediction on average")
        else:
            print("  verdict=tie")


if __name__ == "__main__":
    main()
