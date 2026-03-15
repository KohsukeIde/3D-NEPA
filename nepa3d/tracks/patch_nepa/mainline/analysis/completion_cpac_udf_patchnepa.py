"""CPAC-style UDF completion evaluation for PatchNEPA.

This is a PatchNEPA counterpart of `completion_cpac_udf.py` (QueryNEPA).

Key difference:
  - PatchNEPA consumes *surface patch tokens* as context (raw surface point cloud).
  - Query points (e.g., dense grid) are injected as *query tokens* via
    `PatchTransformerNepa.forward_tokens()`.

The script trains a lightweight ridge readout once (on a chosen split) and
reuses it across different context modalities.

Notes:
  - This implementation is intentionally minimal and focuses on the
    representation transfer + ridge setup. It supports chunked grid inference.
  - Data format: works with both legacy keys (pc_xyz / pt_xyz_pool / pt_dist_udf_pool)
    and v2 keys (surf_xyz / qry_xyz / qry_dist_udf).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import Ridge

from nepa3d.tracks.patch_nepa.mainline.models.patch_nepa import (
    TYPE_BOS,
    TYPE_EOS,
    TYPE_SEP_CTX,
    TYPE_Q_POINT,
    PatchTransformerNepa,
)
from nepa3d.utils import grid as grid_utils


def _mesh_from_udf_grid(udf_grid: np.ndarray, *, level: float) -> Tuple[np.ndarray, np.ndarray]:
    """Marching-cubes mesh from an unsigned distance field on [-1,1]^3 voxel centers."""

    from skimage import measure

    udf = np.asarray(udf_grid, dtype=np.float32)
    if udf.ndim != 3:
        raise ValueError(f"udf_grid must be (G,G,G), got {udf.shape}")
    G = int(udf.shape[0])
    if udf.shape[1] != G or udf.shape[2] != G:
        raise ValueError(f"udf_grid must be cubic, got {udf.shape}")
    if G < 2:
        raise ValueError(f"grid_res must be >=2 for marching cubes, got {G}")

    voxel = 2.0 / float(G - 1)
    verts, faces, _, _ = measure.marching_cubes(
        volume=udf,
        level=float(level),
        spacing=(voxel, voxel, voxel),
    )
    # marching_cubes vertices are in [0,2] with origin at 0; shift to [-1,1]
    verts = verts + np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    return verts.astype(np.float32, copy=False), faces.astype(np.int64, copy=False)


def _npz_get_first(npz: Dict[str, np.ndarray], keys: List[str]) -> Optional[np.ndarray]:
    for k in keys:
        if k in npz:
            return npz[k]
    return None


def _list_npz_files(root: str, split: str, max_files: int = 0) -> List[str]:
    """Recursively list *.npz under root/split."""
    base = os.path.join(root, split)
    out: List[str] = []
    for dirpath, _, filenames in os.walk(base):
        for fn in filenames:
            if fn.endswith(".npz"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    if int(max_files) > 0:
        out = out[: int(max_files)]
    return out


def _key_candidates(override: str, defaults: List[str]) -> List[str]:
    custom = str(override).strip()
    if custom:
        return [custom]
    return list(defaults)


def _default_surf_xyz_keys(context_primitive: str) -> List[str]:
    s = str(context_primitive).strip().lower()
    if ("pc" in s) or ("point" in s) or (s == "p"):
        return ["pc_xyz", "surf_xyz", "pt_xyz_pool"]
    return ["surf_xyz", "pc_xyz", "pt_xyz_pool"]


def _default_qry_xyz_keys(query_primitive: str) -> List[str]:
    s = str(query_primitive).strip().lower()
    if ("mesh" in s) or (s == "m"):
        return ["mesh_qry_xyz", "qry_xyz", "pt_xyz_pool"]
    if ("pc" in s) or ("point" in s) or (s == "p"):
        return ["pc_qry_xyz", "qry_xyz", "pt_xyz_pool"]
    return ["udf_qry_xyz", "qry_xyz", "pt_xyz_pool"]


def _default_qry_dist_keys(query_primitive: str) -> List[str]:
    s = str(query_primitive).strip().lower()
    if ("mesh" in s) or (s == "m"):
        return []
    if ("pc" in s) or ("point" in s) or (s == "p"):
        return []
    return [
        "qry_dist_udf",
        "udf_qry_dist",
        "qry_udf",
        "pt_dist_udf_pool",
        "pt_dist_pool",
    ]


def _sample_rows(arr: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
    n = int(n)
    if arr.shape[0] <= n:
        return arr
    idx = rng.choice(arr.shape[0], size=n, replace=False)
    return arr[idx]


def _sample_rows_aligned(
    arrs: List[np.ndarray], n: int, rng: np.random.RandomState
) -> List[np.ndarray]:
    """Sample the same row indices from multiple 1st-dim aligned arrays."""
    if len(arrs) == 0:
        return []
    n = int(n)
    n0 = int(arrs[0].shape[0])
    for a in arrs[1:]:
        if int(a.shape[0]) != n0:
            raise ValueError("_sample_rows_aligned(): arrays must share the same first dimension")
    if n0 <= n:
        return arrs
    idx = rng.choice(n0, size=n, replace=False)
    return [a[idx] for a in arrs]


@dataclass
class ShapeData:
    surf_xyz: np.ndarray  # [Ns, 3]
    qry_xyz: np.ndarray  # [Nq, 3]
    qry_dist: np.ndarray  # [Nq]
    udf_grid: Optional[np.ndarray] = None  # [R, R, R]


def _load_shape_npz(
    npz_path: str,
    *,
    surf_xyz_key: str = "",
    qry_xyz_key: str = "",
    qry_dist_key: str = "",
    context_primitive: str = "generic",
    query_primitive: str = "udf",
) -> ShapeData:
    with np.load(npz_path, allow_pickle=False) as f:
        surf_xyz = _npz_get_first(
            f,
            _key_candidates(surf_xyz_key, _default_surf_xyz_keys(context_primitive)),
        )
        if surf_xyz is None:
            # Fallback: old pool (not ideal, but keeps script usable).
            surf_xyz = _npz_get_first(f, ["pt_xyz_pool"])
        if surf_xyz is None:
            raise KeyError(f"No surface xyz found in {npz_path}")

        qry_xyz = _npz_get_first(
            f,
            _key_candidates(qry_xyz_key, _default_qry_xyz_keys(query_primitive)),
        )
        if qry_xyz is None:
            raise KeyError(f"No query xyz found in {npz_path}")

        qry_dist_keys = _key_candidates(qry_dist_key, _default_qry_dist_keys(query_primitive))
        qry_dist = _npz_get_first(f, qry_dist_keys)
        if qry_dist is None:
            raise KeyError(
                f"No query distance found in {npz_path} "
                f"(query_primitive={query_primitive}, qry_dist_key={qry_dist_key!r})"
            )

        qry_dist = np.asarray(qry_dist, dtype=np.float32).reshape(-1)
        surf_xyz = np.asarray(surf_xyz, dtype=np.float32).reshape(-1, 3)
        qry_xyz = np.asarray(qry_xyz, dtype=np.float32).reshape(-1, 3)

        udf_grid = _npz_get_first(f, ["udf_grid"])  # may be absent for v2-only files
        if udf_grid is not None:
            udf_grid = np.asarray(udf_grid, dtype=np.float32)

        return ShapeData(surf_xyz=surf_xyz, qry_xyz=qry_xyz, qry_dist=qry_dist, udf_grid=udf_grid)


def _build_patchnepa_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    *,
    answer_in_dim_override: Optional[int] = None,
) -> PatchTransformerNepa:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    if not isinstance(args, dict):
        args = vars(args)

    def _get(name: str, default):
        return args.get(name, default)

    num_groups_val = _get("num_groups", _get("n_patches", 64))
    if isinstance(num_groups_val, str) and num_groups_val.lower() in {"none", ""}:
        num_groups_val = None
    elif num_groups_val is not None:
        num_groups_val = int(num_groups_val)

    ckpt_answer_in_dim = _get("answer_in_dim", None)

    model = PatchTransformerNepa(
        patch_embed=str(_get("patch_embed", "fps_knn")),
        patch_local_encoder=str(_get("patch_local_encoder", "mlp")),
        patch_fps_random_start=bool(int(_get("patch_fps_random_start", 0))),
        n_point=int(_get("n_point", 1024)),
        group_size=int(_get("group_size", _get("patch_size", 32))),
        num_groups=num_groups_val,
        serial_order=str(_get("serial_order", "morton")),
        serial_bits=int(_get("serial_bits", _get("morton_bits", 10))),
        serial_shuffle_within_patch=int(_get("serial_shuffle_within_patch", 0)),
        patch_order_mode=str(_get("patch_order_mode", "none")),
        use_normals=bool(int(_get("use_normals", 0))),
        d_model=int(_get("d_model", _get("dim", 384))),
        n_layers=int(_get("n_layers", _get("depth", 12))),
        n_heads=int(_get("n_heads", _get("heads", 6))),
        mlp_ratio=float(_get("mlp_ratio", 4.0)),
        dropout=float(_get("dropout", _get("drop", 0.0))),
        drop_path_rate=float(_get("drop_path_rate", _get("drop_path", 0.0))),
        qk_norm=int(_get("qk_norm", 1)),
        qk_norm_affine=int(_get("qk_norm_affine", 0)),
        qk_norm_bias=int(_get("qk_norm_bias", 0)),
        layerscale_value=float(_get("layerscale_value", 1e-5)),
        rope_theta=float(_get("rope_theta", 100.0)),
        use_gated_mlp=int(_get("use_gated_mlp", 0)),
        hidden_act=str(_get("hidden_act", "gelu")),
        backbone_mode=str(_get("backbone_mode", "nepa2d")),
        qa_tokens=int(_get("qa_tokens", 1)),
        qa_layout=str(_get("qa_layout", "split_sep")),
        qa_sep_token=bool(int(_get("qa_sep_token", 1))),
        qa_fuse=str(_get("qa_fuse", "add")),
        use_pt_dist=bool(int(_get("use_pt_dist", 1))),
        use_pt_grad=bool(int(_get("use_pt_grad", 0))),
        answer_in_dim=(
            int(answer_in_dim_override)
            if answer_in_dim_override is not None
            else (int(ckpt_answer_in_dim) if ckpt_answer_in_dim is not None else None)
        ),
        answer_mlp_layers=int(_get("answer_mlp_layers", 2)),
        answer_pool=str(_get("answer_pool", "max")),
        q_mask_mode=str(_get("q_mask_mode", "mask_token")),
        max_len=int(_get("max_len", 4096)),
        nepa2d_pos=bool(int(_get("nepa2d_pos", 1))),
        type_specific_pos=bool(int(_get("type_specific_pos", _get("type_pos_emb", 0)))),
        type_pos_max_len=int(_get("type_pos_max_len", 4096)),
        pos_mode=str(_get("pos_mode", "center_mlp")),
        encdec_arch=bool(int(_get("encdec_arch", 0))),
        use_ray_patch=bool(int(_get("use_ray_patch", 0))),
        include_ray_unc=bool(int(_get("include_ray_unc", 0))),
        use_ray_origin=bool(int(_get("ray_use_origin", _get("use_ray_origin", 0)))),
        ray_assign_mode=str(_get("ray_assign_mode", "proxy_sphere")),
        ray_proxy_radius_scale=float(_get("ray_proxy_radius_scale", 1.05)),
        ray_pool_mode=str(_get("ray_pool_mode", "amax")),
        ray_num_groups=int(_get("ray_num_groups", 32)),
        ray_group_size=int(_get("ray_group_size", 32)),
    )

    state = ckpt.get("model_ema", ckpt.get("model", ckpt.get("model_state_dict", None)))
    if state is None:
        raise KeyError(f"Checkpoint {ckpt_path} does not contain 'model' weights")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0:
        print(f"[ckpt] missing keys: {len(missing)} (example: {missing[:5]})")
    if len(unexpected) > 0:
        print(f"[ckpt] unexpected keys: {len(unexpected)} (example: {unexpected[:5]})")
    model.to(device)
    model.eval()
    return model


def _query_point_type_for_primitive(model: PatchTransformerNepa, primitive: str) -> int:
    q_type, _ = model._primitive_to_point_type_pair(str(primitive))
    return int(q_type)


def _extract_query_reps(
    *,
    model: PatchTransformerNepa,
    surf_xyz: np.ndarray,
    q_xyz: np.ndarray,
    rep_source: str,
    n_ctx_points: int,
    chunk_n_query: int,
    device: torch.device,
    seed: int,
    context_primitive: str,
    query_primitive: str,
) -> np.ndarray:
    """Extract representations for query points given surface context."""
    rng = np.random.RandomState(int(seed))

    surf_xyz_s = _sample_rows(surf_xyz, int(n_ctx_points), rng)
    pt_xyz_t = torch.from_numpy(surf_xyz_s[None]).to(device)

    # Context: surface patch tokens.
    with torch.no_grad():
        ctx_tok, ctx_centers, _ = model.encode_patches(pt_xyz_t)

    # Special tokens.
    bos = model.bos_token.expand(1, 1, -1)
    sep = model.sep_ctx_token.expand(1, 1, -1)
    eos = model.eos_token.expand(1, 1, -1)
    zc = torch.zeros((1, 1, 3), device=device, dtype=torch.float32)
    ctx_type = _query_point_type_for_primitive(model, context_primitive)
    qry_type = _query_point_type_for_primitive(model, query_primitive)

    ctx_len = int(ctx_tok.shape[1])
    reps: List[np.ndarray] = []

    qbs = max(1, int(chunk_n_query))
    n_chunks = int(math.ceil(q_xyz.shape[0] / qbs))
    for ci in range(n_chunks):
        s = ci * qbs
        e = min(q_xyz.shape[0], (ci + 1) * qbs)
        q_chunk = q_xyz[s:e]
        q_t = torch.from_numpy(q_chunk[None]).to(device)
        with torch.no_grad():
            q_tok, q_centers = model.encode_point_queries(q_t)

            tokens = torch.cat([bos, ctx_tok, sep, q_tok, eos], dim=1)
            centers = torch.cat([
                zc,
                ctx_centers,
                zc,
                q_centers,
                zc,
            ], dim=1)
            type_id = torch.cat(
                [
                    torch.full((1, 1), TYPE_BOS, device=device, dtype=torch.long),
                    torch.full((1, ctx_len), ctx_type, device=device, dtype=torch.long),
                    torch.full((1, 1), TYPE_SEP_CTX, device=device, dtype=torch.long),
                    torch.full((1, q_tok.shape[1]), qry_type, device=device, dtype=torch.long),
                    torch.full((1, 1), TYPE_EOS, device=device, dtype=torch.long),
                ],
                dim=1,
            )

            out = model.forward_tokens(tokens, type_id, centers_xyz=centers)
            rep_t = out.h if str(rep_source).lower() == "h" else out.z_hat

            q_start = 1 + ctx_len + 1
            rep_chunk = rep_t[:, q_start : q_start + q_tok.shape[1], :]

        reps.append(rep_chunk[0].detach().cpu().numpy())

    return np.concatenate(reps, axis=0)


def _fit_ridge(
    *,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> Ridge:
    reg = Ridge(alpha=float(alpha), fit_intercept=True)
    reg.fit(X, y)
    return reg


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> Dict[str, float]:
    y_true = y_true.reshape(-1).astype(np.float32)
    y_pred = y_pred.reshape(-1).astype(np.float32)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    occ_true = y_true <= float(tau)
    occ_pred = y_pred <= float(tau)
    inter = float(np.logical_and(occ_true, occ_pred).sum())
    union = float(np.logical_or(occ_true, occ_pred).sum())
    iou = float(inter / (union + 1e-8))
    return {"mae": mae, "rmse": rmse, f"iou@{tau}": iou}


def _predict_udf_on_grid(
    *,
    model: PatchTransformerNepa,
    ridge: Ridge,
    surf_xyz: np.ndarray,
    grid_res: int,
    rep_source: str,
    n_ctx_points: int,
    chunk_n_query: int,
    device: torch.device,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict UDF values on a dense grid and also return the grid xyz coordinates."""
    grid_xyz = grid_utils.make_grid_centers_np(int(grid_res)).reshape(-1, 3).astype(np.float32)
    X = _extract_query_reps(
        model=model,
        surf_xyz=surf_xyz,
        q_xyz=grid_xyz,
        rep_source=rep_source,
        n_ctx_points=n_ctx_points,
        chunk_n_query=chunk_n_query,
        device=device,
        seed=seed,
    )
    y_pred = ridge.predict(X).reshape(-1).astype(np.float32)
    y_pred = np.maximum(y_pred, 0.0)
    y_pred_grid = y_pred.reshape(int(grid_res), int(grid_res), int(grid_res))
    return y_pred_grid, grid_xyz


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--head_train_split", type=str, default="train")
    p.add_argument("--eval_split", type=str, default="eval")
    p.add_argument("--max_train_shapes", type=int, default=0)
    p.add_argument("--max_eval_shapes", type=int, default=0)
    p.add_argument("--n_ctx_points", type=int, default=2048)
    p.add_argument("--n_query", type=int, default=4096)
    p.add_argument("--chunk_n_query", type=int, default=2048)
    p.add_argument("--rep_source", type=str, default="h", choices=["h", "zhat"])
    p.add_argument("--answer_in_dim", type=int, default=0, help="Override answer embedding input dim (0=ckpt/default)")
    p.add_argument("--surf_xyz_key", type=str, default="", help="Override context xyz key (default follows context_primitive).")
    p.add_argument("--qry_xyz_key", type=str, default="", help="Override query xyz key (default follows query_primitive).")
    p.add_argument("--qry_dist_key", type=str, default="", help="Override query target-distance key.")
    p.add_argument("--context_primitive", type=str, default="generic", help="Primitive label for context token types (e.g. pc, mesh, udf).")
    p.add_argument("--query_primitive", type=str, default="udf", help="Primitive label for query token types (e.g. udf, mesh, pc).")
    p.add_argument("--ridge_alpha", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--mesh_eval", action="store_true")
    p.add_argument("--mesh_grid_res", type=int, default=24)
    p.add_argument("--mesh_mc_level", type=float, default=0.03)
    p.add_argument("--mesh_fscore_tau", type=float, default=0.01)
    p.add_argument("--mesh_num_samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_json", type=str, default="", help="Optional path to write aggregated metrics as JSON.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ans_dim = int(args.answer_in_dim) if int(args.answer_in_dim) > 0 else None
    model = _build_patchnepa_from_ckpt(args.ckpt, device, answer_in_dim_override=ans_dim)

    # ------------------------------------------------------------------
    # Head training (ridge)
    # ------------------------------------------------------------------
    train_files = _list_npz_files(args.data_root, args.head_train_split, args.max_train_shapes)
    if len(train_files) == 0:
        raise RuntimeError(f"No training npz files under {args.data_root}/{args.head_train_split}")

    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    rng = np.random.RandomState(int(args.seed))
    for fp in train_files:
        sd = _load_shape_npz(
            fp,
            surf_xyz_key=args.surf_xyz_key,
            qry_xyz_key=args.qry_xyz_key,
            qry_dist_key=args.qry_dist_key,
            context_primitive=args.context_primitive,
            query_primitive=args.query_primitive,
        )
        # Sample query points for head training from the provided pool.
        q_xyz, q_dist = _sample_rows_aligned(
            [sd.qry_xyz, sd.qry_dist.reshape(-1, 1)],
            int(args.n_query),
            rng,
        )
        q_dist = q_dist.reshape(-1)

        X = _extract_query_reps(
            model=model,
            surf_xyz=sd.surf_xyz,
            q_xyz=q_xyz,
            rep_source=args.rep_source,
            n_ctx_points=args.n_ctx_points,
            chunk_n_query=args.chunk_n_query,
            device=device,
            seed=args.seed,
            context_primitive=args.context_primitive,
            query_primitive=args.query_primitive,
        )
        X_all.append(X)
        y_all.append(q_dist.reshape(-1, 1))

    X_train = np.concatenate(X_all, axis=0)
    y_train = np.concatenate(y_all, axis=0)
    ridge = _fit_ridge(X=X_train, y=y_train, alpha=args.ridge_alpha)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    eval_files = _list_npz_files(args.data_root, args.eval_split, args.max_eval_shapes)
    if len(eval_files) == 0:
        raise RuntimeError(f"No eval npz files under {args.data_root}/{args.eval_split}")

    metrics_sum: Dict[str, float] = {}
    mesh_sum: Dict[str, float] = {}
    mesh_n = 0
    for fp in eval_files:
        sd = _load_shape_npz(
            fp,
            surf_xyz_key=args.surf_xyz_key,
            qry_xyz_key=args.qry_xyz_key,
            qry_dist_key=args.qry_dist_key,
            context_primitive=args.context_primitive,
            query_primitive=args.query_primitive,
        )
        q_xyz, q_dist = _sample_rows_aligned(
            [sd.qry_xyz, sd.qry_dist.reshape(-1, 1)],
            int(args.n_query),
            rng,
        )
        q_dist = q_dist.reshape(-1)

        X = _extract_query_reps(
            model=model,
            surf_xyz=sd.surf_xyz,
            q_xyz=q_xyz,
            rep_source=args.rep_source,
            n_ctx_points=args.n_ctx_points,
            chunk_n_query=args.chunk_n_query,
            device=device,
            seed=args.seed,
            context_primitive=args.context_primitive,
            query_primitive=args.query_primitive,
        )
        y_pred = ridge.predict(X).reshape(-1)
        m = _compute_metrics(q_dist, y_pred, tau=args.tau)
        for k, v in m.items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + float(v)

        # Optional mesh reconstruction metrics (requires udf_grid in the npz).
        if bool(args.mesh_eval) and (sd.udf_grid is not None):
            # GT grid at desired resolution.
            gt_xyz = grid_utils.make_grid_centers_np(int(args.mesh_grid_res)).reshape(-1, 3)
            gt_udf_flat = grid_utils.trilinear_sample(sd.udf_grid, gt_xyz).reshape(-1)
            gt_udf_grid = gt_udf_flat.reshape(
                int(args.mesh_grid_res), int(args.mesh_grid_res), int(args.mesh_grid_res)
            )

            pred_udf_grid, _ = _predict_udf_on_grid(
                model=model,
                ridge=ridge,
                surf_xyz=sd.surf_xyz,
                grid_res=int(args.mesh_grid_res),
                rep_source=args.rep_source,
                n_ctx_points=args.n_ctx_points,
                chunk_n_query=args.chunk_n_query,
                device=device,
                seed=args.seed,
                context_primitive=args.context_primitive,
                query_primitive=args.query_primitive,
            )

            try:
                pred_v, pred_f = _mesh_from_udf_grid(pred_udf_grid, level=float(args.mesh_mc_level))
                gt_v, gt_f = _mesh_from_udf_grid(gt_udf_grid, level=float(args.mesh_mc_level))
            except Exception as e:
                # Skip shapes where marching cubes fails (e.g., iso-level out of range).
                print(f"(mesh_eval) skipping {os.path.basename(fp)}: marching_cubes failed: {e}")
                continue

            # Compute mesh metrics (API drift handled like in QueryNEPA script).
            from nepa3d.analysis.mesh_metrics import mesh_metrics
            import inspect

            sig = inspect.signature(mesh_metrics)
            if "fscore_tau" in sig.parameters:
                mm = mesh_metrics(
                    pred_v,
                    pred_f,
                    gt_v,
                    gt_f,
                    num_samples=int(args.mesh_num_samples),
                    fscore_tau=float(args.mesh_fscore_tau),
                )
            else:
                mm = mesh_metrics(
                    pred_v,
                    pred_f,
                    gt_v,
                    gt_f,
                    num_samples=int(args.mesh_num_samples),
                )

            for k, v in mm.items():
                mesh_sum[k] = mesh_sum.get(k, 0.0) + float(v)
            mesh_n += 1

    n = float(len(eval_files))
    metrics_avg = {k: v / n for k, v in metrics_sum.items()}
    print("PatchNEPA CPAC(UDF) metrics:")
    for k in sorted(metrics_avg.keys()):
        print(f"  {k}: {metrics_avg[k]:.6f}")

    if bool(args.mesh_eval):
        if mesh_n == 0:
            print("(mesh_eval) skipped: no shapes had udf_grid")
        else:
            mesh_avg = {k: v / float(mesh_n) for k, v in mesh_sum.items()}
            print("PatchNEPA CPAC(UDF) mesh metrics:")
            for k in sorted(mesh_avg.keys()):
                print(f"  {k}: {mesh_avg[k]:.6f}")

    out_json = str(args.out_json).strip()
    if out_json:
        report = {
            "ckpt": str(args.ckpt),
            "data_root": str(args.data_root),
            "head_train_split": str(args.head_train_split),
            "eval_split": str(args.eval_split),
            "n_ctx_points": int(args.n_ctx_points),
            "n_query": int(args.n_query),
            "chunk_n_query": int(args.chunk_n_query),
            "rep_source": str(args.rep_source),
            "surf_xyz_key": str(args.surf_xyz_key),
            "qry_xyz_key": str(args.qry_xyz_key),
            "qry_dist_key": str(args.qry_dist_key),
            "context_primitive": str(args.context_primitive),
            "query_primitive": str(args.query_primitive),
            "ridge_alpha": float(args.ridge_alpha),
            "tau": float(args.tau),
            "seed": int(args.seed),
            "max_train_shapes": int(args.max_train_shapes),
            "max_eval_shapes": int(args.max_eval_shapes),
            "metrics": metrics_avg,
        }
        if bool(args.mesh_eval) and mesh_n > 0:
            report["mesh_metrics"] = mesh_avg
            report["mesh_n"] = int(mesh_n)
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"[saved] out_json={out_json}")


if __name__ == "__main__":
    main()
