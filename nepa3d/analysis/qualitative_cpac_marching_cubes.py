import argparse
import json
import os

import numpy as np
import torch

from .completion_cpac_udf import (
    _ridge_fit,
    _ridge_pred,
    _sample_ctx_query,
    build_model_from_ckpt,
    collect_xy,
    infer_add_eos,
    infer_qa_layout,
    infer_qa_tokens,
    list_npz,
)
from ..token.tokenizer import (
    TYPE_A_POINT,
    TYPE_BOS,
    TYPE_EOS,
    TYPE_POINT,
    TYPE_Q_POINT,
)


def _grid_centers_xyz(res):
    res = int(res)
    lin = -1.0 + (np.arange(res, dtype=np.float32) + 0.5) * (2.0 / float(res))
    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
    return np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32, copy=False)


def _trilinear_sample_grid(grid, xyz):
    g = int(grid.shape[0])
    voxel = 2.0 / float(g)
    x = (xyz[:, 0] + 1.0) / voxel - 0.5
    y = (xyz[:, 1] + 1.0) / voxel - 0.5
    z = (xyz[:, 2] + 1.0) / voxel - 0.5

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    z0 = np.floor(z).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = np.clip(x0, 0, g - 1)
    y0 = np.clip(y0, 0, g - 1)
    z0 = np.clip(z0, 0, g - 1)
    x1 = np.clip(x1, 0, g - 1)
    y1 = np.clip(y1, 0, g - 1)
    z1 = np.clip(z1, 0, g - 1)

    xd = (x - x0).astype(np.float32)
    yd = (y - y0).astype(np.float32)
    zd = (z - z0).astype(np.float32)

    c000 = grid[x0, y0, z0]
    c001 = grid[x0, y0, z1]
    c010 = grid[x0, y1, z0]
    c011 = grid[x0, y1, z1]
    c100 = grid[x1, y0, z0]
    c101 = grid[x1, y0, z1]
    c110 = grid[x1, y1, z0]
    c111 = grid[x1, y1, z1]

    c00 = c000 * (1.0 - zd) + c001 * zd
    c01 = c010 * (1.0 - zd) + c011 * zd
    c10 = c100 * (1.0 - zd) + c101 * zd
    c11 = c110 * (1.0 - zd) + c111 * zd
    c0 = c00 * (1.0 - yd) + c01 * yd
    c1 = c10 * (1.0 - yd) + c11 * yd
    return (c0 * (1.0 - xd) + c1 * xd).astype(np.float32, copy=False)


def _nearest_udf_from_points(points_xyz, query_xyz, chunk=4096):
    """Approximate unsigned distance by nearest observed point distance."""
    pts = np.asarray(points_xyz, dtype=np.float32)
    q = np.asarray(query_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_xyz must be [N,3], got shape={pts.shape}")
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"query_xyz must be [M,3], got shape={q.shape}")
    out = np.empty((q.shape[0],), dtype=np.float32)
    c = max(1, int(chunk))
    for s in range(0, q.shape[0], c):
        e = min(q.shape[0], s + c)
        qq = q[s:e]
        d2 = np.sum((qq[:, None, :] - pts[None, :, :]) ** 2, axis=2)
        out[s:e] = np.sqrt(np.min(d2, axis=1)).astype(np.float32, copy=False)
    return out


def _save_obj(path, verts, faces):
    with open(path, "w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}\n")
        for tri in faces:
            # OBJ is 1-indexed
            f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


def _save_ply_points(path, xyz):
    xyz = np.asarray(xyz, dtype=np.float32)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in xyz:
            f.write(f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f}\n")


def _extract_rep_for_queries(model, ctx_xyz, ctx_dist, q_xyz, add_eos, qa_tokens, qa_layout, rep_source, device):
    n_ctx = int(ctx_xyz.shape[0])
    n_q = int(q_xyz.shape[0])
    bos = np.zeros((1, 15), dtype=np.float32)
    eos = np.zeros((1, 15), dtype=np.float32)

    if bool(qa_tokens):
        layout = str(qa_layout).lower()
        if layout not in ("interleave", "split"):
            raise ValueError(f"unknown qa_layout: {qa_layout}")

        ctx_q = np.zeros((n_ctx, 15), dtype=np.float32)
        ctx_a = np.zeros((n_ctx, 15), dtype=np.float32)
        if n_ctx > 0:
            ctx_q[:, 0:3] = ctx_xyz
            ctx_a[:, 10] = ctx_dist

        qry_q = np.zeros((n_q, 15), dtype=np.float32)
        qry_a = np.zeros((n_q, 15), dtype=np.float32)
        qry_q[:, 0:3] = q_xyz
        qry_a[:, 10] = 0.0

        if layout == "interleave":
            ctx_qa = np.zeros((2 * n_ctx, 15), dtype=np.float32)
            qry_qa = np.zeros((2 * n_q, 15), dtype=np.float32)
            if n_ctx > 0:
                ctx_qa[0::2], ctx_qa[1::2] = ctx_q, ctx_a
            if n_q > 0:
                qry_qa[0::2], qry_qa[1::2] = qry_q, qry_a

            if add_eos:
                feat = np.concatenate([bos, ctx_qa, qry_qa, eos], axis=0)
                type_id = np.concatenate(
                    [
                        np.array([TYPE_BOS], dtype=np.int64),
                        np.tile(np.array([TYPE_Q_POINT, TYPE_A_POINT], dtype=np.int64), n_ctx),
                        np.tile(np.array([TYPE_Q_POINT, TYPE_A_POINT], dtype=np.int64), n_q),
                        np.array([TYPE_EOS], dtype=np.int64),
                    ],
                    axis=0,
                )
            else:
                feat = np.concatenate([bos, ctx_qa, qry_qa], axis=0)
                type_id = np.concatenate(
                    [
                        np.array([TYPE_BOS], dtype=np.int64),
                        np.tile(np.array([TYPE_Q_POINT, TYPE_A_POINT], dtype=np.int64), n_ctx),
                        np.tile(np.array([TYPE_Q_POINT, TYPE_A_POINT], dtype=np.int64), n_q),
                    ],
                    axis=0,
                )
        else:
            q_part = np.concatenate([ctx_q, qry_q], axis=0)
            a_part = np.concatenate([ctx_a, qry_a], axis=0)
            q_type = np.full((q_part.shape[0],), TYPE_Q_POINT, dtype=np.int64)
            a_type = np.full((a_part.shape[0],), TYPE_A_POINT, dtype=np.int64)
            if add_eos:
                feat = np.concatenate([bos, q_part, a_part, eos], axis=0)
                type_id = np.concatenate(
                    [
                        np.array([TYPE_BOS], dtype=np.int64),
                        q_type,
                        a_type,
                        np.array([TYPE_EOS], dtype=np.int64),
                    ],
                    axis=0,
                )
            else:
                feat = np.concatenate([bos, q_part, a_part], axis=0)
                type_id = np.concatenate(
                    [
                        np.array([TYPE_BOS], dtype=np.int64),
                        q_type,
                        a_type,
                    ],
                    axis=0,
                )
    else:
        ctx_feat = np.zeros((n_ctx, 15), dtype=np.float32)
        if n_ctx > 0:
            ctx_feat[:, 0:3] = ctx_xyz
            ctx_feat[:, 10] = ctx_dist

        q_feat = np.zeros((n_q, 15), dtype=np.float32)
        q_feat[:, 0:3] = q_xyz
        q_feat[:, 10] = 0.0

        if add_eos:
            feat = np.concatenate([bos, ctx_feat, q_feat, eos], axis=0)
            type_id = np.concatenate(
                [
                    np.array([TYPE_BOS], dtype=np.int64),
                    np.full((n_ctx,), TYPE_POINT, dtype=np.int64),
                    np.full((n_q,), TYPE_POINT, dtype=np.int64),
                    np.array([TYPE_EOS], dtype=np.int64),
                ],
                axis=0,
            )
        else:
            feat = np.concatenate([bos, ctx_feat, q_feat], axis=0)
            type_id = np.concatenate(
                [
                    np.array([TYPE_BOS], dtype=np.int64),
                    np.full((n_ctx,), TYPE_POINT, dtype=np.int64),
                    np.full((n_q,), TYPE_POINT, dtype=np.int64),
                ],
                axis=0,
            )

    feat_t = torch.from_numpy(feat).unsqueeze(0).to(device).float()
    type_t = torch.from_numpy(type_id).unsqueeze(0).to(device).long()
    _, z_hat, h = model(feat_t, type_t)
    rep = z_hat if rep_source == "zhat" else h
    rep = rep.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    if bool(qa_tokens):
        if str(qa_layout).lower() == "split":
            q_start = 1 + n_ctx
            q_pos = q_start + np.arange(n_q, dtype=np.int64)
            return rep[q_pos]
        q_start = 1 + 2 * n_ctx
        q_pos = q_start + 2 * np.arange(n_q, dtype=np.int64)
        return rep[q_pos]
    q0 = 1 + n_ctx
    return rep[q0:q0 + n_q]


def _save_preview(path, ctx_xyz, pred_verts, gt_verts):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")
    ax1.scatter(ctx_xyz[:, 0], ctx_xyz[:, 1], ctx_xyz[:, 2], s=1)
    ax2.scatter(pred_verts[:, 0], pred_verts[:, 1], pred_verts[:, 2], s=1)
    ax3.scatter(gt_verts[:, 0], gt_verts[:, 1], gt_verts[:, 2], s=1)
    ax1.set_title("context")
    ax2.set_title("pred")
    ax3.set_title("gt")
    for ax in (ax1, ax2, ax3):
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="eval")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument(
        "--max_len",
        type=int,
        default=-1,
        help=(
            "Override model max_len (pos-emb length). If set and differs from checkpoint, "
            "pos_emb is resized by 1D interpolation."
        ),
    )
    ap.add_argument("--context_backend", type=str, default="pointcloud_noray")
    ap.add_argument("--head_train_split", type=str, default="train_udf")
    ap.add_argument("--head_train_backend", type=str, default="udfgrid")
    ap.add_argument("--head_train_max_shapes", type=int, default=4000)
    ap.add_argument("--n_context", type=int, default=256)
    ap.add_argument("--n_query_probe", type=int, default=256)
    ap.add_argument("--grid_res", type=int, default=32)
    ap.add_argument("--mc_level", type=float, default=0.02)
    ap.add_argument("--mesh_metrics", type=int, default=0)
    ap.add_argument("--mesh_samples", type=int, default=20000)
    ap.add_argument("--fscore_taus", type=str, default="0.005,0.01,0.02")
    ap.add_argument("--ridge_lambda", type=float, default=1e-3)
    ap.add_argument("--rep_source", type=str, default="h", choices=["h", "zhat"])
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--max_shapes", type=int, default=8)
    ap.add_argument("--shape_offset", type=int, default=0)
    ap.add_argument("--voxel_grid", type=int, default=64)
    ap.add_argument("--voxel_dilate", type=int, default=1)
    ap.add_argument("--voxel_max_steps", type=int, default=0)
    ap.add_argument("--save_volumes", type=int, default=1)
    ap.add_argument("--save_png", type=int, default=1)
    ap.add_argument("--save_weak_naive_mesh", type=int, default=1)
    ap.add_argument("--weak_naive_chunk", type=int, default=4096)
    ap.add_argument("--save_observed_pc", type=int, default=1)
    ap.add_argument("--save_observed_naive_mesh", type=int, default=1)
    ap.add_argument("--observed_naive_chunk", type=int, default=4096)
    ap.add_argument("--out_dir", type=str, default="results/qual_mc")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = build_model_from_ckpt(args.ckpt, device, max_len_override=args.max_len)
    add_eos = infer_add_eos(ckpt, -1)
    qa_tokens = infer_qa_tokens(ckpt, -1)
    qa_layout = infer_qa_layout(ckpt)

    tr_paths = list_npz(args.cache_root, args.head_train_split)
    if args.head_train_max_shapes and args.head_train_max_shapes > 0:
        tr_paths = tr_paths[: int(args.head_train_max_shapes)]
    if not tr_paths:
        raise RuntimeError("no head-train shapes")

    te_paths_all = list_npz(args.cache_root, args.split)
    te_paths = te_paths_all[int(args.shape_offset): int(args.shape_offset) + int(args.max_shapes)]
    if not te_paths:
        raise RuntimeError("no eval shapes selected")

    os.makedirs(args.out_dir, exist_ok=True)

    xtr, ytr, _, _ = collect_xy(
        model=model,
        paths=tr_paths,
        context_backend=args.head_train_backend,
        n_context=args.n_context,
        n_query=args.n_query_probe,
        add_eos=add_eos,
        seed=args.eval_seed,
        voxel_grid=args.voxel_grid,
        voxel_dilate=args.voxel_dilate,
        voxel_max_steps=args.voxel_max_steps,
        device=device,
        qa_tokens=qa_tokens,
        qa_layout=qa_layout,
        desc="collect train(grid)",
        context_mode="normal",
        disjoint_context_query=True,
        mismatch_shift=1,
        rep_source=args.rep_source,
        query_source="grid",
    )
    w = _ridge_fit(xtr, ytr, lam=float(args.ridge_lambda))
    np.save(os.path.join(args.out_dir, "ridge_w.npy"), w)

    max_len = int(model.pos_emb.shape[1])
    if bool(qa_tokens):
        max_q = (max_len - 1 - (1 if add_eos else 0) - 2 * int(args.n_context)) // 2
    else:
        max_q = max_len - 1 - (1 if add_eos else 0) - int(args.n_context)
    max_q = int(max_q)
    if max_q <= 0:
        raise RuntimeError(f"max query per pass is non-positive: {max_q}")

    grid_xyz = _grid_centers_xyz(int(args.grid_res))
    summary = {
        "ckpt": os.path.abspath(args.ckpt),
        "cache_root": os.path.abspath(args.cache_root),
        "split": args.split,
        "head_train_split": args.head_train_split,
        "head_train_backend": args.head_train_backend,
        "n_head_train_shapes": int(len(tr_paths)),
        "n_eval_shapes": int(len(te_paths)),
        "grid_res": int(args.grid_res),
        "mc_level": float(args.mc_level),
        "rep_source": str(args.rep_source),
        "qa_tokens": bool(qa_tokens),
        "qa_layout": str(qa_layout),
        "add_eos": bool(add_eos),
        "max_query_per_pass": int(max_q),
        "per_shape": [],
    }

    try:
        from skimage.measure import marching_cubes
    except Exception as e:
        raise RuntimeError("scikit-image is required: pip install scikit-image") from e

    for si, p in enumerate(te_paths):
        d = np.load(p, allow_pickle=False)
        if "udf_grid" not in d:
            continue
        gt_grid = d["udf_grid"].astype(np.float32, copy=False)
        g_gt = int(gt_grid.shape[0])

        _, ctx_xyz, ctx_dist, _, _ = _sample_ctx_query(
            path=p,
            context_backend=args.context_backend,
            n_context=args.n_context,
            n_query=1,
            seed=args.eval_seed + 999,
            voxel_grid=args.voxel_grid,
            voxel_dilate=args.voxel_dilate,
            voxel_max_steps=args.voxel_max_steps,
            context_mode="normal",
            disjoint_context_query=True,
            mismatch_path=None,
            query_source="pool",
        )

        pred = np.zeros((grid_xyz.shape[0],), dtype=np.float32)
        ptr = 0
        while ptr < grid_xyz.shape[0]:
            q = grid_xyz[ptr:ptr + max_q]
            xq = _extract_rep_for_queries(
                model=model,
                ctx_xyz=ctx_xyz,
                ctx_dist=ctx_dist,
                q_xyz=q,
                add_eos=add_eos,
                qa_tokens=qa_tokens,
                qa_layout=qa_layout,
                rep_source=args.rep_source,
                device=device,
            )
            pred[ptr:ptr + q.shape[0]] = _ridge_pred(xq, w).reshape(-1)
            ptr += q.shape[0]
        pred_vol = pred.reshape(int(args.grid_res), int(args.grid_res), int(args.grid_res))

        if int(args.grid_res) == g_gt:
            gt_vol = gt_grid.astype(np.float32, copy=False)
        else:
            gt_vol = _trilinear_sample_grid(gt_grid, grid_xyz).reshape(
                int(args.grid_res), int(args.grid_res), int(args.grid_res)
            )

        diff = pred_vol - gt_vol
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        tau = float(args.mc_level)
        gt_occ = gt_vol <= tau
        pr_occ = pred_vol <= tau
        inter = np.sum(gt_occ & pr_occ)
        union = np.sum(gt_occ | pr_occ) + 1e-9
        iou = float(inter / union)

        name = os.path.splitext(os.path.basename(p))[0]
        out_shape = os.path.join(args.out_dir, f"{si:03d}_{name}")
        os.makedirs(out_shape, exist_ok=True)
        context_ply_path = os.path.join(out_shape, "context_pc.ply")
        _save_ply_points(context_ply_path, ctx_xyz)

        entry = {
            "name": name,
            "path": os.path.abspath(p),
            "mae_grid": mae,
            "rmse_grid": rmse,
            "iou@level": iou,
            "out_dir": os.path.abspath(out_shape),
            "context_pc_ply": os.path.abspath(context_ply_path),
        }

        observed_pc = d["pc_xyz"].astype(np.float32, copy=False) if "pc_xyz" in d else None
        if int(args.save_observed_pc) == 1:
            if observed_pc is None or observed_pc.shape[0] <= 0:
                entry["observed_pc_error"] = "pc_xyz is missing or empty"
            else:
                observed_pc_path = os.path.join(out_shape, "observed_pc.ply")
                _save_ply_points(observed_pc_path, observed_pc)
                entry["observed_pc_ply"] = os.path.abspath(observed_pc_path)

        if int(args.save_volumes) == 1:
            pred_vol_path = os.path.join(out_shape, "pred_udf.npy")
            gt_vol_path = os.path.join(out_shape, "gt_udf.npy")
            np.save(pred_vol_path, pred_vol.astype(np.float32))
            np.save(gt_vol_path, gt_vol.astype(np.float32))
            entry["pred_udf_npy"] = os.path.abspath(pred_vol_path)
            entry["gt_udf_npy"] = os.path.abspath(gt_vol_path)

        if int(args.save_weak_naive_mesh) == 1:
            if ctx_xyz is None or ctx_xyz.shape[0] <= 0:
                entry["weak_naive_mc_error"] = "context point cloud is missing or empty"
            else:
                try:
                    weak_udf = _nearest_udf_from_points(
                        ctx_xyz,
                        grid_xyz,
                        chunk=int(args.weak_naive_chunk),
                    ).reshape(int(args.grid_res), int(args.grid_res), int(args.grid_res))
                    voxel = 2.0 / float(int(args.grid_res))
                    origin = -1.0 + 0.5 * voxel
                    wv, wf, _, _ = marching_cubes(
                        weak_udf.astype(np.float32),
                        level=float(args.mc_level),
                        spacing=(voxel, voxel, voxel),
                    )
                    wv = wv + np.array([origin, origin, origin], dtype=np.float32)
                    weak_obj_path = os.path.join(out_shape, "weak_naive_mc.obj")
                    _save_obj(weak_obj_path, wv.astype(np.float32), wf.astype(np.int64))
                    weak_vertices_ply_path = os.path.join(out_shape, "weak_naive_mc_vertices.ply")
                    _save_ply_points(weak_vertices_ply_path, wv.astype(np.float32))
                    entry["weak_naive_mc_obj"] = os.path.abspath(weak_obj_path)
                    entry["weak_naive_mc_vertices_ply"] = os.path.abspath(weak_vertices_ply_path)
                    # Backward-compatible aliases used in earlier logs/scripts.
                    entry["input_naive_mc_obj"] = entry["weak_naive_mc_obj"]
                    entry["input_naive_mc_vertices_ply"] = entry["weak_naive_mc_vertices_ply"]
                except Exception as e:
                    entry["weak_naive_mc_error"] = str(e)
                    entry["input_naive_mc_error"] = str(e)

        if int(args.save_observed_naive_mesh) == 1:
            if observed_pc is None or observed_pc.shape[0] <= 0:
                entry["observed_naive_mc_error"] = "pc_xyz is missing or empty"
            else:
                try:
                    obs_udf = _nearest_udf_from_points(
                        observed_pc,
                        grid_xyz,
                        chunk=int(args.observed_naive_chunk),
                    ).reshape(int(args.grid_res), int(args.grid_res), int(args.grid_res))
                    voxel = 2.0 / float(int(args.grid_res))
                    origin = -1.0 + 0.5 * voxel
                    ov, of, _, _ = marching_cubes(
                        obs_udf.astype(np.float32),
                        level=float(args.mc_level),
                        spacing=(voxel, voxel, voxel),
                    )
                    ov = ov + np.array([origin, origin, origin], dtype=np.float32)
                    observed_obj_path = os.path.join(out_shape, "observed_naive_mc.obj")
                    _save_obj(observed_obj_path, ov.astype(np.float32), of.astype(np.int64))
                    entry["observed_naive_mc_obj"] = os.path.abspath(observed_obj_path)
                    observed_vertices_ply_path = os.path.join(out_shape, "observed_naive_mc_vertices.ply")
                    _save_ply_points(observed_vertices_ply_path, ov.astype(np.float32))
                    entry["observed_naive_mc_vertices_ply"] = os.path.abspath(observed_vertices_ply_path)
                except Exception as e:
                    entry["observed_naive_mc_error"] = str(e)

        # Marching cubes; if level is outside value range, skip mesh export for this shape.
        try:
            voxel = 2.0 / float(int(args.grid_res))
            origin = -1.0 + 0.5 * voxel
            pv, pf, _, _ = marching_cubes(pred_vol.astype(np.float32), level=float(args.mc_level), spacing=(voxel, voxel, voxel))
            gv, gf, _, _ = marching_cubes(gt_vol.astype(np.float32), level=float(args.mc_level), spacing=(voxel, voxel, voxel))
            pv = pv + np.array([origin, origin, origin], dtype=np.float32)
            gv = gv + np.array([origin, origin, origin], dtype=np.float32)
            pred_obj_path = os.path.join(out_shape, "pred_mc.obj")
            gt_obj_path = os.path.join(out_shape, "gt_mc.obj")
            _save_obj(pred_obj_path, pv.astype(np.float32), pf.astype(np.int64))
            _save_obj(gt_obj_path, gv.astype(np.float32), gf.astype(np.int64))
            entry["pred_mc_obj"] = os.path.abspath(pred_obj_path)
            entry["gt_mc_obj"] = os.path.abspath(gt_obj_path)
            pred_vertices_ply_path = os.path.join(out_shape, "pred_mc_vertices.ply")
            gt_vertices_ply_path = os.path.join(out_shape, "gt_mc_vertices.ply")
            _save_ply_points(pred_vertices_ply_path, pv.astype(np.float32))
            _save_ply_points(gt_vertices_ply_path, gv.astype(np.float32))
            entry["pred_mc_vertices_ply"] = os.path.abspath(pred_vertices_ply_path)
            entry["gt_mc_vertices_ply"] = os.path.abspath(gt_vertices_ply_path)
            if int(args.save_png) == 1:
                preview_path = os.path.join(out_shape, "preview.png")
                _save_preview(preview_path, ctx_xyz, pv, gv)
                if os.path.isfile(preview_path):
                    entry["preview_png"] = os.path.abspath(preview_path)
        except Exception as e:
            entry["mc_error"] = str(e)

        summary["per_shape"].append(entry)
        print(f"[done] {out_shape} mae={mae:.5f} rmse={rmse:.5f} iou={iou:.5f}")

    if summary["per_shape"]:
        maes = np.array([s["mae_grid"] for s in summary["per_shape"]], dtype=np.float32)
        rmses = np.array([s["rmse_grid"] for s in summary["per_shape"]], dtype=np.float32)
        ious = np.array([s["iou@level"] for s in summary["per_shape"]], dtype=np.float32)
        summary["mean_mae_grid"] = float(maes.mean())
        summary["mean_rmse_grid"] = float(rmses.mean())
        summary["mean_iou@level"] = float(ious.mean())

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
