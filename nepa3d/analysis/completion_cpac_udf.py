import argparse
import glob
import json
import os

import numpy as np
import torch

from .mesh_metrics import mesh_metrics
from ..utils import grid as grid_utils

from ..backends.mesh_backend import MeshBackend
from ..backends.pointcloud_backend import (
    PointCloudBackend,
    PointCloudMeshRayBackend,
    PointCloudNoRayBackend,
)
from ..backends.udfgrid_backend import UDFGridBackend
from ..backends.voxel_backend import VoxelBackend
from ..models.query_nepa import QueryNepa
from ..utils.ckpt_utils import load_state_dict_flexible, maybe_resize_pos_emb_in_state_dict
from ..token.ordering import morton3d
from ..token.tokenizer import (
    TYPE_BOS,
    TYPE_EOS,
    TYPE_POINT,
    TYPE_Q_POINT,
    TYPE_A_POINT,
)

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def list_npz(cache_root, split):
    return sorted(glob.glob(os.path.join(cache_root, split, "*", "*.npz")))


def _stable_seed(path, seed):
    import hashlib

    h = hashlib.md5(path.encode("utf-8")).hexdigest()[:8]
    return (int(h, 16) + int(seed)) & 0xFFFFFFFF


def _sample_disjoint_indices(n, k_ctx, k_q, rng):
    total = int(k_ctx) + int(k_q)
    if total <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    replace = n < total
    idx = rng.choice(n, size=total, replace=replace).astype(np.int64)
    return idx[: int(k_ctx)], idx[int(k_ctx):]


def _sample_indices(n, k, rng):
    k = int(k)
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    replace = n < k
    return rng.choice(n, size=k, replace=replace).astype(np.int64)


def _parse_int_csv(text, default_vals):
    vals = []
    for tok in str(text or "").split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            vals.append(int(t))
        except Exception:
            continue
    if len(vals) <= 0:
        vals = [int(v) for v in default_vals]
    return vals


def _parse_float_csv(text):
    vals = []
    for tok in str(text or "").split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            vals.append(float(t))
        except Exception:
            continue
    return vals


def _alloc_counts_from_weights(n, weights):
    n = int(max(0, n))
    if n <= 0:
        return [0 for _ in range(len(weights))]
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size <= 0 or float(np.sum(w)) <= 0:
        w = np.ones((len(weights),), dtype=np.float64)
    w = np.maximum(w, 0.0)
    w = w / float(np.sum(w))
    raw = w * float(n)
    base = np.floor(raw).astype(np.int64)
    rem = int(n - int(base.sum()))
    if rem > 0:
        order = np.argsort(-(raw - base.astype(np.float64)))
        for i in range(rem):
            base[int(order[i % order.shape[0]])] += 1
    return [int(x) for x in base.tolist()]


def _lin_to_ijk(lin, res):
    lin = np.asarray(lin, dtype=np.int64).reshape(-1)
    res = int(res)
    i = lin // (res * res)
    j = (lin // res) % res
    k = lin % res
    return np.stack([i, j, k], axis=1).astype(np.int64, copy=False)


def _ijk_to_xyz(ijk, res):
    ijk = np.asarray(ijk, dtype=np.float32).reshape(-1, 3)
    voxel = np.float32(2.0 / float(res))
    return (
        np.float32(-1.0) + (ijk + np.float32(0.5)) * voxel
    ).astype(np.float32, copy=False)


def _ijk_to_udf_nearest(udf, ijk, res):
    g = int(udf.shape[0])
    ijk = np.asarray(ijk, dtype=np.float32).reshape(-1, 3)
    # Map stage-grid centers to nearest source-grid centers.
    u = ((ijk + np.float32(0.5)) * np.float32(g / float(res))) - np.float32(0.5)
    gi = np.clip(np.rint(u[:, 0]).astype(np.int64), 0, g - 1)
    gj = np.clip(np.rint(u[:, 1]).astype(np.int64), 0, g - 1)
    gk = np.clip(np.rint(u[:, 2]).astype(np.int64), 0, g - 1)
    return udf[gi, gj, gk].astype(np.float32, copy=False)


def _sample_uniform_lin(total, n, rng):
    total = int(total)
    n = int(max(0, n))
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    replace = total < n
    return rng.choice(total, size=n, replace=replace).astype(np.int64)


def _sample_child_lin_from_parents(parent_ijk, parent_res, child_res, n, rng, expand=1):
    n = int(max(0, n))
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    parent_ijk = np.asarray(parent_ijk, dtype=np.int64).reshape(-1, 3)
    if parent_ijk.shape[0] <= 0:
        return _sample_uniform_lin(int(child_res) ** 3, n, rng)

    parent_res = int(parent_res)
    child_res = int(child_res)
    expand = int(max(0, expand))
    pick = rng.randint(0, parent_ijk.shape[0], size=n)
    cells = parent_ijk[pick].copy()
    if expand > 0:
        jit = rng.randint(-expand, expand + 1, size=(n, 3))
        cells = np.clip(cells + jit, 0, parent_res - 1)

    scale = float(child_res) / float(parent_res)
    lo = np.floor(cells.astype(np.float64) * scale).astype(np.int64)
    hi = np.ceil((cells.astype(np.float64) + 1.0) * scale).astype(np.int64) - 1
    lo = np.clip(lo, 0, child_res - 1)
    hi = np.clip(hi, 0, child_res - 1)
    hi = np.maximum(hi, lo)

    ii = lo[:, 0] + (rng.rand(n) * (hi[:, 0] - lo[:, 0] + 1).astype(np.float64)).astype(np.int64)
    jj = lo[:, 1] + (rng.rand(n) * (hi[:, 1] - lo[:, 1] + 1).astype(np.float64)).astype(np.int64)
    kk = lo[:, 2] + (rng.rand(n) * (hi[:, 2] - lo[:, 2] + 1).astype(np.float64)).astype(np.int64)
    return (ii * child_res * child_res + jj * child_res + kk).astype(np.int64, copy=False)


def _sample_udf_grid_queries(
    d,
    rng,
    n_query,
    mode="uniform",
    near_tau=0.05,
    near_frac=0.7,
    c2f_res_schedule="16,32,64",
    c2f_expand=1,
    c2f_stage_weights="",
):
    if "udf_grid" not in d:
        raise RuntimeError("query_source=grid requires `udf_grid` in cache npz")
    udf = d["udf_grid"].astype(np.float32, copy=False)
    if udf.ndim != 3 or (udf.shape[0] != udf.shape[1]) or (udf.shape[0] != udf.shape[2]):
        raise RuntimeError(f"udf_grid must be cubic, got shape={udf.shape}")
    g = int(udf.shape[0])
    total = g * g * g
    n = int(min(int(n_query), total))
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    mode = str(mode)
    udf_flat = udf.reshape(-1)

    if mode == "uniform":
        lin = rng.choice(total, size=n, replace=False).astype(np.int64)
    elif mode == "near_surface":
        near_tau = float(near_tau)
        near_frac = float(near_frac)
        near_frac = min(max(near_frac, 0.0), 1.0)

        near_idx = np.nonzero(udf_flat <= near_tau)[0]
        far_idx = np.nonzero(udf_flat > near_tau)[0]

        k_near = int(round(float(n) * near_frac))
        k_near = max(0, min(k_near, n))

        if near_idx.size <= 0:
            k_near = 0
        if far_idx.size <= 0:
            k_near = n

        k_near = min(k_near, int(near_idx.size))
        k_far = n - k_near
        k_far = min(k_far, int(far_idx.size))
        rem = n - (k_near + k_far)
        if rem > 0:
            # Backfill from the larger pool first to avoid replacement.
            add_near = min(rem, int(near_idx.size) - k_near)
            k_near += add_near
            rem -= add_near
            if rem > 0:
                add_far = min(rem, int(far_idx.size) - k_far)
                k_far += add_far
                rem -= add_far
        if (k_near + k_far) < n:
            # As a final fallback (very small grids), allow replacement.
            idx = rng.choice(total, size=n, replace=True).astype(np.int64)
            lin = idx
        else:
            parts = []
            if k_near > 0:
                parts.append(rng.choice(near_idx, size=k_near, replace=False).astype(np.int64))
            if k_far > 0:
                parts.append(rng.choice(far_idx, size=k_far, replace=False).astype(np.int64))
            lin = np.concatenate(parts, axis=0)
            rng.shuffle(lin)
    elif mode == "stratified":
        # Simple quantile-based stratified sampling over UDF magnitude.
        q = np.quantile(udf_flat, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
        bins = np.digitize(udf_flat, q, right=True)
        n_bins = 5
        per = n // n_bins
        rem = n - per * n_bins
        parts = []
        for b in range(n_bins):
            idx_b = np.nonzero(bins == b)[0]
            kb = per + (1 if b < rem else 0)
            if kb <= 0:
                continue
            if idx_b.size >= kb:
                pick = rng.choice(idx_b, size=kb, replace=False).astype(np.int64)
            else:
                pick = rng.choice(total, size=kb, replace=False).astype(np.int64)
            parts.append(pick)
        if len(parts) == 0:
            lin = rng.choice(total, size=n, replace=False).astype(np.int64)
        else:
            lin = np.concatenate(parts, axis=0).astype(np.int64, copy=False)
            if lin.shape[0] > n:
                lin = lin[:n]
            elif lin.shape[0] < n:
                add = rng.choice(total, size=(n - lin.shape[0]), replace=False).astype(np.int64)
                lin = np.concatenate([lin, add], axis=0)
            rng.shuffle(lin)
    elif mode == "coarse_to_fine":
        # Stage-wise adaptive sampling over increasingly fine query grids.
        stage_res = _parse_int_csv(c2f_res_schedule, default_vals=[16, 32, 64])
        stage_res = [int(max(2, min(g, r))) for r in stage_res]
        # keep user-specified order but remove immediate duplicates
        dedup = []
        for r in stage_res:
            if len(dedup) <= 0 or int(dedup[-1]) != int(r):
                dedup.append(int(r))
        stage_res = dedup if len(dedup) > 0 else [min(g, 64)]
        k_stage = len(stage_res)

        weight_vals = _parse_float_csv(c2f_stage_weights)
        if len(weight_vals) != k_stage or float(sum(weight_vals)) <= 0.0:
            # Favor finer stages by default: [1, 2, ..., K]
            weight_vals = [float(i + 1) for i in range(k_stage)]
        stage_counts = _alloc_counts_from_weights(n, weight_vals)

        xyz_parts = []
        dist_parts = []
        parent_near = None
        parent_res = None
        near_tau = float(near_tau)

        for si, res in enumerate(stage_res):
            n_stage = int(stage_counts[si])
            res_total = int(res) ** 3
            if si == 0:
                # Stage-1 guidance from full coarse grid.
                all_lin = np.arange(res_total, dtype=np.int64)
                all_ijk = _lin_to_ijk(all_lin, res)
                all_dist = _ijk_to_udf_nearest(udf, all_ijk, res)
                parent_near = all_ijk[all_dist <= near_tau]
                parent_res = int(res)

                if n_stage > 0:
                    lin_stage = _sample_uniform_lin(res_total, n_stage, rng)
                else:
                    lin_stage = np.zeros((0,), dtype=np.int64)
            else:
                if n_stage > 0:
                    lin_stage = _sample_child_lin_from_parents(
                        parent_ijk=parent_near,
                        parent_res=parent_res,
                        child_res=res,
                        n=n_stage,
                        rng=rng,
                        expand=int(c2f_expand),
                    )
                else:
                    lin_stage = np.zeros((0,), dtype=np.int64)

            ijk_stage = _lin_to_ijk(lin_stage, res) if lin_stage.shape[0] > 0 else np.zeros((0, 3), dtype=np.int64)
            dist_stage = (
                _ijk_to_udf_nearest(udf, ijk_stage, res)
                if ijk_stage.shape[0] > 0
                else np.zeros((0,), dtype=np.float32)
            )
            if ijk_stage.shape[0] > 0:
                xyz_stage = _ijk_to_xyz(ijk_stage, res)
                xyz_parts.append(xyz_stage)
                dist_parts.append(dist_stage)

            if si >= 1:
                near_stage = ijk_stage[dist_stage <= near_tau]
                if near_stage.shape[0] <= 0 and ijk_stage.shape[0] > 0:
                    # Keep refinement alive even when current stage has no near hits.
                    near_stage = ijk_stage
                parent_near = near_stage
                parent_res = int(res)

        if len(xyz_parts) <= 0:
            lin = _sample_uniform_lin(total, n, rng)
            i = lin // (g * g)
            j = (lin // g) % g
            k = lin % g
            voxel = np.float32(2.0 / float(g))
            xyz = np.stack(
                [
                    np.float32(-1.0) + (i.astype(np.float32) + np.float32(0.5)) * voxel,
                    np.float32(-1.0) + (j.astype(np.float32) + np.float32(0.5)) * voxel,
                    np.float32(-1.0) + (k.astype(np.float32) + np.float32(0.5)) * voxel,
                ],
                axis=1,
            ).astype(np.float32, copy=False)
            dist = udf.reshape(-1)[lin].astype(np.float32, copy=False)
            return xyz, dist

        xyz = np.concatenate(xyz_parts, axis=0).astype(np.float32, copy=False)
        dist = np.concatenate(dist_parts, axis=0).astype(np.float32, copy=False)
        if xyz.shape[0] > n:
            keep = rng.choice(xyz.shape[0], size=n, replace=False).astype(np.int64)
            xyz = xyz[keep]
            dist = dist[keep]
        elif xyz.shape[0] < n:
            n_add = n - int(xyz.shape[0])
            lin_add = _sample_uniform_lin(total, n_add, rng)
            ia = lin_add // (g * g)
            ja = (lin_add // g) % g
            ka = lin_add % g
            voxel = np.float32(2.0 / float(g))
            xyz_add = np.stack(
                [
                    np.float32(-1.0) + (ia.astype(np.float32) + np.float32(0.5)) * voxel,
                    np.float32(-1.0) + (ja.astype(np.float32) + np.float32(0.5)) * voxel,
                    np.float32(-1.0) + (ka.astype(np.float32) + np.float32(0.5)) * voxel,
                ],
                axis=1,
            ).astype(np.float32, copy=False)
            dist_add = udf.reshape(-1)[lin_add].astype(np.float32, copy=False)
            xyz = np.concatenate([xyz, xyz_add], axis=0)
            dist = np.concatenate([dist, dist_add], axis=0)
        return xyz.astype(np.float32, copy=False), dist.astype(np.float32, copy=False)
    else:
        raise ValueError(f"unknown grid_sample_mode: {mode}")

    i = lin // (g * g)
    j = (lin // g) % g
    k = lin % g
    voxel = np.float32(2.0 / float(g))
    xyz = np.stack(
        [
            np.float32(-1.0) + (i.astype(np.float32) + np.float32(0.5)) * voxel,
            np.float32(-1.0) + (j.astype(np.float32) + np.float32(0.5)) * voxel,
            np.float32(-1.0) + (k.astype(np.float32) + np.float32(0.5)) * voxel,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    dist = udf.reshape(-1)[lin].astype(np.float32, copy=False)
    return xyz, dist


def _sample_ctx_query(
    path,
    context_backend,
    n_context,
    n_query,
    seed,
    voxel_grid,
    voxel_dilate,
    voxel_max_steps,
    context_mode="normal",
    disjoint_context_query=True,
    mismatch_path=None,
    query_source="pool",
    query_pool_frac=0.5,
    grid_sample_mode="uniform",
    grid_near_tau=0.05,
    grid_near_frac=0.7,
    grid_res_schedule="16,32,64",
    grid_c2f_expand=1,
    grid_c2f_stage_weights="",
):
    rng = np.random.RandomState(_stable_seed(path, seed))
    d = np.load(path, allow_pickle=False)
    pt_xyz_pool = d["pt_xyz_pool"].astype(np.float32, copy=False)
    n_pool = int(pt_xyz_pool.shape[0])

    query_source = str(query_source)
    n_query_total = int(n_query)
    if query_source == "pool":
        n_query_pool = n_query_total
        n_query_grid = 0
    elif query_source == "grid":
        n_query_pool = 0
        n_query_grid = n_query_total
    elif query_source == "hybrid":
        p = float(query_pool_frac)
        p = min(max(p, 0.0), 1.0)
        n_query_pool = int(round(float(n_query_total) * p))
        n_query_pool = max(0, min(n_query_pool, n_query_total))
        n_query_grid = n_query_total - n_query_pool
    else:
        raise ValueError(f"unknown query_source: {query_source}")

    if context_mode == "normal" and bool(disjoint_context_query):
        cidx, qidx_pool = _sample_disjoint_indices(n_pool, int(n_context), int(n_query_pool), rng)
    else:
        cidx = _sample_indices(n_pool, int(n_context), rng)
        qidx_pool = _sample_indices(n_pool, int(n_query_pool), rng)

    if context_mode == "none":
        n_ctx_eff = 0
        ctx_xyz = np.zeros((0, 3), dtype=np.float32)
        ctx_dist = np.zeros((0,), dtype=np.float32)
    else:
        n_ctx_eff = int(n_context)
        ctx_path = mismatch_path if (context_mode == "mismatch" and mismatch_path is not None) else path
        be_ctx = make_backend(context_backend, ctx_path, voxel_grid, voxel_dilate, voxel_max_steps)
        pools_ctx = be_ctx.get_pools()
        pt_xyz_ctx_pool = pools_ctx["pt_xyz_pool"].astype(np.float32, copy=False)
        pt_dist_ctx_pool = pools_ctx["pt_dist_pool"].astype(np.float32, copy=False)
        n_ctx_pool = int(pt_xyz_ctx_pool.shape[0])

        if ctx_path != path or (cidx.size > 0 and int(cidx.max()) >= n_ctx_pool):
            cidx = _sample_indices(n_ctx_pool, int(n_context), rng)

        ctx_xyz = pt_xyz_ctx_pool[cidx].astype(np.float32, copy=False)
        ctx_dist = pt_dist_ctx_pool[cidx].astype(np.float32, copy=False)
        if ctx_xyz.shape[0] > 0:
            corder = np.argsort(morton3d(ctx_xyz))
            ctx_xyz = ctx_xyz[corder]
            ctx_dist = ctx_dist[corder]

    if query_source == "pool":
        q_xyz = pt_xyz_pool[qidx_pool].astype(np.float32, copy=False)
        if "pt_dist_udf_pool" in d:
            q_dist = d["pt_dist_udf_pool"].astype(np.float32, copy=False)[qidx_pool]
        else:
            q_dist = d["pt_dist_pool"].astype(np.float32, copy=False)[qidx_pool]
    elif query_source == "grid":
        q_xyz, q_dist = _sample_udf_grid_queries(
            d,
            rng=rng,
            n_query=n_query_grid,
            mode=grid_sample_mode,
            near_tau=grid_near_tau,
            near_frac=grid_near_frac,
            c2f_res_schedule=grid_res_schedule,
            c2f_expand=grid_c2f_expand,
            c2f_stage_weights=grid_c2f_stage_weights,
        )
    elif query_source == "hybrid":
        q_xyz_pool = pt_xyz_pool[qidx_pool].astype(np.float32, copy=False)
        if "pt_dist_udf_pool" in d:
            q_dist_pool = d["pt_dist_udf_pool"].astype(np.float32, copy=False)[qidx_pool]
        else:
            q_dist_pool = d["pt_dist_pool"].astype(np.float32, copy=False)[qidx_pool]
        q_xyz_grid, q_dist_grid = _sample_udf_grid_queries(
            d,
            rng=rng,
            n_query=n_query_grid,
            mode=grid_sample_mode,
            near_tau=grid_near_tau,
            near_frac=grid_near_frac,
            c2f_res_schedule=grid_res_schedule,
            c2f_expand=grid_c2f_expand,
            c2f_stage_weights=grid_c2f_stage_weights,
        )
        if q_xyz_pool.shape[0] <= 0:
            q_xyz, q_dist = q_xyz_grid, q_dist_grid
        elif q_xyz_grid.shape[0] <= 0:
            q_xyz, q_dist = q_xyz_pool, q_dist_pool
        else:
            q_xyz = np.concatenate([q_xyz_pool, q_xyz_grid], axis=0)
            q_dist = np.concatenate([q_dist_pool, q_dist_grid], axis=0)

    if q_xyz.shape[0] > 0:
        order = np.argsort(morton3d(q_xyz))
        q_xyz = q_xyz[order]
        q_dist = q_dist[order]
    return n_ctx_eff, ctx_xyz, ctx_dist, q_xyz, q_dist


def _nn_copy_predict(ctx_xyz, ctx_dist, q_xyz):
    if q_xyz.shape[0] <= 0:
        return np.zeros((0,), dtype=np.float32)
    if ctx_xyz.shape[0] <= 0:
        return np.zeros((q_xyz.shape[0],), dtype=np.float32)
    d2 = np.sum((q_xyz[:, None, :] - ctx_xyz[None, :, :]) ** 2, axis=-1)
    nn = np.argmin(d2, axis=1)
    return ctx_dist[nn].astype(np.float32, copy=False)


def _eval_nn_copy_baseline(
    paths,
    context_backend,
    n_context,
    n_query,
    seed,
    voxel_grid,
    voxel_dilate,
    voxel_max_steps,
    tau,
    context_mode="normal",
    disjoint_context_query=True,
    mismatch_shift=1,
    query_source="pool",
    query_pool_frac=0.5,
    grid_sample_mode="uniform",
    grid_near_tau=0.05,
    grid_near_frac=0.7,
    grid_res_schedule="16,32,64",
    grid_c2f_expand=1,
    grid_c2f_stage_weights="",
):
    mismatch_paths = [None for _ in range(len(paths))]
    if context_mode == "mismatch" and len(paths) > 1:
        shift = int(mismatch_shift) % len(paths)
        if shift == 0:
            shift = 1
        mismatch_paths = [paths[(i + shift) % len(paths)] for i in range(len(paths))]

    ys = []
    yh = []
    for i, p in enumerate(tqdm(paths, desc="baseline nn_copy")):
        _, ctx_xyz, ctx_dist, q_xyz, q_dist = _sample_ctx_query(
            path=p,
            context_backend=context_backend,
            n_context=n_context,
            n_query=n_query,
            seed=seed,
            voxel_grid=voxel_grid,
            voxel_dilate=voxel_dilate,
            voxel_max_steps=voxel_max_steps,
            context_mode=context_mode,
            disjoint_context_query=bool(disjoint_context_query),
            mismatch_path=mismatch_paths[i],
            query_source=query_source,
            query_pool_frac=query_pool_frac,
            grid_sample_mode=grid_sample_mode,
            grid_near_tau=grid_near_tau,
            grid_near_frac=grid_near_frac,
            grid_res_schedule=grid_res_schedule,
            grid_c2f_expand=grid_c2f_expand,
            grid_c2f_stage_weights=grid_c2f_stage_weights,
        )
        yhat = _nn_copy_predict(ctx_xyz, ctx_dist, q_xyz)
        ys.append(q_dist.reshape(-1))
        yh.append(yhat.reshape(-1))

    y = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.float32)
    yhat = np.concatenate(yh, axis=0) if yh else np.zeros((0,), dtype=np.float32)
    out = _metrics(y, yhat, tau=float(tau))
    out["name"] = "nn_copy"
    return out


def make_backend(name, path, voxel_grid, voxel_dilate, voxel_max_steps):
    if name == "mesh":
        return MeshBackend(path)
    if name == "pointcloud":
        return PointCloudBackend(path)
    if name == "pointcloud_meshray":
        return PointCloudMeshRayBackend(path)
    if name == "pointcloud_noray":
        return PointCloudNoRayBackend(path)
    if name == "voxel":
        return VoxelBackend(path, grid=voxel_grid, dilate=voxel_dilate, max_steps=voxel_max_steps)
    if name == "udfgrid":
        return UDFGridBackend(path)
    raise ValueError(f"unknown backend: {name}")


def infer_add_eos(ckpt, add_eos_arg):
    if add_eos_arg >= 0:
        return bool(add_eos_arg)
    pre_args = ckpt.get("args", {})
    ckpt_n_types = ckpt["model"]["type_emb.weight"].shape[0]
    return bool(pre_args.get("add_eos", ckpt_n_types >= 5))


def infer_qa_tokens(ckpt, qa_tokens_arg):
    if qa_tokens_arg >= 0:
        return bool(qa_tokens_arg)
    pre_args = ckpt.get("args", {})
    ckpt_n_types = ckpt["model"]["type_emb.weight"].shape[0]
    return bool(pre_args.get("qa_tokens", ckpt_n_types >= 9))


def infer_qa_layout(ckpt):
    pre_args = ckpt.get("args", {})
    return str(pre_args.get("qa_layout", "interleave"))


def build_model_from_ckpt(ckpt_path, device, max_len_override: int | None = None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pre_args = ckpt.get("args", {})
    state = ckpt["model"]
    d_model = state["type_emb.weight"].shape[1]
    n_types = state["type_emb.weight"].shape[0]
    nhead = int(pre_args.get("heads", 6))
    num_layers = int(pre_args.get("layers", 8))
    ckpt_max_len = int(state["pos_emb"].shape[1])
    max_len = (
        ckpt_max_len
        if (max_len_override is None or int(max_len_override) < 0)
        else int(max_len_override)
    )
    if max_len != ckpt_max_len:
        print(f"[ckpt] resizing pos_emb: ckpt_len={ckpt_max_len} -> max_len={max_len}")
        state = maybe_resize_pos_emb_in_state_dict(dict(state), max_len)

    model = QueryNepa(
        feat_dim=15,
        d_model=d_model,
        n_types=n_types,
        nhead=nhead,
        num_layers=num_layers,
        max_len=max_len,
        arch=str(pre_args.get("arch", "causal")),
        topo_k=int(pre_args.get("topo_k", 0)),
        topo_include_bos=bool(int(pre_args.get("topo_include_bos", 1))),
        topo_ray_coord=str(pre_args.get("topo_ray_coord", "origin")),
        topo_ray_bbox=float(pre_args.get("topo_ray_bbox", 0.5)),
    )
    load_state_dict_flexible(model, state, strict=True)
    model.eval().to(device)
    return model, ckpt


def _ridge_fit(X, y, lam):
    X = X.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    d = X.shape[1]
    XtX = X.T @ X
    XtX.flat[:: d + 1] += float(lam)
    Xty = X.T @ y
    w = np.linalg.solve(XtX, Xty)
    return w.astype(np.float32, copy=False)


def _ridge_pred(X, w):
    X = X.astype(np.float32, copy=False)
    X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    return (X @ w).astype(np.float32, copy=False)


def _sample_intra_shape_pairs(group_indices, n_pairs, rng):
    n_pairs = int(n_pairs)
    if n_pairs <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    valid = [g for g in group_indices if int(g.shape[0]) >= 2]
    if len(valid) <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    ii = np.empty((n_pairs,), dtype=np.int64)
    jj = np.empty((n_pairs,), dtype=np.int64)
    for t in range(n_pairs):
        arr = valid[int(rng.randint(0, len(valid)))]
        pick = rng.choice(arr, size=2, replace=False).astype(np.int64)
        ii[t] = pick[0]
        jj[t] = pick[1]
    return ii, jj


def _ridge_fit_lipschitz(
    X,
    y,
    q_xyz,
    group_id,
    lam,
    lip_lambda,
    lip_pairs=2048,
    lip_steps=200,
    lip_lr=1e-2,
    lip_batch=8192,
    lip_max_points=200000,
    lip_seed=0,
):
    # Closed-form ridge init (on the same data used for Lipschitz refinement).
    lip_lambda = float(lip_lambda)
    if lip_lambda <= 0.0:
        return _ridge_fit(X, y, lam=lam), {"enabled": False}

    X = X.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    q_xyz = q_xyz.astype(np.float32, copy=False)
    group_id = group_id.astype(np.int64, copy=False).reshape(-1)

    n = int(X.shape[0])
    if n <= 1:
        return _ridge_fit(X, y, lam=lam), {"enabled": True, "skipped": "too_few_points"}

    rng = np.random.RandomState(int(lip_seed) & 0xFFFFFFFF)
    if int(lip_max_points) > 0 and n > int(lip_max_points):
        sel = rng.permutation(n)[: int(lip_max_points)]
        X = X[sel]
        y = y[sel]
        q_xyz = q_xyz[sel]
        group_id = group_id[sel]
        n = int(X.shape[0])

    # Build per-shape index pools for intra-shape pair sampling.
    group_indices = []
    for gid in np.unique(group_id):
        idx = np.nonzero(group_id == gid)[0].astype(np.int64)
        if int(idx.shape[0]) >= 2:
            group_indices.append(idx)
    if len(group_indices) <= 0:
        return _ridge_fit(X, y, lam=lam), {"enabled": True, "skipped": "no_valid_groups"}

    # Recompute ridge solution on the (possibly capped) training set.
    w0 = _ridge_fit(X, y, lam=lam)

    X_aug = np.concatenate([X, np.ones((n, 1), dtype=np.float32)], axis=1)
    X_t = torch.from_numpy(X_aug)
    y_t = torch.from_numpy(y.reshape(-1, 1).astype(np.float32, copy=False))
    q_t = torch.from_numpy(q_xyz)

    w_t = torch.nn.Parameter(torch.from_numpy(w0.astype(np.float32, copy=False)))
    opt = torch.optim.Adam([w_t], lr=float(lip_lr))

    lip_pairs = int(max(0, lip_pairs))
    lip_steps = int(max(1, lip_steps))

    # If ridge solution already satisfies sampled Lipschitz constraints, keep it.
    with torch.no_grad():
        init_lip = w_t.new_tensor(0.0)
        if lip_pairs > 0:
            ii0, jj0 = _sample_intra_shape_pairs(group_indices, lip_pairs, rng)
            if ii0.shape[0] > 0:
                ii0_t = torch.from_numpy(ii0)
                jj0_t = torch.from_numpy(jj0)
                di0 = (X_t[ii0_t] @ w_t).squeeze(-1)
                dj0 = (X_t[jj0_t] @ w_t).squeeze(-1)
                b0 = torch.norm(q_t[ii0_t] - q_t[jj0_t], dim=1)
                init_lip = torch.relu(torch.abs(di0 - dj0) - b0).mean()
        init_lip_val = float(init_lip.detach().cpu().item())
    if init_lip_val <= 1e-12:
        return w0, {
            "enabled": True,
            "skipped": "zero_violation_at_init",
            "lip_lambda": float(lip_lambda),
            "lip_pairs": int(lip_pairs),
            "lip_steps": int(lip_steps),
            "lip_lr": float(lip_lr),
            "lip_batch": int(max(1, min(int(lip_batch), n))),
            "lip_max_points": int(lip_max_points),
            "n_train_points_after_cap": int(n),
            "n_valid_groups": int(len(group_indices)),
            "init_lip": init_lip_val,
        }

    last = {}
    for _ in range(lip_steps):
        pred_all = X_t @ w_t
        mse = ((pred_all - y_t) ** 2).mean()
        ridge = float(lam) * (w_t[:-1] ** 2).mean()

        lip = w_t.new_tensor(0.0)
        if lip_pairs > 0:
            ii, jj = _sample_intra_shape_pairs(group_indices, lip_pairs, rng)
            if ii.shape[0] > 0:
                ii_t = torch.from_numpy(ii)
                jj_t = torch.from_numpy(jj)
                di = (X_t[ii_t] @ w_t).squeeze(-1)
                dj = (X_t[jj_t] @ w_t).squeeze(-1)
                bound = torch.norm(q_t[ii_t] - q_t[jj_t], dim=1)
                lip = torch.relu(torch.abs(di - dj) - bound).mean()

        loss = mse + ridge + lip_lambda * lip
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = {
            "mse": float(mse.detach().cpu().item()),
            "ridge": float(ridge.detach().cpu().item()),
            "lip": float(lip.detach().cpu().item()),
            "loss": float(loss.detach().cpu().item()),
        }

    w = w_t.detach().cpu().numpy().astype(np.float32, copy=False)
    info = {
        "enabled": True,
            "lip_lambda": float(lip_lambda),
            "lip_pairs": int(lip_pairs),
            "lip_steps": int(lip_steps),
            "lip_lr": float(lip_lr),
            "lip_batch": int(max(1, min(int(lip_batch), n))),
            "lip_max_points": int(lip_max_points),
            "n_train_points_after_cap": int(n),
            "n_valid_groups": int(len(group_indices)),
            "init_lip": init_lip_val,
        }
    info.update(last)
    return w, info


def _metrics(y_true, y_pred, tau):
    y_true = y_true.reshape(-1).astype(np.float32)
    y_pred = y_pred.reshape(-1).astype(np.float32)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    gt = y_true <= float(tau)
    pr = y_pred <= float(tau)
    inter = np.sum(gt & pr)
    union = np.sum(gt | pr) + 1e-9
    iou = float(inter / union)
    return {"mae": mae, "rmse": rmse, "iou@tau": iou}


def _metrics_near(y_true, y_pred, tau, near_tau):
    y_true = y_true.reshape(-1).astype(np.float32)
    y_pred = y_pred.reshape(-1).astype(np.float32)
    mask = y_true <= float(near_tau)
    if int(mask.sum()) <= 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "iou@tau": float("nan")}
    m = _metrics(y_true[mask], y_pred[mask], tau=tau)
    m["n"] = int(mask.sum())
    return m


def _transform_target(y, mode, trunc_max, log_scale):
    mode = str(mode)
    y = y.astype(np.float32, copy=False)
    if mode == "none":
        return y
    if mode == "trunc":
        return np.minimum(y, np.float32(trunc_max)).astype(np.float32, copy=False)
    if mode == "log1p":
        s = np.float32(max(float(log_scale), 1e-8))
        return np.log1p(y / s).astype(np.float32, copy=False)
    raise ValueError(f"unknown target_transform: {mode}")


def _inverse_transform_target(y_t, mode, trunc_max, log_scale):
    mode = str(mode)
    y_t = y_t.astype(np.float32, copy=False)
    if mode == "none":
        return y_t
    if mode == "trunc":
        # information-losing transform; inverse is identity in transformed space.
        return y_t
    if mode == "log1p":
        s = np.float32(max(float(log_scale), 1e-8))
        return (np.expm1(y_t) * s).astype(np.float32, copy=False)
    raise ValueError(f"unknown target_transform: {mode}")


def _transform_tau(tau, mode, trunc_max, log_scale):
    mode = str(mode)
    t = float(tau)
    if mode == "none":
        return t
    if mode == "trunc":
        return min(t, float(trunc_max))
    if mode == "log1p":
        s = max(float(log_scale), 1e-8)
        return float(np.log1p(t / s))
    raise ValueError(f"unknown target_transform: {mode}")


@torch.no_grad()
def extract_xy_for_path(
    model,
    path,
    context_backend,
    n_context,
    n_query,
    add_eos,
    seed,
    voxel_grid,
    voxel_dilate,
    voxel_max_steps,
    device,
    qa_tokens,
    qa_layout="interleave",
    context_mode="normal",
    disjoint_context_query=True,
    mismatch_path=None,
    rep_source="h",
    query_source="pool",
    query_pool_frac=0.5,
    grid_sample_mode="uniform",
    grid_near_tau=0.05,
    grid_near_frac=0.7,
    grid_res_schedule="16,32,64",
    grid_c2f_expand=1,
    grid_c2f_stage_weights="",
):
    if context_mode not in ("normal", "none", "mismatch"):
        raise ValueError(f"unknown context_mode: {context_mode}")
    if rep_source not in ("h", "zhat"):
        raise ValueError(f"unknown rep_source: {rep_source}")

    n_ctx_eff, ctx_xyz, ctx_dist, q_xyz, q_dist = _sample_ctx_query(
        path=path,
        context_backend=context_backend,
        n_context=n_context,
        n_query=n_query,
        seed=seed,
        voxel_grid=voxel_grid,
        voxel_dilate=voxel_dilate,
        voxel_max_steps=voxel_max_steps,
        context_mode=context_mode,
        disjoint_context_query=bool(disjoint_context_query),
        mismatch_path=mismatch_path,
        query_source=query_source,
        query_pool_frac=query_pool_frac,
        grid_sample_mode=grid_sample_mode,
        grid_near_tau=grid_near_tau,
        grid_near_frac=grid_near_frac,
        grid_res_schedule=grid_res_schedule,
        grid_c2f_expand=grid_c2f_expand,
        grid_c2f_stage_weights=grid_c2f_stage_weights,
    )

    bos = np.zeros((1, 15), dtype=np.float32)
    eos = np.zeros((1, 15), dtype=np.float32)

    n_q_eff = int(q_xyz.shape[0])

    if bool(qa_tokens):
        layout = str(qa_layout).lower()
        if layout not in ("interleave", "split"):
            raise ValueError(f"unknown qa_layout: {qa_layout}")

        ctx_q = np.zeros((int(n_ctx_eff), 15), dtype=np.float32)
        ctx_a = np.zeros((int(n_ctx_eff), 15), dtype=np.float32)
        if n_ctx_eff > 0:
            ctx_q[:, 0:3] = ctx_xyz
            ctx_a[:, 10] = ctx_dist

        qry_q = np.zeros((n_q_eff, 15), dtype=np.float32)
        qry_a = np.zeros((n_q_eff, 15), dtype=np.float32)
        qry_q[:, 0:3] = q_xyz
        qry_a[:, 10] = 0.0
        if layout == "interleave":
            ctx_qa = np.zeros((2 * int(n_ctx_eff), 15), dtype=np.float32)
            qry_qa = np.zeros((2 * n_q_eff, 15), dtype=np.float32)
            if n_ctx_eff > 0:
                ctx_qa[0::2], ctx_qa[1::2] = ctx_q, ctx_a
            if n_q_eff > 0:
                qry_qa[0::2], qry_qa[1::2] = qry_q, qry_a

            if add_eos:
                feat = np.concatenate([bos, ctx_qa, qry_qa, eos], axis=0)
                type_id = np.concatenate(
                    [
                        np.array([TYPE_BOS], dtype=np.int64),
                        np.tile(np.array([TYPE_Q_POINT, TYPE_A_POINT], dtype=np.int64), int(n_ctx_eff)),
                        np.tile(np.array([TYPE_Q_POINT, TYPE_A_POINT], dtype=np.int64), n_q_eff),
                        np.array([TYPE_EOS], dtype=np.int64),
                    ],
                    axis=0,
                )
            else:
                feat = np.concatenate([bos, ctx_qa, qry_qa], axis=0)
                type_id = np.concatenate(
                    [
                        np.array([TYPE_BOS], dtype=np.int64),
                        np.tile(np.array([TYPE_Q_POINT, TYPE_A_POINT], dtype=np.int64), int(n_ctx_eff)),
                        np.tile(np.array([TYPE_Q_POINT, TYPE_A_POINT], dtype=np.int64), n_q_eff),
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
        ctx_feat = np.zeros((int(n_ctx_eff), 15), dtype=np.float32)
        if n_ctx_eff > 0:
            ctx_feat[:, 0:3] = ctx_xyz
            ctx_feat[:, 10] = ctx_dist

        q_feat = np.zeros((n_q_eff, 15), dtype=np.float32)
        q_feat[:, 0:3] = q_xyz
        q_feat[:, 10] = 0.0

        if add_eos:
            feat = np.concatenate([bos, ctx_feat, q_feat, eos], axis=0)
            type_id = np.concatenate(
                [
                    np.array([TYPE_BOS], dtype=np.int64),
                    np.full((int(n_ctx_eff),), TYPE_POINT, dtype=np.int64),
                    np.full((n_q_eff,), TYPE_POINT, dtype=np.int64),
                    np.array([TYPE_EOS], dtype=np.int64),
                ],
                axis=0,
            )
        else:
            feat = np.concatenate([bos, ctx_feat, q_feat], axis=0)
            type_id = np.concatenate(
                [
                    np.array([TYPE_BOS], dtype=np.int64),
                    np.full((int(n_ctx_eff),), TYPE_POINT, dtype=np.int64),
                    np.full((n_q_eff,), TYPE_POINT, dtype=np.int64),
                ],
                axis=0,
            )

    feat_t = torch.from_numpy(feat).unsqueeze(0).to(device).float()
    type_t = torch.from_numpy(type_id).unsqueeze(0).to(device).long()
    _, z_hat, h = model(feat_t, type_t)

    rep = z_hat if rep_source == "zhat" else h
    rep = rep.squeeze(0).detach().cpu().numpy().astype(np.float32)

    if bool(qa_tokens):
        if str(qa_layout).lower() == "split":
            q_start = 1 + int(n_ctx_eff)
            q_pos = q_start + np.arange(n_q_eff, dtype=np.int64)
            X = rep[q_pos]
        else:
            q_start = 1 + 2 * int(n_ctx_eff)
            q_pos = q_start + 2 * np.arange(n_q_eff, dtype=np.int64)
            X = rep[q_pos]
    else:
        q0 = 1 + int(n_ctx_eff)
        q1 = q0 + n_q_eff
        X = rep[q0:q1]

    y = q_dist.reshape(-1, 1).astype(np.float32, copy=False)
    return X, y, q_xyz.astype(np.float32, copy=False)


def _trilinear_sample_grid_np(grid: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Trilinearly sample a (G,G,G) grid at xyz in [-1, 1].

    Grid convention follows preprocessing: grid indices correspond to x/y/z axes
    in that order (i.e., grid[ix, iy, iz]).
    """

    assert grid.ndim == 3, f"grid must be (G,G,G), got {grid.shape}"
    G = int(grid.shape[0])
    assert grid.shape[1] == G and grid.shape[2] == G, f"grid must be cubic, got {grid.shape}"

    xyz = np.asarray(xyz, dtype=np.float32)
    # Normalize xyz from [-1,1] -> [0, G-1]
    gcoord = (xyz + 1.0) * 0.5 * float(G - 1)
    gx = np.clip(gcoord[:, 0], 0.0, float(G - 1))
    gy = np.clip(gcoord[:, 1], 0.0, float(G - 1))
    gz = np.clip(gcoord[:, 2], 0.0, float(G - 1))

    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    z0 = np.floor(gz).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, G - 1)
    y1 = np.clip(y0 + 1, 0, G - 1)
    z1 = np.clip(z0 + 1, 0, G - 1)

    xd = (gx - x0.astype(np.float32)).astype(np.float32)
    yd = (gy - y0.astype(np.float32)).astype(np.float32)
    zd = (gz - z0.astype(np.float32)).astype(np.float32)

    c000 = grid[x0, y0, z0]
    c100 = grid[x1, y0, z0]
    c010 = grid[x0, y1, z0]
    c110 = grid[x1, y1, z0]
    c001 = grid[x0, y0, z1]
    c101 = grid[x1, y0, z1]
    c011 = grid[x0, y1, z1]
    c111 = grid[x1, y1, z1]

    c00 = c000 * (1.0 - xd) + c100 * xd
    c01 = c001 * (1.0 - xd) + c101 * xd
    c10 = c010 * (1.0 - xd) + c110 * xd
    c11 = c011 * (1.0 - xd) + c111 * xd
    c0 = c00 * (1.0 - yd) + c10 * yd
    c1 = c01 * (1.0 - yd) + c11 * yd
    c = c0 * (1.0 - zd) + c1 * zd
    return c.astype(np.float32, copy=False)


def _mesh_from_udf_grid(udf_grid: np.ndarray, *, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Marching-cubes mesh from an unsigned distance field on [-1,1]^3 voxel centers."""

    from skimage import measure

    udf = np.asarray(udf_grid, dtype=np.float32)
    assert udf.ndim == 3, f"udf_grid must be (G,G,G), got {udf.shape}"
    G = int(udf.shape[0])
    assert udf.shape[1] == G and udf.shape[2] == G, f"udf_grid must be cubic, got {udf.shape}"
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


def _build_feat_type_for_ctx_and_queries(
    ctx_xyz: np.ndarray,
    ctx_dist: np.ndarray,
    q_xyz: np.ndarray,
    *,
    qa_tokens: int,
    qa_layout: str,
    add_eos: int,
) -> tuple[np.ndarray, np.ndarray, slice | np.ndarray]:
    """Build (feat, type_id, q_pos) for a batch with explicit query points.

    Feature layout matches tokenizer's fixed feature dim=15; we only populate:
      - xyz at [:, 0:3]
      - dist at [:, 10]
    All other channels are left as 0.
    """

    ctx_xyz = np.asarray(ctx_xyz, dtype=np.float32)
    ctx_dist = np.asarray(ctx_dist, dtype=np.float32).reshape(-1)
    q_xyz = np.asarray(q_xyz, dtype=np.float32)

    n_ctx = int(ctx_xyz.shape[0])
    n_q = int(q_xyz.shape[0])

    feat_dim = 15
    bos = np.zeros((1, feat_dim), dtype=np.float32)
    bos_type = np.array([TYPE_BOS], dtype=np.int64)

    if int(qa_tokens) > 0:
        # Context
        ctx_q = np.zeros((n_ctx, feat_dim), dtype=np.float32)
        ctx_q[:, 0:3] = ctx_xyz
        ctx_q_type = np.full((n_ctx,), TYPE_Q_POINT, dtype=np.int64)

        ctx_a = np.zeros((n_ctx, feat_dim), dtype=np.float32)
        ctx_a[:, 10] = ctx_dist
        ctx_a_type = np.full((n_ctx,), TYPE_A_POINT, dtype=np.int64)

        # Queries (answers are zeroed)
        q_q = np.zeros((n_q, feat_dim), dtype=np.float32)
        q_q[:, 0:3] = q_xyz
        q_q_type = np.full((n_q,), TYPE_Q_POINT, dtype=np.int64)

        q_a = np.zeros((n_q, feat_dim), dtype=np.float32)
        q_a[:, 10] = 0.0
        q_a_type = np.full((n_q,), TYPE_A_POINT, dtype=np.int64)

        if str(qa_layout) == "split":
            feat = np.concatenate([bos, ctx_q, q_q, ctx_a, q_a], axis=0)
            type_id = np.concatenate(
                [bos_type, ctx_q_type, q_q_type, ctx_a_type, q_a_type], axis=0
            )
            q_pos: slice | np.ndarray = slice(1 + n_ctx, 1 + n_ctx + n_q)
        else:
            ctx_qa = np.stack([ctx_q, ctx_a], axis=1).reshape(2 * n_ctx, feat_dim)
            ctx_qa_type = np.stack([ctx_q_type, ctx_a_type], axis=1).reshape(2 * n_ctx)
            q_qa = np.stack([q_q, q_a], axis=1).reshape(2 * n_q, feat_dim)
            q_qa_type = np.stack([q_q_type, q_a_type], axis=1).reshape(2 * n_q)
            feat = np.concatenate([bos, ctx_qa, q_qa], axis=0)
            type_id = np.concatenate([bos_type, ctx_qa_type, q_qa_type], axis=0)
            base = 1 + 2 * n_ctx
            q_pos = base + 2 * np.arange(n_q, dtype=np.int64)
    else:
        ctx_feat = np.zeros((n_ctx, feat_dim), dtype=np.float32)
        ctx_feat[:, 0:3] = ctx_xyz
        ctx_feat[:, 10] = ctx_dist
        q_feat = np.zeros((n_q, feat_dim), dtype=np.float32)
        q_feat[:, 0:3] = q_xyz
        q_feat[:, 10] = 0.0
        feat = np.concatenate([bos, ctx_feat, q_feat], axis=0)
        type_id = np.concatenate(
            [bos_type, np.full((n_ctx + n_q,), TYPE_POINT, dtype=np.int64)], axis=0
        )
        q_pos = slice(1 + n_ctx, 1 + n_ctx + n_q)

    if int(add_eos) > 0:
        eos = np.zeros((1, feat_dim), dtype=np.float32)
        feat = np.concatenate([feat, eos], axis=0)
        type_id = np.concatenate([type_id, np.array([TYPE_EOS], dtype=np.int64)], axis=0)

    # Add batch dimension
    feat = feat[None, :, :]
    type_id = type_id[None, :]
    return feat, type_id, q_pos


def _predict_udf_grid_ridge(
    *,
    model: torch.nn.Module,
    ridge_w: np.ndarray,
    ctx_xyz: np.ndarray,
    ctx_dist: np.ndarray,
    grid_res: int,
    chunk_n_query: int,
    qa_tokens: int,
    qa_layout: str,
    rep_source: str,
    add_eos: int,
    device: torch.device,
) -> np.ndarray:
    """Predict a dense UDF grid via chunked query evaluation + ridge head."""

    import math as _math

    # Precompute grid query points (voxel centers in [-1, 1]).
    centers = grid_utils.make_grid_centers_np(int(grid_res)).astype(np.float32)
    q_xyz_all = centers.reshape(-1, 3)
    N = int(q_xyz_all.shape[0])

    # Move ridge weights to device for fast matmul.
    w_t = torch.from_numpy(ridge_w.astype(np.float32, copy=False)).to(device)
    W = w_t[:-1, :]
    b = w_t[-1:, :]

    pred = np.zeros((N,), dtype=np.float32)

    # Chunk over query points to keep sequence length manageable.
    qbs = max(1, int(chunk_n_query))
    n_chunks = int(_math.ceil(N / qbs))
    for ci in range(n_chunks):
        s = ci * qbs
        e = min(N, (ci + 1) * qbs)
        q_xyz = q_xyz_all[s:e]

        feat, type_id, q_pos = _build_feat_type_for_ctx_and_queries(
            ctx_xyz,
            ctx_dist,
            q_xyz,
            qa_tokens=qa_tokens,
            qa_layout=qa_layout,
            add_eos=add_eos,
        )
        feat_t = torch.from_numpy(feat).to(device)
        type_t = torch.from_numpy(type_id).to(device)

        with torch.no_grad():
            z, z_hat, h = model(feat_t, type_t)
            rep = h if str(rep_source) == "h" else z_hat
            rep_q = rep[:, q_pos, :].float().squeeze(0)  # (nq, d)
            y = rep_q @ W + b  # (nq, 1)
            pred[s:e] = y.squeeze(-1).detach().cpu().numpy().astype(np.float32, copy=False)

    pred_grid = pred.reshape(int(grid_res), int(grid_res), int(grid_res))
    return np.maximum(pred_grid, 0.0).astype(np.float32, copy=False)


def _mesh_eval_chamfer_for_paths(
    *,
    model: torch.nn.Module | None,
    ridge_w: np.ndarray | None,
    paths: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """Compute mesh metrics (Chamfer/F-score) for a subset of paths."""

    import os
    import trimesh

    ckpt_meta = None
    def _get_ckpt_meta():
        nonlocal ckpt_meta
        if ckpt_meta is None:
            ckpt_meta = torch.load(str(args.ckpt), map_location="cpu")
        return ckpt_meta

    # Keep mesh_eval robust even if caller does not populate these namespace fields.
    qa_tokens = int(getattr(args, "qa_tokens", -1))
    if qa_tokens < 0:
        qa_tokens = int(infer_qa_tokens(_get_ckpt_meta(), -1))
    qa_layout = str(getattr(args, "qa_layout", ""))
    if qa_layout == "":
        qa_layout = str(infer_qa_layout(_get_ckpt_meta()))
    add_eos = int(getattr(args, "add_eos", -1))
    if add_eos < 0:
        add_eos = int(infer_add_eos(_get_ckpt_meta(), -1))

    # Subsample paths if requested.
    max_shapes = int(getattr(args, "mesh_eval_max_shapes", 0) or 0)
    if max_shapes > 0:
        paths = list(paths)[:max_shapes]

    save_dir = str(getattr(args, "mesh_save_dir", "") or "")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Basic safety check: ensure positional embeddings are long enough for the
    # (context + chunked queries) sequence length.
    if ridge_w is not None and model is not None:
        n_ctx = int(getattr(args, "n_context", 0))
        qbs = int(getattr(args, "mesh_chunk_n_query", 512))
        required_len = 1 + (1 + int(qa_tokens)) * (n_ctx + qbs) + (1 if int(add_eos) > 0 else 0)
        try:
            max_len = int(getattr(model, "pos_emb").shape[1])
        except Exception:
            max_len = -1
        if max_len > 0 and required_len > max_len:
            raise ValueError(
                f"mesh_eval requires sequence length {required_len}, but model max_len={max_len}. "
                "Re-run with a larger --max_len (pos_emb will be resized)."
            )

    per_shape = []
    fail_count = 0

    # Use a deterministic order; any randomness is only in mesh surface sampling.
    for p in tqdm(paths, desc="mesh_eval", total=len(paths)):
        # Context sampling follows the normal CPAC path.
        _, ctx_xyz, ctx_dist, _, _ = _sample_ctx_query(
            path=p,
            context_backend=args.context_backend,
            n_context=int(args.n_context),
            seed=int(args.eval_seed),
            voxel_grid=int(args.voxel_grid),
            voxel_dilate=int(args.voxel_dilate),
            voxel_max_steps=int(args.voxel_max_steps),
            query_source="pool",
            n_query=0,
            disjoint_context_query=int(args.disjoint_context_query),
            context_mode=str(args.context_mode_test),
            grid_res_schedule=str(args.grid_res_schedule),
            grid_sample_mode=str(args.grid_sample_mode),
            grid_near_frac=float(args.grid_near_frac),
            grid_near_tau=float(args.grid_near_tau),
            grid_c2f_expand=int(args.grid_c2f_expand),
            grid_c2f_stage_weights=str(args.grid_c2f_stage_weights),
        )

        # GT grid (native res) -> resample to mesh_grid_res.
        try:
            with np.load(p) as d:
                gt_udf_native = d["udf_grid"].astype(np.float32)
        except Exception as e:
            per_shape.append({"path": p, "ok": False, "error": f"missing udf_grid: {e}"})
            fail_count += 1
            continue

        grid_res = int(getattr(args, "mesh_grid_res", 24))
        centers = grid_utils.make_grid_centers_np(grid_res).astype(np.float32)
        q_xyz_all = centers.reshape(-1, 3)
        gt_vals = _trilinear_sample_grid_np(gt_udf_native, q_xyz_all).reshape(grid_res, grid_res, grid_res)
        gt_vals = np.maximum(gt_vals, 0.0).astype(np.float32, copy=False)

        # Prediction.
        if ridge_w is not None and model is not None:
            pred_vals = _predict_udf_grid_ridge(
                model=model,
                ridge_w=ridge_w,
                ctx_xyz=ctx_xyz,
                ctx_dist=ctx_dist,
                grid_res=grid_res,
                chunk_n_query=int(getattr(args, "mesh_chunk_n_query", 512)),
                qa_tokens=int(qa_tokens),
                qa_layout=str(qa_layout),
                rep_source=str(args.rep_source),
                add_eos=int(add_eos),
                device=device,
            )
        else:
            # Fallback: nearest-neighbor distance to context xyz (geometric baseline)
            # This makes mesh_eval usable even with --baseline_only.
            try:
                from scipy.spatial import cKDTree

                tree = cKDTree(ctx_xyz)
                dist, _ = tree.query(q_xyz_all, k=1, workers=-1)
                pred_vals = dist.astype(np.float32).reshape(grid_res, grid_res, grid_res)
            except Exception as e:
                per_shape.append({"path": p, "ok": False, "error": f"baseline_nn_dist failed: {e}"})
                fail_count += 1
                continue

        level = float(getattr(args, "mesh_mc_level", 0.03))
        try:
            v_pred, f_pred = _mesh_from_udf_grid(pred_vals, level=level)
            v_gt, f_gt = _mesh_from_udf_grid(gt_vals, level=level)

            m = mesh_metrics(
                v_pred,
                f_pred,
                v_gt,
                f_gt,
                num_samples=int(getattr(args, "mesh_num_samples", 10000)),
                fscore_tau=float(getattr(args, "mesh_fscore_tau", 0.01)),
            )
            rec = {"path": p, "ok": True, **m}

            if save_dir:
                sid = os.path.splitext(os.path.basename(p))[0]
                trimesh.Trimesh(vertices=v_pred, faces=f_pred, process=False).export(
                    os.path.join(save_dir, f"pred_{sid}.ply")
                )
                trimesh.Trimesh(vertices=v_gt, faces=f_gt, process=False).export(
                    os.path.join(save_dir, f"gt_{sid}.ply")
                )
        except Exception as e:
            rec = {"path": p, "ok": False, "error": str(e)}
            fail_count += 1

        per_shape.append(rec)

    # Aggregate.
    keys = ["chamfer_l2", "chamfer_l1", "fscore"]
    agg = {}
    for k in keys:
        vals = [float(r[k]) for r in per_shape if r.get("ok") and k in r]
        if vals:
            agg[k + "_mean"] = float(np.mean(vals))
            agg[k + "_std"] = float(np.std(vals))
            agg[k + "_n"] = int(len(vals))
        else:
            agg[k + "_mean"] = float("nan")
            agg[k + "_std"] = float("nan")
            agg[k + "_n"] = 0

    pred_mode = "ridge" if (ridge_w is not None and model is not None) else "nn_dist"

    out = {
        "enabled": True,
        "pred_mode": pred_mode,
        "grid_res": int(getattr(args, "mesh_grid_res", 24)),
        "chunk_n_query": int(getattr(args, "mesh_chunk_n_query", 512)),
        "mc_level": float(getattr(args, "mesh_mc_level", 0.03)),
        "num_samples": int(getattr(args, "mesh_num_samples", 10000)),
        "fscore_tau": float(getattr(args, "mesh_fscore_tau", 0.01)),
        "fail_count": int(fail_count),
        "n_eval_shapes": int(len(paths)),
        "summary": agg,
    }
    if int(getattr(args, "mesh_store_per_shape", 0)) > 0:
        out["per_shape"] = per_shape
    return out


def collect_xy(
    model,
    paths,
    context_backend,
    n_context,
    n_query,
    add_eos,
    seed,
    voxel_grid,
    voxel_dilate,
    voxel_max_steps,
    device,
    qa_tokens,
    qa_layout,
    desc,
    context_mode="normal",
    disjoint_context_query=True,
    mismatch_shift=1,
    rep_source="h",
    query_source="pool",
    query_pool_frac=0.5,
    grid_sample_mode="uniform",
    grid_near_tau=0.05,
    grid_near_frac=0.7,
    grid_res_schedule="16,32,64",
    grid_c2f_expand=1,
    grid_c2f_stage_weights="",
):
    mismatch_paths = [None for _ in range(len(paths))]
    if context_mode == "mismatch" and len(paths) > 1:
        shift = int(mismatch_shift) % len(paths)
        if shift == 0:
            shift = 1
        mismatch_paths = [paths[(i + shift) % len(paths)] for i in range(len(paths))]

    xs = []
    ys = []
    qxyzs = []
    gids = []
    for i, p in enumerate(tqdm(paths, desc=desc)):
        X, y, q_xyz = extract_xy_for_path(
            model,
            p,
            context_backend,
            n_context,
            n_query,
            add_eos,
            seed,
            voxel_grid,
            voxel_dilate,
            voxel_max_steps,
            device,
            qa_tokens,
            qa_layout,
            context_mode=context_mode,
            disjoint_context_query=bool(disjoint_context_query),
            mismatch_path=mismatch_paths[i],
            rep_source=rep_source,
            query_source=query_source,
            query_pool_frac=query_pool_frac,
            grid_sample_mode=grid_sample_mode,
            grid_near_tau=grid_near_tau,
            grid_near_frac=grid_near_frac,
            grid_res_schedule=grid_res_schedule,
            grid_c2f_expand=grid_c2f_expand,
            grid_c2f_stage_weights=grid_c2f_stage_weights,
        )
        xs.append(X)
        ys.append(y)
        qxyzs.append(q_xyz)
        gids.append(np.full((q_xyz.shape[0],), i, dtype=np.int64))
    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(qxyzs, axis=0),
        np.concatenate(gids, axis=0),
    )


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
    ap.add_argument("--n_context", type=int, default=256)
    ap.add_argument("--n_query", type=int, default=256)
    ap.add_argument("--add_eos", type=int, default=-1)
    ap.add_argument("--qa_tokens", type=int, default=-1, help="Use Q/A tokenization (1/0). -1: infer from ckpt.")
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Alias for --eval_seed (kept for older wrappers).",
    )
    ap.add_argument("--head_train_ratio", type=float, default=0.2)
    ap.add_argument("--head_train_n", type=int, default=0)
    ap.add_argument(
        "--head_train_split",
        type=str,
        default="train_udf",
        help=("Split used to train the ridge head. "
              "Default: train_udf (non-transductive). "
              "If set to empty or 'none', uses legacy transductive split within --split."),
    )
    ap.add_argument(
        "--head_train_backend",
        type=str,
        default="udfgrid",
        help=("Backend used for head training context extraction. "
              "Default: udfgrid (monolingual target language)."),
    )
    ap.add_argument(
        "--head_train_max_shapes",
        type=int,
        default=0,
        help="Optional cap on number of shapes used to train the head (0=all).",
    )
    ap.add_argument("--ridge_lambda", type=float, default=1e-3)
    ap.add_argument(
        "--ridge_lipschitz_lambda",
        type=float,
        default=0.0,
        help="Optional Lipschitz hinge penalty weight for ridge probe fitting (default: off).",
    )
    ap.add_argument(
        "--ridge_lipschitz_pairs",
        type=int,
        default=2048,
        help="Number of intra-shape pairs sampled per optimization step when Lipschitz is enabled.",
    )
    ap.add_argument(
        "--ridge_lipschitz_steps",
        type=int,
        default=200,
        help="Number of optimization steps for Lipschitz-refined ridge fitting.",
    )
    ap.add_argument(
        "--ridge_lipschitz_lr",
        type=float,
        default=1e-2,
        help="Learning rate for Lipschitz-refined ridge fitting.",
    )
    ap.add_argument(
        "--ridge_lipschitz_batch",
        type=int,
        default=8192,
        help="MSE mini-batch size for Lipschitz-refined ridge fitting.",
    )
    ap.add_argument(
        "--ridge_lipschitz_max_points",
        type=int,
        default=200000,
        help="Optional cap on train points used during Lipschitz refinement (0=all).",
    )
    ap.add_argument(
        "--ridge_lipschitz_seed",
        type=int,
        default=0,
        help="Random seed for Lipschitz pair/mini-batch sampling.",
    )
    ap.add_argument("--tau", type=float, default=0.03)
    ap.add_argument("--max_shapes", type=int, default=0)
    ap.add_argument("--voxel_grid", type=int, default=64)
    ap.add_argument("--voxel_dilate", type=int, default=1)
    ap.add_argument("--voxel_max_steps", type=int, default=0)
    ap.add_argument(
        "--disjoint_context_query",
        type=int,
        default=1,
        help="If 1, sample context/query indices disjointly when context_mode is normal.",
    )
    ap.add_argument(
        "--context_mode_train",
        type=str,
        default="normal",
        choices=["normal", "none", "mismatch"],
        help="Context mode used when fitting the ridge head.",
    )
    ap.add_argument(
        "--context_mode_test",
        type=str,
        default="normal",
        choices=["normal", "none", "mismatch"],
        help="Context mode used when evaluating on split.",
    )
    ap.add_argument(
        "--mismatch_shift",
        type=int,
        default=1,
        help="Deterministic shift used for mismatched context mapping (i -> i+shift).",
    )
    ap.add_argument(
        "--rep_source",
        type=str,
        default="h",
        choices=["h", "zhat"],
        help="Representation source for ridge head: hidden state h or predicted embedding z_hat.",
    )
    ap.add_argument(
        "--query_source",
        type=str,
        default="pool",
        choices=["pool", "grid", "hybrid"],
        help=(
            "Query source: pool=sample from pt_xyz_pool; "
            "grid=sample voxel centers from udf_grid; "
            "hybrid=pool+grid mixture."
        ),
    )
    ap.add_argument(
        "--query_pool_frac",
        type=float,
        default=0.5,
        help="When query_source=hybrid, fraction assigned to pool sampling.",
    )
    ap.add_argument(
        "--grid_sample_mode",
        type=str,
        default="uniform",
        choices=["uniform", "near_surface", "stratified", "coarse_to_fine"],
        help="Grid-query sampler mode.",
    )
    ap.add_argument(
        "--grid_near_tau",
        type=float,
        default=0.05,
        help="Near-surface threshold for grid_sample_mode=near_surface.",
    )
    ap.add_argument(
        "--grid_near_frac",
        type=float,
        default=0.7,
        help="Near-surface ratio for grid_sample_mode=near_surface.",
    )
    ap.add_argument(
        "--grid_res_schedule",
        type=str,
        default="16,32,64",
        help="Resolution schedule for grid_sample_mode=coarse_to_fine (e.g., 16,32,64).",
    )
    ap.add_argument(
        "--grid_c2f_expand",
        type=int,
        default=1,
        help="Parent-cell neighborhood expansion used by coarse_to_fine refinement.",
    )
    ap.add_argument(
        "--grid_c2f_stage_weights",
        type=str,
        default="",
        help=(
            "Optional stage allocation weights for coarse_to_fine (comma-separated). "
            "If empty, defaults to increasing weights [1..K]."
        ),
    )
    ap.add_argument(
        "--target_transform",
        type=str,
        default="none",
        choices=["none", "trunc", "log1p"],
        help="Optional transform on training targets for ridge probe.",
    )
    ap.add_argument(
        "--target_trunc_max",
        type=float,
        default=0.1,
        help="d_max for target_transform=trunc.",
    )
    ap.add_argument(
        "--target_log_scale",
        type=float,
        default=0.03,
        help="scale sigma for target_transform=log1p: y_t = log(1 + y/sigma).",
    )
    ap.add_argument(
        "--report_near_tau",
        type=float,
        default=0.05,
        help="Also report metrics restricted to y_true <= report_near_tau.",
    )
    ap.add_argument(
        "--baseline",
        type=str,
        default="none",
        choices=["none", "nn_copy"],
        help="Optional baseline to compute.",
    )
    ap.add_argument(
        "--baseline_only",
        type=int,
        default=0,
        help="If 1, skip ridge probe and report baseline only.",
    )

    # Optional: mesh reconstruction + Chamfer/F-score evaluation.
    ap.add_argument(
        "--mesh_eval",
        type=int,
        default=0,
        help="If 1, run mesh reconstruction (marching cubes) + Chamfer/F-score on a subset of test shapes.",
    )
    ap.add_argument(
        "--mesh_eval_max_shapes",
        type=int,
        default=50,
        help="How many test shapes to evaluate mesh metrics on (0=all).",
    )
    ap.add_argument(
        "--mesh_grid_res",
        type=int,
        default=24,
        help="Grid resolution used to reconstruct UDF volume for marching cubes.",
    )
    ap.add_argument(
        "--mesh_chunk_n_query",
        type=int,
        default=512,
        help="Number of grid query points per forward pass when reconstructing the UDF grid.",
    )
    ap.add_argument(
        "--mesh_mc_level",
        type=float,
        default=0.03,
        help="Iso-level for marching cubes on UDF (surface at distance=level).",
    )
    ap.add_argument(
        "--mesh_num_samples",
        type=int,
        default=10000,
        help="Number of surface points sampled per mesh for Chamfer/F-score.",
    )
    ap.add_argument(
        "--mesh_fscore_tau",
        type=float,
        default=0.01,
        help="Distance threshold for F-score in mesh evaluation.",
    )
    ap.add_argument(
        "--mesh_save_dir",
        type=str,
        default="",
        help="If set, save predicted/GT meshes as .ply into this directory.",
    )
    ap.add_argument(
        "--mesh_store_per_shape",
        type=int,
        default=0,
        help="If 1, store per-shape mesh metrics in output json (can be large).",
    )
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    # Backward compatibility for existing wrappers.
    if args.seed is not None:
        args.eval_seed = int(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = build_model_from_ckpt(args.ckpt, device, max_len_override=args.max_len)
    add_eos = infer_add_eos(ckpt, args.add_eos)
    qa_tokens = infer_qa_tokens(ckpt, args.qa_tokens)
    qa_layout = infer_qa_layout(ckpt)

    # Evaluation shapes (never used to train the probe head in the default protocol).
    eval_paths = list_npz(args.cache_root, args.split)
    if args.max_shapes and args.max_shapes > 0:
        eval_paths = eval_paths[: int(args.max_shapes)]
    if len(eval_paths) < 1:
        raise RuntimeError("need at least 1 shape for evaluation")

    head_split = (args.head_train_split or "").strip()
    legacy_transductive = (head_split == "") or (head_split.lower() == "none")

    if not legacy_transductive:
        # Non-transductive protocol: train the ridge head on a *different* split (default: train_udf).
        tr_paths = list_npz(args.cache_root, head_split)
        if args.head_train_max_shapes and args.head_train_max_shapes > 0:
            tr_paths = tr_paths[: int(args.head_train_max_shapes)]
        if len(tr_paths) < 1:
            raise RuntimeError(f"no shapes found for head_train_split={head_split}")
        te_paths = eval_paths
        head_backend = args.head_train_backend
    else:
        # Legacy (transductive) protocol: split within eval shapes by ratio.
        if len(eval_paths) < 2:
            raise RuntimeError("need at least 2 shapes for legacy transductive head split")
        rng = np.random.RandomState(int(args.eval_seed) & 0xFFFFFFFF)
        perm = np.arange(len(eval_paths))
        rng.shuffle(perm)
        if args.head_train_n and args.head_train_n > 0:
            n_train = min(int(args.head_train_n), len(eval_paths) - 1)
        else:
            n_train = int(round(float(args.head_train_ratio) * len(eval_paths)))
            n_train = max(1, min(n_train, len(eval_paths) - 1))
        tr_paths = [eval_paths[i] for i in perm[:n_train]]
        te_paths = [eval_paths[i] for i in perm[n_train:]]
        head_backend = args.context_backend

    baseline = None
    if str(args.baseline).lower() == "nn_copy":
        baseline = _eval_nn_copy_baseline(
            paths=te_paths,
            context_backend=args.context_backend,
            n_context=args.n_context,
            n_query=args.n_query,
            seed=args.eval_seed + 999,
            voxel_grid=args.voxel_grid,
            voxel_dilate=args.voxel_dilate,
            voxel_max_steps=args.voxel_max_steps,
            tau=float(args.tau),
            context_mode=args.context_mode_test,
            disjoint_context_query=bool(args.disjoint_context_query),
            mismatch_shift=int(args.mismatch_shift),
            query_source=args.query_source,
            query_pool_frac=float(args.query_pool_frac),
            grid_sample_mode=str(args.grid_sample_mode),
            grid_near_tau=float(args.grid_near_tau),
            grid_near_frac=float(args.grid_near_frac),
            grid_res_schedule=str(args.grid_res_schedule),
            grid_c2f_expand=int(args.grid_c2f_expand),
            grid_c2f_stage_weights=str(args.grid_c2f_stage_weights),
        )

    if int(args.baseline_only) == 1:
        if baseline is None:
            raise ValueError("--baseline_only=1 requires --baseline != none")
        out = dict(baseline)
        out.update(
            {
                "context_backend": args.context_backend,
                "target": "udfgrid_distance_probe",
                "n_context": int(args.n_context),
                "n_query": int(args.n_query),
                "query_source": str(args.query_source),
                "query_pool_frac": float(args.query_pool_frac),
                "grid_sample_mode": str(args.grid_sample_mode),
                "grid_near_tau": float(args.grid_near_tau),
                "grid_near_frac": float(args.grid_near_frac),
                "grid_res_schedule": str(args.grid_res_schedule),
                "grid_c2f_expand": int(args.grid_c2f_expand),
                "grid_c2f_stage_weights": str(args.grid_c2f_stage_weights),
                "eval_split": args.split,
                "head_train_split": (head_split if not legacy_transductive else "legacy_transductive"),
                "head_train_backend": head_backend,
                "legacy_transductive": bool(legacy_transductive),
                "eval_seed": int(args.eval_seed),
                "n_shapes_total": int(len(eval_paths)),
                "n_shapes_head_train": int(len(tr_paths)),
                "n_shapes_head_test": int(len(te_paths)),
                "ridge_lambda": float(args.ridge_lambda),
                "ridge_lipschitz_lambda": float(args.ridge_lipschitz_lambda),
                "ridge_lipschitz_pairs": int(args.ridge_lipschitz_pairs),
                "ridge_lipschitz_steps": int(args.ridge_lipschitz_steps),
                "ridge_lipschitz_lr": float(args.ridge_lipschitz_lr),
                "ridge_lipschitz_batch": int(args.ridge_lipschitz_batch),
                "ridge_lipschitz_max_points": int(args.ridge_lipschitz_max_points),
                "ridge_lipschitz_seed": int(args.ridge_lipschitz_seed),
                "tau": float(args.tau),
                "disjoint_context_query": bool(args.disjoint_context_query),
                "context_mode_train": str(args.context_mode_train),
                "context_mode_test": str(args.context_mode_test),
                "mismatch_shift": int(args.mismatch_shift),
                "rep_source": str(args.rep_source),
                "target_transform": str(args.target_transform),
                "target_trunc_max": float(args.target_trunc_max),
                "target_log_scale": float(args.target_log_scale),
                "report_near_tau": float(args.report_near_tau),
                "add_eos": bool(add_eos),
                "qa_tokens": bool(qa_tokens),
                "qa_layout": str(qa_layout),
                "ckpt": os.path.abspath(args.ckpt),
                "baseline": baseline,
                "baseline_only": True,
            }
        )

        if int(getattr(args, "mesh_eval", 0)) == 1:
            out["mesh_eval"] = _mesh_eval_chamfer_for_paths(
                model=None,
                ridge_w=None,
                paths=te_paths,
                args=args,
                device=device,
            )

        print(json.dumps(out, indent=2))
        if args.out_json:
            os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"[saved] {args.out_json}")
        return

    Xtr, Ytr, Qtr, Gtr = collect_xy(
        model,
        tr_paths,
        head_backend,
        args.n_context,
        args.n_query,
        add_eos,
        args.eval_seed,
        args.voxel_grid,
        args.voxel_dilate,
        args.voxel_max_steps,
        device,
        qa_tokens,
        qa_layout,
        desc="collect train",
        context_mode=args.context_mode_train,
        disjoint_context_query=bool(args.disjoint_context_query),
        mismatch_shift=int(args.mismatch_shift),
        rep_source=args.rep_source,
        query_source=args.query_source,
        query_pool_frac=float(args.query_pool_frac),
        grid_sample_mode=str(args.grid_sample_mode),
        grid_near_tau=float(args.grid_near_tau),
        grid_near_frac=float(args.grid_near_frac),
        grid_res_schedule=str(args.grid_res_schedule),
        grid_c2f_expand=int(args.grid_c2f_expand),
        grid_c2f_stage_weights=str(args.grid_c2f_stage_weights),
    )
    if args.head_train_n and args.head_train_n > 0 and Xtr.shape[0] > int(args.head_train_n):
        rng_cap = np.random.RandomState(int(args.eval_seed) & 0xFFFFFFFF)
        sel = rng_cap.permutation(Xtr.shape[0])[: int(args.head_train_n)]
        Xtr = Xtr[sel]
        Ytr = Ytr[sel]
        Qtr = Qtr[sel]
        Gtr = Gtr[sel]
    Ytr_t = _transform_target(
        Ytr,
        mode=str(args.target_transform),
        trunc_max=float(args.target_trunc_max),
        log_scale=float(args.target_log_scale),
    )
    lip_lambda = float(args.ridge_lipschitz_lambda)
    lip_info = {"enabled": False}
    if lip_lambda > 0.0 and str(args.target_transform) != "none":
        print(
            "[warn] ridge_lipschitz_lambda > 0 with target_transform != none; "
            "disabling Lipschitz refinement for this run."
        )
        lip_lambda = 0.0
    if lip_lambda > 0.0:
        w, lip_info = _ridge_fit_lipschitz(
            Xtr,
            Ytr_t,
            Qtr,
            Gtr,
            lam=float(args.ridge_lambda),
            lip_lambda=lip_lambda,
            lip_pairs=int(args.ridge_lipschitz_pairs),
            lip_steps=int(args.ridge_lipschitz_steps),
            lip_lr=float(args.ridge_lipschitz_lr),
            lip_batch=int(args.ridge_lipschitz_batch),
            lip_max_points=int(args.ridge_lipschitz_max_points),
            lip_seed=int(args.ridge_lipschitz_seed) ^ int(args.eval_seed),
        )
    else:
        w = _ridge_fit(Xtr, Ytr_t, lam=float(args.ridge_lambda))

    Xte, Yte, _, _ = collect_xy(
        model,
        te_paths,
        args.context_backend,
        args.n_context,
        args.n_query,
        add_eos,
        args.eval_seed + 999,
        args.voxel_grid,
        args.voxel_dilate,
        args.voxel_max_steps,
        device,
        qa_tokens,
        qa_layout,
        desc="collect test",
        context_mode=args.context_mode_test,
        disjoint_context_query=bool(args.disjoint_context_query),
        mismatch_shift=int(args.mismatch_shift),
        rep_source=args.rep_source,
        query_source=args.query_source,
        query_pool_frac=float(args.query_pool_frac),
        grid_sample_mode=str(args.grid_sample_mode),
        grid_near_tau=float(args.grid_near_tau),
        grid_near_frac=float(args.grid_near_frac),
        grid_res_schedule=str(args.grid_res_schedule),
        grid_c2f_expand=int(args.grid_c2f_expand),
        grid_c2f_stage_weights=str(args.grid_c2f_stage_weights),
    )
    Yhat_t = _ridge_pred(Xte, w)
    Yhat = _inverse_transform_target(
        Yhat_t,
        mode=str(args.target_transform),
        trunc_max=float(args.target_trunc_max),
        log_scale=float(args.target_log_scale),
    )

    out = _metrics(Yte, Yhat, tau=float(args.tau))
    out["near@tau_report"] = _metrics_near(
        Yte,
        Yhat,
        tau=float(args.tau),
        near_tau=float(args.report_near_tau),
    )
    tau_t = _transform_tau(
        float(args.tau),
        mode=str(args.target_transform),
        trunc_max=float(args.target_trunc_max),
        log_scale=float(args.target_log_scale),
    )
    out["metrics_transformed"] = _metrics(
        _transform_target(
            Yte,
            mode=str(args.target_transform),
            trunc_max=float(args.target_trunc_max),
            log_scale=float(args.target_log_scale),
        ),
        Yhat_t,
        tau=float(tau_t),
    )
    out.update(
        {
            "context_backend": args.context_backend,
            "target": "udfgrid_distance_probe",
            "n_context": int(args.n_context),
            "n_query": int(args.n_query),
            "query_source": str(args.query_source),
            "query_pool_frac": float(args.query_pool_frac),
            "grid_sample_mode": str(args.grid_sample_mode),
            "grid_near_tau": float(args.grid_near_tau),
            "grid_near_frac": float(args.grid_near_frac),
            "grid_res_schedule": str(args.grid_res_schedule),
            "grid_c2f_expand": int(args.grid_c2f_expand),
            "grid_c2f_stage_weights": str(args.grid_c2f_stage_weights),
            "eval_split": args.split,
            "head_train_split": (head_split if not legacy_transductive else "legacy_transductive"),
            "head_train_backend": head_backend,
            "legacy_transductive": bool(legacy_transductive),
            "eval_seed": int(args.eval_seed),
            "n_shapes_total": int(len(eval_paths)),
            "n_shapes_head_train": int(len(tr_paths)),
            "n_shapes_head_test": int(len(te_paths)),
            "ridge_lambda": float(args.ridge_lambda),
            "ridge_lipschitz_lambda": float(args.ridge_lipschitz_lambda),
            "ridge_lipschitz_pairs": int(args.ridge_lipschitz_pairs),
            "ridge_lipschitz_steps": int(args.ridge_lipschitz_steps),
            "ridge_lipschitz_lr": float(args.ridge_lipschitz_lr),
            "ridge_lipschitz_batch": int(args.ridge_lipschitz_batch),
            "ridge_lipschitz_max_points": int(args.ridge_lipschitz_max_points),
            "ridge_lipschitz_seed": int(args.ridge_lipschitz_seed),
            "ridge_lipschitz_info": lip_info,
            "tau": float(args.tau),
            "disjoint_context_query": bool(args.disjoint_context_query),
            "context_mode_train": str(args.context_mode_train),
            "context_mode_test": str(args.context_mode_test),
            "mismatch_shift": int(args.mismatch_shift),
            "rep_source": str(args.rep_source),
            "target_transform": str(args.target_transform),
            "target_trunc_max": float(args.target_trunc_max),
            "target_log_scale": float(args.target_log_scale),
            "report_near_tau": float(args.report_near_tau),
            "add_eos": bool(add_eos),
            "qa_tokens": bool(qa_tokens),
            "qa_layout": str(qa_layout),
            "ckpt": os.path.abspath(args.ckpt),
            "baseline": baseline,
        }
    )

    if int(getattr(args, "mesh_eval", 0)) == 1:
        out["mesh_eval"] = _mesh_eval_chamfer_for_paths(
            model=model,
            ridge_w=w,
            paths=te_paths,
            args=args,
            device=device,
        )

    print(json.dumps(out, indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
