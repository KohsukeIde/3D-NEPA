from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import numpy as np
import torch

from ..backends.mesh_backend import MeshBackend
from ..backends.pointcloud_backend import (
    PointCloudBackend,
    PointCloudMeshRayBackend,
    PointCloudNoRayBackend,
)
from ..backends.udfgrid_backend import UDFGridBackend
from ..backends.voxel_backend import VoxelBackend
from ..models.kplane import build_kplane_from_ckpt
from ..token.ordering import morton3d

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def list_npz(cache_root: str, split: str):
    return sorted(glob.glob(os.path.join(cache_root, split, "*", "*.npz")))


def _stable_seed(path: str, eval_seed: int, k: int):
    h = hashlib.md5(path.encode("utf-8")).hexdigest()[:8]
    base = int(h, 16)
    return (base + int(eval_seed) + 1000003 * int(k)) & 0xFFFFFFFF


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


def _sample_ctx_qry(
    pt_xyz_query,
    pt_dist_query,
    pt_xyz_context,
    pt_dist_context,
    n_context,
    n_query,
    rng,
    disjoint_context_query=True,
):
    n_q_pool = int(pt_xyz_query.shape[0])
    n_c_pool = int(pt_xyz_context.shape[0])
    use_disjoint = bool(disjoint_context_query) and (n_q_pool == n_c_pool)

    if use_disjoint:
        total = int(n_context) + int(n_query)
        replace = n_q_pool < total
        idx = rng.choice(n_q_pool, size=total, replace=replace).astype(np.int64)
        cidx = idx[: int(n_context)]
        qidx = idx[int(n_context):]
    else:
        cidx = rng.choice(n_c_pool, size=int(n_context), replace=n_c_pool < int(n_context)).astype(np.int64)
        qidx = rng.choice(n_q_pool, size=int(n_query), replace=n_q_pool < int(n_query)).astype(np.int64)

    ctx_xyz = pt_xyz_context[cidx].astype(np.float32, copy=False)
    ctx_dist = pt_dist_context[cidx].astype(np.float32, copy=False)
    qry_xyz = pt_xyz_query[qidx].astype(np.float32, copy=False)
    if ctx_xyz.shape[0] > 0:
        order = np.argsort(morton3d(ctx_xyz))
        ctx_xyz = ctx_xyz[order]
        ctx_dist = ctx_dist[order]
    if qry_xyz.shape[0] > 0:
        order = np.argsort(morton3d(qry_xyz))
        qry_xyz = qry_xyz[order]
    return ctx_xyz, ctx_dist, qry_xyz


@torch.no_grad()
def embed_path(
    model,
    path,
    backend,
    n_context,
    n_query,
    eval_seed,
    mc_k,
    voxel_grid,
    voxel_dilate,
    voxel_max_steps,
    device,
    pooling: str = "mean_query",
    ablate_query_xyz: bool = False,
    ablate_context_dist: bool = False,
    context_mode: str = "normal",
    mismatch_path: str | None = None,
    disjoint_context_query: bool = True,
):
    descs = []
    for k in range(max(1, int(mc_k))):
        rng = np.random.RandomState(_stable_seed(path, eval_seed, k))
        be_q = make_backend(backend, path, voxel_grid, voxel_dilate, voxel_max_steps)
        pools_q = be_q.get_pools()

        if context_mode == "none":
            ctx_xyz = np.zeros((0, 3), dtype=np.float32)
            ctx_dist = np.zeros((0,), dtype=np.float32)
            n_q_pool = int(pools_q["pt_xyz_pool"].shape[0])
            qidx = rng.choice(n_q_pool, size=int(n_query), replace=n_q_pool < int(n_query)).astype(np.int64)
            qry_xyz = pools_q["pt_xyz_pool"][qidx].astype(np.float32, copy=False)
            if qry_xyz.shape[0] > 0:
                order = np.argsort(morton3d(qry_xyz))
                qry_xyz = qry_xyz[order]
        else:
            ctx_path = mismatch_path if (context_mode == "mismatch" and mismatch_path is not None) else path
            be_c = make_backend(backend, ctx_path, voxel_grid, voxel_dilate, voxel_max_steps)
            pools_c = be_c.get_pools()
            ctx_xyz, ctx_dist, qry_xyz = _sample_ctx_qry(
                pools_q["pt_xyz_pool"],
                pools_q["pt_dist_pool"],
                pools_c["pt_xyz_pool"],
                pools_c["pt_dist_pool"],
                n_context=n_context,
                n_query=n_query,
                rng=rng,
                disjoint_context_query=bool(disjoint_context_query),
            )

        if context_mode not in ("normal", "none", "mismatch"):
            raise ValueError(f"unknown context_mode: {context_mode}")

        if qry_xyz.shape[0] == 0:
            # Avoid empty query pooling edge case.
            continue
        if ctx_xyz.shape[0] == 0:
            ctx_xyz = np.zeros((0, 3), dtype=np.float32)
            ctx_dist = np.zeros((0,), dtype=np.float32)

        ctx_xyz_t = torch.from_numpy(ctx_xyz).unsqueeze(0).to(device).float()
        ctx_dist_t = torch.from_numpy(ctx_dist).unsqueeze(0).to(device).float()
        qry_xyz_t = torch.from_numpy(qry_xyz).unsqueeze(0).to(device).float()

        _, q_feat, _, planes = model(
            ctx_xyz_t,
            ctx_dist_t,
            qry_xyz_t,
            ablate_query_xyz=bool(ablate_query_xyz),
            ablate_context_dist=bool(ablate_context_dist),
        )

        if pooling == "mean_query":
            v = q_feat.mean(dim=1)
        elif pooling == "plane_gap":
            v = model.global_descriptor_from_planes(planes)
        else:
            raise ValueError(f"unknown pooling: {pooling}")

        v = torch.nn.functional.normalize(v, dim=-1)
        descs.append(v.squeeze(0).detach().cpu().numpy().astype(np.float32))

    out = np.mean(np.stack(descs, axis=0), axis=0)
    out = out / (np.linalg.norm(out) + 1e-9)
    return out.astype(np.float32)


def _ranks_from_score_tieaware(score: np.ndarray, target_idx: np.ndarray) -> np.ndarray:
    order = np.argsort(-score, axis=1, kind="mergesort")
    inv = np.empty_like(order, dtype=np.int32)
    inv[np.arange(order.shape[0])[:, None], order] = np.arange(order.shape[1], dtype=np.int32)[None, :]
    return inv[np.arange(order.shape[0]), target_idx.astype(np.int32)] + 1


def retrieval_metrics(query, gallery, ks=(1, 5, 10), chunk=256, tie_seed=0, tie_break_eps=1e-6):
    q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-9)
    g = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-9)
    n = q.shape[0]
    ranks = np.zeros((n,), dtype=np.int32)
    rng = np.random.RandomState(int(tie_seed) & 0xFFFFFFFF)

    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        score = q[s:e] @ g.T
        if float(tie_break_eps) > 0.0:
            score = score + float(tie_break_eps) * rng.standard_normal(score.shape).astype(np.float32, copy=False)
        target_idx = np.arange(s, e, dtype=np.int32)
        ranks[s:e] = _ranks_from_score_tieaware(score, target_idx)

    out = {}
    for k in ks:
        out[f"r@{int(k)}"] = float(np.mean(ranks <= int(k)))
    out["mAP"] = float(np.mean(1.0 / ranks.astype(np.float32)))
    out["mean_rank"] = float(np.mean(ranks))
    out["median_rank"] = float(np.median(ranks))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="eval")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--query_backend", type=str, required=True)
    ap.add_argument("--gallery_backend", type=str, required=True)
    ap.add_argument("--n_context", type=int, default=256)
    ap.add_argument("--n_query", type=int, default=256)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--eval_seed_gallery", type=int, default=-1)
    ap.add_argument("--mc_k", type=int, default=1)
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--voxel_grid", type=int, default=64)
    ap.add_argument("--voxel_dilate", type=int, default=1)
    ap.add_argument("--voxel_max_steps", type=int, default=0)
    ap.add_argument("--pooling", type=str, default="mean_query", choices=["mean_query", "plane_gap"])
    ap.add_argument("--ablate_query_xyz", action="store_true")
    ap.add_argument("--ablate_context_dist", action="store_true")
    ap.add_argument("--disjoint_context_query", type=int, default=1, choices=[0, 1])
    ap.add_argument("--context_mode_query", type=str, default="normal", choices=["normal", "none", "mismatch"])
    ap.add_argument("--context_mode_gallery", type=str, default="normal", choices=["normal", "none", "mismatch"])
    ap.add_argument("--mismatch_shift_query", type=int, default=1)
    ap.add_argument("--mismatch_shift_gallery", type=int, default=1)
    ap.add_argument("--tie_break_eps", type=float, default=1e-6)
    ap.add_argument(
        "--sanity_constant_embed",
        action="store_true",
        help="Sanity mode: replace all embeddings with a constant vector to verify rank behaves ~random.",
    )
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = build_kplane_from_ckpt(args.ckpt, device)

    paths = list_npz(args.cache_root, args.split)
    if args.max_files and int(args.max_files) > 0:
        paths = paths[: int(args.max_files)]
    if len(paths) < 2:
        raise RuntimeError("need at least 2 files for retrieval")

    eval_seed_gallery = int(args.eval_seed_gallery)
    if eval_seed_gallery < 0:
        eval_seed_gallery = int(args.eval_seed)

    def _build_mismatch_paths(mode: str, shift: int):
        out = [None for _ in range(len(paths))]
        if mode == "mismatch" and len(paths) > 1:
            s = int(shift) % len(paths)
            if s == 0:
                s = 1
            out = [paths[(i + s) % len(paths)] for i in range(len(paths))]
        return out

    mismatch_q = _build_mismatch_paths(args.context_mode_query, int(args.mismatch_shift_query))
    mismatch_g = _build_mismatch_paths(args.context_mode_gallery, int(args.mismatch_shift_gallery))

    Q = []
    G = []
    for i, p in enumerate(tqdm(paths, desc=f"embed query={args.query_backend}")):
        Q.append(
            embed_path(
                model,
                p,
                args.query_backend,
                n_context=args.n_context,
                n_query=args.n_query,
                eval_seed=int(args.eval_seed),
                mc_k=args.mc_k,
                voxel_grid=args.voxel_grid,
                voxel_dilate=args.voxel_dilate,
                voxel_max_steps=args.voxel_max_steps,
                device=device,
                pooling=args.pooling,
                ablate_query_xyz=bool(args.ablate_query_xyz),
                ablate_context_dist=bool(args.ablate_context_dist),
                context_mode=str(args.context_mode_query),
                mismatch_path=mismatch_q[i],
                disjoint_context_query=bool(int(args.disjoint_context_query)),
            )
        )
    for i, p in enumerate(tqdm(paths, desc=f"embed gallery={args.gallery_backend}")):
        G.append(
            embed_path(
                model,
                p,
                args.gallery_backend,
                n_context=args.n_context,
                n_query=args.n_query,
                eval_seed=eval_seed_gallery,
                mc_k=args.mc_k,
                voxel_grid=args.voxel_grid,
                voxel_dilate=args.voxel_dilate,
                voxel_max_steps=args.voxel_max_steps,
                device=device,
                pooling=args.pooling,
                ablate_query_xyz=bool(args.ablate_query_xyz),
                ablate_context_dist=bool(args.ablate_context_dist),
                context_mode=str(args.context_mode_gallery),
                mismatch_path=mismatch_g[i],
                disjoint_context_query=bool(int(args.disjoint_context_query)),
            )
        )

    Q = np.stack(Q, axis=0)
    G = np.stack(G, axis=0)
    if bool(args.sanity_constant_embed):
        Q.fill(1.0)
        G.fill(1.0)

    tie_seed = int(args.eval_seed) * 1000003 + int(eval_seed_gallery)
    out = retrieval_metrics(
        Q,
        G,
        ks=(1, 5, 10),
        chunk=int(args.chunk),
        tie_seed=tie_seed,
        tie_break_eps=float(args.tie_break_eps),
    )
    out.update(
        {
            "query_backend": args.query_backend,
            "gallery_backend": args.gallery_backend,
            "n_files": int(len(paths)),
            "n_context": int(args.n_context),
            "n_query": int(args.n_query),
            "mc_k": int(args.mc_k),
            "eval_seed": int(args.eval_seed),
            "eval_seed_gallery": int(eval_seed_gallery),
            "pooling": str(args.pooling),
            "ablate_query_xyz": bool(args.ablate_query_xyz),
            "ablate_context_dist": bool(args.ablate_context_dist),
            "disjoint_context_query": bool(int(args.disjoint_context_query)),
            "context_mode_query": str(args.context_mode_query),
            "context_mode_gallery": str(args.context_mode_gallery),
            "mismatch_shift_query": int(args.mismatch_shift_query),
            "mismatch_shift_gallery": int(args.mismatch_shift_gallery),
            "tie_break_eps": float(args.tie_break_eps),
            "sanity_constant_embed": bool(args.sanity_constant_embed),
            "expected_random_r@1": float(1.0 / max(1, len(paths))),
            "ckpt": os.path.abspath(args.ckpt),
            "kplane_cfg": ckpt.get("kplane_cfg", {}),
        }
    )
    print(json.dumps(out, indent=2))

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
