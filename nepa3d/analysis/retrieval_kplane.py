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


def _sample_ctx_qry(pt_xyz_pool, pt_dist_pool, n_context, n_query, rng):
    n_pool = int(pt_xyz_pool.shape[0])
    total = int(n_context) + int(n_query)
    replace = n_pool < total
    idx = rng.choice(n_pool, size=total, replace=replace).astype(np.int64)
    cidx = idx[: int(n_context)]
    qidx = idx[int(n_context):]
    ctx_xyz = pt_xyz_pool[cidx].astype(np.float32, copy=False)
    ctx_dist = pt_dist_pool[cidx].astype(np.float32, copy=False)
    qry_xyz = pt_xyz_pool[qidx].astype(np.float32, copy=False)
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
):
    descs = []
    for k in range(max(1, int(mc_k))):
        rng = np.random.RandomState(_stable_seed(path, eval_seed, k))
        be = make_backend(backend, path, voxel_grid, voxel_dilate, voxel_max_steps)
        pools = be.get_pools()

        ctx_xyz, ctx_dist, qry_xyz = _sample_ctx_qry(
            pools["pt_xyz_pool"],
            pools["pt_dist_pool"],
            n_context=n_context,
            n_query=n_query,
            rng=rng,
        )

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


def retrieval_metrics(query, gallery, ks=(1, 5, 10), chunk=256):
    q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-9)
    g = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-9)
    n = q.shape[0]
    ranks = np.zeros((n,), dtype=np.int32)

    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        score = q[s:e] @ g.T
        corr = score[np.arange(e - s), np.arange(s, e)]
        ranks[s:e] = 1 + np.sum(score > corr[:, None], axis=1).astype(np.int32)

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

    Q = []
    G = []
    for p in tqdm(paths, desc=f"embed query={args.query_backend}"):
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
            )
        )
    for p in tqdm(paths, desc=f"embed gallery={args.gallery_backend}"):
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
            )
        )

    Q = np.stack(Q, axis=0)
    G = np.stack(G, axis=0)
    out = retrieval_metrics(Q, G, ks=(1, 5, 10), chunk=int(args.chunk))
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
