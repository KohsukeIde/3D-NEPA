import argparse
import glob
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
from ..models.query_nepa import QueryNepa
from ..token.tokenizer import TYPE_EOS, TYPE_POINT, build_sequence

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def list_npz(cache_root, split):
    return sorted(glob.glob(os.path.join(cache_root, split, "*", "*.npz")))


def _stable_seed(path, eval_seed, k):
    import hashlib

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


def infer_add_eos(ckpt, add_eos_arg):
    if add_eos_arg >= 0:
        return bool(add_eos_arg)
    pre_args = ckpt.get("args", {})
    ckpt_n_types = ckpt["model"]["type_emb.weight"].shape[0]
    return bool(pre_args.get("add_eos", ckpt_n_types >= 5))


def build_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pre_args = ckpt.get("args", {})
    state = ckpt["model"]
    d_model = state["type_emb.weight"].shape[1]
    n_types = state["type_emb.weight"].shape[0]
    nhead = int(pre_args.get("heads", 6))
    num_layers = int(pre_args.get("layers", 8))
    max_len = int(state["pos_emb"].shape[1])

    model = QueryNepa(
        feat_dim=15,
        d_model=d_model,
        n_types=n_types,
        nhead=nhead,
        num_layers=num_layers,
        max_len=max_len,
    )
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, ckpt


@torch.no_grad()
def embed_path(
    model,
    path,
    backend,
    n_point,
    n_ray,
    add_eos,
    eval_seed,
    mc_k,
    voxel_grid,
    voxel_dilate,
    voxel_max_steps,
    device,
    ablate_point_xyz: bool = False,
    ablate_point_dist: bool = False,
):
    embs = []
    for k in range(max(1, int(mc_k))):
        rng = np.random.RandomState(_stable_seed(path, eval_seed, k))
        be = make_backend(backend, path, voxel_grid, voxel_dilate, voxel_max_steps)
        pools = be.get_pools()
        ray_available = bool(pools.get("ray_available", True))

        feat, type_id = build_sequence(
            pools["pt_xyz_pool"],
            pools["pt_dist_pool"],
            pools["ray_o_pool"],
            pools["ray_d_pool"],
            pools["ray_hit_pool"],
            pools["ray_t_pool"],
            pools["ray_n_pool"],
            n_point=n_point,
            n_ray=n_ray,
            drop_ray_prob=0.0,
            ray_available=ray_available,
            add_eos=add_eos,
            rng=rng,
        )

        # Optional ablations (diagnostic): remove access to POINT xyz/dist.
        # This helps detect trivial retrieval caused by shared query coordinates.
        if ablate_point_xyz or ablate_point_dist:
            pt_mask = (type_id == TYPE_POINT)
            if ablate_point_xyz:
                feat[pt_mask, 0:3] = 0.0
            if ablate_point_dist:
                feat[pt_mask, 9] = 0.0

        feat_t = torch.from_numpy(feat).unsqueeze(0).to(device).float()
        type_t = torch.from_numpy(type_id).unsqueeze(0).to(device).long()
        _, _, h = model(feat_t, type_t)

        if add_eos and type_id.shape[0] > 0 and int(type_id[-1]) == TYPE_EOS:
            v = h[:, -1, :]
        else:
            v = h[:, -1, :]
        v = torch.nn.functional.normalize(v, dim=-1)
        embs.append(v.squeeze(0).detach().cpu().numpy().astype(np.float32))

    out = np.mean(np.stack(embs, axis=0), axis=0)
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
    ap.add_argument("--n_point", type=int, default=None)
    ap.add_argument("--n_ray", type=int, default=None)
    ap.add_argument("--add_eos", type=int, default=-1)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument(
        "--eval_seed_gallery",
        type=int,
        default=-1,
        help="If <0, uses --eval_seed. Use a different seed to avoid identical sampling for query/gallery.",
    )
    ap.add_argument("--mc_k", type=int, default=1)
    ap.add_argument(
        "--ablate_point_xyz",
        action="store_true",
        help="Zero-out POINT xyz features before encoding (diagnostic).",
    )
    ap.add_argument(
        "--ablate_point_dist",
        action="store_true",
        help="Zero-out POINT dist features before encoding (diagnostic).",
    )
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--voxel_grid", type=int, default=64)
    ap.add_argument("--voxel_dilate", type=int, default=1)
    ap.add_argument("--voxel_max_steps", type=int, default=0)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    eval_seed_gallery = (
        int(args.eval_seed)
        if int(args.eval_seed_gallery) < 0
        else int(args.eval_seed_gallery)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = build_model_from_ckpt(args.ckpt, device)
    pre_args = ckpt.get("args", {})

    add_eos = infer_add_eos(ckpt, args.add_eos)
    n_point = int(pre_args.get("n_point", 256) if args.n_point is None else args.n_point)
    n_ray = int(pre_args.get("n_ray", 256) if args.n_ray is None else args.n_ray)

    paths = list_npz(args.cache_root, args.split)
    if args.max_files and args.max_files > 0:
        paths = paths[: int(args.max_files)]
    if not paths:
        raise RuntimeError(f"no npz found under {args.cache_root}/{args.split}")

    q_emb = np.zeros((len(paths), model.d_model), dtype=np.float32)
    g_emb = np.zeros((len(paths), model.d_model), dtype=np.float32)

    for i, p in enumerate(tqdm(paths, desc=f"embed query={args.query_backend}")):
        q_emb[i] = embed_path(
            model,
            p,
            args.query_backend,
            n_point,
            n_ray,
            add_eos,
            args.eval_seed,
            args.mc_k,
            args.voxel_grid,
            args.voxel_dilate,
            args.voxel_max_steps,
            device,
            args.ablate_point_xyz,
            args.ablate_point_dist,
        )

    for i, p in enumerate(tqdm(paths, desc=f"embed gallery={args.gallery_backend}")):
        g_emb[i] = embed_path(
            model,
            p,
            args.gallery_backend,
            n_point,
            n_ray,
            add_eos,
            eval_seed_gallery,
            args.mc_k,
            args.voxel_grid,
            args.voxel_dilate,
            args.voxel_max_steps,
            device,
            args.ablate_point_xyz,
            args.ablate_point_dist,
        )

    metrics = retrieval_metrics(q_emb, g_emb, ks=(1, 5, 10), chunk=max(1, int(args.chunk)))
    metrics.update(
        {
            "query_backend": args.query_backend,
            "gallery_backend": args.gallery_backend,
            "n_files": int(len(paths)),
            "n_point": int(n_point),
            "n_ray": int(n_ray),
            "add_eos": bool(add_eos),
            "mc_k": int(args.mc_k),
            "eval_seed": int(args.eval_seed),
            "eval_seed_gallery": int(eval_seed_gallery),
            "ablate_point_xyz": bool(args.ablate_point_xyz),
            "ablate_point_dist": bool(args.ablate_point_dist),
            "ckpt": os.path.abspath(args.ckpt),
        }
    )

    print(json.dumps(metrics, indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
