"""Preprocess ShapeNet meshes into the v2 NPZ format.

This is a *PatchNEPA-oriented* data generator.

Key idea
--------
Store **surface context points** separately from **primitive-specific query points**.

One NPZ (one shape) can contain multiple query sets:
- mesh queries: `mesh_qry_xyz` + normals/curvature proxy
- udf queries:  `udf_qry_xyz`  + unsigned distance + gradient proxy
- pc context/query: `pc_xyz`, `pc_qry_xyz` + PCA normals + density proxy
- ray queries (optional): `ray_o`, `ray_d` + hit/t/normal

Unpaired pretraining
--------------------
Unpairedness is enforced by *splitting shapes into disjoint subsets per primitive*
(see `shapenet_unpaired_split.py` + `preprocess_shapenet_unpaired.py`).
This script intentionally writes a **single NPZ per shape** containing all fields;
which primitive is used during pretrain is decided by the dataset spec (keys
and answer schema).

This mirrors the original codebase design where one NPZ contained all modalities.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import trimesh
except Exception as e:
    raise RuntimeError(
        "preprocess_shapenet_v2 requires trimesh. Install trimesh or run in the project env."
    ) from e

from tqdm import tqdm

from .preprocess_modelnet40 import normalize_mesh
from .convert_npz_to_v2 import estimate_density_knn, estimate_normals_pca


def _infer_synset_model(mesh_path: str) -> Tuple[str, str]:
    # .../<synset>/<model>/models/model_normalized.obj
    parts = mesh_path.replace("\\", "/").split("/")
    if len(parts) < 4:
        return "unknown", os.path.splitext(os.path.basename(mesh_path))[0]
    synset = parts[-4]
    model_id = parts[-3]
    return synset, model_id


def _unit_vectors(rng: np.random.RandomState, n: int) -> np.ndarray:
    v = rng.normal(size=(n, 3)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    return v


def _sample_surface(mesh: "trimesh.Trimesh", n: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    # trimesh.sample.sample_surface uses global np.random by default; we want deterministic per-shape.
    # We implement by temporarily seeding.
    state = np.random.get_state()
    np.random.seed(rng.randint(0, 2**31 - 1))
    try:
        pts, face_idx = trimesh.sample.sample_surface(mesh, n)
    finally:
        np.random.set_state(state)
    return pts.astype(np.float32), face_idx.astype(np.int64)


def _closest_point_and_face(mesh: "trimesh.Trimesh", xyz: np.ndarray, *, chunk: int = 200000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return closest points, distances, and face indices (chunked)."""
    xyz = np.asarray(xyz, dtype=np.float32)
    cps = np.zeros_like(xyz)
    dist = np.zeros((xyz.shape[0],), dtype=np.float32)
    face = np.zeros((xyz.shape[0],), dtype=np.int64)
    for i in range(0, xyz.shape[0], chunk):
        sl = slice(i, min(i + chunk, xyz.shape[0]))
        cp, d, f = trimesh.proximity.closest_point(mesh, xyz[sl])
        cps[sl] = cp.astype(np.float32)
        dist[sl] = d.astype(np.float32)
        face[sl] = f.astype(np.int64)
    return cps, dist, face


def _normal_variation_curv(xyz: np.ndarray, normals: np.ndarray, *, k: int = 20) -> np.ndarray:
    """Curvature proxy via local normal variation.

    Returns a scalar per point in [0, 2] roughly.
    0 => locally planar (normals aligned)
    larger => normals vary
    """
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(xyz)
        kk = min(k + 1, xyz.shape[0])
        _, nn = tree.query(xyz, k=kk)
        nn = nn[:, 1:]
        n0 = normals
        # compute mean 1 - |dot|
        dots = np.abs((n0[:, None, :] * normals[nn]).sum(axis=-1))  # [N,k]
        curv = 1.0 - dots.mean(axis=1)
        return curv.reshape(-1, 1).astype(np.float32)
    except Exception:
        # fallback: no curvature
        return np.zeros((xyz.shape[0], 1), dtype=np.float32)


def _make_partial_noisy_pc(
    surf_xyz: np.ndarray,
    *,
    n_pc: int,
    rng: np.random.RandomState,
    view_crop: float,
    noise_std: float,
    dropout: float,
) -> np.ndarray:
    """Simple scan-like degradation: front-side crop + gaussian noise + dropout."""
    xyz = np.asarray(surf_xyz, dtype=np.float32)

    # crop by random view direction (keep points with dot(p, v) >= thresh)
    v = _unit_vectors(rng, 1)[0]
    proj = (xyz * v[None, :]).sum(axis=1)
    thresh = np.quantile(proj, 1.0 - float(view_crop))  # view_crop=0.5 -> median (half)
    keep = proj >= thresh
    cropped = xyz[keep]
    if cropped.shape[0] < max(64, int(0.2 * n_pc)):
        cropped = xyz

    # dropout
    if dropout > 0.0 and cropped.shape[0] > 0:
        m = rng.rand(cropped.shape[0]) >= float(dropout)
        cropped = cropped[m]
        if cropped.shape[0] == 0:
            cropped = xyz

    # sample to fixed size
    if cropped.shape[0] >= n_pc:
        idx = _choice_np(cropped.shape[0], n_pc, rng)
        pc = cropped[idx]
    else:
        # pad with replacement
        idx = rng.choice(cropped.shape[0], size=n_pc, replace=True)
        pc = cropped[idx]

    # noise
    if noise_std > 0.0:
        pc = pc + rng.normal(scale=float(noise_std), size=pc.shape).astype(np.float32)

    # clamp to bounds
    pc = np.clip(pc, -1.0, 1.0)
    return pc.astype(np.float32)


def _choice_np(n: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    if k >= n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=k, replace=False).astype(np.int64)


def _sample_udf_queries(
    surf_xyz: np.ndarray,
    *,
    n_qry: int,
    rng: np.random.RandomState,
    near_ratio: float,
    near_std: float,
) -> np.ndarray:
    n_near = int(round(float(n_qry) * float(near_ratio)))
    n_uni = int(n_qry) - n_near

    uni = rng.uniform(-1.0, 1.0, size=(n_uni, 3)).astype(np.float32)

    if n_near > 0:
        idx = rng.choice(surf_xyz.shape[0], size=n_near, replace=True)
        near = surf_xyz[idx] + rng.normal(scale=float(near_std), size=(n_near, 3)).astype(np.float32)
        near = np.clip(near, -1.0, 1.0)
        xyz = np.concatenate([uni, near], axis=0)
    else:
        xyz = uni

    # shuffle
    perm = rng.permutation(xyz.shape[0])
    return xyz[perm].astype(np.float32)


def _compute_udf_dist_grad(mesh: "trimesh.Trimesh", qry_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cp, dist, face = _closest_point_and_face(mesh, qry_xyz)

    # grad: direction away from closest surface point
    v = qry_xyz - cp
    denom = dist.reshape(-1, 1) + 1e-8
    grad = v / denom

    # handle (near-)zero distance by using face normal
    zmask = dist < 1e-6
    if np.any(zmask):
        fn = mesh.face_normals[face[zmask]].astype(np.float32)
        fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8
        grad[zmask] = fn

    return dist.reshape(-1, 1).astype(np.float32), grad.astype(np.float32)


def _sample_rays(
    *,
    n_rays: int,
    rng: np.random.RandomState,
    radius: float = 2.5,
    jitter_std: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    o = _unit_vectors(rng, n_rays) * float(radius)
    jitter = rng.normal(scale=float(jitter_std), size=o.shape).astype(np.float32)
    d = -o + jitter
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-8
    return o.astype(np.float32), d.astype(np.float32)


def _ray_intersect(mesh: "trimesh.Trimesh", ray_o: np.ndarray, ray_d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, hit, normal) per ray."""
    # Use trimesh's pure triangle intersector for portability.
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    loc, idx_ray, idx_tri = intersector.intersects_location(ray_o, ray_d, multiple_hits=False)

    n = ray_o.shape[0]
    hit = np.zeros((n, 1), dtype=np.float32)
    t = np.zeros((n, 1), dtype=np.float32)
    nrm = np.zeros((n, 3), dtype=np.float32)

    if loc.shape[0] > 0:
        hit[idx_ray, 0] = 1.0
        # distance along ray
        t[idx_ray, 0] = np.linalg.norm(loc - ray_o[idx_ray], axis=1).astype(np.float32)
        fn = mesh.face_normals[idx_tri].astype(np.float32)
        fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8
        nrm[idx_ray] = fn

    return t, hit, nrm


@dataclass
class V2GenConfig:
    n_surf: int = 8192
    n_mesh_qry: int = 2048
    n_udf_qry: int = 8192
    n_pc: int = 2048
    n_pc_qry: int = 1024
    n_rays: int = 4096

    pc_view_crop: float = 0.5
    pc_noise_std: float = 0.005
    pc_dropout: float = 0.1

    udf_near_ratio: float = 0.5
    udf_near_std: float = 0.05

    curvature_knn: int = 20
    pca_knn: int = 20

    ray_radius: float = 2.5
    ray_jitter_std: float = 0.05
    skip_existing: bool = False


def preprocess_one(mesh_path: str, out_path: str, *, cfg: V2GenConfig, seed: int) -> Optional[str]:
    synset, model_id = _infer_synset_model(mesh_path)
    rng = np.random.RandomState(seed)
    if bool(cfg.skip_existing) and os.path.isfile(out_path):
        return None

    try:
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            # some files load as Scene
            mesh = mesh.dump().sum()
        mesh = normalize_mesh(mesh)
        if mesh.faces.shape[0] == 0 or mesh.vertices.shape[0] == 0:
            return f"empty mesh: {mesh_path}"

        # --- surface context ---
        surf_xyz, _ = _sample_surface(mesh, int(cfg.n_surf), rng)

        # --- mesh query: normals + curvature proxy ---
        mesh_qry_xyz, mesh_qry_f = _sample_surface(mesh, int(cfg.n_mesh_qry), rng)
        mesh_qry_n = mesh.face_normals[mesh_qry_f].astype(np.float32)
        mesh_qry_n /= np.linalg.norm(mesh_qry_n, axis=1, keepdims=True) + 1e-8
        mesh_qry_curv = _normal_variation_curv(mesh_qry_xyz, mesh_qry_n, k=int(cfg.curvature_knn))

        # --- UDF queries: distance + gradient proxy ---
        udf_qry_xyz = _sample_udf_queries(
            surf_xyz,
            n_qry=int(cfg.n_udf_qry),
            rng=rng,
            near_ratio=float(cfg.udf_near_ratio),
            near_std=float(cfg.udf_near_std),
        )
        udf_qry_dist, udf_qry_grad = _compute_udf_dist_grad(mesh, udf_qry_xyz)

        # --- point cloud context/query: PCA normal + density ---
        pc_xyz = _make_partial_noisy_pc(
            surf_xyz,
            n_pc=int(cfg.n_pc),
            rng=rng,
            view_crop=float(cfg.pc_view_crop),
            noise_std=float(cfg.pc_noise_std),
            dropout=float(cfg.pc_dropout),
        )
        # pc queries are a subset of pc points
        pc_qry_idx = _choice_np(pc_xyz.shape[0], int(cfg.n_pc_qry), rng)
        pc_qry_xyz = pc_xyz[pc_qry_idx]

        pc_n_all = estimate_normals_pca(pc_xyz, k=int(cfg.pca_knn))
        pc_d_all = estimate_density_knn(pc_xyz, k=int(cfg.pca_knn))
        if pc_n_all is None:
            pc_n_all = np.zeros_like(pc_xyz, dtype=np.float32)
        if pc_d_all is None:
            pc_d_all = np.zeros((pc_xyz.shape[0], 1), dtype=np.float32)
        pc_qry_n = pc_n_all[pc_qry_idx].astype(np.float32)
        pc_qry_density = pc_d_all[pc_qry_idx].astype(np.float32)

        # --- rays (optional) ---
        ray_o, ray_d = _sample_rays(
            n_rays=int(cfg.n_rays),
            rng=rng,
            radius=float(cfg.ray_radius),
            jitter_std=float(cfg.ray_jitter_std),
        )
        ray_t, ray_hit, ray_n = _ray_intersect(mesh, ray_o, ray_d)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez(
            out_path,
            v2=np.int32(2),
            synset=np.string_(synset),
            model_id=np.string_(model_id),
            surf_xyz=surf_xyz,
            # point cloud
            pc_xyz=pc_xyz,
            pc_qry_xyz=pc_qry_xyz,
            pc_qry_n=pc_qry_n,
            pc_qry_density=pc_qry_density,
            # mesh
            mesh_qry_xyz=mesh_qry_xyz,
            mesh_qry_n=mesh_qry_n,
            mesh_qry_curv=mesh_qry_curv,
            # udf
            udf_qry_xyz=udf_qry_xyz,
            udf_qry_dist=udf_qry_dist,
            udf_qry_grad=udf_qry_grad,
            # rays
            ray_o=ray_o,
            ray_d=ray_d,
            ray_t=ray_t,
            ray_hit=ray_hit,
            ray_n=ray_n,
        )
        return None

    except Exception as e:
        return f"{mesh_path}: {type(e).__name__}: {e}"


def _glob_meshes(shapenet_root: str, synsets: Optional[List[str]] = None) -> List[str]:
    meshes: List[str] = []
    if synsets is None or len(synsets) == 0:
        pattern = os.path.join(shapenet_root, "*", "*", "models", "model_normalized.obj")
        meshes = sorted([p for p in glob_glob(pattern)])
    else:
        for syn in synsets:
            pattern = os.path.join(shapenet_root, syn, "*", "models", "model_normalized.obj")
            meshes.extend(glob_glob(pattern))
        meshes = sorted(meshes)
    return meshes


def glob_glob(pattern: str) -> List[str]:
    import glob

    return glob.glob(pattern)


def _filter_shard(items: List[Tuple[str, str, int]], num_shards: int, shard_id: int) -> List[Tuple[str, str, int]]:
    ns = int(num_shards)
    sid = int(shard_id)
    if ns <= 1:
        return list(items)
    if sid < 0 or sid >= ns:
        raise ValueError(f"invalid shard_id={sid} for num_shards={ns}")
    return [x for i, x in enumerate(items) if (i % ns) == sid]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapenet_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--synsets", type=str, default="", help="comma-separated synset ids (empty=all)")
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--num_shards", type=int, default=1, help="split target task list into N shards.")
    ap.add_argument("--shard_id", type=int, default=0, help="0-based shard id when num_shards>1.")

    # sizes
    ap.add_argument("--n_surf", type=int, default=8192)
    ap.add_argument("--n_mesh_qry", type=int, default=2048)
    ap.add_argument("--n_udf_qry", type=int, default=8192)
    ap.add_argument("--n_pc", type=int, default=2048)
    ap.add_argument("--n_pc_qry", type=int, default=1024)
    ap.add_argument("--n_rays", type=int, default=4096)

    # pc
    ap.add_argument("--pc_view_crop", type=float, default=0.5)
    ap.add_argument("--pc_noise_std", type=float, default=0.005)
    ap.add_argument("--pc_dropout", type=float, default=0.1)

    # udf
    ap.add_argument("--udf_near_ratio", type=float, default=0.5)
    ap.add_argument("--udf_near_std", type=float, default=0.05)

    # knn
    ap.add_argument("--curvature_knn", type=int, default=20)
    ap.add_argument("--pca_knn", type=int, default=20)

    # rays
    ap.add_argument("--ray_radius", type=float, default=2.5)
    ap.add_argument("--ray_jitter_std", type=float, default=0.05)
    ap.add_argument("--skip_existing", action="store_true", help="skip samples whose output NPZ already exists.")
    ap.add_argument(
        "--missing_only",
        action="store_true",
        help="process only tasks whose output NPZ is currently missing.",
    )

    args = ap.parse_args()

    synsets = [s.strip() for s in args.synsets.split(",") if s.strip()]
    meshes = _glob_meshes(args.shapenet_root, synsets)
    if len(meshes) == 0:
        raise SystemExit(f"No meshes found under {args.shapenet_root}")

    rng = random.Random(int(args.seed))
    rng.shuffle(meshes)
    n_train = int(round(len(meshes) * float(args.train_ratio)))
    train_meshes = meshes[:n_train]
    test_meshes = meshes[n_train:]

    cfg = V2GenConfig(
        n_surf=args.n_surf,
        n_mesh_qry=args.n_mesh_qry,
        n_udf_qry=args.n_udf_qry,
        n_pc=args.n_pc,
        n_pc_qry=args.n_pc_qry,
        n_rays=args.n_rays,
        pc_view_crop=args.pc_view_crop,
        pc_noise_std=args.pc_noise_std,
        pc_dropout=args.pc_dropout,
        udf_near_ratio=args.udf_near_ratio,
        udf_near_std=args.udf_near_std,
        curvature_knn=args.curvature_knn,
        pca_knn=args.pca_knn,
        ray_radius=args.ray_radius,
        ray_jitter_std=args.ray_jitter_std,
        skip_existing=bool(args.skip_existing),
    )

    def _out_path(mesh_path: str, split: str) -> str:
        syn, mid = _infer_synset_model(mesh_path)
        return os.path.join(args.out_root, split, syn, f"{mid}.npz")

    tasks_all: List[Tuple[str, str, int]] = []
    for m in train_meshes:
        tasks_all.append((m, _out_path(m, "train"), int(args.seed)))
    for m in test_meshes:
        tasks_all.append((m, _out_path(m, "test"), int(args.seed) + 12345))

    num_tasks_all = len(tasks_all)
    num_missing_tasks_total = 0
    if bool(args.missing_only):
        filtered: List[Tuple[str, str, int]] = []
        for mesh_path, out_path, base_seed in tasks_all:
            if not os.path.isfile(out_path):
                filtered.append((mesh_path, out_path, base_seed))
        tasks_all = filtered
        num_missing_tasks_total = len(tasks_all)

    tasks = _filter_shard(tasks_all, int(args.num_shards), int(args.shard_id))
    print(
        f"[shard] num_shards={int(args.num_shards)} shard_id={int(args.shard_id)} "
        f"tasks={len(tasks)}"
    )
    if bool(args.missing_only):
        print(f"[missing_only] total_missing_now={num_missing_tasks_total} total_all={num_tasks_all}")

    os.makedirs(args.out_root, exist_ok=True)

    # multiprocessing
    if int(args.num_workers) <= 1:
        errors = []
        for mesh_path, out_path, base_seed in tqdm(tasks):
            syn, mid = _infer_synset_model(mesh_path)
            import zlib
            seed = (zlib.crc32(mesh_path.encode("utf-8")) ^ base_seed) & 0x7FFFFFFF
            err = preprocess_one(mesh_path, out_path, cfg=cfg, seed=seed)
            if err is not None:
                errors.append(err)
    else:
        import multiprocessing as mp

        with mp.Pool(processes=int(args.num_workers)) as pool:
            fn = partial(_worker, cfg=cfg)
            errors = list(tqdm(pool.imap(fn, tasks), total=len(tasks)))
        errors = [e for e in errors if e is not None]

    manifest = {
        "shapenet_root": args.shapenet_root,
        "out_root": args.out_root,
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "num_meshes": len(meshes),
        "num_train": len(train_meshes),
        "num_test": len(test_meshes),
        "missing_only": bool(args.missing_only),
        "num_tasks_all": num_tasks_all,
        "num_missing_tasks_total": num_missing_tasks_total,
        "num_tasks_in_shard": len(tasks),
        "config": asdict(cfg),
        "errors": errors[:100],
    }
    if int(args.num_shards) > 1:
        manifest_name = f"v2_manifest.shard{int(args.shard_id):03d}of{int(args.num_shards):03d}.json"
    else:
        manifest_name = "v2_manifest.json"
    with open(os.path.join(args.out_root, manifest_name), "w") as f:
        json.dump(manifest, f, indent=2)

    if len(errors) > 0:
        print(f"[WARN] {len(errors)} meshes failed. See v2_manifest.json for examples.")


def _worker(task: Tuple[str, str, int], *, cfg: V2GenConfig) -> Optional[str]:
    mesh_path, out_path, base_seed = task
    import zlib
    seed = (zlib.crc32(mesh_path.encode("utf-8")) ^ base_seed) & 0x7FFFFFFF
    return preprocess_one(mesh_path, out_path, cfg=cfg, seed=seed)


if __name__ == "__main__":
    main()
