from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import trimesh
except Exception as e:
    raise RuntimeError("augment_world_v3_mesh_aux requires trimesh") from e

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception as e:
    raise RuntimeError("augment_world_v3_mesh_aux requires scipy") from e


def _fibonacci_hemisphere(n: int) -> np.ndarray:
    n = int(max(1, n))
    i = np.arange(n, dtype=np.float32)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    z = (i + np.float32(0.5)) / np.float32(n)
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    theta = phi * i
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    dirs = np.stack([x, y, z], axis=1).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    return dirs


def _safe_decode_npz_scalar(arr) -> str:
    a = np.asarray(arr)
    if a.dtype.kind in {"S", "a"}:
        return bytes(a.reshape(-1)[0]).decode("utf-8")
    if a.dtype.kind == "U":
        return str(a.reshape(-1)[0])
    return str(a.reshape(-1)[0])


def _load_norm_mesh_from_npz(npz_path: Path) -> trimesh.Trimesh:
    with np.load(npz_path, allow_pickle=False) as npz:
        if "mesh_source_path" in npz:
            mesh_path = _safe_decode_npz_scalar(npz["mesh_source_path"])
        elif "mesh_relpath" in npz:
            mesh_path = _safe_decode_npz_scalar(npz["mesh_relpath"])
        else:
            raise KeyError(f"{npz_path}: missing mesh_source_path/mesh_relpath")
        if "norm_center" in npz:
            center = np.asarray(npz["norm_center"], dtype=np.float32).reshape(1, 3)
        else:
            center = np.zeros((1, 3), dtype=np.float32)
        scale = 1.0
        if "norm_scale" in npz:
            scale = float(np.asarray(npz["norm_scale"], dtype=np.float32).reshape(-1)[0])
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    verts = (np.asarray(mesh.vertices, dtype=np.float32) - center) / max(scale, 1e-8)
    return trimesh.Trimesh(
        vertices=verts,
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )


def _orient_dirs_to_normals(dirs_hemi: np.ndarray, normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    out = np.zeros((normals.shape[0], dirs_hemi.shape[0], 3), dtype=np.float32)
    for i, n in enumerate(normals):
        t = np.cross(z, n)
        if np.linalg.norm(t) < 1e-6:
            ref = x_axis if abs(float(n[0])) < 0.9 else y_axis
            t = np.cross(ref, n)
        t = t / (np.linalg.norm(t) + 1e-8)
        b = np.cross(n, t)
        rot = np.stack([t, b, n], axis=1)
        out[i] = (dirs_hemi @ rot.T).astype(np.float32)
    return out


def compute_mesh_ao_hq(
    mesh: trimesh.Trimesh,
    surf_xyz: np.ndarray,
    surf_n: np.ndarray,
    *,
    n_rays: int = 128,
    eps: float = 1e-4,
    max_t: float = 2.5,
    batch_size: int = 256,
) -> np.ndarray:
    pts = np.asarray(surf_xyz, dtype=np.float32)
    nrms = np.asarray(surf_n, dtype=np.float32)
    nrms /= np.linalg.norm(nrms, axis=1, keepdims=True) + 1e-8
    hemi = _fibonacci_hemisphere(int(n_rays))
    dirs = _orient_dirs_to_normals(hemi, nrms)
    ao = np.zeros((pts.shape[0], 1), dtype=np.float32)
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    for start in range(0, pts.shape[0], int(batch_size)):
        end = min(pts.shape[0], start + int(batch_size))
        p = pts[start:end]
        n = nrms[start:end]
        d = dirs[start:end]
        batch, rays, _ = d.shape
        origins = p[:, None, :] + eps * n[:, None, :]
        origins = origins.reshape(batch, 1, 3) + eps * d
        origins2 = origins.reshape(batch * rays, 3).astype(np.float32)
        dirs2 = d.reshape(batch * rays, 3).astype(np.float32)
        loc, idx_ray, _ = intersector.intersects_location(
            origins2,
            dirs2,
            multiple_hits=False,
        )
        vis = np.ones((batch * rays,), dtype=np.float32)
        if loc.shape[0] > 0:
            dist = np.linalg.norm(loc - origins2[idx_ray], axis=1)
            occ = dist < float(max_t)
            if np.any(occ):
                vis[idx_ray[occ]] = 0.0
        ao[start:end, 0] = 1.0 - vis.reshape(batch, rays).mean(axis=1)
    return ao.astype(np.float32)


def _cotangent_laplacian(vertices: np.ndarray, faces: np.ndarray):
    verts = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    i1, i2, i3 = tri[:, 0], tri[:, 1], tri[:, 2]
    v1, v2, v3 = verts[i1], verts[i2], verts[i3]

    def cot(a, b):
        cross = np.linalg.norm(np.cross(a, b), axis=1)
        dot = np.sum(a * b, axis=1)
        return dot / np.clip(cross, 1e-12, None)

    cot1 = cot(v2 - v1, v3 - v1)
    cot2 = cot(v3 - v2, v1 - v2)
    cot3 = cot(v1 - v3, v2 - v3)
    ii = np.concatenate([i2, i3, i3, i1, i1, i2])
    jj = np.concatenate([i3, i2, i1, i3, i2, i1])
    ww = 0.5 * np.concatenate([cot1, cot1, cot2, cot2, cot3, cot3])
    n_vert = verts.shape[0]
    lap = sp.coo_matrix((ww, (ii, jj)), shape=(n_vert, n_vert)).tocsr()
    lap = sp.diags(np.asarray(lap.sum(axis=1)).reshape(-1)) - lap

    tri_area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)
    mass_diag = np.zeros((n_vert,), dtype=np.float64)
    np.add.at(mass_diag, i1, tri_area / 3.0)
    np.add.at(mass_diag, i2, tri_area / 3.0)
    np.add.at(mass_diag, i3, tri_area / 3.0)
    mass = sp.diags(np.clip(mass_diag, 1e-12, None))
    return lap, mass


def compute_mesh_hks(
    mesh: trimesh.Trimesh,
    surf_face_idx: np.ndarray | None,
    surf_bary: np.ndarray | None,
    surf_xyz: np.ndarray,
    *,
    n_eigs: int = 64,
    times: Sequence[float] = (0.05, 0.2, 1.0),
) -> dict[str, np.ndarray]:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    lap, mass = _cotangent_laplacian(verts, faces)
    max_k = max(verts.shape[0] - 2, 2)
    k = int(min(max(8, n_eigs), max_k))
    mass_diag = np.asarray(mass.diagonal()).reshape(-1)
    inv_sqrt_mass = 1.0 / np.sqrt(np.clip(mass_diag, 1e-12, None))
    scale = sp.diags(inv_sqrt_mass)
    lap_n = scale @ lap @ scale
    try:
        vals, u = spla.eigsh(lap_n, k=k, sigma=1e-6, which="LM")
    except Exception:
        vals, u = spla.eigsh(lap_n, k=k, which="SM", tol=1e-4, maxiter=20000)
    vecs = inv_sqrt_mass[:, None] * np.real(u)
    vals = np.clip(np.real(vals), 0.0, None)
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    out: dict[str, np.ndarray] = {}
    if surf_face_idx is not None and surf_bary is not None:
        face_idx = np.asarray(surf_face_idx, dtype=np.int64).reshape(-1)
        bary = np.asarray(surf_bary, dtype=np.float32)
        tri = faces[face_idx]
        phi_pts = vecs[tri]
        for ti, t in enumerate(times):
            expw = np.exp(-vals * float(t)).reshape(1, 1, -1)
            h = np.sum(expw * (phi_pts**2), axis=2)
            h = np.sum(h * bary, axis=1, keepdims=True)
            out[f"mesh_surf_hks_t{ti}"] = np.asarray(h, dtype=np.float32).reshape(-1, 1)
        return out

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(verts)
        _dist, idx = tree.query(np.asarray(surf_xyz, dtype=np.float64), k=1)
        phi_pts = vecs[np.asarray(idx, dtype=np.int64)][:, None, :]
    except Exception:
        idx = np.zeros((len(surf_xyz),), dtype=np.int64)
        phi_pts = vecs[idx][:, None, :]
    for ti, t in enumerate(times):
        expw = np.exp(-vals * float(t)).reshape(1, 1, -1)
        h = np.sum(expw[:, :1, :] * (phi_pts**2), axis=2)
        out[f"mesh_surf_hks_t{ti}"] = np.asarray(h, dtype=np.float32).reshape(-1, 1)
    return out


def _iter_npz(root: Path, split: str) -> Iterable[Path]:
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(split_dir)
    for syn_dir in sorted(split_dir.iterdir()):
        if not syn_dir.is_dir():
            continue
        for npz_path in sorted(syn_dir.glob("*.npz")):
            yield npz_path


def _select_subset(src_root: Path, split: str, limit: int, seed: int) -> list[Path]:
    paths = list(_iter_npz(src_root, split))
    if limit <= 0 or limit >= len(paths):
        return paths
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(len(paths), size=int(limit), replace=False))
    return [paths[int(i)] for i in idx]


def _shard_subset(paths: Sequence[Path], num_shards: int, shard_index: int) -> list[Path]:
    n = int(max(1, num_shards))
    i = int(shard_index)
    if i < 0 or i >= n:
        raise ValueError(f"invalid shard_index={i} for num_shards={n}")
    if n == 1:
        return list(paths)
    return [p for j, p in enumerate(paths) if (j % n) == i]


def _copy_subset(
    src_root: Path,
    dst_root: Path,
    selected: Sequence[Path],
    *,
    refresh: bool,
    copy_meta: bool,
) -> list[Path]:
    if copy_meta:
        src_meta = src_root / "_meta"
        dst_meta = dst_root / "_meta"
        if src_meta.is_dir():
            shutil.copytree(src_meta, dst_meta, dirs_exist_ok=True)
    copied: list[Path] = []
    for src_path in selected:
        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if refresh or not dst_path.exists():
            shutil.copy2(src_path, dst_path)
        copied.append(dst_path)
    return copied


def _augment_npz(
    npz_path: Path,
    *,
    compute_ao_hq: bool,
    ao_rays: int,
    ao_eps: float,
    ao_max_t: float,
    compute_hks: bool,
    hks_eigs: int,
    hks_times: Sequence[float],
    suffix: str,
    refresh: bool,
) -> dict[str, object]:
    result: dict[str, object] = {"path": str(npz_path), "updated": []}
    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            data = {k: npz[k] for k in npz.files}
        updates: dict[str, np.ndarray] = {}
        errors: list[str] = []
        mesh = _load_norm_mesh_from_npz(npz_path)
        if compute_ao_hq:
            key = f"mesh_surf_ao_hq{suffix}"
            if refresh or key not in data:
                try:
                    updates[key] = compute_mesh_ao_hq(
                        mesh,
                        np.asarray(data["surf_xyz"], dtype=np.float32),
                        np.asarray(data["mesh_surf_n"], dtype=np.float32),
                        n_rays=int(ao_rays),
                        eps=float(ao_eps),
                        max_t=float(ao_max_t),
                    )
                except Exception as e:
                    errors.append(f"ao_hq:{type(e).__name__}: {e}")
        if compute_hks:
            face_idx = data["surf_face_idx"] if "surf_face_idx" in data else None
            bary = data["surf_bary"] if "surf_bary" in data else None
            try:
                hks = compute_mesh_hks(
                    mesh,
                    face_idx,
                    bary,
                    np.asarray(data["surf_xyz"], dtype=np.float32),
                    n_eigs=int(hks_eigs),
                    times=hks_times,
                )
                for key, value in hks.items():
                    key2 = f"{key}{suffix}"
                    if refresh or key2 not in data:
                        updates[key2] = value
            except Exception as e:
                errors.append(f"hks:{type(e).__name__}: {e}")
        if updates:
            data.update(updates)
            tmp_path = npz_path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp_path, **data)
            tmp_path.replace(npz_path)
            result["updated"] = sorted(updates.keys())
        if errors:
            result["errors"] = errors
            if not updates:
                result["error"] = "; ".join(errors)
        return result
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result


def main() -> None:
    ap = argparse.ArgumentParser("augment_world_v3_mesh_aux")
    ap.add_argument("--src_cache_root", type=str, required=True)
    ap.add_argument("--dst_cache_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="eval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--refresh", type=int, default=0)
    ap.add_argument("--copy_meta", type=int, default=1)
    ap.add_argument("--compute_ao_hq", type=int, default=1)
    ap.add_argument("--ao_rays", type=int, default=128)
    ap.add_argument("--ao_eps", type=float, default=1e-4)
    ap.add_argument("--ao_max_t", type=float, default=2.5)
    ap.add_argument("--compute_hks", type=int, default=0)
    ap.add_argument("--hks_eigs", type=int, default=64)
    ap.add_argument("--hks_times", type=str, default="0.05,0.2,1.0")
    ap.add_argument("--suffix", type=str, default="")
    ap.add_argument("--output_json", type=str, default="")
    args = ap.parse_args()

    src_root = Path(args.src_cache_root)
    dst_root = Path(args.dst_cache_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    hks_times = [float(x) for x in str(args.hks_times).split(",") if x.strip()]

    selected_all = _select_subset(src_root, args.split, int(args.limit), int(args.sample_seed))
    selected = _shard_subset(selected_all, int(args.num_shards), int(args.shard_index))
    copied = _copy_subset(
        src_root,
        dst_root,
        selected,
        refresh=bool(args.refresh),
        copy_meta=bool(args.copy_meta),
    )
    rows = []
    for npz_path in copied:
        row = _augment_npz(
            npz_path,
            compute_ao_hq=bool(args.compute_ao_hq),
            ao_rays=int(args.ao_rays),
            ao_eps=float(args.ao_eps),
            ao_max_t=float(args.ao_max_t),
            compute_hks=bool(args.compute_hks),
            hks_eigs=int(args.hks_eigs),
            hks_times=hks_times,
            suffix=str(args.suffix),
            refresh=bool(args.refresh),
        )
        rows.append(row)
        print(json.dumps(row))

    summary = {
        "src_cache_root": str(src_root),
        "dst_cache_root": str(dst_root),
        "split": str(args.split),
        "limit": int(args.limit),
        "sample_seed": int(args.sample_seed),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "num_selected_total": len(selected_all),
        "refresh": bool(args.refresh),
        "compute_ao_hq": bool(args.compute_ao_hq),
        "ao_rays": int(args.ao_rays),
        "compute_hks": bool(args.compute_hks),
        "hks_eigs": int(args.hks_eigs),
        "hks_times": hks_times,
        "num_selected": len(selected),
        "num_errors": sum(1 for row in rows if "error" in row),
        "rows": rows,
    }
    if str(args.output_json).strip():
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
