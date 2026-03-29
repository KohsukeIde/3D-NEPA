from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def _require_open3d():
    try:
        import open3d as o3d
    except Exception as e:  # pragma: no cover - explicit hard fail contract
        raise RuntimeError("Open3D is required for degraded completion baselines.") from e
    return o3d


def _as_point_cloud(xyz: np.ndarray):
    o3d = _require_open3d()
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if int(pts.shape[0]) < 8:
        raise ValueError(f"too few points for mesh reconstruction: n={int(pts.shape[0])}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def _mean_nn_distance(xyz: np.ndarray) -> float:
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if int(pts.shape[0]) < 3:
        raise ValueError(f"too few points for nn distance: n={int(pts.shape[0])}")
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    dist += np.eye(int(dist.shape[0]), dtype=np.float64) * 1.0e9
    nn = np.min(dist, axis=1)
    return float(np.mean(nn))


def _estimate_orient_normals(pcd, *, radius: float, max_nn: int = 32, orient_k: int = 24):
    o3d = _require_open3d()
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(radius),
            max_nn=int(max_nn),
        )
    )
    pcd.orient_normals_consistent_tangent_plane(int(orient_k))


def _clean_mesh(mesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    return mesh


def _mesh_to_numpy(mesh) -> Tuple[np.ndarray, np.ndarray]:
    mesh = _clean_mesh(mesh)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    if verts.size == 0 or faces.size == 0:
        raise ValueError("baseline reconstruction produced an empty mesh")
    return verts, faces


def poisson_from_points(
    xyz: np.ndarray,
    *,
    depth: int = 8,
    density_trim_quantile: float = 0.01,
    normal_radius_scale: float = 2.5,
    max_nn: int = 32,
    orient_k: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    o3d = _require_open3d()
    pts = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    mean_nn = _mean_nn_distance(pts)
    pcd = _as_point_cloud(pts)
    _estimate_orient_normals(
        pcd,
        radius=max(float(mean_nn) * float(normal_radius_scale), 1.0e-4),
        max_nn=max_nn,
        orient_k=orient_k,
    )
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=int(depth),
    )
    dens = np.asarray(densities, dtype=np.float64).reshape(-1)
    if dens.size > 0 and float(density_trim_quantile) > 0.0:
        thresh = float(np.quantile(dens, float(density_trim_quantile)))
        mask = dens < thresh
        if bool(np.any(mask)):
            mesh.remove_vertices_by_mask(mask.tolist())
    return _mesh_to_numpy(mesh)


def bpa_from_points(
    xyz: np.ndarray,
    *,
    radius_multipliers: Iterable[float] = (1.5, 3.0),
    normal_radius_scale: float = 2.5,
    max_nn: int = 32,
    orient_k: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    o3d = _require_open3d()
    pts = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    mean_nn = _mean_nn_distance(pts)
    pcd = _as_point_cloud(pts)
    _estimate_orient_normals(
        pcd,
        radius=max(float(mean_nn) * float(normal_radius_scale), 1.0e-4),
        max_nn=max_nn,
        orient_k=orient_k,
    )
    radii = [max(float(mean_nn) * float(m), 1.0e-4) for m in radius_multipliers]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector([float(r) for r in radii]),
    )
    return _mesh_to_numpy(mesh)
