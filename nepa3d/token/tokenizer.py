from __future__ import annotations

import numpy as np
import warnings

from .ordering import (
    morton3d,
    sort_by_ray_direction,
    sort_rays_by_direction_fps,
    sort_rays_by_view_raster,
    sort_rays_by_x_anchor,
)

# -----------------------------------------------------------------------------
# Token type ids
# -----------------------------------------------------------------------------
# Legacy types (v0/v1): one token per primitive (point or ray).
TYPE_BOS = 0
TYPE_POINT = 1
TYPE_RAY = 2
TYPE_MISSING_RAY = 3
TYPE_EOS = 4

# Q/A separated types (v2): two tokens per primitive (query then answer).
#
# IMPORTANT: We keep legacy ids unchanged for backward compatibility with old
# checkpoints / logs / scripts. New ids are appended.
TYPE_Q_POINT = 5
TYPE_A_POINT = 6
TYPE_Q_RAY = 7
TYPE_A_RAY = 8

# Separator token to explicitly mark the boundary between query and answer blocks
# in split layouts.
TYPE_SEP = 9

# Size of the type-id vocabulary.
TYPE_VOCAB_SIZE = TYPE_SEP + 1

_WARNED_FPS_FALLBACK = False


def _choice(n, k, rng=None):
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    replace = n < k
    if rng is None:
        rng = np.random
    return rng.choice(n, size=k, replace=replace)


def _rand01(rng):
    if hasattr(rng, "rand"):
        return float(rng.rand())
    return float(rng.random())


def _clip_vec_norm(v: np.ndarray, max_norm: float, eps: float = 1e-12) -> np.ndarray:
    """Clip vector norms (row-wise) for numerical stability."""
    if max_norm <= 0:
        return v
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    scale = np.minimum(1.0, float(max_norm) / (n + float(eps)))
    return v * scale


def _build_point_grad_from_rays(
    pt_xyz_s: np.ndarray,
    pt_dist_s: np.ndarray,
    ray_o_s: np.ndarray,
    ray_d_s: np.ndarray,
    ray_hit_s: np.ndarray,
    ray_t_s: np.ndarray,
    ray_n_s: np.ndarray,
    mode: str = "raw",
    eps: float = 1e-3,
    clip: float = 10.0,
    orient: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate per-point gradient from nearest valid ray-hit normals.

    Returns:
      grad: (N,3), mag: (N,)
    """
    n_pt = int(pt_xyz_s.shape[0])
    if n_pt <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    grad = np.zeros((n_pt, 3), dtype=np.float32)
    mag = np.zeros((n_pt,), dtype=np.float32)
    valid = ray_hit_s > 0.5
    if not bool(np.any(valid)):
        return grad, mag

    x_hit = ray_o_s[valid] + ray_t_s[valid, None] * ray_d_s[valid]
    n_hit = ray_n_s[valid].astype(np.float32, copy=False)
    d_hit = ray_d_s[valid].astype(np.float32, copy=False)
    if x_hit.shape[0] <= 0:
        return grad, mag

    diff = pt_xyz_s[:, None, :] - x_hit[None, :, :]
    nn_idx = np.argmin(np.sum(diff * diff, axis=-1), axis=1)
    g = n_hit[nn_idx].copy()

    if str(orient).lower() == "ray":
        dref = d_hit[nn_idx]
        dot = np.sum(g * dref, axis=-1, keepdims=True)
        g = np.where(dot > 0, -g, g)

    mode_s = str(mode).lower()
    if mode_s == "log":
        s = 1.0 / np.maximum(pt_dist_s.astype(np.float32), float(eps))
        g = g * s[:, None]
    elif mode_s != "raw":
        raise ValueError(f"unknown pt_grad_mode: {mode}")

    g = _clip_vec_norm(g.astype(np.float32, copy=False), float(clip))
    grad = g
    mag = np.linalg.norm(grad, axis=-1).astype(np.float32)
    return grad, mag


def _build_ray_uncertainty_from_normals(
    ray_o_s: np.ndarray,
    ray_d_s: np.ndarray,
    ray_hit_s: np.ndarray,
    ray_t_s: np.ndarray,
    ray_n_s: np.ndarray,
    k: int = 8,
) -> np.ndarray:
    """Normal-variation uncertainty proxy for ray answers.

    unc_i = 1 - mean_j |dot(n_i, n_j)| over k nearest valid hit points.
    """
    n_ray = int(ray_hit_s.shape[0])
    out = np.zeros((n_ray,), dtype=np.float32)
    valid = ray_hit_s > 0.5
    if not bool(np.any(valid)):
        return out

    x_hit = ray_o_s[valid] + ray_t_s[valid, None] * ray_d_s[valid]
    n_hit = ray_n_s[valid].astype(np.float32, copy=False)
    nv = x_hit.shape[0]
    if nv <= 1:
        return out

    diff = x_hit[:, None, :] - x_hit[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    k_eff = max(1, min(int(k), nv))
    idx = np.argsort(d2, axis=1)[:, :k_eff]
    nn = n_hit[idx]  # (Nv, k, 3)
    dots = np.abs(np.sum(nn * n_hit[:, None, :], axis=-1))
    unc_valid = (1.0 - np.mean(dots, axis=-1)).astype(np.float32)
    out[valid] = unc_valid
    return out


def _sample_point_indices(
    pt_xyz_pool: np.ndarray,
    n_point: int,
    rng: np.random.Generator,
    pt_sample_mode: str = "random",
    pt_fps_order: np.ndarray | None = None,
    pt_rfps_order: np.ndarray | None = None,
    pt_rfps_m: int = 4096,
) -> np.ndarray:
    """Select point indices according to a sampling policy."""
    global _WARNED_FPS_FALLBACK
    n_pool = int(pt_xyz_pool.shape[0])
    n_point = int(n_point)
    if n_point <= 0:
        return np.empty((0,), dtype=np.int64)
    if n_pool <= 0:
        raise ValueError("pt_xyz_pool is empty but n_point > 0")
    k = min(n_point, n_pool)
    mode = str(pt_sample_mode).lower()

    if mode in ("grid", "fixed_grid"):
        # Deterministic fixed-grid queries in normalized space [-1,1]^3.
        # We map each grid query to its nearest point in the pool.
        r = int(np.ceil(float(k) ** (1.0 / 3.0)))
        r = max(r, 1)
        lin = np.linspace(-1.0, 1.0, num=r, dtype=np.float32)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        grid = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)[:k]
        try:
            from scipy.spatial import cKDTree

            _, nn = cKDTree(pt_xyz_pool.astype(np.float32, copy=False)).query(grid, k=1)
            return np.asarray(nn, dtype=np.int64).reshape(-1)
        except Exception:
            # Fallback: brute-force nearest neighbor (chunked to keep memory bounded).
            idx = np.empty((grid.shape[0],), dtype=np.int64)
            pts = pt_xyz_pool.astype(np.float32, copy=False)
            bsz = 64
            for i in range(0, grid.shape[0], bsz):
                q = grid[i : i + bsz]
                d2 = ((q[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1)
                idx[i : i + bsz] = np.argmin(d2, axis=1).astype(np.int64, copy=False)
            return idx

    if mode == "fps":
        order = None
        if pt_fps_order is not None:
            cand = np.asarray(pt_fps_order).reshape(-1)
            if cand.size > 0 and int(cand.min()) >= 0 and int(cand.max()) < n_pool:
                order = cand.astype(np.int64, copy=False)

        if order is None:
            if not _WARNED_FPS_FALLBACK:
                warnings.warn(
                    "pt_sample_mode='fps' but no valid FPS order was provided; "
                    "computing FPS on-the-fly. For strict reproducibility and speed, "
                    "precompute FPS order in cache and set --pt_fps_key accordingly."
                )
                _WARNED_FPS_FALLBACK = True
            from nepa3d.utils.fps import fps_order

            order = fps_order(pt_xyz_pool.astype(np.float32, copy=False), k=k).astype(
                np.int64, copy=False
            )

        k0 = min(k, int(order.shape[0]))
        chosen = order[:k0].astype(np.int64, copy=False)
        if k0 == k:
            return chosen
        used = np.zeros((n_pool,), dtype=bool)
        used[chosen] = True
        rest = np.flatnonzero(~used)
        extra = rng.choice(rest, size=(k - k0), replace=False)
        return np.concatenate([chosen, extra.astype(np.int64)], axis=0)

    if mode == "rfps":
        from nepa3d.utils.fps import rfps_order

        m = min(int(pt_rfps_m), n_pool)
        return rfps_order(pt_xyz_pool, k=k, m=m, rng=rng).astype(np.int64, copy=False)

    if mode == "rfps_cached":
        order = None
        if pt_rfps_order is not None:
            bank = np.asarray(pt_rfps_order)
            if bank.ndim == 1:
                cand = bank.reshape(-1)
                if cand.size > 0 and int(cand.min()) >= 0 and int(cand.max()) < n_pool:
                    order = cand.astype(np.int64, copy=False)
            elif bank.ndim == 2 and bank.size > 0:
                # Preferred format: (n_bank, n_pool).
                if bank.shape[0] > bank.shape[1]:
                    bank = bank.T
                n_bank = int(bank.shape[0])
                if n_bank > 0:
                    if hasattr(rng, "integers"):
                        bid = int(rng.integers(0, n_bank))
                    else:
                        bid = int(rng.randint(0, n_bank))
                    cand = np.asarray(bank[bid]).reshape(-1)
                    if cand.size > 0 and int(cand.min()) >= 0 and int(cand.max()) < n_pool:
                        order = cand.astype(np.int64, copy=False)

        if order is None:
            raise ValueError(
                "pt_sample_mode='rfps_cached' requires a valid RFPS order bank, but none was "
                "provided. Backfill and pass a bank key (e.g., pt_rfps_order_bank)."
            )

        k0 = min(k, int(order.shape[0]))
        chosen = order[:k0].astype(np.int64, copy=False)
        if k0 == k:
            return chosen
        used = np.zeros((n_pool,), dtype=bool)
        used[chosen] = True
        rest = np.flatnonzero(~used)
        extra = rng.choice(rest, size=(k - k0), replace=False)
        return np.concatenate([chosen, extra.astype(np.int64)], axis=0)

    return rng.choice(n_pool, size=k, replace=False).astype(np.int64, copy=False)


def _order_point_tokens(
    pt_xyz_s: np.ndarray,
    pt_dist_s: np.ndarray,
    *,
    rng=None,
    point_order_mode: str = "morton",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an explicit order for point tokens after sampling.

    Supported modes:
      - morton: spatially local Z-order sort (legacy behavior)
      - fps: keep the sampled order as-is (for coarse-to-fine AR when sampling=fps)
      - random: random permutation
    """
    if int(pt_xyz_s.shape[0]) <= 0:
        return pt_xyz_s, pt_dist_s

    mode = str(point_order_mode).lower()
    if mode == "morton":
        order = np.argsort(morton3d(pt_xyz_s))
    elif mode == "fps":
        order = None
    elif mode == "random":
        if rng is None:
            rng = np.random
        if hasattr(rng, "permutation"):
            order = np.asarray(rng.permutation(pt_xyz_s.shape[0]), dtype=np.int64)
        else:
            order = np.random.permutation(pt_xyz_s.shape[0]).astype(np.int64, copy=False)
    else:
        raise ValueError(f"unknown point_order_mode: {point_order_mode}")

    if order is None:
        return pt_xyz_s, pt_dist_s, None
    return pt_xyz_s[order], pt_dist_s[order], order


def _build_sequence_legacy(
    pt_xyz,
    pt_dist,
    ray_o,
    ray_d,
    ray_hit,
    ray_t,
    ray_n,
    n_point=512,
    n_ray=512,
    drop_ray_prob=0.0,
    ray_available=True,
    add_eos=True,
    rng=None,
    pt_sample_mode="random",
    pt_fps_order=None,
    pt_rfps_order=None,
    pt_rfps_m=4096,
    point_order_mode="morton",
    include_pt_grad=False,
    pt_grad_mode="raw",
    pt_grad_eps=1e-3,
    pt_grad_clip=10.0,
    pt_grad_orient="none",
    include_ray_unc=False,
    ray_unc_k=8,
    ray_unc_mode="normal_var",
):
    """Legacy tokenization: [BOS] + POINT* + RAY* (+ [EOS]).

    This is kept to allow reproducing older experiments.
    """
    if rng is None:
        rng = np.random

    pt_xyz = np.asarray(pt_xyz)
    pt_dist = np.asarray(pt_dist)
    pt_dist_1d = pt_dist[:, 0] if pt_dist.ndim == 2 else pt_dist

    p_idx = _sample_point_indices(
        pt_xyz_pool=pt_xyz,
        n_point=n_point,
        rng=rng,
        pt_sample_mode=pt_sample_mode,
        pt_fps_order=pt_fps_order,
        pt_rfps_order=pt_rfps_order,
        pt_rfps_m=pt_rfps_m,
    )

    pt_xyz_s = pt_xyz[p_idx]
    pt_dist_s = pt_dist_1d[p_idx]

    if int(n_ray) > 0:
        if ray_o is None or ray_d is None or ray_hit is None or ray_t is None or ray_n is None:
            raise ValueError("ray pools are required when n_ray > 0")
        ray_o = np.asarray(ray_o)
        ray_d = np.asarray(ray_d)
        ray_hit = np.asarray(ray_hit)
        ray_t = np.asarray(ray_t)
        ray_n = np.asarray(ray_n)
        r_idx = _choice(ray_o.shape[0], n_ray, rng=rng)
        ray_o_s = ray_o[r_idx]
        ray_d_s = ray_d[r_idx]
        ray_hit_s = ray_hit[r_idx].reshape(-1)
        ray_t_s = ray_t[r_idx].reshape(-1)
        ray_n_s = ray_n[r_idx]
    else:
        ray_o_s = np.zeros((0, 3), dtype=np.float32)
        ray_d_s = np.zeros((0, 3), dtype=np.float32)
        ray_hit_s = np.zeros((0,), dtype=np.float32)
        ray_t_s = np.zeros((0,), dtype=np.float32)
        ray_n_s = np.zeros((0, 3), dtype=np.float32)

    pt_xyz_s, pt_dist_s, _p_order = _order_point_tokens(
        pt_xyz_s,
        pt_dist_s,
        rng=rng,
        point_order_mode=point_order_mode,
    )

    if ray_d_s.shape[0] > 0:
        r_order = sort_by_ray_direction(ray_d_s)
        ray_o_s = ray_o_s[r_order]
        ray_d_s = ray_d_s[r_order]
        ray_hit_s = ray_hit_s[r_order]
        ray_t_s = ray_t_s[r_order]
        ray_n_s = ray_n_s[r_order]

    feat_dim = 15
    pt_feat = np.zeros((n_point, feat_dim), dtype=np.float32)
    if n_point > 0:
        pt_feat[:, 0:3] = pt_xyz_s
        pt_feat[:, 10] = pt_dist_s
        if bool(include_pt_grad):
            g, gmag = _build_point_grad_from_rays(
                pt_xyz_s,
                pt_dist_s,
                ray_o_s,
                ray_d_s,
                ray_hit_s,
                ray_t_s,
                ray_n_s,
                mode=str(pt_grad_mode),
                eps=float(pt_grad_eps),
                clip=float(pt_grad_clip),
                orient=str(pt_grad_orient),
            )
            pt_feat[:, 3:6] = g
            pt_feat[:, 8] = gmag

    ray_feat = np.zeros((n_ray, feat_dim), dtype=np.float32)
    if n_ray > 0:
        hit_mask = (ray_hit_s > 0.5).astype(np.float32)
        x_hit = ray_o_s + ray_t_s[:, None] * ray_d_s
        ray_feat[:, 0:3] = x_hit * hit_mask[:, None]
        ray_feat[:, 3:6] = ray_o_s
        ray_feat[:, 6:9] = ray_d_s
        ray_feat[:, 9] = ray_t_s
        if bool(include_ray_unc):
            mode = str(ray_unc_mode).lower()
            if mode != "normal_var":
                raise ValueError(f"unknown ray_unc_mode: {ray_unc_mode}")
            ray_feat[:, 10] = _build_ray_uncertainty_from_normals(
                ray_o_s, ray_d_s, ray_hit_s, ray_t_s, ray_n_s, k=int(ray_unc_k)
            )
        ray_feat[:, 11] = ray_hit_s
        ray_feat[:, 12:15] = ray_n_s * hit_mask[:, None]

    ray_missing = (not bool(ray_available)) or (
        int(n_ray) > 0 and float(drop_ray_prob) > 0.0 and _rand01(rng) < float(drop_ray_prob)
    )
    ray_type = TYPE_MISSING_RAY if ray_missing else TYPE_RAY
    if ray_missing and n_ray > 0:
        ray_feat[:] = 0.0

    bos_feat = np.zeros((1, feat_dim), dtype=np.float32)
    eos_feat = np.zeros((1, feat_dim), dtype=np.float32)

    feat_list = [bos_feat, pt_feat, ray_feat]
    type_list = [
        np.array([TYPE_BOS], dtype=np.int64),
        np.full((n_point,), TYPE_POINT, dtype=np.int64),
        np.full((n_ray,), ray_type, dtype=np.int64),
    ]
    if add_eos:
        feat_list.append(eos_feat)
        type_list.append(np.array([TYPE_EOS], dtype=np.int64))

    feat = np.concatenate(feat_list, axis=0)
    type_id = np.concatenate(type_list, axis=0)
    return feat, type_id


def _build_sequence_qa(
    pt_xyz,
    pt_dist,
    ray_o,
    ray_d,
    ray_hit,
    ray_t,
    ray_n,
    n_point=512,
    n_ray=512,
    drop_ray_prob=0.0,
    ray_available=True,
    add_eos=True,
    rng=None,
    qa_layout="interleave",
    sequence_mode="block",
    event_order_mode="morton",
    ray_order_mode="theta_phi",
    ray_anchor_miss_t=4.0,
    ray_view_tol=1e-6,
    pt_sample_mode="random",
    pt_fps_order=None,
    pt_rfps_order=None,
    pt_rfps_m=4096,
    point_order_mode="morton",
    include_pt_grad=False,
    pt_grad_mode="raw",
    pt_grad_eps=1e-3,
    pt_grad_clip=10.0,
    pt_grad_orient="none",
    include_ray_unc=False,
    ray_unc_k=8,
    ray_unc_mode="normal_var",
):
    """Q/A separated tokenization.

    Default (sequence_mode=block):
      [BOS] + (Qp,Ap)* + (Qr,Ar)* (+ [EOS]).

    Event mode (sequence_mode=event):
      Build a single list of "events" (point events + ray events), order them by a
      shared 3D anchor (x_anchor), and emit tokens in that unified event order.

    - Point query token: xyz only.
    - Point answer token: dist only (xyz is zeroed to reduce coordinate leakage).
    - Ray query token: ray origin+direction only.
    - Ray answer token: hit/t/normal (+ x_hit, optional via existing schema slots).

    Supported layouts:
    - interleave: [BOS] + (Q, A)* + [EOS]
    - split:      [BOS] + Q... + A... + [EOS]
    - split_sep:  [BOS] + Q... + [SEP] + A... + [EOS]
    """
    if rng is None:
        rng = np.random

    pt_xyz = np.asarray(pt_xyz)
    pt_dist = np.asarray(pt_dist)
    pt_dist_1d = pt_dist[:, 0] if pt_dist.ndim == 2 else pt_dist

    p_idx = _sample_point_indices(
        pt_xyz_pool=pt_xyz,
        n_point=n_point,
        rng=rng,
        pt_sample_mode=pt_sample_mode,
        pt_fps_order=pt_fps_order,
        pt_rfps_order=pt_rfps_order,
        pt_rfps_m=pt_rfps_m,
    )

    pt_xyz_s = pt_xyz[p_idx]
    pt_dist_s = pt_dist_1d[p_idx]

    if int(n_ray) > 0:
        if ray_o is None or ray_d is None or ray_hit is None or ray_t is None or ray_n is None:
            raise ValueError("ray pools are required when n_ray > 0")
        ray_o = np.asarray(ray_o)
        ray_d = np.asarray(ray_d)
        ray_hit = np.asarray(ray_hit)
        ray_t = np.asarray(ray_t)
        ray_n = np.asarray(ray_n)
        r_idx = _choice(ray_o.shape[0], n_ray, rng=rng)
        ray_o_s = ray_o[r_idx]
        ray_d_s = ray_d[r_idx]
        ray_hit_s = ray_hit[r_idx].reshape(-1)
        ray_t_s = ray_t[r_idx].reshape(-1)
        ray_n_s = ray_n[r_idx]
    else:
        ray_o_s = np.zeros((0, 3), dtype=np.float32)
        ray_d_s = np.zeros((0, 3), dtype=np.float32)
        ray_hit_s = np.zeros((0,), dtype=np.float32)
        ray_t_s = np.zeros((0,), dtype=np.float32)
        ray_n_s = np.zeros((0, 3), dtype=np.float32)

    pt_xyz_s, pt_dist_s, _p_order = _order_point_tokens(
        pt_xyz_s,
        pt_dist_s,
        rng=rng,
        point_order_mode=point_order_mode,
    )

    if ray_d_s.shape[0] > 0:
        rmode = str(ray_order_mode).lower()
        if rmode in ("theta_phi", "theta-phi", "angle"):
            r_order = sort_by_ray_direction(ray_d_s)
        elif rmode in ("dir_fps", "direction_fps", "ray_fps", "s2_fps"):
            r_order = sort_rays_by_direction_fps(ray_d_s)
        elif rmode in ("view_raster", "view-raster", "raster"):
            r_order = sort_rays_by_view_raster(ray_o_s, ray_d_s, view_tol=float(ray_view_tol))
        elif rmode in ("x_anchor_morton", "x-anchor-morton", "x_anchor"):
            r_order = sort_rays_by_x_anchor(
                ray_o_s,
                ray_d_s,
                ray_hit_s,
                ray_t_s,
                miss_t=float(ray_anchor_miss_t),
                mode="morton",
            )
        elif rmode in ("x_anchor_fps", "x-anchor-fps"):
            r_order = sort_rays_by_x_anchor(
                ray_o_s,
                ray_d_s,
                ray_hit_s,
                ray_t_s,
                miss_t=float(ray_anchor_miss_t),
                mode="fps",
            )
        elif rmode in ("random", "shuffle"):
            r_order = rng.permutation(ray_d_s.shape[0]).astype(np.int32)
        elif rmode in ("none", "keep"):
            r_order = None
        else:
            raise ValueError(f"unknown ray_order_mode: {ray_order_mode}")

        if r_order is not None:
            ray_o_s = ray_o_s[r_order]
            ray_d_s = ray_d_s[r_order]
            ray_hit_s = ray_hit_s[r_order]
            ray_t_s = ray_t_s[r_order]
            ray_n_s = ray_n_s[r_order]

    # Use effective sampled sizes (point pools may be smaller than requested).
    n_point = int(pt_xyz_s.shape[0])
    n_ray = int(ray_o_s.shape[0])

    feat_dim = 15

    layout = str(qa_layout).lower().replace("+", "_")
    if layout == "split_sep":
        pass
    elif layout not in ("interleave", "split"):
        raise ValueError(f"unknown qa_layout: {qa_layout}")

    # POINT: query (xyz), answer (dist)
    pt_q = np.zeros((n_point, feat_dim), dtype=np.float32)
    pt_a = np.zeros((n_point, feat_dim), dtype=np.float32)
    if n_point > 0:
        pt_q[:, 0:3] = pt_xyz_s
        if bool(include_pt_grad):
            g, gmag = _build_point_grad_from_rays(
                pt_xyz_s,
                pt_dist_s,
                ray_o_s,
                ray_d_s,
                ray_hit_s,
                ray_t_s,
                ray_n_s,
                mode=str(pt_grad_mode),
                eps=float(pt_grad_eps),
                clip=float(pt_grad_clip),
                orient=str(pt_grad_orient),
            )
            pt_q[:, 3:6] = g
            pt_q[:, 8] = gmag
        pt_a[:, 10] = pt_dist_s

    # RAY: query (o,d), answer (hit,t,n,+x_hit)
    ray_q = np.zeros((n_ray, feat_dim), dtype=np.float32)
    ray_a = np.zeros((n_ray, feat_dim), dtype=np.float32)
    if n_ray > 0:
        hit_mask = (ray_hit_s > 0.5).astype(np.float32)
        x_hit = ray_o_s + ray_t_s[:, None] * ray_d_s

        # query: o,d
        ray_q[:, 3:6] = ray_o_s
        ray_q[:, 6:9] = ray_d_s

        # answer: (optional) x_hit + t + hit + n
        ray_a[:, 0:3] = x_hit * hit_mask[:, None]
        ray_a[:, 9] = ray_t_s
        if bool(include_ray_unc):
            mode = str(ray_unc_mode).lower()
            if mode != "normal_var":
                raise ValueError(f"unknown ray_unc_mode: {ray_unc_mode}")
            ray_a[:, 10] = _build_ray_uncertainty_from_normals(
                ray_o_s, ray_d_s, ray_hit_s, ray_t_s, ray_n_s, k=int(ray_unc_k)
            )
        ray_a[:, 11] = ray_hit_s
        ray_a[:, 12:15] = ray_n_s * hit_mask[:, None]

    ray_missing = (not bool(ray_available)) or (
        int(n_ray) > 0 and float(drop_ray_prob) > 0.0 and _rand01(rng) < float(drop_ray_prob)
    )

    if ray_missing and n_ray > 0:
        ray_q[:] = 0.0
        ray_a[:] = 0.0

    bos_feat = np.zeros((1, feat_dim), dtype=np.float32)
    eos_feat = np.zeros((1, feat_dim), dtype=np.float32)

    seq_mode = str(sequence_mode).lower()
    if seq_mode not in ("block", "event", "bundle"):
        raise ValueError(f"unknown sequence_mode: {sequence_mode}")
    if seq_mode == "bundle":
        seq_mode = "event"

    if seq_mode == "event":
        # Build a unified list of (point-events + ray-events) ordered by a shared anchor.
        # This helps remove the discontinuity at the point->ray boundary in block layouts.

        # Point events.
        pt_anchor = pt_xyz_s.astype(np.float32, copy=False)
        pt_kind = np.full((n_point,), 0, dtype=np.int64)  # 0=point

        # Ray events.
        if n_ray > 0:
            x_hit = ray_o_s + ray_t_s[:, None] * ray_d_s
            miss_mask = (ray_hit_s <= 0.5)
            if ray_missing:
                miss_mask = np.ones_like(miss_mask, dtype=bool)
            ray_anchor = x_hit.astype(np.float32, copy=False)
            if np.any(miss_mask):
                ray_anchor = ray_anchor.copy()
                ray_anchor[miss_mask] = (ray_o_s + float(ray_anchor_miss_t) * ray_d_s)[miss_mask]
            ray_kind = np.full((n_ray,), 1, dtype=np.int64)  # 1=ray
            if np.any(miss_mask):
                ray_kind = ray_kind.copy()
                ray_kind[miss_mask] = 2  # 2=missing_ray
        else:
            ray_anchor = np.zeros((0, 3), dtype=np.float32)
            ray_kind = np.zeros((0,), dtype=np.int64)

        anchors = np.concatenate([pt_anchor, ray_anchor], axis=0) if (n_point + n_ray) > 0 else np.zeros((0, 3))
        q_all = np.concatenate([pt_q, ray_q], axis=0) if (n_point + n_ray) > 0 else np.zeros((0, feat_dim), dtype=np.float32)
        a_all = np.concatenate([pt_a, ray_a], axis=0) if (n_point + n_ray) > 0 else np.zeros((0, feat_dim), dtype=np.float32)
        kind_all = np.concatenate([pt_kind, ray_kind], axis=0) if (n_point + n_ray) > 0 else np.zeros((0,), dtype=np.int64)

        non_missing = np.nonzero(kind_all != 2)[0]
        missing = np.nonzero(kind_all == 2)[0]

        emode = str(event_order_mode).lower()
        if non_missing.size > 0:
            if emode == "morton":
                code = morton3d(anchors[non_missing])
                non_order = non_missing[np.argsort(code)]
            elif emode == "fps":
                from nepa3d.utils.fps import fps_order

                local = fps_order(anchors[non_missing], non_missing.size)
                non_order = non_missing[local]
            elif emode in ("random", "shuffle"):
                non_order = non_missing[rng.permutation(non_missing.size)]
            else:
                raise ValueError(f"unknown event_order_mode: {event_order_mode}")
        else:
            non_order = non_missing

        # Missing rays are appended in their current order (already ray-ordered).
        order = np.concatenate([non_order, missing], axis=0) if missing.size > 0 else non_order

        q_all = q_all[order]
        a_all = a_all[order]
        kind_all = kind_all[order]

        # Map kind -> token types.
        q_type = np.zeros((order.size,), dtype=np.int64)
        a_type = np.zeros((order.size,), dtype=np.int64)
        if order.size > 0:
            is_pt = kind_all == 0
            is_ray = kind_all == 1
            is_miss = kind_all == 2
            q_type[is_pt] = TYPE_Q_POINT
            a_type[is_pt] = TYPE_A_POINT
            q_type[is_ray] = TYPE_Q_RAY
            a_type[is_ray] = TYPE_A_RAY
            q_type[is_miss] = TYPE_MISSING_RAY
            a_type[is_miss] = TYPE_MISSING_RAY

        if layout == "interleave":
            qa = np.zeros((2 * order.size, feat_dim), dtype=np.float32)
            ty = np.zeros((2 * order.size,), dtype=np.int64)
            if order.size > 0:
                qa[0::2] = q_all
                qa[1::2] = a_all
                ty[0::2] = q_type
                ty[1::2] = a_type
            feat_list = [bos_feat, qa]
            type_list = [np.array([TYPE_BOS], dtype=np.int64), ty]
        else:
            if layout == "split_sep":
                sep_feat = np.zeros((1, feat_dim), dtype=np.float32)
                feat_list = [bos_feat, q_all, sep_feat, a_all]
                type_list = [
                    np.array([TYPE_BOS], dtype=np.int64),
                    q_type,
                    np.array([TYPE_SEP], dtype=np.int64),
                    a_type,
                ]
            else:
                feat_list = [bos_feat, q_all, a_all]
                type_list = [np.array([TYPE_BOS], dtype=np.int64), q_type, a_type]

    else:
        # Default: block layout (points then rays), with optional split separator.
        if layout == "interleave":
            pt_qa = np.zeros((2 * n_point, feat_dim), dtype=np.float32)
            pt_type = np.zeros((2 * n_point,), dtype=np.int64)
            if n_point > 0:
                pt_qa[0::2] = pt_q
                pt_qa[1::2] = pt_a
                pt_type[0::2] = TYPE_Q_POINT
                pt_type[1::2] = TYPE_A_POINT

            ray_qa = np.zeros((2 * n_ray, feat_dim), dtype=np.float32)
            ray_type = np.zeros((2 * n_ray,), dtype=np.int64)
            if n_ray > 0:
                ray_qa[0::2] = ray_q
                ray_qa[1::2] = ray_a
                if ray_missing:
                    ray_type[:] = TYPE_MISSING_RAY
                    ray_qa[:] = 0.0
                else:
                    ray_type[0::2] = TYPE_Q_RAY
                    ray_type[1::2] = TYPE_A_RAY

            feat_list = [bos_feat, pt_qa, ray_qa]
            type_list = [
                np.array([TYPE_BOS], dtype=np.int64),
                pt_type,
                ray_type,
            ]
        else:
            q_parts = []
            q_types = []
            a_parts = []
            a_types = []

            if n_point > 0:
                q_parts.append(pt_q)
                q_types.append(np.full((n_point,), TYPE_Q_POINT, dtype=np.int64))
                a_parts.append(pt_a)
                a_types.append(np.full((n_point,), TYPE_A_POINT, dtype=np.int64))

            if n_ray > 0:
                q_parts.append(ray_q)
                a_parts.append(ray_a)
                if ray_missing:
                    q_types.append(np.full((n_ray,), TYPE_MISSING_RAY, dtype=np.int64))
                    a_types.append(np.full((n_ray,), TYPE_MISSING_RAY, dtype=np.int64))
                else:
                    q_types.append(np.full((n_ray,), TYPE_Q_RAY, dtype=np.int64))
                    a_types.append(np.full((n_ray,), TYPE_A_RAY, dtype=np.int64))

            q_feat = (
                np.concatenate(q_parts, axis=0)
                if len(q_parts) > 0
                else np.zeros((0, feat_dim), dtype=np.float32)
            )
            q_type = (
                np.concatenate(q_types, axis=0)
                if len(q_types) > 0
                else np.zeros((0,), dtype=np.int64)
            )
            a_feat = (
                np.concatenate(a_parts, axis=0)
                if len(a_parts) > 0
                else np.zeros((0, feat_dim), dtype=np.float32)
            )
            a_type = (
                np.concatenate(a_types, axis=0)
                if len(a_types) > 0
                else np.zeros((0,), dtype=np.int64)
            )

            if layout == "split_sep":
                sep_feat = np.zeros((1, feat_dim), dtype=np.float32)
                feat_list = [bos_feat, q_feat, sep_feat, a_feat]
                type_list = [
                    np.array([TYPE_BOS], dtype=np.int64),
                    q_type,
                    np.array([TYPE_SEP], dtype=np.int64),
                    a_type,
                ]
            else:
                feat_list = [bos_feat, q_feat, a_feat]
                type_list = [np.array([TYPE_BOS], dtype=np.int64), q_type, a_type]

    if add_eos:
        feat_list.append(eos_feat)
        type_list.append(np.array([TYPE_EOS], dtype=np.int64))

    feat = np.concatenate(feat_list, axis=0)
    type_id = np.concatenate(type_list, axis=0)
    return feat, type_id


def build_sequence(
    pt_xyz,
    pt_dist,
    ray_o,
    ray_d,
    ray_hit,
    ray_t,
    ray_n,
    n_point=512,
    n_ray=512,
    drop_ray_prob=0.0,
    ray_available=True,
    add_eos=True,
    rng=None,
    qa_tokens=False,
    qa_layout="interleave",
    sequence_mode="block",
    event_order_mode="morton",
    ray_order_mode="theta_phi",
    ray_anchor_miss_t=4.0,
    ray_view_tol=1e-6,
    pt_sample_mode="random",
    pt_fps_order=None,
    pt_rfps_order=None,
    pt_rfps_m=4096,
    point_order_mode="morton",
    include_pt_grad=False,
    pt_grad_mode="raw",
    pt_grad_eps=1e-3,
    pt_grad_clip=10.0,
    pt_grad_orient="none",
    include_ray_unc=False,
    ray_unc_k=8,
    ray_unc_mode="normal_var",
):
    """Build a token sequence from pooled queries/answers.

    Args:
        qa_tokens: if True, use the Q/A separated format (v2).
                   if False, use the legacy format (v0/v1).

    Returns:
        feat: float32 array, shape (T, 15)
        type_id: int64 array, shape (T,)
    """
    if bool(qa_tokens):
        return _build_sequence_qa(
            pt_xyz,
            pt_dist,
            ray_o,
            ray_d,
            ray_hit,
            ray_t,
            ray_n,
            n_point=n_point,
            n_ray=n_ray,
            drop_ray_prob=drop_ray_prob,
            ray_available=ray_available,
            add_eos=add_eos,
            rng=rng,
            qa_layout=qa_layout,
            sequence_mode=sequence_mode,
            event_order_mode=event_order_mode,
            ray_order_mode=ray_order_mode,
            ray_anchor_miss_t=ray_anchor_miss_t,
            ray_view_tol=ray_view_tol,
            pt_sample_mode=pt_sample_mode,
            pt_fps_order=pt_fps_order,
            pt_rfps_order=pt_rfps_order,
            pt_rfps_m=pt_rfps_m,
            point_order_mode=point_order_mode,
            include_pt_grad=include_pt_grad,
            pt_grad_mode=pt_grad_mode,
            pt_grad_eps=pt_grad_eps,
            pt_grad_clip=pt_grad_clip,
            pt_grad_orient=pt_grad_orient,
            include_ray_unc=include_ray_unc,
            ray_unc_k=ray_unc_k,
            ray_unc_mode=ray_unc_mode,
        )
    return _build_sequence_legacy(
        pt_xyz,
        pt_dist,
        ray_o,
        ray_d,
        ray_hit,
        ray_t,
        ray_n,
        n_point=n_point,
        n_ray=n_ray,
        drop_ray_prob=drop_ray_prob,
        ray_available=ray_available,
        add_eos=add_eos,
        rng=rng,
        pt_sample_mode=pt_sample_mode,
        pt_fps_order=pt_fps_order,
        pt_rfps_order=pt_rfps_order,
        pt_rfps_m=pt_rfps_m,
        point_order_mode=point_order_mode,
        include_pt_grad=include_pt_grad,
        pt_grad_mode=pt_grad_mode,
        pt_grad_eps=pt_grad_eps,
        pt_grad_clip=pt_grad_clip,
        pt_grad_orient=pt_grad_orient,
        include_ray_unc=include_ray_unc,
        ray_unc_k=ray_unc_k,
        ray_unc_mode=ray_unc_mode,
    )
