import numpy as np
import warnings

from .ordering import morton3d, sort_by_ray_direction

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

    return rng.choice(n_pool, size=k, replace=False).astype(np.int64, copy=False)


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
    pt_rfps_m=4096,
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

    if pt_xyz_s.shape[0] > 0:
        p_order = np.argsort(morton3d(pt_xyz_s))
        pt_xyz_s = pt_xyz_s[p_order]
        pt_dist_s = pt_dist_s[p_order]

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
    pt_sample_mode="random",
    pt_fps_order=None,
    pt_rfps_m=4096,
    include_pt_grad=False,
    pt_grad_mode="raw",
    pt_grad_eps=1e-3,
    pt_grad_clip=10.0,
    pt_grad_orient="none",
    include_ray_unc=False,
    ray_unc_k=8,
    ray_unc_mode="normal_var",
):
    """Q/A separated tokenization: [BOS] + (Qp,Ap)* + (Qr,Ar)* (+ [EOS]).

    - Point query token: xyz only.
    - Point answer token: dist only (xyz is zeroed to reduce coordinate leakage).
    - Ray query token: ray origin+direction only.
    - Ray answer token: hit/t/normal (+ x_hit, optional via existing schema slots).

    Supported layouts:
    - interleave: [BOS] + (Q, A)* + [EOS]
    - split:      [BOS] + Q... + A... + [EOS]
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

    # ordering (same as legacy)
    if pt_xyz_s.shape[0] > 0:
        p_order = np.argsort(morton3d(pt_xyz_s))
        pt_xyz_s = pt_xyz_s[p_order]
        pt_dist_s = pt_dist_s[p_order]

    if ray_d_s.shape[0] > 0:
        r_order = sort_by_ray_direction(ray_d_s)
        ray_o_s = ray_o_s[r_order]
        ray_d_s = ray_d_s[r_order]
        ray_hit_s = ray_hit_s[r_order]
        ray_t_s = ray_t_s[r_order]
        ray_n_s = ray_n_s[r_order]

    feat_dim = 15

    layout = str(qa_layout).lower()
    if layout not in ("interleave", "split"):
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

        q_feat = np.concatenate(q_parts, axis=0) if len(q_parts) > 0 else np.zeros((0, feat_dim), dtype=np.float32)
        q_type = np.concatenate(q_types, axis=0) if len(q_types) > 0 else np.zeros((0,), dtype=np.int64)
        a_feat = np.concatenate(a_parts, axis=0) if len(a_parts) > 0 else np.zeros((0, feat_dim), dtype=np.float32)
        a_type = np.concatenate(a_types, axis=0) if len(a_types) > 0 else np.zeros((0,), dtype=np.int64)

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
    pt_sample_mode="random",
    pt_fps_order=None,
    pt_rfps_m=4096,
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
            pt_sample_mode=pt_sample_mode,
            pt_fps_order=pt_fps_order,
            pt_rfps_m=pt_rfps_m,
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
        pt_rfps_m=pt_rfps_m,
        include_pt_grad=include_pt_grad,
        pt_grad_mode=pt_grad_mode,
        pt_grad_eps=pt_grad_eps,
        pt_grad_clip=pt_grad_clip,
        pt_grad_orient=pt_grad_orient,
        include_ray_unc=include_ray_unc,
        ray_unc_k=ray_unc_k,
        ray_unc_mode=ray_unc_mode,
    )
