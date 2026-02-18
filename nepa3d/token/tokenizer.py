import numpy as np

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
):
    """Legacy tokenization: [BOS] + POINT* + RAY* (+ [EOS]).

    This is kept to allow reproducing older experiments.
    """
    if rng is None:
        rng = np.random

    pt_xyz = np.asarray(pt_xyz)
    pt_dist = np.asarray(pt_dist)
    pt_dist_1d = pt_dist[:, 0] if pt_dist.ndim == 2 else pt_dist

    p_idx = _choice(pt_xyz.shape[0], n_point, rng=rng)

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

    ray_feat = np.zeros((n_ray, feat_dim), dtype=np.float32)
    if n_ray > 0:
        hit_mask = (ray_hit_s > 0.5).astype(np.float32)
        x_hit = ray_o_s + ray_t_s[:, None] * ray_d_s
        ray_feat[:, 0:3] = x_hit * hit_mask[:, None]
        ray_feat[:, 3:6] = ray_o_s
        ray_feat[:, 6:9] = ray_d_s
        ray_feat[:, 9] = ray_t_s
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
):
    """Q/A separated tokenization: [BOS] + (Qp,Ap)* + (Qr,Ar)* (+ [EOS]).

    - Point query token: xyz only.
    - Point answer token: dist only (xyz is zeroed to reduce coordinate leakage).
    - Ray query token: ray origin+direction only.
    - Ray answer token: hit/t/normal (+ x_hit, optional via existing schema slots).

    NOTE: We **interleave** query/answer per primitive so that in a causal model,
    the query token is the immediate predecessor of its answer token.
    """
    if rng is None:
        rng = np.random

    pt_xyz = np.asarray(pt_xyz)
    pt_dist = np.asarray(pt_dist)
    pt_dist_1d = pt_dist[:, 0] if pt_dist.ndim == 2 else pt_dist

    p_idx = _choice(pt_xyz.shape[0], n_point, rng=rng)

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

    # POINT: query (xyz) then answer (dist)
    pt_q = np.zeros((n_point, feat_dim), dtype=np.float32)
    pt_a = np.zeros((n_point, feat_dim), dtype=np.float32)
    if n_point > 0:
        pt_q[:, 0:3] = pt_xyz_s
        pt_a[:, 10] = pt_dist_s

    pt_qa = np.zeros((2 * n_point, feat_dim), dtype=np.float32)
    pt_type = np.zeros((2 * n_point,), dtype=np.int64)
    if n_point > 0:
        pt_qa[0::2] = pt_q
        pt_qa[1::2] = pt_a
        pt_type[0::2] = TYPE_Q_POINT
        pt_type[1::2] = TYPE_A_POINT

    # RAY: query (o,d) then answer (hit,t,n,+x_hit)
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
        ray_a[:, 11] = ray_hit_s
        ray_a[:, 12:15] = ray_n_s * hit_mask[:, None]

    ray_missing = (not bool(ray_available)) or (
        int(n_ray) > 0 and float(drop_ray_prob) > 0.0 and _rand01(rng) < float(drop_ray_prob)
    )

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

    bos_feat = np.zeros((1, feat_dim), dtype=np.float32)
    eos_feat = np.zeros((1, feat_dim), dtype=np.float32)

    feat_list = [bos_feat, pt_qa, ray_qa]
    type_list = [
        np.array([TYPE_BOS], dtype=np.int64),
        pt_type,
        ray_type,
    ]
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
    )
