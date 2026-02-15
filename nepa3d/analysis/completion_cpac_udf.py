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


def _sample_udf_grid_queries(d, rng, n_query):
    if "udf_grid" not in d:
        raise RuntimeError("query_source=grid requires `udf_grid` in cache npz")
    udf = d["udf_grid"].astype(np.float32, copy=False)
    if udf.ndim != 3 or (udf.shape[0] != udf.shape[1]) or (udf.shape[0] != udf.shape[2]):
        raise RuntimeError(f"udf_grid must be cubic, got shape={udf.shape}")
    g = int(udf.shape[0])
    total = g * g * g
    n = int(min(int(n_query), total))
    lin = rng.choice(total, size=n, replace=False).astype(np.int64)

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
):
    rng = np.random.RandomState(_stable_seed(path, seed))
    d = np.load(path, allow_pickle=False)
    pt_xyz_pool = d["pt_xyz_pool"].astype(np.float32, copy=False)
    n_pool = int(pt_xyz_pool.shape[0])

    if context_mode == "normal" and bool(disjoint_context_query):
        cidx, qidx_pool = _sample_disjoint_indices(n_pool, int(n_context), int(n_query), rng)
    else:
        cidx = _sample_indices(n_pool, int(n_context), rng)
        qidx_pool = _sample_indices(n_pool, int(n_query), rng)

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

    query_source = str(query_source)
    if query_source == "pool":
        q_xyz = pt_xyz_pool[qidx_pool].astype(np.float32, copy=False)
        if "pt_dist_udf_pool" in d:
            q_dist = d["pt_dist_udf_pool"].astype(np.float32, copy=False)[qidx_pool]
        else:
            q_dist = d["pt_dist_pool"].astype(np.float32, copy=False)[qidx_pool]
    elif query_source == "grid":
        q_xyz, q_dist = _sample_udf_grid_queries(d, rng=rng, n_query=int(n_query))
    else:
        raise ValueError(f"unknown query_source: {query_source}")

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
    context_mode="normal",
    disjoint_context_query=True,
    mismatch_path=None,
    rep_source="h",
    query_source="pool",
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
    )

    bos = np.zeros((1, 15), dtype=np.float32)
    eos = np.zeros((1, 15), dtype=np.float32)

    n_q_eff = int(q_xyz.shape[0])

    if bool(qa_tokens):
        ctx_q = np.zeros((int(n_ctx_eff), 15), dtype=np.float32)
        ctx_a = np.zeros((int(n_ctx_eff), 15), dtype=np.float32)
        if n_ctx_eff > 0:
            ctx_q[:, 0:3] = ctx_xyz
            ctx_a[:, 10] = ctx_dist

        qry_q = np.zeros((n_q_eff, 15), dtype=np.float32)
        qry_a = np.zeros((n_q_eff, 15), dtype=np.float32)
        qry_q[:, 0:3] = q_xyz
        qry_a[:, 10] = 0.0

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
        q_start = 1 + 2 * int(n_ctx_eff)
        q_pos = q_start + 2 * np.arange(n_q_eff, dtype=np.int64)
        X = rep[q_pos]
    else:
        q0 = 1 + int(n_ctx_eff)
        q1 = q0 + n_q_eff
        X = rep[q0:q1]

    y = q_dist.reshape(-1, 1).astype(np.float32, copy=False)
    return X, y


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
    desc,
    context_mode="normal",
    disjoint_context_query=True,
    mismatch_shift=1,
    rep_source="h",
    query_source="pool",
):
    mismatch_paths = [None for _ in range(len(paths))]
    if context_mode == "mismatch" and len(paths) > 1:
        shift = int(mismatch_shift) % len(paths)
        if shift == 0:
            shift = 1
        mismatch_paths = [paths[(i + shift) % len(paths)] for i in range(len(paths))]

    xs = []
    ys = []
    for i, p in enumerate(tqdm(paths, desc=desc)):
        X, y = extract_xy_for_path(
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
            context_mode=context_mode,
            disjoint_context_query=bool(disjoint_context_query),
            mismatch_path=mismatch_paths[i],
            rep_source=rep_source,
            query_source=query_source,
        )
        xs.append(X)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="eval")
    ap.add_argument("--ckpt", type=str, required=True)
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
        choices=["pool", "grid"],
        help=("Query source: pool=sample from pt_xyz_pool; grid=sample voxel centers from udf_grid."),
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
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    # Backward compatibility for existing wrappers.
    if args.seed is not None:
        args.eval_seed = int(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = build_model_from_ckpt(args.ckpt, device)
    add_eos = infer_add_eos(ckpt, args.add_eos)
    qa_tokens = infer_qa_tokens(ckpt, args.qa_tokens)

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
                "eval_split": args.split,
                "head_train_split": (head_split if not legacy_transductive else "legacy_transductive"),
                "head_train_backend": head_backend,
                "legacy_transductive": bool(legacy_transductive),
                "eval_seed": int(args.eval_seed),
                "n_shapes_total": int(len(eval_paths)),
                "n_shapes_head_train": int(len(tr_paths)),
                "n_shapes_head_test": int(len(te_paths)),
                "ridge_lambda": float(args.ridge_lambda),
                "tau": float(args.tau),
                "disjoint_context_query": bool(args.disjoint_context_query),
                "context_mode_train": str(args.context_mode_train),
                "context_mode_test": str(args.context_mode_test),
                "mismatch_shift": int(args.mismatch_shift),
                "rep_source": str(args.rep_source),
                "add_eos": bool(add_eos),
                "qa_tokens": bool(qa_tokens),
                "ckpt": os.path.abspath(args.ckpt),
                "baseline": baseline,
                "baseline_only": True,
            }
        )
        print(json.dumps(out, indent=2))
        if args.out_json:
            os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"[saved] {args.out_json}")
        return

    Xtr, Ytr = collect_xy(
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
        desc="collect train",
        context_mode=args.context_mode_train,
        disjoint_context_query=bool(args.disjoint_context_query),
        mismatch_shift=int(args.mismatch_shift),
        rep_source=args.rep_source,
        query_source=args.query_source,
    )
    if args.head_train_n and args.head_train_n > 0 and Xtr.shape[0] > int(args.head_train_n):
        rng_cap = np.random.RandomState(int(args.eval_seed) & 0xFFFFFFFF)
        sel = rng_cap.permutation(Xtr.shape[0])[: int(args.head_train_n)]
        Xtr = Xtr[sel]
        Ytr = Ytr[sel]
    w = _ridge_fit(Xtr, Ytr, lam=float(args.ridge_lambda))

    Xte, Yte = collect_xy(
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
        desc="collect test",
        context_mode=args.context_mode_test,
        disjoint_context_query=bool(args.disjoint_context_query),
        mismatch_shift=int(args.mismatch_shift),
        rep_source=args.rep_source,
        query_source=args.query_source,
    )
    Yhat = _ridge_pred(Xte, w)

    out = _metrics(Yte, Yhat, tau=float(args.tau))
    out.update(
        {
            "context_backend": args.context_backend,
            "target": "udfgrid_distance_probe",
            "n_context": int(args.n_context),
            "n_query": int(args.n_query),
            "query_source": str(args.query_source),
            "eval_split": args.split,
            "head_train_split": (head_split if not legacy_transductive else "legacy_transductive"),
            "head_train_backend": head_backend,
            "legacy_transductive": bool(legacy_transductive),
            "eval_seed": int(args.eval_seed),
            "n_shapes_total": int(len(eval_paths)),
            "n_shapes_head_train": int(len(tr_paths)),
            "n_shapes_head_test": int(len(te_paths)),
            "ridge_lambda": float(args.ridge_lambda),
            "tau": float(args.tau),
            "disjoint_context_query": bool(args.disjoint_context_query),
            "context_mode_train": str(args.context_mode_train),
            "context_mode_test": str(args.context_mode_test),
            "mismatch_shift": int(args.mismatch_shift),
            "rep_source": str(args.rep_source),
            "add_eos": bool(add_eos),
            "qa_tokens": bool(qa_tokens),
            "ckpt": os.path.abspath(args.ckpt),
            "baseline": baseline,
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
