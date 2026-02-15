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


def _stable_seed(path: str, seed: int):
    h = hashlib.md5(path.encode("utf-8")).hexdigest()[:8]
    return (int(h, 16) + int(seed)) & 0xFFFFFFFF


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


def _sample_udf_grid_queries(d: np.lib.npyio.NpzFile, rng: np.random.RandomState, n_query: int):
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

    total = int(n_context) + int(n_query)
    if context_mode == "normal" and bool(disjoint_context_query):
        replace = n_pool < total
        idx = rng.choice(n_pool, size=total, replace=replace).astype(np.int64)
        cidx, qidx_pool = idx[: int(n_context)], idx[int(n_context):]
    else:
        cidx = rng.choice(n_pool, size=int(n_context), replace=n_pool < int(n_context)).astype(np.int64)
        qidx_pool = rng.choice(n_pool, size=int(n_query), replace=n_pool < int(n_query)).astype(np.int64)

    if context_mode == "none":
        ctx_xyz = np.zeros((0, 3), dtype=np.float32)
        ctx_dist = np.zeros((0,), dtype=np.float32)
    else:
        ctx_path = mismatch_path if (context_mode == "mismatch" and mismatch_path is not None) else path
        be_ctx = make_backend(context_backend, ctx_path, voxel_grid, voxel_dilate, voxel_max_steps)
        pools_ctx = be_ctx.get_pools()
        pt_xyz_ctx = pools_ctx["pt_xyz_pool"].astype(np.float32, copy=False)
        pt_dist_ctx = pools_ctx["pt_dist_pool"].astype(np.float32, copy=False)
        n_ctx_pool = int(pt_xyz_ctx.shape[0])
        if ctx_path != path or (cidx.size > 0 and int(cidx.max()) >= n_ctx_pool):
            cidx = rng.choice(n_ctx_pool, size=int(n_context), replace=n_ctx_pool < int(n_context)).astype(np.int64)
        ctx_xyz = pt_xyz_ctx[cidx]
        ctx_dist = pt_dist_ctx[cidx]
        if ctx_xyz.shape[0] > 0:
            order = np.argsort(morton3d(ctx_xyz))
            ctx_xyz = ctx_xyz[order]
            ctx_dist = ctx_dist[order]

    query_source = str(query_source)
    if query_source == "pool":
        qry_xyz = pt_xyz_pool[qidx_pool]
        if "pt_dist_udf_pool" in d:
            qry_dist = d["pt_dist_udf_pool"].astype(np.float32, copy=False)[qidx_pool]
        else:
            qry_dist = d["pt_dist_pool"].astype(np.float32, copy=False)[qidx_pool]
    elif query_source == "grid":
        qry_xyz, qry_dist = _sample_udf_grid_queries(d, rng=rng, n_query=int(n_query))
    else:
        raise ValueError(f"unknown query_source: {query_source}")

    if qry_xyz.shape[0] > 0:
        order = np.argsort(morton3d(qry_xyz))
        qry_xyz = qry_xyz[order]
        qry_dist = qry_dist[order]
    return ctx_xyz.astype(np.float32), ctx_dist.astype(np.float32), qry_xyz.astype(np.float32), qry_dist.astype(np.float32)


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
        ctx_xyz, ctx_dist, q_xyz, q_dist = _sample_ctx_query(
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


@torch.no_grad()
def extract_xy_for_path(
    model,
    path,
    context_backend,
    n_context,
    n_query,
    seed,
    voxel_grid,
    voxel_dilate,
    voxel_max_steps,
    device,
    context_mode="normal",
    disjoint_context_query=True,
    mismatch_path=None,
    query_source="pool",
):
    ctx_xyz, ctx_dist, q_xyz, q_dist = _sample_ctx_query(
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
    ctx_xyz_t = torch.from_numpy(ctx_xyz).unsqueeze(0).to(device).float()
    ctx_dist_t = torch.from_numpy(ctx_dist).unsqueeze(0).to(device).float()
    q_xyz_t = torch.from_numpy(q_xyz).unsqueeze(0).to(device).float()

    _, q_feat, _, _ = model(ctx_xyz_t, ctx_dist_t, q_xyz_t)
    X = q_feat.squeeze(0).detach().cpu().numpy().astype(np.float32)
    y = q_dist.reshape(-1, 1).astype(np.float32, copy=False)
    return X, y


def collect_xy(
    model,
    paths,
    context_backend,
    n_context,
    n_query,
    seed,
    voxel_grid,
    voxel_dilate,
    voxel_max_steps,
    device,
    desc,
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

    xs, ys = [], []
    for i, p in enumerate(tqdm(paths, desc=desc)):
        X, y = extract_xy_for_path(
            model,
            p,
            context_backend,
            n_context,
            n_query,
            seed,
            voxel_grid,
            voxel_dilate,
            voxel_max_steps,
            device,
            context_mode=context_mode,
            disjoint_context_query=bool(disjoint_context_query),
            mismatch_path=mismatch_paths[i],
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
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--head_train_ratio", type=float, default=0.2)
    ap.add_argument("--head_train_n", type=int, default=0)
    ap.add_argument("--head_train_split", type=str, default="train_udf")
    ap.add_argument("--head_train_backend", type=str, default="udfgrid")
    ap.add_argument("--head_train_max_shapes", type=int, default=0)
    ap.add_argument("--ridge_lambda", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.03)
    ap.add_argument("--max_shapes", type=int, default=0)
    ap.add_argument("--voxel_grid", type=int, default=64)
    ap.add_argument("--voxel_dilate", type=int, default=1)
    ap.add_argument("--voxel_max_steps", type=int, default=0)
    ap.add_argument("--disjoint_context_query", type=int, default=1)
    ap.add_argument("--context_mode_train", type=str, default="normal", choices=["normal", "none", "mismatch"])
    ap.add_argument("--context_mode_test", type=str, default="normal", choices=["normal", "none", "mismatch"])
    ap.add_argument("--mismatch_shift", type=int, default=1)
    ap.add_argument("--query_source", type=str, default="pool", choices=["pool", "grid"])
    ap.add_argument("--baseline", type=str, default="none", choices=["none", "nn_copy"])
    ap.add_argument("--baseline_only", type=int, default=0)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = build_kplane_from_ckpt(args.ckpt, device)

    eval_paths = list_npz(args.cache_root, args.split)
    if args.max_shapes and int(args.max_shapes) > 0:
        eval_paths = eval_paths[: int(args.max_shapes)]
    if len(eval_paths) < 1:
        raise RuntimeError("need at least 1 shape for evaluation")

    head_split = (args.head_train_split or "").strip()
    legacy_transductive = (head_split == "") or (head_split.lower() == "none")

    if not legacy_transductive:
        tr_paths = list_npz(args.cache_root, head_split)
        if args.head_train_max_shapes and int(args.head_train_max_shapes) > 0:
            tr_paths = tr_paths[: int(args.head_train_max_shapes)]
        if len(tr_paths) < 1:
            raise RuntimeError(f"no shapes found for head_train_split={head_split}")
        te_paths = eval_paths
        head_backend = args.head_train_backend
    else:
        if len(eval_paths) < 2:
            raise RuntimeError("need at least 2 shapes for legacy transductive head split")
        rng = np.random.RandomState(int(args.eval_seed) & 0xFFFFFFFF)
        perm = np.arange(len(eval_paths))
        rng.shuffle(perm)
        if args.head_train_n and int(args.head_train_n) > 0:
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
                "ckpt": os.path.abspath(args.ckpt),
                "kplane_cfg": ckpt.get("kplane_cfg", {}),
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
        model=model,
        paths=tr_paths,
        context_backend=head_backend,
        n_context=args.n_context,
        n_query=args.n_query,
        seed=args.eval_seed,
        voxel_grid=args.voxel_grid,
        voxel_dilate=args.voxel_dilate,
        voxel_max_steps=args.voxel_max_steps,
        device=device,
        desc="collect train",
        context_mode=args.context_mode_train,
        disjoint_context_query=bool(args.disjoint_context_query),
        mismatch_shift=int(args.mismatch_shift),
        query_source=args.query_source,
    )
    if args.head_train_n and int(args.head_train_n) > 0 and Xtr.shape[0] > int(args.head_train_n):
        rng_cap = np.random.RandomState(int(args.eval_seed) & 0xFFFFFFFF)
        sel = rng_cap.permutation(Xtr.shape[0])[: int(args.head_train_n)]
        Xtr = Xtr[sel]
        Ytr = Ytr[sel]
    w = _ridge_fit(Xtr, Ytr, lam=float(args.ridge_lambda))

    Xte, Yte = collect_xy(
        model=model,
        paths=te_paths,
        context_backend=args.context_backend,
        n_context=args.n_context,
        n_query=args.n_query,
        seed=args.eval_seed + 999,
        voxel_grid=args.voxel_grid,
        voxel_dilate=args.voxel_dilate,
        voxel_max_steps=args.voxel_max_steps,
        device=device,
        desc="collect test",
        context_mode=args.context_mode_test,
        disjoint_context_query=bool(args.disjoint_context_query),
        mismatch_shift=int(args.mismatch_shift),
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
            "ckpt": os.path.abspath(args.ckpt),
            "kplane_cfg": ckpt.get("kplane_cfg", {}),
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
