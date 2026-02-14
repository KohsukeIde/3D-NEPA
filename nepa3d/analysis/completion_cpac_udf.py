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
from ..token.tokenizer import TYPE_BOS, TYPE_EOS, TYPE_POINT, build_sequence

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
):
    rng = np.random.RandomState(_stable_seed(path, seed))
    be_ctx = make_backend(context_backend, path, voxel_grid, voxel_dilate, voxel_max_steps)
    pools = be_ctx.get_pools()
    ray_available = bool(pools.get("ray_available", True))

    feat_ctx, _ = build_sequence(
        pools["pt_xyz_pool"],
        pools["pt_dist_pool"],
        pools["ray_o_pool"],
        pools["ray_d_pool"],
        pools["ray_hit_pool"],
        pools["ray_t_pool"],
        pools["ray_n_pool"],
        n_point=int(n_context),
        n_ray=0,
        drop_ray_prob=0.0,
        ray_available=ray_available,
        add_eos=False,
        rng=rng,
    )
    ctx_pts = feat_ctx[1 : 1 + int(n_context)].astype(np.float32, copy=False)

    d = np.load(path, allow_pickle=False)
    pt_xyz_pool = d["pt_xyz_pool"].astype(np.float32, copy=False)
    idx = rng.choice(pt_xyz_pool.shape[0], size=int(n_query), replace=(pt_xyz_pool.shape[0] < int(n_query)))
    q_xyz = pt_xyz_pool[idx].astype(np.float32, copy=False)

    if "pt_dist_udf_pool" in d:
        q_dist = d["pt_dist_udf_pool"].astype(np.float32, copy=False)[idx]
    else:
        q_dist = d["pt_dist_pool"].astype(np.float32, copy=False)[idx]

    order = np.argsort(morton3d(q_xyz))
    q_xyz = q_xyz[order]
    q_dist = q_dist[order]

    q_feat = np.zeros((int(n_query), 15), dtype=np.float32)
    q_feat[:, 0:3] = q_xyz
    q_feat[:, 10] = 0.0

    bos = np.zeros((1, 15), dtype=np.float32)
    eos = np.zeros((1, 15), dtype=np.float32)

    if add_eos:
        feat = np.concatenate([bos, ctx_pts, q_feat, eos], axis=0)
        type_id = np.concatenate(
            [
                np.array([TYPE_BOS], dtype=np.int64),
                np.full((int(n_context),), TYPE_POINT, dtype=np.int64),
                np.full((int(n_query),), TYPE_POINT, dtype=np.int64),
                np.array([TYPE_EOS], dtype=np.int64),
            ],
            axis=0,
        )
    else:
        feat = np.concatenate([bos, ctx_pts, q_feat], axis=0)
        type_id = np.concatenate(
            [
                np.array([TYPE_BOS], dtype=np.int64),
                np.full((int(n_context),), TYPE_POINT, dtype=np.int64),
                np.full((int(n_query),), TYPE_POINT, dtype=np.int64),
            ],
            axis=0,
        )

    feat_t = torch.from_numpy(feat).unsqueeze(0).to(device).float()
    type_t = torch.from_numpy(type_id).unsqueeze(0).to(device).long()
    _, _, h = model(feat_t, type_t)
    h = h.squeeze(0).detach().cpu().numpy().astype(np.float32)

    q0 = 1 + int(n_context)
    q1 = q0 + int(n_query)
    X = h[q0:q1]
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
    desc,
):
    xs = []
    ys = []
    for p in tqdm(paths, desc=desc):
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
    ap.add_argument("--seed", type=int, default=0)
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
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = build_model_from_ckpt(args.ckpt, device)
    add_eos = infer_add_eos(ckpt, args.add_eos)

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
        rng = np.random.RandomState(int(args.seed) & 0xFFFFFFFF)
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

    Xtr, Ytr = collect_xy(
        model,
        tr_paths,
        head_backend,
        args.n_context,
        args.n_query,
        add_eos,
        args.seed,
        args.voxel_grid,
        args.voxel_dilate,
        args.voxel_max_steps,
        device,
        desc="collect train",
    )
    w = _ridge_fit(Xtr, Ytr, lam=float(args.ridge_lambda))

    Xte, Yte = collect_xy(
        model,
        te_paths,
        args.context_backend,
        args.n_context,
        args.n_query,
        add_eos,
        args.seed + 999,
        args.voxel_grid,
        args.voxel_dilate,
        args.voxel_max_steps,
        device,
        desc="collect test",
    )
    Yhat = _ridge_pred(Xte, w)

    out = _metrics(Yte, Yhat, tau=float(args.tau))
    out.update(
        {
            "context_backend": args.context_backend,
            "target": "udfgrid_distance_probe",
            "n_context": int(args.n_context),
            "n_query": int(args.n_query),
            "eval_split": args.split,
            "head_train_split": (head_split if not legacy_transductive else "legacy_transductive"),
            "head_train_backend": head_backend,
            "legacy_transductive": bool(legacy_transductive),
            "n_shapes_total": int(len(eval_paths)),
            "n_shapes_head_train": int(len(tr_paths)),
            "n_shapes_head_test": int(len(te_paths)),
            "ridge_lambda": float(args.ridge_lambda),
            "tau": float(args.tau),
            "add_eos": bool(add_eos),
            "ckpt": os.path.abspath(args.ckpt),
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