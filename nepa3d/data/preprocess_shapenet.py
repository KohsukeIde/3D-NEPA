import argparse
import glob
import hashlib
import multiprocessing as mp
import os
from collections import defaultdict

from tqdm import tqdm

from .preprocess_modelnet40 import distance_transform_edt, preprocess_one


def _stable_int(s):
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def _infer_synset_model(mesh_path):
    parts = mesh_path.replace("\\", "/").split("/")
    if "models" in parts:
        i = parts.index("models")
        if i >= 2:
            return parts[i - 2], parts[i - 1]
    # Fallback: .../<synset>/<model>.<ext>
    if len(parts) >= 3:
        synset = parts[-3]
        model = os.path.splitext(parts[-1])[0]
        return synset, model
    synset = "unknown"
    model = os.path.splitext(os.path.basename(mesh_path))[0]
    return synset, model


def _build_split_map(mesh_paths, test_ratio=0.1, split_seed=0):
    groups = defaultdict(list)
    for p in mesh_paths:
        synset, model = _infer_synset_model(p)
        groups[synset].append((model, p))

    split_map = {}
    for synset, arr in groups.items():
        arr = sorted(arr, key=lambda x: x[0])
        n = len(arr)
        n_test = int(round(float(test_ratio) * n))
        if n >= 10:
            n_test = max(1, n_test)
        else:
            n_test = max(0, n_test)
        n_test = min(n_test, n)
        perm = sorted(
            range(n),
            key=lambda i: _stable_int(f"{synset}/{arr[i][0]}:{int(split_seed)}"),
        )
        test_idx = set(perm[:n_test])
        for i, (_, p) in enumerate(arr):
            split_map[p] = "test" if i in test_idx else "train"
    return split_map


def _worker(task):
    (
        mesh_path,
        out_path,
        pc_points,
        pt_pool,
        ray_pool,
        n_views,
        rays_per_view,
        seed,
        pc_grid,
        pc_dilate,
        pc_max_steps,
        compute_pc_rays,
        df_grid,
        df_dilate,
        compute_udf,
        pt_surface_ratio,
        pt_surface_sigma,
        pt_query_chunk,
        ray_query_chunk,
        pt_dist_mode,
        dist_ref_points,
    ) = task
    ok = preprocess_one(
        mesh_path,
        out_path,
        pc_points=pc_points,
        pt_pool=pt_pool,
        ray_pool=ray_pool,
        n_views=n_views,
        rays_per_view=rays_per_view,
        seed=seed,
        pc_grid=pc_grid,
        pc_dilate=pc_dilate,
        pc_max_steps=pc_max_steps,
        compute_pc_rays=compute_pc_rays,
        df_grid=df_grid,
        df_dilate=df_dilate,
        compute_udf=compute_udf,
        pt_surface_ratio=pt_surface_ratio,
        pt_surface_sigma=pt_surface_sigma,
        pt_query_chunk=pt_query_chunk,
        ray_query_chunk=ray_query_chunk,
        pt_dist_mode=pt_dist_mode,
        dist_ref_points=dist_ref_points,
    )
    return (mesh_path, ok)


def _write_split_manifest(out_root, split_map):
    split_dir = os.path.join(out_root, "_splits")
    os.makedirs(split_dir, exist_ok=True)
    train_txt = os.path.join(split_dir, "train.txt")
    test_txt = os.path.join(split_dir, "test.txt")
    with open(train_txt, "w", encoding="utf-8") as fw_tr, open(
        test_txt, "w", encoding="utf-8"
    ) as fw_te:
        for p in sorted(split_map):
            if split_map[p] == "train":
                fw_tr.write(p + "\n")
            else:
                fw_te.write(p + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapenet_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--split", type=str, choices=["train", "test", "all"], default="all")
    ap.add_argument(
        "--mesh_glob",
        type=str,
        default="*/*/models/model_normalized.obj",
        help="glob pattern under --shapenet_root for mesh files",
    )
    ap.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="test split ratio used when split files are not provided",
    )
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--pc_points", type=int, default=2048)
    ap.add_argument("--pt_pool", type=int, default=20000)
    ap.add_argument("--ray_pool", type=int, default=8000)
    ap.add_argument("--n_views", type=int, default=20)
    ap.add_argument("--rays_per_view", type=int, default=400)
    ap.add_argument("--pc_grid", type=int, default=64)
    ap.add_argument("--pc_dilate", type=int, default=1)
    ap.add_argument("--pc_max_steps", type=int, default=0)
    ap.add_argument("--no_pc_rays", action="store_true")
    ap.add_argument("--df_grid", type=int, default=64)
    ap.add_argument("--df_dilate", type=int, default=1)
    ap.add_argument("--no_udf", action="store_true")
    ap.add_argument("--pt_surface_ratio", type=float, default=0.5)
    ap.add_argument("--pt_surface_sigma", type=float, default=0.02)
    ap.add_argument("--pt_query_chunk", type=int, default=2048)
    ap.add_argument("--ray_query_chunk", type=int, default=2048)
    ap.add_argument(
        "--pt_dist_mode",
        type=str,
        choices=["mesh", "kdtree"],
        default="kdtree",
    )
    ap.add_argument("--dist_ref_points", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=1)
    ap.add_argument("--max_tasks_per_child", type=int, default=2)
    args = ap.parse_args()

    if (not args.no_udf) and (distance_transform_edt is None):
        raise RuntimeError(
            "SciPy distance_transform_edt is required for UDF. "
            "Install scipy or rerun with --no_udf."
        )

    mesh_pattern = os.path.join(args.shapenet_root, args.mesh_glob)
    mesh_paths = sorted(glob.glob(mesh_pattern))
    if not mesh_paths:
        raise FileNotFoundError(f"no mesh found with pattern: {mesh_pattern}")

    split_map = _build_split_map(
        mesh_paths, test_ratio=float(args.test_ratio), split_seed=int(args.split_seed)
    )
    _write_split_manifest(args.out_root, split_map)

    if args.split == "all":
        target_paths = mesh_paths
    else:
        target_paths = [p for p in mesh_paths if split_map.get(p) == args.split]

    seed_base = None if args.seed < 0 else int(args.seed)
    workers = max(1, int(args.workers))
    chunk_size = max(1, int(args.chunk_size))
    max_tasks = None if int(args.max_tasks_per_child) <= 0 else int(args.max_tasks_per_child)

    tasks = []
    skip_count = 0
    for i, mesh_path in enumerate(target_paths):
        synset, model = _infer_synset_model(mesh_path)
        split_name = split_map.get(mesh_path, "train")
        out_path = os.path.join(args.out_root, split_name, synset, f"{model}.npz")
        if os.path.exists(out_path) and not args.overwrite:
            skip_count += 1
            continue
        seed = None if seed_base is None else seed_base + i
        tasks.append(
            (
                mesh_path,
                out_path,
                args.pc_points,
                args.pt_pool,
                args.ray_pool,
                args.n_views,
                args.rays_per_view,
                seed,
                args.pc_grid,
                args.pc_dilate,
                args.pc_max_steps,
                (not args.no_pc_rays),
                args.df_grid,
                args.df_dilate,
                (not args.no_udf),
                args.pt_surface_ratio,
                args.pt_surface_sigma,
                args.pt_query_chunk,
                args.ray_query_chunk,
                args.pt_dist_mode,
                args.dist_ref_points,
            )
        )

    if skip_count > 0:
        print(f"skip existing: {skip_count}")
    if not tasks:
        print("nothing to do")
        return

    if workers == 1:
        for task in tqdm(tasks, desc=f"preprocess shapenet ({args.split})"):
            mesh_path, ok = _worker(task)
            if not ok:
                print(f"skip: {mesh_path}")
    else:
        with mp.Pool(processes=workers, maxtasksperchild=max_tasks) as pool:
            for mesh_path, ok in tqdm(
                pool.imap_unordered(_worker, tasks, chunksize=chunk_size),
                total=len(tasks),
                desc=f"preprocess shapenet ({args.split}) [w={workers}]",
            ):
                if not ok:
                    print(f"skip: {mesh_path}")


if __name__ == "__main__":
    main()

