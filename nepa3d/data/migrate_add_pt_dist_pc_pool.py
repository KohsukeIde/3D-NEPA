import argparse
import glob
import os
import tempfile
import multiprocessing as mp

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def _list_npz(cache_root: str, splits):
    pats = []
    if splits == ["all"]:
        pats.append(os.path.join(cache_root, "**", "*.npz"))
    else:
        for sp in splits:
            pats.append(os.path.join(cache_root, sp, "**", "*.npz"))
    files = []
    for pat in pats:
        files.extend(glob.glob(pat, recursive=True))
    return sorted(set(files))


def _compute_pc_dist(pc_xyz: np.ndarray, pt_xyz: np.ndarray, chunk: int = 4096) -> np.ndarray:
    pc_xyz = pc_xyz.astype(np.float32, copy=False)
    pt_xyz = pt_xyz.astype(np.float32, copy=False)
    if cKDTree is not None:
        kdt = cKDTree(pc_xyz)
        dist, _ = kdt.query(pt_xyz, k=1)
        return dist.astype(np.float32, copy=False)

    # Fallback: brute-force in chunks (slow, but keeps correctness when SciPy is absent).
    out = np.empty((pt_xyz.shape[0],), dtype=np.float32)
    chunk = max(1, int(chunk))
    for s in range(0, pt_xyz.shape[0], chunk):
        e = min(pt_xyz.shape[0], s + chunk)
        diff = pt_xyz[s:e, None, :] - pc_xyz[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        out[s:e] = np.sqrt(np.min(d2, axis=1)).astype(np.float32)
    return out


def _process_one(args):
    path, chunk, overwrite = args
    real_path = os.path.realpath(path)
    try:
        d = np.load(real_path, allow_pickle=False)
        if (not overwrite) and ("pt_dist_pc_pool" in d):
            arr = d["pt_dist_pc_pool"]
            if np.isfinite(arr).all() and np.all(arr >= 0):
                return (path, "skip")
        pc_xyz = d["pc_xyz"]
        pt_xyz = d["pt_xyz_pool"]
        dist_pc = _compute_pc_dist(pc_xyz, pt_xyz, chunk=chunk)
        # sanitize
        if not np.isfinite(dist_pc).all():
            finite = np.isfinite(dist_pc)
            fill = float(np.max(dist_pc[finite])) if finite.any() else 1.0
            dist_pc = np.nan_to_num(dist_pc, nan=fill, posinf=fill, neginf=0.0).astype(np.float32)

        # materialize all arrays and rewrite
        payload = {k: d[k] for k in d.files}
        payload["pt_dist_pc_pool"] = dist_pc.astype(np.float32, copy=False)

        tmp_dir = os.path.dirname(real_path)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_npz_", suffix=".npz", dir=tmp_dir)
        os.close(fd)
        try:
            np.savez_compressed(tmp_path, **payload)
            os.replace(tmp_path, real_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        return (path, "ok")
    except Exception as e:
        return (path, f"fail:{type(e).__name__}")


def main():
    ap = argparse.ArgumentParser(description="Add pt_dist_pc_pool to existing cache npz files (non-destructive migration).")
    ap.add_argument("--cache_root", type=str, required=True)
    ap.add_argument("--splits", type=str, default="all", help="Comma-separated list: train,test,eval,... or 'all'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=4096, help="Chunk size for brute-force fallback")
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if pt_dist_pc_pool exists")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        splits = ["all"]
    files = _list_npz(args.cache_root, splits)
    if args.limit and args.limit > 0:
        files = files[: int(args.limit)]
    if not files:
        raise FileNotFoundError(f"no npz found under: {args.cache_root}")

    tasks = [(p, int(args.chunk), bool(args.overwrite)) for p in files]
    workers = max(1, int(args.workers))

    ok = skip = fail = 0
    if workers == 1:
        for t in tasks:
            _, st = _process_one(t)
            if st == "ok":
                ok += 1
            elif st == "skip":
                skip += 1
            else:
                fail += 1
    else:
        with mp.Pool(processes=workers) as pool:
            for _, st in pool.imap_unordered(_process_one, tasks, chunksize=1):
                if st == "ok":
                    ok += 1
                elif st == "skip":
                    skip += 1
                else:
                    fail += 1

    print(f"done: ok={ok} skip={skip} fail={fail} total={len(files)}")


if __name__ == "__main__":
    main()