"""Migration: add precomputed RFPS order bank to cache npz files.

Writes an RFPS order bank for fast `pt_sample_mode=rfps_cached`.

Output shape:
  (bank_size, k_eff) where k_eff=min(rfps_k, n_pool)

Typical usage:
  python -m nepa3d.data.migrate_add_pt_rfps_order_bank \
    --cache_root data/shapenet_cache_v0 \
    --splits train,test \
    --pt_key pc_xyz \
    --out_key pc_rfps_order_bank \
    --rfps_k 2048 \
    --rfps_m 4096 \
    --bank_size 8 \
    --workers 32 \
    --write_mode append
"""

from __future__ import annotations

import argparse
import io
import os
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from nepa3d.utils.fps import rfps_order


@dataclass
class Stats:
    ok: int = 0
    skip: int = 0
    fail: int = 0


def _iter_npz(cache_root: Path, splits: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for sp in splits:
        p = cache_root / sp
        if not p.exists():
            continue
        out.extend(sorted(p.rglob("*.npz")))
    return out


def _rewrite_npz(path: Path, key: str, arr: np.ndarray) -> None:
    d = np.load(path, allow_pickle=False)
    payload = {k: d[k] for k in d.files}
    payload[key] = arr

    tmp_dir = str(path.parent)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_npz_", suffix=".npz", dir=tmp_dir)
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _append_npz_key(path: Path, key: str, arr: np.ndarray) -> bool:
    member = f"{key}.npy"
    with zipfile.ZipFile(path, mode="a") as zf:
        if member in set(zf.namelist()):
            return False
        bio = io.BytesIO()
        np.lib.format.write_array(bio, np.asarray(arr), allow_pickle=False)
        zf.writestr(member, bio.getvalue(), compress_type=zipfile.ZIP_STORED)
    return True


def _filter_shard(files: list[Path], num_shards: int, shard_id: int) -> list[Path]:
    if int(num_shards) <= 1:
        return files
    ns = int(num_shards)
    sid = int(shard_id)
    if sid < 0 or sid >= ns:
        raise ValueError(f"invalid shard_id={sid} for num_shards={ns}")
    return [p for i, p in enumerate(files) if (i % ns) == sid]


def _worker(
    npz_path: str,
    *,
    pt_key: str,
    out_key: str,
    rfps_k: int,
    rfps_m: int,
    bank_size: int,
    seed: int,
    overwrite: bool,
    write_mode: str,
) -> tuple[str, str]:
    p = Path(npz_path)
    try:
        d = np.load(p, allow_pickle=False)
        if (not overwrite) and (out_key in d.files):
            return (npz_path, "skip")
        if pt_key not in d.files:
            raise KeyError(f"missing {pt_key}")

        pts = np.asarray(d[pt_key], dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"{pt_key} must be (N,3), got {pts.shape}")
        n_pool = int(pts.shape[0])
        if n_pool <= 0:
            raise ValueError(f"empty pool: {pt_key}")

        kk = min(int(rfps_k), n_pool)
        mm = min(int(rfps_m), n_pool)
        nb = max(1, int(bank_size))
        bank = np.empty((nb, kk), dtype=np.int32)
        for i in range(nb):
            # Deterministic per-bank seed for reproducible cached banks.
            rng = np.random.RandomState((int(seed) + 10007 * i) & 0xFFFFFFFF)
            bank[i] = rfps_order(pts, k=kk, m=mm, rng=rng).astype(np.int32, copy=False)

        wm = str(write_mode).lower()
        if (wm == "append") and (not overwrite):
            _append_npz_key(p, out_key, bank)
        else:
            _rewrite_npz(p, out_key, bank)
        return (npz_path, "ok")
    except Exception as e:
        return (npz_path, f"fail:{type(e).__name__}:{e}")


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=str, required=True)
    ap.add_argument("--splits", type=str, default="train,test")
    ap.add_argument("--pt_key", type=str, default="pc_xyz")
    ap.add_argument("--out_key", type=str, default="pc_rfps_order_bank")
    ap.add_argument("--rfps_k", type=int, default=2048)
    ap.add_argument("--rfps_m", type=int, default=4096)
    ap.add_argument("--bank_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--write_mode",
        type=str,
        choices=["append", "rewrite"],
        default="append",
        help=(
            "append: add new key into existing npz zip without rewriting other arrays "
            "(fast, no overwrite support). rewrite: rewrite whole npz payload (slow)."
        ),
    )
    ap.add_argument("--num_shards", type=int, default=1, help="split file list into N shards.")
    ap.add_argument("--shard_id", type=int, default=0, help="0-based shard id when num_shards>1.")
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--chunksize", type=int, default=32, help="executor.map chunksize")
    args = ap.parse_args(argv)

    cache_root = Path(args.cache_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    files = _iter_npz(cache_root, splits)
    files = _filter_shard(files, int(args.num_shards), int(args.shard_id))

    st = Stats()
    if len(files) == 0:
        print(
            f"No npz found under {cache_root} for splits={splits} "
            f"(num_shards={args.num_shards}, shard_id={args.shard_id})"
        )
        return

    print(
        f"[migrate_add_pt_rfps_order_bank] cache_root={cache_root} splits={splits} "
        f"rfps_k={args.rfps_k} rfps_m={args.rfps_m} bank_size={args.bank_size} "
        f"out_key={args.out_key} workers={args.workers} write_mode={args.write_mode} "
        f"num_shards={args.num_shards} shard_id={args.shard_id} files={len(files)}"
    )

    done = 0
    log_every = max(1, int(args.log_every))
    chunksize = max(1, int(args.chunksize))
    worker_fn = partial(
        _worker,
        pt_key=args.pt_key,
        out_key=args.out_key,
        rfps_k=args.rfps_k,
        rfps_m=args.rfps_m,
        bank_size=args.bank_size,
        seed=args.seed,
        overwrite=args.overwrite,
        write_mode=args.write_mode,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for _, status in ex.map(worker_fn, map(str, files), chunksize=chunksize):
            if status == "ok":
                st.ok += 1
            elif status == "skip":
                st.skip += 1
            else:
                st.fail += 1
            done += 1
            if (done % log_every) == 0 or done == len(files):
                print(
                    f"progress: {done}/{len(files)} "
                    f"(ok={st.ok} skip={st.skip} fail={st.fail})"
                )

    print(f"done: ok={st.ok} skip={st.skip} fail={st.fail}")


if __name__ == "__main__":
    main()

