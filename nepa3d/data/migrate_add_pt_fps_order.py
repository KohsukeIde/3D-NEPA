"""Migration: add precomputed FPS order to cache npz files.

Computes deterministic FPS order from `pt_xyz_pool` and writes it as
`pt_fps_order` for classification-time sampling.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from nepa3d.utils.fps import fps_order


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


def _worker(
    npz_path: str,
    *,
    pt_key: str,
    out_key: str,
    fps_k: int,
    overwrite: bool,
) -> tuple[str, str]:
    p = Path(npz_path)
    try:
        d = np.load(p, allow_pickle=False)
        if (not overwrite) and (out_key in d.files):
            return (npz_path, "skip")
        if pt_key not in d.files:
            raise KeyError(f"missing {pt_key}")
        pts = d[pt_key]
        order = fps_order(pts, fps_k)
        _rewrite_npz(p, out_key, order.astype(np.int32, copy=False))
        return (npz_path, "ok")
    except Exception as e:
        return (npz_path, f"fail:{type(e).__name__}:{e}")


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=str, required=True)
    ap.add_argument("--splits", type=str, default="train,test")
    ap.add_argument("--fps_k", type=int, default=2048)
    ap.add_argument("--pt_key", type=str, default="pt_xyz_pool")
    ap.add_argument("--out_key", type=str, default="pt_fps_order")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    cache_root = Path(args.cache_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    files = _iter_npz(cache_root, splits)

    st = Stats()
    if len(files) == 0:
        print(f"No npz found under {cache_root} for splits={splits}")
        return

    print(
        f"[migrate_add_pt_fps_order] cache_root={cache_root} splits={splits} "
        f"fps_k={args.fps_k} out_key={args.out_key} workers={args.workers}"
    )

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                _worker,
                str(p),
                pt_key=args.pt_key,
                out_key=args.out_key,
                fps_k=args.fps_k,
                overwrite=args.overwrite,
            )
            for p in files
        ]
        for fut in as_completed(futs):
            _, status = fut.result()
            if status == "ok":
                st.ok += 1
            elif status == "skip":
                st.skip += 1
            else:
                st.fail += 1

    print(f"done: ok={st.ok} skip={st.skip} fail={st.fail}")


if __name__ == "__main__":
    main()
