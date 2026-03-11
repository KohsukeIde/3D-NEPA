"""Small helper to inspect a v2 world-package NPZ."""
from __future__ import annotations

import argparse
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=str)
    args = ap.parse_args()
    with np.load(args.npz, allow_pickle=False) as npz:
        for k in sorted(npz.files):
            arr = np.asarray(npz[k])
            print(f"{k:28s} shape={arr.shape!s:18s} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
