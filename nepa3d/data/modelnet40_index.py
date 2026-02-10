import glob
import os
from collections import defaultdict

import numpy as np


def list_npz(cache_root, split):
    pattern = os.path.join(cache_root, split, "*", "*.npz")
    return sorted(glob.glob(pattern))


def label_from_path(npz_path):
    return npz_path.split(os.sep)[-2]


def build_label_map(paths):
    classes = sorted({label_from_path(p) for p in paths})
    return {c: i for i, c in enumerate(classes)}


def stratified_train_val_split(paths, val_ratio=0.1, seed=0):
    """Deterministic stratified split by class directory name.

    Args:
        paths: list of sample paths (.../<split>/<class>/<id>.npz)
        val_ratio: fraction per class to use as validation.
        seed: RNG seed for shuffling within each class.

    Returns:
        train_paths, val_paths (both sorted lists)

    Notes:
        - Split happens *within each class* to avoid class imbalance.
        - Deterministic given (paths, val_ratio, seed).
    """
    paths = list(paths)
    if val_ratio <= 0.0:
        return sorted(paths), []

    groups = defaultdict(list)
    for p in paths:
        groups[label_from_path(p)].append(p)

    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)

    train, val = [], []
    for cls, cls_paths in groups.items():
        cls_paths = sorted(cls_paths)
        rng.shuffle(cls_paths)
        n = len(cls_paths)
        n_val = int(n * float(val_ratio))
        # keep at least 1 val sample when possible
        if n_val <= 0 and n >= 2:
            n_val = 1
        val.extend(cls_paths[:n_val])
        train.extend(cls_paths[n_val:])

    return sorted(train), sorted(val)
