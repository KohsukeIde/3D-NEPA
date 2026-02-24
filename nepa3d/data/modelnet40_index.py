import glob
import os
import re
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


_SCANOBJ_VARIANT_TOKENS = (
    "_augmented25_norot",
    "_augmented25rot",
    "_augmentedrot_scale75_newsplit",
    "_augmentedrot_scale75",
    "_augmentedrot",
)
_TAIL_ID_RE = re.compile(r"_(\d+)$")


def scanobjectnn_group_key(npz_path):
    """Canonical group key for ScanObjectNN multi-variant caches.

    Expected filename pattern examples:
      training_objectdataset_000123.npz
      training_objectdataset_augmented25rot_000123.npz
    """
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    m = _TAIL_ID_RE.search(stem)
    if m is None:
        return stem
    idx = m.group(1)
    prefix = stem[: m.start()]
    for tok in _SCANOBJ_VARIANT_TOKENS:
        prefix = prefix.replace(tok, "")
    return f"{prefix}_{idx}"


def stratified_train_val_split(paths, val_ratio=0.1, seed=0, group_key_fn=None):
    """Deterministic stratified split by class directory name.

    Args:
        paths: list of sample paths (.../<split>/<class>/<id>.npz)
        val_ratio: fraction per class to use as validation.
        seed: RNG seed for shuffling within each class.
        group_key_fn: optional callable(path)->group_id.
            When set, split is done at group granularity (group-aware stratified split),
            then expanded back to files. This prevents same-group leakage across train/val.

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
        if group_key_fn is None:
            rng.shuffle(cls_paths)
            n = len(cls_paths)
            n_val = int(n * float(val_ratio))
            # keep at least 1 val sample when possible
            if n_val <= 0 and n >= 2:
                n_val = 1
            val.extend(cls_paths[:n_val])
            train.extend(cls_paths[n_val:])
            continue

        # Group-aware split inside each class.
        group_to_paths = defaultdict(list)
        for p in cls_paths:
            group_to_paths[str(group_key_fn(p))].append(p)
        group_ids = sorted(group_to_paths.keys())
        rng.shuffle(group_ids)

        n_group = len(group_ids)
        n_val_group = int(n_group * float(val_ratio))
        if n_val_group <= 0 and n_group >= 2:
            n_val_group = 1
        # keep at least one train group when possible
        if n_val_group >= n_group and n_group >= 2:
            n_val_group = n_group - 1

        val_groups = set(group_ids[:n_val_group])
        for gid in group_ids:
            bucket = group_to_paths[gid]
            if gid in val_groups:
                val.extend(bucket)
            else:
                train.extend(bucket)

    return sorted(train), sorted(val)
