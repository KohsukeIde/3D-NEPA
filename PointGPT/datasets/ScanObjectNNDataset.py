import os
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .build import DATASETS
from utils.logger import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def _load_h5_pair(path):
    h5 = h5py.File(path, 'r')
    points = np.array(h5['data']).astype(np.float32)
    labels = np.array(h5['label']).astype(int)
    h5.close()
    return points, labels


def _stratified_train_val_indices(labels, val_ratio, val_seed):
    if not (0.0 < float(val_ratio) < 1.0):
        raise ValueError(f'val_ratio must be in (0,1), got {val_ratio}')

    labels = np.asarray(labels).reshape(-1)
    rng = np.random.RandomState(int(val_seed))
    train_parts = []
    val_parts = []

    for cls in np.unique(labels):
        cls_idx = np.flatnonzero(labels == cls)
        rng.shuffle(cls_idx)
        if len(cls_idx) <= 1:
            train_parts.append(cls_idx)
            continue
        n_val = int(round(len(cls_idx) * float(val_ratio)))
        n_val = max(1, min(len(cls_idx) - 1, n_val))
        val_parts.append(cls_idx[:n_val])
        train_parts.append(cls_idx[n_val:])

    train_idx = np.concatenate(train_parts, axis=0)
    val_idx = np.concatenate(val_parts, axis=0)
    train_idx.sort()
    val_idx.sort()
    return train_idx, val_idx


def _load_scan_subset(root, train_file, test_file, subset, val_ratio=None, val_seed=0):
    if subset == 'test':
        return _load_h5_pair(os.path.join(root, test_file))

    points, labels = _load_h5_pair(os.path.join(root, train_file))
    if subset == 'train':
        if val_ratio is None:
            return points, labels
        train_idx, _ = _stratified_train_val_indices(labels, val_ratio, val_seed)
        return points[train_idx], labels[train_idx]
    if subset == 'val':
        if val_ratio is None:
            raise ValueError('subset=val requires val_ratio')
        _, val_idx = _stratified_train_val_indices(labels, val_ratio, val_seed)
        return points[val_idx], labels[val_idx]
    raise NotImplementedError(f'unsupported subset: {subset}')


class _BaseScanObjectNN(Dataset):
    train_file = ''
    test_file = ''

    def __init__(self, config, **kwargs):
        super().__init__()
        self.subset = config.subset
        self.root = config.ROOT
        self.val_ratio = getattr(config, 'val_ratio', None)
        self.val_seed = getattr(config, 'val_seed', 0)
        self.points, self.labels = _load_scan_subset(
            self.root,
            self.train_file,
            self.test_file,
            self.subset,
            val_ratio=self.val_ratio,
            val_seed=self.val_seed,
        )
        print(
            f'Successfully load ScanObjectNN subset={self.subset} '
            f'shape={self.points.shape} val_ratio={self.val_ratio}'
        )

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])  # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]
        return 'ScanObjectNN', 'sample', (current_points, label)

    def __len__(self):
        return self.points.shape[0]


@DATASETS.register_module()
class ScanObjectNN(_BaseScanObjectNN):
    train_file = 'training_objectdataset.h5'
    test_file = 'test_objectdataset.h5'


@DATASETS.register_module()
class ScanObjectNN_hardest(_BaseScanObjectNN):
    train_file = 'training_objectdataset_augmentedrot_scale75.h5'
    test_file = 'test_objectdataset_augmentedrot_scale75.h5'
