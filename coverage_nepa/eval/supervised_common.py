from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from coverage_nepa.data.coverage_dataset import CoverageStateDataset
from coverage_nepa.data.coverage_utils import raw_stats_features, voxel_keys
from coverage_nepa.models.simple_point_encoder import SimplePointEncoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def infer_num_levels(ds: CoverageStateDataset) -> int:
    levels = [int(ds[i]["level"].item()) for i in range(len(ds))]
    return max(levels) + 1 if levels else 0


def build_category_map(*datasets: CoverageStateDataset) -> dict[str, int]:
    cats: set[str] = set()
    for ds in datasets:
        for i in range(len(ds)):
            cats.add(str(ds[i]["category"]))
    return {c: i for i, c in enumerate(sorted(cats))}


def dataset_indices_for_level(ds: CoverageStateDataset, level: int) -> list[int]:
    return [i for i in range(len(ds)) if int(ds[i]["level"].item()) == level]


def collect_labels(
    ds: CoverageStateDataset,
    indices: Iterable[int] | None,
    task: str,
    category_map: dict[str, int] | None = None,
) -> torch.Tensor:
    idxs = range(len(ds)) if indices is None else indices
    labels: list[int] = []
    for i in idxs:
        row = ds[i]
        if task == "level":
            labels.append(int(row["level"].item()))
        elif task == "category":
            if category_map is None:
                raise ValueError("category_map is required for category labels")
            labels.append(category_map[str(row["category"])])
        else:
            raise ValueError(f"unknown task: {task}")
    return torch.tensor(labels, dtype=torch.long)


def collect_feature_matrix(
    ds: CoverageStateDataset,
    indices: Iterable[int] | None,
    feature_kind: str,
) -> torch.Tensor:
    idxs = range(len(ds)) if indices is None else indices
    feats: list[np.ndarray] = []
    for i in idxs:
        row = ds[i]
        if feature_kind == "raw_union_stats":
            feat = row["raw_stats"].numpy().astype(np.float32)
        elif feature_kind == "input_stats":
            feat = raw_stats_features(row["points"].numpy())
        elif feature_kind == "raw_union_count":
            feat = np.asarray([float(row["raw_count"].item())], dtype=np.float32)
        elif feature_kind == "fixed_input_count":
            feat = np.asarray([float(row["points"].shape[0])], dtype=np.float32)
        elif feature_kind == "input_unique_voxel_count":
            feat = np.asarray([float(len(voxel_keys(row["points"].numpy())))], dtype=np.float32)
        elif feature_kind == "input_duplicate_rate":
            pts = row["points"].numpy()
            unique_exact = np.unique(np.round(pts, 6), axis=0).shape[0]
            feat = np.asarray([float(1.0 - unique_exact / max(len(pts), 1))], dtype=np.float32)
        else:
            raise ValueError(f"unknown feature kind: {feature_kind}")
        feats.append(feat)
    if not feats:
        return torch.empty(0, 1)
    return torch.tensor(np.stack(feats, axis=0), dtype=torch.float32)


class PointLabelDataset(Dataset):
    def __init__(
        self,
        base: CoverageStateDataset,
        indices: Iterable[int] | None,
        task: str,
        category_map: dict[str, int] | None = None,
    ) -> None:
        self.base = base
        self.indices = list(range(len(base))) if indices is None else list(indices)
        self.task = task
        self.category_map = category_map

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.base[self.indices[idx]]
        if self.task == "level":
            label = int(row["level"].item())
        elif self.task == "category":
            if self.category_map is None:
                raise ValueError("category_map is required for category labels")
            label = self.category_map[str(row["category"])]
        else:
            raise ValueError(f"unknown task: {self.task}")
        return row["points"].float(), torch.tensor(label, dtype=torch.long)


def materialize_point_dataset(ds: Dataset) -> TensorDataset:
    if isinstance(ds, TensorDataset):
        return ds
    points = []
    labels = []
    for i in range(len(ds)):
        p, y = ds[i]
        points.append(p.float())
        labels.append(y.long())
    if not points:
        return TensorDataset(torch.empty(0, 0, 3), torch.empty(0, dtype=torch.long))
    return TensorDataset(torch.stack(points, dim=0), torch.stack(labels, dim=0))


class FeatureMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointNetSmall(nn.Module):
    def __init__(self, num_classes: int, hidden: int = 96, feat_dim: int = 192) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, feat_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim * 2),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        h = self.point_mlp(points)
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        return self.head(pooled)


class SimpleEncoderClassifier(nn.Module):
    def __init__(self, num_classes: int, dim: int = 256) -> None:
        super().__init__()
        self.encoder = SimplePointEncoder(dim=dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(points))


@dataclass
class TrainResult:
    train_acc: float
    test_acc: float
    majority_test_acc: float


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.numel() == 0:
        return 0.0
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def majority_test_acc(train_y: torch.Tensor, test_y: torch.Tensor) -> float:
    if train_y.numel() == 0 or test_y.numel() == 0:
        return 0.0
    values, counts = torch.unique(train_y, return_counts=True)
    majority = values[counts.argmax()]
    return float((test_y == majority).float().mean().item())


def train_feature_mlp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    num_classes: int,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> TrainResult:
    set_seed(seed)
    mean = train_x.mean(0, keepdim=True)
    std = train_x.std(0, keepdim=True, unbiased=False).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    model = FeatureMLP(train_x.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        train_acc = accuracy_from_logits(model(train_x.to(device)).cpu(), train_y)
        test_acc = accuracy_from_logits(model(test_x.to(device)).cpu(), test_y)
    return TrainResult(train_acc, test_acc, majority_test_acc(train_y, test_y))


def train_point_classifier(
    train_ds: Dataset,
    test_ds: Dataset,
    model_name: str,
    num_classes: int,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
) -> TrainResult:
    set_seed(seed)
    if model_name == "pointnet_small":
        model = PointNetSmall(num_classes=num_classes)
    elif model_name == "simple_encoder":
        model = SimpleEncoderClassifier(num_classes=num_classes)
    else:
        raise ValueError(f"unknown point model: {model_name}")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_ds = materialize_point_dataset(train_ds)
    test_ds = materialize_point_dataset(test_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for _ in range(epochs):
        model.train()
        for points, labels in train_loader:
            points = points.to(device)
            labels = labels.to(device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(points), labels)
            loss.backward()
            opt.step()
    model.eval()

    def eval_loader(loader: DataLoader) -> tuple[float, torch.Tensor]:
        correct = 0
        total = 0
        labels_all = []
        with torch.no_grad():
            for points, labels in loader:
                logits = model(points.to(device)).cpu()
                correct += int((logits.argmax(1) == labels).sum().item())
                total += int(labels.numel())
                labels_all.append(labels)
        y = torch.cat(labels_all, dim=0) if labels_all else torch.empty(0, dtype=torch.long)
        return (float(correct / total) if total else 0.0), y

    train_acc, train_y = eval_loader(DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers))
    test_acc, test_y = eval_loader(test_loader)
    return TrainResult(train_acc, test_acc, majority_test_acc(train_y, test_y))


def per_class_counts(labels: torch.Tensor) -> dict[str, int]:
    values, counts = torch.unique(labels, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}
