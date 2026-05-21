#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from coverage_nepa.eval.common import load_ckpt
from coverage_nepa.models.simple_point_encoder import SimplePointEncoder

SCAN_FILES = {
    "obj_bg": ("h5_files/main_split/training_objectdataset.h5", "h5_files/main_split/test_objectdataset.h5"),
    "obj_only": ("h5_files/main_split_nobg/training_objectdataset.h5", "h5_files/main_split_nobg/test_objectdataset.h5"),
    "pb_t50_rs": ("h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5", "h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5"),
}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_h5(path: Path):
    with h5py.File(path, "r") as f:
        return np.asarray(f["data"], dtype=np.float32), np.asarray(f["label"], dtype=np.int64).reshape(-1)


def norm_pc(x: torch.Tensor) -> torch.Tensor:
    x = x[:, :, :3]
    x = x - x.mean(dim=1, keepdim=True)
    s = torch.norm(x, dim=-1).amax(dim=1, keepdim=True).clamp_min(1e-6).unsqueeze(-1)
    return x / s


def sample_points_np(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n <= 0 or points.shape[1] == n:
        return points[:, :, :3]
    out = []
    for i, p in enumerate(points):
        rng = np.random.default_rng(seed + i * 1_000_003)
        replace = p.shape[0] < n
        idx = rng.choice(p.shape[0], size=n, replace=replace)
        out.append(p[idx, :3])
    return np.stack(out).astype(np.float32)


def stratified_subset(points, labels, max_n, seed):
    if max_n <= 0 or max_n >= len(labels): return points, labels
    rng = np.random.default_rng(seed)
    chosen = []
    classes = np.unique(labels)
    per = max_n // len(classes)
    for c in classes:
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        chosen.extend(idx[:per].tolist())
    if len(chosen) < max_n:
        rest = [i for i in range(len(labels)) if i not in set(chosen)]
        rng.shuffle(rest); chosen.extend(rest[:max_n-len(chosen)])
    chosen = np.asarray(chosen[:max_n], dtype=np.int64); rng.shuffle(chosen)
    return points[chosen], labels[chosen]


def extract(model, points, labels, npoints, batch_size, device, seed, feature_source):
    pts = sample_points_np(points, npoints, seed)
    ds = TensorDataset(torch.from_numpy(pts).float(), torch.from_numpy(labels).long())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    feats, labs = [], []
    with torch.no_grad():
        for x, y in loader:
            x = norm_pc(x).to(device)
            if hasattr(model, "encode_target") and feature_source == "target":
                z = model.encode_target(x)
            elif hasattr(model, "encode_online"):
                z = model.encode_online(x)
            else:
                z = model(x)
            feats.append(z.cpu()); labs.append(y)
    return torch.cat(feats), torch.cat(labs)


def train_probe(train_x, train_y, test_x, test_y, epochs, lr, wd, seed, device):
    set_seed(seed)
    mean = train_x.mean(0, keepdim=True); std = train_x.std(0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean) / std; test_x = (test_x - mean) / std
    nc = int(max(train_y.max().item(), test_y.max().item()) + 1)
    head = nn.Linear(train_x.shape[1], nc).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=2048, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(xb), yb)
            loss.backward(); opt.step()
    with torch.no_grad():
        train_acc = float((head(train_x.to(device)).argmax(1).cpu() == train_y).float().mean())
        test_acc = float((head(test_x.to(device)).argmax(1).cpu() == test_y).float().mean())
    return train_acc, test_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--random-encoder", action="store_true")
    ap.add_argument("--data-root", default="/groups/gag51402/datasets/scanobjectnn")
    ap.add_argument("--split", default="pb_t50_rs", choices=sorted(SCAN_FILES))
    ap.add_argument("--npoints", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--probe-epochs", type=int, default=100)
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    ap.add_argument("--probe-weight-decay", type=float, default=0.01)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--feature-source", default="target", choices=["online", "target"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.random_encoder:
        model = SimplePointEncoder(dim=256).to(device).eval()
        model.encode_online = model.forward  # type: ignore
    else:
        model = load_ckpt(args.ckpt, device)
    tr_rel, te_rel = SCAN_FILES[args.split]
    tr_x, tr_y = load_h5(Path(args.data_root) / tr_rel)
    te_x, te_y = load_h5(Path(args.data_root) / te_rel)
    tr_x, tr_y = stratified_subset(tr_x, tr_y, args.max_train, args.seed)
    te_x, te_y = stratified_subset(te_x, te_y, args.max_test, args.seed + 999)
    ftr, ytr = extract(model, tr_x, tr_y, args.npoints, args.batch_size, device, args.seed, args.feature_source)
    fte, yte = extract(model, te_x, te_y, args.npoints, args.batch_size, device, args.seed + 99_991, args.feature_source)
    train_acc, test_acc = train_probe(ftr, ytr, fte, yte, args.probe_epochs, args.probe_lr, args.probe_weight_decay, args.seed, device)
    out = {
        "split": args.split,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "feature_dim": int(ftr.shape[1]),
        "train_n": int(len(ytr)),
        "test_n": int(len(yte)),
        "npoints": int(args.npoints),
        "feature_source": args.feature_source,
        "ckpt": args.ckpt or "random",
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(out)

if __name__ == "__main__": main()
