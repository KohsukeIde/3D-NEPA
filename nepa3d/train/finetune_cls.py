import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..data.dataset import ModelNet40QueryDataset, collate
from ..data.modelnet40_index import (
    build_label_map,
    label_from_path,
    list_npz,
    stratified_train_val_split,
)
from ..models.query_nepa import QueryNepa
from ..utils.seed import set_seed
from ..utils.ckpt_utils import load_state_dict_flexible, maybe_resize_pos_emb_in_state_dict


def stratified_kshot(paths, k, seed=0):
    """Select up to K samples per class from a list of paths (.../<split>/<class>/<id>.npz)."""
    from collections import defaultdict
    from ..data.modelnet40_index import label_from_path

    paths = list(paths)
    if k <= 0:
        return sorted(paths)
    groups = defaultdict(list)
    for p in paths:
        groups[label_from_path(p)].append(p)

    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)

    out = []
    for cls, cls_paths in groups.items():
        cls_paths = sorted(cls_paths)
        rng.shuffle(cls_paths)
        out.extend(cls_paths[: min(len(cls_paths), int(k))])
    return sorted(out)


def stratified_nway(paths, n_way, seed=0):
    """Select N classes and keep all samples from those classes."""
    from collections import defaultdict

    paths = list(paths)
    if n_way <= 0:
        classes = sorted({label_from_path(p) for p in paths})
        return sorted(paths), classes

    groups = defaultdict(list)
    for p in paths:
        groups[label_from_path(p)].append(p)
    classes = sorted(groups.keys())
    if n_way > len(classes):
        raise ValueError(f"n_way={n_way} exceeds available classes={len(classes)}")

    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    picked_idx = rng.choice(len(classes), size=int(n_way), replace=False)
    picked = sorted(classes[i] for i in picked_idx.tolist())
    picked_set = set(picked)
    out = [p for p in paths if label_from_path(p) in picked_set]
    return sorted(out), picked


class ClsWrapper(nn.Module):
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.d_model, n_classes)

    def forward(self, feat, type_id):
        _, _, h = self.backbone(feat, type_id)
        pooled = h[:, -1, :]
        return self.head(pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=str, default="data/modelnet40_cache")
    ap.add_argument(
        "--backend",
        type=str,
        default="mesh",
        choices=["mesh", "pointcloud", "pointcloud_meshray", "pointcloud_noray", "voxel", "udfgrid"],
        help="shorthand: sets both --train_backend and --eval_backend unless they are explicitly provided",
    )
    ap.add_argument(
        "--train_backend",
        type=str,
        default=None,
        choices=["mesh", "pointcloud", "pointcloud_meshray", "pointcloud_noray", "voxel", "udfgrid"],
        help="backend for TRAIN split (fine-tuning / probe training)",
    )
    ap.add_argument(
        "--eval_backend",
        type=str,
        default=None,
        choices=["mesh", "pointcloud", "pointcloud_meshray", "pointcloud_noray", "voxel", "udfgrid"],
        help="backend for VAL/TEST split (evaluation). Use this for cross-backend probing.",
    )
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="", help="optional: save best/last checkpoints here")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--n_point", type=int, default=None)
    ap.add_argument("--n_ray", type=int, default=None)
    ap.add_argument(
        "--allow_scale_up",
        type=int,
        default=0,
        help=(
            "Allow n_point/n_ray to exceed the pretrain checkpoint settings (requires resizing pos_emb). "
            "Default keeps legacy behavior (cap to pretrain sizes)."
        ),
    )
    ap.add_argument(
        "--max_len",
        type=int,
        default=-1,
        help=(
            "Override model max_len (pos-emb length). If <0, auto = max(ckpt_max_len, required_seq_len). "
            "If set and differs from checkpoint, pos_emb is resized by 1D interpolation."
        ),
    )
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_ratio", type=float, default=0.1, help="stratified val split ratio from TRAIN")
    ap.add_argument("--val_seed", type=int, default=0, help="seed for stratified train/val split")
    ap.add_argument("--fewshot_k", type=int, default=0, help="K-shot fine-tuning: number of training samples per class (0=full train)")
    ap.add_argument("--fewshot_seed", type=int, default=0, help="seed for K-shot subset selection")
    ap.add_argument(
        "--fewshot_n_way",
        type=int,
        default=0,
        help="N-way episodic setting: keep only N classes for train/val/test (0=all classes)",
    )
    ap.add_argument(
        "--fewshot_way_seed",
        type=int,
        default=-1,
        help="seed for N-way class selection (-1: use --fewshot_seed)",
    )

    ap.add_argument("--eval_seed", type=int, default=0, help="deterministic eval seed (per-sample)")
    ap.add_argument("--mc_eval_k", type=int, default=1, help="MC eval: number of query resamples per test sample")
    ap.add_argument(
        "--mc_eval_k_val",
        type=int,
        default=-1,
        help="MC eval resamples for validation (-1: use --mc_eval_k)",
    )
    ap.add_argument(
        "--mc_eval_k_test",
        type=int,
        default=-1,
        help="MC eval resamples for final test (-1: use --mc_eval_k)",
    )
    ap.add_argument("--drop_ray_prob_train", type=float, default=0.0)
    ap.add_argument("--force_missing_ray", action="store_true")
    ap.add_argument("--add_eos", type=int, default=-1, help="1/0 to override, -1 to infer from checkpoint")
    ap.add_argument("--qa_tokens", type=int, default=-1, help="Use Q/A separated tokenization (0/1). -1: infer from checkpoint")
    ap.add_argument("--voxel_grid", type=int, default=64)
    ap.add_argument("--voxel_dilate", type=int, default=1)
    ap.add_argument("--voxel_max_steps", type=int, default=0)
    ap.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="linear probe mode: freeze backbone, train classifier head only",
    )
    ap.add_argument(
        "--ablate_point_dist",
        action="store_true",
        help="ablation: zero out point-distance channel before tokenization",
    )
    args = ap.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_backend = args.train_backend or args.backend
    eval_backend = args.eval_backend or args.backend
    mc_eval_k_val = args.mc_eval_k if args.mc_eval_k_val < 0 else int(args.mc_eval_k_val)
    mc_eval_k_test = args.mc_eval_k if args.mc_eval_k_test < 0 else int(args.mc_eval_k_test)

    train_paths_full = list_npz(args.cache_root, "train")
    test_paths = list_npz(args.cache_root, "test")
    way_seed = args.fewshot_seed if args.fewshot_way_seed < 0 else args.fewshot_way_seed
    picked_classes = None
    if args.fewshot_n_way and args.fewshot_n_way > 0:
        train_paths_full, picked_classes = stratified_nway(
            train_paths_full,
            n_way=args.fewshot_n_way,
            seed=way_seed,
        )
        picked_set = set(picked_classes)
        test_paths = sorted([p for p in test_paths if label_from_path(p) in picked_set])
        print(
            f"[fewshot-nway] n_way={args.fewshot_n_way} way_seed={way_seed} "
            f"picked_classes={picked_classes}"
        )
        print(
            f"[fewshot-nway] filtered_train={len(train_paths_full)} "
            f"filtered_test={len(test_paths)}"
        )

    train_paths, val_paths = stratified_train_val_split(
        train_paths_full, val_ratio=args.val_ratio, seed=args.val_seed
    )
    if args.fewshot_k and args.fewshot_k > 0:
        train_paths = stratified_kshot(train_paths, k=args.fewshot_k, seed=args.fewshot_seed)
        print(f"[fewshot] k={args.fewshot_k} selected_train={len(train_paths)} (from split-train)")

    label_map = build_label_map(train_paths_full)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    pre_args = ckpt["args"]
    ckpt_n_types = ckpt["model"]["type_emb.weight"].shape[0]
    if args.add_eos < 0:
        add_eos = bool(pre_args.get("add_eos", ckpt_n_types >= 5))
    else:
        add_eos = bool(args.add_eos)

    if args.qa_tokens >= 0:
        qa_tokens = bool(args.qa_tokens)
    else:
        qa_tokens = bool(pre_args.get("qa_tokens", ckpt_n_types >= 9))

    pre_n_point = int(pre_args["n_point"])
    pre_n_ray = int(pre_args["n_ray"])

    allow_scale_up = bool(int(args.allow_scale_up))

    if args.n_point is None:
        args.n_point = pre_n_point
    elif args.n_point > pre_n_point and (not allow_scale_up):
        print(
            f"[sizes] requested n_point={args.n_point} > pretrain n_point={pre_n_point}; capping (set --allow_scale_up 1 to override)"
        )
        args.n_point = pre_n_point
    elif args.n_point > pre_n_point and allow_scale_up:
        print(f"[sizes] scale-up enabled: n_point {pre_n_point} -> {args.n_point}")
    elif args.n_point <= 0:
        raise ValueError(f"--n_point must be positive, got {args.n_point}")

    if args.n_ray is None:
        args.n_ray = pre_n_ray
    elif args.n_ray > pre_n_ray and (not allow_scale_up):
        print(
            f"[sizes] requested n_ray={args.n_ray} > pretrain n_ray={pre_n_ray}; capping (set --allow_scale_up 1 to override)"
        )
        args.n_ray = pre_n_ray
    elif args.n_ray > pre_n_ray and allow_scale_up:
        print(f"[sizes] scale-up enabled: n_ray {pre_n_ray} -> {args.n_ray}")
    elif args.n_ray < 0:
        raise ValueError(f"--n_ray must be >= 0, got {args.n_ray}")

    train_ds = ModelNet40QueryDataset(
        train_paths,
        backend=train_backend,
        n_point=args.n_point,
        n_ray=args.n_ray,
        drop_ray_prob=args.drop_ray_prob_train,
        force_missing_ray=args.force_missing_ray,
        add_eos=add_eos,
        qa_tokens=qa_tokens,
        voxel_grid=args.voxel_grid,
        voxel_dilate=args.voxel_dilate,
        voxel_max_steps=args.voxel_max_steps,
        ablate_point_dist=args.ablate_point_dist,
        return_label=True,
        label_map=label_map,
    )

    val_ds = ModelNet40QueryDataset(
        val_paths,
        backend=eval_backend,
        n_point=args.n_point,
        n_ray=args.n_ray,
        mode="eval",
        eval_seed=args.eval_seed,
        mc_eval_k=mc_eval_k_val,
        drop_ray_prob=0.0,
        force_missing_ray=args.force_missing_ray,
        add_eos=add_eos,
        qa_tokens=qa_tokens,
        voxel_grid=args.voxel_grid,
        voxel_dilate=args.voxel_dilate,
        voxel_max_steps=args.voxel_max_steps,
        ablate_point_dist=args.ablate_point_dist,
        return_label=True,
        label_map=label_map,
    )

    test_ds = ModelNet40QueryDataset(
        test_paths,
        backend=eval_backend,
        n_point=args.n_point,
        n_ray=args.n_ray,
        mode="eval",
        eval_seed=args.eval_seed,
        mc_eval_k=mc_eval_k_test,
        drop_ray_prob=0.0,
        force_missing_ray=args.force_missing_ray,
        add_eos=add_eos,
        qa_tokens=qa_tokens,
        voxel_grid=args.voxel_grid,
        voxel_dilate=args.voxel_dilate,
        voxel_max_steps=args.voxel_max_steps,
        ablate_point_dist=args.ablate_point_dist,
        return_label=True,
        label_map=label_map,
    )

    def _worker_init_fn(worker_id: int):
        # Ensure numpy RNG differs across dataloader workers.
        import random

        base = (torch.initial_seed() + worker_id) % (2**32)
        np.random.seed(base)
        random.seed(base)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        worker_init_fn=_worker_init_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        worker_init_fn=_worker_init_fn,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        worker_init_fn=_worker_init_fn,
    )

    # Allow scale-up by resizing learned positional embeddings.
    if "pos_emb" not in ckpt["model"]:
        raise KeyError("checkpoint missing pos_emb; cannot infer/resize max_len")

    ckpt_max_len = int(ckpt["model"]["pos_emb"].shape[1])
    required_len = (
        1 + 2 * int(args.n_point) + 2 * int(args.n_ray) + (1 if add_eos else 0)
        if qa_tokens
        else 1 + int(args.n_point) + int(args.n_ray) + (1 if add_eos else 0)
    )

    if int(args.max_len) >= 0:
        model_max_len = int(args.max_len)
        if model_max_len < required_len:
            raise ValueError(
                f"--max_len too small: max_len={model_max_len} < required_seq_len={required_len} "
                f"(qa_tokens={qa_tokens}, add_eos={add_eos}, n_point={args.n_point}, n_ray={args.n_ray})."
            )
    else:
        # Auto: keep ckpt length unless we need to scale up.
        model_max_len = max(ckpt_max_len, required_len)

    state = ckpt["model"]
    if model_max_len != ckpt_max_len:
        print(f"[ckpt] resizing pos_emb: ckpt_len={ckpt_max_len} -> max_len={model_max_len}")
        state = maybe_resize_pos_emb_in_state_dict(dict(state), model_max_len)

    backbone = QueryNepa(
        feat_dim=15,
        d_model=pre_args["d_model"],
        n_types=ckpt_n_types,
        nhead=pre_args["heads"],
        num_layers=pre_args["layers"],
        max_len=model_max_len,
    )
    load_state_dict_flexible(backbone, state, strict=True)
    backbone.to(device)

    if args.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()

    model = ClsWrapper(backbone, n_classes=len(label_map)).to(device)
    opt_params = model.head.parameters() if args.freeze_backbone else model.parameters()
    opt = optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    def eval_acc(dl):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dl:
                feat = batch["feat"].to(device).float()
                type_id = batch["type_id"].to(device).long()
                y = batch["label"].to(device).long()

                # MC eval: if feat is (B,K,T,F), average logits over K resamples
                if feat.dim() == 4:
                    B, K, T, F = feat.shape
                    feat_ = feat.view(B * K, T, F)
                    type_ = type_id.view(B * K, T)
                    logits_ = model(feat_, type_)
                    logits = logits_.view(B, K, -1).mean(dim=1)
                else:
                    logits = model(feat, type_id)
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.numel()
        return correct / max(total, 1)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    best_val = -1.0
    best_ep = -1
    best_state = None

    print(
        f"train_backend={train_backend} eval_backend={eval_backend} "
        f"val_ratio={args.val_ratio} val_seed={args.val_seed} "
        f"mc_eval_k_val={mc_eval_k_val} mc_eval_k_test={mc_eval_k_test} "
        f"freeze_backbone={bool(args.freeze_backbone)} "
        f"ablate_point_dist={bool(args.ablate_point_dist)} "
        f"n_point={args.n_point}/{pre_n_point} n_ray={args.n_ray}/{pre_n_ray} "
        f"model_max_len={model_max_len}"
    )
    print(f"num_train={len(train_paths)} num_val={len(val_paths)} num_test={len(test_paths)}")

    for ep in range(args.epochs):
        model.train()
        if args.freeze_backbone:
            # keep frozen backbone deterministic even in train loop
            model.backbone.eval()

        for batch in train_dl:
            feat = batch["feat"].to(device).float()
            type_id = batch["type_id"].to(device).long()
            y = batch["label"].to(device).long()

            opt.zero_grad(set_to_none=True)
            logits = model(feat, type_id)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

        val_acc = eval_acc(val_dl)
        improved = val_acc > best_val + 1e-12
        if improved:
            best_val = val_acc
            best_ep = ep
            # Keep best weights.
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if args.save_dir:
                torch.save(
                    {
                        "model": best_state,
                        "epoch": best_ep,
                        "val_acc": best_val,
                        "args": vars(args),
                        "pretrain_ckpt": args.ckpt,
                    },
                    os.path.join(args.save_dir, "best.pt"),
                )

        print(f"ep={ep} val_acc={val_acc:.4f} best_val={best_val:.4f} best_ep={best_ep}")

    # Final test evaluation on the best VAL checkpoint.
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    test_acc = eval_acc(test_dl)
    print(f"best_val={best_val:.4f} best_ep={best_ep} test_acc={test_acc:.4f}")

    if args.save_dir:
        torch.save(
            {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": args.epochs - 1,
                "val_acc": best_val,
                "test_acc": test_acc,
                "args": vars(args),
                "pretrain_ckpt": args.ckpt,
            },
            os.path.join(args.save_dir, "last.pt"),
        )


if __name__ == "__main__":
    main()
