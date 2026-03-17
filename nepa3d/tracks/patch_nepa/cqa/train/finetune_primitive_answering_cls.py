from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from nepa3d.data.cls_patch_dataset import PatchClsPointDataset, PointAugConfig
from nepa3d.tracks.patch_nepa.cqa.data.cqa_codec import ANSWER_VOCAB_SIZE, QUERY_TYPE_VOCAB_SIZE
from nepa3d.data.modelnet40_index import list_npz
from nepa3d.tracks.patch_nepa.cqa.models.primitive_answering import PrimitiveAnsweringClassifier, PrimitiveAnsweringModel


def _build_label_map(paths: List[str]) -> Dict[str, int]:
    classes = sorted({Path(p).parent.name for p in paths})
    return {c: i for i, c in enumerate(classes)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("finetune_primitive_answering_cls")
    p.add_argument("--cache_root", type=str, required=True)
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--val_split", type=str, default="test")
    p.add_argument("--val_split_mode", type=str, default="pointmae", choices=["pointmae", "explicit"])
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="runs/cqa_cls")
    p.add_argument("--run_name", type=str, default="ft_debug")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--n_point", type=int, default=2048)
    p.add_argument("--sample_mode_train", type=str, default="random", choices=["random", "fps", "fps_then_sample"])
    p.add_argument("--sample_mode_eval", type=str, default="random", choices=["random", "fps", "fps_then_sample"])
    p.add_argument("--pool", type=str, default="mean", choices=["mean", "bos"])
    p.add_argument("--pointmae_aug", type=int, default=1, choices=[0, 1])
    return p.parse_args()


def _load_pretrained_model(ckpt_path: str) -> PrimitiveAnsweringModel:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    model = PrimitiveAnsweringModel(
        d_model=int(args.get("d_model", 384)),
        n_layers=int(args.get("n_layers", 12)),
        n_heads=int(args.get("n_heads", 6)),
        mlp_ratio=float(args.get("mlp_ratio", 4.0)),
        dropout=float(args.get("dropout", 0.0)),
        drop_path=float(args.get("drop_path", 0.0)),
        backbone_impl=str(args.get("backbone_impl", "nepa2d")),
        num_groups=int(args.get("num_groups", 64)),
        group_size=int(args.get("group_size", 32)),
        patch_center_mode=str(args.get("patch_center_mode", "fps")),
        patch_fps_random_start=bool(args.get("patch_fps_random_start", 1)),
        local_encoder=str(args.get("local_encoder", "pointmae_conv")),
        query_type_vocab=int(args.get("query_type_vocab", QUERY_TYPE_VOCAB_SIZE)),
        answer_vocab=int(args.get("answer_vocab", ANSWER_VOCAB_SIZE)),
        generator_depth=int(args.get("generator_depth", 2)),
    )
    model.load_state_dict(ckpt["model"], strict=False)
    return model


def _evaluate(model, loader, accelerator):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["xyz"])
            pred = logits.argmax(dim=-1)
            correct += int((pred == batch["label"]).sum().item())
            total += int(batch["label"].numel())
    t = torch.tensor([correct, total], device=accelerator.device, dtype=torch.long)
    if accelerator.num_processes > 1:
        t = accelerator.reduce(t, reduction="sum")
    return float(t[0].item()) / max(int(t[1].item()), 1)


def main() -> None:
    args = parse_args()
    ddp = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp])
    set_seed(int(args.seed))

    cache_root_abs_l = os.path.abspath(str(args.cache_root)).lower()
    resolved_train_split = str(args.train_split)
    resolved_val_split = str(args.val_split)
    resolved_val_split_mode = "explicit"
    if str(args.val_split_mode) == "pointmae":
        if "scanobjectnn" in cache_root_abs_l:
            resolved_train_split = "train"
            resolved_val_split = "test"
            resolved_val_split_mode = "pointmae(test-as-val)"
        else:
            resolved_val_split_mode = "pointmae(fallback-explicit)"

    train_paths = list_npz(args.cache_root, resolved_train_split)
    val_paths = list_npz(args.cache_root, resolved_val_split)
    label_map = _build_label_map(train_paths + val_paths)

    aug_cfg = PointAugConfig(
        prob=1.0,
        scale_min=(2.0 / 3.0),
        scale_max=(3.0 / 2.0),
        shift_std=0.2,
        pointmae_exact=bool(args.pointmae_aug),
    )
    train_set = PatchClsPointDataset(
        train_paths,
        cache_root=args.cache_root,
        label_map=label_map,
        n_point=int(args.n_point),
        sample_mode=str(args.sample_mode_train),
        use_normals=False,
        aug=True,
        aug_cfg=aug_cfg,
        rng_seed=int(args.seed),
    )
    val_set = PatchClsPointDataset(
        val_paths,
        cache_root=args.cache_root,
        label_map=label_map,
        n_point=int(args.n_point),
        sample_mode=str(args.sample_mode_eval),
        use_normals=False,
        aug=False,
        rng_seed=int(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers), pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=True)

    pretrained = _load_pretrained_model(args.ckpt)
    model = PrimitiveAnsweringClassifier(pretrained, n_cls=len(label_map), pool=str(args.pool))
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    model, opt, train_loader, val_loader = accelerator.prepare(model, opt, train_loader, val_loader)

    save_dir = Path(args.save_dir) / args.run_name
    save_args = dict(vars(args))
    save_args["resolved_train_split"] = resolved_train_split
    save_args["resolved_val_split"] = resolved_val_split
    save_args["resolved_val_split_mode"] = resolved_val_split_mode
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "args.json", "w") as f:
            json.dump(save_args, f, indent=2)
        print(
            f"[protocol] cache_root={args.cache_root} "
            f"train_split={resolved_train_split} val_split={resolved_val_split} "
            f"val_split_mode={resolved_val_split_mode}"
        )

    best = -1.0
    for ep in range(int(args.epochs)):
        model.train()
        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"ep {ep:03d}")
        for batch in pbar:
            logits = model(batch["xyz"])
            loss = F.cross_entropy(logits, batch["label"])
            accelerator.backward(loss)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if accelerator.is_main_process:
                pbar.set_postfix(loss=float(loss.detach().cpu()))
        acc = _evaluate(model, val_loader, accelerator)
        if accelerator.is_main_process:
            print(f"[val] ep={ep:03d} acc={acc:.4f}")
            if acc > best:
                best = acc
                ckpt = {"model": accelerator.unwrap_model(model).state_dict(), "args": save_args, "best_val_acc": best}
                torch.save(ckpt, save_dir / "ckpt_best.pt")
            torch.save({"model": accelerator.unwrap_model(model).state_dict(), "args": save_args, "best_val_acc": best}, save_dir / "ckpt_latest.pt")


if __name__ == "__main__":
    main()
