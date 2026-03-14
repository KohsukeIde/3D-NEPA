from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from nepa3d.data.cqa_codec import ANSWER_VOCAB_SIZE, CQA_VOCAB_VERSION, QUERY_TYPE_VOCAB_SIZE
from nepa3d.data.dataset_cqa import cqa_collate_fn
from nepa3d.data.mixed_pretrain_cqa import build_mixed_pretrain_cqa
from nepa3d.models.primitive_answering import PrimitiveAnsweringModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("pretrain_primitive_answering")
    p.add_argument("--mix_config_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="runs/cqa")
    p.add_argument("--run_name", type=str, default="cqa_debug")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--n_ctx", type=int, default=2048)
    p.add_argument("--n_qry", type=int, default=64)

    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--drop_path", type=float, default=0.0)
    p.add_argument("--backbone_impl", type=str, default="nepa2d", choices=["nepa2d", "pointmae", "legacy"])
    p.add_argument("--num_groups", type=int, default=64)
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--patch_center_mode", type=str, default="fps", choices=["fps", "first"])
    p.add_argument("--patch_fps_random_start", type=int, default=1, choices=[0, 1])
    p.add_argument("--local_encoder", type=str, default="pointmae_conv", choices=["mlp", "pointmae_conv"])
    p.add_argument("--answer_vocab", type=int, default=ANSWER_VOCAB_SIZE)
    p.add_argument("--query_type_vocab", type=int, default=QUERY_TYPE_VOCAB_SIZE)
    p.add_argument("--generator_depth", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ddp = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp])
    set_seed(int(args.seed))

    dataset, sampler, info = build_mixed_pretrain_cqa(
        args.mix_config_path,
        n_ctx=int(args.n_ctx),
        n_qry=int(args.n_qry),
        mode="train",
        eval_seed=int(args.seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=cqa_collate_fn,
        drop_last=True,
    )

    model = PrimitiveAnsweringModel(
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        drop_path=float(args.drop_path),
        backbone_impl=str(args.backbone_impl),
        num_groups=int(args.num_groups),
        group_size=int(args.group_size),
        patch_center_mode=str(args.patch_center_mode),
        patch_fps_random_start=bool(args.patch_fps_random_start),
        local_encoder=str(args.local_encoder),
        query_type_vocab=int(args.query_type_vocab),
        answer_vocab=int(args.answer_vocab),
        generator_depth=int(args.generator_depth),
    )
    opt = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    model, opt, loader = accelerator.prepare(model, opt, loader)

    save_dir = Path(args.save_dir) / args.run_name
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        with open(save_dir / "mix_info.json", "w") as f:
            json.dump(info, f, indent=2)
        with open(save_dir / "vocab_spec.json", "w") as f:
            json.dump(
                {
                    "vocab_version": CQA_VOCAB_VERSION,
                    "answer_vocab_size": int(args.answer_vocab),
                    "query_type_vocab": int(args.query_type_vocab),
                },
                f,
                indent=2,
            )

    global_step = 0
    for ep in range(int(args.epochs)):
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(ep)
        model.train()
        pbar = tqdm(loader, disable=not accelerator.is_main_process, desc=f"ep {ep:03d}")
        for batch in pbar:
            out = model(
                ctx_xyz=batch["ctx_xyz"],
                qry_xyz=batch["qry_xyz"],
                qry_type=batch["qry_type"],
                answer_code=batch["answer_code"],
            )
            logits = out.logits.reshape(-1, out.logits.shape[-1])
            target = batch["answer_code"].reshape(-1)
            loss = F.cross_entropy(logits, target)
            accelerator.backward(loss)
            opt.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1
            if accelerator.is_main_process:
                pbar.set_postfix(loss=float(loss.detach().cpu()))

        if accelerator.is_main_process and ((ep + 1) % int(args.save_every) == 0 or (ep + 1) == int(args.epochs)):
            ckpt = {
                "model": accelerator.unwrap_model(model).state_dict(),
                "args": vars(args),
                "epoch": int(ep + 1),
                "global_step": int(global_step),
                "vocab_version": CQA_VOCAB_VERSION,
            }
            torch.save(ckpt, save_dir / f"ckpt_ep{ep+1:04d}.pt")
            torch.save(ckpt, save_dir / "ckpt_latest.pt")


if __name__ == "__main__":
    main()
