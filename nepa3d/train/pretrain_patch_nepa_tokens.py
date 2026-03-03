"""Token-stream PatchNEPA pretraining on v2 surface/query datasets.

Sequence format:
  [BOS, ctx_patch_1..P, SEP, qry_Q1, qry_A1, ..., qry_QN, qry_AN, EOS]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterator, Tuple

import torch
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from nepa3d.data.dataset_v2 import v2_collate_fn
from nepa3d.data.mixed_pretrain import build_mixed_pretrain
from nepa3d.models.patch_nepa import PatchTransformerNepa
from nepa3d.token.tokenizer import TYPE_A_POINT, TYPE_BOS, TYPE_EOS, TYPE_POINT, TYPE_Q_POINT, TYPE_SEP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("pretrain_patch_nepa_tokens")
    p.add_argument("--mix_config_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="runs_patch_nepa_tokens")
    p.add_argument("--run_name", type=str, default="debug_tokens")
    p.add_argument("--save_every", type=int, default=1000)

    p.add_argument("--n_surf", type=int, default=2048)
    p.add_argument("--n_qry", type=int, default=1024)
    p.add_argument("--n_ray", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--pm_pc_norm", type=int, default=1, choices=[0, 1], help="Apply Point-MAE pc_norm on v2 samples.")
    p.add_argument(
        "--pm_scale_translate",
        type=int,
        default=1,
        choices=[0, 1],
        help="Apply Point-MAE scale+translate augmentation on v2 train samples.",
    )
    p.add_argument("--pm_scale_low", type=float, default=(2.0 / 3.0))
    p.add_argument("--pm_scale_high", type=float, default=(3.0 / 2.0))
    p.add_argument("--pm_translate", type=float, default=0.2)
    p.add_argument(
        "--pm_transform_answers",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, transform ans_feat consistently under pm_pc_norm/pm_scale_translate.",
    )

    p.add_argument("--patch_embed", type=str, default="fps_knn", choices=["serial", "pointgpt", "fps_knn"])
    p.add_argument("--patch_local_encoder", type=str, default="mlp", choices=["mlp", "pointmae_conv"])
    p.add_argument("--patch_fps_random_start", type=int, default=0, choices=[0, 1])
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--num_groups", type=int, default=64)
    p.add_argument("--serial_order", type=str, default="morton")
    p.add_argument("--serial_bits", type=int, default=10)
    p.add_argument("--serial_shuffle_within_patch", type=int, default=0, choices=[0, 1])
    p.add_argument("--patch_order_mode", type=str, default="none")

    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--drop_path_rate", type=float, default=0.0)
    p.add_argument("--qk_norm", type=int, default=1)
    p.add_argument("--qk_norm_affine", type=int, default=0)
    p.add_argument("--qk_norm_bias", type=int, default=0)
    p.add_argument("--layerscale_value", type=float, default=1e-5)
    p.add_argument("--rope_theta", type=float, default=100.0)
    p.add_argument("--use_gated_mlp", type=int, default=0)
    p.add_argument("--hidden_act", type=str, default="gelu")
    p.add_argument("--backbone_mode", type=str, default="nepa2d", choices=["nepa2d", "vanilla", "pointmae"])

    p.add_argument("--answer_in_dim", type=int, default=0, help="0 => infer from first sample ans_feat dim")
    p.add_argument("--answer_mlp_layers", type=int, default=2)
    p.add_argument("--answer_pool", type=str, default="max", choices=["max", "mean"])
    p.add_argument("--loss_target_mode", type=str, default="content_tokens", choices=["full_z", "content_tokens"])
    p.add_argument("--skip_k", type=int, default=1)

    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=0.0)
    p.add_argument("--q_mask_prob", type=float, default=0.0)
    p.add_argument("--q_mask_mode", type=str, default="mask_token", choices=["mask_token", "zero"])
    p.add_argument("--dual_mask_near", type=float, default=0.0)
    p.add_argument("--dual_mask_far", type=float, default=0.0)
    p.add_argument("--dual_mask_window", type=int, default=0)
    p.add_argument("--dual_mask_type_aware", type=int, default=0, choices=[0, 1])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_qa_sequence(
    model: PatchTransformerNepa,
    surf_xyz: torch.Tensor,
    qry_xyz: torch.Tensor,
    ans_feat: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build [tokens, type_id, centers_xyz] for forward_tokens()."""
    b = surf_xyz.shape[0]
    dev = surf_xyz.device

    ctx_tok, ctx_centers, _ = model.encode_patches(surf_xyz)
    q_tok, q_centers = model.encode_point_queries(qry_xyz)
    a_tok, a_centers = model.encode_point_answers(ans_feat, qry_xyz)

    p = int(ctx_tok.shape[1])
    n = int(q_tok.shape[1])
    qa_tok = torch.stack([q_tok, a_tok], dim=2).reshape(b, 2 * n, -1)
    qa_centers = torch.stack([q_centers, a_centers], dim=2).reshape(b, 2 * n, 3)

    z0 = torch.zeros((b, 1, 3), device=dev, dtype=surf_xyz.dtype)
    tokens = torch.cat(
        [
            model.bos_token.expand(b, 1, -1),
            ctx_tok,
            model.sep_token.expand(b, 1, -1),
            qa_tok,
            model.eos_token.expand(b, 1, -1),
        ],
        dim=1,
    )
    centers = torch.cat([z0, ctx_centers, z0, qa_centers, z0], dim=1)

    type_id = torch.empty((b, 1 + p + 1 + 2 * n + 1), device=dev, dtype=torch.long)
    type_id[:, 0] = int(TYPE_BOS)
    type_id[:, 1 : 1 + p] = int(TYPE_POINT)
    type_id[:, 1 + p] = int(TYPE_SEP)
    type_id[:, 1 + p + 1 : -1 : 2] = int(TYPE_Q_POINT)
    type_id[:, 1 + p + 2 : -1 : 2] = int(TYPE_A_POINT)
    type_id[:, -1] = int(TYPE_EOS)
    return tokens, type_id, centers


def _infinite_loader(dl: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        for batch in dl:
            yield batch


def _save_ckpt(path: Path, model: PatchTransformerNepa, optimizer: optim.Optimizer, step: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "args": vars(args),
        },
        str(path),
    )


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=int(args.grad_accum))
    set_seed(int(args.seed))

    save_root = Path(args.save_dir) / args.run_name
    if accelerator.is_main_process:
        save_root.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    dataset, sampler, info = build_mixed_pretrain(
        mix_config_path=str(args.mix_config_path),
        n_point=int(args.n_surf),
        n_ray=int(args.n_ray),
        num_workers=int(args.num_workers),
        mode="train",
        eval_seed=int(args.seed),
        return_raw=True,
        v2_n_qry=(None if int(args.n_qry) <= 0 else int(args.n_qry)),
        v2_pointmae_pc_norm=bool(int(args.pm_pc_norm)),
        v2_pointmae_scale_translate=bool(int(args.pm_scale_translate)),
        v2_pointmae_scale_low=float(args.pm_scale_low),
        v2_pointmae_scale_high=float(args.pm_scale_high),
        v2_pointmae_translate=float(args.pm_translate),
        v2_transform_answers=bool(int(args.pm_transform_answers)),
    )
    probe = dataset[0]
    if probe.get("ans_feat", None) is None:
        raise RuntimeError("v2 token pretrain requires ans_feat in dataset items.")
    answer_in_dim = int(args.answer_in_dim) if int(args.answer_in_dim) > 0 else int(probe["ans_feat"].shape[-1])
    if answer_in_dim <= 0:
        raise RuntimeError(f"invalid answer_in_dim={answer_in_dim}")

    dl = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        collate_fn=v2_collate_fn,
    )

    model = PatchTransformerNepa(
        patch_embed=str(args.patch_embed),
        patch_local_encoder=str(args.patch_local_encoder),
        patch_fps_random_start=bool(int(args.patch_fps_random_start)),
        n_point=int(args.n_surf),
        group_size=int(args.group_size),
        num_groups=int(args.num_groups),
        serial_order=str(args.serial_order),
        serial_bits=int(args.serial_bits),
        serial_shuffle_within_patch=int(args.serial_shuffle_within_patch),
        patch_order_mode=str(args.patch_order_mode),
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        drop_path_rate=float(args.drop_path_rate),
        qk_norm=int(args.qk_norm),
        qk_norm_affine=int(args.qk_norm_affine),
        qk_norm_bias=int(args.qk_norm_bias),
        layerscale_value=float(args.layerscale_value),
        rope_theta=float(args.rope_theta),
        use_gated_mlp=int(args.use_gated_mlp),
        hidden_act=str(args.hidden_act),
        backbone_mode=str(args.backbone_mode),
        qa_tokens=1,
        qa_layout="split",
        qa_sep_token=True,
        qa_fuse="add",
        use_pt_dist=False,
        use_pt_grad=False,
        answer_in_dim=int(answer_in_dim),
        answer_mlp_layers=int(args.answer_mlp_layers),
        answer_pool=str(args.answer_pool),
        q_mask_mode=str(args.q_mask_mode),
        use_ray_patch=False,
    )
    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
    model_raw = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        accelerator.print(f"[token_pretrain] answer_in_dim={answer_in_dim}")
        accelerator.print(f"[token_pretrain] mix_info={info}")
        accelerator.print(
            "[token_pretrain] pointmae_compat "
            f"pc_norm={bool(int(args.pm_pc_norm))} "
            f"scale_translate={bool(int(args.pm_scale_translate))} "
            f"scale=[{float(args.pm_scale_low):.4f},{float(args.pm_scale_high):.4f}] "
            f"translate={float(args.pm_translate):.4f} "
            f"transform_answers={bool(int(args.pm_transform_answers))}"
        )

    data_iter = _infinite_loader(dl)
    model.train()
    step = 0
    pbar = tqdm(total=int(args.max_steps), disable=not accelerator.is_main_process)
    while step < int(args.max_steps):
        batch = next(data_iter)
        surf_xyz = batch["surf_xyz"]
        qry_xyz = batch["qry_xyz"]
        ans_feat = batch["ans_feat"]
        if qry_xyz is None or ans_feat is None:
            raise RuntimeError("batch missing qry_xyz/ans_feat; ensure v2 config uses return_qry=true.")
        surf_xyz = surf_xyz.to(accelerator.device, non_blocking=True)
        qry_xyz = qry_xyz.to(accelerator.device, non_blocking=True)
        ans_feat = ans_feat.to(accelerator.device, non_blocking=True)

        with accelerator.accumulate(model):
            tokens, type_id, centers = build_qa_sequence(model, surf_xyz, qry_xyz, ans_feat)
            out = model.forward_tokens(
                tokens=tokens,
                type_id=type_id,
                centers_xyz=centers,
                is_causal=True,
                q_mask_prob=float(args.q_mask_prob),
                dual_mask_near=float(args.dual_mask_near),
                dual_mask_far=float(args.dual_mask_far),
                dual_mask_window=int(args.dual_mask_window),
                dual_mask_type_aware=bool(int(args.dual_mask_type_aware)),
            )
            target = out.z if str(args.loss_target_mode) == "full_z" else out.tokens
            loss = model_raw.nepa_loss(
                out.z,
                out.z_hat,
                out.type_id,
                skip_k=int(args.skip_k),
                target=target,
            )
            accelerator.backward(loss)
            if float(args.max_grad_norm) > 0:
                accelerator.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            step += 1
            if accelerator.is_main_process:
                pbar.update(1)
                pbar.set_description(f"loss={loss.item():.4f}")
            if int(args.save_every) > 0 and (step % int(args.save_every) == 0):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    _save_ckpt(save_root / f"ckpt_step{step}.pt", model_raw, optimizer, step, args)
    pbar.close()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_ckpt(save_root / "ckpt_final.pt", model_raw, optimizer, step, args)
        accelerator.print(f"[done] saved to {save_root}")


if __name__ == "__main__":
    main()
