#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import builder
from utils import misc
from utils.config import cfg_from_yaml_file


SCANOBJECTNN_CLASS_NAMES = [
    "bag",
    "bin",
    "box",
    "cabinet",
    "chair",
    "desk",
    "display",
    "door",
    "shelf",
    "table",
    "bed",
    "pillow",
    "sink",
    "sofa",
    "toilet",
]


def parse_args():
    p = argparse.ArgumentParser("ScanObjectNN readout audit for PointGPT family")
    p.add_argument("--config", required=True, help="finetune config path")
    p.add_argument("--ckpt", required=True, help="fine-tuned checkpoint path")
    p.add_argument("--train_split", default="train", choices=["train", "val", "test"])
    p.add_argument("--test_split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_test_batches", type=int, default=0)
    p.add_argument("--pair", type=str, default="", help="optional class ids a,b")
    p.add_argument("--output_json", type=str, default="")
    p.add_argument("--output_md", type=str, default="")
    return p.parse_args()


def load_config(config_path, batch_size, split):
    cfg = cfg_from_yaml_file(config_path)
    cfg.dataset.test.others.bs = batch_size
    cfg.dataset.test.others.subset = split
    return cfg


def build_loader(cfg, split, num_workers):
    cfg_split = cfg.dataset.test
    cfg_split.others.subset = split
    args = SimpleNamespace(distributed=False, num_workers=num_workers)
    _, loader = builder.dataset_builder(args, cfg_split)
    return loader


def build_model(cfg, ckpt):
    model = builder.model_builder(cfg.model)
    builder.load_model(model, ckpt, logger=None)
    model = model.cuda()
    model.eval()
    return model


def extract_split(model, loader, npoints, max_batches=0):
    feats = []
    logits = []
    labels = []
    with torch.no_grad():
        for batch_idx, (_, _, data) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            points = data[0].cuda(non_blocking=True)
            label = data[1].cuda(non_blocking=True).view(-1)
            points = misc.fps(points, npoints)
            batch_logits, _, batch_feat = model(points, compute_recon=False, return_features=True)
            feats.append(batch_feat.detach().cpu())
            logits.append(batch_logits.detach().cpu())
            labels.append(label.detach().cpu())
    feats = torch.cat(feats, dim=0)
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, logits, labels


def confusion_matrix_from_logits(logits, labels, num_classes):
    pred = logits.argmax(dim=-1)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(labels.view(-1), pred.view(-1)):
        cm[int(t), int(p)] += 1
    return cm


def find_hardest_pair(cm):
    row_sum = cm.sum(dim=1).clamp(min=1)
    err = cm.float() / row_sum.unsqueeze(1).float()
    err.fill_diagonal_(0.0)
    flat_idx = int(err.argmax().item())
    a = flat_idx // err.shape[1]
    b = flat_idx % err.shape[1]
    return a, b, float(err[a, b].item()), int(cm[a, b].item())


def topk_stats(logits, labels, ks=(1, 2, 5)):
    out = {}
    max_k = max(ks)
    topk = torch.topk(logits, k=max_k, dim=-1).indices
    for k in ks:
        hit = (topk[:, :k] == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        out[f"top{k}_hit"] = float(hit)
    return out


def pair_subset_metrics(logits, labels, a, b):
    mask = (labels == a) | (labels == b)
    sub_logits = logits[mask]
    sub_labels = labels[mask]
    pred = sub_logits.argmax(dim=-1)
    acc = (pred == sub_labels).float().mean().item() if len(sub_labels) else float("nan")
    a_to_b = ((sub_labels == a) & (pred == b)).float()
    b_to_a = ((sub_labels == b) & (pred == a)).float()
    a_total = max(1, int((sub_labels == a).sum().item()))
    b_total = max(1, int((sub_labels == b).sum().item()))
    margin = (sub_logits[:, a] - sub_logits[:, b]).mean().item() if len(sub_logits) else float("nan")
    return {
        "pair_samples": int(mask.sum().item()),
        "direct_top1_acc": float(acc),
        "a_to_b_rate": float(a_to_b.sum().item() / a_total),
        "b_to_a_rate": float(b_to_a.sum().item() / b_total),
        "mean_margin_a_minus_b": float(margin),
    }


def fit_pair_probe(train_feat, train_labels, test_feat, test_labels, a, b):
    train_mask = (train_labels == a) | (train_labels == b)
    test_mask = (test_labels == a) | (test_labels == b)
    x_tr = train_feat[train_mask].numpy()
    x_te = test_feat[test_mask].numpy()
    y_tr = (train_labels[train_mask] == b).numpy().astype(np.int64)
    y_te = (test_labels[test_mask] == b).numpy().astype(np.int64)
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return {"pair_probe_bal_acc": float("nan"), "pair_probe_acc": float("nan")}
    clf = LinearSVC(random_state=0, dual=False, max_iter=10000)
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    acc = float((pred == y_te).mean())
    pos = y_te == 1
    neg = y_te == 0
    tpr = float((pred[pos] == 1).mean()) if pos.any() else float("nan")
    tnr = float((pred[neg] == 0).mean()) if neg.any() else float("nan")
    bal = float(np.nanmean([tpr, tnr]))
    return {"pair_probe_bal_acc": bal, "pair_probe_acc": acc}


def per_class_accuracy(cm):
    out = {}
    for idx in range(cm.shape[0]):
        total = int(cm[idx].sum().item())
        acc = float(cm[idx, idx].item() / total) if total > 0 else float("nan")
        name = SCANOBJECTNN_CLASS_NAMES[idx] if idx < len(SCANOBJECTNN_CLASS_NAMES) else str(idx)
        out[name] = acc
    return out


def to_md(summary):
    lines = []
    lines.append("# ScanObjectNN Readout Audit")
    lines.append("")
    lines.append(f"- config: `{summary['config']}`")
    lines.append(f"- ckpt: `{summary['ckpt']}`")
    lines.append(f"- train split: `{summary['train_split']}`")
    lines.append(f"- test split: `{summary['test_split']}`")
    lines.append("")
    lines.append("## Global")
    lines.append("")
    lines.append(f"- top1 acc: `{summary['top1_acc']:.4f}`")
    lines.append(f"- top2 hit: `{summary['top2_hit']:.4f}`")
    lines.append(f"- top5 hit: `{summary['top5_hit']:.4f}`")
    lines.append("")
    hp = summary["hardest_pair"]
    lines.append("## Hardest Pair")
    lines.append("")
    lines.append(
        f"- pair: `{hp['a_name']} ({hp['a_id']}) -> {hp['b_name']} ({hp['b_id']})`"
    )
    lines.append(f"- off-diagonal count: `{hp['count']}`")
    lines.append(f"- normalized confusion: `{hp['rate']:.4f}`")
    lines.append(f"- pair direct top1 acc: `{hp['direct_top1_acc']:.4f}`")
    lines.append(f"- `{hp['a_name']} -> {hp['b_name']}`: `{hp['a_to_b_rate']:.4f}`")
    lines.append(f"- `{hp['b_name']} -> {hp['a_name']}`: `{hp['b_to_a_rate']:.4f}`")
    lines.append(f"- mean logit margin ({hp['a_name']} - {hp['b_name']}): `{hp['mean_margin_a_minus_b']:.4f}`")
    lines.append(f"- binary probe acc: `{hp['pair_probe_acc']:.4f}`")
    lines.append(f"- binary probe bal acc: `{hp['pair_probe_bal_acc']:.4f}`")
    lines.append("")
    lines.append("## Per-Class Acc")
    lines.append("")
    lines.append("| class | acc |")
    lines.append("|---|---:|")
    for name, acc in summary["per_class_acc"].items():
        lines.append(f"| `{name}` | `{acc:.4f}` |")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    cfg = load_config(args.config, args.batch_size, args.test_split)
    model = build_model(cfg, args.ckpt)

    train_loader = build_loader(cfg, args.train_split, args.num_workers)
    test_loader = build_loader(cfg, args.test_split, args.num_workers)

    train_feat, train_logits, train_labels = extract_split(
        model, train_loader, cfg.npoints, max_batches=args.max_train_batches
    )
    test_feat, test_logits, test_labels = extract_split(
        model, test_loader, cfg.npoints, max_batches=args.max_test_batches
    )

    topk = topk_stats(test_logits, test_labels, ks=(1, 2, 5))
    cm = confusion_matrix_from_logits(test_logits, test_labels, num_classes=test_logits.shape[1])

    if args.pair:
        a, b = [int(x) for x in args.pair.split(",")]
        rate = float(cm[a, b].item() / max(1, int(cm[a].sum().item())))
        count = int(cm[a, b].item())
    else:
        a, b, rate, count = find_hardest_pair(cm)

    hp = {
        "a_id": int(a),
        "b_id": int(b),
        "a_name": SCANOBJECTNN_CLASS_NAMES[a] if a < len(SCANOBJECTNN_CLASS_NAMES) else str(a),
        "b_name": SCANOBJECTNN_CLASS_NAMES[b] if b < len(SCANOBJECTNN_CLASS_NAMES) else str(b),
        "rate": float(rate),
        "count": int(count),
    }
    hp.update(pair_subset_metrics(test_logits, test_labels, a, b))
    hp.update(fit_pair_probe(train_feat, train_labels, test_feat, test_labels, a, b))

    pred = test_logits.argmax(dim=-1)
    top1_acc = float((pred == test_labels).float().mean().item())

    summary = {
        "config": args.config,
        "ckpt": args.ckpt,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "top1_acc": top1_acc,
        **topk,
        "hardest_pair": hp,
        "per_class_acc": per_class_accuracy(cm),
    }

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(to_md(summary))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
