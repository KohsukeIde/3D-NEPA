#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
DATA_ROOT="${DATA_ROOT:-${WORKDIR}/data}"
SHAPENETPART_ROOT="${SHAPENETPART_ROOT:-${DATA_ROOT}/shapenetcore_partanno_segmentation_benchmark_v0_normal}"
RUN_ROOT="${RUN_ROOT:-${WORKDIR}/pointnepa/results/pointgpt_nomask_orderrandom_full}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/pointgpt_nomask_orderrandom_full}"
MARKER_ROOT="${RUN_ROOT}/markers"

PRETRAIN_EXP="${PRETRAIN_EXP:-pgpt_s_nomask_ordrand_e300_20260504}"
PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-cfgs/PointGPT-S/pretrain_nomask_orderrandom.yaml}"
PRETRAIN_NPROC="${PRETRAIN_NPROC:-4}"
PRETRAIN_GPUS="${PRETRAIN_GPUS:-0,1,2,3}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FT_NUM_WORKERS="${FT_NUM_WORKERS:-8}"
AUDIT_NUM_WORKERS="${AUDIT_NUM_WORKERS:-8}"
PART_BATCH_SIZE="${PART_BATCH_SIZE:-16}"
PART_EPOCH="${PART_EPOCH:-300}"
PART_WARMUP_EPOCH="${PART_WARMUP_EPOCH:-30}"
PART_LR="${PART_LR:-0.0002}"
SEED="${SEED:-0}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_MODE="${WANDB_MODE:-disabled}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${POINTGPT_DIR}:${PYTHONPATH:-}"
export USE_WANDB
export WANDB_MODE

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${MARKER_ROOT}"

if [[ ! -d "${POINTGPT_DIR}" ]]; then
  echo "[error] PointGPT dir not found: ${POINTGPT_DIR}" >&2
  exit 2
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[error] data root not found: ${DATA_ROOT}" >&2
  exit 2
fi

mkdir -p "${POINTGPT_DIR}"
ln -sfn "${DATA_ROOT}" "${POINTGPT_DIR}/data"

cd "${POINTGPT_DIR}"

run_logged() {
  local marker="$1"
  local logfile="$2"
  shift 2
  if [[ -f "${marker}" ]]; then
    echo "[skip] ${marker}"
    return 0
  fi
  mkdir -p "$(dirname "${logfile}")" "$(dirname "${marker}")"
  echo "[run] $*" | tee "${logfile}"
  "$@" 2>&1 | tee -a "${logfile}"
  touch "${marker}"
}

wait_all() {
  local status=0
  for pid in "$@"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  return "${status}"
}

pretrain_path() {
  printf "%s/experiments/%s/%s/%s" \
    "${POINTGPT_DIR}" \
    "$(basename "${PRETRAIN_CONFIG%.*}")" \
    "$(basename "$(dirname "${PRETRAIN_CONFIG}")")" \
    "${PRETRAIN_EXP}"
}

PRETRAIN_DIR="$(pretrain_path)"
PRETRAIN_CKPT="${PRETRAIN_DIR}/ckpt-last.pth"

run_pretrain() {
  if [[ -f "${PRETRAIN_CKPT}" ]]; then
    touch "${MARKER_ROOT}/pretrain.done"
    echo "[skip] pretrain checkpoint exists: ${PRETRAIN_CKPT}"
    return 0
  fi
  run_logged \
    "${MARKER_ROOT}/pretrain.done" \
    "${LOG_ROOT}/pretrain_nomask_orderrandom.log" \
    env CUDA_VISIBLE_DEVICES="${PRETRAIN_GPUS}" \
      torchrun --standalone --nproc_per_node="${PRETRAIN_NPROC}" \
      main.py \
      --launcher pytorch \
      --config "${PRETRAIN_CONFIG}" \
      --exp_name "${PRETRAIN_EXP}" \
      --num_workers "${NUM_WORKERS}" \
      --seed "${SEED}"
}

scan_config() {
  case "$1" in
    obj_bg) printf "cfgs/PointGPT-S/finetune_scan_objbg.yaml" ;;
    obj_only) printf "cfgs/PointGPT-S/finetune_scan_objonly.yaml" ;;
    pb_t50_rs) printf "cfgs/PointGPT-S/finetune_scan_hardest.yaml" ;;
    *) echo "[error] unknown ScanObjectNN variant: $1" >&2; return 2 ;;
  esac
}

scan_exp() {
  case "$1" in
    obj_bg) printf "pgpt_s_nomask_ordrand_objbg_e300_20260504" ;;
    obj_only) printf "pgpt_s_nomask_ordrand_objonly_e300_20260504" ;;
    pb_t50_rs) printf "pgpt_s_nomask_ordrand_hardest_e300_20260504" ;;
    *) echo "[error] unknown ScanObjectNN variant: $1" >&2; return 2 ;;
  esac
}

scan_ckpt() {
  local variant="$1"
  local cfg exp
  cfg="$(scan_config "${variant}")"
  exp="$(scan_exp "${variant}")"
  printf "%s/experiments/%s/%s/%s/ckpt-best.pth" \
    "${POINTGPT_DIR}" \
    "$(basename "${cfg%.*}")" \
    "$(basename "$(dirname "${cfg}")")" \
    "${exp}"
}

run_scan_ft_one() {
  local variant="$1"
  local gpu="$2"
  local cfg exp ckpt marker logfile
  cfg="$(scan_config "${variant}")"
  exp="$(scan_exp "${variant}")"
  ckpt="$(scan_ckpt "${variant}")"
  marker="${MARKER_ROOT}/scan_${variant}_ft.done"
  logfile="${LOG_ROOT}/scan_${variant}_ft.log"
  if [[ -f "${ckpt}" ]]; then
    touch "${marker}"
    echo "[skip] ${variant} FT checkpoint exists: ${ckpt}"
    return 0
  fi
  run_logged \
    "${marker}" \
    "${logfile}" \
    env CUDA_VISIBLE_DEVICES="${gpu}" FT_RECON_WEIGHT=0 SAVE_LAST_EVERY_EPOCH=0 \
      torchrun --standalone --nproc_per_node=1 \
      main.py \
      --launcher pytorch \
      --config "${cfg}" \
      --exp_name "${exp}" \
      --num_workers "${FT_NUM_WORKERS}" \
      --val_freq 1 \
      --ft_recon_weight 0 \
      --save_last_every_epoch 0 \
      --finetune_model \
      --ckpts "${PRETRAIN_CKPT}" \
      --seed "${SEED}"
}

run_scan_ft_all() {
  run_scan_ft_one obj_bg 0 &
  local p0=$!
  run_scan_ft_one obj_only 1 &
  local p1=$!
  run_scan_ft_one pb_t50_rs 2 &
  local p2=$!
  wait_all "${p0}" "${p1}" "${p2}"
}

PART_EXP="${PART_EXP:-pgpt_s_shapenetpart_nomask_ordrand_e300_20260504}"
PART_CKPT="${POINTGPT_DIR}/segmentation/log/part_seg/${PART_EXP}/checkpoints/best_model.pth"

run_part_ft() {
  if [[ -f "${PART_CKPT}" ]]; then
    touch "${MARKER_ROOT}/shapenetpart_ft.done"
    echo "[skip] ShapeNetPart FT checkpoint exists: ${PART_CKPT}"
    return 0
  fi
  if [[ ! -d "${SHAPENETPART_ROOT}" ]]; then
    echo "[blocked] ShapeNetPart root not found: ${SHAPENETPART_ROOT}" | tee "${RUN_ROOT}/shapenetpart_blocked.txt"
    touch "${MARKER_ROOT}/shapenetpart_ft.blocked"
    return 0
  fi
  run_logged \
    "${MARKER_ROOT}/shapenetpart_ft.done" \
    "${LOG_ROOT}/shapenetpart_ft.log" \
    env CUDA_VISIBLE_DEVICES=3 \
      WORKDIR="${WORKDIR}" \
      POINTGPT_DIR="${POINTGPT_DIR}" \
      ROOT="${SHAPENETPART_ROOT}" \
      CKPT_PATH="${PRETRAIN_CKPT}" \
      RUN_NAME="${PART_EXP}" \
      EPOCH="${PART_EPOCH}" \
      WARMUP_EPOCH="${PART_WARMUP_EPOCH}" \
      BATCH_SIZE="${PART_BATCH_SIZE}" \
      LEARNING_RATE="${PART_LR}" \
      SEED="${SEED}" \
      GROUP_MODE=fps_knn \
      bash "${WORKDIR}/pointnepa/scripts/local/pointgpt_s_shapenetpart_ft.sh"
}

run_downstream_all() {
  run_scan_ft_all &
  local scan_pid=$!
  run_part_ft &
  local part_pid=$!
  wait_all "${scan_pid}" "${part_pid}"
}

audit_scan_one() {
  local variant="$1"
  local gpu="$2"
  local cfg ckpt prefix marker logfile
  cfg="$(scan_config "${variant}")"
  ckpt="$(scan_ckpt "${variant}")"
  prefix="${RUN_ROOT}/scanobjectnn_${variant}_nomask_ordrand"
  marker="${MARKER_ROOT}/scan_${variant}_audits.done"
  logfile="${LOG_ROOT}/scan_${variant}_audits.log"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[error] missing FT checkpoint for ${variant}: ${ckpt}" >&2
    return 2
  fi
  if [[ -f "${marker}" ]]; then
    echo "[skip] ${marker}"
    return 0
  fi
  mkdir -p "$(dirname "${logfile}")" "$(dirname "${marker}")"
  echo "[run] ScanObjectNN ${variant} audits" | tee "${logfile}"
  env CUDA_VISIBLE_DEVICES="${gpu}" python tools/eval_scanobjectnn_readout_audit.py \
      --config "${cfg}" \
      --ckpt "${ckpt}" \
      --batch_size 32 \
      --num_workers "${AUDIT_NUM_WORKERS}" \
      --output_json "${prefix}_readout.json" \
      --output_md "${prefix}_readout.md" 2>&1 | tee -a "${logfile}"
  env CUDA_VISIBLE_DEVICES="${gpu}" python tools/eval_scanobjectnn_support_stress.py \
      --config "${cfg}" \
      --ckpt "${ckpt}" \
      --batch_size 32 \
      --num_workers "${AUDIT_NUM_WORKERS}" \
      --output_json "${prefix}_support.json" \
      --output_md "${prefix}_support.md" 2>&1 | tee -a "${logfile}"
  env CUDA_VISIBLE_DEVICES="${gpu}" python tools/eval_scanobjectnn_grouping_ablation.py \
      --config "${cfg}" \
      --ckpt "${ckpt}" \
      --batch_size 32 \
      --num_workers "${AUDIT_NUM_WORKERS}" \
      --support-conditions clean,random_keep20,structured_keep20,xyz_zero \
      --output_json "${prefix}_grouping.json" \
      --output_csv "${prefix}_grouping.csv" \
      --output_md "${prefix}_grouping.md" 2>&1 | tee -a "${logfile}"
  touch "${marker}"
}

audit_scan_all() {
  audit_scan_one obj_bg 0 &
  local p0=$!
  audit_scan_one obj_only 1 &
  local p1=$!
  audit_scan_one pb_t50_rs 2 &
  local p2=$!
  wait_all "${p0}" "${p1}" "${p2}"
}

audit_part() {
  local prefix marker logfile
  prefix="${RUN_ROOT}/shapenetpart_nomask_ordrand"
  marker="${MARKER_ROOT}/shapenetpart_audits.done"
  logfile="${LOG_ROOT}/shapenetpart_audits.log"
  if [[ ! -f "${PART_CKPT}" ]]; then
    if [[ -f "${MARKER_ROOT}/shapenetpart_ft.blocked" ]]; then
      echo "[skip] ShapeNetPart audits blocked by missing data root"
      return 0
    fi
    echo "[error] missing ShapeNetPart checkpoint: ${PART_CKPT}" >&2
    return 2
  fi
  if [[ -f "${marker}" ]]; then
    echo "[skip] ${marker}"
    return 0
  fi
  mkdir -p "$(dirname "${logfile}")" "$(dirname "${marker}")"
  echo "[run] ShapeNetPart audits" | tee "${logfile}"
  env CUDA_VISIBLE_DEVICES=3 python segmentation/eval_shapenetpart_support_stress.py \
      --ckpt "${PART_CKPT}" \
      --root "${SHAPENETPART_ROOT}" \
      --batch_size "${PART_BATCH_SIZE}" \
      --num_workers "${AUDIT_NUM_WORKERS}" \
      --seed "${SEED}" \
      --output_json "${prefix}_support_unique.json" \
      --output_md "${prefix}_support_unique.md" 2>&1 | tee -a "${logfile}"
  env CUDA_VISIBLE_DEVICES=3 python segmentation/eval_shapenetpart_grouping_ablation.py \
      --ckpt "${PART_CKPT}" \
      --root "${SHAPENETPART_ROOT}" \
      --batch_size "${PART_BATCH_SIZE}" \
      --num_workers "${AUDIT_NUM_WORKERS}" \
      --seed "${SEED}" \
      --output_json "${prefix}_grouping_unique.json" \
      --output_csv "${prefix}_grouping_unique.csv" \
      --output_md "${prefix}_grouping_unique.md" 2>&1 | tee -a "${logfile}"
  touch "${marker}"
}

run_audits_all() {
  audit_scan_all &
  local scan_pid=$!
  audit_part &
  local part_pid=$!
  wait_all "${scan_pid}" "${part_pid}"
}

write_summary() {
  python - "${RUN_ROOT}" "${PRETRAIN_CKPT}" "${PART_CKPT}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
pretrain_ckpt = Path(sys.argv[2])
part_ckpt = Path(sys.argv[3])

def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())

def scan_readout_row(split: str):
    p = run_root / f"scanobjectnn_{split}_nomask_ordrand_readout.json"
    d = read_json(p)
    if not d:
        return None
    hp = d.get("hardest_pair", {})
    pair = hp.get("pair")
    if pair is None and hp.get("a_name") is not None and hp.get("b_name") is not None:
        pair = [hp.get("a_name"), hp.get("b_name")]
    return split, d.get("top1_acc"), d.get("top2_hit"), d.get("top5_hit"), pair

def scan_support_row(split: str):
    p = run_root / f"scanobjectnn_{split}_nomask_ordrand_support.json"
    d = read_json(p)
    if not d:
        return None
    rows = {r.get("condition", r.get("name")): r["acc"] for r in d.get("conditions", [])}
    return split, rows

lines = [
    "# PointGPT-S No-Mask Order-Randomized Diagnostics",
    "",
    "Purpose: rerun the PointGPT no-mask + order-randomized object diagnostics with the current local code and unique-retained ShapeNetPart scoring.",
    "",
    "## Checkpoints",
    "",
    f"- pretrain: `{pretrain_ckpt}`" if pretrain_ckpt.exists() else f"- pretrain: `MISSING {pretrain_ckpt}`",
    f"- ShapeNetPart FT: `{part_ckpt}`" if part_ckpt.exists() else f"- ShapeNetPart FT: `MISSING/BLOCKED {part_ckpt}`",
    "",
    "## Result Files",
    "",
]
for split in ["obj_bg", "obj_only", "pb_t50_rs"]:
    prefix = f"scanobjectnn_{split}_nomask_ordrand"
    lines.extend([
        f"- ScanObjectNN `{split}` readout: `{run_root / (prefix + '_readout.md')}`",
        f"- ScanObjectNN `{split}` support: `{run_root / (prefix + '_support.md')}`",
        f"- ScanObjectNN `{split}` eval-time grouping: `{run_root / (prefix + '_grouping.md')}`",
    ])
lines.extend([
    f"- ShapeNetPart support: `{run_root / 'shapenetpart_nomask_ordrand_support_unique.md'}`",
    f"- ShapeNetPart eval-time grouping: `{run_root / 'shapenetpart_nomask_ordrand_grouping_unique.md'}`",
    "",
    "## ScanObjectNN Readout",
    "",
    "| split | top1 | top2 hit | top5 hit | hardest pair |",
    "|---|---:|---:|---:|---|",
])
for split in ["obj_bg", "obj_only", "pb_t50_rs"]:
    row = scan_readout_row(split)
    if row is None:
        lines.append(f"| `{split}` | n/a | n/a | n/a | n/a |")
    else:
        split, top1, top2, top5, pair = row
        pair_txt = " -> ".join(pair) if isinstance(pair, list) else str(pair)
        lines.append(f"| `{split}` | `{top1:.4f}` | `{top2:.4f}` | `{top5:.4f}` | `{pair_txt}` |")
lines.extend([
    "",
    "## ScanObjectNN Support",
    "",
    "| split | clean | random80 | random50 | random20 | random10 | structured80 | structured50 | structured20 | structured10 | xyz_zero |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
])
for split in ["obj_bg", "obj_only", "pb_t50_rs"]:
    row = scan_support_row(split)
    if row is None:
        lines.append(f"| `{split}` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    else:
        split, rows = row
        keys = ["clean", "random_keep80", "random_keep50", "random_keep20", "random_keep10", "structured_keep80", "structured_keep50", "structured_keep20", "structured_keep10", "xyz_zero"]
        vals = [rows.get(k) for k in keys]
        lines.append("| `{}` | {} |".format(split, " | ".join("n/a" if v is None else f"`{v:.4f}`" for v in vals)))
lines.extend([
    "",
    "## Notes",
    "",
    "- ShapeNetPart support metrics use unique retained original point indices; fixed-size forward resampling is aggregated back by original point.",
    "- ScanObjectNN rows use the PointGPT single-label classification support/readout protocol; random/structured keep conditions include 80/50/20/10.",
    "- Grouping rows are eval-time patchization perturbations with checkpoint/readout fixed.",
])
(run_root / "pointgpt_nomask_orderrandom_diagnostics_summary.md").write_text("\n".join(lines) + "\n")
PY
}

run_pretrain
if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
  echo "[error] pretrain did not produce checkpoint: ${PRETRAIN_CKPT}" >&2
  exit 1
fi

run_downstream_all
run_audits_all
write_summary

echo "[done] PointGPT-S no-mask order-randomized diagnostics"
echo "summary=${RUN_ROOT}/pointgpt_nomask_orderrandom_diagnostics_summary.md"
