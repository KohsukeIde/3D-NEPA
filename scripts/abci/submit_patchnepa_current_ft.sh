#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

DEFAULT_CKPT="runs/patchnepa_tokens/patchnepa_recong2_full300_20260306_072643/pt_pc33mesh33udf33_reconch_g2_e300/ckpt_final.pt"
CKPT="${CKPT:-${DEFAULT_CKPT}}"

RUN_SET="${RUN_SET:-patchnepa_abci_ft_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/sanity/patchnepa_ft/${RUN_SET}}"
SAVE_DIR="${SAVE_DIR:-runs/sanity/patchnepa_ft_abci}"

export WORKDIR="${ROOT_DIR}"
export RUN_SET
export LOG_ROOT
export SAVE_DIR
export CKPT

export VARIANTS="${VARIANTS:-obj_bg,obj_only,pb_t50_rs}"
export RT_QF="${RT_QF:-1}"
export WALLTIME="${WALLTIME:-24:00:00}"
export GROUP_LIST="${GROUP_LIST:-qgah50055}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export SEED="${SEED:-0}"
export EPOCHS="${EPOCHS:-300}"
export BATCH="${BATCH:-64}"
export BATCH_MODE="${BATCH_MODE:-global}"
export LR="${LR:-5e-4}"
export WD="${WD:-0.05}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
export N_POINT="${N_POINT:-2048}"
export NUM_GROUPS="${NUM_GROUPS:-64}"
export GROUP_SIZE="${GROUP_SIZE:-32}"

export MODEL_SOURCE="${MODEL_SOURCE:-patchnepa}"
export USE_RAY_PATCH="${USE_RAY_PATCH:-0}"
export N_RAY="${N_RAY:-256}"
export PT_SAMPLE_MODE_TRAIN="${PT_SAMPLE_MODE_TRAIN:-random}"
export PT_SAMPLE_MODE_EVAL="${PT_SAMPLE_MODE_EVAL:-fps}"
export POOLING="${POOLING:-cls_max}"
export HEAD_MODE="${HEAD_MODE:-pointmae_mlp}"
export PATCHNEPA_FT_MODE="${PATCHNEPA_FT_MODE:-qa_zeroa}"
export PATCHNEPA_CLS_TOKEN_SOURCE="${PATCHNEPA_CLS_TOKEN_SOURCE:-last_q}"
export PATCHNEPA_FREEZE_PATCH_EMBED="${PATCHNEPA_FREEZE_PATCH_EMBED:-1}"
export LLRD_START="${LLRD_START:-1.0}"
export LLRD_END="${LLRD_END:-1.0}"
export LLRD_SCHEDULER="${LLRD_SCHEDULER:-static}"
export LLRD_MODE="${LLRD_MODE:-linear}"
export AUG_PRESET="${AUG_PRESET:-pointmae}"
export AUG_EVAL="${AUG_EVAL:-1}"
export MC_EVAL_K_TEST="${MC_EVAL_K_TEST:-10}"
export VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-file}"
export ALLOW_SCAN_UNISCALE_V2="${ALLOW_SCAN_UNISCALE_V2:-0}"

export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-patchnepa-finetune}"
export WANDB_GROUP="${WANDB_GROUP:-${RUN_SET}}"
export WANDB_TAGS="${WANDB_TAGS:-abci,patchnepa,current,finetune}"
export WANDB_MODE="${WANDB_MODE:-online}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[error] CKPT not found: ${CKPT}"
  exit 2
fi

echo "[abci-ft] ckpt=${CKPT}"
echo "[abci-ft] variants=${VARIANTS}"
echo "[abci-ft] log_root=${LOG_ROOT}"
echo "[abci-ft] save_dir=${SAVE_DIR}"

exec "${ROOT_DIR}/scripts/sanity/submit_patchnepa_finetune_variants_qf.sh"
