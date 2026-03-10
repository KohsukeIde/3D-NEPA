#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

MIX_VARIANT="${MIX_VARIANT:-pc33mesh33udf33}"  # pc100 | mesh50udf50 | pc33mesh33udf33

case "${MIX_VARIANT}" in
  pc100)
    MIX_CONFIG="nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_pc100.yaml"
    RUN_STEM="pt_pc100_reconch_g2_e300"
    ;;
  mesh50udf50)
    MIX_CONFIG="nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_mesh50_udf50.yaml"
    RUN_STEM="pt_mesh50udf50_reconch_g2_e300"
    ;;
  pc33mesh33udf33)
    MIX_CONFIG="nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_pc33_mesh33_udf33.yaml"
    RUN_STEM="pt_pc33mesh33udf33_reconch_g2_e300"
    ;;
  *)
    echo "[error] unknown MIX_VARIANT=${MIX_VARIANT} (use: pc100 | mesh50udf50 | pc33mesh33udf33)"
    exit 2
    ;;
esac

RUN_SET="${RUN_SET:-patchnepa_abci_recong2_$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-${RUN_STEM}}"
SAVE_DIR="${SAVE_DIR:-runs/patchnepa_tokens/${RUN_SET}/${RUN_STEM}}"
LOG_ROOT="${LOG_ROOT:-logs/patch_nepa_pretrain_tokens/${RUN_SET}}"

export WORKDIR="${ROOT_DIR}"
export GROUP_LIST="${GROUP_LIST:-qgah50055}"
export RT_QF="${RT_QF:-4}"
export WALLTIME="${WALLTIME:-72:00:00}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MASTER_PORT="${MASTER_PORT:-29420}"
export STAGE2_REQUIRE_GLOBAL_BATCH128="${STAGE2_REQUIRE_GLOBAL_BATCH128:-1}"

export RUN_SET
export RUN_TAG
export SAVE_DIR
export LOG_ROOT
export MIX_CONFIG

export MAX_STEPS="${MAX_STEPS:-10000}"
export EPOCHS="${EPOCHS:-300}"
export SAVE_EVERY="${SAVE_EVERY:-1000}"
export BATCH="${BATCH:-8}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export SEED="${SEED:-0}"
export LR="${LR:-3e-4}"
export LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.025}"
export MIN_LR="${MIN_LR:-1e-6}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"

export N_SURF="${N_SURF:-2048}"
export N_QRY="${N_QRY:-1024}"
export N_RAY="${N_RAY:-0}"
export TOKEN_QA_LAYOUT="${TOKEN_QA_LAYOUT:-split_sep}"
export PATCH_EMBED="${PATCH_EMBED:-fps_knn}"
export PATCH_LOCAL_ENCODER="${PATCH_LOCAL_ENCODER:-pointmae_conv}"
export PATCH_FPS_RANDOM_START="${PATCH_FPS_RANDOM_START:-1}"
export GROUP_SIZE="${GROUP_SIZE:-32}"
export NUM_GROUPS="${NUM_GROUPS:-64}"
export PATCH_ORDER_MODE="${PATCH_ORDER_MODE:-morton}"
export BACKBONE_MODE="${BACKBONE_MODE:-nepa2d}"

export ANSWER_IN_DIM="${ANSWER_IN_DIM:-9}"
export PRETRAIN_OBJECTIVE="${PRETRAIN_OBJECTIVE:-recon_chamfer}"
export RECON_LOSS_MODE="${RECON_LOSS_MODE:-composite}"
export RECON_GENERATOR_DEPTH="${RECON_GENERATOR_DEPTH:-2}"
export RECON_CHAMFER_METRIC="${RECON_CHAMFER_METRIC:-l2}"
export LOSS_TARGET_MODE="${LOSS_TARGET_MODE:-content_tokens}"
export LOSS_MASK_MODE="${LOSS_MASK_MODE:-answer_and_point_context}"
export REG_VAR_WEIGHT="${REG_VAR_WEIGHT:-0.0}"
export REG_COV_WEIGHT="${REG_COV_WEIGHT:-0.0}"

export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-patchnepa-pretrain}"
export WANDB_GROUP="${WANDB_GROUP:-${RUN_SET}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_SET}_${RUN_STEM}}"
export WANDB_TAGS="${WANDB_TAGS:-abci,patchnepa,current,recong2,${MIX_VARIANT}}"
export WANDB_MODE="${WANDB_MODE:-online}"

echo "[abci-pretrain] mix_variant=${MIX_VARIANT}"
echo "[abci-pretrain] mix_config=${MIX_CONFIG}"
echo "[abci-pretrain] save_dir=${SAVE_DIR}"
echo "[abci-pretrain] log_root=${LOG_ROOT}"

exec "${ROOT_DIR}/scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh"
