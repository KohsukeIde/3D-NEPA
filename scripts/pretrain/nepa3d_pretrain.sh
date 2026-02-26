#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=24:00:00
#PBS -P gag51403
#PBS -N nepa3d_pretrain
#PBS -o nepa3d_pretrain.out
#PBS -e nepa3d_pretrain.err

set -eu

# Environment setup
. /etc/profile.d/modules.sh
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6}"
if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" 2>/dev/null || echo "[warn] module load ${CUDA_MODULE} failed; continue with current module set."
fi

# Move to working directory
WORKDIR="${WORKDIR:-${PBS_O_WORKDIR:-$(pwd)}}"
cd "${WORKDIR}"

# Activate Python virtual environment
if [ -f ".venv/bin/activate" ]; then
  . .venv/bin/activate
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE_ROOT="${CACHE_ROOT:-data/modelnet40_cache_v0}"
MIX_CONFIG="${MIX_CONFIG:-}"
MIX_NUM_SAMPLES="${MIX_NUM_SAMPLES:-0}"
MIX_SEED="${MIX_SEED:-0}"
BACKEND="${BACKEND:-mesh}"
BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-3e-4}"
# 2D NEPA style linear LR scaling:
#   LEARNING_RATE = BASE_LEARNING_RATE * TOTAL_BATCH_SIZE / 256
# Backward-compatible default keeps LR at current default when TOTAL_BATCH_SIZE=32.
LR_SCALE_ENABLE="${LR_SCALE_ENABLE:-0}"          # 1: enable linear scaling, 0: disable
LR_SCALE_REF_BATCH="${LR_SCALE_REF_BATCH:-256}"  # denominator in scaling rule
LR_BASE_TOTAL_BATCH="${LR_BASE_TOTAL_BATCH:-32}" # baseline total batch for deriving BASE_LEARNING_RATE
BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-}"     # if empty, derived from LR and LR_BASE_TOTAL_BATCH
N_POINT="${N_POINT:-256}"
N_RAY="${N_RAY:-256}"
MAX_LEN="${MAX_LEN:--1}"  # -1 = auto from (qa_tokens/add_eos/n_point/n_ray/schedules)
N_POINT_SCHEDULE="${N_POINT_SCHEDULE:-}"  # e.g., '0:256,10:512,20:1024'
N_RAY_SCHEDULE="${N_RAY_SCHEDULE:-}"      # e.g., '0:256,10:512'
D_MODEL="${D_MODEL:-768}"
LAYERS="${LAYERS:-12}"
HEADS="${HEADS:-12}"
DROP_PATH="${DROP_PATH:-0.0}"
BACKBONE_IMPL="${BACKBONE_IMPL:-nepa2d}"
QKV_BIAS="${QKV_BIAS:-1}"
QK_NORM="${QK_NORM:-1}"
QK_NORM_AFFINE="${QK_NORM_AFFINE:-0}"
QK_NORM_BIAS="${QK_NORM_BIAS:-0}"
LAYERSCALE_VALUE="${LAYERSCALE_VALUE:-1e-5}"
ROPE_THETA="${ROPE_THETA:-100.0}"
LAYER_NORM_EPS="${LAYER_NORM_EPS:-1e-12}"
HIDDEN_DROPOUT_PROB="${HIDDEN_DROPOUT_PROB:-0.0}"
ATTENTION_PROBS_DROPOUT_PROB="${ATTENTION_PROBS_DROPOUT_PROB:-0.0}"
USE_GATED_MLP="${USE_GATED_MLP:-0}"
FINAL_LAYERNORM="${FINAL_LAYERNORM:-1}"
SAVE_DIR="${SAVE_DIR:-runs/querynepa3d_meshpre_v0}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SAVE_LAST="${SAVE_LAST:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME="${RESUME:-}"
RESUME_OPTIMIZER="${RESUME_OPTIMIZER:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-auto}" # auto/no/fp16/bf16
NUM_PROCESSES="${NUM_PROCESSES:-1}"       # >1 enables Accelerate DDP
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-127.0.0.1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
ACCELERATE_PYTHON="${ACCELERATE_PYTHON:-${PYTHON_BIN}}"
ACCELERATE_LAUNCH_MODULE="${ACCELERATE_LAUNCH_MODULE:-accelerate.commands.launch}"
DROP_RAY_PROB="${DROP_RAY_PROB:-0.0}"
FORCE_MISSING_RAY="${FORCE_MISSING_RAY:-0}"
ADD_EOS="${ADD_EOS:-1}"
QA_TOKENS="${QA_TOKENS:-0}"
QA_LAYOUT="${QA_LAYOUT:-interleave}"
SEQUENCE_MODE="${SEQUENCE_MODE:-block}"
EVENT_ORDER_MODE="${EVENT_ORDER_MODE:-morton}"
RAY_ORDER_MODE="${RAY_ORDER_MODE:-theta_phi}"
RAY_ANCHOR_MISS_T="${RAY_ANCHOR_MISS_T:-4.0}"
RAY_VIEW_TOL="${RAY_VIEW_TOL:-1e-6}"
TYPE_SPECIFIC_POS="${TYPE_SPECIFIC_POS:-0}"
PT_XYZ_KEY="${PT_XYZ_KEY:-pt_xyz_pool}"
PT_DIST_KEY="${PT_DIST_KEY:-pt_dist_pool}"
ABLATE_POINT_DIST="${ABLATE_POINT_DIST:-0}"
PT_SAMPLE_MODE_TRAIN="${PT_SAMPLE_MODE_TRAIN:-random}"
PT_FPS_KEY="${PT_FPS_KEY:-auto}"
PT_RFPS_M="${PT_RFPS_M:-4096}"
POINT_ORDER_MODE="${POINT_ORDER_MODE:-morton}"
AUG_ROTATE_Z="${AUG_ROTATE_Z:-0}"
AUG_SCALE_MIN="${AUG_SCALE_MIN:-1.0}"
AUG_SCALE_MAX="${AUG_SCALE_MAX:-1.0}"
AUG_TRANSLATE="${AUG_TRANSLATE:-0.0}"
AUG_JITTER_SIGMA="${AUG_JITTER_SIGMA:-0.0}"
AUG_JITTER_CLIP="${AUG_JITTER_CLIP:-0.0}"
AUG_RECOMPUTE_DIST="${AUG_RECOMPUTE_DIST:-0}"
VOXEL_GRID="${VOXEL_GRID:-64}"
VOXEL_DILATE="${VOXEL_DILATE:-1}"
VOXEL_MAX_STEPS="${VOXEL_MAX_STEPS:-0}"
OBJECTIVE="${OBJECTIVE:-nepa}"
MASK_RATIO="${MASK_RATIO:-0.4}"
# B-2/B-3 auxiliary losses (default OFF)
AUX_B2_WEIGHT="${AUX_B2_WEIGHT:-0.0}"
AUX_B2_HIT_WEIGHT="${AUX_B2_HIT_WEIGHT:-1.0}"
AUX_B2_T_WEIGHT="${AUX_B2_T_WEIGHT:-1.0}"
AUX_B2_RANK_WEIGHT="${AUX_B2_RANK_WEIGHT:-1.0}"
AUX_B2_RANK_PAIRS="${AUX_B2_RANK_PAIRS:-128}"
AUX_B2_RANK_MARGIN="${AUX_B2_RANK_MARGIN:-0.0}"
AUX_B3_WEIGHT="${AUX_B3_WEIGHT:-0.0}"
AUX_B3_NEAR_TAU="${AUX_B3_NEAR_TAU:-0.05}"
# C-0 teacher-student refresh
TEACHER_CKPT="${TEACHER_CKPT:-}"
TEACHER_DISTILL_WEIGHT="${TEACHER_DISTILL_WEIGHT:-0.0}"
TEACHER_ANSWER_DROP_PROB="${TEACHER_ANSWER_DROP_PROB:-0.0}"
CYCLE_WEIGHT="${CYCLE_WEIGHT:-0.0}"
CYCLE_ANSWER_DROP_PROB="${CYCLE_ANSWER_DROP_PROB:-0.3}"
# D/E auxiliaries (default OFF)
D_HARD_WEIGHT="${D_HARD_WEIGHT:-0.0}"
D_HARD_TOP_FRAC="${D_HARD_TOP_FRAC:-0.25}"
D_HARD_MIN_TOKENS="${D_HARD_MIN_TOKENS:-32}"
AUX_E_WEIGHT="${AUX_E_WEIGHT:-0.0}"
DUAL_MASK_NEAR="${DUAL_MASK_NEAR:-0.0}"
DUAL_MASK_FAR="${DUAL_MASK_FAR:-0.0}"
DUAL_MASK_WINDOW="${DUAL_MASK_WINDOW:-32}"
DUAL_MASK_WARMUP_FRAC="${DUAL_MASK_WARMUP_FRAC:-0.05}"
DUAL_MASK_TYPE_AWARE="${DUAL_MASK_TYPE_AWARE:-0}"
DUAL_MASK_WINDOW_SCALE="${DUAL_MASK_WINDOW_SCALE:-linear}"
DUAL_MASK_WINDOW_REF_TOTAL="${DUAL_MASK_WINDOW_REF_TOTAL:--1}"

EXTRA_FORCE=""
if [ "${FORCE_MISSING_RAY}" = "1" ]; then
  EXTRA_FORCE="--force_missing_ray"
fi
EXTRA_AUG_ROTATE_Z=""
if [ "${AUG_ROTATE_Z}" = "1" ]; then
  EXTRA_AUG_ROTATE_Z="--aug_rotate_z"
fi

TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-$((BATCH * NUM_PROCESSES))}"
if [ "${LR_SCALE_ENABLE}" = "1" ]; then
  if [ -z "${BASE_LEARNING_RATE}" ]; then
    BASE_LEARNING_RATE="$("${PYTHON_BIN}" -c "print(float('${LR}') * 256.0 / float('${LR_BASE_TOTAL_BATCH}'))")"
  fi
  LR="$("${PYTHON_BIN}" -c "print(float('${BASE_LEARNING_RATE}') * float('${TOTAL_BATCH_SIZE}') / float('${LR_SCALE_REF_BATCH}'))")"
  echo "[lr-scale] enabled: base_lr=${BASE_LEARNING_RATE} total_batch=${TOTAL_BATCH_SIZE} ref_batch=${LR_SCALE_REF_BATCH} lr=${LR}"
else
  echo "[lr-scale] disabled: lr=${LR}"
fi

LAUNCH_MIXED_PRECISION="${MIXED_PRECISION}"
if [ "${LAUNCH_MIXED_PRECISION}" = "auto" ]; then
  LAUNCH_MIXED_PRECISION="$("${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    print("no")
elif hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
    print("bf16")
else:
    print("fp16")
PY
)"
fi
echo "[accelerate] launch mixed_precision=${LAUNCH_MIXED_PRECISION} (requested=${MIXED_PRECISION})"

if [ "${NUM_PROCESSES}" -gt 1 ]; then
  "${ACCELERATE_PYTHON}" -m "${ACCELERATE_LAUNCH_MODULE}" \
    --multi_gpu \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines "${NUM_MACHINES}" \
    --machine_rank "${MACHINE_RANK}" \
    --main_process_ip "${MAIN_PROCESS_IP}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    --mixed_precision "${LAUNCH_MIXED_PRECISION}" \
    -m nepa3d.train.pretrain \
  --cache_root "${CACHE_ROOT}" \
  --mix_config "${MIX_CONFIG}" \
  --mix_num_samples "${MIX_NUM_SAMPLES}" \
  --mix_seed "${MIX_SEED}" \
  --backend "${BACKEND}" \
  --batch "${BATCH}" --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --n_point "${N_POINT}" --n_ray "${N_RAY}" \
  --max_len "${MAX_LEN}" \
  --n_point_schedule "${N_POINT_SCHEDULE}" \
  --n_ray_schedule "${N_RAY_SCHEDULE}" \
  --d_model "${D_MODEL}" --layers "${LAYERS}" --heads "${HEADS}" \
  --drop_path "${DROP_PATH}" \
  --backbone_impl "${BACKBONE_IMPL}" \
  --qkv_bias "${QKV_BIAS}" \
  --qk_norm "${QK_NORM}" \
  --qk_norm_affine "${QK_NORM_AFFINE}" \
  --qk_norm_bias "${QK_NORM_BIAS}" \
  --layerscale_value "${LAYERSCALE_VALUE}" \
  --rope_theta "${ROPE_THETA}" \
  --layer_norm_eps "${LAYER_NORM_EPS}" \
  --hidden_dropout_prob "${HIDDEN_DROPOUT_PROB}" \
  --attention_probs_dropout_prob "${ATTENTION_PROBS_DROPOUT_PROB}" \
  --use_gated_mlp "${USE_GATED_MLP}" \
  --final_layernorm "${FINAL_LAYERNORM}" \
  --num_workers "${NUM_WORKERS}" \
  --drop_ray_prob "${DROP_RAY_PROB}" \
  --add_eos "${ADD_EOS}" \
  --qa_tokens "${QA_TOKENS}" \
  --qa_layout "${QA_LAYOUT}" \
  --sequence_mode "${SEQUENCE_MODE}" \
  --event_order_mode "${EVENT_ORDER_MODE}" \
  --ray_order_mode "${RAY_ORDER_MODE}" \
  --ray_anchor_miss_t "${RAY_ANCHOR_MISS_T}" \
  --ray_view_tol "${RAY_VIEW_TOL}" \
  --type_specific_pos "${TYPE_SPECIFIC_POS}" \
  --pt_xyz_key "${PT_XYZ_KEY}" \
  --pt_dist_key "${PT_DIST_KEY}" \
  --ablate_point_dist "${ABLATE_POINT_DIST}" \
  --pt_sample_mode_train "${PT_SAMPLE_MODE_TRAIN}" \
  --pt_fps_key "${PT_FPS_KEY}" \
  --pt_rfps_m "${PT_RFPS_M}" \
  --point_order_mode "${POINT_ORDER_MODE}" \
  ${EXTRA_AUG_ROTATE_Z} \
  --aug_scale_min "${AUG_SCALE_MIN}" \
  --aug_scale_max "${AUG_SCALE_MAX}" \
  --aug_translate "${AUG_TRANSLATE}" \
  --aug_jitter_sigma "${AUG_JITTER_SIGMA}" \
  --aug_jitter_clip "${AUG_JITTER_CLIP}" \
  --aug_recompute_dist "${AUG_RECOMPUTE_DIST}" \
  --objective "${OBJECTIVE}" \
  --dual_mask_near "${DUAL_MASK_NEAR}" \
  --dual_mask_far "${DUAL_MASK_FAR}" \
  --dual_mask_window "${DUAL_MASK_WINDOW}" \
  --dual_mask_warmup_frac "${DUAL_MASK_WARMUP_FRAC}" \
  --dual_mask_type_aware "${DUAL_MASK_TYPE_AWARE}" \
  --dual_mask_window_scale "${DUAL_MASK_WINDOW_SCALE}" \
  --dual_mask_window_ref_total "${DUAL_MASK_WINDOW_REF_TOTAL}" \
  --mask_ratio "${MASK_RATIO}" \
  --mixed_precision "${LAUNCH_MIXED_PRECISION}" \
  --aux_b2_weight "${AUX_B2_WEIGHT}" \
  --aux_b2_hit_weight "${AUX_B2_HIT_WEIGHT}" \
  --aux_b2_t_weight "${AUX_B2_T_WEIGHT}" \
  --aux_b2_rank_weight "${AUX_B2_RANK_WEIGHT}" \
  --aux_b2_rank_pairs "${AUX_B2_RANK_PAIRS}" \
  --aux_b2_rank_margin "${AUX_B2_RANK_MARGIN}" \
  --aux_b3_weight "${AUX_B3_WEIGHT}" \
  --aux_b3_near_tau "${AUX_B3_NEAR_TAU}" \
  --teacher_ckpt "${TEACHER_CKPT}" \
  --teacher_distill_weight "${TEACHER_DISTILL_WEIGHT}" \
  --teacher_answer_drop_prob "${TEACHER_ANSWER_DROP_PROB}" \
  --cycle_weight "${CYCLE_WEIGHT}" \
  --cycle_answer_drop_prob "${CYCLE_ANSWER_DROP_PROB}" \
  --d_hard_weight "${D_HARD_WEIGHT}" \
  --d_hard_top_frac "${D_HARD_TOP_FRAC}" \
  --d_hard_min_tokens "${D_HARD_MIN_TOKENS}" \
  --aux_e_weight "${AUX_E_WEIGHT}" \
  --save_every "${SAVE_EVERY}" \
  --save_last "${SAVE_LAST}" \
  --auto_resume "${AUTO_RESUME}" \
  --resume "${RESUME}" \
  --resume_optimizer "${RESUME_OPTIMIZER}" \
  --voxel_grid "${VOXEL_GRID}" \
  --voxel_dilate "${VOXEL_DILATE}" \
  --voxel_max_steps "${VOXEL_MAX_STEPS}" \
  --seed "${SEED}" \
  ${EXTRA_FORCE} \
  --save_dir "${SAVE_DIR}"
else
  "${PYTHON_BIN}" -m nepa3d.train.pretrain \
  --cache_root "${CACHE_ROOT}" \
  --mix_config "${MIX_CONFIG}" \
  --mix_num_samples "${MIX_NUM_SAMPLES}" \
  --mix_seed "${MIX_SEED}" \
  --backend "${BACKEND}" \
  --batch "${BATCH}" --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --n_point "${N_POINT}" --n_ray "${N_RAY}" \
  --max_len "${MAX_LEN}" \
  --n_point_schedule "${N_POINT_SCHEDULE}" \
  --n_ray_schedule "${N_RAY_SCHEDULE}" \
  --d_model "${D_MODEL}" --layers "${LAYERS}" --heads "${HEADS}" \
  --drop_path "${DROP_PATH}" \
  --backbone_impl "${BACKBONE_IMPL}" \
  --qkv_bias "${QKV_BIAS}" \
  --qk_norm "${QK_NORM}" \
  --qk_norm_affine "${QK_NORM_AFFINE}" \
  --qk_norm_bias "${QK_NORM_BIAS}" \
  --layerscale_value "${LAYERSCALE_VALUE}" \
  --rope_theta "${ROPE_THETA}" \
  --layer_norm_eps "${LAYER_NORM_EPS}" \
  --hidden_dropout_prob "${HIDDEN_DROPOUT_PROB}" \
  --attention_probs_dropout_prob "${ATTENTION_PROBS_DROPOUT_PROB}" \
  --use_gated_mlp "${USE_GATED_MLP}" \
  --final_layernorm "${FINAL_LAYERNORM}" \
  --num_workers "${NUM_WORKERS}" \
  --drop_ray_prob "${DROP_RAY_PROB}" \
  --add_eos "${ADD_EOS}" \
  --qa_tokens "${QA_TOKENS}" \
  --qa_layout "${QA_LAYOUT}" \
  --sequence_mode "${SEQUENCE_MODE}" \
  --event_order_mode "${EVENT_ORDER_MODE}" \
  --ray_order_mode "${RAY_ORDER_MODE}" \
  --ray_anchor_miss_t "${RAY_ANCHOR_MISS_T}" \
  --ray_view_tol "${RAY_VIEW_TOL}" \
  --type_specific_pos "${TYPE_SPECIFIC_POS}" \
  --pt_xyz_key "${PT_XYZ_KEY}" \
  --pt_dist_key "${PT_DIST_KEY}" \
  --ablate_point_dist "${ABLATE_POINT_DIST}" \
  --pt_sample_mode_train "${PT_SAMPLE_MODE_TRAIN}" \
  --pt_fps_key "${PT_FPS_KEY}" \
  --pt_rfps_m "${PT_RFPS_M}" \
  --point_order_mode "${POINT_ORDER_MODE}" \
  ${EXTRA_AUG_ROTATE_Z} \
  --aug_scale_min "${AUG_SCALE_MIN}" \
  --aug_scale_max "${AUG_SCALE_MAX}" \
  --aug_translate "${AUG_TRANSLATE}" \
  --aug_jitter_sigma "${AUG_JITTER_SIGMA}" \
  --aug_jitter_clip "${AUG_JITTER_CLIP}" \
  --aug_recompute_dist "${AUG_RECOMPUTE_DIST}" \
  --objective "${OBJECTIVE}" \
  --dual_mask_near "${DUAL_MASK_NEAR}" \
  --dual_mask_far "${DUAL_MASK_FAR}" \
  --dual_mask_window "${DUAL_MASK_WINDOW}" \
  --dual_mask_warmup_frac "${DUAL_MASK_WARMUP_FRAC}" \
  --dual_mask_type_aware "${DUAL_MASK_TYPE_AWARE}" \
  --dual_mask_window_scale "${DUAL_MASK_WINDOW_SCALE}" \
  --dual_mask_window_ref_total "${DUAL_MASK_WINDOW_REF_TOTAL}" \
  --mask_ratio "${MASK_RATIO}" \
  --mixed_precision "${LAUNCH_MIXED_PRECISION}" \
  --aux_b2_weight "${AUX_B2_WEIGHT}" \
  --aux_b2_hit_weight "${AUX_B2_HIT_WEIGHT}" \
  --aux_b2_t_weight "${AUX_B2_T_WEIGHT}" \
  --aux_b2_rank_weight "${AUX_B2_RANK_WEIGHT}" \
  --aux_b2_rank_pairs "${AUX_B2_RANK_PAIRS}" \
  --aux_b2_rank_margin "${AUX_B2_RANK_MARGIN}" \
  --aux_b3_weight "${AUX_B3_WEIGHT}" \
  --aux_b3_near_tau "${AUX_B3_NEAR_TAU}" \
  --teacher_ckpt "${TEACHER_CKPT}" \
  --teacher_distill_weight "${TEACHER_DISTILL_WEIGHT}" \
  --teacher_answer_drop_prob "${TEACHER_ANSWER_DROP_PROB}" \
  --cycle_weight "${CYCLE_WEIGHT}" \
  --cycle_answer_drop_prob "${CYCLE_ANSWER_DROP_PROB}" \
  --d_hard_weight "${D_HARD_WEIGHT}" \
  --d_hard_top_frac "${D_HARD_TOP_FRAC}" \
  --d_hard_min_tokens "${D_HARD_MIN_TOKENS}" \
  --aux_e_weight "${AUX_E_WEIGHT}" \
  --save_every "${SAVE_EVERY}" \
  --save_last "${SAVE_LAST}" \
  --auto_resume "${AUTO_RESUME}" \
  --resume "${RESUME}" \
  --resume_optimizer "${RESUME_OPTIMIZER}" \
  --voxel_grid "${VOXEL_GRID}" \
  --voxel_dilate "${VOXEL_DILATE}" \
  --voxel_max_steps "${VOXEL_MAX_STEPS}" \
  --seed "${SEED}" \
  ${EXTRA_FORCE} \
  --save_dir "${SAVE_DIR}"
fi
