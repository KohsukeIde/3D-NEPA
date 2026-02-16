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
module load cuda/12.6

# Move to working directory
cd /groups/gag51403/ide/3D-NEPA

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
N_POINT="${N_POINT:-256}"
N_RAY="${N_RAY:-256}"
MAX_LEN="${MAX_LEN:--1}"  # -1 = auto from (qa_tokens/add_eos/n_point/n_ray/schedules)
N_POINT_SCHEDULE="${N_POINT_SCHEDULE:-}"  # e.g., '0:256,10:512,20:1024'
N_RAY_SCHEDULE="${N_RAY_SCHEDULE:-}"      # e.g., '0:256,10:512'
D_MODEL="${D_MODEL:-384}"
LAYERS="${LAYERS:-8}"
HEADS="${HEADS:-6}"
SAVE_DIR="${SAVE_DIR:-runs/querynepa3d_meshpre_v0}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SAVE_LAST="${SAVE_LAST:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME="${RESUME:-}"
RESUME_OPTIMIZER="${RESUME_OPTIMIZER:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
DROP_RAY_PROB="${DROP_RAY_PROB:-0.0}"
FORCE_MISSING_RAY="${FORCE_MISSING_RAY:-0}"
ADD_EOS="${ADD_EOS:-1}"
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

EXTRA_FORCE=""
if [ "${FORCE_MISSING_RAY}" = "1" ]; then
  EXTRA_FORCE="--force_missing_ray"
fi

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
  --num_workers "${NUM_WORKERS}" \
  --drop_ray_prob "${DROP_RAY_PROB}" \
  --add_eos "${ADD_EOS}" \
  --objective "${OBJECTIVE}" \
  --mask_ratio "${MASK_RATIO}" \
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
