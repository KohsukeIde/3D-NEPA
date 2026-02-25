#!/bin/bash
set -euo pipefail

# A-only (n_point=256, n_ray=256) query-rethink ablation:
# pretrain 9 configs -> eval 2 protocols each (SOTA-fair / NEPA-full), with CPAC+mesh(chamfer).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PRETRAIN_SCRIPT="${WORKDIR}/scripts/pretrain/nepa3d_pretrain_multinode_pbsdsh.sh"
EVAL_SCRIPT="${WORKDIR}/scripts/eval/nepa3d_eval_cls_cpac_qf.sh"

if ! command -v qsub >/dev/null 2>&1; then
  echo "[error] qsub not found"
  exit 1
fi

RUN_SET="${RUN_SET:-a256_queryrethink_$(date +%Y%m%d_%H%M%S)}"

# ----- Pretrain defaults (small scale) -----
PT_GROUP_LIST="${PT_GROUP_LIST:-qgah50055}"
PT_NODES_PER_RUN="${PT_NODES_PER_RUN:-1}"
PT_WALLTIME="${PT_WALLTIME:-24:00:00}"
PT_NUM_WORKERS="${PT_NUM_WORKERS:-8}"
PT_BATCH="${PT_BATCH:-16}"
PT_EPOCHS="${PT_EPOCHS:-100}"
PT_LR="${PT_LR:-3e-4}"
PT_SEED_BASE="${PT_SEED_BASE:-1200}"
PT_MAX_LEN="${PT_MAX_LEN:-1300}"
PT_RFPS_M="${PT_RFPS_M:-1024}"
PT_DROP_PATH="${PT_DROP_PATH:-0.0}"

# mild pretrain aug
AUG_ROTATE_Z="${AUG_ROTATE_Z:-1}"
AUG_SCALE_MIN="${AUG_SCALE_MIN:-0.8}"
AUG_SCALE_MAX="${AUG_SCALE_MAX:-1.25}"
AUG_TRANSLATE="${AUG_TRANSLATE:-0.0}"
AUG_JITTER_SIGMA="${AUG_JITTER_SIGMA:-0.01}"
AUG_JITTER_CLIP="${AUG_JITTER_CLIP:-0.05}"
AUG_RECOMPUTE_DIST="${AUG_RECOMPUTE_DIST:-0}"

# keep dual mask OFF for this serialization-focused ablation
DUAL_MASK_NEAR="${DUAL_MASK_NEAR:-0.0}"
DUAL_MASK_FAR="${DUAL_MASK_FAR:-0.0}"
DUAL_MASK_WINDOW="${DUAL_MASK_WINDOW:-32}"
DUAL_MASK_WARMUP_FRAC="${DUAL_MASK_WARMUP_FRAC:-0.05}"
DUAL_MASK_TYPE_AWARE="${DUAL_MASK_TYPE_AWARE:-0}"
DUAL_MASK_WINDOW_SCALE="${DUAL_MASK_WINDOW_SCALE:-linear}"
DUAL_MASK_WINDOW_REF_TOTAL="${DUAL_MASK_WINDOW_REF_TOTAL:--1}"

PRETRAIN_SAVE_ROOT="${PRETRAIN_SAVE_ROOT:-runs/pretrain_a256_queryrethink_${RUN_SET}}"
PRETRAIN_LOG_ROOT="${PRETRAIN_LOG_ROOT:-logs/pretrain/${RUN_SET}}"
mkdir -p "${WORKDIR}/${PRETRAIN_LOG_ROOT}"

# ----- Eval defaults -----
EV_GROUP_LIST="${EV_GROUP_LIST:-qgah50055}"
EV_RT_QF="${EV_RT_QF:-1}"
EV_WALLTIME="${EV_WALLTIME:-24:00:00}"
EV_NPROC_PER_NODE="${EV_NPROC_PER_NODE:-4}"
EV_NUM_WORKERS="${EV_NUM_WORKERS:-8}"
EV_BATCH_SCAN="${EV_BATCH_SCAN:-96}"
EV_BATCH_MODELNET="${EV_BATCH_MODELNET:-128}"
EV_EPOCHS_CLS="${EV_EPOCHS_CLS:-100}"
EV_LR_CLS="${EV_LR_CLS:-1e-4}"
EV_N_POINT_CLS="${EV_N_POINT_CLS:-256}"
EV_N_RAY_CLS="${EV_N_RAY_CLS:-0}"
EV_SEED_BASE="${EV_SEED_BASE:-2200}"

EVAL_ROOT="${EVAL_ROOT:-runs/eval_a256_queryrethink_${RUN_SET}}"
RESULTS_ROOT="${RESULTS_ROOT:-results/a256_queryrethink_${RUN_SET}}"
LOG_ROOT="${LOG_ROOT:-logs/eval/a256_queryrethink_${RUN_SET}}"
mkdir -p "${WORKDIR}/${LOG_ROOT}" "${WORKDIR}/${RESULTS_ROOT}" "${WORKDIR}/${EVAL_ROOT}"

# CPAC + mesh/chamfer
RUN_SCAN="${RUN_SCAN:-1}"
RUN_MODELNET="${RUN_MODELNET:-1}"
RUN_CPAC="${RUN_CPAC:-1}"
CPAC_N_CONTEXT="${CPAC_N_CONTEXT:-256}"
CPAC_N_QUERY="${CPAC_N_QUERY:-256}"
CPAC_MAX_LEN="${CPAC_MAX_LEN:-1300}"
CPAC_MAX_SHAPES="${CPAC_MAX_SHAPES:-800}"
CPAC_MESH_EVAL="${CPAC_MESH_EVAL:-1}"
CPAC_MESH_EVAL_MAX_SHAPES="${CPAC_MESH_EVAL_MAX_SHAPES:-800}"
CPAC_MESH_GRID_RES="${CPAC_MESH_GRID_RES:-24}"
CPAC_MESH_CHUNK_N_QUERY="${CPAC_MESH_CHUNK_N_QUERY:-512}"
CPAC_MESH_MC_LEVEL="${CPAC_MESH_MC_LEVEL:-0.03}"
CPAC_MESH_NUM_SAMPLES="${CPAC_MESH_NUM_SAMPLES:-10000}"
CPAC_MESH_FSCORE_TAU="${CPAC_MESH_FSCORE_TAU:-0.01}"
CPAC_MESH_STORE_PER_SHAPE="${CPAC_MESH_STORE_PER_SHAPE:-0}"

COMMON_PRETRAIN="WORKDIR=${WORKDIR},NUM_WORKERS=${PT_NUM_WORKERS},BATCH=${PT_BATCH},EPOCHS=${PT_EPOCHS},LR=${PT_LR},N_POINT=256,N_RAY=256,MAX_LEN=${PT_MAX_LEN},PT_SAMPLE_MODE_TRAIN=rfps,PT_FPS_KEY=auto,PT_RFPS_M=${PT_RFPS_M},MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_scan_mainsplit.yaml,QA_TOKENS=1,PT_XYZ_KEY=pt_xyz_pool,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=0,POINT_ORDER_MODE=fps,DROP_PATH=${PT_DROP_PATH},AUG_ROTATE_Z=${AUG_ROTATE_Z},AUG_SCALE_MIN=${AUG_SCALE_MIN},AUG_SCALE_MAX=${AUG_SCALE_MAX},AUG_TRANSLATE=${AUG_TRANSLATE},AUG_JITTER_SIGMA=${AUG_JITTER_SIGMA},AUG_JITTER_CLIP=${AUG_JITTER_CLIP},AUG_RECOMPUTE_DIST=${AUG_RECOMPUTE_DIST},DUAL_MASK_NEAR=${DUAL_MASK_NEAR},DUAL_MASK_FAR=${DUAL_MASK_FAR},DUAL_MASK_WINDOW=${DUAL_MASK_WINDOW},DUAL_MASK_WARMUP_FRAC=${DUAL_MASK_WARMUP_FRAC},DUAL_MASK_TYPE_AWARE=${DUAL_MASK_TYPE_AWARE},DUAL_MASK_WINDOW_SCALE=${DUAL_MASK_WINDOW_SCALE},DUAL_MASK_WINDOW_REF_TOTAL=${DUAL_MASK_WINDOW_REF_TOTAL}"

# rid|qa_layout|sequence_mode|event_order_mode|ray_order_mode|type_specific_pos
defs=(
  "b00_interleave_theta|interleave|block|morton|theta_phi|0"
  "b01_split_theta|split_sep|block|morton|theta_phi|0"
  "b02_split_theta_typepos|split_sep|block|morton|theta_phi|1"
  "b03_split_viewraster_typepos|split_sep|block|morton|view_raster|1"
  "b04_split_xanchor_morton_typepos|split_sep|block|morton|x_anchor_morton|1"
  "b05_split_xanchor_fps_typepos|split_sep|block|morton|x_anchor_fps|1"
  "b06_split_dirfps_typepos|split_sep|block|morton|dir_fps|1"
  "b07_event_xanchor_typepos|split_sep|event|morton|x_anchor_morton|1"
  "b08_event_dirfps_typepos|split_sep|event|morton|dir_fps|1"
)

declare -A PT_JOB_BY_RID

echo "[submit] pretrain ablation runs (${#defs[@]})"
for i in "${!defs[@]}"; do
  IFS='|' read -r rid qa_layout seq_mode event_mode ray_mode type_pos <<< "${defs[$i]}"
  run_tag="A256_${rid}_${RUN_SET}"
  save_dir="${PRETRAIN_SAVE_ROOT}_${rid}"
  seed="$((PT_SEED_BASE + i))"
  vars="${COMMON_PRETRAIN},RUN_TAG=${run_tag},SEED=${seed},QA_LAYOUT=${qa_layout},SEQUENCE_MODE=${seq_mode},EVENT_ORDER_MODE=${event_mode},RAY_ORDER_MODE=${ray_mode},TYPE_SPECIFIC_POS=${type_pos},SAVE_DIR=${save_dir}"

  job_id="$(qsub -l "rt_QF=${PT_NODES_PER_RUN}" -l "walltime=${PT_WALLTIME}" -W "group_list=${PT_GROUP_LIST}" -N "pt_${rid}" -v "${vars}" "${PRETRAIN_SCRIPT}")"
  PT_JOB_BY_RID["${rid}"]="${job_id}"
  echo "  - ${rid}: ${job_id}"
done

JOBS_TXT="${WORKDIR}/${LOG_ROOT}/submitted_jobs_${RUN_SET}.txt"
: > "${JOBS_TXT}"

echo "[submit] eval runs (2 protocols per pretrain)"
for i in "${!defs[@]}"; do
  IFS='|' read -r rid qa_layout seq_mode event_mode ray_mode type_pos <<< "${defs[$i]}"
  dep="afterok:${PT_JOB_BY_RID[$rid]}"
  ckpt="${WORKDIR}/${PRETRAIN_SAVE_ROOT}_${rid}/last.pt"

  # SOTA-fair
  run_tag_sf="${rid}_sotafair"
  seed_sf="$((EV_SEED_BASE + i*2 + 0))"
  eval_vars_sf="WORKDIR=${WORKDIR},RUN_TAG=${run_tag_sf},CKPT=${ckpt},EVAL_ROOT=${EVAL_ROOT},RESULTS_ROOT=${RESULTS_ROOT},LOG_ROOT=${LOG_ROOT},SEED=${seed_sf},NUM_WORKERS=${EV_NUM_WORKERS},BATCH_SCAN=${EV_BATCH_SCAN},BATCH_MODELNET=${EV_BATCH_MODELNET},EPOCHS_CLS=${EV_EPOCHS_CLS},LR_CLS=${EV_LR_CLS},N_POINT_CLS=${EV_N_POINT_CLS},N_RAY_CLS=${EV_N_RAY_CLS},PT_XYZ_KEY_CLS=pc_xyz,PT_DIST_KEY_CLS=pt_dist_pool,NPROC_PER_NODE=${EV_NPROC_PER_NODE},DDP_FIND_UNUSED_PARAMETERS=1,CLS_POOLING=mean_q,POINT_ORDER_MODE=morton,ABLATE_POINT_DIST=1,USE_FC_NORM=0,LABEL_SMOOTHING=0.0,WEIGHT_DECAY_CLS=0.05,WEIGHT_DECAY_NORM=0.0,LR_SCHEDULER=cosine,WARMUP_EPOCHS=10,WARMUP_START_FACTOR=0.1,MIN_LR=1e-6,LLRD=1.0,LLRD_MODE=exp,DROP_PATH=0.0,GRAD_ACCUM_STEPS=1,MAX_GRAD_NORM=1.0,VAL_SPLIT_MODE=group_auto,PT_SAMPLE_MODE_TRAIN_CLS=fps,PT_SAMPLE_MODE_EVAL_CLS=fps,PT_RFPS_M_CLS=4096,AUG_EVAL=1,AUG_RECOMPUTE_DIST=1,RUN_SCAN=${RUN_SCAN},RUN_MODELNET=${RUN_MODELNET},RUN_CPAC=${RUN_CPAC},SCAN_CACHE_ROOT=data/scanobjectnn_main_split_v2,MODELNET_CACHE_ROOT=data/modelnet40_cache_v2,UNPAIRED_CACHE_ROOT=data/shapenet_unpaired_cache_v1,SCAN_AUG_PRESET=scanobjectnn,MODELNET_AUG_PRESET=modelnet40,MC_EVAL_K_VAL=1,MC_EVAL_K_TEST=10,CPAC_N_CONTEXT=${CPAC_N_CONTEXT},CPAC_N_QUERY=${CPAC_N_QUERY},CPAC_MAX_LEN=${CPAC_MAX_LEN},CPAC_MAX_SHAPES=${CPAC_MAX_SHAPES},CPAC_MESH_EVAL=${CPAC_MESH_EVAL},CPAC_MESH_EVAL_MAX_SHAPES=${CPAC_MESH_EVAL_MAX_SHAPES},CPAC_MESH_GRID_RES=${CPAC_MESH_GRID_RES},CPAC_MESH_CHUNK_N_QUERY=${CPAC_MESH_CHUNK_N_QUERY},CPAC_MESH_MC_LEVEL=${CPAC_MESH_MC_LEVEL},CPAC_MESH_NUM_SAMPLES=${CPAC_MESH_NUM_SAMPLES},CPAC_MESH_FSCORE_TAU=${CPAC_MESH_FSCORE_TAU},CPAC_MESH_STORE_PER_SHAPE=${CPAC_MESH_STORE_PER_SHAPE}"
  out_sf="${WORKDIR}/${LOG_ROOT}/${run_tag_sf}.out"
  err_sf="${WORKDIR}/${LOG_ROOT}/${run_tag_sf}.err"
  job_sf="$(qsub -l "rt_QF=${EV_RT_QF}" -l "walltime=${EV_WALLTIME}" -W "group_list=${EV_GROUP_LIST}" -N "ev_${run_tag_sf}" -o "${out_sf}" -e "${err_sf}" -v "${eval_vars_sf}" -W "depend=${dep}" "${EVAL_SCRIPT}")"
  echo "${run_tag_sf} ${job_sf}" >> "${JOBS_TXT}"
  echo "  - ${run_tag_sf}: ${job_sf} (dep=${dep})"

  # NEPA-full
  run_tag_nf="${rid}_nepafull"
  seed_nf="$((EV_SEED_BASE + i*2 + 1))"
  eval_vars_nf="WORKDIR=${WORKDIR},RUN_TAG=${run_tag_nf},CKPT=${ckpt},EVAL_ROOT=${EVAL_ROOT},RESULTS_ROOT=${RESULTS_ROOT},LOG_ROOT=${LOG_ROOT},SEED=${seed_nf},NUM_WORKERS=${EV_NUM_WORKERS},BATCH_SCAN=${EV_BATCH_SCAN},BATCH_MODELNET=${EV_BATCH_MODELNET},EPOCHS_CLS=${EV_EPOCHS_CLS},LR_CLS=${EV_LR_CLS},N_POINT_CLS=${EV_N_POINT_CLS},N_RAY_CLS=${EV_N_RAY_CLS},PT_XYZ_KEY_CLS=pt_xyz_pool,PT_DIST_KEY_CLS=pt_dist_pool,NPROC_PER_NODE=${EV_NPROC_PER_NODE},DDP_FIND_UNUSED_PARAMETERS=1,CLS_POOLING=mean_q,POINT_ORDER_MODE=fps,ABLATE_POINT_DIST=0,USE_FC_NORM=0,LABEL_SMOOTHING=0.0,WEIGHT_DECAY_CLS=0.05,WEIGHT_DECAY_NORM=0.0,LR_SCHEDULER=cosine,WARMUP_EPOCHS=10,WARMUP_START_FACTOR=0.1,MIN_LR=1e-6,LLRD=1.0,LLRD_MODE=exp,DROP_PATH=0.0,GRAD_ACCUM_STEPS=1,MAX_GRAD_NORM=1.0,VAL_SPLIT_MODE=group_auto,PT_SAMPLE_MODE_TRAIN_CLS=fps,PT_SAMPLE_MODE_EVAL_CLS=fps,PT_RFPS_M_CLS=4096,AUG_EVAL=1,AUG_RECOMPUTE_DIST=1,RUN_SCAN=${RUN_SCAN},RUN_MODELNET=${RUN_MODELNET},RUN_CPAC=${RUN_CPAC},SCAN_CACHE_ROOT=data/scanobjectnn_main_split_v2,MODELNET_CACHE_ROOT=data/modelnet40_cache_v2,UNPAIRED_CACHE_ROOT=data/shapenet_unpaired_cache_v1,SCAN_AUG_PRESET=scanobjectnn,MODELNET_AUG_PRESET=modelnet40,MC_EVAL_K_VAL=1,MC_EVAL_K_TEST=10,CPAC_N_CONTEXT=${CPAC_N_CONTEXT},CPAC_N_QUERY=${CPAC_N_QUERY},CPAC_MAX_LEN=${CPAC_MAX_LEN},CPAC_MAX_SHAPES=${CPAC_MAX_SHAPES},CPAC_MESH_EVAL=${CPAC_MESH_EVAL},CPAC_MESH_EVAL_MAX_SHAPES=${CPAC_MESH_EVAL_MAX_SHAPES},CPAC_MESH_GRID_RES=${CPAC_MESH_GRID_RES},CPAC_MESH_CHUNK_N_QUERY=${CPAC_MESH_CHUNK_N_QUERY},CPAC_MESH_MC_LEVEL=${CPAC_MESH_MC_LEVEL},CPAC_MESH_NUM_SAMPLES=${CPAC_MESH_NUM_SAMPLES},CPAC_MESH_FSCORE_TAU=${CPAC_MESH_FSCORE_TAU},CPAC_MESH_STORE_PER_SHAPE=${CPAC_MESH_STORE_PER_SHAPE}"
  out_nf="${WORKDIR}/${LOG_ROOT}/${run_tag_nf}.out"
  err_nf="${WORKDIR}/${LOG_ROOT}/${run_tag_nf}.err"
  job_nf="$(qsub -l "rt_QF=${EV_RT_QF}" -l "walltime=${EV_WALLTIME}" -W "group_list=${EV_GROUP_LIST}" -N "ev_${run_tag_nf}" -o "${out_nf}" -e "${err_nf}" -v "${eval_vars_nf}" -W "depend=${dep}" "${EVAL_SCRIPT}")"
  echo "${run_tag_nf} ${job_nf}" >> "${JOBS_TXT}"
  echo "  - ${run_tag_nf}: ${job_nf} (dep=${dep})"
done

echo "[done] run_set=${RUN_SET}"
echo "[done] eval job list: ${JOBS_TXT}"
