#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-patchnepa_visocc_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/local_patchnepa_visocc/${RUN_TAG}}"
PIPELINE_LOG="${LOG_ROOT}/pipeline.log"
DECISION_JSON="${LOG_ROOT}/decision.json"
PRETRAIN_LOG_ROOT="${PRETRAIN_LOG_ROOT:-${ROOT_DIR}/logs/local_patchnepa_pretrain}"
CPAC_LOG_ROOT="${CPAC_LOG_ROOT:-${ROOT_DIR}/logs/local_patchnepa_cpac}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results/local_patchnepa_cpac}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT_DIR}/runs/local_patchnepa}"
GPU_BASE="${GPU_BASE:-0}"
GPU_VISOCC="${GPU_VISOCC:-1}"
GPU_OPTIONAL="${GPU_OPTIONAL:-0}"

SHAPENET_ROOT="${SHAPENET_ROOT:-data/ShapeNetCore.v2/synsets}"
SOURCE_CACHE="${SOURCE_CACHE:-data/shapenet_cache_v2_20260306_visocc}"
SPLIT_JSON_BASE="${SPLIT_JSON_BASE:-data/shapenet_unpaired_splits_v2_20260306_visocc.json}"
SPLIT_JSON_PC33="${SPLIT_JSON_PC33:-data/shapenet_unpaired_splits_v2_pc33_mesh33_udf33_visocc.json}"
SPLIT_JSON_M50U50="${SPLIT_JSON_M50U50:-data/shapenet_unpaired_splits_v2_mesh50_udf50_visocc.json}"
UNPAIRED_ROOT="${UNPAIRED_ROOT:-data/shapenet_unpaired_cache_v2_20260306_visocc}"
UNPAIRED_DROP1_ROOT="${UNPAIRED_DROP1_ROOT:-data/shapenet_unpaired_cache_v2_20260306_visocc_drop1}"
UNPAIRED_PC33_ROOT="${UNPAIRED_PC33_ROOT:-data/shapenet_unpaired_cache_v2_pc33_mesh33_udf33_visocc}"
UNPAIRED_M50U50_ROOT="${UNPAIRED_M50U50_ROOT:-data/shapenet_unpaired_cache_v2_mesh50_udf50_visocc}"

MIX_BASE="${MIX_BASE:-nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_pc33_mesh33_udf33_visocc_base.yaml}"
MIX_VISOCC="${MIX_VISOCC:-nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_pc33_mesh33_udf33_visocc.yaml}"

MAX_STEPS="${MAX_STEPS:-2000}"
BATCH="${BATCH:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
MAX_TRAIN_SHAPES="${MAX_TRAIN_SHAPES:-64}"
MAX_EVAL_SHAPES="${MAX_EVAL_SHAPES:-64}"
N_CTX_POINTS="${N_CTX_POINTS:-1024}"
N_QUERY="${N_QUERY:-1024}"
CHUNK_N_QUERY="${CHUNK_N_QUERY:-1024}"

mkdir -p "${LOG_ROOT}" "${PRETRAIN_LOG_ROOT}" "${CPAC_LOG_ROOT}" "${RESULTS_ROOT}" "${RUNS_ROOT}"

log() {
  printf "[%s] %s\n" "$(date '+%F %T')" "$*" | tee -a "${PIPELINE_LOG}"
}

run_step() {
  local name="$1"
  shift
  local log_path="${LOG_ROOT}/${name}.log"
  log "start ${name}"
  (
    set -euo pipefail
    "$@"
  ) 2>&1 | tee "${log_path}"
  local rc=${PIPESTATUS[0]}
  if [[ "${rc}" -ne 0 ]]; then
    log "fail ${name} rc=${rc} log=${log_path}"
    exit "${rc}"
  fi
  log "done ${name} log=${log_path}"
}

run_bg() {
  local name="$1"
  shift
  local log_path="${LOG_ROOT}/${name}.log"
  log "start-bg ${name}"
  (
    set -euo pipefail
    "$@"
  ) > >(tee "${log_path}") 2>&1 &
  echo $!
}

extract_pretrain_diag() {
  local log_path="$1"
  "${PYTHON_BIN}" - "${log_path}" <<'PY'
import json, re, sys
path = sys.argv[1]
pat_q = re.compile(r"recon_lift_q=([+-]?[0-9]*\.?[0-9]+)")
pat_a = re.compile(r"recon_lift_a=([+-]?[0-9]*\.?[0-9]+)")
q = None
a = None
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        mq = pat_q.search(line)
        ma = pat_a.search(line)
        if mq:
            q = float(mq.group(1))
        if ma:
            a = float(ma.group(1))
print(json.dumps({"recon_lift_q": q, "recon_lift_a": a}))
PY
}

extract_cpac_metrics() {
  local json_path="$1"
  "${PYTHON_BIN}" - "${json_path}" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)
m = payload.get("metrics", {})
print(json.dumps({
    "mae": m.get("mae"),
    "rmse": m.get("rmse"),
    "iou@0.01": m.get("iou@0.01"),
}))
PY
}

run_step "build_source_cache" env \
  SHAPENET_ROOT="${SHAPENET_ROOT}" \
  OUT_ROOT="${SOURCE_CACHE}" \
  WORKERS="${BUILD_WORKERS:-32}" \
  N_RAYS="${N_RAYS:-0}" \
  MESH_VIS_ENABLE="1" \
  MESH_VIS_DIRS="${MESH_VIS_DIRS:-16}" \
  MESH_VIS_MAX_T="${MESH_VIS_MAX_T:-0.25}" \
  MESH_VIS_EPS="${MESH_VIS_EPS:-1e-3}" \
  SKIP_EXISTING="${SKIP_EXISTING:-0}" \
  AUGMENT_EXISTING="${AUGMENT_EXISTING:-0}" \
  bash scripts/preprocess/preprocess_shapenet_v2.sh

run_step "split_base" env \
  CACHE_ROOT="${SOURCE_CACHE}" \
  OUT_JSON="${SPLIT_JSON_BASE}" \
  RATIOS="0.34:0.33:0.33" \
  ALLOW_EMPTY_SPLITS="0" \
  bash scripts/preprocess/make_shapenet_unpaired_split.sh

run_step "materialize_base" env \
  SRC_CACHE_ROOT="${SOURCE_CACHE}" \
  SPLIT_JSON="${SPLIT_JSON_BASE}" \
  OUT_ROOT="${UNPAIRED_ROOT}" \
  LINK_MODE="symlink" \
  OVERWRITE="1" \
  bash scripts/preprocess/preprocess_shapenet_unpaired.sh

run_step "materialize_drop1" env \
  SRC_CACHE_ROOT="${SOURCE_CACHE}" \
  SPLIT_JSON="${SPLIT_JSON_BASE}" \
  OUT_ROOT="${UNPAIRED_DROP1_ROOT}" \
  LINK_MODE="symlink" \
  OVERWRITE="1" \
  bash scripts/preprocess/preprocess_shapenet_unpaired.sh

run_step "split_pc33" env \
  CACHE_ROOT="${SOURCE_CACHE}" \
  OUT_JSON="${SPLIT_JSON_PC33}" \
  RATIOS="0.33:0.33:0.34" \
  ALLOW_EMPTY_SPLITS="0" \
  bash scripts/preprocess/make_shapenet_unpaired_split.sh

run_step "materialize_pc33" env \
  SRC_CACHE_ROOT="${SOURCE_CACHE}" \
  SPLIT_JSON="${SPLIT_JSON_PC33}" \
  OUT_ROOT="${UNPAIRED_PC33_ROOT}" \
  LINK_MODE="symlink" \
  OVERWRITE="1" \
  bash scripts/preprocess/preprocess_shapenet_unpaired.sh

run_step "split_mesh50udf50" env \
  CACHE_ROOT="${SOURCE_CACHE}" \
  OUT_JSON="${SPLIT_JSON_M50U50}" \
  RATIOS="0.5:0.0:0.5" \
  ALLOW_EMPTY_SPLITS="1" \
  bash scripts/preprocess/make_shapenet_unpaired_split.sh

run_step "materialize_mesh50udf50" env \
  SRC_CACHE_ROOT="${SOURCE_CACHE}" \
  SPLIT_JSON="${SPLIT_JSON_M50U50}" \
  OUT_ROOT="${UNPAIRED_M50U50_ROOT}" \
  LINK_MODE="symlink" \
  OVERWRITE="1" \
  bash scripts/preprocess/preprocess_shapenet_unpaired.sh

BASE_RUN_TAG="local_l000a_visocc_base_pc33_s2000"
VISOCC_RUN_TAG="local_l000a_visocc_pc33_s2000"
BASE_SAVE_DIR="${RUNS_ROOT}/l000a_visocc_base_pc33_s2000"
VISOCC_SAVE_DIR="${RUNS_ROOT}/l000a_visocc_pc33_s2000"

pid_base="$(run_bg pretrain_base env CUDA_VISIBLE_DEVICES="${GPU_BASE}" \
  MIX_CONFIG="${MIX_BASE}" RUN_TAG="${BASE_RUN_TAG}" SAVE_DIR="${BASE_SAVE_DIR}" \
  LOG_ROOT="${PRETRAIN_LOG_ROOT}" MAX_STEPS="${MAX_STEPS}" EPOCHS="0" SAVE_EVERY="1000" \
  BATCH="${BATCH}" NUM_WORKERS="${NUM_WORKERS}" SEED="${SEED}" \
  PRETRAIN_OBJECTIVE="recon_chamfer" RECON_LOSS_MODE="composite" \
  RECON_CHAMFER_METRIC="l2" RECON_GENERATOR_DEPTH="2" \
  USE_WANDB="1" WANDB_MODE="offline" WANDB_PROJECT="patchnepa-pretrain-local" \
  WANDB_GROUP="local_visocc_screen" WANDB_RUN_NAME="${BASE_RUN_TAG}" \
  NUM_PROCESSES="1" bash scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh)"

pid_visocc="$(run_bg pretrain_visocc env CUDA_VISIBLE_DEVICES="${GPU_VISOCC}" \
  MIX_CONFIG="${MIX_VISOCC}" RUN_TAG="${VISOCC_RUN_TAG}" SAVE_DIR="${VISOCC_SAVE_DIR}" \
  LOG_ROOT="${PRETRAIN_LOG_ROOT}" MAX_STEPS="${MAX_STEPS}" EPOCHS="0" SAVE_EVERY="1000" \
  BATCH="${BATCH}" NUM_WORKERS="${NUM_WORKERS}" SEED="${SEED}" \
  PRETRAIN_OBJECTIVE="recon_chamfer" RECON_LOSS_MODE="composite" \
  RECON_CHAMFER_METRIC="l2" RECON_GENERATOR_DEPTH="2" \
  USE_WANDB="1" WANDB_MODE="offline" WANDB_PROJECT="patchnepa-pretrain-local" \
  WANDB_GROUP="local_visocc_screen" WANDB_RUN_NAME="${VISOCC_RUN_TAG}" \
  NUM_PROCESSES="1" bash scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh)"

wait "${pid_base}"
wait "${pid_visocc}"
log "short pretrain arms finished"

BASE_CKPT="${BASE_SAVE_DIR}/ckpt_final.pt"
VISOCC_CKPT="${VISOCC_SAVE_DIR}/ckpt_final.pt"
[[ -f "${BASE_CKPT}" ]] || { log "missing baseline ckpt: ${BASE_CKPT}"; exit 2; }
[[ -f "${VISOCC_CKPT}" ]] || { log "missing visocc ckpt: ${VISOCC_CKPT}"; exit 2; }

BASE_CPAC_JSON="${RESULTS_ROOT}/l000b_visocc_base_pc33.json"
VISOCC_CPAC_JSON="${RESULTS_ROOT}/l000b_visocc_pc33.json"

pid_cpac_base="$(run_bg cpac_base env CUDA_VISIBLE_DEVICES="${GPU_BASE}" \
  CKPT="${BASE_CKPT}" DATA_ROOT="${UNPAIRED_PC33_ROOT}" RUN_TAG="local_l000b_visocc_base_pc33_cpac" \
  LOG_ROOT="${CPAC_LOG_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" OUT_JSON="${BASE_CPAC_JSON}" \
  HEAD_TRAIN_SPLIT="train_udf" EVAL_SPLIT="eval" REP_SOURCE="h" RIDGE_ALPHA="1.0" TAU="0.01" \
  MINI_CPAC="1" MAX_TRAIN_SHAPES="${MAX_TRAIN_SHAPES}" MAX_EVAL_SHAPES="${MAX_EVAL_SHAPES}" \
  N_CTX_POINTS="${N_CTX_POINTS}" N_QUERY="${N_QUERY}" CHUNK_N_QUERY="${CHUNK_N_QUERY}" \
  CONTEXT_PRIMITIVE="pc" QUERY_PRIMITIVE="udf" SURF_XYZ_KEY="pc_xyz" QRY_XYZ_KEY="udf_qry_xyz" \
  QRY_DIST_KEY="udf_qry_dist" bash scripts/analysis/nepa3d_cpac_udf_patchnepa_qf.sh)"

pid_cpac_visocc="$(run_bg cpac_visocc env CUDA_VISIBLE_DEVICES="${GPU_VISOCC}" \
  CKPT="${VISOCC_CKPT}" DATA_ROOT="${UNPAIRED_PC33_ROOT}" RUN_TAG="local_l000b_visocc_pc33_cpac" \
  LOG_ROOT="${CPAC_LOG_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" OUT_JSON="${VISOCC_CPAC_JSON}" \
  HEAD_TRAIN_SPLIT="train_udf" EVAL_SPLIT="eval" REP_SOURCE="h" RIDGE_ALPHA="1.0" TAU="0.01" \
  MINI_CPAC="1" MAX_TRAIN_SHAPES="${MAX_TRAIN_SHAPES}" MAX_EVAL_SHAPES="${MAX_EVAL_SHAPES}" \
  N_CTX_POINTS="${N_CTX_POINTS}" N_QUERY="${N_QUERY}" CHUNK_N_QUERY="${CHUNK_N_QUERY}" \
  CONTEXT_PRIMITIVE="pc" QUERY_PRIMITIVE="udf" SURF_XYZ_KEY="pc_xyz" QRY_XYZ_KEY="udf_qry_xyz" \
  QRY_DIST_KEY="udf_qry_dist" bash scripts/analysis/nepa3d_cpac_udf_patchnepa_qf.sh)"

wait "${pid_cpac_base}"
wait "${pid_cpac_visocc}"
log "mini-CPAC arms finished"

base_diag="$(extract_pretrain_diag "${PRETRAIN_LOG_ROOT}/${BASE_RUN_TAG}.log")"
visocc_diag="$(extract_pretrain_diag "${PRETRAIN_LOG_ROOT}/${VISOCC_RUN_TAG}.log")"
base_metrics="$(extract_cpac_metrics "${BASE_CPAC_JSON}")"
visocc_metrics="$(extract_cpac_metrics "${VISOCC_CPAC_JSON}")"

"${PYTHON_BIN}" - "${base_diag}" "${visocc_diag}" "${base_metrics}" "${visocc_metrics}" "${DECISION_JSON}" <<'PY'
import json, sys
base_diag = json.loads(sys.argv[1])
vis_diag = json.loads(sys.argv[2])
base_m = json.loads(sys.argv[3])
vis_m = json.loads(sys.argv[4])
out_path = sys.argv[5]

healthy = (
    (vis_diag.get("recon_lift_q") is not None) and
    (vis_diag.get("recon_lift_a") is not None) and
    (float(vis_diag["recon_lift_q"]) >= 0.0) and
    (float(vis_diag["recon_lift_a"]) > 0.0)
)
iou_base = float(base_m.get("iou@0.01") or 0.0)
iou_vis = float(vis_m.get("iou@0.01") or 0.0)
rmse_base = float(base_m.get("rmse") or 0.0)
rmse_vis = float(vis_m.get("rmse") or 0.0)
improved = ((iou_vis >= iou_base + 0.005) or (rmse_vis <= rmse_base - 0.002))
catastrophic = ((iou_vis < iou_base - 0.003) or (rmse_vis > rmse_base + 0.003))
promote = bool(healthy and improved and (not catastrophic))
payload = {
    "baseline_pretrain": base_diag,
    "visocc_pretrain": vis_diag,
    "baseline_cpac": base_m,
    "visocc_cpac": vis_m,
    "healthy": healthy,
    "improved": improved,
    "catastrophic": catastrophic,
    "promote": promote,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

promote="$("${PYTHON_BIN}" - "${DECISION_JSON}" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)
print("1" if payload.get("promote") else "0")
PY
)"

if [[ "${promote}" == "1" ]]; then
  log "visibility branch passed promotion gate; launching answer_only arm"
  ANSWER_RUN_TAG="local_l000a_visocc_answeronly_pc33_s2000"
  ANSWER_SAVE_DIR="${RUNS_ROOT}/l000a_visocc_answeronly_pc33_s2000"
  ANSWER_CPAC_JSON="${RESULTS_ROOT}/l000b_visocc_answeronly_pc33.json"

  run_step "pretrain_visocc_answeronly" env CUDA_VISIBLE_DEVICES="${GPU_OPTIONAL}" \
    MIX_CONFIG="${MIX_VISOCC}" RUN_TAG="${ANSWER_RUN_TAG}" SAVE_DIR="${ANSWER_SAVE_DIR}" \
    LOG_ROOT="${PRETRAIN_LOG_ROOT}" MAX_STEPS="${MAX_STEPS}" EPOCHS="0" SAVE_EVERY="1000" \
    BATCH="${BATCH}" NUM_WORKERS="${NUM_WORKERS}" SEED="${SEED}" \
    PRETRAIN_OBJECTIVE="recon_chamfer" RECON_LOSS_MODE="answer_only" \
    RECON_CHAMFER_METRIC="l2" RECON_GENERATOR_DEPTH="2" \
    USE_WANDB="1" WANDB_MODE="offline" WANDB_PROJECT="patchnepa-pretrain-local" \
    WANDB_GROUP="local_visocc_screen" WANDB_RUN_NAME="${ANSWER_RUN_TAG}" \
    NUM_PROCESSES="1" bash scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh

  run_step "cpac_visocc_answeronly" env CUDA_VISIBLE_DEVICES="${GPU_OPTIONAL}" \
    CKPT="${ANSWER_SAVE_DIR}/ckpt_final.pt" DATA_ROOT="${UNPAIRED_PC33_ROOT}" \
    RUN_TAG="local_l000b_visocc_answeronly_pc33_cpac" LOG_ROOT="${CPAC_LOG_ROOT}" \
    RESULTS_ROOT="${RESULTS_ROOT}" OUT_JSON="${ANSWER_CPAC_JSON}" \
    HEAD_TRAIN_SPLIT="train_udf" EVAL_SPLIT="eval" REP_SOURCE="h" RIDGE_ALPHA="1.0" TAU="0.01" \
    MINI_CPAC="1" MAX_TRAIN_SHAPES="${MAX_TRAIN_SHAPES}" MAX_EVAL_SHAPES="${MAX_EVAL_SHAPES}" \
    N_CTX_POINTS="${N_CTX_POINTS}" N_QUERY="${N_QUERY}" CHUNK_N_QUERY="${CHUNK_N_QUERY}" \
    CONTEXT_PRIMITIVE="pc" QUERY_PRIMITIVE="udf" SURF_XYZ_KEY="pc_xyz" QRY_XYZ_KEY="udf_qry_xyz" \
    QRY_DIST_KEY="udf_qry_dist" bash scripts/analysis/nepa3d_cpac_udf_patchnepa_qf.sh
else
  log "visibility branch did not pass promotion gate; answer_only arm skipped"
fi

touch "${LOG_ROOT}/done.marker"
log "all done decision_json=${DECISION_JSON}"
