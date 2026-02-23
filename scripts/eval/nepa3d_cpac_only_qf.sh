#!/bin/bash
#PBS -l rt_QF=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N nepa3d_cpac_only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"

RUN_TAG="${RUN_TAG:?set RUN_TAG}"
CKPT="${CKPT:?set CKPT}"

UNPAIRED_CACHE_ROOT="${UNPAIRED_CACHE_ROOT:-data/shapenet_unpaired_cache_v1}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
LOG_ROOT="${LOG_ROOT:-logs/eval/abcd_cpac_only}"

CPAC_SPLIT="${CPAC_SPLIT:-eval}"
CPAC_N_CONTEXT="${CPAC_N_CONTEXT:-1024}"
CPAC_N_QUERY="${CPAC_N_QUERY:-1024}"
CPAC_MAX_SHAPES="${CPAC_MAX_SHAPES:-800}"
CPAC_MAX_LEN="${CPAC_MAX_LEN:--1}"
CPAC_HEAD_TRAIN_SPLIT="${CPAC_HEAD_TRAIN_SPLIT:-train_udf}"
CPAC_HEAD_TRAIN_BACKEND="${CPAC_HEAD_TRAIN_BACKEND:-udfgrid}"
CPAC_HEAD_TRAIN_RATIO="${CPAC_HEAD_TRAIN_RATIO:-0.2}"
CPAC_RIDGE_LAMBDA="${CPAC_RIDGE_LAMBDA:-1e-3}"
CPAC_TAU="${CPAC_TAU:-0.03}"
CPAC_EVAL_SEED="${CPAC_EVAL_SEED:-0}"

cd "${WORKDIR}"
mkdir -p "${LOG_ROOT}" "${RESULTS_ROOT}"

if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

echo "=== CPAC-ONLY JOB INFO ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "run_tag=${RUN_TAG}"
echo "ckpt=${CKPT}"
echo "python=$(which python)"
python -V || true
echo

if [[ ! -f "${CKPT}" ]]; then
  echo "[error] missing ckpt: ${CKPT}"
  exit 2
fi

if [[ ! -d "${UNPAIRED_CACHE_ROOT}" ]]; then
  echo "[error] missing unpaired cache for CPAC: ${UNPAIRED_CACHE_ROOT}"
  exit 2
fi

# Fail fast when CPAC token length is incompatible with checkpoint max_len.
python - "${CKPT}" "${CPAC_N_CONTEXT}" "${CPAC_N_QUERY}" "${CPAC_MAX_LEN}" <<'PY'
import sys
import torch

ckpt_path = sys.argv[1]
n_context = int(sys.argv[2])
n_query = int(sys.argv[3])
max_len_override = int(sys.argv[4])
ckpt = torch.load(ckpt_path, map_location="cpu")
state = ckpt["model"]
pre_args = ckpt.get("args", {})
ckpt_n_types = int(state["type_emb.weight"].shape[0])
qa_tokens = bool(pre_args.get("qa_tokens", ckpt_n_types >= 9))
add_eos = bool(pre_args.get("add_eos", ckpt_n_types >= 5))
ckpt_max_len = int(state["pos_emb"].shape[1])
max_len = ckpt_max_len if max_len_override < 0 else max_len_override
required = 1 + (2 if qa_tokens else 1) * (n_context + n_query) + (1 if add_eos else 0)
if required > max_len:
    raise SystemExit(
        f"[error] CPAC precheck failed: required_seq_len={required} > effective_max_len={max_len} "
        f"(ckpt_max_len={ckpt_max_len}, qa_tokens={int(qa_tokens)}, add_eos={int(add_eos)}, "
        f"n_context={n_context}, n_query={n_query}, max_len_override={max_len_override})."
    )
print(
    f"[ok] CPAC precheck: required_seq_len={required} <= effective_max_len={max_len} "
    f"(ckpt_max_len={ckpt_max_len}, qa_tokens={int(qa_tokens)}, add_eos={int(add_eos)})"
)
PY

CPAC_JSON="${RESULTS_ROOT}/cpac_abcd_1024_${RUN_TAG}.json"
CPAC_LOG="${LOG_ROOT}/${RUN_TAG}_cpac.log"

echo "=== CPAC ==="
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root "${UNPAIRED_CACHE_ROOT}" \
  --split "${CPAC_SPLIT}" \
  --ckpt "${CKPT}" \
  --context_backend pointcloud_noray \
  --n_context "${CPAC_N_CONTEXT}" \
  --n_query "${CPAC_N_QUERY}" \
  --max_len "${CPAC_MAX_LEN}" \
  --max_shapes "${CPAC_MAX_SHAPES}" \
  --head_train_split "${CPAC_HEAD_TRAIN_SPLIT}" \
  --head_train_backend "${CPAC_HEAD_TRAIN_BACKEND}" \
  --head_train_ratio "${CPAC_HEAD_TRAIN_RATIO}" \
  --ridge_lambda "${CPAC_RIDGE_LAMBDA}" \
  --tau "${CPAC_TAU}" \
  --eval_seed "${CPAC_EVAL_SEED}" \
  --disjoint_context_query 1 \
  --context_mode_test normal \
  --rep_source h \
  --query_source pool \
  --out_json "${CPAC_JSON}" \
  2>&1 | tee "${CPAC_LOG}"

echo "=== DONE ==="
echo "cpac_json=${CPAC_JSON}"
