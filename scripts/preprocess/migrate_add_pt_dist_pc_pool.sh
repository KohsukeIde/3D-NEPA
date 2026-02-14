#!/bin/sh
#PBS -q rt_HC
#PBS -l select=1:ncpus=16:ngpus=0
#PBS -l walltime=04:00:00
#PBS -P gag51403
#PBS -N nepa3d_migrate_ptdistpc
#PBS -o nepa3d_migrate_ptdistpc.out
#PBS -e nepa3d_migrate_ptdistpc.err

set -eu

. /etc/profile.d/modules.sh
cd /groups/gag51403/ide/3D-NEPA

if [ -f ".venv/bin/activate" ]; then
  . .venv/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE_ROOT="${CACHE_ROOT:?set CACHE_ROOT=...}"
SPLITS="${SPLITS:-all}"
WORKERS="${WORKERS:-16}"
OVERWRITE="${OVERWRITE:-0}"
LIMIT="${LIMIT:-0}"

EXTRA=""
if [ "${OVERWRITE}" = "1" ]; then
  EXTRA="${EXTRA} --overwrite"
fi
if [ "${LIMIT}" != "0" ]; then
  EXTRA="${EXTRA} --limit ${LIMIT}"
fi

"${PYTHON_BIN}" -m nepa3d.data.migrate_add_pt_dist_pc_pool \
  --cache_root "${CACHE_ROOT}" \
  --splits "${SPLITS}" \
  --workers "${WORKERS}" \
  ${EXTRA}
