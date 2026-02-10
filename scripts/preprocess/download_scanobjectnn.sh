#!/bin/sh
#PBS -q rt_HC
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=12:00:00
#PBS -P gag51403
#PBS -N nepa3d_dl_scan
#PBS -o nepa3d_dl_scan.out
#PBS -e nepa3d_dl_scan.err

set -euo pipefail

. /etc/profile.d/modules.sh

cd /groups/gag51403/ide/3D-NEPA
if [ -f ".venv/bin/activate" ]; then
  . .venv/bin/activate
fi

OUT_DIR="${OUT_DIR:-data/ScanObjectNN}"
SCANOBJECTNN_URL="${SCANOBJECTNN_URL:-https://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip}"
export SCANOBJECTNN_URL

bash nepa3d/data/download_scanobjectnn.sh "${OUT_DIR}"
