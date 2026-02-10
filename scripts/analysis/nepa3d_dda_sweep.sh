#!/bin/sh
#PBS -q rt_HC
#PBS -l select=1:ncpus=16:mem=64gb
#PBS -l walltime=08:00:00
#PBS -P gag51403
#PBS -N nepa3d_dda_sweep
#PBS -o nepa3d_dda_sweep.out
#PBS -e nepa3d_dda_sweep.err

set -eu

. /etc/profile.d/modules.sh

cd /groups/gag51403/ide/3D-NEPA
if [ -f ".venv/bin/activate" ]; then
  . .venv/bin/activate
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-python}"
MODELNET_ROOT="${MODELNET_ROOT:-data/ModelNet40}"
OUT_BASE="${OUT_BASE:-data/modelnet40_cache_sweep}"
SPLIT="${SPLIT:-test}"
PC_GRIDS="${PC_GRIDS:-32 64 128}"
PC_DILATES="${PC_DILATES:-0 1 2}"
RAY_SUBSAMPLE="${RAY_SUBSAMPLE:-0}"
RESERVOIR="${RESERVOIR:-200000}"
SEED="${SEED:-0}"

mkdir -p results/dda_sweep
SUMMARY_CSV="results/dda_sweep/summary_${SPLIT}.csv"
echo "cache_root,pc_grid,pc_dilate,split,hit_acc,precision,recall,f1,depth_abs_mean,depth_abs_median,depth_abs_p90,depth_abs_p99,normal_cos_mean,normal_cos_median" > "${SUMMARY_CSV}"

for g in ${PC_GRIDS}; do
  for d in ${PC_DILATES}; do
    out_root="${OUT_BASE}_g${g}_d${d}"
    echo "[sweep] preprocess split=${SPLIT} grid=${g} dilate=${d} -> ${out_root}"
    "${PYTHON_BIN}" nepa3d/data/preprocess_modelnet40.py \
      --modelnet_root "${MODELNET_ROOT}" \
      --out_root "${out_root}" \
      --split "${SPLIT}" \
      --pc_grid "${g}" \
      --pc_dilate "${d}"

    echo "[sweep] metrics split=${SPLIT} grid=${g} dilate=${d}"
    out_csv="results/dda_sweep/per_class_g${g}_d${d}_${SPLIT}.csv"
    out_plot="results/dda_sweep/figs_g${g}_d${d}_${SPLIT}"
    metrics_out=$( "${PYTHON_BIN}" -m nepa3d.analysis.dda_metrics \
      --cache_root "${out_root}" \
      --split "${SPLIT}" \
      --ray_subsample "${RAY_SUBSAMPLE}" \
      --reservoir "${RESERVOIR}" \
      --seed "${SEED}" \
      --out_csv "${out_csv}" \
      --plot_dir "${out_plot}" )
    echo "${metrics_out}"

    hit_acc=$(echo "${metrics_out}" | awk -F': ' '/^hit_acc:/{print $2}')
    precision=$(echo "${metrics_out}" | awk -F': ' '/^precision:/{print $2}')
    recall=$(echo "${metrics_out}" | awk -F': ' '/^recall:/{print $2}')
    f1=$(echo "${metrics_out}" | awk -F': ' '/^f1:/{print $2}')
    depth_abs_mean=$(echo "${metrics_out}" | awk -F': ' '/^depth_abs_mean:/{print $2}')
    depth_abs_median=$(echo "${metrics_out}" | awk -F': ' '/^depth_abs_median:/{print $2}')
    depth_abs_p90=$(echo "${metrics_out}" | awk -F': ' '/^depth_abs_p90:/{print $2}')
    depth_abs_p99=$(echo "${metrics_out}" | awk -F': ' '/^depth_abs_p99:/{print $2}')
    normal_cos_mean=$(echo "${metrics_out}" | awk -F': ' '/^normal_cos_mean:/{print $2}')
    normal_cos_median=$(echo "${metrics_out}" | awk -F': ' '/^normal_cos_median:/{print $2}')

    echo "${out_root},${g},${d},${SPLIT},${hit_acc},${precision},${recall},${f1},${depth_abs_mean},${depth_abs_median},${depth_abs_p90},${depth_abs_p99},${normal_cos_mean},${normal_cos_median}" >> "${SUMMARY_CSV}"
  done
done

echo "wrote ${SUMMARY_CSV}"
