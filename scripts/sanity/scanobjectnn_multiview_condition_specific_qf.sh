#!/bin/bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=04:00:00
#PBS -N mv2d_cond
#PBS -j oe
#PBS -o /groups/qgah50055/ide/concerto-shortcut-mvp/3D-NEPA/logs/sanity/

set -euo pipefail

cd /groups/qgah50055/ide/concerto-shortcut-mvp
source 3D-NEPA/.venv-pointgpt/bin/activate
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load cuda/12.6/12.6.2 2>/dev/null || true
  module load gcc/11.4.1 2>/dev/null || true
fi
if command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME_AUTO="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  export CUDA_HOME="${CUDA_HOME:-${CUDA_HOME_AUTO}}"
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

IMAGE_SIZE="${IMAGE_SIZE:-96}"
VIEWS="${VIEWS:-10}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-48}"
LOCAL_NOISE_SIGMA="${LOCAL_NOISE_SIGMA:-0.08}"
OUT_ROOT="3D-NEPA/results/multiview_2d/condition_specific_v${VIEWS}_s${IMAGE_SIZE}_e${EPOCHS}"
CONDITIONS="${CONDITIONS:-clean,random_keep80,random_keep50,random_keep20,random_keep10,structured_keep80,structured_keep50,structured_keep20,structured_keep10,local_jitter80,local_jitter50,local_jitter20,local_jitter10,local_replace80,local_replace50,local_replace20,local_replace10,xyz_zero}"
export OUT_ROOT IMAGE_SIZE VIEWS EPOCHS

echo "=== ScanObjectNN condition-specific multi-view recognizability ==="
date --iso-8601=seconds
hostname
echo "pbs_jobid=${PBS_JOBID:-none}"
echo "conditions=${CONDITIONS}"
echo "epochs=${EPOCHS} image_size=${IMAGE_SIZE} views=${VIEWS}"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "is_available", torch.cuda.is_available())
PY

IFS=',' read -r -a COND_ARRAY <<< "${CONDITIONS}"
for cond in "${COND_ARRAY[@]}"; do
  cond="$(echo "${cond}" | xargs)"
  [[ -n "${cond}" ]] || continue
  out="${OUT_ROOT}/${cond}"
  echo "[run] train_condition=${cond} output=${out}"
  python 3D-NEPA/multiview_2d/eval_scanobjectnn_multiview_recognizability.py \
    --image_size "${IMAGE_SIZE}" \
    --views "${VIEWS}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --train_conditions "${cond}" \
    --local_noise_sigma "${LOCAL_NOISE_SIGMA}" \
    --num_examples 0 \
    --save_individual_views 0 \
    --output_dir "${out}"
done

python - <<'PY'
from pathlib import Path
import csv
import json
import os

root = Path("3D-NEPA/results/multiview_2d")
out_root = Path(os.environ["OUT_ROOT"])
all_cond_summary = root / "scanobjectnn_recognizability_all_v10_s96_e40" / "summary.csv"
clean_summary = root / "scanobjectnn_recognizability_v10_s96_e120" / "summary.csv"
out_csv = out_root / "condition_specific_summary.csv"
out_md = out_root / "condition_specific_summary.md"

def read_rows(path):
    if not path.exists():
        return {}
    with path.open() as f:
        return {row["condition"]: float(row["acc"]) for row in csv.DictReader(f)}

all_rows = read_rows(all_cond_summary)
clean_rows = read_rows(clean_summary)
rows = []
for d in sorted([p for p in out_root.iterdir() if p.is_dir()]):
    summary = d / "summary.csv"
    if not summary.exists():
        continue
    cond_rows = read_rows(summary)
    train_cond = d.name
    rows.append({
        "condition": train_cond,
        "condition_specific_acc": cond_rows.get(train_cond, float("nan")),
        "clean_trained_acc": clean_rows.get(train_cond, float("nan")),
        "all_condition_trained_acc": all_rows.get(train_cond, float("nan")),
    })

out_root.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["condition", "condition_specific_acc", "clean_trained_acc", "all_condition_trained_acc"])
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# ScanObjectNN Condition-Specific Multi-view Recognizability",
    "",
    "Condition-specific accuracy trains the rendered-view classifier on exactly the same stress condition and evaluates heldout objects under that condition.",
    "This is the most paper-facing recognizability value: it asks whether the stressed geometry itself is class-discriminative, rather than whether a clean-trained classifier is OOD-robust.",
    "",
    "| condition | condition-specific acc | all-condition acc | clean-trained acc | claim line |",
    "|---|---:|---:|---:|---|",
]

def claim_line(cond, acc):
    if cond == "xyz_zero":
        return "null reference"
    if acc >= 0.65:
        return "strong: visually/class-discriminative"
    if acc >= 0.50:
        return "moderate: partially discriminative"
    return "weak: evidence-destroying or hard"

for row in rows:
    acc = row["condition_specific_acc"]
    lines.append(
        f"| `{row['condition']}` | {acc:.4f} | {row['all_condition_trained_acc']:.4f} | {row['clean_trained_acc']:.4f} | {claim_line(row['condition'], acc)} |"
    )

lines += [
    "",
    "## Paper-safe Use",
    "",
    "- Use `strong` rows as evidence that a perturbation preserves object identity in a 2D rendered-view proxy.",
    "- Use `moderate` rows as intermediate stress.",
    "- Do not use `weak` rows to claim that 3D models miss human-obvious evidence; use them as destructive stress references.",
]
out_md.write_text("\n".join(lines) + "\n")
print(json.dumps({"summary_csv": str(out_csv), "summary_md": str(out_md)}, indent=2))
PY

echo "[done] ${OUT_ROOT}"
