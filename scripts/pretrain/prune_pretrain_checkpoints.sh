#!/usr/bin/env bash
set -eu

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

KEEP_EVERY="${KEEP_EVERY:-10}"
KEEP_LAST="${KEEP_LAST:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$#" -gt 0 ]; then
  TARGETS=("$@")
else
  mapfile -t TARGETS < <(find runs -maxdepth 2 -type f -name 'ckpt_ep*.pt' -printf '%h\n' | sort -u)
fi

if [ "${#TARGETS[@]}" -eq 0 ]; then
  echo "[info] no checkpoint directories found"
  exit 0
fi

for d in "${TARGETS[@]}"; do
  if [ ! -d "${d}" ]; then
    continue
  fi

  mapfile -t files < <(find "${d}" -maxdepth 1 -type f -name 'ckpt_ep*.pt' | sort)
  if [ "${#files[@]}" -le 1 ]; then
    continue
  fi

  max_ep=-1
  for f in "${files[@]}"; do
    bn="$(basename "${f}")"
    ep_str="${bn#ckpt_ep}"
    ep_str="${ep_str%.pt}"
    ep=$((10#${ep_str}))
    if [ "${ep}" -gt "${max_ep}" ]; then
      max_ep="${ep}"
    fi
  done

  keep=0
  del=0
  for f in "${files[@]}"; do
    bn="$(basename "${f}")"
    ep_str="${bn#ckpt_ep}"
    ep_str="${ep_str%.pt}"
    ep=$((10#${ep_str}))
    keep_flag=0
    if [ "$((ep % KEEP_EVERY))" -eq 0 ]; then
      keep_flag=1
    fi
    if [ "${KEEP_LAST}" = "1" ] && [ "${ep}" -eq "${max_ep}" ]; then
      keep_flag=1
    fi

    if [ "${keep_flag}" -eq 1 ]; then
      keep=$((keep + 1))
    else
      del=$((del + 1))
      if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry-run] rm ${f}"
      else
        rm -f "${f}"
      fi
    fi
  done

  echo "[done] ${d} keep=${keep} delete=${del} (keep_every=${KEEP_EVERY}, keep_last=${KEEP_LAST})"
done
