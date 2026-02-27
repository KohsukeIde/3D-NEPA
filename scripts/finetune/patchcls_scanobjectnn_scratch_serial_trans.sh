#!/usr/bin/env bash
set -euo pipefail

# PTv3-style trans serialization baseline:
# XY-swapped Morton ("z-trans") -> contiguous chunk groups -> mini-PointNet.
#
# This wraps patchcls_scanobjectnn_scratch_serial.sh and overrides SERIAL_ORDER.

if [ -n "${WORKDIR:-}" ]; then
  ROOT_DIR="${WORKDIR}"
elif [ -n "${PBS_O_WORKDIR:-}" ]; then
  ROOT_DIR="${PBS_O_WORKDIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

export WORKDIR="${ROOT_DIR}"
export SERIAL_ORDER="${SERIAL_ORDER:-z-trans}"

exec "${ROOT_DIR}/scripts/finetune/patchcls_scanobjectnn_scratch_serial.sh"

