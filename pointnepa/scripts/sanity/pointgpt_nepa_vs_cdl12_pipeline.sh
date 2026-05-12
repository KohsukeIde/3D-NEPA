#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
exec "${WORKDIR}/scripts/local/pointgpt_nepa_vs_cdl12_pipeline.sh" "$@"
