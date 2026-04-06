#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

configs=(
  "cfgs/geopcp/pcp_worldvis_base_100ep.yaml"
  "cfgs/geopcp/geopcp_worldvis_base_normal_100ep.yaml"
  "cfgs/geopcp/geopcp_worldvis_base_normal_thickness_100ep.yaml"
)
names=(
  "pcp_worldvis_base_100ep"
  "geopcp_worldvis_base_normal_100ep"
  "geopcp_worldvis_base_normal_thickness_100ep"
)

for i in "${!configs[@]}"; do
  CONFIG="${configs[$i]}" \
  EXP_NAME="${names[$i]}" \
  RUN_TAG="${names[$i]}" \
  FOREGROUND="${FOREGROUND:-1}" \
  "${ROOT_DIR}/scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_pretrain_local.sh"
done
