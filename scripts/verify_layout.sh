#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$(pwd)}"

required_paths=(
  "${REPO_DIR}/repos/Isaac-GR00T"
  "${REPO_DIR}/repos/Isaac-GR00T/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate"
  "${REPO_DIR}/repos/unitree_sim_isaaclab"
  "${REPO_DIR}/docker/docker-compose.groot-unitree.yml"
  "${REPO_DIR}/docker/.env.groot-unitree"
)

for path in "${required_paths[@]}"; do
  if [[ -e "${path}" ]]; then
    echo "ok: ${path}"
  else
    echo "missing: ${path}" >&2
    exit 1
  fi
done

echo "Layout looks ready."
