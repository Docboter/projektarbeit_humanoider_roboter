#!/usr/bin/env bash
set -euo pipefail

cd /home/code/unitree_sim_isaaclab

TASK="${UNITREE_TASK:-Isaac-Move-Cylinder-G129-Dex3-Wholebody}"
ROBOT_TYPE="${ROBOT_TYPE:-g129}"
GROOT_HOST="${GROOT_HOST:-127.0.0.1}"
GROOT_PORT="${GROOT_PORT:-5555}"
GROOT_TIMEOUT_MS="${GROOT_TIMEOUT_MS:-120000}"
GROOT_ACTION_STEP="${GROOT_ACTION_STEP:-0}"
CAMERA_WIDTH="${CAMERA_WIDTH:-320}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-240}"
LIVESTREAM_TYPE="${LIVESTREAM_TYPE:-1}"
PUBLIC_IP="${PUBLIC_IP:-127.0.0.1}"
DEX_HAND_ARG="${DEX_HAND_ARG:---enable_dex3_dds}"

base_args=(
  python3 sim_main.py
  --headless
  --no_render
  --livestream_type "${LIVESTREAM_TYPE}"
  --public_ip "${PUBLIC_IP}"
  --device cuda
  --enable_cameras
  --task "${TASK}"
  --robot_type "${ROBOT_TYPE}"
  --action_source groot
  --enable_wholebody_dds
  --minimal_scene
  --camera_width "${CAMERA_WIDTH}"
  --camera_height "${CAMERA_HEIGHT}"
  --groot_host "${GROOT_HOST}"
  --groot_port "${GROOT_PORT}"
  --groot_timeout_ms "${GROOT_TIMEOUT_MS}"
  --groot_action_step "${GROOT_ACTION_STEP}"
  --prompt_fifo groot_prompt.pipe
  --prompt_path groot_prompt.txt
)

if [[ -n "${DEX_HAND_ARG}" ]]; then
  base_args+=("${DEX_HAND_ARG}")
fi

# UNITREE_EXTRA_ARGS is intentionally split by the shell so values like
# "--groot_debug --groot_dry_run" behave the same as in docker compose.
# shellcheck disable=SC2086
exec conda run --no-capture-output -n unitree_sim_env "${base_args[@]}" ${UNITREE_EXTRA_ARGS:-}
