#!/usr/bin/env bash
set -euo pipefail

cd /workspace/unitree_sim_isaaclab

required_assets=(
  "assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd"
  "assets/objects/table_with_yellowbox.usd"
  "assets/objects/small_warehouse_digital_twin/small_warehouse_digital_twin.usd"
)

for asset in "${required_assets[@]}"; do
  if [[ ! -f "${asset}" ]]; then
    echo "ERROR: required Unitree asset is missing: ${asset}" >&2
    echo "Run ./scripts/fetch_unitree_assets.sh on the host before starting Compose." >&2
    exit 2
  fi
done

args=(
  /isaac-sim/python.sh sim_main.py
  --headless
  --device cuda
  --enable_cameras
  --task "${UNITREE_TASK:-Isaac-Stack-RgyBlock-G129-Dex3-Joint}"
  --robot_type g129
  --enable_dex3_dds
  --action_source idle
  --livestream_type "${LIVESTREAM_TYPE:-2}"
  --public_ip "${PUBLIC_IP:-127.0.0.1}"
  --camera_width "${CAMERA_WIDTH:-640}"
  --camera_height "${CAMERA_HEIGHT:-480}"
  --stats_interval "${STATS_INTERVAL:-30}"
  --episode_length_s "${EPISODE_LENGTH_S:-3600}"
)

if [[ -n "${UNITREE_SIM_EXTRA_ARGS:-}" ]]; then
  # This variable is an explicit escape hatch for additional sim_main.py flags.
  # shellcheck disable=SC2206
  extra_args=(${UNITREE_SIM_EXTRA_ARGS})
  args+=("${extra_args[@]}")
fi

echo "Starting stable Unitree scene: ${UNITREE_TASK:-Isaac-Stack-RgyBlock-G129-Dex3-Joint}"
echo "WebRTC host: ${PUBLIC_IP:-127.0.0.1}; signaling TCP 49100; media UDP 47998"
exec "${args[@]}"
