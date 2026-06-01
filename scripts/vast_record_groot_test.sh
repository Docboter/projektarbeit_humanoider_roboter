#!/usr/bin/env bash
set -euo pipefail

# Run from the project root. Copies current test code to a Vast instance,
# starts GR00T if needed, records a short sim clip, and downloads outputs.

VAST_HOST="${VAST_HOST:-45.81.32.13}"
VAST_PORT="${VAST_PORT:-22924}"
VAST_USER="${VAST_USER:-root}"
PROMPT="${PROMPT:-pick up the cylinder}"
RECORD_NAME="${RECORD_NAME:-groot-dryrun}"
RECORD_FRAMES="${RECORD_FRAMES:-1200}"
RECORD_EVERY="${RECORD_EVERY:-5}"
RECORD_FPS="${RECORD_FPS:-20}"
LOCAL_OUT="${LOCAL_OUT:-vast-recordings/${RECORD_NAME}}"
REMOTE_SIM_DIR="/home/code/unitree_sim_isaaclab"
REMOTE_RECORD_DIR="/workspace/recordings/${RECORD_NAME}"
REMOTE_LOG="/workspace/recordings/${RECORD_NAME}.log"

SSH=(ssh -p "${VAST_PORT}" "${VAST_USER}@${VAST_HOST}")
SCP=(scp -P "${VAST_PORT}")

echo "==> Copy current test files to Vast"
"${SCP[@]}" \
  repos/unitree_sim_isaaclab/sim_main.py \
  "${VAST_USER}@${VAST_HOST}:${REMOTE_SIM_DIR}/sim_main.py"
"${SCP[@]}" \
  repos/unitree_sim_isaaclab/tools/test_groot_server_action.py \
  "${VAST_USER}@${VAST_HOST}:${REMOTE_SIM_DIR}/tools/test_groot_server_action.py"

echo "==> Prepare tmux session on Vast"
"${SSH[@]}" "tmux kill-session -t groot_test 2>/dev/null || true"
"${SSH[@]}" "mkdir -p /workspace/recordings && tmux new-session -d -s groot_test"

echo "==> Start GR00T server in tmux window 0"
"${SSH[@]}" "tmux send-keys -t groot_test:0 'CUDA_VISIBLE_DEVICES=0 start_groot_server.sh 2>&1 | tee /workspace/groot-server.log' C-m"

echo "==> Wait for GR00T readiness"
"${SSH[@]}" "bash -lc 'for i in {1..120}; do grep -q \"Server is ready and listening\" /workspace/groot-server.log 2>/dev/null && exit 0; sleep 5; done; tail -80 /workspace/groot-server.log; exit 1'"

echo "==> Smoke-test GR00T action endpoint"
"${SSH[@]}" "cd ${REMOTE_SIM_DIR} && conda run --no-capture-output -n unitree_sim_env python3 tools/test_groot_server_action.py --host 127.0.0.1 --port 5555 --timeout-ms 120000 --prompt '${PROMPT}'"

echo "==> Start sim recording in tmux window 1"
"${SSH[@]}" "tmux new-window -t groot_test -n sim"
"${SSH[@]}" "tmux send-keys -t groot_test:sim 'cd ${REMOTE_SIM_DIR} && mkdir -p /workspace/recordings && rm -rf ${REMOTE_RECORD_DIR} && export UNITREE_EXTRA_ARGS=\"--groot_debug --groot_dry_run --record_camera front_camera --record_dir ${REMOTE_RECORD_DIR} --record_every ${RECORD_EVERY} --record_max_frames ${RECORD_FRAMES} --record_fps ${RECORD_FPS}\" && CUDA_VISIBLE_DEVICES=0 start_unitree_sim.sh 2>&1 | tee ${REMOTE_LOG}' C-m"

echo "==> Wait for recorder to start, then send prompt"
"${SSH[@]}" "bash -lc 'for i in {1..120}; do grep -q \"\\[recorder\\] wrote 1 frames\" ${REMOTE_LOG} 2>/dev/null && exit 0; sleep 5; done; tail -120 ${REMOTE_LOG}; exit 1'"
"${SSH[@]}" "cd ${REMOTE_SIM_DIR} && printf '%s\n' '${PROMPT}' > groot_prompt.txt"

echo "==> Wait for MP4"
"${SSH[@]}" "bash -lc 'for i in {1..180}; do test -s ${REMOTE_RECORD_DIR}/front_camera.mp4 && exit 0; sleep 5; done; tail -160 ${REMOTE_LOG}; exit 1'"

echo "==> Download outputs to ${LOCAL_OUT}"
mkdir -p "${LOCAL_OUT}"
"${SCP[@]}" "${VAST_USER}@${VAST_HOST}:${REMOTE_RECORD_DIR}/front_camera.mp4" "${LOCAL_OUT}/"
"${SCP[@]}" "${VAST_USER}@${VAST_HOST}:${REMOTE_LOG}" "${LOCAL_OUT}/"

echo "==> Done"
echo "Video: ${LOCAL_OUT}/front_camera.mp4"
echo "Log:   ${LOCAL_OUT}/${RECORD_NAME}.log"
