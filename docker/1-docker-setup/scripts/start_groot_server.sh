#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Isaac-GR00T

MODEL_PATH="${GROOT_MODEL_PATH:-checkpoints/GR00T-N1.6-G1-PnPAppleToPlate}"
EMBODIMENT_TAG="${GROOT_EMBODIMENT_TAG:-UNITREE_G1}"
HOST="${GROOT_HOST_BIND:-0.0.0.0}"
PORT="${GROOT_PORT:-5555}"
CUDA_DEVICE="${GROOT_CUDA_DEVICE:-0}"

exec /workspace/Isaac-GR00T/.venv/bin/python gr00t/eval/run_gr00t_server.py \
  --model-path "${MODEL_PATH}" \
  --embodiment-tag "${EMBODIMENT_TAG}" \
  --device "cuda:${CUDA_DEVICE}" \
  --host "${HOST}" \
  --port "${PORT}"
