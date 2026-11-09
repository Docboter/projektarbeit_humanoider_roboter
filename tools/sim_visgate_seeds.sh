#!/usr/bin/env bash
# TL;DR: Host-Werkzeug — visuelles Gate (Protokoll Lauf 80, Seeds 0/1/2) für eine Liste von HF-Checkpoints, lädt fehlende mit dem HF_TOKEN des Containers.
#
# sim_visgate_seeds.sh — mehrere Checkpoints nacheinander durch `server_rl_run.sh visgate`.
#
# Aufruf:
#   tools/sim_visgate_seeds.sh CONTAINER GROOT_VERSION GPU REPO...
#   REPO = HF-Repo ohne Konto, z. B. groot-g1dex3-cotrain-v3-cube4-n17-ckpt44000
# Beispiel (Lauf 92, N1.7 auf GPU 1):
#   tools/sim_visgate_seeds.sh groot-rl-eval17 1.7 1 \
#     groot-g1dex3-cotrain-v3-cube4-n17-ckpt44000 groot-g1dex3-cotrain-v3-cube4-n17-ckpt40000
#
# Je Checkpoint:
#   1. Fehlt /data/checkpoints/<REPO ohne "groot-g1dex3-"> im Container, wird er dort
#      heruntergeladen (ohne optimizer*). Das nutzt den HF_TOKEN des Containers; der Host
#      braucht keinen (server_rl_run.sh würde sonst ein leeres -e HF_TOKEN= durchreichen).
#   2. global_step, run_id und SHA256 der Shards ins Log (Identitätsprüfung).
#   3. visgate mit VISGATE_SEED=0, 1, 2. Ergebnis: sim_results/visgate_<dir>_s<seed>_*.json.
# Kennzahl: Mittel der Seed-Mediane MSE Hände; der Median über die 4 Episoden ist der echte
# Median (Mittel der beiden mittleren), wie Simulation/scripts/visgate_summary.py.
# GROOT_VERSION immer explizit: der Host-Default ist 1.6.
# Konto: HF_ACCOUNT (Default luca-mue). Seeds: SEEDS (Default "0 1 2").

set -u
if [[ $# -lt 4 ]]; then
  sed -n '5,11p' "$0"
  exit 2
fi
C=$1 V=$2 G=$3; shift 3
HF_ACCOUNT=${HF_ACCOUNT:-luca-mue} SEEDS=${SEEDS:-0 1 2}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_DIR" || exit 1

echo "=== Start $(date '+%F %T') $C GROOT_VERSION=$V GPU=$G"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

for repo in "$@"; do
  dir=${repo#groot-g1dex3-}
  p=/data/checkpoints/$dir
  echo "=== $repo $(date +%T)"
  if ! docker exec "$C" test -f "$p/config.json"; then
    echo "--- Download $HF_ACCOUNT/$repo -> $p"
    docker exec "$C" bash -lc "huggingface-cli download '$HF_ACCOUNT/$repo' --local-dir '$p' \
      --exclude 'optimizer*' 2>&1 | tail -2" || { echo "FAILED download $repo"; continue; }
  fi
  docker exec "$C" bash -c "python3 -c \"import json;print('global_step',json.load(open('$p/trainer_state.json'))['global_step'])\"; \
    grep -o '\"run_id\"[^,]*' $p/wandb_config.json 2>/dev/null; cd $p && sha256sum *.safetensors"
  for s in $SEEDS; do
    echo "--- $repo seed $s $(date +%T)"
    MENU=0 GROOT_VERSION=$V RL_CONTAINER=$C RL_EXEC_GPU=$G HF_CHECKPOINT_REPO=$HF_ACCOUNT/$repo \
      CHECKPOINT_PATH=$p VISGATE_SEED=$s ./Simulation/server_rl_run.sh visgate || echo "FAILED $repo $s"
  done
done
echo "=== Warteschlange fertig $(date '+%F %T')"
