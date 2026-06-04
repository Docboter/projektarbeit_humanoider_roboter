#!/usr/bin/env bash
# entrypoint_replay.sh — Open-Loop-Dataset-Replay (Diagnose Config vs. Training).
#
# Eigenständig & additiv: nutzt KEINEN GR00T-Server und KEIN Modell. Spielt die echten
# aufgezeichneten Dataset-Aktionen (gebündelte Episode 0) in dieselbe Isaac-Lab-Env.
# Kann parallel zum normalen Modell-Eval (entrypoint_sim.sh) in einer zweiten Instanz laufen.
# Ausgaben getrennt: /data/sim_videos_replay, /data/sim_results_replay.
#
# Nutzung auf vast.ai: gleiches Image, aber Entrypoint überschreiben:
#   Docker Options:  --ipc=host --shm-size=16g -p 22
#   Command/Entrypoint:  bash /scripts/entrypoint_replay.sh
# Env-Vars: HF_TOKEN + HF_CHECKPOINT_REPO (für das USD-Asset) ODER ASSET_PATH direkt.

set -euo pipefail

# ── Instance-Log (SSH-Zugriff) ────────────────────────────────────────────────
# Alle Ausgaben in /data/logs/entrypoint_replay.log spiegeln (zusätzlich zu stdout).
# Per SSH erreichbar: tail -f /data/logs/entrypoint_replay.log
mkdir -p "${DATA_DIR:-/data}/logs"
exec > >(tee -a "${DATA_DIR:-/data}/logs/entrypoint_replay.log") 2>&1

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m v \033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m!! \033[0m %s\n' "$*" >&2; }

if [[ $# -gt 0 ]]; then exec "$@"; fi

DATA_DIR="${DATA_DIR:-/data}"
GROOT_ROOT="${GROOT_ROOT:-/app/Groot-1.6}"
HF_CHECKPOINT_REPO="${HF_CHECKPOINT_REPO:-}"
ASSET_PATH="${ASSET_PATH:-}"

mkdir -p "$DATA_DIR/sim_videos_replay" "$DATA_DIR/sim_results_replay" "$DATA_DIR/logs"

# SSH (für vastai ssh), wie im Haupt-Entrypoint
mkdir -p /root/.ssh && chmod 700 /root/.ssh
[[ -n "${PUBLIC_KEY:-}" ]] && echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
/usr/sbin/sshd 2>/dev/null || true

log "GPU-Check"
nvidia-smi -L | sed 's/^/    /'

# Asset beschaffen: entweder ASSET_PATH gesetzt, oder Checkpoint-Repo (enthält g1_dex3.usd) laden
if [[ -z "$ASSET_PATH" && -n "$HF_CHECKPOINT_REPO" ]]; then
    CKPT_DIR="$DATA_DIR/checkpoints/$(basename "$HF_CHECKPOINT_REPO")"
    if [[ ! -f "$CKPT_DIR/g1_dex3.usd" ]]; then
        log "Lade USD-Asset von HuggingFace: $HF_CHECKPOINT_REPO"
        "$GROOT_ROOT/.venv/bin/python" - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${HF_CHECKPOINT_REPO}", local_dir="${CKPT_DIR}",
                  token="${HF_TOKEN:-None}")
EOF
    fi
    ASSET_PATH="$CKPT_DIR/g1_dex3.usd"
fi

if [[ -z "$ASSET_PATH" || ! -f "$ASSET_PATH" ]]; then
    err "Kein USD-Asset gefunden. ASSET_PATH setzen oder HF_TOKEN+HF_CHECKPOINT_REPO angeben."
    exit 1
fi
ok "Asset: $ASSET_PATH"

# Replay starten — VIRTUAL_ENV unset, damit isaaclab.sh sein eigenes Python nutzt.
unset VIRTUAL_ENV
export PYTHONUNBUFFERED=1
# GRASP_TEST=1 → Würfel exakt an die aufgezeichneten Greifpunkte (kontrollierter Greif-Physik-Test).
# sonst zufällige Würfelpositionen im Greifraum.
GRASP_TEST_ARG=""
[[ "${GRASP_TEST:-0}" == "1" ]] && GRASP_TEST_ARG="--grasp-test"
log "Starte Open-Loop-Replay ${GRASP_TEST_ARG:+(grasp-test)}"
${ISAACLAB_PATH}/isaaclab.sh -p /workspace/g1_dex3_sim/run_g1_dex3_replay.py \
    --headless \
    --enable_cameras \
    $GRASP_TEST_ARG \
    --asset-path     "$ASSET_PATH" \
    --video-dir      "$DATA_DIR/sim_videos_replay" \
    --results-file   "$DATA_DIR/sim_results_replay/results.json"

ok "Replay abgeschlossen."
echo "  Video:      $DATA_DIR/sim_videos_replay/replay_episode0.mp4"
echo "  Ergebnis:   $DATA_DIR/sim_results_replay/results.json"
