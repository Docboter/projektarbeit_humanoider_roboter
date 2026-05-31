#!/usr/bin/env bash
# entrypoint_sim.sh — Sim-Container-Entrypoint für vast.ai
#
# Startet autonom:
#   1. GPU-Check
#   2. Checkpoint prüfen / von HuggingFace herunterladen
#   3. GR00T-Policy-Server im Hintergrund starten
#   4. Auf Server-Bereitschaft warten
#   5. Isaac-Lab-Sim-Client starten (Vordergrund)
#
# Pflicht-Env-Vars:
#   CHECKPOINT_PATH      — Pfad zum feingetunten Checkpoint-Verzeichnis
#                          (z. B. /data/checkpoints/checkpoint-3000)
#                          ODER HF_CHECKPOINT_REPO setzen für automatischen Download.
#
# Optionale Env-Vars (Defaults im Dockerfile gesetzt):
#   HF_TOKEN             — HuggingFace-Token (Pflicht wenn HF_CHECKPOINT_REPO gesetzt)
#   HF_CHECKPOINT_REPO   — HF-Modell-Repo für Checkpoint-Download (z. B. user/my-model)
#   NUM_EPISODES         — Anzahl Eval-Episoden           (default 20)
#   EXECUTION_HORIZON    — Steps pro Action-Chunk          (default 8)
#   TASK_DESCRIPTION     — Language-Prompt                 (default "stack the blocks")
#   ZMQ_PORT             — ZMQ-Port für Server-Kommunikation (default 5555)
#   ASSET_PATH           — Pfad zum g1_dex3.usd            (default /workspace/assets/g1_dex3.usd)
#   NO_FLASH_ATTN        — 0=flash-attn an, 1=aus          (default 0)
#   SKIP_DOWNLOAD        — 1=Checkpoint-Download überspringen (default 0)
#   SHELL_ON_ERROR       — 1=bei Fehler in Shell fallen    (default 0)
#
# Auf vast.ai:
#   Image:          lucam03/projekt-humanoider-roboter-sim-vastai:latest
#   GPU:            RTX 3090 / RTX 4090 / A6000 (≥24 GB, Ampere+, RT-Cores!)
#   Docker Options: --ipc=host --shm-size=16g
#   Disk:           Checkpoint-Volume nach /data mounten
#   Environment:    CHECKPOINT_PATH=/data/checkpoints/checkpoint-3000

set -euo pipefail

# ── Logging ───────────────────────────────────────────────────────────────────
log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m v \033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m ! \033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m!! \033[0m %s\n' "$*" >&2; }

trap_err() {
    err "Entrypoint mit Fehler beendet (Zeile $1)."
    if [[ "${SHELL_ON_ERROR:-0}" == "1" ]]; then
        warn "SHELL_ON_ERROR=1 — falle in interaktive Shell zur Diagnose."
        exec /bin/bash
    fi
    exit 1
}
trap 'trap_err $LINENO' ERR

# Argumente durchreichen (z. B. 'docker run … bash')
if [[ $# -gt 0 ]]; then
    log "Entrypoint: führe übergebenen Befehl aus: $*"
    exec "$@"
fi

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "\033[1;35m╔══════════════════════════════════════════════════════════════════╗\033[0m"
echo -e "\033[1;35m║   GR00T N1.6 Closed-Loop Sim — vast.ai Container                  ║\033[0m"
echo -e "\033[1;35m╚══════════════════════════════════════════════════════════════════╝\033[0m"
echo ""

# ── Konfiguration ─────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/data}"
GROOT_ROOT="${GROOT_ROOT:-/app/Groot-1.6}"
ZMQ_PORT="${ZMQ_PORT:-5555}"
NUM_EPISODES="${NUM_EPISODES:-20}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-8}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-stack the blocks}"
ASSET_PATH="${ASSET_PATH:-/workspace/assets/g1_dex3.usd}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$DATA_DIR/checkpoints}"
HF_CHECKPOINT_REPO="${HF_CHECKPOINT_REPO:-}"
NO_FLASH_ATTN="${NO_FLASH_ATTN:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

mkdir -p "$DATA_DIR/sim_videos" "$DATA_DIR/sim_results" "$DATA_DIR/logs"

# ── GPU-Check ──────────────────────────────────────────────────────────────────
log "GPU-Check"
if ! nvidia-smi -L &>/dev/null; then
    err "Keine GPU sichtbar. Container ohne --gpus all gestartet?"
    exit 1
fi
nvidia-smi -L | sed 's/^/    /'
ok "GPU verfügbar"
echo ""

# ── HuggingFace-Token ──────────────────────────────────────────────────────────
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    ok "HF_TOKEN gesetzt"
elif [[ "$SKIP_DOWNLOAD" != "1" && -n "$HF_CHECKPOINT_REPO" ]]; then
    err "HF_TOKEN ist Pflicht wenn HF_CHECKPOINT_REPO gesetzt und SKIP_DOWNLOAD=0."
    exit 1
fi
echo ""

# ── Checkpoint prüfen / herunterladen ─────────────────────────────────────────
log "Schritt 1/3 — Checkpoint"

if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    warn "SKIP_DOWNLOAD=1 — Download übersprungen."
    if [[ ! -d "$CHECKPOINT_PATH" ]]; then
        err "CHECKPOINT_PATH '$CHECKPOINT_PATH' existiert nicht."
        exit 1
    fi
    ok "Checkpoint: $CHECKPOINT_PATH"
elif [[ -n "$HF_CHECKPOINT_REPO" ]]; then
    CHECKPOINT_PATH="$DATA_DIR/checkpoints/$(basename "$HF_CHECKPOINT_REPO")"
    if [[ -d "$CHECKPOINT_PATH" && -n "$(ls -A "$CHECKPOINT_PATH" 2>/dev/null)" ]]; then
        ok "Checkpoint bereits vorhanden: $CHECKPOINT_PATH"
    else
        log "Lade Checkpoint von HuggingFace: $HF_CHECKPOINT_REPO"
        "$GROOT_ROOT/.venv/bin/python" - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${HF_CHECKPOINT_REPO}",
    local_dir="${CHECKPOINT_PATH}",
    token="${HF_TOKEN:-None}",
)
print("Download abgeschlossen.")
EOF
        ok "Checkpoint heruntergeladen: $CHECKPOINT_PATH"
    fi
else
    if [[ ! -d "$CHECKPOINT_PATH" ]] || [[ -z "$(ls -A "$CHECKPOINT_PATH" 2>/dev/null)" ]]; then
        err "CHECKPOINT_PATH '$CHECKPOINT_PATH' ist leer oder existiert nicht."
        err ""
        err "Optionen:"
        err "  1. Volume mit Checkpoint nach /data mounten und CHECKPOINT_PATH setzen:"
        err "       -v /host/checkpoint:/data/checkpoints"
        err "       -e CHECKPOINT_PATH=/data/checkpoints/checkpoint-3000"
        err "  2. Checkpoint von HuggingFace Hub herunterladen:"
        err "       -e HF_TOKEN=hf_... -e HF_CHECKPOINT_REPO=user/my-model"
        exit 1
    fi
    ok "Checkpoint gefunden: $CHECKPOINT_PATH"
fi
echo ""

# ── GR00T-Policy-Server starten ───────────────────────────────────────────────
log "Schritt 2/3 — GR00T-Policy-Server (Port $ZMQ_PORT)"

GROOT_SERVER_LOG="$DATA_DIR/logs/groot_server.log"

FLASH_ATTN_ARG=""
[[ "$NO_FLASH_ATTN" == "1" ]] && FLASH_ATTN_ARG="--no-flash-attn"

"$GROOT_ROOT/.venv/bin/python" "$GROOT_ROOT/gr00t/eval/run_gr00t_server.py" \
    --model-path "$CHECKPOINT_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --use-sim-policy-wrapper \
    $FLASH_ATTN_ARG \
    --port "$ZMQ_PORT" \
    > "$GROOT_SERVER_LOG" 2>&1 &
GROOT_SERVER_PID=$!
ok "GR00T-Server gestartet (PID $GROOT_SERVER_PID)"
ok "Server-Log: $GROOT_SERVER_LOG"

cleanup() {
    warn "Stoppe GR00T-Server (PID $GROOT_SERVER_PID) …"
    kill "$GROOT_SERVER_PID" 2>/dev/null || true
    wait "$GROOT_SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Auf TCP-Port warten
echo "    Warte auf Server-Bereitschaft (Port $ZMQ_PORT) …"
MAX_WAIT=300
WAITED=0
while ! timeout 1 bash -c "echo > /dev/tcp/localhost/$ZMQ_PORT" 2>/dev/null; do
    # Server-Prozess noch am Leben?
    if ! kill -0 "$GROOT_SERVER_PID" 2>/dev/null; then
        err "GR00T-Server-Prozess ist unerwartet beendet. Log:"
        tail -20 "$GROOT_SERVER_LOG" >&2
        exit 1
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        err "Server nicht bereit nach ${MAX_WAIT}s. Letzte Log-Zeilen:"
        tail -20 "$GROOT_SERVER_LOG" >&2
        exit 1
    fi
    printf "    … %ds\n" "$WAITED"
done
ok "Server bereit (nach ${WAITED}s)"
echo ""

# ── Isaac-Lab-Sim-Client starten ──────────────────────────────────────────────
log "Schritt 3/3 — Isaac-Lab-Sim-Client"
printf "    %-22s %s\n" "Server:"         "tcp://localhost:$ZMQ_PORT"
printf "    %-22s %s\n" "Episoden:"       "$NUM_EPISODES"
printf "    %-22s %s\n" "Exec-Horizon:"   "$EXECUTION_HORIZON"
printf "    %-22s %s\n" "Task:"           "$TASK_DESCRIPTION"
printf "    %-22s %s\n" "Checkpoint:"     "$CHECKPOINT_PATH"
printf "    %-22s %s\n" "Asset:"          "$ASSET_PATH"
printf "    %-22s %s\n" "Videos:"         "$DATA_DIR/sim_videos"
echo ""

${ISAACLAB_PATH}/isaaclab.sh -p /workspace/g1_dex3_sim/run_g1_dex3_sim_eval.py \
    --headless \
    --enable_cameras \
    --server "tcp://localhost:$ZMQ_PORT" \
    --num-episodes   "$NUM_EPISODES" \
    --execution-horizon "$EXECUTION_HORIZON" \
    --task-description  "$TASK_DESCRIPTION" \
    --video-dir      "$DATA_DIR/sim_videos" \
    --results-file   "$DATA_DIR/sim_results/results.json" \
    --asset-path     "$ASSET_PATH" \
    --ping-retries   20

ok "Sim-Eval abgeschlossen."
echo ""
echo "  Ergebnisse: $DATA_DIR/sim_results/results.json"
echo "  Videos:     $DATA_DIR/sim_videos/"
echo ""
echo "  Daten sichern (vom Host):"
echo "    docker cp <container_id>:/data/sim_results ./sim_results"
echo "    docker cp <container_id>:/data/sim_videos  ./sim_videos"
echo ""
