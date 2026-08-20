#!/usr/bin/env bash
# TL;DR: Sim-Container-Entrypoint: spielt echte Dataset-Aktionen ab, Open-Loop ohne GR00T-Server.
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
#
# LIVE-Variante: LIVESTREAM=2 (privates Netz) streamt den Isaac-Sim-Viewport per WebRTC
# zum nativen Client auf dem Arbeitsrechner — statt hinterher ein MP4 zu holen. Beim
# Greif-Test ist das der nützlichste der vier Läufe, weil man die Finger beim Zugreifen
# aus jedem Winkel betrachten kann. Anleitung: docs/simulation/live-ansicht.md

set -euo pipefail

# ── Instance-Log (SSH-Zugriff) ────────────────────────────────────────────────
# Alle Ausgaben in /data/logs/entrypoint_replay.log spiegeln (zusätzlich zu stdout).
# Per SSH erreichbar: tail -f /data/logs/entrypoint_replay.log
mkdir -p "${DATA_DIR:-/data}/logs"
exec > >(tee -a "${DATA_DIR:-/data}/logs/entrypoint_replay.log") 2>&1

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m v \033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m ! \033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m!! \033[0m %s\n' "$*" >&2; }

if [[ $# -gt 0 ]]; then exec "$@"; fi

DATA_DIR="${DATA_DIR:-/data}"
GROOT_ROOT="${GROOT_ROOT:-/app/Groot-1.6}"
HF_CHECKPOINT_REPO="${HF_CHECKPOINT_REPO:-}"
ASSET_PATH="${ASSET_PATH:-}"

# LIVE-Variante (WebRTC-Viewport statt MP4) — gemeinsame Lib, s. entrypoint_sim.sh.
if [[ -r /scripts/lib_livestream.sh ]]; then
    # shellcheck source=lib_livestream.sh
    source /scripts/lib_livestream.sh
    livestream_init
else
    warn "lib_livestream.sh fehlt im Image — LIVE-Variante nicht verfügbar (headless)."
    LIVESTREAM=0
    livestream_active()    { return 1; }
    livestream_app_flags() { LS_APP_FLAGS=( --headless ); }
    livestream_video_dir() { printf '%s' "$1"; }
    livestream_banner()    { return 0; }
fi
VIDEO_DIR="$(livestream_video_dir "$DATA_DIR/sim_videos_replay")"

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
livestream_app_flags
livestream_banner
log "Starte Open-Loop-Replay ${GRASP_TEST_ARG:+(grasp-test)}"
${ISAACLAB_PATH}/isaaclab.sh -p /workspace/g1_dex3_sim/run_g1_dex3_replay.py \
    "${LS_APP_FLAGS[@]}" \
    --enable_cameras \
    $GRASP_TEST_ARG \
    --asset-path     "$ASSET_PATH" \
    --video-dir      "$VIDEO_DIR" \
    --results-file   "$DATA_DIR/sim_results_replay/results.json"

ok "Replay abgeschlossen."
if [[ -n "$VIDEO_DIR" ]]; then
    echo "  Video:      $VIDEO_DIR/replay_episode0.mp4"
else
    echo "  Video:      keines (LIVE-Variante lief; LIVE_KEEP_VIDEO=1 schreibt es zusätzlich)"
fi
echo "  Ergebnis:   $DATA_DIR/sim_results_replay/results.json"
