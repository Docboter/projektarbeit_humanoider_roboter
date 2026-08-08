#!/usr/bin/env bash
# entrypoint_rl.sh — RL-Fine-tuning (FPO) im kombinierten Isaac-Sim + GR00T-Container.
#
# Läuft autonom: lädt den BC-Checkpoint (HF oder lokal) und startet das
# FPO-RL-Fine-tuning des GR00T-Action-Heads (rl_finetune.py) in der Block-Stacking-Sim.
#
# WICHTIG: braucht eine RT-Core-GPU (Ampere+ mit RT-Cores: L40, RTX 4090, A6000) für
# das 4-Kamera-Rendering — A100/H100 haben KEINE RT-Cores (siehe
# docs/weiterfuehrend/reinforcement-learning-plan.md §3.5 und CLAUDE.md).
#
# Steuerung über Env-Vars (auf vast.ai im Feld "Docker options" als -e KEY=VAL):
#   HF_TOKEN              (Pflicht für HF-Download)
#   HF_CHECKPOINT_REPO   (optional) BC-Checkpoint-Repo auf HF (wird nach CHECKPOINT_PATH geladen)
#   CHECKPOINT_PATH      (default /data/checkpoints/groot-g1dex3-checkpoint) BC-Startpunkt
#   RL_NUM_ENVS          (default 16)   parallele Sim-Envs (Render-Durchsatz beachten!)
#   RL_ITERATIONS        (default 500)  RL-Iterationen
#   RL_ROLLOUT_STEPS     (default 32)   Env-Steps pro Rollout
#   RL_LR                (default 1e-5)
#   RL_KL_COEF           (default 0.1)  KL-Regularisierung gegen den BC-Checkpoint
#   RL_CLIP              (default 0.2)  PPO/FPO-Clip-Epsilon
#   RL_SAVE_EVERY        (default 100)  Checkpoint alle N Iterationen (ganzes Modell ~6 GB/Ckpt)
#   WANDB_API_KEY        (optional)     ohne → ohne W&B
#   RL_WANDB_VIDEO_EVERY (default 0)    alle N Iterationen einen Rollout als W&B-Video (0 = aus)
#   SHELL_ON_ERROR       (default 0)    bei Fehler in Shell fallen
#
# Live-Ansicht im Browser (opt-in, docs/weiterfuehrend/livestream-plan.md „Spur B"):
#   LIVE_VIEW            (default 0)    1 = MJPEG-Stream des Rollouts
#   LIVE_VIEW_PORT       (default 8900) HTTP-Port (Container-Port mappen!)
#   LIVE_VIEW_EVERY_N    (default 1)    nur jedes n-te Frame senden
#   LIVE_VIEW_CAMS       (default cam_scene) Kameras, kommagetrennt
# Diese vier werden hier BEWUSST NICHT als CLI-Flags durchgereicht: rl_finetune.py liest
# sie selbst als argparse-Defaults. Damit wirken sie auch mit einem älteren Image — auf dem
# Server ist g1_dex3_sim gemountet, dieses Skript hier dagegen fest im Image.

set -euo pipefail

# ── Instance-Log ──────────────────────────────────────────────────────────────
# Alle Ausgaben zusätzlich nach /data/logs/entrypoint_rl.log spiegeln — identisch zu
# entrypoint_sim.sh / _replay.sh / _baseline.sh, die das längst tun; der RL-Entrypoint
# war der einzige ohne. Bei einem Lauf über Stunden ist der Terminal-Scrollback als
# einzige Quelle zu wenig. Erreichbar mit: tail -f /data/logs/entrypoint_rl.log
mkdir -p "${DATA_DIR:-/data}/logs"
exec > >(tee -a "${DATA_DIR:-/data}/logs/entrypoint_rl.log") 2>&1

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m v \033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m ! \033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m!! \033[0m %s\n' "$*" >&2; }

trap_err() {
    err "RL-Entrypoint mit Fehler beendet (Zeile $1)."
    if [[ "${SHELL_ON_ERROR:-0}" == "1" ]]; then
        warn "SHELL_ON_ERROR=1 — falle in interaktive Shell."
        exec /bin/bash
    fi
    exit 1
}
trap 'trap_err $LINENO' ERR

# Manuelle Override-Befehle durchreichen.
if [[ $# -gt 0 ]]; then
    log "RL-Entrypoint: führe übergebenen Befehl aus: $*"
    exec "$@"
fi

SIM_DIR="${SIM_DIR:-/workspace/g1_dex3_sim}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/data/checkpoints/groot-g1dex3-checkpoint}"
# USD-Roboter-Asset (wie bei der Sim-Eval). Default: g1_dex3.usd im Checkpoint-Verzeichnis,
# falls per HF_CHECKPOINT_REPO mitgeladen; sonst explizit setzen.
ASSET_PATH="${ASSET_PATH:-$CHECKPOINT_PATH/g1_dex3.usd}"
RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-/data/g1_dex3_rl}"

RL_NUM_ENVS="${RL_NUM_ENVS:-16}"
RL_ITERATIONS="${RL_ITERATIONS:-500}"
RL_ROLLOUT_STEPS="${RL_ROLLOUT_STEPS:-32}"
RL_LR="${RL_LR:-1e-5}"
RL_KL_COEF="${RL_KL_COEF:-0.1}"
RL_CLIP="${RL_CLIP:-0.2}"
RL_SAVE_EVERY="${RL_SAVE_EVERY:-100}"

echo ""
echo -e "\033[1;35m╔══════════════════════════════════════════════════════════════════╗\033[0m"
echo -e "\033[1;35m║   GR00T N1.6 — RL-Fine-tuning (FPO) auf Block-Stacking           ║\033[0m"
echo -e "\033[1;35m╚══════════════════════════════════════════════════════════════════╝\033[0m"
echo ""

# ── GPU-Check (RT-Cores erforderlich) ─────────────────────────────────────────
log "GPU-Check"
if ! nvidia-smi -L &>/dev/null; then
    err "Keine GPU sichtbar. Container ohne --gpus all gestartet?"
    exit 1
fi
nvidia-smi -L | sed 's/^/    /'
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 || true)"
case "$GPU_NAME" in
    *A100*|*H100*)
        warn "GPU '$GPU_NAME' hat KEINE RT-Cores — das 4-Kamera-Rendering ist hier ineffizient/instabil."
        warn "Empfohlen: L40 / RTX 4090 / A6000 (siehe reinforcement-learning-plan.md §3.5)." ;;
esac
ok "GPU: $GPU_NAME"
echo ""

# ── Checkpoint laden (HF optional) ────────────────────────────────────────────
if [[ -n "${HF_CHECKPOINT_REPO:-}" && ! -d "$CHECKPOINT_PATH" ]]; then
    [[ -n "${HF_TOKEN:-}" ]] || { err "HF_CHECKPOINT_REPO gesetzt, aber HF_TOKEN fehlt."; exit 1; }
    log "Lade BC-Checkpoint von HF: $HF_CHECKPOINT_REPO → $CHECKPOINT_PATH"
    export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    huggingface-cli download "$HF_CHECKPOINT_REPO" --local-dir "$CHECKPOINT_PATH"
fi
[[ -d "$CHECKPOINT_PATH" ]] || { err "BC-Checkpoint fehlt: $CHECKPOINT_PATH (HF_CHECKPOINT_REPO setzen?)"; exit 1; }
ok "BC-Checkpoint: $CHECKPOINT_PATH"

# USD-Asset prüfen (RL-Env spawnt den Roboter daraus)
ASSET_FLAG=()
if [[ -f "$ASSET_PATH" ]]; then
    ASSET_FLAG=(--asset-path "$ASSET_PATH")
    ok "USD-Asset: $ASSET_PATH"
else
    warn "USD-Asset nicht unter $ASSET_PATH — Env nutzt den cfg-Default (g1_dex3_cfg.py)."
    warn "Bei Bedarf ASSET_PATH setzen oder g1_dex3.usd ins Checkpoint-Repo legen."
fi

# ── W&B ───────────────────────────────────────────────────────────────────────
WANDB_FLAG=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY WANDB_MODE="${WANDB_MODE:-offline}"
    WANDB_FLAG=(--wandb)
    ok "W&B aktiv (Modus: ${WANDB_MODE})."
else
    warn "Kein WANDB_API_KEY — RL läuft ohne W&B."
fi

# ── Live-Ansicht (rein informativ; rl_finetune.py liest LIVE_VIEW* selbst) ─────
if [[ "${LIVE_VIEW:-0}" != "0" ]]; then
    ok "Live-Ansicht an → http://<server-ip>:${LIVE_VIEW_PORT:-8900}/  (Port mappen nicht vergessen)"
else
    warn "Live-Ansicht aus (LIVE_VIEW=1 setzen, um im Browser zuzusehen)."
fi

mkdir -p "$RL_OUTPUT_DIR"

# ── RL starten ────────────────────────────────────────────────────────────────
log "Starte FPO-RL-Fine-tuning (num_envs=$RL_NUM_ENVS, iterations=$RL_ITERATIONS, save_every=$RL_SAVE_EVERY)"
cd "$SIM_DIR"
# rl_finetune.py initialisiert Isaac Sim via AppLauncher — muss im Isaac-Sim-Python laufen.
# VIRTUAL_ENV muss ungesetzt sein, sonst lenkt isaaclab.sh auf das GR00T-venv um.
unset VIRTUAL_ENV
exec "${ISAACLAB_PATH}/isaaclab.sh" -p "$SIM_DIR/rl_finetune.py" \
    --checkpoint     "$CHECKPOINT_PATH" \
    "${ASSET_FLAG[@]}" \
    --output-dir     "$RL_OUTPUT_DIR" \
    --num-envs       "$RL_NUM_ENVS" \
    --iterations     "$RL_ITERATIONS" \
    --rollout-steps  "$RL_ROLLOUT_STEPS" \
    --lr             "$RL_LR" \
    --kl-coef        "$RL_KL_COEF" \
    --clip           "$RL_CLIP" \
    --save-every     "$RL_SAVE_EVERY" \
    "${WANDB_FLAG[@]}"
