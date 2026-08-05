#!/usr/bin/env bash
# server_rl_run.sh — RL-Fine-tuning (FPO) auf einem generischen Docker-GPU-Server
#   (z. B. 2× RTX PRO 6000 Blackwell) statt auf vast.ai.
#
# Docker-Pendant zu docs/weiterfuehrend/rl-anleitung.md (dort vast.ai-Instanz-Miete).
# Fährt den kombinierten Isaac-Lab+GR00T-Container (Dockerfile.vastai) und startet
# entrypoint_rl.sh / rl_finetune.py — verfeinert den BC-Checkpoint per FPO in der
# Block-Stacking-Sim.
#   Hintergrund: docs/weiterfuehrend/reinforcement-learning-plan.md
#   Bedienung:   docs/weiterfuehrend/rl-anleitung.md
#
# ⚠️ KRITISCH — Image-Rebuild-Pflicht vor dem ersten Lauf:
#   `lucam03/projekt-humanoider-roboter-sim-vastai:latest` wurde zuletzt am 2026-06-15
#   gepusht. Commit 8e01979 (2026-07-19) hat DANACH zwei für RL zwingende Fixes in
#   Dockerfile.vastai/entrypoint_rl.sh gemacht (gr00t+flash-attn im Isaac-Sim-Python
#   3.11 installiert; Start über `isaaclab.sh -p` statt nacktem `python`). Der aktuell
#   gepushte Image-Tag hat diese Fixes NICHT. Vor dem ersten `check`/`rl`-Lauf hier:
#     ./Simulation/update_sim_image.sh --vastai      # baut Dockerfile.vastai neu + pusht
#   `preflight` unten prüft genau das (gr00t-Import im Isaac-Sim-Python) und schlägt
#   fehl, falls das gepullte Image noch der alte Stand ist.
#
# RT-CORES: Die RTX PRO 6000 Blackwell haben RT-Cores (anders als KISSKI A100/H100) —
# das Kamera-Rendering der RL-Env läuft hier grundsätzlich, ohne vast.ai-Miete.
# Isaac Sim 2.3.2 (Basis-Image) wurde nicht offiziell gegen Blackwell getestet —
# `check` ist der reale Nachweis, dass Rendering + Env-Konstruktion funktionieren.
#
# CONTAINER-MODELL: ein langlebiger "Workbench"-Container (sleep infinity), in dem
# Setup/Check/RL-Lauf per `docker exec` laufen. So bleiben HF-Checkpoint-Cache und
# Isaac-Sim-Shader-Cache (OMNI_CACHE, unter /data) zwischen Läufen erhalten — erst
# `clean` entfernt den Container.
#
# NUTZUNG:
#   ./Simulation/server_rl_run.sh preflight                    # Image-Frische + GPU prüfen (kein HF_TOKEN nötig)
#   HF_TOKEN=hf_... ./Simulation/server_rl_run.sh setup        # Checkpoint+USD von HF laden (einmalig)
#   HF_TOKEN=hf_... ./Simulation/server_rl_run.sh check        # LIVE-CHECK: Aufbau ohne Training (--check, 2 Envs)
#   HF_TOKEN=hf_... ./Simulation/server_rl_run.sh rl           # echter RL-Lauf (Vordergrund, lange Laufzeit)
#   ./Simulation/server_rl_run.sh shell|clean|help
#
# Überschreibbar via Env (Defaults für diesen Server):
#   RL_HOST_DATA_DIR (/home/lmuecke/project/data/RL), RL_IMAGE, RL_CONTAINER,
#   RL_GPUS ("device=0" — RT-Core-Rendering + Training auf einer GPU, zweite frei),
#   HF_TOKEN, HF_CHECKPOINT_REPO (luca-mue/groot-g1dex3-checkpoint),
#   RL_NUM_ENVS, RL_ITERATIONS, RL_ROLLOUT_STEPS, RL_LR, RL_KL_COEF, RL_CLIP,
#   RL_SAVE_EVERY, WANDB_API_KEY, SHELL_ON_ERROR — an entrypoint_rl.sh durchgereicht.

set -euo pipefail

# ── Logging-Helfer (Hausstil, vgl. server_robocasa_ref_run.sh) ───────────────
log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m v \033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m ! \033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m!! \033[0m %s\n' "$*" >&2; }

# ── Konfiguration (alle via Env überschreibbar) ──────────────────────────────
IMAGE="${RL_IMAGE:-lucam03/projekt-humanoider-roboter-sim-vastai:latest}"
CONTAINER="${RL_CONTAINER:-groot-rl}"
HOST_DATA_DIR="${RL_HOST_DATA_DIR:-/home/lmuecke/project/data/RL}"
GPUS="${RL_GPUS:-\"device=0\"}"        # RT-Core-Rendering + Backprop teilen sich eine GPU; RL_GPUS=all für beide
SHM_SIZE="${RL_SHM_SIZE:-16g}"

HF_CHECKPOINT_REPO="${HF_CHECKPOINT_REPO:-luca-mue/groot-g1dex3-checkpoint}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/data/checkpoints/groot-g1dex3-checkpoint}"
ASSET_PATH="${ASSET_PATH:-$CHECKPOINT_PATH/g1_dex3.usd}"

ISAAC_PY="/workspace/isaaclab/_isaac_sim/python.sh"
SIM_DIR="/workspace/g1_dex3_sim"

# ── Kleine Helfer ─────────────────────────────────────────────────────────────
require_docker() { command -v docker >/dev/null 2>&1 || { err "docker nicht gefunden."; exit 1; }; }

require_hf_token() {
  [[ -n "${HF_TOKEN:-}" ]] || { err "HF_TOKEN nicht gesetzt (für Checkpoint-Download nötig)."; \
      err "  Aufruf z. B.: HF_TOKEN=hf_... $0 $ACTION"; exit 1; }
}

container_state() { docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo missing; }

ensure_container() {
  local state; state="$(container_state)"
  if [[ "$state" == "true" ]]; then
    return 0
  elif [[ "$state" == "false" ]]; then
    log "Container '$CONTAINER' vorhanden (gestoppt) — starte ihn."
    docker start "$CONTAINER" >/dev/null
    return 0
  fi
  mkdir -p "$HOST_DATA_DIR"
  log "Erzeuge langlebigen Container '$CONTAINER' (Image: $IMAGE, GPU: $GPUS)."
  log "  /data -> $HOST_DATA_DIR  (Checkpoint-Cache, RL-Checkpoints, Isaac-Sim-Shader-Cache)"
  local create_env=( -e PYTHONUNBUFFERED=1 )
  [[ -n "${HF_TOKEN:-}" ]]         && create_env+=( -e "HF_TOKEN=$HF_TOKEN" )
  [[ -n "${WANDB_API_KEY:-}" ]]    && create_env+=( -e "WANDB_API_KEY=$WANDB_API_KEY" )
  docker run -d --name "$CONTAINER" --gpus "$GPUS" --ipc=host --shm-size="$SHM_SIZE" \
    "${create_env[@]}" \
    -v "$HOST_DATA_DIR:/data" \
    --entrypoint bash "$IMAGE" -lc "sleep infinity" >/dev/null
  ok "Container läuft."
}

# Baut das -e-Array für RL-Execs (Checkpoint/Asset + RL_*-Hyperparameter, falls gesetzt)
build_rl_env() {
  RL_ENV=(
    -e "HF_CHECKPOINT_REPO=$HF_CHECKPOINT_REPO"
    -e "CHECKPOINT_PATH=$CHECKPOINT_PATH"
    -e "ASSET_PATH=$ASSET_PATH"
  )
  [[ -n "${HF_TOKEN:-}" ]]         && RL_ENV+=( -e "HF_TOKEN=$HF_TOKEN" )
  [[ -n "${WANDB_API_KEY:-}" ]]    && RL_ENV+=( -e "WANDB_API_KEY=$WANDB_API_KEY" )
  [[ -n "${RL_NUM_ENVS:-}" ]]      && RL_ENV+=( -e "RL_NUM_ENVS=$RL_NUM_ENVS" )
  [[ -n "${RL_ITERATIONS:-}" ]]    && RL_ENV+=( -e "RL_ITERATIONS=$RL_ITERATIONS" )
  [[ -n "${RL_ROLLOUT_STEPS:-}" ]] && RL_ENV+=( -e "RL_ROLLOUT_STEPS=$RL_ROLLOUT_STEPS" )
  [[ -n "${RL_LR:-}" ]]            && RL_ENV+=( -e "RL_LR=$RL_LR" )
  [[ -n "${RL_KL_COEF:-}" ]]       && RL_ENV+=( -e "RL_KL_COEF=$RL_KL_COEF" )
  [[ -n "${RL_CLIP:-}" ]]          && RL_ENV+=( -e "RL_CLIP=$RL_CLIP" )
  [[ -n "${RL_SAVE_EVERY:-}" ]]    && RL_ENV+=( -e "RL_SAVE_EVERY=$RL_SAVE_EVERY" )
  [[ -n "${SHELL_ON_ERROR:-}" ]]   && RL_ENV+=( -e "SHELL_ON_ERROR=$SHELL_ON_ERROR" )
}

# Lädt BC-Checkpoint + USD-Asset von HF, falls noch nicht im Container vorhanden
# (identische Download-Logik wie in entrypoint_rl.sh — wird hier separat gebraucht,
# weil ein direkter rl_finetune.py-Aufruf für `check` den Entrypoint umgeht).
ensure_checkpoint() {
  ensure_container
  if docker exec "$CONTAINER" test -d "$CHECKPOINT_PATH"; then
    ok "BC-Checkpoint bereits vorhanden: $CHECKPOINT_PATH"
    return 0
  fi
  require_hf_token
  log "Lade BC-Checkpoint von HF: $HF_CHECKPOINT_REPO -> $CHECKPOINT_PATH (~10 GB, einmalig)"
  docker exec -e "HF_TOKEN=$HF_TOKEN" -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" "$CONTAINER" \
    bash -lc "huggingface-cli download '$HF_CHECKPOINT_REPO' --local-dir '$CHECKPOINT_PATH'"
  ok "Checkpoint geladen."
}

# ── Aktionen ──────────────────────────────────────────────────────────────────
do_preflight() {
  log "Preflight: Image-Frische (gr00t+flash-attn im Isaac-Sim-Python) + GPU/RT-Cores prüfen."
  docker pull "$IMAGE" || warn "docker pull fehlgeschlagen — nutze lokal vorhandenes Image."
  if docker run --rm --gpus "$GPUS" "$IMAGE" "$ISAAC_PY" - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| dev", torch.cuda.get_device_name(0))
x = torch.randn(2048, 2048, device="cuda")
print("matmul ok  :", float((x @ x).sum()))
import gr00t  # noqa: F401  -- nur vorhanden, wenn Image den Fix aus Commit 8e01979 enthaelt
from flash_attn import flash_attn_func
import flash_attn
q = k = v = torch.randn(1, 8, 4, 64, device="cuda", dtype=torch.float16)
print("flash-attn :", flash_attn.__version__, "->", tuple(flash_attn_func(q, k, v).shape))
print("gr00t importierbar im Isaac-Sim-Python: OK")
PY
  then
    ok "Preflight bestanden — Image ist aktuell (enthaelt Commit-8e01979-Fixes), GPU nutzbar."
  else
    err "Preflight fehlgeschlagen. Wahrscheinlichste Ursache: Image ist der alte Stand"
    err "  (gepusht 2026-06-15, vor den RL-Fixes aus Commit 8e01979). Beheben mit:"
    err "    ./Simulation/update_sim_image.sh --vastai"
    return 1
  fi
}

do_setup() { ensure_checkpoint; ok "Setup abgeschlossen. Weiter mit:  $0 check"; }

do_check() {
  ensure_checkpoint
  log "LIVE-CHECK (Schritt 5/7 in rl-anleitung.md): Aufbau von Env+Policy+Critic, KEIN Training."
  docker exec -w "$SIM_DIR" "$CONTAINER" bash -lc "
    unset VIRTUAL_ENV
    '$ISAAC_PY' '$SIM_DIR/rl_finetune.py' \
        --checkpoint '$CHECKPOINT_PATH' \
        --asset-path '$ASSET_PATH' \
        --num-envs 2 --check"
  ok "Check bestanden — Isaac Sim + RT-Core-Rendering + GR00T-Policy laufen auf dieser GPU."
}

do_rl() {
  ensure_checkpoint
  build_rl_env
  log "Starte echten RL-Lauf (Vordergrund, laeuft je nach RL_ITERATIONS lange)."
  log "  Checkpoints -> $HOST_DATA_DIR/g1_dex3_rl/  (alle RL_SAVE_EVERY Iterationen)"
  docker exec "${RL_ENV[@]}" "$CONTAINER" bash -lc "bash /scripts/entrypoint_rl.sh"
}

do_shell()  { ensure_container; docker exec -it "$CONTAINER" bash -l; }
do_clean()  { log "Entferne Container '$CONTAINER' (Daten in $HOST_DATA_DIR bleiben)."; \
              docker rm -f "$CONTAINER" 2>/dev/null || warn "Container existierte nicht."; ok "Weg."; }

usage() {
  cat <<EOF
server_rl_run.sh — RL-Fine-tuning (FPO), Docker-Server statt vast.ai

⚠️  Vor dem ersten Lauf: Image ggf. neu bauen+pushen (siehe Kopf dieser Datei):
      ./Simulation/update_sim_image.sh --vastai
    'preflight' unten weist das verbindlich nach.

Aktionen:
  preflight   Image-Frische (gr00t+flash-attn im Isaac-Sim-Python) + GPU/RT-Cores testen. Kein HF_TOKEN nötig.
  setup       BC-Checkpoint + USD-Asset von HF laden (einmalig, ~10 GB).
  check       LIVE-CHECK: Env/Policy/Critic aufbauen, 2 Envs, KEIN Training (--check).
  rl          Echter RL-Lauf (Vordergrund). Checkpoints unter $HOST_DATA_DIR/g1_dex3_rl/.
  shell       Interaktive Shell im Container.
  clean       Container entfernen (Daten unter $HOST_DATA_DIR bleiben).
  help        Diese Hilfe.

Beispiele:
  ./Simulation/server_rl_run.sh preflight
  HF_TOKEN=hf_... ./Simulation/server_rl_run.sh check
  HF_TOKEN=hf_... WANDB_API_KEY=... RL_NUM_ENVS=4 ./Simulation/server_rl_run.sh rl

Datenverzeichnis (Host): $HOST_DATA_DIR   ->  Container /data
Image:                    $IMAGE
EOF
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
require_docker
ACTION="${1:-help}"
case "$ACTION" in
  preflight)  do_preflight ;;
  setup)      do_setup ;;
  check)      do_check ;;
  rl)         do_rl ;;
  shell)      do_shell ;;
  clean|down) do_clean ;;
  help|-h|--help) usage ;;
  *) err "Unbekannte Aktion: '$ACTION'"; echo; usage; exit 2 ;;
esac
