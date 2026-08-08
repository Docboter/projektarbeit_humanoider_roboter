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
#   Der zuletzt gepushte Tag (2026-06-15) ist doppelt veraltet:
#     1. Commit 8e01979 (2026-07-19) brachte für RL zwingende Fixes (gr00t+flash-attn ins
#        Isaac-Sim-Python; Start über `isaaclab.sh -p` statt nacktem `python`).
#     2. Der Isaac-Sim-6.0-Port (2026-08-07) wechselt das Basis-Image von isaac-lab 2.3.2
#        auf 3.0.0-beta2-post1 — nötig, weil Isaac Sim 5.1 auf der RTX PRO 6000 Blackwell
#        mit dem (nicht änderbaren) Treiber-Branch 610.x segfaultet.
#   Vor dem ersten `check`/`rl`-Lauf daher zwingend:
#     ./Simulation/update_sim_image.sh --vastai      # baut Dockerfile.vastai neu + pusht
#   `preflight` unten prüft Python-Version, torch, flash-attn und gr00t-Import.
#
# RT-CORES: Die RTX PRO 6000 Blackwell haben RT-Cores (anders als KISSKI A100/H100) —
# das Kamera-Rendering der RL-Env läuft hier grundsätzlich, ohne vast.ai-Miete.
# ⚠️ Der Isaac-Sim-6.0-Port ist NOCH NICHT auf Hardware verifiziert (Isaac Lab 3.0.0-beta2
# ist Beta; das Bundle springt auf numpy 2.5, gr00t ist gegen numpy 1.26 entwickelt).
# `preflight` prüft die Python-Seite, `check` ist der reale Nachweis, dass Rendering +
# Env-Konstruktion auf dieser GPU laufen. Restrisiken: docs/weiterfuehrend/rl-anleitung.md.
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
#   ./Simulation/server_rl_run.sh gap                          # Domain-Gap real vs. sim messen (nach 'cams')
#   ./Simulation/server_rl_run.sh shell|clean|help
#
# Überschreibbar via Env (Defaults für diesen Server):
#   RL_HOST_DATA_DIR (/home/lmuecke/project/data/RL), RL_IMAGE, RL_CONTAINER,
#   RL_GPUS ("device=1,0" — beide Karten; erste trägt Rendering+Training, zweite nur
#            das eingefrorene Referenzmodell), RL_REF_DEVICE (auto|same|cuda:N),
#   HF_TOKEN, HF_CHECKPOINT_REPO (luca-mue/groot-g1dex3-checkpoint),
#   RL_NUM_ENVS, RL_ITERATIONS, RL_ROLLOUT_STEPS, RL_LR, RL_KL_COEF, RL_CLIP,
#   RL_MINIBATCH_SIZE, RL_FPO_MC_SAMPLES, RL_EPOCHS_PER_ITER (Speicher-Stellschrauben),
#   RL_SAVE_EVERY, WANDB_API_KEY, WANDB_MODE, RL_WANDB_VIDEO_EVERY, SHELL_ON_ERROR,
#   LIVE_VIEW, LIVE_VIEW_PORT, LIVE_VIEW_EVERY_N, LIVE_VIEW_CAMS — durchgereicht.

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
# Beide Karten, ABER in dieser Reihenfolge: die zuerst genannte wird im Container zu
# cuda:0 und traegt Rendering + Policy + Optimizer; die zweite bekommt nur das
# eingefrorene Referenzmodell (~6-7 GB, nur no_grad). Physische GPU 1 steht vorn, weil
# dort am 2026-08-08 mehr frei war (llama-server: 41 GB auf GPU 0, 37 GB auf GPU 1).
# → Vor einem langen Lauf `nvidia-smi` prüfen und ggf. auf "device=0,1" drehen.
# Einzelkarte: RL_GPUS='"device=0"' — das Referenzmodell rückt dann automatisch mit auf.
GPUS="${RL_GPUS:-\"device=1,0\"}"
SHM_SIZE="${RL_SHM_SIZE:-16g}"

HF_CHECKPOINT_REPO="${HF_CHECKPOINT_REPO:-luca-mue/groot-g1dex3-checkpoint}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/data/checkpoints/groot-g1dex3-checkpoint}"
ASSET_PATH="${ASSET_PATH:-$CHECKPOINT_PATH/g1_dex3.usd}"

# Live-Ansicht (MJPEG im Browser, Spur B des Livestream-Plans). Der Port wird IMMER
# gemappt — auch bei LIVE_VIEW=0 —, weil -p nur beim ANLEGEN des langlebigen Containers
# wirkt: sonst müsste man für ein späteres LIVE_VIEW=1 erst `clean` fahren.
LIVE_VIEW_PORT="${LIVE_VIEW_PORT:-8900}"

ISAAC_PY="/workspace/isaaclab/_isaac_sim/python.sh"
SIM_DIR="/workspace/g1_dex3_sim"

# Repo-Wurzel (Elternverzeichnis dieses Skripts) — für den Live-Mount des Sim-Codes.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${RL_REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# ── Log-Spiegelung ────────────────────────────────────────────────────────────
# Jede nicht-interaktive Aktion landet zusätzlich in einer Datei unter dem gemounteten
# Datenverzeichnis — auf dem Host also direkt lesbar, ohne `docker cp`.
# Grund: `rl` läuft Stunden im Vordergrund und tmux-Scrollback ist endlich; und Ausgaben
# wie die Kamera-Pose-Tabelle aus `cams` will man nachträglich noch lesen können.
# Muster wie in entrypoint_sim.sh (dort /data/logs/entrypoint.log).
start_logging() {
  LOG_DIR="$HOST_DATA_DIR/logs"
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/$1-$(date +%Y%m%d-%H%M%S).log"
  exec > >(tee -a "$LOG_FILE") 2>&1
  log "Log dieses Aufrufs: $LOG_FILE"
}

# ── Kleine Helfer ─────────────────────────────────────────────────────────────
require_docker() { command -v docker >/dev/null 2>&1 || { err "docker nicht gefunden."; exit 1; }; }

require_hf_token() {
  [[ -n "${HF_TOKEN:-}" ]] || { err "HF_TOKEN nicht gesetzt (für Checkpoint-Download nötig)."; \
      err "  Aufruf z. B.: HF_TOKEN=hf_... $0 $ACTION"; exit 1; }
}

container_state() { docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo missing; }

# Ist der Host-Port frei? Reines Bash (kein ss/netstat/lsof im Image-losen Fall nötig).
port_free() { ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

# -e-Flags für die Live-Ansicht. rl_finetune.py liest LIVE_VIEW* SELBST als CLI-Defaults —
# deshalb wirkt das auch ohne Image-Rebuild (g1_dex3_sim ist gemountet, /scripts nicht).
live_view_env() {
  LIVE_ENV=( -e "LIVE_VIEW=${LIVE_VIEW:-0}" -e "LIVE_VIEW_PORT=$LIVE_VIEW_PORT" )
  [[ -n "${LIVE_VIEW_EVERY_N:-}" ]] && LIVE_ENV+=( -e "LIVE_VIEW_EVERY_N=$LIVE_VIEW_EVERY_N" )
  [[ -n "${LIVE_VIEW_CAMS:-}" ]]    && LIVE_ENV+=( -e "LIVE_VIEW_CAMS=$LIVE_VIEW_CAMS" )
  return 0   # siehe Kommentar in build_rl_env: letzte Zeile darf kein `[[ … ]] &&` sein
}

# Sowohl -p als auch --gpus wirken NUR beim Anlegen des Containers. Ein langlebiger
# Container aus einem früheren Lauf hat sie also nicht, egal was hier gesetzt ist —
# und das äußert sich stumm: der Stream lauscht nur container-intern, bzw. das
# Referenzmodell rückt mangels zweiter Karte auf die erste zurück.
warn_if_container_stale() {
  if [[ "${LIVE_VIEW:-0}" != "0" ]]; then
    local ports; ports="$(docker inspect -f '{{json .NetworkSettings.Ports}}' "$CONTAINER" 2>/dev/null || echo '{}')"
    if [[ "$ports" != *"\"$LIVE_VIEW_PORT/tcp\":[{"* ]]; then
      warn "LIVE_VIEW=1, aber Container '$CONTAINER' hat Port $LIVE_VIEW_PORT NICHT veröffentlicht."
      warn "  Einmalig neu anlegen:  $0 clean   (Daten unter $HOST_DATA_DIR bleiben)"
    fi
  fi
  if [[ "$GPUS" == *,* || "$GPUS" == all ]] && [[ "${RL_REF_DEVICE:-auto}" != "same" ]]; then
    local n; n="$(docker exec "$CONTAINER" bash -lc 'nvidia-smi -L 2>/dev/null | wc -l' 2>/dev/null || echo 0)"
    if [[ "${n:-0}" -lt 2 ]]; then
      warn "Container '$CONTAINER' sieht nur $n GPU(s) — das Referenzmodell bleibt auf der ersten."
      warn "  Einmalig neu anlegen:  $0 clean   (Daten unter $HOST_DATA_DIR bleiben)"
    fi
  fi
}

ensure_container() {
  local state; state="$(container_state)"
  if [[ "$state" == "true" ]]; then
    warn_if_container_stale
    return 0
  elif [[ "$state" == "false" ]]; then
    log "Container '$CONTAINER' vorhanden (gestoppt) — starte ihn."
    docker start "$CONTAINER" >/dev/null
    warn_if_container_stale
    return 0
  fi
  mkdir -p "$HOST_DATA_DIR"
  log "Erzeuge langlebigen Container '$CONTAINER' (Image: $IMAGE, GPU: $GPUS)."
  log "  /data -> $HOST_DATA_DIR  (Checkpoint-Cache, RL-Checkpoints, Isaac-Sim-Shader-Cache)"
  local create_env=( -e PYTHONUNBUFFERED=1 )
  [[ -n "${HF_TOKEN:-}" ]]         && create_env+=( -e "HF_TOKEN=$HF_TOKEN" )
  [[ -n "${WANDB_API_KEY:-}" ]]    && create_env+=( -e "WANDB_API_KEY=$WANDB_API_KEY" )
  # g1_dex3_sim aus dem Repo ÜBER die Image-Kopie mounten (Muster wie kisski_submit.sh /
  # server_robocasa_ref_run.sh): LIVE-CHECK-Iterationen an rl_finetune.py & Co. brauchen dann
  # nur `git pull` auf dem Server — kein Image-Rebuild, kein `clean`.
  log "  Sim-Code-Mount -> $REPO_DIR/Simulation/g1_dex3_sim"
  local port_flag=()
  if port_free "$LIVE_VIEW_PORT"; then
    port_flag=( -p "$LIVE_VIEW_PORT:$LIVE_VIEW_PORT" )
    log "  Live-Ansicht  -> Port $LIVE_VIEW_PORT veröffentlicht (nutzbar mit LIVE_VIEW=1)"
  else
    warn "Host-Port $LIVE_VIEW_PORT ist belegt — Live-Ansicht bleibt unveröffentlicht."
    warn "  Anderen Port wählen:  LIVE_VIEW_PORT=8901 $0 clean && … $0 rl"
  fi
  docker run -d --name "$CONTAINER" --gpus "$GPUS" --ipc=host --shm-size="$SHM_SIZE" \
    "${create_env[@]}" "${port_flag[@]}" \
    -v "$HOST_DATA_DIR:/data" \
    -v "$REPO_DIR/Simulation/g1_dex3_sim:$SIM_DIR:ro" \
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
  # Ohne das hier wäre WANDB_MODE=online wirkungslos: entrypoint_rl.sh setzt sonst
  # stur 'offline', und die Rollout-Videos (RL_WANDB_VIDEO_EVERY) lägen unerreichbar
  # im Container statt im Dashboard.
  [[ -n "${WANDB_MODE:-}" ]]       && RL_ENV+=( -e "WANDB_MODE=$WANDB_MODE" )
  [[ -n "${RL_NUM_ENVS:-}" ]]      && RL_ENV+=( -e "RL_NUM_ENVS=$RL_NUM_ENVS" )
  [[ -n "${RL_ITERATIONS:-}" ]]    && RL_ENV+=( -e "RL_ITERATIONS=$RL_ITERATIONS" )
  [[ -n "${RL_ROLLOUT_STEPS:-}" ]] && RL_ENV+=( -e "RL_ROLLOUT_STEPS=$RL_ROLLOUT_STEPS" )
  [[ -n "${RL_LR:-}" ]]            && RL_ENV+=( -e "RL_LR=$RL_LR" )
  [[ -n "${RL_KL_COEF:-}" ]]       && RL_ENV+=( -e "RL_KL_COEF=$RL_KL_COEF" )
  [[ -n "${RL_CLIP:-}" ]]          && RL_ENV+=( -e "RL_CLIP=$RL_CLIP" )
  [[ -n "${RL_SAVE_EVERY:-}" ]]    && RL_ENV+=( -e "RL_SAVE_EVERY=$RL_SAVE_EVERY" )
  # Speicher-Stellschrauben bei OOM (rl_finetune.py liest sie selbst als CLI-Defaults).
  [[ -n "${RL_MINIBATCH_SIZE:-}" ]]  && RL_ENV+=( -e "RL_MINIBATCH_SIZE=$RL_MINIBATCH_SIZE" )
  [[ -n "${RL_FPO_MC_SAMPLES:-}" ]]  && RL_ENV+=( -e "RL_FPO_MC_SAMPLES=$RL_FPO_MC_SAMPLES" )
  [[ -n "${RL_EPOCHS_PER_ITER:-}" ]] && RL_ENV+=( -e "RL_EPOCHS_PER_ITER=$RL_EPOCHS_PER_ITER" )
  [[ -n "${RL_REF_DEVICE:-}" ]]      && RL_ENV+=( -e "RL_REF_DEVICE=$RL_REF_DEVICE" )
  # Render-/Belichtungshebel, damit ein per `cams` gefundener Wert auch im RL-Lauf gilt.
  [[ -n "${RL_AA_MODE:-}" ]]         && RL_ENV+=( -e "RL_AA_MODE=$RL_AA_MODE" )
  [[ -n "${RL_DOME_INTENSITY:-}" ]]  && RL_ENV+=( -e "RL_DOME_INTENSITY=$RL_DOME_INTENSITY" )
  [[ -n "${RL_CAMERA_CLASS:-}" ]]    && RL_ENV+=( -e "RL_CAMERA_CLASS=$RL_CAMERA_CLASS" )
  [[ -n "${DR_ENABLED:-}" ]]         && RL_ENV+=( -e "DR_ENABLED=$DR_ENABLED" )
  # Gegen Fragmentierung — der OOM-Traceback empfahl es selbst (1,13 GB reserviert,
  # aber unbenutzt). Ueberschreibbar, falls es auf dieser Torch-Version stoert.
  RL_ENV+=( -e "PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}" )
  [[ -n "${SHELL_ON_ERROR:-}" ]]   && RL_ENV+=( -e "SHELL_ON_ERROR=$SHELL_ON_ERROR" )
  [[ -n "${RL_WANDB_VIDEO_EVERY:-}" ]] && RL_ENV+=( -e "RL_WANDB_VIDEO_EVERY=$RL_WANDB_VIDEO_EVERY" )
  live_view_env
  RL_ENV+=( "${LIVE_ENV[@]}" )
  # PFLICHT: Ist die letzte Zeile ein nicht zutreffendes `[[ … ]] && …`, gibt die Funktion 1
  # zurück und `set -e` beendet das Skript STILL — genau vor dem RL-Start (beobachtet 2026-08-07,
  # als SHELL_ON_ERROR ungesetzt war). Nie durch eine weitere Bedingung ersetzen.
  return 0
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
  log "Preflight: Isaac-Sim-Python (3.12), torch, flash-attn und gr00t-Import auf der GPU prüfen."
  # RL_SKIP_PULL=1, wenn das Image LOKAL frisch gebaut wurde (update_sim_image.sh --skip-push):
  # ein docker pull würde das lokale :latest sonst mit dem älteren Docker-Hub-Stand überschreiben.
  if [[ "${RL_SKIP_PULL:-0}" == "1" ]]; then
    warn "RL_SKIP_PULL=1 — überspringe docker pull, nutze lokales Image."
  else
    docker pull "$IMAGE" || warn "docker pull fehlgeschlagen — nutze lokal vorhandenes Image."
  fi
  if docker run --rm -i --gpus "$GPUS" "$IMAGE" "$ISAAC_PY" - <<'PY'
import sys
print("python     :", ".".join(map(str, sys.version_info[:2])), "(erwartet 3.12 fuer Isaac Sim 6.0)")
import torch
print("torch      :", torch.__version__, "| cuda", torch.version.cuda, "| dev", torch.cuda.get_device_name(0))
x = torch.randn(2048, 2048, device="cuda")
print("matmul ok  :", float((x @ x).sum()))
import gr00t  # noqa: F401  -- nur vorhanden, wenn das Image aus Dockerfile.vastai gebaut wurde
from flash_attn import flash_attn_func
import flash_attn
q = k = v = torch.randn(1, 8, 4, 64, device="cuda", dtype=torch.float16)
print("flash-attn :", flash_attn.__version__, "->", tuple(flash_attn_func(q, k, v).shape))
print("OK: gr00t + flash-attn im Isaac-Sim-Python nutzbar")
PY
  then
    ok "Preflight bestanden — Image nutzbar (Python 3.12, torch 2.10, flash-attn, gr00t)."
  else
    err "Preflight fehlgeschlagen. Haeufigste Ursache: Image noch nicht neu gebaut"
    err "  (Isaac-Sim-6.0-Port). Beheben mit:"
    err "    ./Simulation/update_sim_image.sh --vastai"
    err "  Details: docs/weiterfuehrend/rl-anleitung.md (Troubleshooting)"
    return 1
  fi
}

do_setup() { ensure_checkpoint; ok "Setup abgeschlossen. Weiter mit:  $0 check"; }

do_check() {
  ensure_checkpoint
  log "LIVE-CHECK (Schritt 5/7 in rl-anleitung.md): Aufbau von Env+Policy+Critic, KEIN Training."
  # Exit-Code des Kit-Pythons ist NICHT belastbar: Isaac Sim beendet auch nach einem
  # Python-Traceback mit 0 (beobachtet 2026-08-07: TypeError → trotzdem Exit 0). Erfolg
  # daher ausschließlich am positiven Marker von rl_finetune.py --check festmachen.
  local out
  # LIVE_VIEW mitgeben: rl_finetune.py legt die Live-Ansicht VOR dem --check-Return an,
  # der Check weist damit auch Pillow + Port-Bindung im Kit-Python nach.
  live_view_env
  out=$(docker exec -w "$SIM_DIR" "${LIVE_ENV[@]}" \
        -e "RL_REF_DEVICE=${RL_REF_DEVICE:-auto}" "$CONTAINER" bash -lc "
    unset VIRTUAL_ENV
    '$ISAAC_PY' '$SIM_DIR/rl_finetune.py' \
        --checkpoint '$CHECKPOINT_PATH' \
        --asset-path '$ASSET_PATH' \
        --num-envs 2 --check" 2>&1 | tee /dev/stderr) || true
  if grep -q "Aufbau OK" <<<"$out"; then
    ok "Check bestanden — Isaac Sim + RT-Core-Rendering + GR00T-Policy laufen auf dieser GPU."
  else
    err "Check FEHLGESCHLAGEN — Erfolgsmarker ('Aufbau OK') fehlt in der Ausgabe."
    err "  Traceback oben beachten; LIVE-CHECK-Stellen: docs/weiterfuehrend/rl-anleitung.md Schritt 7."
    return 1
  fi
}

do_rl() {
  ensure_checkpoint
  build_rl_env
  log "Starte echten RL-Lauf (Vordergrund, laeuft je nach RL_ITERATIONS lange)."
  log "  Checkpoints -> $HOST_DATA_DIR/g1_dex3_rl/  (alle RL_SAVE_EVERY Iterationen)"
  if [[ "${LIVE_VIEW:-0}" != "0" ]]; then
    local ip; ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    log "  Live-Ansicht -> http://${ip:-<server-ip>}:$LIVE_VIEW_PORT/"
  fi
  docker exec "${RL_ENV[@]}" "$CONTAINER" bash -lc "bash /scripts/entrypoint_rl.sh"
}

# Kamera-Diagnose: konfigurierte gegen tatsächlich gerenderte Pose + PNG je Kamera.
# Anlass: cam_scene zeigte in der Live-Ansicht nur Hintergrund, obwohl die konfigurierte
# Pose nachweislich Tisch + Roboter erfassen müsste.
do_cams() {
  ensure_checkpoint
  # Messhebel für die leeren Kamerabilder (je ein Lauf pro Wert, siehe rl-anleitung.md):
  #   RL_SETTLE_STEPS   Render-Konvergenz
  #   RL_AA_MODE        Anti-Aliasing-Modus
  #   RL_DOME_INTENSITY Belichtung — Basiswert der Szene
  #   RL_DOME_SWEEP     Belichtung — mehrere Werte in EINEM Lauf, z. B. "500,120,30"
  #   DR_ENABLED=0      visuelle Domain Randomization aus (WIDERLEGT, Lauf 11)
  #   RL_CAMERA_CLASS=camera  gewöhnliche Camera statt TiledCamera (Halbierungstest)
  log "Kamera-Posen dumpen (num_envs=${RL_NUM_ENVS:-4}, settle=${RL_SETTLE_STEPS:-8}," \
      "aa=${RL_AA_MODE:-<Isaac-Default>}, dome=${RL_DOME_INTENSITY:-2000}," \
      "sweep=${RL_DOME_SWEEP:-<aus>}, DR=${DR_ENABLED:-1}," \
      "cam=${RL_CAMERA_CLASS:-tiled}) → $HOST_DATA_DIR/cam_dump/"
  docker exec -w "$SIM_DIR" \
    -e "RL_AA_MODE=${RL_AA_MODE:-}" \
    -e "RL_DOME_INTENSITY=${RL_DOME_INTENSITY:-2000}" \
    -e "RL_DOME_SWEEP=${RL_DOME_SWEEP:-}" \
    -e "DR_ENABLED=${DR_ENABLED:-1}" \
    -e "RL_CAMERA_CLASS=${RL_CAMERA_CLASS:-tiled}" \
    "$CONTAINER" bash -lc "
    unset VIRTUAL_ENV
    '$ISAAC_PY' '$SIM_DIR/dump_camera_poses.py' \
        --headless --enable_cameras \
        --asset-path '$ASSET_PATH' \
        --settle-steps ${RL_SETTLE_STEPS:-8} \
        --num-envs ${RL_NUM_ENVS:-4}" 2>&1 | tee /dev/stderr | grep -q "\[dump\] fertig" \
    && ok "PNGs + Posen unter $HOST_DATA_DIR/cam_dump/" \
    || { err "Kamera-Dump ohne Erfolgsmarker beendet (Traceback oben)."; return 1; }
}

# Domain-Gap: Kosinus-Distanz real gegen sim je Policy-Kamera durch den eingefrorenen
# SigLIP-ViT (= GR00Ts Vision-Backbone, da BC mit tune_visual=false lief).
# Braucht die Frames aus 'cams' — misst also genau die Bilder, die auch die Policy sieht.
#
# Skript und Referenzbilder werden per `docker cp` hineingelegt statt gemountet:
#   - /scripts ist ins Image gebacken (kein Mount) -> ein Edit braeuchte sonst einen Rebuild;
#   - Simulation/camera_reference/ liegt im Image ueberhaupt nicht;
#   - ein zusaetzliches -v wirkt nur beim ANLEGEN des Containers, verlangte also 'clean'.
# `docker cp` in den laufenden Container umgeht alle drei Punkte.
do_gap() {
  ensure_container
  local sim_dir="${GAP_SIM_DIR:-/data/cam_dump}"

  if ! docker exec "$CONTAINER" test -f "$sim_dir/cam_left_high.png"; then
    err "Keine Sim-Frames unter $sim_dir im Container."
    err "  Zuerst den Kamera-Dump fahren:  HF_TOKEN=hf_... $0 cams"
    return 1
  fi

  log "Referenzbilder + Messskript in den Container kopieren."
  docker exec "$CONTAINER" mkdir -p /workspace/camera_reference
  docker cp "$REPO_DIR/Simulation/camera_reference/." "$CONTAINER:/workspace/camera_reference/"
  docker cp "$REPO_DIR/Simulation/scripts/measure_domain_gap.py" \
            "$CONTAINER:/workspace/measure_domain_gap.py"

  # HF_HOME unter /data: der SigLIP-Download (~1,6 GB) ueberlebt so ein 'clean'.
  # HF_TOKEN nur durchreichen, wenn gesetzt — ein leerer Wert gilt huggingface_hub als
  # gesetzter, ungueltiger Token. Fuer das oeffentliche SigLIP wird er ohnehin nicht gebraucht.
  local gap_env=( -e "HF_HOME=${HF_HOME:-/data/hf_cache}" )
  [[ -n "${HF_TOKEN:-}" ]] && gap_env+=( -e "HF_TOKEN=$HF_TOKEN" )

  log "Domain-Gap messen (SigLIP-ViT, Sim-Frames aus $sim_dir)."
  docker exec "${gap_env[@]}" "$CONTAINER" bash -lc "
    unset VIRTUAL_ENV
    '$ISAAC_PY' /workspace/measure_domain_gap.py \
        --real-dir /workspace/camera_reference \
        --sim-dir '$sim_dir'" 2>&1 | tee /dev/stderr | grep -q "\[gap\] fertig" \
    && ok "Ergebnisse auch als JSON unter $HOST_DATA_DIR/${sim_dir#/data/}/domain_gap_results.json" \
    || { err "Domain-Gap-Messung ohne Erfolgsmarker beendet (Traceback oben)."; return 1; }
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
  cams        Kamera-Diagnose: konfigurierte vs. gerenderte Pose + ein PNG je Kamera.
  gap         Domain-Gap real vs. sim je Policy-Kamera (SigLIP-ViT). Setzt 'cams' voraus.
  rl          Echter RL-Lauf (Vordergrund). Checkpoints unter $HOST_DATA_DIR/g1_dex3_rl/.
  shell       Interaktive Shell im Container.
  clean       Container entfernen (Daten unter $HOST_DATA_DIR bleiben).
  help        Diese Hilfe.

Beispiele:
  ./Simulation/server_rl_run.sh preflight
  HF_TOKEN=hf_... ./Simulation/server_rl_run.sh check
  HF_TOKEN=hf_... WANDB_API_KEY=... RL_NUM_ENVS=4 ./Simulation/server_rl_run.sh rl
  # mit Live-Ansicht im Browser + W&B-Video alle 10 Iterationen:
  HF_TOKEN=hf_... WANDB_API_KEY=... LIVE_VIEW=1 RL_WANDB_VIDEO_EVERY=10 \\
      RL_NUM_ENVS=4 ./Simulation/server_rl_run.sh rl

Live-Ansicht (opt-in, docs/weiterfuehrend/livestream-plan.md Spur B):
  LIVE_VIEW=1            MJPEG-Stream des Rollouts im Browser
  LIVE_VIEW_PORT         Host- und Container-Port (Default 8900; wird beim Anlegen gemappt)
  LIVE_VIEW_EVERY_N      nur jedes n-te Frame senden (Default 1)
  LIVE_VIEW_CAMS         Kameras, kommagetrennt (Default cam_left_high,cam_left_wrist —
                         die kalibrierten Policy-Kameras, also die Modell-Eingabe.
                         cam_scene ist unvalidiert, siehe Aktion 'cams')
  RL_WANDB_VIDEO_EVERY   alle N Iterationen einen Rollout ins W&B-Dashboard (0 = aus)
  -> Aufruf im Browser:  http://<server-ip>:$LIVE_VIEW_PORT/
     nur SSH?            ssh -L $LIVE_VIEW_PORT:localhost:$LIVE_VIEW_PORT <server>

Logs (jede Aktion außer 'shell' wird gespiegelt):
  Host-Seite:      $HOST_DATA_DIR/logs/<aktion>-<zeitstempel>.log
  Container-Seite: $HOST_DATA_DIR/logs/entrypoint_rl.log   (= /data/logs/… im Container)

Datenverzeichnis (Host): $HOST_DATA_DIR   ->  Container /data
Image:                    $IMAGE
EOF
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
require_docker
ACTION="${1:-help}"
# 'shell' bleibt ungespiegelt (interaktives -it verträgt die Pipe nicht), 'help'/'clean'
# haben nichts zu protokollieren.
case "$ACTION" in
  preflight|setup|check|cams|gap|rl) start_logging "$ACTION" ;;
esac
case "$ACTION" in
  preflight)  do_preflight ;;
  setup)      do_setup ;;
  check)      do_check ;;
  cams)       do_cams ;;
  gap)        do_gap ;;
  rl)         do_rl ;;
  shell)      do_shell ;;
  clean|down) do_clean ;;
  help|-h|--help) usage ;;
  *) err "Unbekannte Aktion: '$ACTION'"; echo; usage; exit 2 ;;
esac
