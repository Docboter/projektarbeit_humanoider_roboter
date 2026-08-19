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
# HOST-KONFIGURATION (seit 2026-08): Dieses Skript enthält KEINE rechnerspezifischen
# Pfade mehr. Alles Host-Abhängige kommt aus einer gitignorierten .env.local im
# Repo-Wurzelverzeichnis (Vorlage: .env.local.example). Ohne sie landet /data unter
# $HOME/groot-rl-data. Auf dem IKR-Server kann ein benutzerunabhängiger Projektpfad
# gesetzt werden:
#   echo ': "${RL_HOST_DATA_DIR:=$HOME/project/data/RL}"' >> .env.local
# Vollständige Anleitung inkl. KISSKI: docs/portabilitaet.md
#
# NUTZUNG:
#   ./Simulation/server_rl_run.sh preflight                    # Image-Frische + GPU prüfen (kein HF_TOKEN nötig)
#   ./Simulation/server_rl_run.sh view                         # nur die Szene ansehen — OHNE Modell/Checkpoint/HF_TOKEN
#   HF_TOKEN=hf_... ./Simulation/server_rl_run.sh setup        # Checkpoint+USD von HF laden (einmalig)
#   HF_TOKEN=hf_... ./Simulation/server_rl_run.sh check        # LIVE-CHECK: Aufbau ohne Training (--check, 2 Envs)
#   HF_TOKEN=hf_... ./Simulation/server_rl_run.sh rl           # echter RL-Lauf (Vordergrund, lange Laufzeit)
#   ./Simulation/server_rl_run.sh gap                          # Domain-Gap real vs. sim messen (nach 'cams')
#   ./Simulation/server_rl_run.sh livecheck                    # Phase 0 der LIVE-Variante: NVENC/Extension/Ports
#   LIVESTREAM=2 HF_TOKEN=hf_... ./Simulation/server_rl_run.sh eval   # LIVE statt Videos (WebRTC-Viewport)
#   GROOT_INFERENCE_BACKEND=compile ./Simulation/server_rl_run.sh eval # DiT mit torch.compile
#   GROOT_INFERENCE_BACKEND=tensorrt CAMERA_RENDER_EVERY_N=8 \
#       SCENE_CAM=0 ./Simulation/server_rl_run.sh eval          # TensorRT + synchrones Rendering
#   ./Simulation/server_rl_run.sh shell|clean|help
#
# Überschreibbar via Env (Defaults für diesen Server):
#   RL_HOST_DATA_DIR ($HOME/groot-rl-data — s. HOST-KONFIGURATION oben), RL_IMAGE, RL_CONTAINER,
#   RL_GPUS ("device=1,0" — beide Karten; erste trägt Rendering+Training, zweite nur
#            das eingefrorene Referenzmodell), RL_REF_DEVICE (auto|same|cuda:N),
#   HF_TOKEN, HF_CHECKPOINT_REPO (luca-mue/groot-g1dex3-checkpoint),
#   RL_NUM_ENVS, RL_ITERATIONS, RL_ROLLOUT_STEPS, RL_LR, RL_KL_COEF, RL_CLIP,
#   RL_MINIBATCH_SIZE, RL_FPO_MC_SAMPLES, RL_EPOCHS_PER_ITER (Speicher-Stellschrauben),
#   RL_SAVE_EVERY, WANDB_API_KEY, WANDB_MODE, RL_WANDB_VIDEO_EVERY, SHELL_ON_ERROR,
#   LIVE_VIEW, LIVE_VIEW_PORT, LIVE_VIEW_EVERY_N, LIVE_VIEW_CAMS — durchgereicht,
#   LIVESTREAM, LIVESTREAM_PORT, LIVESTREAM_MEDIA_PORT, LIVE_KEEP_VIDEO (LIVE-Variante).
#   GROOT_INFERENCE_BACKEND (eager|compile|tensorrt; Default eager),
#   GROOT_TRT_ENGINE_PATH (optional; leer = Engine per Checkpoint-Fingerprint finden),
#   CAMERA_RENDER_EVERY_N (Default 1; schneller Modus = EXECUTION_HORIZON, meist 8).
#   Backend-Details und Build: docs/simulation/inferenz-optimierung.md
#
# ZWEI LIVE-WEGE, bewusst getrennt (docs/simulation/live-ansicht.md):
#   LIVE_VIEW=1   „Spur B" — MJPEG-Bilder im Browser. Zustandslos, beliebig viele
#                 Zuschauer, per ssh -L tunnelbar. Für den tagelangen RL-Lauf gedacht.
#   LIVESTREAM=2  „Spur A" — der komplette Isaac-Sim-Viewport per WebRTC, geöffnet vom
#                 nativen „Isaac Sim WebRTC Streaming Client" auf dem Arbeitsrechner:
#                 freie Kamera, Szene drehen, Isaac-Sim-UI. Genau ein Zuschauer, braucht
#                 UDP. Für eval/grasp/baseline — dort will man sich die Greifpose ansehen.
#                 Ersetzt standardmäßig die MP4-Aufzeichnung (LIVE_KEEP_VIDEO=1 = beides).

set -euo pipefail

# ── Logging-Helfer (Hausstil, vgl. server_robocasa_ref_run.sh) ───────────────
log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m v \033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m ! \033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m!! \033[0m %s\n' "$*" >&2; }

# ── Repo-Wurzel + lokale Host-Konfiguration ──────────────────────────────────
# Steht bewusst VOR dem Konfigblock: .env.local darf dessen Defaults vorbelegen.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${RL_REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# Alles Rechner-Spezifische (Datenverzeichnis, Tokens) gehört NICHT ins Skript, sondern
# in eine gitignorierte .env.local im Repo-Wurzelverzeichnis. Vorlage: .env.local.example,
# Erklärung: docs/portabilitaet.md.
# Vorrang: explizite Umgebungsvariable > .env.local > eingebauter Default. Damit das gilt,
# nutzt die Datei die Form  : "${VAR:=wert}"  — ein nacktes VAR=wert würde eine bereits
# gesetzte Variable überschreiben. `set -a` exportiert das Gesetzte, damit es auch
# Unterprozesse (docker, apptainer) erreicht.
if [[ -f "$REPO_DIR/.env.local" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_DIR/.env.local"
  set +a
fi

# ── Konfiguration (alle via Env überschreibbar) ──────────────────────────────
IMAGE="${RL_IMAGE:-lucam03/projekt-humanoider-roboter-sim-vastai:latest}"
CONTAINER="${RL_CONTAINER:-groot-rl}"
# Host-Verzeichnis, das im Container zu /data wird: HF-Checkpoint-Cache, RL-Checkpoints,
# Isaac-Sim-Shader-Cache und die Logspiegelung. Wächst auf viele GB — auf eine Partition
# mit Platz legen. Der Default ist bewusst host-agnostisch; der konkrete Pfad EINES Servers
# gehört in .env.local, nicht hierher (bis 2026-08 stand hier fest /home/lmuecke/project/
# data/RL — auf jedem anderen Rechner ein "Permission denied" beim ersten mkdir).
HOST_DATA_DIR="${RL_HOST_DATA_DIR:-$HOME/groot-rl-data}"
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
# Default = schwarzhändiges Asset (Domain-Gap-Fix vom Juni, s. ensure_black_hands).
# kisski_sim_submit.sh tat das seit Juni, dieses Skript nicht — deshalb rendert der Server
# bis runs/20260808/21 weiße Hände gegen schwarze im Datensatz. BLACK_HANDS=0 -> Original.
BLACK_HANDS="${BLACK_HANDS:-1}"
if [[ "$BLACK_HANDS" == "1" ]]; then
  ASSET_PATH="${ASSET_PATH:-$CHECKPOINT_PATH/g1_dex3_blackhands.usd}"
else
  ASSET_PATH="${ASSET_PATH:-$CHECKPOINT_PATH/g1_dex3.usd}"
fi

# Live-Ansicht (MJPEG im Browser, Spur B des Livestream-Plans). Der Port wird IMMER
# gemappt — auch bei LIVE_VIEW=0 —, weil -p nur beim ANLEGEN des langlebigen Containers
# wirkt: sonst müsste man für ein späteres LIVE_VIEW=1 erst `clean` fahren.
LIVE_VIEW_PORT="${LIVE_VIEW_PORT:-8900}"

# LIVE-Variante (WebRTC-Viewport, Spur A). Gleiche Überlegung wie oben: beide Ports
# werden beim Anlegen gemappt, damit ein späteres LIVESTREAM=2 kein `clean` verlangt.
LIVESTREAM="${LIVESTREAM:-0}"
LIVESTREAM_PORT="${LIVESTREAM_PORT:-49100}"
LIVESTREAM_MEDIA_PORT="${LIVESTREAM_MEDIA_PORT:-47998}"
LIVE_KEEP_VIDEO="${LIVE_KEEP_VIDEO:-0}"

ISAAC_PY="/workspace/isaaclab/_isaac_sim/python.sh"
SIM_DIR="/workspace/g1_dex3_sim"
# GR00T-venv im Container: fuer reine Policy-Inferenz ohne Isaac Sim (Aktion 'span').
# Gleicher Pfad wie in entrypoint_sim.sh.
GROOT_ROOT="${GROOT_ROOT:-/app/Groot-1.6}"

# Dieselbe Livestream-Logik wie in den Entrypoints — hier auf dem HOST gesourct, damit
# `grasp` (das rl-fremde Skripte direkt aufruft) nicht seine eigene Kopie braucht.
# shellcheck source=scripts/lib_livestream.sh
source "$SCRIPT_DIR/scripts/lib_livestream.sh"
livestream_init

# ── Log-Spiegelung ────────────────────────────────────────────────────────────
# Jede nicht-interaktive Aktion landet zusätzlich in einer Datei unter dem gemounteten
# Datenverzeichnis — auf dem Host also direkt lesbar, ohne `docker cp`.
# Grund: `rl` läuft Stunden im Vordergrund und tmux-Scrollback ist endlich; und Ausgaben
# wie die Kamera-Pose-Tabelle aus `cams` will man nachträglich noch lesen können.
# Muster wie in entrypoint_sim.sh (dort /data/logs/entrypoint.log).
# Legt das Host-Datenverzeichnis an — und bricht mit einer brauchbaren Anleitung ab,
# statt kommentarlos an "Permission denied" zu sterben. Erste Stelle, die HOST_DATA_DIR
# anfasst, ist die Logspiegelung; ensure_container ruft denselben Guard vor dem Mount.
ensure_host_data_dir() {
  [[ -d "$HOST_DATA_DIR" ]] && return 0
  if mkdir -p "$HOST_DATA_DIR" 2>/dev/null; then
    log "Host-Datenverzeichnis neu angelegt: $HOST_DATA_DIR"
    return 0
  fi
  err "Host-Datenverzeichnis nicht anlegbar: $HOST_DATA_DIR"
  err "  Dorthin mountet der Container /data (Checkpoints, Shader-Cache, Logs)."
  err "  Einmalig einen passenden Pfad hinterlegen:"
  err "    echo ': \"\${RL_HOST_DATA_DIR:=/pfad/mit/platz}\"' >> $REPO_DIR/.env.local"
  err "  oder pro Aufruf:  RL_HOST_DATA_DIR=/pfad/mit/platz $0 <aktion>"
  err "  Details: docs/portabilitaet.md"
  exit 1
}

start_logging() {
  ensure_host_data_dir
  LOG_DIR="$HOST_DATA_DIR/logs"
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/$1-$(date +%Y%m%d-%H%M%S).log"
  exec > >(tee -a "$LOG_FILE") 2>&1
  log "Log dieses Aufrufs: $LOG_FILE"
  log "Host-Datenverzeichnis: $HOST_DATA_DIR  (Container: /data)"
}

# ── Kleine Helfer ─────────────────────────────────────────────────────────────
require_docker() { command -v docker >/dev/null 2>&1 || { err "docker nicht gefunden."; exit 1; }; }

require_hf_token() {
  [[ -n "${HF_TOKEN:-}" ]] || { err "HF_TOKEN nicht gesetzt (für Checkpoint-Download nötig)."; \
      err "  Aufruf z. B.: HF_TOKEN=hf_... $0 $ACTION"; exit 1; }
}

container_state() { docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo missing; }

container_data_source() {
  docker inspect -f '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Source}}{{end}}{{end}}' \
    "$CONTAINER" 2>/dev/null || true
}

# Ein langlebiger Container behält seinen ursprünglichen /data-Mount. Wenn später
# RL_HOST_DATA_DIR geändert wird, würden Logs auf Host A landen, während Datensatz und
# Ergebnisse im Container weiterhin Host B benutzen. Dieser stille Split ist zu gefährlich.
validate_container_data_mount() {
  local actual expected
  actual="$(container_data_source)"
  [[ -n "$actual" ]] || { err "Container '$CONTAINER' hat keinen /data-Mount."; return 1; }
  expected="$(cd "$HOST_DATA_DIR" && pwd -P)"
  [[ -d "$actual" ]] && actual="$(cd "$actual" && pwd -P)"
  if [[ "$actual" != "$expected" ]]; then
    err "Datenpfade widersprechen sich:"
    err "  laufender Container /data -> $actual"
    err "  RL_HOST_DATA_DIR erwartet  -> $expected"
    err "Wenn der bestehende Datenbestand richtig ist, in .env.local setzen:"
    err "  : \"\${RL_HOST_DATA_DIR:=$actual}\""
    err "Soll stattdessen der neue Pfad gelten: '$0 clean' und danach erneut aufrufen."
    err "'clean' entfernt nur den Container, nicht die Datenverzeichnisse."
    return 1
  fi
}

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

# Adresse, unter der der Server vom Arbeitsrechner aus erreichbar ist. Der Container
# selbst kennt sie nicht (er sähe nur seine 172.x-Bridge-Adresse), deshalb ermitteln wir
# sie hier und reichen sie als LIVESTREAM_HOST_ADDR durch — nur für die Ausgabe.
host_addr() { echo "${LIVESTREAM_HOST_ADDR:-$(hostname -I 2>/dev/null | awk '{print $1}')}"; }

# -e-Flags für die LIVE-Variante (WebRTC-Viewport). Wirkt auf entrypoint_sim.sh,
# entrypoint_rl.sh und rl_finetune.py gleichermaßen, weil alle drei dieselben Var-Namen
# lesen.
livestream_docker_env() {
  LS_ENV=(
    -e "LIVESTREAM=${LIVESTREAM:-0}"
    -e "LIVESTREAM_PORT=$LIVESTREAM_PORT"
    -e "LIVESTREAM_MEDIA_PORT=$LIVESTREAM_MEDIA_PORT"
    -e "LIVE_KEEP_VIDEO=${LIVE_KEEP_VIDEO:-0}"
    -e "LIVESTREAM_HOST_ADDR=$(host_addr)"
  )
  [[ -n "${LIVESTREAM_SETTINGS_STYLE:-}" ]] && LS_ENV+=( -e "LIVESTREAM_SETTINGS_STYLE=$LIVESTREAM_SETTINGS_STYLE" )
  [[ -n "${LIVESTREAM_KIT_ARGS:-}" ]]       && LS_ENV+=( -e "LIVESTREAM_KIT_ARGS=$LIVESTREAM_KIT_ARGS" )
  [[ -n "${LIVESTREAM_UPDATE_EVERY_N:-}" ]] && LS_ENV+=( -e "LIVESTREAM_UPDATE_EVERY_N=$LIVESTREAM_UPDATE_EVERY_N" )
  return 0   # PFLICHT, s. build_rl_env
}

# /scripts liegt im Image GEBACKEN (nur $SIM_DIR ist gemountet). Jede Repo-Änderung an
# einem Entrypoint — und an lib_livestream.sh, die er sourct — muss deshalb vor dem Lauf
# hineinkopiert werden, sonst läuft still die alte Fassung aus dem Image.
sync_scripts() {
  local f
  for f in lib_livestream.sh "$@"; do
    docker cp "$REPO_DIR/Simulation/scripts/$f" "$CONTAINER:/scripts/$f"
  done
}

# Sowohl -p als auch --gpus wirken NUR beim Anlegen des Containers. Ein langlebiger
# Container aus einem früheren Lauf hat sie also nicht, egal was hier gesetzt ist —
# und das äußert sich stumm: der Stream lauscht nur container-intern, bzw. das
# Referenzmodell rückt mangels zweiter Karte auf die erste zurück.
warn_if_container_stale() {
  # Im Host-Netzwerkmodus gibt es keine Port-Mappings, die fehlen könnten — die Prüfungen
  # unten würden dann durchweg falschen Alarm schlagen.
  local netmode; netmode="$(docker inspect -f '{{.HostConfig.NetworkMode}}' "$CONTAINER" 2>/dev/null || echo '')"
  if [[ "$netmode" == "host" ]]; then
    [[ "${RL_NETWORK_MODE:-bridge}" == "host" ]] \
      || log "Container läuft im host-Netzwerkmodus (Ports liegen direkt auf dem Host)."
    return 0
  fi
  if [[ "${RL_NETWORK_MODE:-bridge}" == "host" && -n "$netmode" ]]; then
    warn "RL_NETWORK_MODE=host, aber '$CONTAINER' läuft im Modus '$netmode' —"
    warn "  --network wirkt nur beim ANLEGEN. Umstellen:  $0 clean   (Daten bleiben)"
  fi
  local ports; ports="$(docker inspect -f '{{json .NetworkSettings.Ports}}' "$CONTAINER" 2>/dev/null || echo '{}')"
  if [[ "${LIVE_VIEW:-0}" != "0" ]]; then
    if [[ "$ports" != *"\"$LIVE_VIEW_PORT/tcp\":[{"* ]]; then
      warn "LIVE_VIEW=1, aber Container '$CONTAINER' hat Port $LIVE_VIEW_PORT NICHT veröffentlicht."
      warn "  Einmalig neu anlegen:  $0 clean   (Daten unter $HOST_DATA_DIR bleiben)"
    fi
  fi
  # Dasselbe für die LIVE-Variante — und hier ist es besonders leicht zu übersehen: der
  # Signaling-Port (TCP) kann durchgereicht sein, der Medien-Port (UDP) aber nicht. Dann
  # verbindet sich der Client, und das Bild bleibt schwarz.
  if livestream_active; then
    if [[ "$ports" != *"\"$LIVESTREAM_PORT/tcp\":[{"* ]]; then
      warn "LIVESTREAM=$LIVESTREAM, aber Port $LIVESTREAM_PORT/tcp ist am Container NICHT veröffentlicht."
      warn "  Einmalig neu anlegen:  $0 clean   (Daten unter $HOST_DATA_DIR bleiben)"
    fi
    if [[ "$ports" != *"\"$LIVESTREAM_MEDIA_PORT/udp\":[{"* ]]; then
      warn "Medien-Port $LIVESTREAM_MEDIA_PORT/udp ist am Container NICHT veröffentlicht —"
      warn "  der Client verbindet sich dann zwar, das Bild bleibt aber schwarz."
      warn "  Einmalig neu anlegen:  $0 clean"
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
    ensure_host_data_dir
    validate_container_data_mount || return 1
    warn_if_container_stale
    return 0
  elif [[ "$state" == "false" ]]; then
    ensure_host_data_dir
    validate_container_data_mount || return 1
    log "Container '$CONTAINER' vorhanden (gestoppt) — starte ihn."
    docker start "$CONTAINER" >/dev/null
    warn_if_container_stale
    return 0
  fi
  ensure_host_data_dir
  log "Erzeuge langlebigen Container '$CONTAINER' (Image: $IMAGE, GPU: $GPUS)."
  log "  /data -> $HOST_DATA_DIR  (Checkpoint-Cache, RL-Checkpoints, Isaac-Sim-Shader-Cache)"
  local create_env=( -e PYTHONUNBUFFERED=1 )
  [[ -n "${HF_TOKEN:-}" ]]         && create_env+=( -e "HF_TOKEN=$HF_TOKEN" )
  [[ -n "${WANDB_API_KEY:-}" ]]    && create_env+=( -e "WANDB_API_KEY=$WANDB_API_KEY" )
  # g1_dex3_sim aus dem Repo ÜBER die Image-Kopie mounten (Muster wie kisski_submit.sh /
  # server_robocasa_ref_run.sh): LIVE-CHECK-Iterationen an rl_finetune.py & Co. brauchen dann
  # nur `git pull` auf dem Server — kein Image-Rebuild, kein `clean`.
  log "  Sim-Code-Mount -> $REPO_DIR/Simulation/g1_dex3_sim"
  # ── Netzwerkmodus (RL_NETWORK_MODE) ────────────────────────────────────────
  # NVIDIAs Docker-Anleitung zu Isaac Sim 6.0 sagt ausdrücklich: „--network=host is required
  # for WebRTC livestreaming" — das Streaming-SDK brauche direkten Zugriff auf die
  # Netzwerk-Interfaces, um seine UDP-Sockets korrekt zu binden. Wir fahren hier seit jeher
  # Bridge + 1:1-Port-Mapping, was theoretisch reichen sollte (intern == extern, damit die
  # SDP-Aushandlung stimmt) und für Spur B (reines HTTP) auch nachweislich reicht.
  # Verifiziert ist der WebRTC-Fall bei uns aber NICHT. Bleibt der Viewport schwarz, obwohl
  # 47998/udp offen ist, ist das hier der erste Verdacht:
  #     RL_NETWORK_MODE=host $0 clean   &&   … LIVESTREAM=2 $0 view
  # Default bleibt bridge — ein stiller Wechsel des Netzwerkmodells wäre die Sorte Änderung,
  # die man später nicht mehr aus den Ergebnissen herausrechnen kann.
  local net_flag=()
  if [[ "${RL_NETWORK_MODE:-bridge}" == "host" ]]; then
    net_flag=( --network=host )
    log "  Netzwerk      -> host (RL_NETWORK_MODE=host). Port-Mappings entfallen dabei —"
    log "                   im Host-Modus lauschen alle Dienste direkt auf den Host-Ports."
    if ! docker run -d --name "$CONTAINER" --gpus "$GPUS" --ipc=host --shm-size="$SHM_SIZE" \
      "${create_env[@]}" "${net_flag[@]}" \
      -v "$HOST_DATA_DIR:/data" \
      -v "$REPO_DIR/Simulation/g1_dex3_sim:$SIM_DIR:ro" \
      --entrypoint bash "$IMAGE" -lc "sleep infinity" >/dev/null; then
      err "Container '$CONTAINER' konnte im Host-Netzwerkmodus nicht gestartet werden."
      return 1
    fi
    ok "Container läuft (Netzwerk: host)."
    return 0
  fi

  local port_flag=()
  if port_free "$LIVE_VIEW_PORT"; then
    port_flag=( -p "$LIVE_VIEW_PORT:$LIVE_VIEW_PORT" )
    log "  Live-Ansicht  -> Port $LIVE_VIEW_PORT veröffentlicht (nutzbar mit LIVE_VIEW=1)"
  else
    warn "Host-Port $LIVE_VIEW_PORT ist belegt — Live-Ansicht bleibt unveröffentlicht."
    warn "  Anderen Port wählen:  LIVE_VIEW_PORT=8901 $0 clean && … $0 rl"
  fi
  # LIVE-Variante: Signaling (TCP) + Medien (UDP). Auch hier unabhängig davon mappen, ob
  # LIVESTREAM gerade an ist — -p wirkt nur beim Anlegen. Intern==extern ist Pflicht, weil
  # WebRTC den Port in die SDP-Aushandlung einbettet.
  local ls_port_flag=()
  if port_free "$LIVESTREAM_PORT"; then
    ls_port_flag=( -p "$LIVESTREAM_PORT:$LIVESTREAM_PORT/tcp" -p "$LIVESTREAM_MEDIA_PORT:$LIVESTREAM_MEDIA_PORT/udp" )
    log "  LIVE-Variante -> $LIVESTREAM_PORT/tcp + $LIVESTREAM_MEDIA_PORT/udp veröffentlicht (nutzbar mit LIVESTREAM=2)"
  else
    warn "Host-Port $LIVESTREAM_PORT ist belegt — WebRTC-Viewport bleibt unveröffentlicht."
    warn "  Anderen Port wählen:  LIVESTREAM_PORT=49101 $0 clean && … LIVESTREAM=2 $0 eval"
  fi
  # Fehlschlag-Pfad: den UDP-Port kann `port_free` (reines /dev/tcp) nicht prüfen. Ist er
  # belegt, bricht `docker run` mit "port is already allocated" ab — dann lieber ohne die
  # Livestream-Ports weitermachen, als den ganzen Workflow zu blockieren.
  if ! docker run -d --name "$CONTAINER" --gpus "$GPUS" --ipc=host --shm-size="$SHM_SIZE" \
    "${create_env[@]}" "${port_flag[@]}" "${ls_port_flag[@]}" \
    -v "$HOST_DATA_DIR:/data" \
    -v "$REPO_DIR/Simulation/g1_dex3_sim:$SIM_DIR:ro" \
    --entrypoint bash "$IMAGE" -lc "sleep infinity" >/dev/null 2>&1; then
    if [[ ${#ls_port_flag[@]} -gt 0 ]]; then
      warn "Container-Start mit den Livestream-Ports fehlgeschlagen (belegt?) — versuche es ohne."
      warn "  Die LIVE-Variante ist dann erst nach '$0 clean' mit freien Ports nutzbar."
      docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
      docker run -d --name "$CONTAINER" --gpus "$GPUS" --ipc=host --shm-size="$SHM_SIZE" \
        "${create_env[@]}" "${port_flag[@]}" \
        -v "$HOST_DATA_DIR:/data" \
        -v "$REPO_DIR/Simulation/g1_dex3_sim:$SIM_DIR:ro" \
        --entrypoint bash "$IMAGE" -lc "sleep infinity" >/dev/null
    else
      err "Container '$CONTAINER' konnte nicht gestartet werden."
      return 1
    fi
  fi
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
  # Domain-Gap-Hebel Hintergrund (Albedo statt Belichtung, gemessen in runs/20260808/20).
  # Die Handfarbe läuft NICHT hierüber, sondern offline übers Asset — s. ensure_black_hands.
  [[ -n "${RL_GROUND_COLOR:-}" ]]    && RL_ENV+=( -e "RL_GROUND_COLOR=$RL_GROUND_COLOR" )
  [[ -n "${DR_ENABLED:-}" ]]         && RL_ENV+=( -e "DR_ENABLED=$DR_ENABLED" )
  # Gegen Fragmentierung — der OOM-Traceback empfahl es selbst (1,13 GB reserviert,
  # aber unbenutzt). Ueberschreibbar, falls es auf dieser Torch-Version stoert.
  RL_ENV+=( -e "PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}" )
  [[ -n "${SHELL_ON_ERROR:-}" ]]   && RL_ENV+=( -e "SHELL_ON_ERROR=$SHELL_ON_ERROR" )
  [[ -n "${RL_WANDB_VIDEO_EVERY:-}" ]] && RL_ENV+=( -e "RL_WANDB_VIDEO_EVERY=$RL_WANDB_VIDEO_EVERY" )
  live_view_env
  RL_ENV+=( "${LIVE_ENV[@]}" )
  livestream_docker_env
  RL_ENV+=( "${LS_ENV[@]}" )
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

# Stellt den ECHTEN Trainingsdatensatz im Container sicher (fuer die Aktion 'span').
#
# Der Sim-Container laedt von sich aus nur Checkpoint + USD. Der Datensatz braucht DREI
# Schritte, nicht nur einen Download — genau die Reihenfolge aus Training/scripts/entrypoint.sh:
#   1. HF-Download (~18 GB, LeRobot v3.0)
#   2. Konvertierung v3.0 -> v2.1 (legt zusaetzlich ein Backup *_v3.0 an -> Platzbedarf ~2x)
#   3. modality_4cam.json -> meta/modality.json   (erst das macht ihn fuer gr00t lesbar)
# Ein blosser `huggingface-cli download` reicht NICHT: ohne Schritt 2+3 fehlt modality.json
# und der Loader scheitert.
#
# Automatisch, weil ensure_checkpoint es mit seinen ~10 GB genauso haelt und HF_TOKEN
# ohnehin schon uebergeben wurde. Abschalten mit SPAN_AUTO_FETCH=0.
ensure_dataset() {
  local ds="$1"
  if docker exec "$CONTAINER" test -f "$ds/meta/modality.json"; then
    ok "Datensatz einsatzbereit: $ds"
    return 0
  fi

  if [[ "${SPAN_AUTO_FETCH:-1}" != "1" ]]; then
    err "Kein Datensatz mit meta/modality.json unter $ds (SPAN_AUTO_FETCH=0)."
    err "  Vorhandenen Pfad angeben:  SPAN_DATASET=/data/... $0 span"
    return 1
  fi

  require_hf_token
  local repo="unitreerobotics/G1_Dex3_BlockStacking_Dataset"

  if ! docker exec "$CONTAINER" bash -lc "[ -n \"\$(ls -A '$ds' 2>/dev/null)\" ]"; then
    warn "Datensatz fehlt. Lade ihn jetzt: ~18 GB Download, danach Konvertierung mit"
    warn "  Backup-Kopie — rechne mit ~40 GB Spitzenbedarf unter $HOST_DATA_DIR und"
    warn "  einer knappen Stunde. Einmalig; ein 'clean' loescht ihn nicht."
    log "Download $repo -> $ds"
    docker exec -e "HF_TOKEN=$HF_TOKEN" -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" "$CONTAINER" \
      bash -lc "huggingface-cli download '$repo' --repo-type dataset --local-dir '$ds'" \
      || { err "Download fehlgeschlagen."; return 1; }
    ok "Download abgeschlossen."
  else
    log "Datensatz vorhanden, aber ohne modality.json — nur Konvertierung nachholen."
  fi

  # ffmpeg: die Konvertierung schneidet die zusammenhaengenden MP4s in Einzel-Episoden
  # (_extract_video_segment ruft es als Subprozess). Im Training-Image ist es drin, im
  # Sim-Image fehlte es bis 2026-08-12 — der Sim-Pfad brauchte den Datensatz nie.
  # Dockerfile.vastai hat es jetzt; bis zum naechsten Rebuild wird es hier nachinstalliert,
  # damit ein 60-Minuten-Rebuild nicht zwischen dir und der Messung steht.
  if ! docker exec "$CONTAINER" bash -lc "command -v ffmpeg >/dev/null"; then
    warn "ffmpeg fehlt im Container (Image aelter als der Dockerfile-Fix) — installiere es."
    docker exec "$CONTAINER" bash -lc \
      "apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg" \
      || { err "ffmpeg-Installation fehlgeschlagen. Image neu bauen:"
           err "  ./Simulation/update_sim_image.sh --vastai"; return 1; }
    ok "ffmpeg installiert (nur in diesem Container; ueberlebt 'clean' nicht)."
  fi

  # Konvertierung + modality.json: identisch zu entrypoint.sh, nur mit dem venv-Python
  # statt `uv run` (im Sim-Image ist das GR00T-venv der Interpreter fuer gr00t).
  log "Konvertiere LeRobot v3.0 -> v2.1 (dauert; schreibt Backup ${ds##*/}_v3.0)."
  docker exec "$CONTAINER" bash -lc "
    unset VIRTUAL_ENV
    cd '$GROOT_ROOT' && '$GROOT_ROOT/.venv/bin/python' \
        scripts/lerobot_conversion/convert_v3_to_v2_standalone.py \
        --repo-id '$repo' --root /data" \
    || { err "Konvertierung fehlgeschlagen."; return 1; }

  docker exec "$CONTAINER" bash -lc \
    "cp '$GROOT_ROOT/examples/G1_DEX3/modality_4cam.json' '$ds/meta/modality.json'" \
    || { err "modality.json konnte nicht kopiert werden."; return 1; }

  docker exec "$CONTAINER" test -f "$ds/meta/modality.json" \
    || { err "modality.json fehlt nach der Konvertierung — Ablauf pruefen."; return 1; }
  ok "Datensatz einsatzbereit: $ds"
}

# Stellt das schwarzhändige Asset sicher (Domain-Gap: reale DEX3 schwarz, URDF-Asset weiß).
# Reines USD-Authoring, keine GPU, wenige Sekunden — deshalb bei jedem Lauf geprüft statt
# einmalig dokumentiert. Schlägt der Recolor fehl, fällt ASSET_PATH aufs Original zurück,
# damit ein kosmetischer Fehler keinen Lauf verhindert.
ensure_black_hands() {
  [[ "$BLACK_HANDS" == "1" ]] || return 0
  [[ "$ASSET_PATH" == *g1_dex3_blackhands.usd ]] || return 0
  local orig="$CHECKPOINT_PATH/g1_dex3.usd"

  if docker exec "$CONTAINER" test -f "$ASSET_PATH"; then
    ok "Schwarzhändiges Asset vorhanden: $ASSET_PATH"
    return 0
  fi
  if ! docker exec "$CONTAINER" test -f "$orig"; then
    warn "Weder $ASSET_PATH noch $orig im Container — Asset-Pfad prüfen."
    return 0
  fi
  # Ausgabe MUSS neben das Original: der Wrapper referenziert configuration/ relativ.
  log "Erzeuge schwarzhändiges Asset (Recolor, offline auf dem USD)."
  if docker exec "$CONTAINER" bash -lc "
      unset VIRTUAL_ENV
      '$ISAAC_PY' '$SIM_DIR/recolor_hands_black.py' --in '$orig' --out '$ASSET_PATH'"; then
    ok "Asset erzeugt: $ASSET_PATH"
  else
    warn "Recolor fehlgeschlagen — Fallback auf $orig (weiße Hände)."
    ASSET_PATH="$orig"
  fi
}

# Stellt ein USD-Asset im Container sicher — OHNE HuggingFace, ohne Checkpoint, ohne Token.
#
# Der Unterschied zu ensure_checkpoint(): dort kommen ~10 GB Modellgewichte herunter, weil
# jede messende Aktion eine Policy braucht. 'view' braucht keine — nur die Geometrie. Und
# die liegt bereits im Repo (data/g1_dex3.usd + data/configuration/, zusammen ~40 MB,
# erzeugt von convert_urdf_to_usd.py). Deshalb wird der Reihe nach im Container gesucht und
# erst als letzter Schritt vom Host hineinkopiert.
#
# Der Wrapper referenziert configuration/ RELATIV zu sich selbst — Datei und Verzeichnis
# muessen also zusammen an denselben Ort. Ziel ist /data/assets: liegt im Bind-Mount und
# ueberlebt damit ein 'clean'.
#
# Setzt ASSET_PATH auf den gefundenen bzw. kopierten Pfad.
ensure_asset_local() {
  ensure_container
  local dst="${VIEW_ASSET_DIR:-/data/assets}"

  # Reihenfolge: ein bereits gesetztes ASSET_PATH gewinnt, danach alles, was frühere Läufe
  # oder das Image hinterlassen haben. BLACK_HANDS entscheidet, welche Variante bevorzugt wird.
  local names=( g1_dex3_blackhands.usd g1_dex3.usd )
  [[ "$BLACK_HANDS" == "1" ]] || names=( g1_dex3.usd g1_dex3_blackhands.usd )

  local cands=( "$ASSET_PATH" ) dir nm c
  for dir in "$CHECKPOINT_PATH" /workspace/assets "$dst"; do
    for nm in "${names[@]}"; do cands+=( "$dir/$nm" ); done
  done
  for c in "${cands[@]}"; do
    [[ -n "$c" ]] || continue
    if docker exec "$CONTAINER" test -f "$c"; then
      ASSET_PATH="$c"
      ok "USD-Asset im Container gefunden: $ASSET_PATH"
      return 0
    fi
  done

  # Nichts im Container — vom Host kopieren.
  local host_dir="${VIEW_HOST_ASSET_DIR:-$REPO_DIR/data}"
  local want=""
  for nm in "${names[@]}"; do
    [[ -f "$host_dir/$nm" ]] && { want="$nm"; break; }
  done
  if [[ -z "$want" || ! -d "$host_dir/configuration" ]]; then
    err "Kein USD-Asset gefunden — weder im Container noch unter $host_dir."
    err "  Erwartet: $host_dir/g1_dex3.usd  UND  $host_dir/configuration/  (gehoeren zusammen,"
    err "  der Wrapper referenziert configuration/ relativ zu sich selbst)."
    err "  Drei Wege:"
    err "    1. Anderen Ort angeben:  VIEW_HOST_ASSET_DIR=/pfad/zu/usd $0 $ACTION"
    err "    2. Aus dem URDF erzeugen (braucht Isaac Sim, laeuft IM Container —"
    err "       docs/simulation/vastai-anleitung.md Schritt 3):"
    err "         $0 shell"
    err "         unset VIRTUAL_ENV && '\$ISAACLAB_PATH/isaaclab.sh' -p \\"
    err "             $SIM_DIR/convert_urdf_to_usd.py --headless \\"
    err "             --urdf /data/assets/unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf \\"
    err "             --output ${dst}/g1_dex3.usd"
    err "    3. Checkpoint samt USD von HF holen (braucht HF_TOKEN, ~10 GB):  $0 setup"
    return 1
  fi

  log "Kopiere USD-Asset vom Host in den Container: $host_dir/$want (+ configuration/, ~40 MB)."
  docker exec "$CONTAINER" mkdir -p "$dst/configuration"
  docker cp "$host_dir/$want" "$CONTAINER:$dst/$want" \
    || { err "Kopieren von $want fehlgeschlagen."; return 1; }
  docker cp "$host_dir/configuration/." "$CONTAINER:$dst/configuration/" \
    || { err "Kopieren von configuration/ fehlgeschlagen."; return 1; }
  ASSET_PATH="$dst/$want"
  ok "USD-Asset bereit: $ASSET_PATH  (bleibt in $HOST_DATA_DIR/${dst#/data/}, überlebt 'clean')"
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
  if docker run --rm --gpus "$GPUS" "$IMAGE" \
      "$GROOT_ROOT/.venv/bin/python" -c \
      "import onnx,tensorrt as trt; print('onnx',onnx.__version__,'tensorrt',trt.__version__); assert trt.Builder(trt.Logger())"; then
    ok "ONNX + TensorRT-cu12 im GR00T-venv nutzbar."
  else
    err "ONNX/TensorRT-Preflight fehlgeschlagen — Image mit aktuellem Dockerfile.vastai bauen."
    return 1
  fi
}

do_setup() { ensure_checkpoint; ok "Setup abgeschlossen. Weiter mit:  $0 check"; }

do_check() {
  ensure_checkpoint
  ensure_black_hands
  log "LIVE-CHECK (Schritt 5/7 in rl-anleitung.md): Aufbau von Env+Policy+Critic, KEIN Training."
  # Exit-Code des Kit-Pythons ist NICHT belastbar: Isaac Sim beendet auch nach einem
  # Python-Traceback mit 0 (beobachtet 2026-08-07: TypeError → trotzdem Exit 0). Erfolg
  # daher ausschließlich am positiven Marker von rl_finetune.py --check festmachen.
  local out
  # LIVE_VIEW mitgeben: rl_finetune.py legt die Live-Ansicht VOR dem --check-Return an,
  # der Check weist damit auch Pillow + Port-Bindung im Kit-Python nach.
  # LIVESTREAM ebenso: mit LIVESTREAM=2 ist `check` der schnellste Nachweis, dass Isaac Sim
  # im Livestream-Modus ueberhaupt hochkommt (Sekunden statt einer ganzen Eval) — der
  # Viewport steht dann allerdings nur kurz, bis der Aufbau geprueft ist.
  live_view_env
  livestream_docker_env
  out=$(docker exec -w "$SIM_DIR" "${LIVE_ENV[@]}" "${LS_ENV[@]}" \
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
  ensure_black_hands   # muss VOR build_rl_env laufen: der Fallback ändert ASSET_PATH
  # Wie in do_eval: /scripts liegt im Image, nicht im Mount. Ohne diese Zeile liefe ein
  # Repo-Edit an entrypoint_rl.sh ins Leere und der Lauf nähme still die alte Fassung.
  sync_scripts entrypoint_rl.sh
  build_rl_env
  log "Starte echten RL-Lauf (Vordergrund, laeuft je nach RL_ITERATIONS lange)."
  log "  Checkpoints -> $HOST_DATA_DIR/g1_dex3_rl/  (alle RL_SAVE_EVERY Iterationen)"
  if [[ "${LIVE_VIEW:-0}" != "0" ]]; then
    log "  Live-Ansicht -> http://$(host_addr):$LIVE_VIEW_PORT/"
  fi
  if livestream_active; then
    warn "LIVESTREAM=$LIVESTREAM im RL-Lauf: der WebRTC-Viewport haelt genau EINE Sitzung"
    warn "  und ueberlebt keinen Netzabbruch — ueber Stunden/Tage ist LIVE_VIEW=1 (Spur B)"
    warn "  die passendere Wahl. Beide zugleich sind moeglich."
    livestream_banner "$(host_addr)"
  fi
  docker exec "${RL_ENV[@]}" "$CONTAINER" bash -lc "bash /scripts/entrypoint_rl.sh"
}

# Schritt 3 der Diagnosekette (docs/ergebnisse/diagnose-chronik.md): BC-Erfolgsrate in der Sim.
#
# Misst die Zielgroesse direkt statt ueber den Proxy Domain-Gap — und entscheidet damit,
# ob sich ein TUNE_VISUAL=1-Lauf (~48 h H100) lohnt. Zugleich der Nullpunkt, gegen den
# jeder spaetere RL-Lauf zu vergleichen ist: ohne diese Zahl ist "RL hat geholfen" nicht
# belegbar.
#
# Fuehrt NICHT rl_finetune.py aus, sondern /scripts/entrypoint_sim.sh — das ist die
# vollstaendige Closed-Loop-Pipeline (GR00T-Policy-Server auf ZMQ + Isaac-Lab-Client) und
# unterscheidet sich in einem fuer die Zahl entscheidenden Punkt vom RL-Rollout: der Client
# fuehrt EXECUTION_HORIZON Schritte eines 16er-Chunks aus, waehrend rl_finetune.py jeden
# Step neu plant und 15 von 16 vorhergesagten Schritten wegwirft. Gemessen werden soll der
# Betriebsmodus, nicht der Trainingsmodus.
do_eval() {
  ensure_checkpoint
  ensure_black_hands   # muss VOR dem Lauf passieren: der Fallback aendert ASSET_PATH
  local eps="${NUM_EPISODES:-20}"
  local horizon="${EXECUTION_HORIZON:-8}"
  local ep_len="${EPISODE_LENGTH_S:-0}"
  log "BC-Erfolgsrate messen: $eps Episoden, Exec-Horizon $horizon," \
      "Episodenlaenge ${ep_len}s$([[ "$ep_len" == "0" ]] && echo ' (cfg-Default 300s)')," \
      "DR=${DR_ENABLED:-1}"
  if [[ "$ep_len" == "0" ]]; then
    warn "Ohne EPISODE_LENGTH_S laeuft jede Episode bis zu 9000 Steps mit 4 gerenderten"
    warn "  Kameras. Das kann bei 20 Episoden Stunden dauern. Anhaltspunkt: die"
    warn "  menschliche Teleop-Demo (replay_episode0.npz) braucht 1173 Steps = 39 s."
  fi
  # /scripts ist ins Image GEBACKEN (nur $SIM_DIR ist gemountet) — eine Aenderung an
  # entrypoint_sim.sh im Repo erreicht den Container sonst nie, und der Lauf liefe still
  # mit der alten Fassung. Gleiches Muster wie bei measure_domain_gap.py in do_gap.
  sync_scripts entrypoint_sim.sh groot_inference_backend.py run_groot_optimized_server.py
  if livestream_active; then
    livestream_banner "$(host_addr)"
  fi
  livestream_docker_env
  # SKIP_DOWNLOAD=1: ensure_checkpoint hat den Checkpoint bereits sichergestellt; ein
  # zweiter HF-Zugriff im Entrypoint braeuchte nur wieder einen Token.
  # BLACK_HANDS wird mitgereicht, obwohl entrypoint_sim.sh dieselbe Logik selbst hat —
  # es findet das von ensure_black_hands erzeugte *_blackhands.usd dann einfach vor.
  docker exec \
    -e "SKIP_DOWNLOAD=1" \
    -e "CHECKPOINT_PATH=$CHECKPOINT_PATH" \
    -e "ASSET_PATH=$ASSET_PATH" \
    -e "BLACK_HANDS=$BLACK_HANDS" \
    -e "NUM_EPISODES=$eps" \
    -e "EXECUTION_HORIZON=$horizon" \
    -e "GROOT_INFERENCE_BACKEND=${GROOT_INFERENCE_BACKEND:-eager}" \
    -e "GROOT_TRT_ENGINE_PATH=${GROOT_TRT_ENGINE_PATH:-}" \
    -e "CAMERA_RENDER_EVERY_N=${CAMERA_RENDER_EVERY_N:-1}" \
    -e "EPISODE_LENGTH_S=$ep_len" \
    -e "TASK_DESCRIPTION=${TASK_DESCRIPTION:-stack the blocks}" \
    -e "DR_ENABLED=${DR_ENABLED:-1}" \
    -e "RL_GROUND_COLOR=${RL_GROUND_COLOR:-}" \
    `# Render-Hebel (gemessen 2026-08-13: 94 % der Laufzeit sind Sim+Render, 5 % Inferenz).` \
    `# SCENE_CAM=0 ist gratis — cam_scene geht nur ins MP4, die Policy sieht sie nie.` \
    `# CAM_RES_SCALE veraendert dagegen die Modell-Eingabe: nur zum Zuschauen.` \
    -e "SCENE_CAM=${SCENE_CAM:-1}" \
    -e "CAM_RES_SCALE=${CAM_RES_SCALE:-1}" \
    -e "RL_AA_MODE=${RL_AA_MODE:-}" \
    "${LS_ENV[@]}" \
    "$CONTAINER" bash -lc "bash /scripts/entrypoint_sim.sh" 2>&1 \
    `# grep -c statt -q: -q steigt beim ersten Treffer aus, das vorgeschaltete tee bekommt` \
    `# SIGPIPE, und unter 'set -o pipefail' (Z. 53) galt der Lauf dann als gescheitert —` \
    `# in Lauf 24 als falscher Alarm "ohne Erfolgsmarker", obwohl der Marker da war.` \
    `# Isaac Sim schreibt nach '[eval] fertig.' noch Shutdown-Zeilen, daher trat es genau` \
    `# hier auf und bei dump/gap nicht. -c liest bis EOF und kann nicht früher schließen.` \
    | tee /dev/stderr | grep -c "\[eval\] fertig" >/dev/null \
    || { err "Sim-Eval ohne Erfolgsmarker beendet (Traceback oben)."; return 1; }
  ok "Ergebnisse: $HOST_DATA_DIR/sim_results/results.json"
  if livestream_active && [[ "${LIVE_KEEP_VIDEO:-0}" != "1" ]]; then
    ok "Videos:     keine — LIVE-Variante lief (LIVE_KEEP_VIDEO=1 schreibt sie zusaetzlich)"
  else
    ok "Videos:     $HOST_DATA_DIR/sim_videos/"
  fi
  docker exec "$CONTAINER" bash -lc \
    "python3 -c \"import json;d=json.load(open('/data/sim_results/results.json'));\
print('Erfolgsrate: %d/%d = %.1f%%' % (d['num_success'], d['num_episodes'], 100*d['success_rate']))\"" \
    2>/dev/null || true
}

# ONNX -> TensorRT auf der Ziel-GPU, plus reproduzierbarer Policy-/Closed-Loop-Benchmark.
# Ein TensorRT-Plan ist absichtlich NICHT portabel: Metadaten binden ihn an Checkpoint,
# GPU-Modell/Compute-Capability und TensorRT-Version. Deshalb laeuft diese Aktion auf dem
# IKR-Server und nicht beim Image-Build.
do_optimize() {
  local phase="${1:-all}"
  case "$phase" in export|build|validate|benchmark|all) ;; *)
    err "optimize-Phase '$phase' ungueltig (export|build|validate|benchmark|all)."; return 2 ;;
  esac
  ensure_checkpoint
  ensure_black_hands
  sync_scripts optimize_groot_inference.py groot_inference_backend.py \
               run_groot_optimized_server.py summarize_optimization.py

  local root="${GROOT_OPTIMIZED_ROOT:-/data/optimized}"
  local iters="${OPTIMIZE_ITERATIONS:-20}"
  local warmup="${OPTIMIZE_WARMUP:-5}"
  log "GR00T-Inferenz optimieren: Phase=$phase, Checkpoint=$CHECKPOINT_PATH"
  docker exec "$CONTAINER" bash -lc "
    unset VIRTUAL_ENV
    '$GROOT_ROOT/.venv/bin/python' /scripts/optimize_groot_inference.py '$phase' \
      --model-path '$CHECKPOINT_PATH' \
      --output-root '$root' \
      --iterations '$iters' --warmup '$warmup' \
      --workspace-mb '${OPTIMIZE_WORKSPACE_MB:-8192}' \
      --mean-tolerance '${OPTIMIZE_MEAN_TOLERANCE:-0.005}' \
      --max-tolerance '${OPTIMIZE_MAX_TOLERANCE:-0.05}'" 2>&1 \
    | tee /dev/stderr | grep -c "\[optimize\] fertig" >/dev/null \
    || { err "ONNX/TensorRT-Phase ohne Erfolgsmarker beendet (Traceback oben)."; return 1; }

  if [[ "$phase" == "all" ]]; then
    local artifact
    artifact="$(docker exec "$CONTAINER" python3 -c \
      "import json;print(json.load(open('$root/latest.json'))['artifact_dir'])")"
    local eps="${OPTIMIZE_SIM_EPISODES:-1}"
    local seconds="${OPTIMIZE_SIM_EPISODE_LENGTH_S:-20}"
    local horizon="${EXECUTION_HORIZON:-8}"
    log "Closed-Loop A/B: $eps Episode(n) à ${seconds}s; Eager/Render=1 gegen TensorRT/Render=$horizon."

    ( GROOT_INFERENCE_BACKEND=eager CAMERA_RENDER_EVERY_N=1 SCENE_CAM=1 \
      NUM_EPISODES="$eps" EPISODE_LENGTH_S="$seconds" DR_ENABLED=0 do_eval )
    docker exec "$CONTAINER" cp /data/sim_results/results.json \
      "$artifact/sim_baseline_eager.json"

    ( GROOT_INFERENCE_BACKEND=tensorrt CAMERA_RENDER_EVERY_N="$horizon" SCENE_CAM=0 \
      NUM_EPISODES="$eps" EPISODE_LENGTH_S="$seconds" DR_ENABLED=0 do_eval )
    docker exec "$CONTAINER" cp /data/sim_results/results.json \
      "$artifact/sim_optimized_tensorrt.json"

    docker exec "$CONTAINER" python3 /scripts/summarize_optimization.py \
      --baseline "$artifact/sim_baseline_eager.json" \
      --optimized "$artifact/sim_optimized_tensorrt.json" \
      --report "$artifact/benchmark_report.json" \
      --target-speedup "${OPTIMIZE_TARGET_SPEEDUP:-2.0}"
    ok "A/B-Bericht: $HOST_DATA_DIR/${artifact#/data/}/benchmark_report.json"
  fi
  ok "Optimierungsartefakte: $HOST_DATA_DIR/${root#/data/}/"
}

# Kamera-Diagnose: konfigurierte gegen tatsächlich gerenderte Pose + PNG je Kamera.
# Anlass: cam_scene zeigte in der Live-Ansicht nur Hintergrund, obwohl die konfigurierte
# Pose nachweislich Tisch + Roboter erfassen müsste.
do_cams() {
  ensure_checkpoint
  ensure_black_hands
  # Messhebel für die leeren Kamerabilder (je ein Lauf pro Wert, siehe docs/ergebnisse/diagnose-chronik.md):
  #   RL_SETTLE_STEPS   Render-Konvergenz
  #   RL_AA_MODE        Anti-Aliasing-Modus
  #   RL_DOME_INTENSITY Belichtung — Basiswert der Szene
  #   RL_DOME_SWEEP     Belichtung — mehrere Werte in EINEM Lauf, z. B. "500,120,30"
  #                     (WIDERLEGT als Domain-Gap-Hebel, runs/20260808/20: 25-fache
  #                      Lichtspanne bewegt den Gap um 0,013)
  #   RL_GROUND_COLOR   Bodenfarbe, z. B. "0.35,0.35,0.36" statt Isaacs schwarzem Raster
  #                     (0.75,0.73,0.70 war zu hell: Kontrast 39 gegen real 59, runs/…/21)
  #   BLACK_HANDS=0     Recolor der Hände aus (Default 1, s. ensure_black_hands)
  #   DR_ENABLED=0      visuelle Domain Randomization aus (WIDERLEGT, Lauf 11)
  #   RL_CAMERA_CLASS=camera  gewöhnliche Camera statt TiledCamera (Halbierungstest)
  # Alte Sweep-Varianten weg: sie überleben sonst den nächsten Lauf und 'gap' vergleicht
  # frische Basisbilder gegen Varianten von gestern (passiert in runs/20260808/21 —
  # dessen Sweep-Zeilen sind byteidentisch mit Lauf 20).
  docker exec "$CONTAINER" bash -lc "rm -f /data/cam_dump/*__*.png" 2>/dev/null || true
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
    -e "RL_GROUND_COLOR=${RL_GROUND_COLOR:-}" \
    "$CONTAINER" bash -lc "
    unset VIRTUAL_ENV
    '$ISAAC_PY' '$SIM_DIR/dump_camera_poses.py' \
        --headless --enable_cameras \
        --asset-path '$ASSET_PATH' \
        --settle-steps ${RL_SETTLE_STEPS:-8} \
        --num-envs ${RL_NUM_ENVS:-4}" 2>&1 | tee /dev/stderr | grep -c "\[dump\] fertig" >/dev/null \
    && ok "PNGs + Posen unter $HOST_DATA_DIR/cam_dump/" \
    || { err "Kamera-Dump ohne Erfolgsmarker beendet (Traceback oben)."; return 1; }
}

# Greif-Physik isoliert: Open-Loop-Replay der ECHTEN Dataset-Aktionen, kein Modell, kein
# Server. Mit --grasp-test liegen die Wuerfel exakt an den aufgezeichneten Greifpunkten —
# damit haengt das Ergebnis nur noch an Reibung/Kontakt/Fingerschluss, nicht mehr an der
# Politik. Trennt die beiden Erklaerungen, die 'eval' offenlaesst, wenn die Fingerspitzen
# den Wuerfel erreichen (4 cm, runs/20260808/25), aber kein Stapel entsteht:
#   Anhebung > ~2 cm  -> Greifen funktioniert -> es liegt an der Politik (TUNE_VISUAL=1)
#   Anhebung ~ 0      -> Kontaktparameter zuerst, ein ViT-Lauf waere verschwendet
do_grasp() {
  ensure_checkpoint
  ensure_black_hands
  # GRASP_MODE=test  → Würfel an die geschätzten Greifpunkte (Platzierung + Physik zusammen)
  # GRASP_MODE=hold  → Würfel im Moment des Zugreifens zwischen die Fingerspitzen setzen;
  #                    prüft die Greif-Physik allein, ohne jede Platzierungs-Annahme.
  local grasp_flag="--grasp-test"
  [ "${GRASP_MODE:-test}" = "hold" ] && grasp_flag="--grasp-hold"
  # App-Flags aus derselben Lib wie die Entrypoints: entweder --headless oder
  # --livestream N [--kit_args …]. Der Aufruf hier geht direkt an das Python (kein
  # Entrypoint), deshalb muessen die Flags in den Kommando-String — %q quotet sie so,
  # dass die Kit-Settings-Zeile mit ihren Leerzeichen EIN Argument bleibt.
  livestream_app_flags
  local ls_flags="" f
  for f in "${LS_APP_FLAGS[@]}"; do ls_flags+=" $(printf '%q' "$f")"; done
  local vid; vid="$(livestream_video_dir /data/sim_videos_replay)"
  log "Greif-Physik-Test (Open-Loop-Replay, Dataset-Aktionen, Modus ${GRASP_MODE:-test})" \
      "→ $HOST_DATA_DIR/sim_results_replay/"
  if livestream_active; then
    livestream_banner "$(host_addr)"
  fi
  docker exec -w "$SIM_DIR" \
    -e "DR_ENABLED=${DR_ENABLED:-0}" \
    -e "RL_GROUND_COLOR=${RL_GROUND_COLOR:-}" \
    -e "SCENE_CAM=${SCENE_CAM:-1}" \
    -e "CAM_RES_SCALE=${CAM_RES_SCALE:-1}" \
    -e "RL_AA_MODE=${RL_AA_MODE:-}" \
    "$CONTAINER" bash -lc "
    unset VIRTUAL_ENV
    '$ISAAC_PY' '$SIM_DIR/run_g1_dex3_replay.py' \
        $ls_flags --enable_cameras \
        --asset-path '$ASSET_PATH' \
        --video-dir '$vid' \
        $grasp_flag" 2>&1 | tee /dev/stderr | grep -c "\[replay\] fertig" >/dev/null \
    && ok "Ergebnis: $HOST_DATA_DIR/sim_results_replay/results.json${vid:+, Video unter sim_videos_replay/}" \
    || { err "Greif-Test ohne Erfolgsmarker beendet (Traceback oben)."; return 1; }
}

# Phase 0 der LIVE-Variante (livestream-plan.md §6): die drei Dinge pruefen, die den
# WebRTC-Viewport im Container scheitern lassen — bevor eine ganze Eval dafuer laeuft.
#   1. NVENC: ohne Hardware-Encoder kein WebRTC. Die RTX PRO 6000 Blackwell hat vier
#      Engines; A100/H100 haetten keine — dieselbe GPU-Regel wie fuers RT-Core-Rendering.
#   2. Livestream-Extension im Isaac-Sim-Bundle vorhanden?
#   3. Welche Kit-Settings-Pfade gelten (Isaac Sim 6.0 hat sie umbenannt, D2)? Kit
#      ignoriert falsche Pfade STILL — deshalb hier die Version ablesen, statt zu raten.
do_livecheck() {
  ensure_container
  log "Phase 0 der LIVE-Variante: NVENC, Livestream-Extension, Kit-Settings-Pfade."
  docker exec "$CONTAINER" bash -lc '
    echo "── Isaac-Sim-Version ──"
    for f in "${ISAACLAB_PATH:-/workspace/isaaclab}/_isaac_sim/VERSION" /isaac-sim/VERSION; do
      [ -r "$f" ] && { echo "  $f: $(head -n1 "$f")"; break; }
    done
    echo "── NVENC (Encoder-Sitzungen) ──"
    nvidia-smi -q -d ENCODER 2>/dev/null | grep -iA3 "Encoder Stats" | head -12 \
      || echo "  nvidia-smi liefert keine Encoder-Statistik"
    echo "── Livestream-Extensions im Bundle ──"
    find "${ISAACLAB_PATH:-/workspace/isaaclab}/_isaac_sim" -maxdepth 3 -type d \
         -name "*livestream*" 2>/dev/null | sed "s|^|  |" | head -10 \
      || echo "  keine gefunden"
  ' || warn "Teile des Checks liefen ins Leere — Ausgabe oben bewerten."
  echo ""
  log "Host-Seite:"
  printf "    %-24s %s\n" "Signaling (TCP):"  "$(host_addr):$LIVESTREAM_PORT"
  printf "    %-24s %s\n" "Medien (UDP):"     "$(host_addr):$LIVESTREAM_MEDIA_PORT"
  local ports; ports="$(docker inspect -f '{{json .NetworkSettings.Ports}}' "$CONTAINER" 2>/dev/null || echo '{}')"
  if [[ "$ports" == *"\"$LIVESTREAM_PORT/tcp\":[{"* ]]; then
    ok "Port $LIVESTREAM_PORT/tcp ist am Container veröffentlicht."
  else
    warn "Port $LIVESTREAM_PORT/tcp NICHT veröffentlicht — '$0 clean' und neu anlegen."
  fi
  if [[ "$ports" == *"\"$LIVESTREAM_MEDIA_PORT/udp\":[{"* ]]; then
    ok "Port $LIVESTREAM_MEDIA_PORT/udp ist am Container veröffentlicht."
  else
    warn "Port $LIVESTREAM_MEDIA_PORT/udp NICHT veröffentlicht — '$0 clean' und neu anlegen."
  fi
  echo ""
  log "Vom Arbeitsrechner aus prüfen (UDP ist der wahrscheinlichste Stolperstein):"
  echo "    nc -vz  $(host_addr) $LIVESTREAM_PORT        # Signaling erreichbar?"
  echo "    nc -vzu $(host_addr) $LIVESTREAM_MEDIA_PORT       # UDP durch die Firewall?"
  echo ""
  log "Danach:  LIVESTREAM=2 NUM_EPISODES=2 EPISODE_LENGTH_S=120 $0 eval"
}

# Fingerspanne der Policy auf ECHTEN Datensatz-Bildern (offene Schleife).
#
# Trennt die letzten beiden Erklaerungen fuer die gestauchten Fingerkommandos aus Lauf 30
# (0,39 rad statt 2,09 rad): Domain-Gap gegen "das Modell hat den Griff nie gelernt".
# Nur bei Ersterem ist ein TUNE_VISUAL-Lauf ueber ~48 GPU-Stunden begruendet.
#
# Laeuft im GR00T-venv, NICHT im Isaac-Python: hier wird kein Isaac Sim gebraucht, nur
# Policy-Inferenz auf Datensatz-Frames. Skript per `docker cp` wie bei do_gap — /scripts
# ist ins Image gebacken, ein Edit wuerde sonst einen Rebuild verlangen.
do_span() {
  ensure_checkpoint
  local ds="${SPAN_DATASET:-/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset}"
  local trajs="${SPAN_TRAJ_IDS:-0 1 2 3 4}"

  ensure_dataset "$ds" || return 1

  docker cp "$REPO_DIR/Simulation/scripts/finger_span_openloop.py" \
            "$CONTAINER:/workspace/finger_span_openloop.py"

  log "Fingerspanne auf echten Datensatz-Bildern messen (Trajektorien: $trajs)."
  docker exec "$CONTAINER" bash -lc "
    unset VIRTUAL_ENV
    '$GROOT_ROOT/.venv/bin/python' /workspace/finger_span_openloop.py \
        --model-path '$CHECKPOINT_PATH' \
        --dataset-path '$ds' \
        --json-out /data/finger_span_openloop.json \
        --traj-ids $trajs" 2>&1 | tee /dev/stderr \
    | grep -c "Median Vorhersage" >/dev/null \
    && ok "Ergebnis auch als JSON: $HOST_DATA_DIR/finger_span_openloop.json" \
    || { err "Fingerspannen-Test ohne Ergebniszeile beendet (Traceback oben)."; return 1; }
}

# Reine Policy-Latenz — die einzige hier messbare Zahl, die auch fuer echte Hardware gilt.
#
# Die Sim-Eval laeuft 3,5x langsamer als Echtzeit, aber 94 % davon sind Kamera-Rendering,
# das es auf einem realen Roboter nicht gibt. Was dort uebrig bleibt, ist die Zeit vom
# Observation-Dict bis zum Action-Chunk — und die misst dieses Skript in-process, also
# ohne ZMQ-Roundtrip (~3,7 MB unkomprimierte Bilder je Aufruf) und ohne Isaac Sim.
# Die Differenz zur Zahl aus der Eval ("Inferenz X ms/Aufruf") ist der Transport-Overhead.
#
# Laeuft im GR00T-venv wie 'span', nicht im Isaac-Python: kein Isaac Sim noetig.
do_latency() {
  ensure_checkpoint
  docker cp "$REPO_DIR/Simulation/scripts/policy_latency.py" \
            "$CONTAINER:/workspace/policy_latency.py"
  local iters="${LATENCY_ITERS:-50}"
  local horizon="${EXECUTION_HORIZON:-8}"
  log "Policy-Latenz messen ($iters Aufrufe, Budget-Bezug: Exec-Horizon $horizon bei 30 Hz)."
  docker exec "$CONTAINER" bash -lc "
    unset VIRTUAL_ENV
    '$GROOT_ROOT/.venv/bin/python' /workspace/policy_latency.py \
        --model-path '$CHECKPOINT_PATH' \
        --iterations $iters \
        --execution-horizon $horizon \
        --denoising-sweep '${LATENCY_SWEEP:-1,2,4}' \
        ${LATENCY_DEVICE:+--device '$LATENCY_DEVICE'} \
        --json-out /data/policy_latency.json" 2>&1 | tee /dev/stderr \
    | grep -c "\[latency\] fertig" >/dev/null \
    && ok "Ergebnis auch als JSON: $HOST_DATA_DIR/policy_latency.json" \
    || { err "Latenz-Messung ohne Erfolgsmarker beendet (Traceback oben)."; return 1; }
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

  # Warnung, keine Abbruchbedingung: mit DR=1 würfelt _randomize_visuals() die Dome-Intensität
  # je Episode aus [1000, 3800]. Die Frames sind dann bei EINER zufälligen Beleuchtung
  # entstanden — für Absolutwerte und erst recht für einen Sweep unbrauchbar.
  warn "Gilt nur für die Beleuchtung, unter der 'cams' lief. Für vergleichbare Zahlen:"
  warn "  DR_ENABLED=0 $0 cams        (feste Dome-Intensität statt zufälliger je Episode)"
  warn "  DR_ENABLED=0 RL_DOME_SWEEP=1000,500,200 $0 cams   -> 'gap' misst alle Stufen mit"

  log "Domain-Gap messen (SigLIP-ViT, Sim-Frames aus $sim_dir)."
  docker exec "${gap_env[@]}" "$CONTAINER" bash -lc "
    unset VIRTUAL_ENV
    '$ISAAC_PY' /workspace/measure_domain_gap.py \
        --real-dir /workspace/camera_reference \
        --sim-dir '$sim_dir'" 2>&1 | tee /dev/stderr | grep -c "\[gap\] fertig" >/dev/null \
    && ok "Ergebnisse auch als JSON unter $HOST_DATA_DIR/${sim_dir#/data/}/domain_gap_results.json" \
    || { err "Domain-Gap-Messung ohne Erfolgsmarker beendet (Traceback oben)."; return 1; }
}

# Gerenderten Co-Training-Datensatz erzeugen (Schritt 4 aus next-steps.md).
#
# Zwei Stufen in einem Aufruf: 'scan' findet je Episode den Punkt, an dem sich die Finger
# schliessen (billig, Kameras auf 1/10), 'render' legt die Wuerfel dorthin und schreibt den
# Datensatz in kalibrierter Aufloesung. Ohne den Scan laegen die Wuerfel zufaellig, und die
# Bild-Aktions-Paare waeren visuell entkoppelt — das Gegenteil dessen, was der Encoder lernen soll.
#
# Laufzeit: Rendern kostet rund 2-3 min je Episode (8,6 Steps/s gemessen, Episode = 1173
# Steps). 60 Episoden sind also ~3 h. Der Lauf ist fortsetzbar: fertige Episoden werden
# uebersprungen, ein Abbruch kostet hoechstens die angefangene Episode.
# Wuerfellage aus dem REALBILD lesen (Farbblob -> Strahl auf die Wuerfelebene).
#
# Warum das noetig ist: der Greifpunkt aus scan.json ist das Minimum der Fingeroeffnung ueber
# die GANZE Episode. Beim Pick-and-Place bleibt die Hand vom Zugreifen bis zum Ablegen
# geschlossen, das Minimum liegt also irgendwo auf dem Transportweg — 48 von 116 Griffen des
# Laufs vom 2026-08-17 jenseits von 60 % der Episode. Der Wuerfel landete fern vom echten
# Pick, und der Arm griff ins Leere, in x, y UND z.
#
# Braucht keine GPU und kein Isaac: numpy + imageio + Pillow. Minuten statt Stunden.
do_layout() {
  ensure_container
  local ds="${SPAN_DATASET:-/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset}"
  ensure_dataset "$ds" || return 1
  local out="${LAYOUT_OUT:-/data/cotrain/layout.json}"
  local eps="${RENDER_EPISODES:-60}"
  local dbg="${LAYOUT_DEBUG_DIR:-/data/cotrain/layout_debug}"
  local extra=""
  [[ -n "${LAYOUT_BIAS:-}" ]] && extra+=" --bias ${LAYOUT_BIAS}"
  [[ "${LAYOUT_OVERWRITE:-0}" == "1" ]] && extra+=" --overwrite"

  if ! docker exec "$CONTAINER" test -f "$SIM_DIR/extract_block_layout.py"; then
    err "extract_block_layout.py fehlt unter $SIM_DIR — auf dem Server:  git pull"
    return 1
  fi

  log "Wuerfellage aus den Realbildern lesen: $eps Episoden, Ziel $out"
  log "  Markierte Kontrollbilder: $HOST_DATA_DIR/${dbg#/data/}"
  docker exec -w "$SIM_DIR" "$CONTAINER" bash -lc "
    unset VIRTUAL_ENV
    '$ISAAC_PY' '$SIM_DIR/extract_block_layout.py' extract \
        --dataset-path '$ds' \
        --out '$out' \
        --num-episodes '$eps' \
        --debug-dir '$dbg' \
        $extra" 2>&1 | tee /dev/stderr | grep -c "\[layout\] fertig" >/dev/null \
    || { err "Layout-Extraktion ohne Erfolgsmarker beendet (Traceback oben)."; return 1; }
  ok "Layout: $HOST_DATA_DIR/${out#/data/}"
  echo "  Kontrollbilder ansehen (Kreuz = gefundener Wuerfel):"
  echo "    ls $HOST_DATA_DIR/${dbg#/data/}"
  echo "  Danach rendern:  RENDER_LAYOUT=$out ./Simulation/server_rl_run.sh render"
}

# Das Kameramodell gegen den RENDERER pruefen, nicht gegen den eigenen Verstand.
#
# In Lauf 13 (2026-08-08) lagen konfigurierte USD-Pose und cam.data 95,6° auseinander und
# drei Sim-Laeufe waren umsonst. Dieselbe Falle steht hier: eine Rueckprojektion ist nur so
# gut wie die Annahme, dass Isaac mit der konfigurierten Pose auch rendert. Deshalb wird ein
# GERENDERTES Bild mit BEKANNTEN Wuerfelpositionen durch denselben Detektor geschickt; das
# Residuum ist der Fehler von Kameramodell und Blob-Schwerpunkt zusammen.
do_layoutcheck() {
  ensure_container
  local frame="${LAYOUTCHECK_FRAME:?LAYOUTCHECK_FRAME=<PNG oder MP4 im Container> setzen}"
  local expect="${LAYOUTCHECK_EXPECT:?LAYOUTCHECK_EXPECT='[[x,y,z],[x,y,z],[x,y,z]]' setzen (render_manifest.json -> cubes_xyz)}"
  local cam="${LAYOUTCHECK_CAM:-cam_left_high}"
  local dbg="${LAYOUT_DEBUG_DIR:-/data/cotrain/layout_debug}"

  log "Kameramodell gegen gerendertes Bild pruefen: $frame ($cam)"
  docker exec -w "$SIM_DIR" "$CONTAINER" bash -lc "
    unset VIRTUAL_ENV
    '$ISAAC_PY' '$SIM_DIR/extract_block_layout.py' detect '$frame' \
        --camera '$cam' --expect '$expect' --debug-dir '$dbg'" || return 1
  echo
  echo "  Lesart: MITTEL = Bias des Schaetzers (der Blob-Schwerpunkt ist der Schwerpunkt der"
  echo "  sichtbaren Flaechen, nicht die Projektion des Wuerfelmittelpunkts) — per"
  echo "  LAYOUT_BIAS=\"dx dy\" (Meter) in 'layout' herausrechnen. STREUUNG = der Rest, der"
  echo "  bleibt; erst die entscheidet, ob das Layout brauchbar ist."
}

# Gemeinsame, als Bash-Array aufgebaute Episodenauswahl für die neuen Replay-Aktionen.
# Keine String-Konkatenation: explizite IDs bleiben einzelne, validierte CLI-Argumente.
replay_selection_args() {
  local count="${REPLAY_NUM_EPISODES:-10}"
  local start="${REPLAY_START_EPISODE:-0}"
  [[ "$count" =~ ^[1-9][0-9]*$ ]] \
    || { err "REPLAY_NUM_EPISODES muss eine positive Ganzzahl sein: '$count'"; return 1; }
  [[ "$start" =~ ^[0-9]+$ ]] \
    || { err "REPLAY_START_EPISODE muss eine nichtnegative Ganzzahl sein: '$start'"; return 1; }
  REPLAY_SELECTION_ARGS=( --num-episodes "$count" --start-episode "$start" )
  if [[ -n "${REPLAY_EPISODE_IDS:-}" ]]; then
    local ids=() id
    read -r -a ids <<<"$REPLAY_EPISODE_IDS"
    for id in "${ids[@]}"; do
      [[ "$id" =~ ^[0-9]+$ ]] \
        || { err "Ungültige Episode in REPLAY_EPISODE_IDS: '$id'"; return 1; }
    done
    REPLAY_SELECTION_ARGS+=( --episode-ids "${ids[@]}" )
  fi
}

ensure_replay_script() {
  local script="$1"
  if ! docker exec "$CONTAINER" test -f "$SIM_DIR/$script"; then
    err "$script fehlt unter $SIM_DIR im Container."
    err "  Das Verzeichnis ist read-only aus dem Repo gemountet; auf dem Server git pull."
    return 1
  fi
}

# Vollständigen v2.1-Quelldatensatz und das lokale G1+DEX3-Asset bereitstellen. Es werden
# bewusst keine GR00T-Modellgewichte geladen: der Replay benutzt ausschließlich Dataset-Actions.
do_replay_prepare() {
  ensure_asset_local || return 1
  local ds="${REPLAY_DATASET:-/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset}"
  ensure_dataset "$ds" || return 1
  ensure_replay_script reconstruct_cube_poses.py || return 1
  replay_selection_args || return 1

  log "Prüfe Replay-Quelldatensatz: $ds"
  docker exec -w "$SIM_DIR" "$CONTAINER" env -u VIRTUAL_ENV \
    "$ISAAC_PY" "$SIM_DIR/reconstruct_cube_poses.py" inspect \
      --dataset-path "$ds" "${REPLAY_SELECTION_ARGS[@]}" \
    2>&1 | tee /dev/stderr | grep -c "\[replay-prepare\] fertig" >/dev/null \
    || { err "Replay-Datensatzprüfung fehlgeschlagen (Ausgabe oben)."; return 1; }
  ok "Replay vorbereitet; Asset: $ASSET_PATH"
}

# Prüft die rekonstruierte Kopfkamera-Skala an den realen Würfelsilhouetten und schreibt
# die für Pose-Fit und Replay gemeinsame Geometrie. Kein Isaac-Start, also schnell.
do_replay_calibrate() {
  ensure_container
  local ds="${REPLAY_DATASET:-/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset}"
  ensure_dataset "$ds" || return 1
  ensure_replay_script reconstruct_cube_poses.py || return 1
  replay_selection_args || return 1
  local work="${REPLAY_WORK:-/data/cube_replay/work}"
  local calibration="${REPLAY_CALIBRATION:-$work/geometry_calibration.json}"
  local debug="${REPLAY_CALIBRATION_DEBUG_DIR:-$work/calibration_overlays}"

  log "Kalibriere Replay-Geometrie aus maximal ${REPLAY_NUM_EPISODES:-10} Episoden."
  docker exec -w "$SIM_DIR" "$CONTAINER" env -u VIRTUAL_ENV \
    "$ISAAC_PY" "$SIM_DIR/reconstruct_cube_poses.py" calibrate \
      --dataset-path "$ds" --out "$calibration" --debug-dir "$debug" \
      "${REPLAY_SELECTION_ARGS[@]}" \
    2>&1 | tee /dev/stderr | grep -c "\[replay-calibrate\] fertig" >/dev/null \
    || { err "Replay-Kalibrierung fehlgeschlagen (Ausgabe oben)."; return 1; }
  ok "Kalibrierung: $HOST_DATA_DIR/${calibration#/data/}"
  echo "  Kontrollbilder: $HOST_DATA_DIR/${debug#/data/}"
}

# Rekonstruiert genau eine Anfangspose je Würfel und Episode. Fehlende Farben führen zum
# Überspringen der Episode; es gibt keinen Greifpunkt- oder Zufalls-Fallback.
do_replay_poses() {
  ensure_container
  local ds="${REPLAY_DATASET:-/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset}"
  ensure_dataset "$ds" || return 1
  ensure_replay_script reconstruct_cube_poses.py || return 1
  replay_selection_args || return 1
  local work="${REPLAY_WORK:-/data/cube_replay/work}"
  local calibration="${REPLAY_CALIBRATION:-$work/geometry_calibration.json}"
  local poses="${REPLAY_POSES:-$work/cube_poses.json}"
  local debug="${REPLAY_POSE_DEBUG_DIR:-$work/pose_overlays}"
  local overwrite=()
  [[ "${REPLAY_OVERWRITE:-0}" == "1" ]] && overwrite=( --overwrite )

  if ! docker exec "$CONTAINER" test -f "$calibration"; then
    err "Replay-Kalibrierung fehlt: $calibration"
    err "  Zuerst: ./Simulation/server_rl_run.sh replay-calibrate"
    return 1
  fi
  log "Bestimme einmalige Würfel-Startposen für maximal ${REPLAY_NUM_EPISODES:-10} Episoden."
  docker exec -w "$SIM_DIR" "$CONTAINER" env -u VIRTUAL_ENV \
    "$ISAAC_PY" "$SIM_DIR/reconstruct_cube_poses.py" poses \
      --dataset-path "$ds" --calibration "$calibration" --out "$poses" \
      --debug-dir "$debug" "${overwrite[@]}" "${REPLAY_SELECTION_ARGS[@]}" \
    2>&1 | tee /dev/stderr | grep -c "\[replay-poses\] fertig" >/dev/null \
    || { err "Würfelpose-Rekonstruktion fehlgeschlagen (Ausgabe oben)."; return 1; }
  ok "Würfelposen: $HOST_DATA_DIR/${poses#/data/}"
  echo "  Kontrollbilder: $HOST_DATA_DIR/${debug#/data/}"
}

# Spielt die Originalaktionen bei exakt 30 Hz ab und schreibt fünf MP4s je Episode.
# Die Würfel werden nach env.reset genau einmal gesetzt und danach nie wieder beschrieben.
do_replay_render() {
  ensure_asset_local || return 1
  local ds="${REPLAY_DATASET:-/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset}"
  ensure_dataset "$ds" || return 1
  ensure_replay_script run_dataset_replay_videos.py || return 1
  replay_selection_args || return 1
  local work="${REPLAY_WORK:-/data/cube_replay/work}"
  local poses="${REPLAY_POSES:-$work/cube_poses.json}"
  local out="${REPLAY_OUT:-/data/cube_replay/videos}"
  local extra=()
  [[ "${REPLAY_OVERWRITE:-0}" == "1" ]] && extra+=( --overwrite )
  [[ "${REPLAY_MAX_FRAMES:-0}" != "0" ]] \
    && extra+=( --max-frames "${REPLAY_MAX_FRAMES}" )

  if ! docker exec "$CONTAINER" test -f "$poses"; then
    err "Würfelposen fehlen: $poses"
    err "  Zuerst: ./Simulation/server_rl_run.sh replay-poses"
    return 1
  fi
  log "Rendere maximal ${REPLAY_NUM_EPISODES:-10} physikbasierte Replay-Episoden nach $out."
  docker exec -w "$SIM_DIR" -e "DR_ENABLED=${DR_ENABLED:-0}" "$CONTAINER" \
    env -u VIRTUAL_ENV "$ISAAC_PY" "$SIM_DIR/run_dataset_replay_videos.py" \
      --headless --enable_cameras --dataset-path "$ds" --poses "$poses" \
      --out-dir "$out" --asset-path "$ASSET_PATH" \
      "${extra[@]}" "${REPLAY_SELECTION_ARGS[@]}" \
    2>&1 | tee /dev/stderr | grep -c "\[replay-render\] fertig" >/dev/null \
    || { err "Replay-Rendering fehlgeschlagen (Ausgabe oben)."; return 1; }
  ok "Replay-Videos: $HOST_DATA_DIR/${out#/data/}"
}

do_render() {
  ensure_checkpoint
  ensure_black_hands
  local ds="${SPAN_DATASET:-/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset}"
  ensure_dataset "$ds" || return 1

  local out="${RENDER_OUT:-/data/cotrain/g1_dex3_rendered}"
  local eps="${RENDER_EPISODES:-60}"
  local stage="${RENDER_STAGE:-both}"
  local extra=""
  [[ "${RENDER_MAX_FRAMES:-0}" != "0" ]] && extra+=" --max-frames-per-episode ${RENDER_MAX_FRAMES}"
  [[ "${RENDER_OVERWRITE:-0}" == "1" ]] && extra+=" --overwrite"
  [[ -n "${RENDER_EPISODE_IDS:-}" ]] && extra+=" --episode-ids ${RENDER_EPISODE_IDS}"
  # Nur bis zum Griff rendern. Ab dem Griff entscheidet die Kontaktphysik ueber die
  # Wuerfellage, und die greift im Replay meist nicht — das Bild zeigt dann etwas anderes,
  # als die Aktion beschreibt. Default AN, weil die Alternative falsch beschriftete Paare
  # sind; RENDER_STOP_AT_GRASP=0 stellt das alte Verhalten wieder her.
  # Wuerfellage aus dem Realbild. Default ist Pflicht, nicht Angebot: ohne Layout landen die
  # Wuerfel am Greifpunkt aus scan.json, und der liegt bei knapp der Haelfte der Griffe auf
  # dem Transportweg statt am Pick. RENDER_LAYOUT=none erzwingt den alten Weg.
  local layout="${RENDER_LAYOUT:-/data/cotrain/layout.json}"
  if [[ "$layout" == "none" ]]; then
    warn "RENDER_LAYOUT=none — Wuerfel kommen vom Greifpunkt aus scan.json."
    warn "  Das ist der Modus, in dem der Arm ins Leere greift. Nur fuer Vergleichslaeufe."
  elif docker exec "$CONTAINER" test -f "$layout"; then
    extra+=" --layout ${layout}"
  else
    err "Layout fehlt: $layout"
    err "  Erst:  RENDER_EPISODES=${eps} ./Simulation/server_rl_run.sh layout"
    err "  Bewusst ohne Layout rendern:  RENDER_LAYOUT=none"
    return 1
  fi

  if [[ "${RENDER_STOP_AT_GRASP:-1}" == "1" ]]; then
    extra+=" --stop-at-grasp"
    [[ "${RENDER_GRASP_WINDOW:-0}" != "0" ]] && extra+=" --grasp-window ${RENDER_GRASP_WINDOW}"
    [[ -n "${RENDER_MIN_WINDOW:-}" ]] && extra+=" --min-window ${RENDER_MIN_WINDOW}"
  fi

  # Kein docker cp noetig — und auch nicht moeglich: $SIM_DIR ist das Repo-Verzeichnis
  # Simulation/g1_dex3_sim, READ-ONLY hineingemountet (s. ensure_container). Ein 'git pull'
  # auf dem Server genuegt also. Genau deshalb wird hier geprueft statt kopiert.
  if ! docker exec "$CONTAINER" test -f "$SIM_DIR/render_cotrain_dataset.py"; then
    err "render_cotrain_dataset.py fehlt unter $SIM_DIR im Container."
    err "  $SIM_DIR ist read-only aus dem Repo gemountet — auf dem Server:  git pull"
    return 1
  fi

  local st
  for st in scan render; do
    [[ "$stage" == "both" || "$stage" == "$st" ]] || continue
    log "Stufe '$st' — Datensatz $ds, Ziel $out, $eps Episoden."
    [[ "$st" == "render" ]] && log "  Das dauert. Bei Verbindungsabbruch: Lauf in tmux/screen legen."
    docker exec -w "$SIM_DIR" \
      -e "DR_ENABLED=${DR_ENABLED:-1}" \
      -e "RL_GROUND_COLOR=${RL_GROUND_COLOR:-}" \
      -e "RL_AA_MODE=${RL_AA_MODE:-}" \
      "$CONTAINER" bash -lc "
      unset VIRTUAL_ENV
      '$ISAAC_PY' '$SIM_DIR/render_cotrain_dataset.py' \
          --headless --enable_cameras \
          --stage '$st' \
          --dataset-path '$ds' \
          --out-path '$out' \
          --num-episodes '$eps' \
          --asset-path '$ASSET_PATH' \
          $extra" 2>&1 | tee /dev/stderr | grep -c "\[$st\] fertig" >/dev/null \
      || { err "Stufe '$st' ohne Erfolgsmarker beendet (Traceback oben)."; return 1; }
    ok "Stufe '$st' fertig."
  done

  ok "Gerenderter Datensatz: $HOST_DATA_DIR/${out#/data/}"
  echo "  Weiter (auf der Trainings-Seite):"
  echo "    huggingface-cli upload <user>/<repo> $HOST_DATA_DIR/${out#/data/} --repo-type dataset"
  echo "    USE_COTRAIN=1 COTRAIN_HF_REPO=<user>/<repo> …  (docs/training/co-training.md)"
}

# Die Szene ansehen — ohne Modell, ohne Checkpoint, ohne HF_TOKEN.
#
# Der einzige Lauf hier, der KEINE Gewichte braucht: es wird nur die Isaac-Lab-Env
# aufgebaut (Roboter, Tisch, Wuerfel, Licht, Kameras) und der Roboter haelt seine
# Home-Pose. Damit ist die LIVE-Variante ausprobierbar, bevor ein echter Lauf davon
# abhaengt — jeder andere Weg dorthin (check/eval/grasp) verlangt erst den ~10-GB-
# Checkpoint. Zugleich der billigste Nachweis, dass Isaac Sim und RT-Core-Rendering
# auf dieser GPU ueberhaupt laufen.
#
# Gemessen wird nichts. Genau deshalb sind hier Render-Sparhebel erlaubt, die in einem
# Messlauf die Zahlen unvergleichbar machen wuerden (s. CAM_RES_SCALE unten).
do_view() {
  # Den Live-Weg VOR ensure_container waehlen, damit warn_if_container_stale die
  # Port-Veroeffentlichung schon gegen den tatsaechlichen Modus prueft.
  if [[ "${LIVESTREAM:-0}" == "0" && "${LIVE_VIEW:-0}" == "0" ]]; then
    LIVESTREAM=2
    livestream_init
    log "Weder LIVESTREAM noch LIVE_VIEW gesetzt → LIVESTREAM=2 (WebRTC-Viewport)."
    log "  Nur Bilder im Browser stattdessen:  LIVESTREAM=0 LIVE_VIEW=1 $0 view"
  fi

  ensure_asset_local || return 1

  # $SIM_DIR ist read-only aus dem Repo gemountet (s. ensure_container) — ein docker cp
  # ist weder noetig noch moeglich. Gleiche Pruefung wie in do_render.
  if ! docker exec "$CONTAINER" test -f "$SIM_DIR/view_sim.py"; then
    err "view_sim.py fehlt unter $SIM_DIR im Container."
    err "  Normalfall — der Mount ist da, das Repo ist alt:  auf dem Server 'git pull'."
    err "  Sonderfall — der Container stammt von VOR dem 2026-08-13 und hat den Mount gar"
    err "  nicht (damals kam '-v …/g1_dex3_sim:$SIM_DIR:ro' dazu, und -v wirkt nur beim"
    err "  ANLEGEN). Dann hilft nur:  $0 clean   (Daten unter $HOST_DATA_DIR bleiben)."
    err "  Ein Image-Rebuild ist in beiden Faellen NICHT noetig."
    return 1
  fi

  # SCENE_CAM=0: cam_scene geht sonst nur ins MP4, das hier gar nicht geschrieben wird —
  # eine von fuenf Kameras je Step gratis gespart. Ausnahme: wer sie ausdruecklich im
  # Browser sehen will, bekommt sie.
  local scene_cam="${SCENE_CAM:-0}"
  [[ "${LIVE_VIEW_CAMS:-}" == *cam_scene* ]] && scene_cam="${SCENE_CAM:-1}"
  # CAM_RES_SCALE=0.5: in jedem anderen Lauf verboten (die 640x480 sind gegen die
  # Dataset-Referenzframes kalibriert), hier folgenlos — es gibt keine Modell-Eingabe.
  # 94 % der Wanduhr stecken im Rendern (gemessen 2026-08-13), halbe Kantenlaenge ist ein
  # Viertel der Pixel: der Viewport wird spuerbar fluessiger.
  local cam_scale="${CAM_RES_SCALE:-0.5}"
  # DR aus: fuers blosse Ansehen ist eine stabile Beleuchtung nuetzlicher als eine, die
  # je Episode neu aus [1000, 3800] gewuerfelt wird.
  local dr="${DR_ENABLED:-0}"

  livestream_app_flags
  local ls_flags="" f
  for f in "${LS_APP_FLAGS[@]}"; do ls_flags+=" $(printf '%q' "$f")"; done
  live_view_env
  livestream_docker_env

  # Die Laufzeit ist immer endlich. Ein "0 = bis Strg-C" gab es hier frueher und war ein
  # Fehler: der Lauf haengt an einem `docker exec` OHNE TTY, Strg-C beendet nur diesen
  # Client auf dem Host. Im Container liefe die Sim weiter (GPU belegt), der Erfolgsmarker
  # '[view] fertig' unten kaeme nie, und der Lauf gaelte faelschlich als gescheitert.
  # Alte Aufrufe mit 0 nicht abwuergen, sondern auf den Default zurueckdrehen.
  local view_dur="${VIEW_DURATION_S:-3600}"
  if ! awk -v d="$view_dur" 'BEGIN{exit !(d+0 > 0)}'; then
    warn "VIEW_DURATION_S='$view_dur' ist kein Dauerlauf-Schalter (mehr) — Strg-C erreicht den"
    warn "  Prozess im Container nicht. Nutze 3600 s. Laenger zusehen: VIEW_DURATION_S=14400."
    view_dur=3600
  fi

  log "Szene ansehen — kein Modell, kein Checkpoint. Asset: $ASSET_PATH"
  log "  Envs ${VIEW_NUM_ENVS:-1}, Laufzeit ${view_dur}s," \
      "SCENE_CAM=$scene_cam, CAM_RES_SCALE=$cam_scale, DR=$dr"
  if livestream_active; then
    livestream_banner "$(host_addr)"
  fi
  if [[ "${LIVE_VIEW:-0}" != "0" ]]; then
    log "  Live-Ansicht → http://$(host_addr):$LIVE_VIEW_PORT/"
  fi

  docker exec -w "$SIM_DIR" \
    "${LIVE_ENV[@]}" "${LS_ENV[@]}" \
    -e "DR_ENABLED=$dr" \
    -e "SCENE_CAM=$scene_cam" \
    -e "CAM_RES_SCALE=$cam_scale" \
    -e "RL_GROUND_COLOR=${RL_GROUND_COLOR:-}" \
    -e "RL_AA_MODE=${RL_AA_MODE:-}" \
    -e "RL_DOME_INTENSITY=${RL_DOME_INTENSITY:-2000}" \
    "$CONTAINER" bash -lc "
    unset VIRTUAL_ENV
    '$ISAAC_PY' '$SIM_DIR/view_sim.py' \
        $ls_flags --enable_cameras \
        --asset-path '$ASSET_PATH' \
        --num-envs ${VIEW_NUM_ENVS:-1} \
        --duration-s ${view_dur}" 2>&1 \
    `# grep -c statt -q — s. do_eval: -q steigt frueh aus, das tee bekommt SIGPIPE und` \
    `# der Lauf gaelte unter 'set -o pipefail' faelschlich als gescheitert.` \
    | tee /dev/stderr | grep -c "\[view\] fertig" >/dev/null \
    || { err "Ansicht ohne Erfolgsmarker beendet (Traceback oben)."; return 1; }
  ok "Ansicht beendet."
}

# Browser-Client für den WebRTC-Viewport ("Spur A im Browser") — Maus/Tastatur inklusive.
#
# Isaac Sim 6.0 hat keinen eingebauten Browser-Client mehr (der alte auf 8211 entfiel mit
# 5.x, Defekt D1). NVIDIA liefert stattdessen einen separaten Web-Viewer; hier läuft er als
# EIGENER, kleiner Container neben groot-rl. Er enthält keinen Simulator — nur eine
# Webseite. Der Stream kommt weiter aus groot-rl (LIVESTREAM=2), und der Browser verbindet
# sich DIREKT auf dessen 49100/tcp + 47998/udp.
#
# Daraus folgen zwei Dinge, die man leicht falsch erwartet:
#   1. `webview` allein zeigt nichts. Es braucht parallel einen Lauf mit LIVESTREAM=2
#      (z. B. `view`) — sonst ist die Seite da, aber ohne Bild.
#   2. Die Firewall-Anforderung bleibt IDENTISCH. Der Browser spart die App-Installation,
#      nicht den UDP-Port. Ist 47998/udp zu, bleibt das Bild hier genauso schwarz.
do_webview() {
  local port="${WEBVIEW_PORT:-8210}"
  local wv_container="${WEBVIEW_CONTAINER:-groot-webview}"

  # Vor allem anderen: 'stop' braucht weder Adresse noch Image.
  if [[ "${1:-}" == "stop" ]]; then
    docker rm -f "$wv_container" >/dev/null 2>&1 && ok "Web-Viewer gestoppt." \
      || warn "Kein laufender Web-Viewer."
    return 0
  fi

  # Leerzeichen abschneiden: ein LIVESTREAM_HOST_ADDR=" " käme sonst durch die -z-Prüfung
  # und erzeugte einen ungültigen Image-Tag ("groot-webview: -49100-47998").
  local addr; addr="$(host_addr | tr -d '[:space:]')"
  # Die Verbindungsdaten stecken im JS-BUNDLE (Build-Zeit, nicht Laufzeit). Sie deshalb in
  # den Tag schreiben: ändert sich die Server-IP oder ein Port, entsteht automatisch ein
  # anderer Tag und es wird neu gebaut. Ohne das zeigte ein altes Image stumm ins Leere.
  local tag="${addr}-${LIVESTREAM_PORT}-${LIVESTREAM_MEDIA_PORT}"
  local image="${WEBVIEW_IMAGE:-groot-webview}:$tag"

  if [[ -z "$addr" || "$addr" == "<server-ip>" ]]; then
    err "Server-Adresse nicht ermittelbar — sie wird ins JS-Bundle eingebacken, raten geht nicht."
    err "  Explizit setzen:  LIVESTREAM_HOST_ADDR=<lan-ip> $0 webview"
    return 1
  fi

  if ! docker image inspect "$image" >/dev/null 2>&1; then
    log "Baue Web-Viewer-Image $image (einmalig, ~30 s bei freiem Netz)."
    log "  Eingebacken wird: $addr:$LIVESTREAM_PORT/tcp + $LIVESTREAM_MEDIA_PORT/udp"
    warn "Der Build braucht Netzzugang zu ZWEI Registries:"
    warn "  - registry.npmjs.org                (React, Vite & Co.)"
    warn "  - edge.urm.nvidia.com/artifactory   (der @nvidia-Scope; anonym, ohne Token)"
    warn "  Ein 404 auf '@nvidia/…' heißt: die zweite ist nicht erreichbar oder gefiltert —"
    warn "  nicht, dass Paketname oder Version falsch wären."
    docker build -f "$REPO_DIR/Simulation/Dockerfile.webviewer" \
      --build-arg "ISAACSIM_HOST=$addr" \
      --build-arg "ISAACSIM_SIGNAL_PORT=$LIVESTREAM_PORT" \
      --build-arg "ISAACSIM_STREAM_PORT=$LIVESTREAM_MEDIA_PORT" \
      --build-arg "WEB_VIEWER_PORT=$port" \
      -t "$image" "$REPO_DIR/Simulation" \
      || { err "Build des Web-Viewers fehlgeschlagen (Ausgabe oben)."; return 1; }
    ok "Image gebaut: $image"
  else
    ok "Web-Viewer-Image vorhanden: $image"
  fi

  # Einen Container aus einem ALTEN Tag ersetzen — sonst liefe die Seite mit einer
  # veralteten, einbetonierten Adresse weiter.
  local running_img
  running_img="$(docker inspect -f '{{.Config.Image}}' "$wv_container" 2>/dev/null || echo '')"
  if [[ -n "$running_img" && "$running_img" != "$image" ]]; then
    log "Vorhandener Web-Viewer zeigt auf '$running_img' — ersetze ihn durch $image."
    docker rm -f "$wv_container" >/dev/null 2>&1 || true
    running_img=""
  fi
  if [[ -z "$running_img" ]]; then
    if ! port_free "$port"; then
      err "Host-Port $port ist belegt. Anderen wählen:  WEBVIEW_PORT=8211 $0 webview"
      return 1
    fi
    docker run -d --name "$wv_container" -p "$port:$port" "$image" >/dev/null \
      || { err "Web-Viewer-Container konnte nicht gestartet werden."; return 1; }
  elif [[ "$(docker inspect -f '{{.State.Running}}' "$wv_container" 2>/dev/null)" != "true" ]]; then
    docker start "$wv_container" >/dev/null
  fi

  ok "Web-Viewer läuft."
  echo ""
  log "Im Browser öffnen (Chromium/Chrome/Edge — Firefox ist nicht unterstützt):"
  echo "      http://$addr:$port/"
  echo ""
  log "Damit ein Bild kommt, muss PARALLEL ein Lauf mit LIVESTREAM=2 laufen, z. B.:"
  echo "      LIVESTREAM=2 VIEW_DURATION_S=14400 $0 view    # 4 h statt der 1 h Default"
  echo ""
  log "Erreichbar sein müssen vom Browser-Rechner aus:"
  echo "      $port/tcp    (diese Seite)"
  echo "      $LIVESTREAM_PORT/tcp   (Signaling)"
  echo "      $LIVESTREAM_MEDIA_PORT/udp   (Video — der Browser spart die App, NICHT diesen Port)"
  warn "Ohne Authentifizierung und ohne Verschlüsselung — nur im privaten Netz/VPN betreiben."
  echo ""
  log "Stoppen:  $0 webview stop"
}

do_shell()  { ensure_container; docker exec -it "$CONTAINER" bash -l; }
do_clean()  { log "Entferne Container '$CONTAINER' (Daten in $HOST_DATA_DIR bleiben)."; \
              docker rm -f "$CONTAINER" 2>/dev/null || warn "Container existierte nicht."; \
              # Der Web-Viewer ist ein eigener Container — sonst überlebt er 'clean' und
              # zeigt danach auf einen Sim-Container, den es nicht mehr gibt.
              docker rm -f "${WEBVIEW_CONTAINER:-groot-webview}" >/dev/null 2>&1 \
                && log "Web-Viewer-Container ebenfalls entfernt (Image bleibt)."; \
              ok "Weg."; }

usage() {
  cat <<EOF
server_rl_run.sh — RL-Fine-tuning (FPO), Docker-Server statt vast.ai

⚠️  Vor dem ersten Lauf: Image ggf. neu bauen+pushen (siehe Kopf dieser Datei):
      ./Simulation/update_sim_image.sh --vastai
    'preflight' unten weist das verbindlich nach.

Aktionen:
  preflight   Image-Frische (gr00t+flash-attn im Isaac-Sim-Python) + GPU/RT-Cores testen. Kein HF_TOKEN nötig.
  setup       BC-Checkpoint + USD-Asset von HF laden (einmalig, ~10 GB).
  view        Szene ansehen — OHNE Modell, OHNE Checkpoint, OHNE HF_TOKEN. Baut nur die
              Isaac-Lab-Env auf (Roboter, Tisch, Wuerfel, Kameras); der Roboter haelt seine
              Home-Pose. Der einzige Lauf, der ganz ohne Gewichte auskommt — damit ist die
              LIVE-Variante testbar, bevor ein echter Lauf davon abhaengt.
              Ohne LIVESTREAM/LIVE_VIEW wird automatisch LIVESTREAM=2 gesetzt.
              Das USD holt er sich selbst: erst im Container suchen, sonst data/g1_dex3.usd
              + data/configuration/ aus dem Repo hineinkopieren (~40 MB, einmalig).
              VIEW_NUM_ENVS (1), VIEW_DURATION_S (3600, immer > 0 — laenger zusehen: 14400),
              VIEW_HOST_ASSET_DIR (<repo>/data), VIEW_ASSET_DIR (/data/assets),
              EPISODE_LENGTH_S (0 = 300 s bis zum Auto-Reset mit neuen Wuerfelpositionen).
              Eigene Defaults (misst nichts, darf also sparen): SCENE_CAM=0,
              CAM_RES_SCALE=0.5, DR_ENABLED=0 — alle drei ueberschreibbar.
  check       LIVE-CHECK: Env/Policy/Critic aufbauen, 2 Envs, KEIN Training (--check).
  cams        Kamera-Diagnose: konfigurierte vs. gerenderte Pose + ein PNG je Kamera.
  gap         Domain-Gap real vs. sim je Policy-Kamera (SigLIP-ViT). Setzt 'cams' voraus.
              Sweep-Varianten (RL_DOME_SWEEP) werden automatisch mitgemessen.
  eval        BC-Erfolgsrate in der Sim (Closed Loop, GR00T-Server + Isaac-Lab-Client).
              Schritt 3 der Diagnosekette und der Nullpunkt fuer jeden RL-Vergleich.
              NUM_EPISODES (20), EXECUTION_HORIZON (8), EPISODE_LENGTH_S (0 = 300 s).
              GROOT_INFERENCE_BACKEND=eager|compile|tensorrt (Default eager).
              TensorRT braucht vorher 'optimize all'; GROOT_TRT_ENGINE_PATH ist optional.
              CAMERA_RENDER_EVERY_N=8 spart Zwischenframes und muss dann exakt dem
              EXECUTION_HORIZON entsprechen. Details: docs/simulation/inferenz-optimierung.md
  grasp       Greif-Physik isoliert: Open-Loop-Replay der Dataset-Aktionen, kein Modell.
              Beantwortet, ob ein Wuerfel ueberhaupt angehoben werden KANN.
              GRASP_MODE=test (Default): Wuerfel an den geschaetzten Greifpunkten.
              GRASP_MODE=hold: Wuerfel im Moment des Zugreifens zwischen die Fingerspitzen
              gesetzt — misst die Greif-Physik ohne jede Platzierungs-Annahme.
  span        Fingerspanne der Policy auf ECHTEN Datensatz-Bildern (offene Schleife, kein
              Isaac Sim). Trennt Domain-Gap von "Modell hat den Griff nie gelernt" — das
              Gate vor einem TUNE_VISUAL-Lauf. Holt den echten Datensatz bei Bedarf selbst
              (Download + v3->v2-Konvertierung + modality.json, ~18 GB, einmalig).
              SPAN_DATASET (Default /data/unitreerobotics/G1_Dex3_BlockStacking_Dataset),
              SPAN_TRAJ_IDS (Default "0 1 2 3 4"), SPAN_AUTO_FETCH=0 schaltet das Holen ab.
  layout      Wuerfellage aus den REALBILDERN lesen -> layout.json. Farbblob (rot/gruen/gelb)
              im ersten Frame, Strahl durch den Schwerpunkt auf die Wuerfelebene. Braucht
              keine GPU, Minuten statt Stunden. Das ist die richtige Quelle fuer 'render':
              der Greifpunkt aus scan.json ist das Minimum der Fingeroeffnung ueber die ganze
              Episode und liegt bei knapp der Haelfte der Griffe auf dem Transportweg statt
              am Pick — dort greift der Arm dann ins Leere.
              LAYOUT_OUT (/data/cotrain/layout.json), RENDER_EPISODES (60),
              LAYOUT_DEBUG_DIR, LAYOUT_BIAS ("dx dy" in Metern), LAYOUT_OVERWRITE=1.
  layoutcheck Kameramodell gegen ein GERENDERTES Bild pruefen, bevor 'layout' geglaubt wird.
              In Lauf 13 lagen konfigurierte Pose und cam.data 95,6° auseinander und drei
              Laeufe waren umsonst. LAYOUTCHECK_FRAME (Bild im Container),
              LAYOUTCHECK_EXPECT (bekannte Wuerfelpositionen als JSON, aus
              render_manifest.json -> cubes_xyz), LAYOUTCHECK_CAM (cam_left_high).
  replay-prepare  Vollständigen Real-Datensatz holen/konvertieren und Schema, vier Kameras,
              30 Hz sowie 28-DoF-State/Actions prüfen. Stellt das lokale G1+DEX3-Asset
              bereit und lädt keine Modellgewichte.
  replay-calibrate  Kopfkamera-Skala an realen Würfelsilhouetten prüfen und gemeinsame
              5-cm-Würfel-/Tischgeometrie unter REPLAY_WORK ablegen.
  replay-poses  Für maximal REPLAY_NUM_EPISODES (Default 10) die einmalige Anfangspose
              aller Würfel aus beiden Kopfkameras fitten. Fehlende Farben überspringen
              die Episode; kein Zufalls- oder Greifpunkt-Fallback.
  replay-render  Originale 28-DoF-Actions bei exakt 30 Hz abspielen. Jeder Würfel wird
              einmal vor Frame 0 gesetzt und danach ausschließlich von PhysX bewegt.
              Ausgabe: fünf MP4s je Episode, kein Trainingsdatensatz.
              REPLAY_NUM_EPISODES (10), REPLAY_START_EPISODE (0), REPLAY_EPISODE_IDS,
              REPLAY_MAX_FRAMES (0), REPLAY_OVERWRITE (0),
              REPLAY_OUT (/data/cube_replay/videos), REPLAY_WORK (/data/cube_replay/work).
  render      Gerenderten Co-Training-Datensatz erzeugen (Schritt 4): echte Dataset-Aktionen
              in der Sim abspielen und dabei die vier Policy-Kameras aufzeichnen. Ergebnis
              ist ein LeRobot-v2.1-Datensatz, den run_finetuning_cotrain.sh dazumischt.
              Zwei Stufen (scan -> render), fortsetzbar, ~2-3 min je Episode.
              RENDER_EPISODES (60), RENDER_OUT (/data/cotrain/g1_dex3_rendered),
              RENDER_STAGE (both|scan|render), RENDER_MAX_FRAMES (0 = ganze Episode),
              RENDER_EPISODE_IDS ("0 4 8"), RENDER_OVERWRITE=1, DR_ENABLED (1).
              Zurueckgehaltene Test-Episoden werden nie gerendert.
              RENDER_LAYOUT (/data/cotrain/layout.json) ist PFLICHT — 'layout' zuerst
              fahren. RENDER_LAYOUT=none erzwingt den alten Greifpunkt-Weg (Vergleichslauf).
              RENDER_STOP_AT_GRASP (1) schneidet jede Episode am ersten Zugreifen ab —
              ab dort entscheidet die Kontaktphysik ueber die Wuerfellage und das Bild
              zeigt etwas anderes, als die Aktion beschreibt. RENDER_GRASP_WINDOW (0 =
              ab Frame 0) rendert nur die letzten N Frames davor, RENDER_MIN_WINDOW (60)
              ueberspringt zu kurze Fenster. Das Manifest bekommt je Episode ein
              consistency-Feld (Abstand Kuppen<->Wuerfel beim Griff) und der Lauf am Ende
              eine Zusammenfassung daraus — das ist die QA-Zahl, nicht scan.json:ok.
  rl          Echter RL-Lauf (Vordergrund). Checkpoints unter $HOST_DATA_DIR/g1_dex3_rl/.
  latency     Reine Policy-Latenz (ms je Action-Chunk), in-process ohne Sim und ohne ZMQ.
              Die einzige hier messbare Zahl, die auch auf echter Hardware gilt — dort
              faellt das Rendering weg, das in der Eval 94 % der Zeit frisst.
              LATENCY_ITERS (50), EXECUTION_HORIZON (8, nur als Budget-Bezug).
  optimize    N1.6-DiT nach ONNX exportieren, GPU-spezifische TensorRT-Engine bauen und
              validieren. Phasen: export|build|validate|benchmark|all (Default all).
              'all' fuehrt zusaetzlich einen kurzen Closed-Loop-A/B-Lauf aus und schreibt
              benchmark_report.json unter /data/optimized/<checkpoint-fingerprint>/.
              OPTIMIZE_ITERATIONS (20), OPTIMIZE_WARMUP (5),
              OPTIMIZE_SIM_EPISODES (1), OPTIMIZE_SIM_EPISODE_LENGTH_S (20),
              OPTIMIZE_WORKSPACE_MB (8192), OPTIMIZE_TARGET_SPEEDUP (2.0).
  webview     Browser-Client fuer den WebRTC-Viewport starten — mit Maus/Tastatur, also
              echtes Steuern der Kamera, ohne die native App zu installieren. Laeuft als
              EIGENER kleiner Container (kein Simulator drin) und serviert nur eine Seite;
              der Browser verbindet sich direkt auf 49100/tcp + 47998/udp von groot-rl.
              Zeigt allein nichts — es braucht PARALLEL einen Lauf mit LIVESTREAM=2.
              Nur Chromium/Chrome/Edge. Spart die App-Installation, NICHT den UDP-Port.
              WEBVIEW_PORT (8210), WEBVIEW_IMAGE/-CONTAINER (groot-webview).
              '$0 webview stop' beendet ihn; 'clean' entfernt ihn mit.
  livecheck   Phase 0 der LIVE-Variante: NVENC, Livestream-Extension, Isaac-Sim-Version
              und Port-Veroeffentlichung pruefen — vor dem ersten LIVESTREAM=2-Lauf.
  shell       Interaktive Shell im Container.
  clean       Container entfernen (Daten unter $HOST_DATA_DIR bleiben).
  help        Diese Hilfe.

Beispiele:
  ./Simulation/server_rl_run.sh preflight
  ./Simulation/server_rl_run.sh optimize all
  GROOT_INFERENCE_BACKEND=eager ./Simulation/server_rl_run.sh eval
  GROOT_INFERENCE_BACKEND=compile ./Simulation/server_rl_run.sh eval
  GROOT_INFERENCE_BACKEND=tensorrt CAMERA_RENDER_EVERY_N=8 SCENE_CAM=0 \\
      ./Simulation/server_rl_run.sh eval
  HF_TOKEN=hf_... ./Simulation/server_rl_run.sh check
  # Schritt 3 — BC-Erfolgsrate, 3x das Zeitbudget der menschlichen Demo (39 s):
  HF_TOKEN=hf_... NUM_EPISODES=20 EPISODE_LENGTH_S=120 DR_ENABLED=0 \\
      ./Simulation/server_rl_run.sh eval
  # Greif-Physik ohne Platzierungs-Annahme (Wuerfel wird in die Greifoeffnung gesetzt):
  HF_TOKEN=hf_... GRASP_MODE=hold ./Simulation/server_rl_run.sh grasp
  # Gate vor TUNE_VISUAL — greift die Policy auf ECHTEN Bildern?
  HF_TOKEN=hf_... ./Simulation/server_rl_run.sh span
  # Schritt 4 — erst der Rauchtest (2 Episoden a 60 Frames), dann der lange Lauf:
  HF_TOKEN=hf_... RENDER_EPISODES=2 RENDER_MAX_FRAMES=60 ./Simulation/server_rl_run.sh render
  HF_TOKEN=hf_... RENDER_EPISODES=60 ./Simulation/server_rl_run.sh render
  # Physikbasierte Videos aus zunächst höchstens zehn Real-Episoden:
  HF_TOKEN=hf_... ./Simulation/server_rl_run.sh replay-prepare
  REPLAY_NUM_EPISODES=10 ./Simulation/server_rl_run.sh replay-calibrate
  REPLAY_NUM_EPISODES=10 ./Simulation/server_rl_run.sh replay-poses
  DR_ENABLED=0 REPLAY_NUM_EPISODES=10 ./Simulation/server_rl_run.sh replay-render
  HF_TOKEN=hf_... WANDB_API_KEY=... RL_NUM_ENVS=4 ./Simulation/server_rl_run.sh rl
  # mit Live-Ansicht im Browser + W&B-Video alle 10 Iterationen:
  HF_TOKEN=hf_... WANDB_API_KEY=... LIVE_VIEW=1 RL_WANDB_VIDEO_EVERY=10 \\
      RL_NUM_ENVS=4 ./Simulation/server_rl_run.sh rl

Tempo der Sim (gemessen 2026-08-13: 94 % Sim+Render, 5 % Inferenz, 3,5x langsamer
als Echtzeit). Die Zahlen stehen live in der "Step …"-Zeile und in results.json:
  SCENE_CAM=0        Uebersichtskamera cam_scene weglassen — eine von fuenf Kameras je
                     Step. GRATIS: die Policy sieht cam_scene nie, Erfolgsraten bleiben
                     vergleichbar. Kostet nur MP4/Uebersichtsbild.
  CAM_RES_SCALE=0.5  Kameraaufloesung halbieren (Pixel gehen quadratisch ein). Aendert
                     die MODELL-EINGABE -> nur zum Zuschauen, nicht fuer Messlaeufe.
  RL_AA_MODE=Off     Anti-Aliasing aus (sonst Isaac-Defaults).
  EXECUTION_HORIZON  wirkt nur auf die 5 % Inferenz — hier fast wirkungslos.

LIVE-Variante — Isaac Sim auf dem eigenen Rechner (opt-in, "Spur A"):
  Statt hinterher MP4s zu holen, streamt Isaac Sim seinen 3D-Viewport per WebRTC. Auf dem
  Arbeitsrechner oeffnet ihn die App "Isaac Sim WebRTC Streaming Client" — freie Kamera,
  Szene drehen, Isaac-Sim-UI. Anleitung: docs/simulation/live-ansicht.md

  LIVESTREAM=2           an, privates Netz/VPN (auf diesem Server der richtige Wert)
  LIVESTREAM=1           an, oeffentliches Netz (vast.ai; setzt PUBLIC_IP, ungeschuetzt!)
  LIVESTREAM_PORT        Signaling, TCP (Default 49100; intern==extern, wird gemappt)
  LIVESTREAM_MEDIA_PORT  Medien, UDP  (Default 47998; MUSS durch die Firewall)
  LIVE_KEEP_VIDEO=1      zusaetzlich MP4s schreiben (Default: live STATT Video)
  -> gilt fuer:  view, eval, grasp, rl, check   (baseline ueber entrypoint_baseline.sh)
  -> Client:     <server-ip>:49100 in der App eintragen, Connect
  -> vorher:     $0 livecheck

  Beispiele:
    # ohne jede Gewichte — der schnellste Weg zum ersten Bild:
    $0 view
    HF_TOKEN=hf_... LIVESTREAM=2 NUM_EPISODES=2 EPISODE_LENGTH_S=120 $0 eval
    HF_TOKEN=hf_... LIVESTREAM=2 GRASP_MODE=hold $0 grasp

Live-Ansicht im Browser (opt-in, docs/weiterfuehrend/livestream-plan.md Spur B):
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
  preflight|setup|check|cams|gap|eval|grasp|span|rl|livecheck|latency|optimize|render|view|webview|layout|layoutcheck|replay-prepare|replay-calibrate|replay-poses|replay-render) start_logging "$ACTION" ;;
esac
case "$ACTION" in
  preflight)  do_preflight ;;
  setup)      do_setup ;;
  check)      do_check ;;
  view)       do_view ;;
  webview)    do_webview "${2:-}" ;;
  cams)       do_cams ;;
  gap)        do_gap ;;
  eval)       do_eval ;;
  grasp)      do_grasp ;;
  span)       do_span ;;
  layout)     do_layout ;;
  layoutcheck) do_layoutcheck ;;
  replay-prepare) do_replay_prepare ;;
  replay-calibrate) do_replay_calibrate ;;
  replay-poses) do_replay_poses ;;
  replay-render) do_replay_render ;;
  render)     do_render ;;
  latency)    do_latency ;;
  optimize)   do_optimize "${2:-all}" ;;
  livecheck)  do_livecheck ;;
  rl)         do_rl ;;
  shell)      do_shell ;;
  clean|down) do_clean ;;
  help|-h|--help) usage ;;
  *) err "Unbekannte Aktion: '$ACTION'"; echo; usage; exit 2 ;;
esac
