#!/usr/bin/env bash
# server_robocasa_ref_run.sh — RoboCasa GR-1 Referenz-Eval auf einem generischen
#   Docker-GPU-Server (z. B. 2× RTX PRO 6000 Blackwell).
#
# Docker-Pendant zu kisski_robocasa_ref_submit.sh (dort Apptainer + SLURM).
# Fährt das BASISMODELL `nvidia/GR00T-N1.6-3B` (zero-shot, Embodiment GR1) auf dem
# RoboCasa GR-1 Tabletop Benchmark (robosuite/MuJoCo, EGL — KEINE RT-Cores nötig)
# und prüft, ob unsere GR00T-Inferenz-Pipeline NVIDIAs publizierte Erfolgsquoten
# reproduziert (Top-Task 78,7 %, Mittel 47,6 %).
#   Hintergrund/Plan:  docs/simulation/basismodell-referenzaufgabe.md
#   Bedienung/Caveats: docs/simulation/robocasa-referenz-eval.md
#
# ISOLATION: nutzt das vorhandene TRAININGS-Image + das additive Scaffold in
# Simulation/robocasa_reference (eigene robocasa_uv-Venv, ZMQ-Port 5757, RC_*-Env,
# Ergebnisse unter /data/robocasa_ref). Verändert kein bestehendes Skript/Image.
#
# CONTAINER-MODELL: ein langlebiger "Workbench"-Container (sleep infinity), in dem
# Setup und Eval per `docker exec` laufen. So bleibt der Container zwischen
# Läufen erhalten (Modell-Cache/Venv persistent) — erst `clean` entfernt ihn.
#
# BLACKWELL: Image hat torch 2.7.1/cu128 (sm_120) → kein Rebuild nötig. Einzige
# Unbekannte ist flash-attn (vorgebautes Wheel). `preflight` testet beides;
# `fix-flash-attn` installiert bei Bedarf ein sm_120-fähiges Wheel nach (kein
# Rebuild). Details: siehe Kopf von docs/simulation/robocasa-referenz-eval.md.
#
# NUTZUNG:
#   ./Simulation/server_robocasa_ref_run.sh preflight        # Torch+flash-attn auf GPU testen
#   HF_TOKEN=hf_... ./Simulation/server_robocasa_ref_run.sh setup     # einmalig (Internet, ~10–20 min)
#   HF_TOKEN=hf_... ./Simulation/server_robocasa_ref_run.sh smoke     # schneller Kettentest (1 Task/5 Ep.)
#   HF_TOKEN=hf_... ./Simulation/server_robocasa_ref_run.sh eval      # Validierung (Default: top, 100 Ep.)
#   ./Simulation/server_robocasa_ref_run.sh fix-flash-attn   # nur falls Preflight flash-attn scheitert
#   ./Simulation/server_robocasa_ref_run.sh shell|clean|help
#
# Überschreibbar via Env (Defaults für diesen Server):
#   RC_HOST_DATA_DIR (/home/lmuecke/project/data/RoboCasa), RC_IMAGE, RC_CONTAINER,
#   RC_GPUS (all), RC_PRESET (top|smoke|full|custom), RC_N_EPISODES, RC_N_ENVS,
#   RC_TASKS, RC_MODEL_PATH, RC_PORT, HF_TOKEN.

set -euo pipefail

# ── Logging-Helfer (Hausstil, vgl. run_robocasa_ref_eval.sh) ─────────────────
log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m v \033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m ! \033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m!! \033[0m %s\n' "$*" >&2; }

# ── Konfiguration (alle via Env überschreibbar) ──────────────────────────────
IMAGE="${RC_IMAGE:-lucam03/projekt-humanoider-roboter:latest}"
CONTAINER="${RC_CONTAINER:-groot-robocasa-ref}"
HOST_DATA_DIR="${RC_HOST_DATA_DIR:-/home/lmuecke/project/data/RoboCasa}"
GPUS="${RC_GPUS:-all}"                 # z. B. RC_GPUS='"device=0"' zum Pinnen
SHM_SIZE="${RC_SHM_SIZE:-16g}"

# Repo-Wurzel = Elternverzeichnis dieses Skripts (Simulation/..)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${RC_REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SCAFFOLD_DIR="$REPO_DIR/Simulation/robocasa_reference"

# Eval-Parameter (an run_robocasa_ref_eval.sh durchgereicht)
RC_PRESET="${RC_PRESET:-top}"
RC_N_EPISODES="${RC_N_EPISODES:-100}"
RC_N_ENVS="${RC_N_ENVS:-8}"
RC_N_ACTION_STEPS="${RC_N_ACTION_STEPS:-8}"
RC_MAX_EPISODE_STEPS="${RC_MAX_EPISODE_STEPS:-720}"
RC_TASKS="${RC_TASKS:-}"
RC_MODEL_PATH="${RC_MODEL_PATH:-nvidia/GR00T-N1.6-3B}"   # HF-ID; lokal: /data/models/GR00T-N1.6-3B
RC_PORT="${RC_PORT:-5757}"
HF_HOME_IN="${RC_HF_HOME:-/data/hf_cache}"               # Modell-Cache im externen /data

FLASH_ATTN_SPEC="${RC_FLASH_ATTN_SPEC:-flash-attn>=2.8.0}"

VENV_PY="/app/Groot-1.6/.venv/bin/python"
RUNNER_IN="/workspace/robocasa_reference/run_robocasa_ref_eval.sh"

# ── Kleine Helfer ─────────────────────────────────────────────────────────────
require_docker() { command -v docker >/dev/null 2>&1 || { err "docker nicht gefunden."; exit 1; }; }

require_hf_token() {
  [[ -n "${HF_TOKEN:-}" ]] || { err "HF_TOKEN nicht gesetzt (für Setup/Modell-Download nötig)."; \
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
  # Neu anlegen (langlebiger idle-Container)
  [[ -d "$SCAFFOLD_DIR" ]] || { err "Scaffold fehlt: $SCAFFOLD_DIR"; \
      err "  Ist das Repo auf dem Server aktuell (Branch mit robocasa_reference gepullt)?"; exit 1; }
  mkdir -p "$HOST_DATA_DIR"
  log "Erzeuge langlebigen Container '$CONTAINER' (Image: $IMAGE, GPU: $GPUS)."
  log "  /data          -> $HOST_DATA_DIR"
  log "  Scaffold-Mount -> $SCAFFOLD_DIR"
  local create_env=( -e "HF_HOME=$HF_HOME_IN" -e "MUJOCO_GL=egl" \
                     -e "PYOPENGL_PLATFORM=egl" -e "PYTHONUNBUFFERED=1" )
  [[ -n "${HF_TOKEN:-}" ]] && create_env+=( -e "HF_TOKEN=$HF_TOKEN" )
  docker run -d --name "$CONTAINER" --gpus "$GPUS" --ipc=host --shm-size="$SHM_SIZE" \
    "${create_env[@]}" \
    -v "$HOST_DATA_DIR:/data" \
    -v "$SCAFFOLD_DIR:/workspace/robocasa_reference" \
    --entrypoint bash "$IMAGE" -lc "sleep infinity" >/dev/null
  ok "Container läuft."
}

# Baut das -e-Array für Eval-Execs (RC_* + optional HF_TOKEN/RC_TASKS)
build_eval_env() {
  EVAL_ENV=(
    -e "RC_PRESET=$RC_PRESET"
    -e "RC_N_EPISODES=$RC_N_EPISODES"
    -e "RC_N_ENVS=$RC_N_ENVS"
    -e "RC_N_ACTION_STEPS=$RC_N_ACTION_STEPS"
    -e "RC_MAX_EPISODE_STEPS=$RC_MAX_EPISODE_STEPS"
    -e "RC_MODEL_PATH=$RC_MODEL_PATH"
    -e "RC_PORT=$RC_PORT"
  )
  [[ -n "$RC_TASKS" ]]        && EVAL_ENV+=( -e "RC_TASKS=$RC_TASKS" )
  [[ -n "${HF_TOKEN:-}" ]]    && EVAL_ENV+=( -e "HF_TOKEN=$HF_TOKEN" )
  # PFLICHT: Trifft die letzte `[[ … ]] && …`-Zeile nicht zu, liefert die Funktion 1 und
  # `set -e` beendet das Skript STILL. Bisher unentdeckt, weil HF_TOKEN bei HF-Modell-IDs
  # immer gesetzt ist — mit lokalem RC_MODEL_PATH (/data/models/…) wäre es aufgeschlagen.
  return 0
}

# ── Aktionen ──────────────────────────────────────────────────────────────────
do_preflight() {
  log "Preflight: Torch- und flash-attn-Kernels auf GPU '$GPUS' prüfen."
  docker pull "$IMAGE" || warn "docker pull fehlgeschlagen — nutze lokal vorhandenes Image."
  if docker run --rm -i --gpus "$GPUS" "$IMAGE" "$VENV_PY" - <<'PY'
import sys, torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| dev", torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda")
print("matmul ok  :", float((x @ x).sum()))
try:
    from flash_attn import flash_attn_func
    import flash_attn
    q = k = v = torch.randn(1, 8, 4, 64, device="cuda", dtype=torch.float16)
    print("flash-attn :", flash_attn.__version__, "->", tuple(flash_attn_func(q, k, v).shape))
except Exception as e:
    print("FLASH-ATTN FAIL:", repr(e))
    sys.exit(3)
PY
  then
    ok "Preflight bestanden — Image ist Blackwell-tauglich, KEIN Rebuild nötig."
  else
    warn "Preflight-Problem. Bei 'FLASH-ATTN FAIL'/sm_120-Fehler:  $0 fix-flash-attn"
    return 1
  fi
}

do_setup() {
  require_hf_token
  ensure_container
  log "Einmaliges RoboCasa-Setup (robosuite-Submodul + robocasa_uv-Venv + Assets)."
  log "  Braucht Internet + HF_TOKEN, dauert ~10–20 min."
  local senv=( -e RC_SETUP_ONLY=1 )
  [[ -n "${HF_TOKEN:-}" ]] && senv+=( -e "HF_TOKEN=$HF_TOKEN" )
  docker exec "${senv[@]}" "$CONTAINER" bash -lc "bash $RUNNER_IN"
  ok "Setup abgeschlossen. Weiter mit:  $0 smoke   bzw.   $0 eval"
}

# Übernimmt die vom Client aufgezeichneten Rollout-Videos ins externe
# Ergebnisverzeichnis. Der Runner reicht kein Video-Verzeichnis durch, und
# rollout_policy.py schreibt fest nach /tmp/sim_eval_videos_* (mit UUID) — daher
# aus dem flüchtigen Container-/tmp nach /data/robocasa_ref/videos verschieben.
collect_videos() {
  docker exec "$CONTAINER" bash -lc '
    dest=/data/robocasa_ref/videos; mkdir -p "$dest"
    n=$(find /tmp -maxdepth 1 -type d -name "sim_eval_videos_*" | wc -l)
    if [ "$n" -gt 0 ]; then
      mv /tmp/sim_eval_videos_* "$dest"/ 2>/dev/null || true
      echo "  $n Video-Ordner -> $dest"
    else
      echo "  (keine neuen Videos in /tmp gefunden)"
    fi' || warn "Video-Übernahme fehlgeschlagen (nicht fatal)."
}

do_eval() {
  # HF-ID (kein absoluter lokaler Pfad) -> Token nötig; ggf. auch für Auto-Setup.
  [[ "$RC_MODEL_PATH" == /* ]] || require_hf_token
  ensure_container
  build_eval_env
  log "Eval starten:  preset=$RC_PRESET  n_episodes=$RC_N_EPISODES  n_envs=$RC_N_ENVS"
  log "  Modell:      $RC_MODEL_PATH"
  log "  Ergebnisse:  $HOST_DATA_DIR/robocasa_ref/summary-<ts>.json"
  docker exec "${EVAL_ENV[@]}" "$CONTAINER" bash -lc "bash $RUNNER_IN"
  log "Übernehme Rollout-Videos ins externe Verzeichnis…"
  collect_videos
  ok "Fertig. Summary + Logs + Videos unter $HOST_DATA_DIR/robocasa_ref/"
}

do_smoke() {
  # smoke-Preset erzwingt im Runner ohnehin 1 Task/5 Ep./5 Envs — hier spiegeln,
  # damit die Log-Zeile in do_eval die tatsächlich laufenden Werte zeigt.
  RC_PRESET="smoke"; RC_N_EPISODES=5; RC_N_ENVS=5
  log "Smoke-Test (RC_PRESET=smoke: 1 Task, 5 Episoden) — prüft die ganze Kette schnell."
  do_eval
}

do_videos() {
  ensure_container
  log "Übernehme aufgezeichnete Rollout-Videos ins externe Verzeichnis."
  collect_videos
  ok "Videos unter $HOST_DATA_DIR/robocasa_ref/videos/  (per scp abholbar)"
}

do_fix_flash_attn() {
  ensure_container
  warn "Installiere '$FLASH_ATTN_SPEC' in die GR00T-Venv (nur nötig, wenn Preflight flash-attn scheitert)."
  warn "Existiert kein passendes sm_120-Wheel, wird aus Quellen gebaut (kann dauern; Threadripper hilft)."
  docker exec \
    -e "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}" \
    -e "MAX_JOBS=${MAX_JOBS:-16}" \
    "$CONTAINER" bash -lc "uv pip install --python $VENV_PY --no-build-isolation '$FLASH_ATTN_SPEC'"
  ok "flash-attn aktualisiert (im laufenden Container; geht bei 'clean' verloren)."
  log "Erneut prüfen:  $0 preflight   (bzw. direkt $0 smoke)"
}

do_shell()  { ensure_container; docker exec -it "$CONTAINER" bash -l; }
do_clean()  { log "Entferne Container '$CONTAINER' (Ergebnisse in $HOST_DATA_DIR bleiben)."; \
              docker rm -f "$CONTAINER" 2>/dev/null || warn "Container existierte nicht."; ok "Weg."; }

usage() {
  cat <<EOF
server_robocasa_ref_run.sh — RoboCasa GR-1 Referenz-Eval (Docker, generischer GPU-Server)

Aktionen:
  preflight        Torch + flash-attn auf der GPU testen (Blackwell/sm_120). Kein HF_TOKEN nötig.
  setup            Einmaliges RoboCasa-Setup im langlebigen Container (Internet + HF_TOKEN).
  smoke            Schneller Kettentest (1 Task, 5 Episoden).
  eval             Referenz-Eval (Default RC_PRESET=top, 100 Ep. -> Soll 78,7 %).
  fix-flash-attn   Blackwell-fähiges flash-attn nachinstallieren (nur falls Preflight scheitert).
  videos           Aufgezeichnete Rollout-Videos ins externe Verzeichnis (/data/robocasa_ref/videos) übernehmen.
  shell            Interaktive Shell im Container.
  clean            Container entfernen (Daten unter $HOST_DATA_DIR bleiben).
  help             Diese Hilfe.

Beispiele:
  ./Simulation/server_robocasa_ref_run.sh preflight
  HF_TOKEN=hf_... ./Simulation/server_robocasa_ref_run.sh setup
  HF_TOKEN=hf_... ./Simulation/server_robocasa_ref_run.sh smoke
  HF_TOKEN=hf_... RC_PRESET=full ./Simulation/server_robocasa_ref_run.sh eval    # alle 24 Tasks -> Soll ø47,6 %

Datenverzeichnis (Host): $HOST_DATA_DIR   ->  Container /data
Image:                    $IMAGE
EOF
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
require_docker
ACTION="${1:-help}"
case "$ACTION" in
  preflight)       do_preflight ;;
  setup)           do_setup ;;
  smoke)           do_smoke ;;
  eval)            do_eval ;;
  fix-flash-attn)  do_fix_flash_attn ;;
  videos)          do_videos ;;
  shell)           do_shell ;;
  clean|down)      do_clean ;;
  help|-h|--help)  usage ;;
  *)               err "Unbekannte Aktion: '$ACTION'"; echo; usage; exit 2 ;;
esac
