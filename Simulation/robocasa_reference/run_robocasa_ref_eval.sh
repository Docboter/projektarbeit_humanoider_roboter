#!/usr/bin/env bash
# run_robocasa_ref_eval.sh — Referenz-Eval: GR00T-N1.6 Basismodell (zero-shot)
# auf dem RoboCasa GR-1 Tabletop Benchmark (robosuite/MuJoCo).
#
# ZWECK
#   Validiert unsere GR00T-Inferenz-/Server-Pipeline gegen einen externen
#   Goldstandard: NVIDIA publiziert für `nvidia/GR00T-N1.6-3B` mit Embodiment
#   GR1 Zero-Shot-Erfolgsquoten (Einzeltask bis 78,7 %, ø 47,6 %). Reproduzieren
#   wir diese Quoten innerhalb statistischer Toleranz, ist die Server/Wrapper/
#   ZMQ/Obs-Action-Hälfte unserer Pipeline bestätigt.
#   Hintergrund + Plan: docs/simulation/basismodell-referenzaufgabe.md
#   Bedienung:          docs/simulation/robocasa-referenz-eval.md
#
# ISOLATION (stört KEINE bestehende Implementierung)
#   - Eigene, isolierte Python-Venv `robocasa_uv/.venv` (NVIDIA-Setup), getrennt
#     von der GR00T-Venv (.venv) und vom Isaac-Lab-Sim.
#   - Alle Ergebnisse/Logs unter $RC_RESULTS_DIR (Default /data/robocasa_ref).
#   - Eigene Env-Vars (Prefix RC_), eigener Server-Port-Default.
#   - Verändert keine bestehenden Skripte/Entrypoints/Dockerfiles.
#
# ZWEI-PROZESS-ARCHITEKTUR (beide auf derselben Maschine)
#   Server: GR00T-Venv  -> gr00t/eval/run_gr00t_server.py  (ZMQ, Port RC_PORT)
#   Client: robocasa_uv -> gr00t/eval/rollout_policy.py     (robosuite/MuJoCo)
#
# MODI
#   RC_SETUP_ONLY=1  -> nur NVIDIA-Setup ausführen (auf Node mit Internet!), Ende.
#   RC_SKIP_SETUP=1  -> Setup überspringen (robocasa_uv muss existieren).
#   sonst            -> Setup nur, wenn robocasa_uv fehlt; danach Eval.

set -euo pipefail

# ── Logging-Helfer (Hausstil) ────────────────────────────────────────────────
log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m v \033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m ! \033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m!! \033[0m %s\n' "$*" >&2; }

# ── Konfiguration (alle überschreibbar via Env, Prefix RC_) ──────────────────
GROOT_ROOT="${GROOT_ROOT:-/app/Groot-1.6}"
ROBOCASA_DIR="${ROBOCASA_DIR:-$GROOT_ROOT/gr00t/eval/sim/robocasa-gr1-tabletop-tasks}"
SETUP_SCRIPT="$ROBOCASA_DIR/setup_RoboCasaGR1TabletopTasks.sh"
SERVER_PY="$GROOT_ROOT/gr00t/eval/run_gr00t_server.py"
CLIENT_PY="$GROOT_ROOT/gr00t/eval/rollout_policy.py"

# Python-Interpreter der beiden getrennten Venvs
SERVER_PYTHON="${RC_SERVER_PYTHON:-$GROOT_ROOT/.venv/bin/python}"
CLIENT_PYTHON="${RC_CLIENT_PYTHON:-$ROBOCASA_DIR/robocasa_uv/.venv/bin/python}"

# Modell / Embodiment
RC_MODEL_PATH="${RC_MODEL_PATH:-/data/models/GR00T-N1.6-3B}"   # lokal, sonst HF-ID
RC_EMBODIMENT_TAG="${RC_EMBODIMENT_TAG:-GR1}"
RC_PORT="${RC_PORT:-5757}"                                     # eigener Port (nicht 5555)

# Eval-Parameter
RC_PRESET="${RC_PRESET:-top}"            # top | smoke | full | custom
RC_N_EPISODES="${RC_N_EPISODES:-50}"
RC_N_ENVS="${RC_N_ENVS:-8}"
RC_N_ACTION_STEPS="${RC_N_ACTION_STEPS:-8}"
RC_MAX_EPISODE_STEPS="${RC_MAX_EPISODE_STEPS:-720}"
RC_TASKS="${RC_TASKS:-}"                 # bei RC_PRESET=custom: space/komma-separiert

# Ablauf-Steuerung
RC_SETUP_ONLY="${RC_SETUP_ONLY:-0}"
RC_SKIP_SETUP="${RC_SKIP_SETUP:-0}"
RC_RESULTS_DIR="${RC_RESULTS_DIR:-/data/robocasa_ref}"
RC_SERVER_WAIT_TIMEOUT="${RC_SERVER_WAIT_TIMEOUT:-900}"   # s, Modell-Load ~6 GB
RC_SERVER_SETTLE="${RC_SERVER_SETTLE:-20}"                # s Puffer nach Port-open

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# ── Publizierte Goldstandard-Quoten (README robocasa-gr1-tabletop-tasks) ──────
declare -A REF=(
  ["gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env"]=51.5
  ["gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env"]=13.0
  ["gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"]=8.5
  ["gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"]=14.0
  ["gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"]=41.5
  ["gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env"]=16.5
  ["gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"]=58.0
  ["gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"]=46.5
  ["gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env"]=68.5
  ["gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env"]=65.0
  ["gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env"]=46.5
  ["gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"]=58.5
  ["gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env"]=57.5
  ["gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"]=63.0
  ["gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env"]=28.5
  ["gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env"]=57.0
  ["gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"]=43.5
  ["gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env"]=51.0
  ["gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"]=78.7
  ["gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"]=51.5
  ["gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"]=71.0
  ["gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env"]=64.5
  ["gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env"]=57.0
  ["gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env"]=31.5
)

# Bester Einzeltask (höchste publizierte Quote) — Default für schnelle Validierung
TOP_TASK="gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"

# ── Task-Liste aus Preset bestimmen ──────────────────────────────────────────
declare -a TASKS=()
case "$RC_PRESET" in
  top)   TASKS=("$TOP_TASK") ;;
  smoke) TASKS=("$TOP_TASK") ; RC_N_EPISODES=5 ; RC_N_ENVS=5 ;;
  full)  TASKS=("${!REF[@]}") ;;
  custom)
    if [[ -z "$RC_TASKS" ]]; then err "RC_PRESET=custom, aber RC_TASKS leer."; exit 2; fi
    IFS=', ' read -r -a TASKS <<< "$RC_TASKS" ;;
  *) err "Unbekanntes RC_PRESET='$RC_PRESET' (top|smoke|full|custom)."; exit 2 ;;
esac

mkdir -p "$RC_RESULTS_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
SERVER_LOG="$RC_RESULTS_DIR/server-$TS.log"
SUMMARY_JSON="$RC_RESULTS_DIR/summary-$TS.json"

# ── Existenz-Checks ───────────────────────────────────────────────────────────
[[ -f "$SETUP_SCRIPT" ]] || { err "Setup-Skript fehlt: $SETUP_SCRIPT (Submodul ausgecheckt?)"; exit 1; }
[[ -f "$SERVER_PY"   ]] || { err "Server-Skript fehlt: $SERVER_PY"; exit 1; }
[[ -f "$CLIENT_PY"   ]] || { err "Client-Skript fehlt: $CLIENT_PY"; exit 1; }

# ── NVIDIA-Setup (braucht Internet: git submodule + uv pip + HF-Assets) ───────
run_setup() {
  log "RoboCasa-Setup (NVIDIA) — benötigt Internet + git + HF-Token."
  ( cd "$GROOT_ROOT" && bash "$SETUP_SCRIPT" )
  ok "Setup abgeschlossen — Venv: $ROBOCASA_DIR/robocasa_uv/.venv"
}

if [[ "$RC_SETUP_ONLY" == "1" ]]; then
  run_setup
  ok "RC_SETUP_ONLY=1 — fertig (keine Eval)."
  exit 0
fi

if [[ ! -x "$CLIENT_PYTHON" ]]; then
  if [[ "$RC_SKIP_SETUP" == "1" ]]; then
    err "robocasa_uv-Venv fehlt ($CLIENT_PYTHON) und RC_SKIP_SETUP=1."
    err "Einmalig auf einem Node mit Internet: RC_SETUP_ONLY=1 $0"
    exit 1
  fi
  run_setup
fi
[[ -x "$CLIENT_PYTHON" ]] || { err "Client-Python weiterhin nicht vorhanden: $CLIENT_PYTHON"; exit 1; }

# ── Modell-Pfad: lokal bevorzugt, sonst als HF-ID durchreichen ────────────────
MODEL_ARG="$RC_MODEL_PATH"
if [[ ! -d "$RC_MODEL_PATH" ]]; then
  warn "Lokales Modellverzeichnis fehlt ($RC_MODEL_PATH) — gebe '$RC_MODEL_PATH' an Server weiter (HF-Download, braucht Internet/HF_TOKEN)."
fi

# ── Server starten (Hintergrund) ──────────────────────────────────────────────
log "Starte GR00T-Server (Embodiment=$RC_EMBODIMENT_TAG, Port=$RC_PORT)"
log "  Modell:      $MODEL_ARG"
log "  Server-Log:  $SERVER_LOG"
(
  cd "$GROOT_ROOT" && exec "$SERVER_PYTHON" "$SERVER_PY" \
    --model-path "$MODEL_ARG" \
    --embodiment-tag "$RC_EMBODIMENT_TAG" \
    --use-sim-policy-wrapper \
    --port "$RC_PORT"
) >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    log "Beende GR00T-Server (PID $SERVER_PID)."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ── Auf Server-Bereitschaft warten (Port offen + Settle) ─────────────────────
log "Warte auf Server-Port $RC_PORT (Timeout ${RC_SERVER_WAIT_TIMEOUT}s, Modell-Load ~6 GB)…"
waited=0
until timeout 1 bash -c ">/dev/tcp/127.0.0.1/$RC_PORT" 2>/dev/null; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    err "Server-Prozess vorzeitig beendet. Letzte Log-Zeilen:"; tail -n 40 "$SERVER_LOG" >&2 || true; exit 1
  fi
  if (( waited >= RC_SERVER_WAIT_TIMEOUT )); then
    err "Timeout: Server-Port $RC_PORT nach ${RC_SERVER_WAIT_TIMEOUT}s nicht offen."; tail -n 40 "$SERVER_LOG" >&2 || true; exit 1
  fi
  sleep 5; waited=$((waited+5))
done
ok "Server-Port offen nach ${waited}s — warte ${RC_SERVER_SETTLE}s Puffer."
sleep "$RC_SERVER_SETTLE"

# ── Eval-Schleife über Tasks ──────────────────────────────────────────────────
declare -a JSON_ENTRIES=()
printf '\n'
log "Starte Eval: ${#TASKS[@]} Task(s), n_episodes=$RC_N_EPISODES, n_envs=$RC_N_ENVS, n_action_steps=$RC_N_ACTION_STEPS"
for task in "${TASKS[@]}"; do
  [[ -z "$task" ]] && continue
  short="${task#gr1_unified/}"; short="${short%_GR1ArmsAndWaistFourierHands_Env}"
  client_log="$RC_RESULTS_DIR/client-$short-$TS.log"
  expected="${REF[$task]:-}"

  log "→ Task: $short  (erwartet: ${expected:-?}%)"
  set +e
  "$CLIENT_PYTHON" "$CLIENT_PY" \
    --policy_client_host 127.0.0.1 \
    --policy_client_port "$RC_PORT" \
    --env_name "$task" \
    --n_episodes "$RC_N_EPISODES" \
    --n_envs "$RC_N_ENVS" \
    --n_action_steps "$RC_N_ACTION_STEPS" \
    --max_episode_steps "$RC_MAX_EPISODE_STEPS" \
    2>&1 | tee "$client_log"
  rc=${PIPESTATUS[0]}
  set -e

  # success rate aus stdout parsen: 'success rate:  <float>'
  rate="$(grep -aoE 'success rate:[[:space:]]*[0-9.]+' "$client_log" | tail -n1 | grep -oE '[0-9.]+$' || true)"
  if [[ -n "$rate" ]]; then
    measured_pct="$(awk -v r="$rate" 'BEGIN{printf "%.1f", r*100}')"
    if [[ -n "$expected" ]]; then
      delta="$(awk -v m="$measured_pct" -v e="$expected" 'BEGIN{printf "%+.1f", m-e}')"
    else
      delta="n/a"
    fi
    ok "   gemessen: ${measured_pct}%  (erwartet ${expected:-?}%, Δ ${delta} pp)"
    JSON_ENTRIES+=("$(printf '{"task":"%s","n_episodes":%s,"success_rate":%s,"measured_pct":%s,"expected_pct":%s,"delta_pp":"%s","client_rc":%s}' \
      "$task" "$RC_N_EPISODES" "$rate" "$measured_pct" "${expected:-null}" "$delta" "$rc")")
  else
    err "   Keine success-rate im Output gefunden (rc=$rc). Log: $client_log"
    JSON_ENTRIES+=("$(printf '{"task":"%s","n_episodes":%s,"success_rate":null,"measured_pct":null,"expected_pct":%s,"delta_pp":"n/a","client_rc":%s}' \
      "$task" "$RC_N_EPISODES" "${expected:-null}" "$rc")")
  fi
done

# ── Zusammenfassung schreiben ────────────────────────────────────────────────
{
  printf '{\n  "timestamp": "%s",\n  "model_path": "%s",\n  "embodiment_tag": "%s",\n  "preset": "%s",\n  "results": [\n' \
    "$TS" "$MODEL_ARG" "$RC_EMBODIMENT_TAG" "$RC_PRESET"
  for i in "${!JSON_ENTRIES[@]}"; do
    sep=","; [[ $i -eq $((${#JSON_ENTRIES[@]}-1)) ]] && sep=""
    printf '    %s%s\n' "${JSON_ENTRIES[$i]}" "$sep"
  done
  printf '  ]\n}\n'
} > "$SUMMARY_JSON"

printf '\n'
ok "Fertig. Zusammenfassung: $SUMMARY_JSON"
log "Akzeptanz: gemessene Quote im 95%%-KI um den erwarteten Wert (bei n_episodes≥100 Halbbreite ~±5–6 pp)."
