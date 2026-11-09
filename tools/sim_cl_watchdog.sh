#!/usr/bin/env bash
# TL;DR: Host-Werkzeug — Closed Loop (Protokoll Lauf 82) mit Hänger-Wächter: steht das Log STALL_S lang, Eval im Container beenden und neu starten.
#
# sim_cl_watchdog.sh — ein Closed Loop über server_rl_run.sh eval, der Hänger übersteht.
#
# Hintergrund: Am 2026-10-25 hingen zwei Closed Loops mitten in einer Episode (Hauptthread
# dreht, Policy-Server und GPU untätig, keine Fehlermeldung; Chronik Läufe 89/90). Ursache
# offen. Dieser Wächter beendet in dem Fall alle Eval-Prozesse im Container und startet den
# Lauf neu (höchstens RETRIES-mal). In einem Container darf nur ein Closed Loop laufen
# (fester Port 5555), deshalb ist „alle Eval-Prozesse im Container“ eindeutig.
#
# Aufruf:
#   tools/sim_cl_watchdog.sh TAG CONTAINER GPU GROOT_VERSION HF_REPO CHECKPOINT_PATH
# Beispiel (Lauf 93):
#   SIM_CUBE_EDGE=0.04 tools/sim_cl_watchdog.sh cl_cotrain_v3_cube4_n17_edge4_scene70_dy3 \
#     groot-rl-eval17 1 1.7 luca-mue/groot-g1dex3-cotrain-v3-cube4-n17-ckpt44000 \
#     /data/checkpoints/cotrain-v3-cube4-n17-ckpt44000
#
# Protokoll-Defaults = Lauf 82 (Szene aus Lauf 70, Held-out-Layout dy3, 20 × 120 s, DR aus):
#   NUM_EPISODES=20 EPISODE_LENGTH_S=120 EXECUTION_HORIZON=8 DR_ENABLED=0 DR_SEED=0
#   EVAL_LAYOUT=/data/eval/layout_heldout_dy3.json
#   SIM_TABLE_SIZE="0.8 2.0" SIM_WRIST_EYE="0.02 0.0 0.08" SIM_WRIST_TARGET="0.30 0.0 -0.08"
#   SIM_WRIST_HFOV=85
# Jede davon lässt sich per Umgebung überschreiben; SIM_CUBE_EDGE wird nur durchgereicht,
# wenn gesetzt (4-cm-Modelle: SIM_CUBE_EDGE=0.04).
# Wächter: STALL_S (Default 900 s ohne Log-Wachstum), RETRIES (Default 2).
# Log: $HOST_DATA_DIR/logs/<TAG>_<JJJJMMTT>.log, Videos: /data/sim_videos/<TAG>/.

set -u
if [[ $# -ne 6 ]]; then
  sed -n '12,16p' "$0"
  exit 2
fi
TAG=$1 C=$2 G=$3 V=$4 REPO=$5 CKPT=$6
STALL_S=${STALL_S:-900} RETRIES=${RETRIES:-2}
HOST_DATA_DIR=${HOST_DATA_DIR:-$HOME/groot-rl-data}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
LOG=$HOST_DATA_DIR/logs/${TAG}_$(date +%Y%m%d).log
mkdir -p "$HOST_DATA_DIR/logs"

export NUM_EPISODES=${NUM_EPISODES:-20} EPISODE_LENGTH_S=${EPISODE_LENGTH_S:-120}
export EXECUTION_HORIZON=${EXECUTION_HORIZON:-8} DR_ENABLED=${DR_ENABLED:-0} DR_SEED=${DR_SEED:-0}
export EVAL_LAYOUT=${EVAL_LAYOUT:-/data/eval/layout_heldout_dy3.json}
export SIM_TABLE_SIZE="${SIM_TABLE_SIZE:-0.8 2.0}" SIM_WRIST_EYE="${SIM_WRIST_EYE:-0.02 0.0 0.08}"
export SIM_WRIST_TARGET="${SIM_WRIST_TARGET:-0.30 0.0 -0.08}" SIM_WRIST_HFOV="${SIM_WRIST_HFOV:-85}"
export SIM_VIDEO_DIR=/data/sim_videos/$TAG

kill_eval() {
  local pids
  pids=$(docker exec "$C" ps -eo pid,cmd \
    | grep -E 'entrypoint_sim|run_g1_dex3_sim_eval|isaaclab.cli|run_gr00t_server' \
    | grep -v grep | awk '{print $1}' | tr '\n' ' ')
  # shellcheck disable=SC2086
  [[ -n "$pids" ]] && docker exec "$C" kill $pids
  sleep 15
  pids=$(docker exec "$C" ps -eo pid,cmd | grep -E 'run_g1_dex3_sim_eval|run_gr00t_server' \
    | grep -v grep | awk '{print $1}' | tr '\n' ' ')
  # shellcheck disable=SC2086
  [[ -n "$pids" ]] && docker exec "$C" kill -9 $pids
  sleep 5
}

cd "$REPO_DIR" || exit 1
for attempt in $(seq 0 "$RETRIES"); do
  {
    echo "=== $TAG Versuch $attempt Start $(date '+%F %T')"
    echo "=== Protokoll: NUM_EPISODES=$NUM_EPISODES EPISODE_LENGTH_S=$EPISODE_LENGTH_S" \
      "EVAL_LAYOUT=$EVAL_LAYOUT SIM_CUBE_EDGE=${SIM_CUBE_EDGE:-(Default)} GPU=$G GROOT_VERSION=$V"
  } >> "$LOG"
  MENU=0 RL_CONTAINER=$C RL_EXEC_GPU=$G GROOT_VERSION=$V HF_CHECKPOINT_REPO=$REPO \
    CHECKPOINT_PATH=$CKPT ./Simulation/server_rl_run.sh eval >> "$LOG" 2>&1 &
  runner=$!
  stalled=0
  while kill -0 "$runner" 2>/dev/null; do
    sleep 60
    age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
    if (( age > STALL_S )); then
      echo "=== $TAG HAENGER: Log seit ${age}s unveraendert, $(date '+%F %T'), Versuch $attempt" >> "$LOG"
      kill_eval
      kill "$runner" 2>/dev/null
      wait "$runner" 2>/dev/null
      stalled=1
      break
    fi
  done
  (( stalled )) && continue
  wait "$runner"; rc=$?
  echo "=== $TAG Ende rc=$rc $(date '+%F %T')" >> "$LOG"
  exit "$rc"
done
echo "=== $TAG aufgegeben nach $RETRIES Neustarts $(date '+%F %T')" >> "$LOG"
exit 1
