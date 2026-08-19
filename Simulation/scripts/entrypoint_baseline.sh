#!/usr/bin/env bash
# TL;DR: Sim-Container-Entrypoint: Baseline-Eval mit un-finetuntem GR00T + stock-G1-Dex1-Greifer.
# entrypoint_baseline.sh — Sim-Container-Entrypoint für den BASELINE-Test:
#   un-finetuntes GR00T-N1.6-3B + stock Unitree G1 mit Dex1-Parallelgreifer
#   (Embodiment UNITREE_G1) auf dem Block-Stacking-Task.
#
# Wird über entrypoint_sim.sh per SIM_MODE=baseline angesprungen (oder direkt).
# Spiegelt entrypoint_sim.sh, aber:
#   - GR00T-Server mit --embodiment-tag UNITREE_G1 (statt NEW_EMBODIMENT)
#   - kein Hand-Recolor (DEX3-spezifisch)
#   - Pro-Gruppe-Dims werden vor dem Eval introspiziert (dump_unitree_g1_dims.py)
#   - Sim-Client = g1_gripper_sim/run_g1_gripper_sim_eval.py
#
# Pflicht-Env-Vars:
#   HF_CHECKPOINT_REPO=nvidia/GR00T-N1.6-3B + HF_TOKEN   (Basismodell-Download)
#     ODER CHECKPOINT_PATH auf ein vorhandenes Modell-Verzeichnis.
#   ASSET_PATH  — Pfad zu g1_gripper.usd (Dex1-Greifer-USD; muss vorhanden/gemountet sein,
#                 Erzeugung via Simulation/g1_dex3_sim/convert_urdf_to_usd.py mit
#                 --urdf .../g1_29dof_mode_15_with_dex1_1.urdf)
#
# NUR GR00T N1.6. Der Baseline-Test hängt am Embodiment-Tag UNITREE_G1, und der bedeutet
# in beiden Versionen etwas anderes:
#   N1.6: UNITREE_G1 = "unitree_g1"                                   (Tag des Basismodells)
#   N1.7: UNITREE_G1 = "unitree_g1_full_body_with_waist_height_nav_cmd"
#         — ein Sim-Full-Body-Post-Train-Tag mit Waist/Height/Nav-Kommandos, für den das
#         Basismodell keinen Zero-Shot-Kopf mitbringt. Ein Lauf damit würde entweder
#         scheitern oder etwas messen, das mit dieser Baseline nichts zu tun hat.
# Mit GROOT_VERSION=1.7 (oder einem N1.7-Basismodell) bricht das Skript deshalb ab.
# Der nächstliegende N1.7-Kandidat wäre REAL_G1 ("real_g1_relative_eef_relative_joints") —
# ungeprüft und anderes Aktionsformat (relative EEF), also bewusst nicht eingebaut.
#
# Optionale Env-Vars (Defaults unten / im Dockerfile):
#   GROOT_VERSION (nur 1.6 zulässig, s. o.),
#   NUM_EPISODES, EXECUTION_HORIZON, TASK_DESCRIPTION, ZMQ_PORT, SKIP_DOWNLOAD,
#   SHELL_ON_ERROR, LIVESTREAM, LIVESTREAM_PORT, LIVESTREAM_MEDIA_PORT, LIVE_KEEP_VIDEO,
#   PUBLIC_IP  (alle wie entrypoint_sim.sh; die LIVE-Variante steckt gemeinsam in
#   /scripts/lib_livestream.sh — Anleitung: docs/simulation/live-ansicht.md)

set -euo pipefail

# ── Instance-Log (SSH-Zugriff) ────────────────────────────────────────────────
mkdir -p "${DATA_DIR:-/data}/logs"
exec > >(tee -a "${DATA_DIR:-/data}/logs/entrypoint.log") 2>&1

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

if [[ $# -gt 0 ]]; then
    log "Entrypoint: führe übergebenen Befehl aus: $*"
    exec "$@"
fi

echo ""
echo -e "\033[1;35m╔══════════════════════════════════════════════════════════════════╗\033[0m"
echo -e "\033[1;35m║   GR00T N1.6 BASELINE — stock G1 + Dex1-Greifer (UNITREE_G1)      ║\033[0m"
echo -e "\033[1;35m╚══════════════════════════════════════════════════════════════════╝\033[0m"
echo ""

# ── Konfiguration ─────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/data}"

# ── GR00T-Version: hier ausschließlich N1.6 (Begründung im Kopf dieser Datei) ──
# Anders als bei entrypoint_sim.sh entscheidet NICHT der Checkpoint, sondern der
# Embodiment-Tag: UNITREE_G1 heißt in N1.7 etwas anderes. Default deshalb 1.6.
export GROOT_VERSION_DEFAULT="1.6"
if [[ -r /scripts/lib_groot_version.sh ]]; then
    # shellcheck source=lib_groot_version.sh
    source /scripts/lib_groot_version.sh
    GROOT_WANT="$(groot_normalize_version "${GROOT_VERSION:-}")"
    if [[ "$GROOT_WANT" == "1.7" ]]; then
        err "GROOT_VERSION=1.7 — der Baseline-Test läuft nur mit GR00T N1.6."
        err "  Grund: Embodiment-Tag UNITREE_G1 bedeutet in N1.7"
        err "  'unitree_g1_full_body_with_waist_height_nav_cmd' (Sim-Full-Body-Post-Train,"
        err "  kein Zero-Shot-Kopf im Basismodell) statt wie in N1.6 schlicht 'unitree_g1'."
        err "  Entweder GROOT_VERSION=1.6 setzen (Basismodell nvidia/GR00T-N1.6-3B) oder die"
        err "  Baseline weglassen. Ein N1.7-Gegenstück (REAL_G1) ist ungeprüft."
        exit 1
    fi
    # `auto` ergibt hier 1.6: es gibt nichts zu erkennen, der Tag legt die Version fest.
    groot_resolve 1.6
else
    warn "lib_groot_version.sh fehlt im Image — bleibe fest bei GR00T N1.6."
    GROOT_VERSION="1.6"
    GROOT_ROOT="${GROOT_ROOT:-/app/Groot-1.6}"
    GROOT_VENV_PY="$GROOT_ROOT/.venv/bin/python"
    groot_detect_version() { return 1; }
fi

ZMQ_PORT="${ZMQ_PORT:-5555}"
NUM_EPISODES="${NUM_EPISODES:-20}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-8}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-stack the blocks}"
ASSET_PATH="${ASSET_PATH:-/workspace/assets/g1_gripper.usd}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$DATA_DIR/checkpoints}"
HF_CHECKPOINT_REPO="${HF_CHECKPOINT_REPO:-}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
DIMS_FILE="${DIMS_FILE:-$DATA_DIR/g1_baseline_dims.json}"
EMBODIMENT_TAG="UNITREE_G1"

# LIVE-Variante (WebRTC-Viewport) — identisch zu entrypoint_sim.sh, weil beide dieselbe
# Lib nutzen statt je einer eigenen Kopie des Blocks.
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
VIDEO_DIR="$(livestream_video_dir "$DATA_DIR/sim_videos")"

mkdir -p "$DATA_DIR/sim_videos" "$DATA_DIR/sim_results" "$DATA_DIR/logs"

# ── SSH-Server (vast.ai) ─────────────────────────────────────────────────────
mkdir -p /root/.ssh && chmod 700 /root/.ssh
[[ -n "${PUBLIC_KEY:-}" ]] && echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
/usr/sbin/sshd 2>/dev/null || true

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
log "Schritt 1/4 — Checkpoint (Basismodell)"
if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    warn "SKIP_DOWNLOAD=1 — Download übersprungen."
    [[ -d "$CHECKPOINT_PATH" ]] || { err "CHECKPOINT_PATH '$CHECKPOINT_PATH' existiert nicht."; exit 1; }
    ok "Checkpoint: $CHECKPOINT_PATH"
elif [[ -n "$HF_CHECKPOINT_REPO" ]]; then
    CHECKPOINT_PATH="$DATA_DIR/checkpoints/$(basename "$HF_CHECKPOINT_REPO")"
    if [[ -d "$CHECKPOINT_PATH" && -n "$(ls -A "$CHECKPOINT_PATH" 2>/dev/null)" ]]; then
        ok "Checkpoint bereits vorhanden: $CHECKPOINT_PATH"
    else
        log "Lade Basismodell von HuggingFace: $HF_CHECKPOINT_REPO"
        "$GROOT_VENV_PY" - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${HF_CHECKPOINT_REPO}", local_dir="${CHECKPOINT_PATH}", token="${HF_TOKEN:-None}")
print("Download abgeschlossen.")
EOF
        ok "Checkpoint heruntergeladen: $CHECKPOINT_PATH"
    fi
else
    if [[ ! -d "$CHECKPOINT_PATH" || -z "$(ls -A "$CHECKPOINT_PATH" 2>/dev/null)" ]]; then
        err "CHECKPOINT_PATH '$CHECKPOINT_PATH' ist leer/fehlt."
        err "Setze -e HF_TOKEN=hf_... -e HF_CHECKPOINT_REPO=nvidia/GR00T-N1.6-3B"
        exit 1
    fi
    ok "Checkpoint gefunden: $CHECKPOINT_PATH"
fi

# Auch ein N1.7-BASISMODELL (nvidia/GR00T-N1.7-3B) ist hier falsch — aus demselben
# Grund wie GROOT_VERSION=1.7 oben. Lieber jetzt abbrechen als nach dem Modell-Laden.
CKPT_VERSION="$(groot_detect_version "$CHECKPOINT_PATH" 2>/dev/null || true)"
if [[ "$CKPT_VERSION" == "1.7" ]]; then
    err "'$CHECKPOINT_PATH' ist ein N1.7-Modell (model_type Gr00tN1d7 in config.json)."
    err "  Der Baseline-Test ist an den N1.6-Embodiment-Tag UNITREE_G1 ('unitree_g1')"
    err "  gebunden; in N1.7 heißt derselbe Name etwas anderes (Full-Body-Sim-Tag)."
    err "  Basismodell nvidia/GR00T-N1.6-3B verwenden (HF_CHECKPOINT_REPO)."
    exit 1
fi
echo ""

# ── USD-Asset prüfen ──────────────────────────────────────────────────────────
log "Schritt 2/4 — USD-Asset (Dex1-Greifer)"
if [[ ! -f "$ASSET_PATH" ]]; then
    err "ASSET_PATH '$ASSET_PATH' fehlt."
    err "Erzeuge g1_gripper.usd einmalig (lokal/Container) aus der Dex1-URDF:"
    err "  isaaclab.sh -p /workspace/g1_dex3_sim/convert_urdf_to_usd.py \\"
    err "    --urdf .../g1_description/g1_29dof_mode_15_with_dex1_1.urdf \\"
    err "    --output $ASSET_PATH"
    err "und mounte/bake es nach \$ASSET_PATH."
    exit 1
fi
ok "Asset: $ASSET_PATH"
echo ""

# ── Pro-Gruppe-Dims introspizieren ────────────────────────────────────────────
log "Schritt 3/4 — State/Action-Dims aus Checkpoint lesen ($EMBODIMENT_TAG)"
"$GROOT_VENV_PY" /scripts/dump_unitree_g1_dims.py \
    --model-path "$CHECKPOINT_PATH" \
    --embodiment-tag unitree_g1 \
    --out "$DIMS_FILE"
ok "Dims geschrieben: $DIMS_FILE"
echo ""

# ── GR00T-Policy-Server starten (UNITREE_G1) ──────────────────────────────────
log "Schritt 4/4 — GR00T-Policy-Server (Port $ZMQ_PORT, Embodiment $EMBODIMENT_TAG)"
LIBCUDA_SO1=$(ldconfig -p 2>/dev/null | grep "libcuda\.so\.1" | awk '{print $NF}' | head -1 || true)
if [[ -n "$LIBCUDA_SO1" ]]; then
    ln -sf "$LIBCUDA_SO1" /usr/lib/x86_64-linux-gnu/libcuda.so 2>/dev/null || true
    ldconfig 2>/dev/null || true
fi

GROOT_SERVER_LOG="$DATA_DIR/logs/groot_server.log"
"$GROOT_VENV_PY" "$GROOT_ROOT/gr00t/eval/run_gr00t_server.py" \
    --model-path "$CHECKPOINT_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --use-sim-policy-wrapper \
    --port "$ZMQ_PORT" \
    > "$GROOT_SERVER_LOG" 2>&1 &
GROOT_SERVER_PID=$!
ok "GR00T-Server gestartet (PID $GROOT_SERVER_PID) — Log: $GROOT_SERVER_LOG"

cleanup() {
    warn "Stoppe GR00T-Server (PID $GROOT_SERVER_PID) …"
    kill "$GROOT_SERVER_PID" 2>/dev/null || true
    wait "$GROOT_SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "    Warte auf Server-Bereitschaft (Port $ZMQ_PORT) …"
MAX_WAIT=300; WAITED=0
while ! timeout 1 bash -c "echo > /dev/tcp/localhost/$ZMQ_PORT" 2>/dev/null; do
    if ! kill -0 "$GROOT_SERVER_PID" 2>/dev/null; then
        err "GR00T-Server-Prozess unerwartet beendet. Log:"; tail -20 "$GROOT_SERVER_LOG" >&2; exit 1
    fi
    sleep 2; WAITED=$((WAITED + 2))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        err "Server nicht bereit nach ${MAX_WAIT}s. Log:"; tail -20 "$GROOT_SERVER_LOG" >&2; exit 1
    fi
    printf "    … %ds\n" "$WAITED"
done
ok "Server bereit (nach ${WAITED}s)"
echo ""

# ── Isaac-Lab-Sim-Client starten ──────────────────────────────────────────────
log "Isaac-Lab-Sim-Client (Baseline)"
unset VIRTUAL_ENV
export PYTHONUNBUFFERED=1

APP_FLAGS=( --enable_cameras )
livestream_app_flags
APP_FLAGS+=( "${LS_APP_FLAGS[@]}" )
livestream_banner
echo ""

${ISAACLAB_PATH}/isaaclab.sh -p /workspace/g1_gripper_sim/run_g1_gripper_sim_eval.py \
    "${APP_FLAGS[@]}" \
    --server "tcp://localhost:$ZMQ_PORT" \
    --dims-file      "$DIMS_FILE" \
    --num-episodes   "$NUM_EPISODES" \
    --execution-horizon "$EXECUTION_HORIZON" \
    --task-description  "$TASK_DESCRIPTION" \
    --video-dir      "$VIDEO_DIR" \
    --results-file   "$DATA_DIR/sim_results/results.json" \
    --asset-path     "$ASSET_PATH" \
    --ping-retries   20

RESULTS_FILE="$DATA_DIR/sim_results/results.json"
if [[ ! -s "$RESULTS_FILE" ]]; then
    err "Sim-Eval hat keine Ergebnis-Datei erzeugt ($RESULTS_FILE)."
    err "Der Sim-Client ist vermutlich abgestürzt — letzte Container-Ausgaben oben prüfen."
    exit 1
fi

ok "Baseline-Sim-Eval abgeschlossen."
echo ""
echo "  Ergebnisse: $RESULTS_FILE"
if [[ -n "$VIDEO_DIR" ]]; then
    echo "  Videos:     $VIDEO_DIR/"
else
    echo "  Videos:     keine (LIVE-Variante lief; LIVE_KEEP_VIDEO=1 schreibt sie zusätzlich)"
fi
echo ""
