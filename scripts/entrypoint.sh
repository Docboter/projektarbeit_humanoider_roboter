#!/usr/bin/env bash
# entrypoint.sh — Container-Entrypoint für vast.ai & andere Cloud-GPU-Plattformen.
#
# Lauft beim Containerstart komplett autonom durch:
#   1. Env-Vars prüfen (HF_TOKEN ist Pflicht, WANDB_API_KEY optional)
#   2. HuggingFace-Token exportieren
#   3. Modell + Datensatz nach /data herunterladen (falls noch nicht vorhanden)
#   4. Datensatz von LeRobot v3.0 → v2.1 konvertieren (falls noch nicht geschehen)
#   5. Fine-tuning starten
#
# Steuerung über Env-Vars (vast.ai: im Feld "Docker options" als -e KEY=VAL setzen):
#   HF_TOKEN              (Pflicht) HuggingFace-Token mit Zugriff auf das Modell
#   WANDB_API_KEY         (optional) W&B-API-Key — ohne diesen wird ohne W&B trainiert
#   MAX_STEPS             (default 30000)
#   GLOBAL_BATCH_SIZE     (default 8)
#   NUM_GPUS              (default 1)
#   WANDB_PROJECT         (default gr00t-g1-dex3)
#   DATA_DIR              (default /data)  — auf vast.ai z. B. einen Disk-Volume nach /data mounten
#   SKIP_DOWNLOAD         (default 0)      — auf 1 setzen, wenn Daten bereits vorhanden sind
#   SKIP_CONVERT          (default 0)      — auf 1 setzen, wenn modality.json bereits existiert
#   SKIP_TRAIN            (default 0)      — auf 1 setzen, um nur Setup zu fahren (für Debug)
#   SHELL_ON_ERROR        (default 0)      — auf 1 setzen, um bei Fehler in eine Shell zu fallen
#
# Bei Aufruf mit Argumenten wird das Skript nicht aktiv — stattdessen wird das
# Argument direkt ausgeführt (nützlich für `docker run … bash`).

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

# ── Manuelle Override-Befehle durchreichen ────────────────────────────────────
# Wenn der Container z. B. mit `docker run … bash` gestartet wird, einfach
# diesen Befehl ausführen statt zu trainieren.
if [[ $# -gt 0 ]]; then
    log "Entrypoint: führe übergebenen Befehl aus: $*"
    exec "$@"
fi

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "\033[1;35m╔══════════════════════════════════════════════════════════════════╗\033[0m"
echo -e "\033[1;35m║   GR00T N1.6 Fine-tuning — Container-Entrypoint (vast.ai-ready)  ║\033[0m"
echo -e "\033[1;35m╚══════════════════════════════════════════════════════════════════╝\033[0m"
echo ""

# ── Konfiguration ─────────────────────────────────────────────────────────────
GROOT_ROOT="${GROOT_ROOT:-/app/Groot-1.6}"
DATA_DIR="${DATA_DIR:-/data}"
export DATA_DIR

MAX_STEPS="${MAX_STEPS:-30000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-gr00t-g1-dex3}"

SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

export MAX_STEPS GLOBAL_BATCH_SIZE NUM_GPUS WANDB_PROJECT

# ── GPU-Check ─────────────────────────────────────────────────────────────────
log "GPU-Check"
if ! nvidia-smi -L &>/dev/null; then
    err "Keine GPU sichtbar. Container ohne --gpus all gestartet oder kein NVIDIA-Treiber?"
    exit 1
fi
nvidia-smi -L | sed 's/^/    /'
ok "GPU verfügbar"
echo ""

# ── HuggingFace-Token ─────────────────────────────────────────────────────────
log "HuggingFace-Token prüfen"
if [[ -z "${HF_TOKEN:-}" ]]; then
    err "HF_TOKEN ist nicht gesetzt."
    err "Auf vast.ai: im Feld 'Docker options' -e HF_TOKEN=hf_... eintragen."
    exit 1
fi
# huggingface-cli liest HUGGING_FACE_HUB_TOKEN bzw. HF_TOKEN; wir setzen beide sicherheitshalber.
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_TOKEN
ok "HF_TOKEN gesetzt (${#HF_TOKEN} Zeichen)"
echo ""

# ── Verzeichnisse anlegen ─────────────────────────────────────────────────────
mkdir -p "$DATA_DIR/models" \
         "$DATA_DIR/unitreerobotics" \
         "$DATA_DIR/G1_Dex3_BlockStacking" \
         "$DATA_DIR/g1_dex3_finetune" \
         "$DATA_DIR/logs"

# ── Download ──────────────────────────────────────────────────────────────────
MODEL_DIR="$DATA_DIR/models/GR00T-N1.6-3B"
DATASET_DIR="$DATA_DIR/unitreerobotics/G1_Dex3_BlockStacking_Dataset"
MODALITY_FILE="$DATASET_DIR/meta/modality.json"

# Heuristik: Modell- und Datensatzordner mit Inhalt → Download überspringen
data_present() {
    [[ -d "$MODEL_DIR"   && -n "$(ls -A "$MODEL_DIR"   2>/dev/null || true)" ]] && \
    [[ -d "$DATASET_DIR" && -n "$(ls -A "$DATASET_DIR" 2>/dev/null || true)" ]]
}

log "Schritt 1/3 — Download (Modell + Datensatz)"
if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    warn "SKIP_DOWNLOAD=1 — Download übersprungen."
elif data_present; then
    ok "Daten bereits vorhanden — Download übersprungen."
    ok "    Modell:    $MODEL_DIR"
    ok "    Datensatz: $DATASET_DIR"
else
    log "Lade Modell + Datensatz nach $DATA_DIR (~25 GB)…"
    bash /scripts/download_data.sh
    ok "Download abgeschlossen."
fi
echo ""

# ── Konvertierung v3.0 → v2.1 ─────────────────────────────────────────────────
log "Schritt 2/3 — Konvertierung LeRobot v3.0 → v2.1"
if [[ "$SKIP_CONVERT" == "1" ]]; then
    warn "SKIP_CONVERT=1 — Konvertierung übersprungen."
elif [[ -f "$MODALITY_FILE" ]]; then
    ok "modality.json bereits vorhanden — Konvertierung übersprungen."
else
    cd "$GROOT_ROOT"
    log "Konvertiere Datensatz…"
    uv run python scripts/lerobot_conversion/convert_v3_to_v2_standalone.py \
        --repo-id unitreerobotics/G1_Dex3_BlockStacking_Dataset \
        --root "$DATA_DIR"
    log "Kopiere modality_4cam.json → modality.json"
    cp "$GROOT_ROOT/examples/G1_DEX3/modality_4cam.json" "$MODALITY_FILE"
    ok "Konvertierung abgeschlossen."
fi
echo ""

# ── W&B-Login (nur via Env, kein interaktiver Fallback) ───────────────────────
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export USE_WANDB=1
    export WANDB_API_KEY
    ok "WANDB_API_KEY gesetzt — Training loggt nach W&B (Projekt: $WANDB_PROJECT)."
else
    export USE_WANDB=0
    warn "Kein WANDB_API_KEY — Training läuft ohne W&B-Logging."
fi
echo ""

# ── Training ──────────────────────────────────────────────────────────────────
log "Schritt 3/3 — Fine-tuning starten"
if [[ "$SKIP_TRAIN" == "1" ]]; then
    warn "SKIP_TRAIN=1 — Training übersprungen. Falle in Shell."
    exec /bin/bash
fi

echo "  Konfiguration:"
printf "    %-20s %s\n" "MAX_STEPS"         "$MAX_STEPS"
printf "    %-20s %s\n" "GLOBAL_BATCH_SIZE" "$GLOBAL_BATCH_SIZE"
printf "    %-20s %s\n" "NUM_GPUS"          "$NUM_GPUS"
printf "    %-20s %s\n" "WANDB_PROJECT"     "$WANDB_PROJECT"
printf "    %-20s %s\n" "DATA_DIR"          "$DATA_DIR"
echo ""

exec bash /scripts/run_finetuning.sh
