#!/usr/bin/env bash
# TL;DR: Laedt den echten Trainingsdatensatz von HuggingFace und konvertiert v3.0 → v2.1.
# download_source_dataset.sh — laeuft im Container, aufgerufen von entrypoint.sh.
#
# Nutzt denselben Datensatz-Pfad wie das Training-Image ($DATA_DIR/unitreerobotics/...), damit
# der spaeter kopierte modality.json-Inhalt exakt der Datei entspricht, die das Training-Image
# tatsaechlich verwendet. Konvertierung nutzt den aus dem GR00T-Fork vendorten Konverter
# (siehe Dockerfile Stage "groot-convert-src") — NICHT das komplette GR00T-Trainings-Setup.
#
# Voraussetzungen: HF_TOKEN/HUGGING_FACE_HUB_TOKEN gesetzt (von entrypoint.sh geprueft).

set -euo pipefail

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m v \033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m ! \033[0m %s\n' "$*"; }

DATA_DIR="${DATA_DIR:-/data}"
SOURCE_DATASET_REPO="${SOURCE_DATASET_REPO:-unitreerobotics/G1_Dex3_BlockStacking_Dataset}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"

DATASET_DIR="$DATA_DIR/unitreerobotics/G1_Dex3_BlockStacking_Dataset"
MODALITY_FILE="$DATASET_DIR/meta/modality.json"

mkdir -p "$DATA_DIR/unitreerobotics"

# ── Download (v3.0-Rohformat, inkl. Videos) ────────────────────────────────────
# Fertig-Marker NEBEN dem Datensatzordner, nicht darin: der Konverter verschiebt den Ordner
# nach <name>_v30 und legt das v2.1-Ergebnis neu an — ein Marker darin ginge dabei verloren.
# Frueher galt jedes nicht leere Verzeichnis als vollstaendig, also auch ein abgebrochener
# Download.
DOWNLOAD_DONE="$DATA_DIR/unitreerobotics/.$(basename "$DATASET_DIR").download_done"

download_complete() {
    [[ -f "$DOWNLOAD_DONE" ]] && return 0
    # Bestandsordner aus der Zeit vor dem Marker: eine vorhandene modality.json heisst, dass
    # Download UND Konvertierung schon durchgelaufen sind — Marker nachtragen.
    if [[ -f "$MODALITY_FILE" ]]; then
        touch "$DOWNLOAD_DONE"
        return 0
    fi
    return 1
}

if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    warn "SKIP_DOWNLOAD=1 — Download uebersprungen."
elif download_complete; then
    ok "Datensatz bereits vollstaendig — Download uebersprungen ($DATASET_DIR)."
else
    if [[ -d "$DATASET_DIR" && -n "$(ls -A "$DATASET_DIR" 2>/dev/null || true)" ]]; then
        # Nicht leer, aber ohne Marker: abgebrochener Download. huggingface-cli setzt fort
        # und laedt nur fehlende Dateien nach.
        warn "Datensatzordner unvollstaendig (kein Fertig-Marker) — Download wird fortgesetzt."
    fi
    log "Lade $SOURCE_DATASET_REPO (~18 GB, inkl. Videos)…"
    /opt/venv-tools/bin/huggingface-cli download "$SOURCE_DATASET_REPO" \
        --repo-type dataset \
        --local-dir "$DATASET_DIR" \
        --local-dir-use-symlinks False
    touch "$DOWNLOAD_DONE"
    ok "Download abgeschlossen."
fi

# ── Konvertierung v3.0 → v2.1 (vendorter Konverter aus dem GR00T-Fork) ─────────
if [[ "$SKIP_CONVERT" == "1" ]]; then
    warn "SKIP_CONVERT=1 — Konvertierung uebersprungen."
elif [[ -f "$MODALITY_FILE" ]]; then
    ok "modality.json bereits vorhanden — Konvertierung uebersprungen."
else
    log "Konvertiere $SOURCE_DATASET_REPO nach LeRobot v2.1…"
    "$TOOLS_PYTHON" /app/groot-convert/lerobot_conversion/convert_v3_to_v2_standalone.py \
        --repo-id "$SOURCE_DATASET_REPO" \
        --root "$DATA_DIR"
    log "Kopiere modality_4cam.json → meta/modality.json (unveraendert, fuer den"
    log "COTRAIN-Gleichheitscheck in run_finetuning_cotrain.sh)"
    cp /app/groot-convert/modality_4cam.json "$MODALITY_FILE"
    ok "Konvertierung abgeschlossen."
fi
