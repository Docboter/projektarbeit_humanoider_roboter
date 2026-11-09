#!/bin/bash
# TL;DR: Läuft im Container — lädt Modellgewichte und Datensatz von HuggingFace nach /data.
# Lädt Modellgewichte und Datensätze von HuggingFace herunter.
# Ausführen: bash scripts/download_data.sh
#
# Voraussetzungen:
#   pip install huggingface_hub
#   huggingface-cli login   (oder HF_TOKEN als Umgebungsvariable setzen)
#
# GROOT_VERSION=1.6|1.7 (Default 1.6) steuert, welches Modell geladen wird
# (siehe lib_groot_version.sh). Für 1.7 wird zusätzlich das gated Backbone
# nvidia/Cosmos-Reason2-2B in den HF_HOME-Cache geladen (kein --local-dir,
# damit from_pretrained() den Cache trifft).
#
# Vollstaendigkeit (seit 2026-10-13, Review T5): jeder Teil-Download bekommt einen Marker unter
# $DATA_DIR/.download_state/ — <teil>.incomplete waehrend des Ladens, <teil>.done danach. Bis
# dahin galt ein Ordner schon als vollstaendig, sobald er nicht leer war; ein abgebrochener
# Download wurde so nie fortgesetzt. Bereits fertige Teile ueberspringt das Skript, deshalb ruft
# entrypoint.sh es bei jedem Start auf (billig, wenn alles da ist) — so laedt es auch das N1.7-
# Backbone nach, wenn Modell und Datensatz schon vorhanden sind.
# Altbestand ohne Marker (vor 2026-10-13 geladen): ein nicht leerer Ordner OHNE .incomplete gilt
# weiter als vollstaendig, der .done-Marker wird nachgetragen. Erneut laden: Ordner loeschen.

set -e

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "FEHLER: HF_TOKEN ist nicht gesetzt." >&2
    echo "Bitte 'export HF_TOKEN=hf_...' vor dem Aufruf setzen." >&2
    exit 1
fi

# huggingface-cli erwartet HUGGING_FACE_HUB_TOKEN — beide Varianten unterstützen.
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export HF_TOKEN="${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}"

DATA_DIR="${DATA_DIR:-/data}"
export DATA_DIR

# ── GR00T-Version auflösen (N1.6 | N1.7) ───────────────────────────────────────
source "$(dirname "${BASH_SOURCE[0]}")/lib_groot_version.sh"
groot_resolve || { echo "FEHLER: GR00T-Versionsauflösung fehlgeschlagen (GROOT_VERSION=${GROOT_VERSION:-})." >&2; exit 1; }
echo "==> $(groot_summary)"

echo "==> Zielverzeichnis: $DATA_DIR"
mkdir -p "$DATA_DIR/models" "$DATA_DIR/unitreerobotics" "$DATA_DIR/G1_Dex3_BlockStacking"

DL_STATE_DIR="${DL_STATE_DIR:-$DATA_DIR/.download_state}"
mkdir -p "$DL_STATE_DIR"

# dl_done <teil> <ordner> — Rueckgabe 0, wenn der Teil vollstaendig vorliegt.
dl_done() {
    local name="$1" dir="$2"
    [[ -f "$DL_STATE_DIR/$name.done" ]] && return 0
    [[ -f "$DL_STATE_DIR/$name.incomplete" ]] && return 1   # Abbruch eines frueheren Laufs
    if [[ -d "$dir" && -n "$(ls -A "$dir" 2>/dev/null || true)" ]]; then
        echo "==> $name: Bestand ohne Marker (vor 2026-10-13 geladen) — gilt als vollstaendig, Marker nachgetragen."
        echo "    Zweifel an der Vollstaendigkeit? '$dir' loeschen und neu starten."
        : > "$DL_STATE_DIR/$name.done"
        return 0
    fi
    return 1
}

# dl_item <teil> <ordner> <argumente fuer huggingface-cli download …>
# Unter `set -e` bricht ein gescheiterter Download hier ab und laesst <teil>.incomplete
# liegen — der naechste Start laedt dann weiter (huggingface-cli setzt fort).
dl_item() {
    local name="$1" dir="$2"; shift 2
    if dl_done "$name" "$dir"; then
        echo "==> $name bereits vollstaendig ($dir) — übersprungen."
        return 0
    fi
    : > "$DL_STATE_DIR/$name.incomplete"
    huggingface-cli download "$@"
    rm -f "$DL_STATE_DIR/$name.incomplete"
    : > "$DL_STATE_DIR/$name.done"
}

# GR00T-Modellgewichte (~6 GB; Repo/Name versionsabhängig aus lib_groot_version.sh)
MODEL_TARGET="${MODEL_DIR:-$DATA_DIR/models/$GROOT_MODEL_NAME}"
echo "==> Lade $GROOT_MODEL_NAME Modell ($GROOT_MODEL_REPO)..."
dl_item "model_$(basename "$MODEL_TARGET")" "$MODEL_TARGET" \
    "$GROOT_MODEL_REPO" \
    --local-dir "$MODEL_TARGET" \
    --local-dir-use-symlinks False

# ── N1.7: gated Backbone in den HF_HOME-Cache laden ───────────────────────────
# Cosmos-Reason2-2B wird von N1.7 bei JEDEM Checkpoint-Laden per from_pretrained()
# vom Hub gezogen — ohne --local-dir, damit es im normalen HF-Cache (HF_HOME)
# landet und der Aufruf im Container den Cache trifft (ggf. offline via
# HF_HUB_OFFLINE=1). Idempotent: übersprungen, wenn schon im Cache.
if [[ -n "${GROOT_BACKBONE_REPO:-}" ]]; then
    backbone_cache_glob="$HF_HOME/hub/models--${GROOT_BACKBONE_REPO//\//--}/snapshots/*/config.json"
    # shellcheck disable=SC2086 # bewusst ungequotet für Glob-Expansion
    if compgen -G "$backbone_cache_glob" > /dev/null 2>&1; then
        echo "==> Backbone $GROOT_BACKBONE_REPO bereits im HF-Cache ($HF_HOME) — Download übersprungen."
    else
        echo "==> Lade gated Backbone $GROOT_BACKBONE_REPO in den HF-Cache ($HF_HOME)..."
        # Bewusst OHNE --local-dir: from_pretrained("$GROOT_BACKBONE_REPO") im Training
        # muss denselben Cache treffen. HF_HUB_OFFLINE darf hier nicht gesetzt sein.
        if ! env -u HF_HUB_OFFLINE huggingface-cli download "$GROOT_BACKBONE_REPO"; then
            echo "FEHLER: Download von $GROOT_BACKBONE_REPO fehlgeschlagen." >&2
            echo "Das Repo ist gated — Zugang muss mit demselben HF_TOKEN beantragt werden:" >&2
            echo "    https://huggingface.co/$GROOT_BACKBONE_REPO" >&2
            exit 1
        fi
        ok_check="$(compgen -G "$backbone_cache_glob" || true)"
        if [[ -z "$ok_check" ]]; then
            echo "FEHLER: $GROOT_BACKBONE_REPO nach Download nicht im erwarteten Cache-Pfad gefunden." >&2
            echo "Erwartet: $backbone_cache_glob" >&2
            exit 1
        fi
        echo "==> Backbone-Download abgeschlossen."
    fi
fi

# Unitree G1 Dex3 Block-Stacking Datensatz v3.0 (~18 GB, inkl. Videos)
echo "==> Lade G1_Dex3_BlockStacking_Dataset_v3.0..."
dl_item "dataset_G1_Dex3_BlockStacking_Dataset" "$DATA_DIR/unitreerobotics/G1_Dex3_BlockStacking_Dataset" \
    unitreerobotics/G1_Dex3_BlockStacking_Dataset \
    --repo-type dataset \
    --local-dir "$DATA_DIR/unitreerobotics/G1_Dex3_BlockStacking_Dataset" \
    --local-dir-use-symlinks False

# Nur Metadaten (kein Video) – für schnellen Test
echo "==> Lade G1_Dex3_BlockStacking Metadaten (ohne Videos)..."
dl_item "meta_G1_Dex3_BlockStacking" "$DATA_DIR/G1_Dex3_BlockStacking" \
    unitreerobotics/G1_Dex3_BlockStacking_Dataset \
    --repo-type dataset \
    --include "meta/*" \
    --local-dir "$DATA_DIR/G1_Dex3_BlockStacking" \
    --local-dir-use-symlinks False

echo "==> Fertig. Daten liegen unter $DATA_DIR"
