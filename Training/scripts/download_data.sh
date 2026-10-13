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

# GR00T-Modellgewichte (~6 GB; Repo/Name versionsabhängig aus lib_groot_version.sh)
echo "==> Lade $GROOT_MODEL_NAME Modell ($GROOT_MODEL_REPO)..."
huggingface-cli download "$GROOT_MODEL_REPO" \
    --local-dir "$DATA_DIR/models/$GROOT_MODEL_NAME" \
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
huggingface-cli download unitreerobotics/G1_Dex3_BlockStacking_Dataset \
    --repo-type dataset \
    --local-dir "$DATA_DIR/unitreerobotics/G1_Dex3_BlockStacking_Dataset" \
    --local-dir-use-symlinks False

# Nur Metadaten (kein Video) – für schnellen Test
echo "==> Lade G1_Dex3_BlockStacking Metadaten (ohne Videos)..."
huggingface-cli download unitreerobotics/G1_Dex3_BlockStacking_Dataset \
    --repo-type dataset \
    --include "meta/*" \
    --local-dir "$DATA_DIR/G1_Dex3_BlockStacking" \
    --local-dir-use-symlinks False

echo "==> Fertig. Daten liegen unter $DATA_DIR"
