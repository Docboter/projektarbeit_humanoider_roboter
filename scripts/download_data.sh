#!/bin/bash
# Lädt Modellgewichte und Datensätze von HuggingFace herunter.
# Ausführen: bash scripts/download_data.sh
#
# Voraussetzungen:
#   pip install huggingface_hub
#   huggingface-cli login   (oder HF_TOKEN als Umgebungsvariable setzen)

set -e

DATA_DIR="${DATA_DIR:-/data}"

echo "==> Zielverzeichnis: $DATA_DIR"
mkdir -p "$DATA_DIR/models" "$DATA_DIR/unitreerobotics" "$DATA_DIR/G1_Dex3_BlockStacking"

# GR00T N1.6-3B Modellgewichte (~6 GB)
echo "==> Lade GR00T-N1.6-3B Modell..."
huggingface-cli download nvidia/GR00T-N1.6-3B \
    --local-dir "$DATA_DIR/models/GR00T-N1.6-3B" \
    --local-dir-use-symlinks False

# Unitree G1 Dex3 Block-Stacking Datensatz v3.0 (~18 GB, inkl. Videos)
echo "==> Lade G1_Dex3_BlockStacking_Dataset_v3.0..."
huggingface-cli download unitreerobotics/G1_Dex3_BlockStacking_Dataset \
    --repo-type dataset \
    --local-dir "$DATA_DIR/unitreerobotics/G1_Dex3_BlockStacking_Dataset_v3.0" \
    --local-dir-use-symlinks False

# Nur Metadaten (kein Video) – für schnellen Test
echo "==> Lade G1_Dex3_BlockStacking Metadaten (ohne Videos)..."
huggingface-cli download unitreerobotics/G1_Dex3_BlockStacking_Dataset \
    --repo-type dataset \
    --include "meta/*" \
    --local-dir "$DATA_DIR/G1_Dex3_BlockStacking" \
    --local-dir-use-symlinks False

echo "==> Fertig. Daten liegen unter $DATA_DIR"
