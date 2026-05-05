#!/bin/bash
set -e

echo "=== Prüfe / lade Assets ==="
python trainer/download_assets.py

echo "=== Konvertiere Daten ==="
python trainer/convert_data.py

echo "=== Starte Training ==="
python trainer/train.py --config trainer/config.yaml