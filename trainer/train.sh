#!/bin/bash
set -e

DATA_DIR="/workspace/data"
OUTPUT_DIR="/workspace/output"

DATASET_ID="unitreerobotics/G1_Dex3_BlockStacking_Dataset"

echo "== GPU CHECK =="
python3 -c "import torch; print(torch.cuda.is_available())"

if [ ! -f "$DATA_DIR/.done" ]; then
  echo "Downloading dataset..."
  python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$DATASET_ID",
    repo_type="dataset",
    local_dir="$DATA_DIR"
)
EOF
  touch $DATA_DIR/.done
fi

echo "== START TRAINING =="

python3 /workspace/train_policy.py