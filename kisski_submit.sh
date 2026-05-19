#!/usr/bin/env bash
# kisski_submit.sh — SLURM-Job-Script für GR00T N1.6 Fine-tuning auf KISSKI
#
# Voraussetzungen (einmalig):
#   1. Image konvertieren:
#        module load apptainer
#        apptainer pull $HOME/images/projekt-humanoider-roboter.sif \
#            docker://lucam03/projekt-humanoider-roboter:latest
#
#   2. Daten auf Scratch übertragen (von lokalem Rechner):
#        rsync -avz ./data/ <username>@transfer.hpc.gwdg.de:/scratch/<username>/data/
#
#   3. Tokens setzen und Job einreichen:
#        export HF_TOKEN=hf_...
#        export WANDB_API_KEY=...
#        sbatch kisski_submit.sh
#
# Optionale Überschreibungen (vor sbatch als export):
#   MAX_STEPS, GLOBAL_BATCH_SIZE, NUM_GPUS, WANDB_PROJECT, DATA_DIR
#   SKIP_DOWNLOAD, SKIP_CONVERT, SKIP_TRAIN

#SBATCH --job-name=groot-finetune
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

# ── Konfiguration ─────────────────────────────────────────────────────────────
SIF_IMAGE="${SIF_IMAGE:-$HOME/images/projekt-humanoider-roboter.sif}"
DATA_DIR="${DATA_DIR:-/scratch/$USER/data}"

MAX_STEPS="${MAX_STEPS:-30000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-gr00t-g1-dex3}"

SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

# ── Voraussetzungen prüfen ────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "FEHLER: HF_TOKEN ist nicht gesetzt. Vor sbatch exportieren: export HF_TOKEN=hf_..." >&2
    exit 1
fi

if [[ ! -f "$SIF_IMAGE" ]]; then
    echo "FEHLER: SIF-Image nicht gefunden: $SIF_IMAGE" >&2
    echo "Einmalig erstellen:" >&2
    echo "    module load apptainer" >&2
    echo "    mkdir -p \$HOME/images" >&2
    echo "    apptainer pull \$HOME/images/projekt-humanoider-roboter.sif docker://lucam03/projekt-humanoider-roboter:latest" >&2
    exit 1
fi

mkdir -p logs

# ── Laufumgebung anzeigen ─────────────────────────────────────────────────────
echo "==> SLURM Job: $SLURM_JOB_ID auf $(hostname)"
echo "    SIF-Image:         $SIF_IMAGE"
echo "    DATA_DIR:          $DATA_DIR"
echo "    MAX_STEPS:         $MAX_STEPS"
echo "    GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE"
echo "    NUM_GPUS:          $NUM_GPUS"
echo "    WANDB_PROJECT:     $WANDB_PROJECT"
echo ""

module load apptainer

apptainer run \
    --nv \
    --bind "$DATA_DIR:/data" \
    --env HF_TOKEN="$HF_TOKEN" \
    --env MAX_STEPS="$MAX_STEPS" \
    --env GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    --env NUM_GPUS="$NUM_GPUS" \
    --env WANDB_PROJECT="$WANDB_PROJECT" \
    --env SKIP_DOWNLOAD="$SKIP_DOWNLOAD" \
    --env SKIP_CONVERT="$SKIP_CONVERT" \
    --env SKIP_TRAIN="$SKIP_TRAIN" \
    ${WANDB_API_KEY:+--env WANDB_API_KEY="$WANDB_API_KEY"} \
    "$SIF_IMAGE"
