#!/usr/bin/env bash
# kisski_open_loop_eval.sh — Open-Loop-Evaluation des GR00T-Checkpoints auf KISSKI
#
# Läuft den GR00T open_loop_eval.py auf echten Dataset-Trajektorien:
#   - Lädt Checkpoint (lokal, kein Server nötig)
#   - Führt Inferenz auf N Trajektorien durch
#   - Berechnet MAE/MSE pro Trajektorie und über alle Trajektorien
#   - Speichert Plots (GT- vs. Predicted-Actions) auf Projektspeicher
#
# Einreichen:
#   sbatch Training/kisski_open_loop_eval.sh
#
# Optionale Überschreibungen (vor sbatch als export):
#   CHECKPOINT, TRAJ_IDS, STEPS, DENOISING_STEPS

# ── SLURM-Direktiven ──────────────────────────────────────────────────────────
#SBATCH --job-name=groot-open-loop-eval
#SBATCH -p kisski
#SBATCH --gres=gpu:A100:1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --output=/user/luca.muecke/u28320/.project/dir.project/logs/slurm-open-loop-%j.out
#SBATCH --error=/user/luca.muecke/u28320/.project/dir.project/logs/slurm-open-loop-%j.err
#SBATCH --export=ALL

set -euo pipefail

# ── Konfiguration ─────────────────────────────────────────────────────────────
SERVER_SIF="${SERVER_SIF:-/user/luca.muecke/u28320/.project/dir.project/images/projekt-humanoider-roboter.sif}"
GROOT_FORK_DIR="${GROOT_FORK_DIR:-/mnt/vast-kisski/projects/kisski-humrob/repo-groot}"
DATA_DIR="${DATA_DIR:-/mnt/vast-kisski/projects/kisski-humrob/data}"

# Checkpoint: neuester (checkpoint-3000)
CHECKPOINT="${CHECKPOINT:-/data/g1_dex3_finetune/blockstacking/g1_dex3_blockstacking_v1/checkpoints/20260529/checkpoint-3000}"

# Dataset (LeRobot-Format, GR00T-konvertiert — enthält episodes.jsonl + modality.json)
DATASET_PATH="${DATASET_PATH:-/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset}"

# Trajektorien-IDs: gleichmäßig über 301 Episoden verteilt
TRAJ_IDS="${TRAJ_IDS:-0 10 20 50 100 150 200 250 300}"

# Steps pro Trajektorie (capped auf traj_length)
STEPS="${STEPS:-300}"

# Denoising-Steps (4 = schnell, 16 = genauer)
DENOISING_STEPS="${DENOISING_STEPS:-4}"

# Ausgabe-Plots → bind-mount auf /tmp/open_loop_eval im Container
PLOTS_DIR="$DATA_DIR/open_loop_plots"

# ── Voraussetzungen ───────────────────────────────────────────────────────────
if [[ ! -f "$SERVER_SIF" ]]; then
    echo "FEHLER: SIF nicht gefunden: $SERVER_SIF" >&2
    exit 1
fi

mkdir -p "$PLOTS_DIR"

echo "==> SLURM Job: ${SLURM_JOB_ID:-local} auf $(hostname)"
echo "    GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unbekannt')"
echo "    SIF:        $SERVER_SIF"
echo "    Checkpoint: $CHECKPOINT"
echo "    Dataset:    $DATASET_PATH"
echo "    Trajektorie-IDs: $TRAJ_IDS"
echo "    Steps:      $STEPS"
echo "    Plots:      $PLOTS_DIR"
echo ""

module load apptainer

export APPTAINER_CACHEDIR="/mnt/vast-kisski/projects/kisski-humrob/apptainer-cache"
export APPTAINER_TMPDIR="/mnt/vast-kisski/projects/kisski-humrob/apptainer-tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# ── Open-Loop-Eval ────────────────────────────────────────────────────────────
APPTAINER_ARGS=(
    --nv
    --bind "$DATA_DIR:/data"
    --bind "$PLOTS_DIR:/tmp/open_loop_eval"
    --env "TMPDIR=/tmp"
    --env "UV_OFFLINE=1"
)

# gr00t-Modul aus Fork einbinden (falls vorhanden)
if [[ -d "$GROOT_FORK_DIR/gr00t" ]]; then
    APPTAINER_ARGS+=(--bind "$GROOT_FORK_DIR/gr00t:/app/Groot-1.6/gr00t")
    echo "    Fork:       $GROOT_FORK_DIR/gr00t"
fi
if [[ -d "$GROOT_FORK_DIR/examples/G1_DEX3" ]]; then
    APPTAINER_ARGS+=(--bind "$GROOT_FORK_DIR/examples/G1_DEX3:/app/Groot-1.6/examples/G1_DEX3")
fi

# Trajektorien-IDs als tyro-kompatible Argumente aufbauen
# tyro erwartet: --traj-ids 0 10 20 ...
TRAJ_ARGS=""
for id in $TRAJ_IDS; do
    TRAJ_ARGS="$TRAJ_ARGS $id"
done

echo "==> Starte Open-Loop-Eval …"
apptainer exec "${APPTAINER_ARGS[@]}" "$SERVER_SIF" \
    bash -lc "cd /app/Groot-1.6 && \
        .venv/bin/python gr00t/eval/open_loop_eval.py \
            --model-path $CHECKPOINT \
            --dataset-path $DATASET_PATH \
            --traj-ids $TRAJ_ARGS \
            --steps $STEPS \
            --action-horizon 16 \
            --denoising-steps $DENOISING_STEPS \
            --embodiment-tag NEW_EMBODIMENT"

echo ""
echo "==> Open-Loop-Eval abgeschlossen."
echo ""

# ── Plots anzeigen ────────────────────────────────────────────────────────────
echo "==> Gespeicherte Plots:"
ls -lh "$PLOTS_DIR"/*.jpeg 2>/dev/null || echo "    (keine Plots gefunden)"

echo ""
echo "==> Plots abrufen (vom Laptop):"
echo "    rsync -avz $(whoami)@transfer.hpc.gwdg.de:$PLOTS_DIR/ ./open_loop_plots/"
