#!/usr/bin/env bash
# kisski_openloop_eval.sh — Open-Loop-Modell-Eval (predicted vs. Ground-Truth-Actions) auf KISSKI
#
# DIAGNOSE 2 (komplementär zum Sim-Replay): Trennt "Modell-Problem" von "Sim-Observation-Problem".
#   Das Modell bekommt die ECHTEN aufgezeichneten Dataset-Beobachtungen (4 Kameras + State +
#   Sprach-Prompt) und sagt Actions vorher. Verglichen wird gegen die GT-Actions → MSE/MAE + Plot.
#   KEINE Isaac Sim, KEIN ZMQ-Server, KEINE RT-Cores → läuft auf der A100-Partition (kisski).
#
# Interpretation:
#   - Niedriges MSE / Predicted-Kurve folgt GT eng  → Modell ist gesund. Der Closed-Loop-Fehler
#     liegt dann an der SIM-OBSERVATION-PIPELINE (Kameras/State/Prompt in Simulation/.../client.py),
#     d.h. das Modell sieht im Sim OOD-Bilder. → dort fixen.
#   - Hohes MSE / Predicted weicht stark von GT ab   → echtes MODELL-/TRAININGS-Problem
#     (Covariate Shift, zu wenig/zu enge Daten). → Datenseite / Trainingsrezept angehen.
#
# Nutzt das fork-eigene Tool: gr00t/eval/open_loop_eval.py (im Trainings-Image gebacken).
#
# Voraussetzungen (einmalig auf dem Login-Node):
#   module load apptainer
#   apptainer pull $HOME/images/projekt-humanoider-roboter.sif \
#       docker://lucam03/projekt-humanoider-roboter:latest
#
#   # Job einreichen (Checkpoint ggf. überschreiben, s.u.):
#   sbatch Training/kisski_openloop_eval.sh
#   CHECKPOINT=/data/.../checkpoint-175000 TRAJ_IDS="0 1 2 3 4" sbatch Training/kisski_openloop_eval.sh
#
# Optionale Überschreibungen (vor sbatch als export oder inline mit --export=ALL):
#   SIF_IMAGE, DATA_DIR, CHECKPOINT, DATASET_PATH, TRAJ_IDS, STEPS, ACTION_HORIZON, DENOISING_STEPS

#SBATCH --job-name=groot-openloop-eval
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH --output=logs/slurm-openloop-%j.out
#SBATCH --error=logs/slurm-openloop-%j.err
#SBATCH --export=ALL

set -euo pipefail

# ── Konfiguration ─────────────────────────────────────────────────────────────
SIF_IMAGE="${SIF_IMAGE:-/user/luca.muecke/u28320/.project/dir.project/images/projekt-humanoider-roboter.sif}"
DATA_DIR="${DATA_DIR:-/mnt/vast-kisski/projects/kisski-humrob/data}"

# Zu evaluierender Checkpoint (Pfad INNERHALB des Containers, also unter /data).
# Default: finaler 175k-Checkpoint. Bei abweichendem Datum-Unterordner anpassen.
CHECKPOINT="${CHECKPOINT:-/data/g1_dex3_finetune/blockstacking/g1_dex3_blockstacking_v1/checkpoints/20260602/checkpoint-175000}"

# Konvertierter Trainings-Datensatz (v2.1, mit modality.json) — exakt der Pfad aus dem Training.
DATASET_PATH="${DATASET_PATH:-/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset}"

# Welche Trajektorien (Episoden) auswerten, wie viele Steps, welcher Action-Horizon.
TRAJ_IDS="${TRAJ_IDS:-0 1 2}"
STEPS="${STEPS:-200}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"   # = Training (model.action_horizon=16)
DENOISING_STEPS="${DENOISING_STEPS:-4}"  # = Training (num_inference_timesteps=4)

# Optional: gr00t-Modul aus einem ausgecheckten Fork einbinden (statt der gebackenen Version).
GROOT_FORK_DIR="${GROOT_FORK_DIR:-}"

OUT_DIR="$DATA_DIR/openloop_eval"

# ── Voraussetzungen prüfen ────────────────────────────────────────────────────
if [[ ! -f "$SIF_IMAGE" ]]; then
    echo "FEHLER: SIF-Image nicht gefunden: $SIF_IMAGE" >&2
    echo "Einmalig erstellen: module load apptainer && apptainer pull $SIF_IMAGE docker://lucam03/projekt-humanoider-roboter:latest" >&2
    exit 1
fi
if [[ ! -d "$(dirname "$DATA_DIR")" ]]; then
    echo "FEHLER: Elternverzeichnis von DATA_DIR nicht erreichbar: $(dirname "$DATA_DIR")" >&2
    exit 1
fi

mkdir -p logs "$OUT_DIR"

CKPT_NAME="$(basename "$CHECKPOINT")"

echo "==> SLURM Job: ${SLURM_JOB_ID:-local} auf $(hostname)"
echo "    GPU:             $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'unbekannt')"
echo "    SIF-Image:       $SIF_IMAGE"
echo "    Checkpoint:      $CHECKPOINT"
echo "    Datensatz:       $DATASET_PATH"
echo "    Trajektorien:    $TRAJ_IDS"
echo "    Steps:           $STEPS"
echo "    Action-Horizon:  $ACTION_HORIZON   Denoising-Steps: $DENOISING_STEPS"
echo "    Ausgabe-Plots:   $OUT_DIR"
echo ""

module load apptainer

export APPTAINER_CACHEDIR="/mnt/vast-kisski/projects/kisski-humrob/apptainer-cache"
export APPTAINER_TMPDIR="/mnt/vast-kisski/projects/kisski-humrob/apptainer-tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# ── Apptainer-Argumente ───────────────────────────────────────────────────────
APPTAINER_ARGS=(
    --nv
    --bind "$DATA_DIR:/data"
    --env "TMPDIR=/tmp"
    --env "UV_OFFLINE=1"
    --env "PYTHONUNBUFFERED=1"
    --env "MPLBACKEND=Agg"          # headless-Plotting (kein Display)
    --env "MPLCONFIGDIR=/tmp/mpl"   # schreibbares matplotlib-Configdir (Container-HOME read-only)
    --env "HF_HUB_OFFLINE=1"        # Checkpoint + Datensatz liegen lokal — kein HF-Zugriff nötig
)
# Optional aktuelleren Fork-Code einbinden
if [[ -n "$GROOT_FORK_DIR" && -d "$GROOT_FORK_DIR/gr00t" ]]; then
    APPTAINER_ARGS+=(--bind "$GROOT_FORK_DIR/gr00t:/app/Groot-1.6/gr00t")
    echo "    gr00t-Modul aus Fork: $GROOT_FORK_DIR/gr00t"
fi

# ── Im Container: pro Trajektorie open_loop_eval ausführen ────────────────────
# (Ein Aufruf pro traj_id, damit jeder Vergleichs-Plot eine eigene Datei bekommt —
#  --save-plot-path wird sonst von der nächsten Trajektorie überschrieben.)
apptainer exec "${APPTAINER_ARGS[@]}" "$SIF_IMAGE" bash -lc '
    set -euo pipefail
    CKPT="'"$CHECKPOINT"'"; DS="'"$DATASET_PATH"'"; OUT="/data/openloop_eval"
    CKPT_NAME="'"$CKPT_NAME"'"
    PY=/app/Groot-1.6/.venv/bin/python
    EVAL=/app/Groot-1.6/gr00t/eval/open_loop_eval.py

    if [[ ! -d "$CKPT" ]]; then
        echo "FEHLER: Checkpoint nicht gefunden: $CKPT" >&2
        echo "Verfügbare Checkpoints:" >&2
        find /data/g1_dex3_finetune -maxdepth 5 -type d -name "checkpoint-*" 2>/dev/null | sort >&2 || true
        exit 1
    fi
    if [[ ! -e "$DS" ]]; then
        echo "FEHLER: Datensatz nicht gefunden: $DS" >&2
        exit 1
    fi

    for tid in '"$TRAJ_IDS"'; do
        echo ""; echo "=================================================="
        echo "  Open-Loop-Eval — Trajektorie $tid"
        echo "=================================================="
        "$PY" "$EVAL" \
            --model-path      "$CKPT" \
            --dataset-path    "$DS" \
            --embodiment-tag  NEW_EMBODIMENT \
            --action-horizon  '"$ACTION_HORIZON"' \
            --denoising-steps '"$DENOISING_STEPS"' \
            --steps           '"$STEPS"' \
            --traj-ids        "$tid" \
            --save-plot-path  "${OUT}/${CKPT_NAME}_traj_${tid}.jpeg"
    done
'

echo ""
echo "==> Open-Loop-Eval abgeschlossen."
echo "    MSE/MAE stehen oben im Log (pro Trajektorie + Durchschnitt)."
echo "    Vergleichs-Plots (GT vs. predicted, pro Action-Dim): $OUT_DIR/${CKPT_NAME}_traj_*.jpeg"
echo ""
echo "==> Plots abrufen (vom Laptop):"
echo "    rsync -avz $(whoami)@transfer.hpc.gwdg.de:$OUT_DIR ./openloop_eval/"
