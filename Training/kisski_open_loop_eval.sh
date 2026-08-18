#!/usr/bin/env bash
# kisski_open_loop_eval.sh — Checkpoint-Auswahl per Open-Loop-Eval auf KISSKI
#
# Läuft checkpoint_sweep.py über ALLE Checkpoints eines Trainingslaufs:
#   - Lädt jeden Checkpoint einzeln (lokal, kein Server nötig)
#   - Wertet ihn auf ZURÜCKGEHALTENEN Episoden aus (split="test")
#   - Berechnet MSE/MAE je Episode und im Mittel je Checkpoint
#   - Nennt den besten Checkpoint und schreibt checkpoint_sweep.json
#   - Speichert Plots (GT- vs. Predicted-Actions) auf Projektspeicher
#
# Voraussetzung: Der Trainingslauf lief mit TRAIN_TEST_SPLIT=1. Ohne
# zurückgehaltene Episoden misst der Sweep gegen Trainingsdaten und kann
# Overfitting prinzipiell nicht zeigen — er warnt dann laut.
#
# Einreichen:
#   sbatch Training/kisski_open_loop_eval.sh
#
# Optionale Überschreibungen (vor sbatch als export):
#   RUN_DIR, DATASET_PATH, EVAL_SPLIT, EVAL_NUM_TRAJ, EVAL_STEPS, EVAL_CHECKPOINTS

# ── SLURM-Direktiven ──────────────────────────────────────────────────────────
#SBATCH --job-name=groot-ckpt-sweep
#SBATCH -p kisski
#SBATCH --gres=gpu:A100:1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 02:00:00
# Logpfade sind RELATIV zum Verzeichnis, aus dem `sbatch` aufgerufen wird — in
# #SBATCH-Zeilen kann SLURM keine Umgebungsvariablen auflösen, ein Knopf wie
# KISSKI_PROJECT_DIR wirkt hier also nicht. Vor dem ersten Absenden einmal
#   mkdir -p logs
# (SLURM legt die Datei an, aber nicht das Verzeichnis — fehlt es, startet der Job
# gar nicht). Anderer Ort ohne Skript-Änderung:
#   sbatch --output=/pfad/logs/slurm-ckpt-sweep-%j.out --error=/pfad/logs/slurm-ckpt-sweep-%j.err <skript>
#SBATCH --output=logs/slurm-ckpt-sweep-%j.out
#SBATCH --error=logs/slurm-ckpt-sweep-%j.err
#SBATCH --export=ALL

set -euo pipefail

# ── Portable Pfad-Auflösung ───────────────────────────────────────────────────
# Bewusst in JEDEM SLURM-Skript dupliziert statt in ein lib-Skript ausgelagert: SLURM
# kopiert das Batch-Skript vor der Ausführung in sein Spool-Verzeichnis, darum zeigen $0
# und BASH_SOURCE im Job NICHT mehr ins Repo — ein `source "$(dirname "$0")/lib_…"` würde
# fehlschlagen. Die paar Zeilen hier sind der Preis für Selbstgenügsamkeit.
#
# EIN Knopf für den Projektspeicher, alles Weitere leitet sich davon ab:
#     KISSKI_PROJECT_DIR=/mnt/vast-kisski/projects/<projekt> sbatch <dieses Skript>
# Das wirkt beim Absenden, weil oben `#SBATCH --export=ALL` steht — die Umgebung der
# Submit-Shell landet unverändert im Job.
KISSKI_PROJECT_DIR="${KISSKI_PROJECT_DIR:-/mnt/vast-kisski/projects/kisski-humrob}"

# SIF-Ablage. Reihenfolge: eigenes $HOME/images zuerst — genau dorthin lädt der in
# CLAUDE.md und docs/training/kisski-hpc.md dokumentierte `apptainer pull`; danach eine
# gemeinsame Kopie auf dem Projektspeicher. Eigener Ort: KISSKI_SIF_DIR=… setzen.
KISSKI_SIF_DIR="${KISSKI_SIF_DIR:-$HOME/images}"
pick_sif() {   # erste EXISTIERENDE Datei gewinnt; sonst der erste Kandidat, damit die
    local f    # Fehlermeldung weiter unten einen konkreten Pfad nennen kann
    for f in "$@"; do [[ -f "$f" ]] && { printf '%s\n' "$f"; return; }; done
    printf '%s\n' "$1"
}

# ── Konfiguration ─────────────────────────────────────────────────────────────
SERVER_SIF="${SERVER_SIF:-$(pick_sif \
    "$KISSKI_SIF_DIR/projekt-humanoider-roboter.sif" \
    "$KISSKI_PROJECT_DIR/images/projekt-humanoider-roboter.sif")}"
GROOT_FORK_DIR="${GROOT_FORK_DIR:-$KISSKI_PROJECT_DIR/repo-groot}"
DATA_DIR="${DATA_DIR:-$KISSKI_PROJECT_DIR/data}"
# Wie in kisski_submit.sh: fester Pfad auf dem Projektspeicher. NICHT über BASH_SOURCE
# herleiten — sbatch führt eine Kopie aus dem SLURM-Spool aus, nicht die Datei im Repo.
REPO_DIR="${REPO_DIR:-$KISSKI_PROJECT_DIR/repo}"

# Lauf-Verzeichnis (nicht ein einzelner Checkpoint — der Sweep sucht selbst).
#   Standard-Lauf : /data/g1_dex3_finetune/blockstacking
#   Vision-Lauf   : /data/g1_dex3_finetune/blockstacking_vision
RUN_DIR="${RUN_DIR:-/data/g1_dex3_finetune/blockstacking}"

# Dataset (LeRobot-Format, GR00T-konvertiert — enthält episodes.jsonl + modality.json)
DATASET_PATH="${DATASET_PATH:-/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset}"

# Split: "test" = die vom Training zurückgehaltenen Episoden. Das ist der Punkt.
EVAL_SPLIT="${EVAL_SPLIT:-test}"

# Anzahl gleichmäßig über den Split verteilter Episoden
EVAL_NUM_TRAJ="${EVAL_NUM_TRAJ:-6}"

# Steps pro Episode (capped auf die Episodenlänge)
EVAL_STEPS="${EVAL_STEPS:-300}"

# Optional: nur bestimmte Step-Nummern auswerten (leer = alle gefundenen)
EVAL_CHECKPOINTS="${EVAL_CHECKPOINTS:-}"

# Ausgabe (liegt über den Bind-Mount direkt auf dem Projektspeicher)
EVAL_OUT="${EVAL_OUT:-$RUN_DIR/checkpoint_sweep.json}"
EVAL_PLOT_DIR="${EVAL_PLOT_DIR:-$RUN_DIR/open_loop_plots}"

# ── Voraussetzungen ───────────────────────────────────────────────────────────
if [[ ! -f "$SERVER_SIF" ]]; then
    echo "FEHLER: SIF nicht gefunden: $SERVER_SIF" >&2
    exit 1
fi

echo "==> SLURM Job: ${SLURM_JOB_ID:-local} auf $(hostname)"
echo "    GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unbekannt')"
echo "    SIF:        $SERVER_SIF"
echo "    Lauf:       $RUN_DIR"
echo "    Dataset:    $DATASET_PATH"
echo "    Split:      $EVAL_SPLIT   (Episoden: $EVAL_NUM_TRAJ, Steps: $EVAL_STEPS)"
echo "    JSON:       $EVAL_OUT"
echo "    Plots:      $EVAL_PLOT_DIR"
echo ""

module load apptainer

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$KISSKI_PROJECT_DIR/apptainer-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$KISSKI_PROJECT_DIR/apptainer-tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# ── Checkpoint-Sweep ──────────────────────────────────────────────────────────
APPTAINER_ARGS=(
    --nv
    --bind "$DATA_DIR:/data"
    --env "TMPDIR=/tmp"
    --env "UV_OFFLINE=1"
)

# Skripte aus dem Repo einbinden, damit Änderungen ohne Image-Rebuild wirken
# (gleiche Konvention wie kisski_submit.sh).
if [[ -d "$REPO_DIR/Training/scripts" ]]; then
    APPTAINER_ARGS+=(--bind "$REPO_DIR/Training/scripts:/scripts")
    echo "    Skripte:    $REPO_DIR/Training/scripts (Bind-Mount)"
fi

# gr00t-Modul aus Fork einbinden (falls vorhanden)
if [[ -d "$GROOT_FORK_DIR/gr00t" ]]; then
    APPTAINER_ARGS+=(--bind "$GROOT_FORK_DIR/gr00t:/app/Groot-1.6/gr00t")
    echo "    Fork:       $GROOT_FORK_DIR/gr00t"
fi
if [[ -d "$GROOT_FORK_DIR/examples/G1_DEX3" ]]; then
    APPTAINER_ARGS+=(--bind "$GROOT_FORK_DIR/examples/G1_DEX3:/app/Groot-1.6/examples/G1_DEX3")
fi

SWEEP_ARGS=(
    --run-dir        "$RUN_DIR"
    --dataset-path   "$DATASET_PATH"
    --split          "$EVAL_SPLIT"
    --num-trajectories "$EVAL_NUM_TRAJ"
    --steps          "$EVAL_STEPS"
    --out            "$EVAL_OUT"
    --plot-dir       "$EVAL_PLOT_DIR"
)
[[ -n "$EVAL_CHECKPOINTS" ]] && SWEEP_ARGS+=(--checkpoints "$EVAL_CHECKPOINTS")

echo "==> Starte Checkpoint-Sweep …"
apptainer exec "${APPTAINER_ARGS[@]}" "$SERVER_SIF" \
    bash -lc "cd /app/Groot-1.6 && .venv/bin/python /scripts/checkpoint_sweep.py $(printf '%q ' "${SWEEP_ARGS[@]}")"

echo ""
echo "==> Checkpoint-Sweep abgeschlossen."
echo ""

# ── Ergebnisse anzeigen ───────────────────────────────────────────────────────
HOST_OUT="${EVAL_OUT/#\/data/$DATA_DIR}"
HOST_PLOTS="${EVAL_PLOT_DIR/#\/data/$DATA_DIR}"

if [[ -f "$HOST_OUT" ]]; then
    echo "==> Bester Checkpoint laut $HOST_OUT:"
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('   ', d['best_checkpoint_path']); print('    MSE', d['checkpoints'][[c['step'] for c in d['checkpoints']].index(d['best_step'])]['mean_mse'])" "$HOST_OUT" 2>/dev/null \
        || echo "    (JSON vorhanden, Auswertung hier nicht möglich)"
else
    echo "==> Kein JSON unter $HOST_OUT gefunden."
fi

echo ""
echo "==> Ergebnisse abrufen (vom Laptop):"
echo "    rsync -avz $(whoami)@transfer.hpc.gwdg.de:$HOST_OUT ./"
echo "    rsync -avz $(whoami)@transfer.hpc.gwdg.de:$HOST_PLOTS/ ./open_loop_plots/"
