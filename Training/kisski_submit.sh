#!/usr/bin/env bash
# kisski_submit.sh — SLURM-Job-Script für GR00T N1.6 Fine-tuning auf KISSKI
#
# Voraussetzungen (einmalig):
#   1. Nur dieses Skript auf den Cluster kopieren:
#        scp kisski_submit.sh <username>@glogin-gpu.hpc.gwdg.de:~/
#
#   2. SIF-Image erstellen (auf dem Cluster, einmalig):
#        module load apptainer
#        mkdir -p $HOME/images
#        apptainer pull $HOME/images/projekt-humanoider-roboter.sif \
#            docker://lucam03/projekt-humanoider-roboter:latest
#
#   3. Tokens setzen und Job einreichen:
#        export HF_TOKEN=hf_...
#        export WANDB_API_KEY=...          # optional
#        sbatch kisski_submit.sh
#
# Das Skript klont automatisch das GitHub-Repo und lädt Modell + Datensatz
# von HuggingFace herunter — kein manuelles rsync nötig.
#
# Optionale Überschreibungen (vor sbatch als export):
#   GITHUB_REPO, GITHUB_BRANCH, GITHUB_TOKEN, REPO_DIR
#   MAX_STEPS, GLOBAL_BATCH_SIZE, NUM_GPUS, WANDB_PROJECT, DATA_DIR
#   SKIP_GIT_PULL, SKIP_DOWNLOAD, SKIP_CONVERT, SKIP_TRAIN

#SBATCH --job-name=groot-finetune
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --export=ALL

set -euo pipefail

# ── Konfiguration ─────────────────────────────────────────────────────────────
SIF_IMAGE="${SIF_IMAGE:-/user/luca.muecke/u28320/.project/dir.project/images/projekt-humanoider-roboter.sif}"
DATA_DIR="${DATA_DIR:-/mnt/vast-kisski/projects/kisski-humrob/data}"

# Prüfen ob DATA_DIR-Pfad vom Compute-Node aus erreichbar ist
if [[ ! -d "$(dirname "$DATA_DIR")" ]]; then
    echo "FEHLER: Elternverzeichnis von DATA_DIR nicht erreichbar: $(dirname "$DATA_DIR")" >&2
    echo "       /mnt/vast-kisski ist auf diesem Node möglicherweise nicht gemountet." >&2
    echo "       Alternativen: export DATA_DIR=\$SHARED_TMPDIR/data  (temporär)" >&2
    exit 1
fi

GITHUB_REPO="${GITHUB_REPO:-https://github.com/Docboter/projektarbeit_humanoider_roboter.git}"
GITHUB_BRANCH="${GITHUB_BRANCH:-training-luca-KISSKI}"
# Repo muss vorab auf dem Login-Node geklont werden (Compute-Nodes haben keinen Internet-Zugang).
# Einmalig: git clone --branch $GITHUB_BRANCH --depth 1 $GITHUB_REPO /mnt/vast-kisski/projects/kisski-humrob/repo
REPO_DIR="${REPO_DIR:-/mnt/vast-kisski/projects/kisski-humrob/repo}"
SKIP_GIT_PULL="${SKIP_GIT_PULL:-1}"

MAX_STEPS="${MAX_STEPS:-175000}"          # ~5 Epochen (Datensatz: 281k Frames / Batch 8 ≈ 35k Schritte/Epoche)
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-gr00t-g1-dex3}"

# Checkpoints: alle 5000 Schritte (~52 min) ein Checkpoint, Limit hoch genug,
# dass NICHTS gelöscht wird → 35 Checkpoints gleichmäßig über den ganzen Lauf
# verteilt, sodass sich der Trainingsfortschritt im Nachhinein rekonstruieren lässt.
# Achtung: jeder Checkpoint ≈ 22 GB (13 GB optimizer.pt + 9 GB Gewichte) → ~770 GB gesamt.
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-40}"

SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-1}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

# ── Voraussetzungen prüfen ────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" && "${SKIP_DOWNLOAD:-1}" != "1" ]]; then
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
mkdir -p "$DATA_DIR"

# ── Laufumgebung anzeigen ─────────────────────────────────────────────────────
echo "==> SLURM Job: ${SLURM_JOB_ID:-local} auf $(hostname)"
echo "    SIF-Image:         $SIF_IMAGE"
echo "    DATA_DIR:          $DATA_DIR"
echo "    REPO_DIR:          $REPO_DIR"
echo "    GITHUB_REPO:       $GITHUB_REPO (Branch: $GITHUB_BRANCH)"
echo "    MAX_STEPS:         $MAX_STEPS"
echo "    SAVE_STEPS:        $SAVE_STEPS  (SAVE_TOTAL_LIMIT=$SAVE_TOTAL_LIMIT)"
echo "    GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE"
echo "    NUM_GPUS:          $NUM_GPUS"
echo "    WANDB_PROJECT:     $WANDB_PROJECT"
echo ""

module load apptainer

# ── Schritt 1: GitHub-Repo klonen / aktualisieren ────────────────────────────
echo "==> Schritt 1/2 — GitHub-Repo (Skripte + Konfiguration)"
if [[ "$SKIP_GIT_PULL" == "1" ]]; then
    echo "    SKIP_GIT_PULL=1 — übersprungen."
else
    # Bei privaten Repos: export GITHUB_TOKEN=ghp_... vor sbatch setzen
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        REPO_URL="https://${GITHUB_TOKEN}@${GITHUB_REPO#https://}"
    else
        REPO_URL="$GITHUB_REPO"
    fi

    if [[ -d "$REPO_DIR/.git" ]]; then
        echo "    Aktualisiere $REPO_DIR auf Branch $GITHUB_BRANCH …"
        git -C "$REPO_DIR" fetch origin
        git -C "$REPO_DIR" checkout "$GITHUB_BRANCH"
        git -C "$REPO_DIR" pull --ff-only origin "$GITHUB_BRANCH"
    else
        echo "    Klone nach $REPO_DIR …"
        git clone --branch "$GITHUB_BRANCH" --depth 1 "$REPO_URL" "$REPO_DIR"
    fi
    echo "    Repo aktuell: $(git -C "$REPO_DIR" log -1 --oneline)"
fi
echo ""

# ── Schritt 2: Container starten ─────────────────────────────────────────────
# entrypoint.sh im Container lädt automatisch Modell + Datensatz von HuggingFace
# (/scripts/download_data.sh im Container) und startet dann das Training.
echo "==> Schritt 2/2 — Container starten (HF-Download + Training)"

APPTAINER_ARGS=(
    --nv
    --bind "$DATA_DIR:/data"
    --env "TMPDIR=/tmp"
    --env "UV_OFFLINE=1"
    ${HF_TOKEN:+--env "HF_TOKEN=$HF_TOKEN"}
    --env "MAX_STEPS=$MAX_STEPS"
    --env "SAVE_STEPS=$SAVE_STEPS"
    --env "SAVE_TOTAL_LIMIT=$SAVE_TOTAL_LIMIT"
    --env "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
    --env "NUM_GPUS=$NUM_GPUS"
    --env "WANDB_PROJECT=$WANDB_PROJECT"
    --env "SKIP_DOWNLOAD=$SKIP_DOWNLOAD"
    --env "SKIP_CONVERT=$SKIP_CONVERT"
    --env "SKIP_TRAIN=$SKIP_TRAIN"
)

# Repo-Skripte in den Container mounten, damit GitHub-Änderungen sofort wirken
# (überschreibt die ins SIF-Image gebackenen Versionen ohne Image-Rebuild).
# Pflicht: ohne diesen Mount laufen veraltete Container-Scripts (fehlendes --no-sync).
if [[ ! -d "${REPO_DIR}/Training/scripts" ]]; then
    echo "FEHLER: ${REPO_DIR}/Training/scripts nicht gefunden." >&2
    echo "       Repo einmalig klonen:" >&2
    echo "       git clone --branch $GITHUB_BRANCH --depth 1 $GITHUB_REPO $REPO_DIR" >&2
    exit 1
fi
APPTAINER_ARGS+=(--bind "${REPO_DIR}/Training/scripts:/scripts")
echo "    Skripte aus Repo: ${REPO_DIR}/Training/scripts"

# G1_DEX3-Konfiguration + LeRobot-Konverter aus dem lucam06/Isaac-GR00T Fork mounten
# (im Container-Image fehlen diese Dateien — nur die offizielle NVIDIA-Version ist eingebackt).
# Einmalig: git clone --branch luca/g1-dex3 --depth 1 https://github.com/lucam06/Isaac-GR00T.git /mnt/vast-kisski/projects/kisski-humrob/repo-groot
GROOT_FORK_DIR="${GROOT_FORK_DIR:-/mnt/vast-kisski/projects/kisski-humrob/repo-groot}"
if [[ ! -d "$GROOT_FORK_DIR/examples/G1_DEX3" ]]; then
    echo "FEHLER: $GROOT_FORK_DIR/examples/G1_DEX3 nicht gefunden." >&2
    echo "       lucam06-Fork einmalig klonen:" >&2
    echo "       git clone --branch luca/g1-dex3 --depth 1 https://github.com/lucam06/Isaac-GR00T.git $GROOT_FORK_DIR" >&2
    exit 1
fi
APPTAINER_ARGS+=(--bind "$GROOT_FORK_DIR/examples/G1_DEX3:/app/Groot-1.6/examples/G1_DEX3")
APPTAINER_ARGS+=(--bind "$GROOT_FORK_DIR/scripts/lerobot_conversion:/app/Groot-1.6/scripts/lerobot_conversion")
# gr00t-Modul aus Fork: Container hat nur gr00t_n1d7-Code, Modell ist aber N1.6 (Gr00tN1d6).
# Editable install (egg-info) lädt Code direkt aus /app/Groot-1.6/gr00t/.
APPTAINER_ARGS+=(--bind "$GROOT_FORK_DIR/gr00t:/app/Groot-1.6/gr00t")
echo "    G1_DEX3-Config aus Fork: $GROOT_FORK_DIR/examples/G1_DEX3"
echo "    gr00t-Modul (N1.6)  aus Fork: $GROOT_FORK_DIR/gr00t"

[[ -n "${WANDB_API_KEY:-}" ]] && APPTAINER_ARGS+=(--env "WANDB_API_KEY=$WANDB_API_KEY")
# Compute-Nodes haben kein Internet — W&B immer im Offline-Modus betreiben.
APPTAINER_ARGS+=(--env "WANDB_MODE=offline")
APPTAINER_ARGS+=(--env "WANDB_DIR=/data/g1_dex3_finetune")

apptainer run "${APPTAINER_ARGS[@]}" "$SIF_IMAGE"
