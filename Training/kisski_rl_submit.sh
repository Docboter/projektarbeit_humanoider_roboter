#!/usr/bin/env bash
# TL;DR: SLURM-Job-VORLAGE für RL-Fine-tuning (FPO) auf KISSKI — braucht eine RT-Core-GPU-Partition.
# kisski_rl_submit.sh — SLURM-Job für RL-Fine-tuning (FPO) des GR00T-Action-Heads.
#
# ⚠️  VORLAGE / GRUPPE-0-ENTSCHEIDUNG OFFEN ⚠️
# RL braucht bildbasiertes Isaac-Sim-Rendering → eine GPU mit RT-Cores
# (L40 / RTX 4090 / A6000). Die KISSKI-Kernpartitionen `kisski` (A100) und
# `kisski-h100` (H100) haben KEINE RT-Cores und können das 4-Kamera-Rendering
# nicht effizient/stabil ausführen (siehe reinforcement-learning-plan.md §3.5/§6).
# Dieses Skript ist daher eine VORLAGE: vor dem Einreichen muss eine RT-Core-fähige
# Partition (Ada/L40) bestätigt ODER der vast.ai-Pfad (L40/A6000) gewählt werden.
# Der unten stehende RT-Core-Guard bricht auf A100/H100 bewusst ab.
#
# Es nutzt das SIM-SIF (kombiniertes Isaac-Sim + GR00T, aus Simulation/Dockerfile.vastai),
# NICHT das BC-Trainingsimage. Startpunkt ist ein BC-Checkpoint (RL verfeinert ihn).
#
# Einreichen (sobald RT-Core-Partition feststeht — Platzhalter unten anpassen):
#   sbatch Training/kisski_rl_submit.sh
#
# Optionale Überschreibungen (Inline-Prefix vor sbatch, mit --export=ALL):
#   SIM_SIF, DATA_DIR, CHECKPOINT_PATH, HF_CHECKPOINT_REPO,
#   RL_NUM_ENVS, RL_ITERATIONS, RL_ROLLOUT_STEPS, RL_LR, RL_KL_COEF, RL_CLIP
#   GROOT_VERSION (nur 1.6 unterstützt — 1.7 bricht mit Hinweis auf Phase 6 ab)

#SBATCH --job-name=groot-rl
# >>> ANPASSEN: RT-Core-fähige Partition/GPU eintragen (z. B. eine Ada/L40-Partition).
# Die folgenden Zeilen sind PLATZHALTER und müssen vor dem ersten Lauf gesetzt werden.
#SBATCH -p PLACEHOLDER_RTCORE_PARTITION
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH --output=logs/slurm-rl-%j.out
#SBATCH --error=logs/slurm-rl-%j.err
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
SIM_SIF="${SIM_SIF:-$(pick_sif \
    "$KISSKI_SIF_DIR/projekt-humanoider-roboter-sim.sif" \
    "$KISSKI_PROJECT_DIR/images/projekt-humanoider-roboter-sim.sif")}"
DATA_DIR="${DATA_DIR:-$KISSKI_PROJECT_DIR/data}"
REPO_DIR="${REPO_DIR:-$KISSKI_PROJECT_DIR/repo}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/data/checkpoints/groot-g1dex3-checkpoint}"
HF_CHECKPOINT_REPO="${HF_CHECKPOINT_REPO:-}"

RL_NUM_ENVS="${RL_NUM_ENVS:-8}"
RL_ITERATIONS="${RL_ITERATIONS:-500}"
RL_ROLLOUT_STEPS="${RL_ROLLOUT_STEPS:-32}"
RL_LR="${RL_LR:-1e-5}"
RL_KL_COEF="${RL_KL_COEF:-0.1}"
RL_CLIP="${RL_CLIP:-0.2}"

# GR00T-Version: nur 1.6 unterstützt (Default). RL (rl_finetune.py) ist an N1.6-Interna
# gekoppelt (Gr00tN1d6ActionHead-Forward, Processor-Maske) — siehe Phase 6 des Migrationsplans.
# Nur groot_normalize_version() aus der lib sourcen (reine Funktion, keine Exports).
GROOT_VERSION="${GROOT_VERSION:-1.6}"
GROOT_LIB="$REPO_DIR/Training/scripts/lib_groot_version.sh"
if [[ -f "$GROOT_LIB" ]]; then
    source "$GROOT_LIB"
    GROOT_VERSION="$(groot_normalize_version "$GROOT_VERSION")" || exit 1
fi
if [[ "$GROOT_VERSION" == "1.7" ]]; then
    echo "FEHLER: RL-Fine-tuning (FPO) ist für GR00T N1.7 noch NICHT implementiert." >&2
    echo "       rl_finetune.py repliziert bislang N1.6-Interna (Gr00tN1d6ActionHead.forward," >&2
    echo "       Processor-Maske) und ist noch nicht auf N1.7 (Gr00tN1d7ActionHead," >&2
    echo "       state_history_length, options-Dict) umgestellt. Siehe:" >&2
    echo "       docs/weiterfuehrend/groot-n17-migration.md, Abschnitt \"Phase 6 — RL &" >&2
    echo "       Co-Training nachziehen\"." >&2
    echo "       GROOT_VERSION=1.6 setzen (Default) oder Phase 6 zuerst umsetzen." >&2
    exit 1
fi

# ── Tokens aus Dateien (wie kisski_submit.sh) ─────────────────────────────────
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.hf_token}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$HOME/.wandb_key}"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"; [[ -n "$HF_TOKEN" && "$HF_TOKEN" != REPLACE_* ]] && export HF_TOKEN || unset HF_TOKEN
fi
if [[ -z "${WANDB_API_KEY:-}" && -f "$WANDB_KEY_FILE" ]]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_KEY_FILE")"; [[ -n "$WANDB_API_KEY" && "$WANDB_API_KEY" != REPLACE_* ]] && export WANDB_API_KEY || unset WANDB_API_KEY
fi

# ── RT-Core-Guard ─────────────────────────────────────────────────────────────
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo unknown)"
case "$GPU_NAME" in
    *A100*|*H100*)
        echo "FEHLER: GPU '$GPU_NAME' hat keine RT-Cores — bildbasiertes RL-Rendering nicht möglich." >&2
        echo "       Eine Ada/L40-Partition wählen ODER den vast.ai-Pfad (L40/A6000) nutzen." >&2
        echo "       Siehe docs/weiterfuehrend/reinforcement-learning-plan.md §3.5/§6." >&2
        exit 1 ;;
esac

if [[ ! -f "$SIM_SIF" ]]; then
    echo "FEHLER: SIM-SIF nicht gefunden: $SIM_SIF" >&2
    echo "Einmalig erstellen (aus Simulation/Dockerfile.vastai gebautem Docker-Image):" >&2
    echo "    apptainer pull \$HOME/images/projekt-humanoider-roboter-sim.sif docker://<sim-image>:latest" >&2
    exit 1
fi

mkdir -p logs "$DATA_DIR"
module load apptainer

echo "==> RL-Job ${SLURM_JOB_ID:-local} auf $(hostname) — GPU: $GPU_NAME"
echo "    GROOT_VERSION=$GROOT_VERSION  SIM_SIF=$SIM_SIF  CHECKPOINT=$CHECKPOINT_PATH  num_envs=$RL_NUM_ENVS  iters=$RL_ITERATIONS"

APPTAINER_ARGS=(
    --nv
    --bind "$DATA_DIR:/data"
    --env "TMPDIR=/tmp"
    ${HF_TOKEN:+--env "HF_TOKEN=$HF_TOKEN"}
    ${HF_CHECKPOINT_REPO:+--env "HF_CHECKPOINT_REPO=$HF_CHECKPOINT_REPO"}
    --env "CHECKPOINT_PATH=$CHECKPOINT_PATH"
    --env "RL_NUM_ENVS=$RL_NUM_ENVS"
    --env "RL_ITERATIONS=$RL_ITERATIONS"
    --env "RL_ROLLOUT_STEPS=$RL_ROLLOUT_STEPS"
    --env "RL_LR=$RL_LR"
    --env "RL_KL_COEF=$RL_KL_COEF"
    --env "RL_CLIP=$RL_CLIP"
    --env "GROOT_VERSION=$GROOT_VERSION"
)
[[ -n "${WANDB_API_KEY:-}" ]] && APPTAINER_ARGS+=(--env "WANDB_API_KEY=$WANDB_API_KEY" --env "WANDB_MODE=offline")

# Repo-Sim-Skripte mounten, damit Änderungen ohne Image-Rebuild wirken (analog kisski_submit.sh).
if [[ -d "${REPO_DIR}/Simulation/g1_dex3_sim" ]]; then
    APPTAINER_ARGS+=(--bind "${REPO_DIR}/Simulation/g1_dex3_sim:/workspace/g1_dex3_sim")
    APPTAINER_ARGS+=(--bind "${REPO_DIR}/Simulation/scripts:/scripts")
fi

# entrypoint_rl.sh im Sim-Image orchestriert Checkpoint-Load + RL-Start.
apptainer run "${APPTAINER_ARGS[@]}" "$SIM_SIF" /scripts/entrypoint_rl.sh
