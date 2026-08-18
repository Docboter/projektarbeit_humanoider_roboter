#!/usr/bin/env bash
# kisski_robocasa_ref_submit.sh — RoboCasa GR-1 Tabletop Referenz-Eval auf KISSKI
#   (Basismodell GR00T-N1.6-3B, zero-shot, Embodiment GR1; robosuite/MuJoCo).
#
# ZWECK / KONTEXT
#   Validiert die GR00T-Inferenz-Pipeline gegen einen externen Goldstandard.
#   Plan + Hintergrund:  docs/simulation/basismodell-referenzaufgabe.md
#   Bedienung + Caveats: docs/simulation/robocasa-referenz-eval.md
#
#   Nutzt das TRAININGS-Image (GR00T-Venv, kein Isaac Sim) und braucht KEINE
#   RT-Cores -> Partition `kisski` (A100) ist ausreichend; MuJoCo rendert per EGL.
#
# WICHTIG — TEMPLATE (zwei Caveats vor dem ersten Lauf, siehe Ops-Doc §"Klärung"):
#   1) EINMALIGES SETUP nötig (Internet + Schreibzugriff): erstellt die isolierte
#      robocasa_uv-Venv + lädt robosuite/Assets. KISSKI-Compute-Nodes sind ggf.
#      offline und das Apptainer-Image ist read-only -> Setup auf dem LOGIN-Node
#      in einem schreibbaren Sandbox/--writable-tmpfs ausführen (Ops-Doc Schritt 1).
#      Danach hier mit RC_SKIP_SETUP=1 (Default) nur noch die Eval fahren.
#   2) EGL ist im Trainings-Image bereits vorhanden (libegl1 + NVIDIA-EGL-ICD +
#      MUJOCO_GL=egl, siehe Training/Dockerfile) -> kein Blocker. Nur libGLU ist
#      ungeprüft; falls MuJoCo es verlangt, MUJOCO_GL=osmesa als CPU-Fallback.
#
# Einreichen (nach erfolgtem Setup):
#   sbatch Simulation/kisski_robocasa_ref_submit.sh
# Überschreibungen (vor sbatch als export):
#   RC_PRESET (top|smoke|full|custom), RC_N_EPISODES, RC_N_ENVS, RC_TASKS,
#   RC_MODEL_PATH, SERVER_SIF, GROOT_FORK_DIR, RC_SKIP_SETUP

# ── SLURM-Direktiven ──────────────────────────────────────────────────────────
#SBATCH --job-name=groot-robocasa-ref
#SBATCH -p kisski
#SBATCH --gres=gpu:A100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
# Logpfade sind RELATIV zum Verzeichnis, aus dem `sbatch` aufgerufen wird — in
# #SBATCH-Zeilen kann SLURM keine Umgebungsvariablen auflösen, ein Knopf wie
# KISSKI_PROJECT_DIR wirkt hier also nicht. Vor dem ersten Absenden einmal
#   mkdir -p logs
# (SLURM legt die Datei an, aber nicht das Verzeichnis — fehlt es, startet der Job
# gar nicht). Anderer Ort ohne Skript-Änderung:
#   sbatch --output=/pfad/logs/slurm-robocasa-ref-%j.out --error=/pfad/logs/slurm-robocasa-ref-%j.err <skript>
#SBATCH --output=logs/slurm-robocasa-ref-%j.out
#SBATCH --error=logs/slurm-robocasa-ref-%j.err
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
REPO_DIR="${REPO_DIR:-$KISSKI_PROJECT_DIR/repo}"
DATA_DIR="${DATA_DIR:-$KISSKI_PROJECT_DIR/data}"

# Referenz-Eval-Parameter (an run_robocasa_ref_eval.sh durchgereicht)
RC_PRESET="${RC_PRESET:-top}"
RC_N_EPISODES="${RC_N_EPISODES:-50}"
RC_N_ENVS="${RC_N_ENVS:-8}"
RC_N_ACTION_STEPS="${RC_N_ACTION_STEPS:-8}"
RC_MAX_EPISODE_STEPS="${RC_MAX_EPISODE_STEPS:-720}"
RC_TASKS="${RC_TASKS:-}"
RC_MODEL_PATH="${RC_MODEL_PATH:-/data/models/GR00T-N1.6-3B}"
RC_PORT="${RC_PORT:-5757}"
RC_SKIP_SETUP="${RC_SKIP_SETUP:-1}"   # Setup vorab einmalig (Ops-Doc Schritt 1)

# ── Voraussetzungen ───────────────────────────────────────────────────────────
if [[ ! -f "$SERVER_SIF" ]]; then
    echo "FEHLER: SIF nicht gefunden: $SERVER_SIF" >&2
    exit 1
fi
RUNNER="$REPO_DIR/Simulation/robocasa_reference/run_robocasa_ref_eval.sh"
if [[ ! -f "$RUNNER" ]]; then
    echo "FEHLER: Runner nicht gefunden: $RUNNER (REPO_DIR korrekt?)" >&2
    exit 1
fi

echo "==> SLURM Job: ${SLURM_JOB_ID:-local} auf $(hostname)"
echo "    GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unbekannt')"
echo "    SIF:        $SERVER_SIF"
echo "    Modell:     $RC_MODEL_PATH"
echo "    Preset:     $RC_PRESET  (n_episodes=$RC_N_EPISODES, n_envs=$RC_N_ENVS)"
echo "    SkipSetup:  $RC_SKIP_SETUP"
echo ""

module load apptainer

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$KISSKI_PROJECT_DIR/apptainer-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$KISSKI_PROJECT_DIR/apptainer-tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# ── Bind-Mounts ───────────────────────────────────────────────────────────────
# gr00t + external_dependencies SCHREIBBAR aus dem VAST-Fork-Checkout, damit die
# robocasa_uv-Venv und das robocasa-Submodul persistent sind (Setup einmalig).
APPTAINER_ARGS=(
    --nv
    --bind "$DATA_DIR:/data"
    --bind "$REPO_DIR/Simulation/robocasa_reference:/workspace/robocasa_reference"
    --env "TMPDIR=/tmp"
    --env "UV_OFFLINE=1"
    --env "PYTHONUNBUFFERED=1"
    --env "MUJOCO_GL=${MUJOCO_GL:-egl}"
    --env "PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}"
    --env "RC_PRESET=$RC_PRESET"
    --env "RC_N_EPISODES=$RC_N_EPISODES"
    --env "RC_N_ENVS=$RC_N_ENVS"
    --env "RC_N_ACTION_STEPS=$RC_N_ACTION_STEPS"
    --env "RC_MAX_EPISODE_STEPS=$RC_MAX_EPISODE_STEPS"
    --env "RC_MODEL_PATH=$RC_MODEL_PATH"
    --env "RC_PORT=$RC_PORT"
    --env "RC_SKIP_SETUP=$RC_SKIP_SETUP"
    --env "RC_RESULTS_DIR=/data/robocasa_ref"
)
[[ -n "$RC_TASKS" ]] && APPTAINER_ARGS+=(--env "RC_TASKS=$RC_TASKS")
[[ -n "${HF_TOKEN:-}" ]] && APPTAINER_ARGS+=(--env "HF_TOKEN=$HF_TOKEN")

if [[ -d "$GROOT_FORK_DIR/gr00t" ]]; then
    APPTAINER_ARGS+=(--bind "$GROOT_FORK_DIR/gr00t:/app/Groot-1.6/gr00t")
    echo "    Fork gr00t: $GROOT_FORK_DIR/gr00t"
fi
if [[ -d "$GROOT_FORK_DIR/external_dependencies" ]]; then
    APPTAINER_ARGS+=(--bind "$GROOT_FORK_DIR/external_dependencies:/app/Groot-1.6/external_dependencies")
fi

echo "==> Starte RoboCasa-Referenz-Eval …"
apptainer exec "${APPTAINER_ARGS[@]}" "$SERVER_SIF" \
    bash -lc "bash /workspace/robocasa_reference/run_robocasa_ref_eval.sh"

echo ""
echo "==> Fertig. Ergebnisse unter $DATA_DIR/robocasa_ref/ (summary-*.json)."
echo "    Abrufen:  rsync -avz $(whoami)@transfer.hpc.gwdg.de:$DATA_DIR/robocasa_ref/ ./robocasa_ref/"
