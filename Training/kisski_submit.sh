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
#   3. Tokens EINMALIG in Dateien hinterlegen (mode 600):
#        printf 'hf_DEIN_TOKEN\n'    > ~/.hf_token  && chmod 600 ~/.hf_token
#        printf 'DEIN_WANDB_KEY\n'   > ~/.wandb_key && chmod 600 ~/.wandb_key
#      Das Skript liest beide automatisch ein (siehe Block "Tokens aus Dateien").
#      Danach genügt zum Einreichen:
#        sbatch kisski_submit.sh
#
#      Grund für den Datei-Weg: KISSKI setzt SBATCH_EXPORT=none auf dem
#      Login-Node. Das überstimmt das #SBATCH --export=ALL, sodass inline vor
#      sbatch übergebene Variablen (HF_TOKEN=... sbatch ...) NICHT im Job
#      ankommen. Willst du Variablen doch über die Umgebung übergeben, MUSS
#      --export=ALL auf der Kommandozeile stehen (schlägt die Env-Variable):
#        HF_TOKEN=hf_... WANDB_API_KEY=... sbatch --export=ALL kisski_submit.sh
#      (W&B ist optional; ohne Key/Datei wird ohne W&B-Logging trainiert.)
#
# Das Skript klont automatisch das GitHub-Repo und lädt Modell + Datensatz
# von HuggingFace herunter — kein manuelles rsync nötig.
#
# Optionale Überschreibungen — ebenfalls als Inline-Prefix vor sbatch:
#   GITHUB_REPO, GITHUB_BRANCH, GITHUB_TOKEN, REPO_DIR
#   MAX_STEPS, SAVE_STEPS, SAVE_TOTAL_LIMIT, GLOBAL_BATCH_SIZE, NUM_GPUS,
#   WANDB_PROJECT, DATA_DIR, SKIP_GIT_PULL, SKIP_DOWNLOAD, SKIP_CONVERT, SKIP_TRAIN,
#   TUNE_VISUAL  (=1 → Vision-Encoder mittrainieren)
#   USE_COTRAIN  (=1 → Co-Training echt + gerendert; COTRAIN_HF_REPO / COTRAIN_MIX_RATIO)
#
# Vision-Encoder-Training auf KISSKI einreichen:
#   TUNE_VISUAL=1 sbatch --export=ALL kisski_submit.sh
#   (oder im Skript oben TUNE_VISUAL-Default auf 1 setzen)
#
# Co-Training auf echten UND gerenderten Bildern (Schritt 4, docs/training/co-training.md):
#   USE_COTRAIN=1 COTRAIN_HF_REPO=<user>/<repo> COTRAIN_MIX_RATIO=0.25 \
#       sbatch --export=ALL kisski_submit.sh
#   Der gerenderte Datensatz entsteht vorher auf dem Sim-Server:
#       ./Simulation/server_rl_run.sh render

#SBATCH --job-name=groot-finetune
#SBATCH -p kisski
#SBATCH -G A100:4
#SBATCH -c 96
#SBATCH --mem=384G
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

# 4× A100 (DeepSpeed ZeRO-2). global_batch_size MUSS durch NUM_GPUS teilbar sein
# (Assertion in experiment.py: global_batch_size % num_gpus == 0) → per_device = 32/4 = 8.
# ~5 Epochen: 281k Frames / Batch 32 ≈ 8,8k Schritte/Epoche → 5 Epochen ≈ 44k Schritte
# (1-GPU-Referenz war 175k bei Batch 8; durch den 4× größeren Batch sinkt die Schrittzahl auf 1/4).
MAX_STEPS="${MAX_STEPS:-44000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
NUM_GPUS="${NUM_GPUS:-4}"
# Dataloader-Worker PRO RANK. Bei 4 GPUs ergeben 4 Worker × 4 Ranks = 16 Prozesse, die
# Shards in RAM cachen. 8/Rank (= 32 Prozesse) sprengten den Host-RAM → OOM-Kill der Worker
# (SIGKILL) → DataLoader-Abbruch. 4/Rank füttern eine A100 locker und halbieren den RAM-Druck.
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
# TUNE_VISUAL=1 → Vision-Encoder mittrainieren (Entrypoint startet run_finetuning_vision.sh,
# eigener Output-Namespace /data/g1_dex3_finetune/blockstacking_vision). Default 0 = Standardlauf.
# Steht hier oben, weil LR + Warmup davon abhängen.
TUNE_VISUAL="${TUNE_VISUAL:-0}"

# USE_COTRAIN=1 → Co-Training auf echten UND gerenderten Bildern (Schritt 4; Entrypoint
# startet run_finetuning_cotrain.sh, Namespace /data/g1_dex3_finetune/blockstacking_cotrain).
# Setzt --tune_visual selbst, also dieselbe LR-Behandlung wie ein Vision-Lauf.
# Anleitung: docs/training/co-training.md
USE_COTRAIN="${USE_COTRAIN:-0}"
COTRAIN_DATASET_PATH="${COTRAIN_DATASET_PATH:-/data/cotrain/g1_dex3_rendered}"
COTRAIN_HF_REPO="${COTRAIN_HF_REPO:-}"
COTRAIN_MIX_RATIO="${COTRAIN_MIX_RATIO:-0.25}"

# Lernrate + Warmup hängen davon ab, ob der Vision-Encoder mittrainiert wird:
#  • Standardlauf (nur Projector + Diffusion): LR sqrt-skaliert für den 4× größeren
#    effektiven Batch (1e-4 × √4 = 2e-4), Warmup 0.05 — wie bisher, unverändert.
#  • Vision-Lauf: der große vortrainierte Eagle-ViT teilt sich dieselbe globale LR;
#    2e-4 würde die Visual-Features destabilisieren → konservativ 1e-4 + längeres
#    Warmup 0.1. Beide Werte jederzeit per Env-Var überschreibbar.
if [[ "$TUNE_VISUAL" == "1" || "$USE_COTRAIN" == "1" ]]; then
    LEARNING_RATE="${LEARNING_RATE:-1e-4}"
    WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
else
    LEARNING_RATE="${LEARNING_RATE:-2e-4}"
    WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
fi
WANDB_PROJECT="${WANDB_PROJECT:-gr00t-g1-dex3}"

# Checkpoints: alle 5000 Schritte (~52 min) ein Checkpoint, Limit hoch genug,
# dass NICHTS gelöscht wird → 35 Checkpoints gleichmäßig über den ganzen Lauf
# verteilt, sodass sich der Trainingsfortschritt im Nachhinein rekonstruieren lässt.
# Achtung: jeder Checkpoint ≈ 22 GB (13 GB optimizer.pt + 9 GB Gewichte) → ~770 GB gesamt.
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-40}"

# Train-Test-Split + Augmentierung: müssen hier definiert und unten explizit an den
# Container durchgereicht werden — Apptainer erbt die Job-Umgebung NICHT automatisch.
# Ohne den Durchgriff bliebe ein `TRAIN_TEST_SPLIT=1 sbatch …` wirkungslos, es gäbe
# keine zurückgehaltenen Episoden und die Checkpoint-Auswahl bliebe blind.
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-0}"
TRAIN_SPLIT_RATIO="${TRAIN_SPLIT_RATIO:-0.8}"

# Namespace des Laufs. Leer = Default des jeweiligen Trainings-Skripts.
# WICHTIG: Der Fork ruft trainer.train(resume_from_checkpoint=True) fest verdrahtet auf —
# ein Lauf in ein bereits belegtes OUTPUT_DIR/EXPERIMENT_NAME SETZT FORT statt neu zu
# trainieren. Für einen neuen Lauf also einen neuen Namen vergeben. RESUME=1 erlaubt das
# Fortsetzen bewusst (z. B. nach Walltime-Abbruch).
OUTPUT_DIR="${OUTPUT_DIR:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
RESUME="${RESUME:-0}"
USE_AUGMENTATION="${USE_AUGMENTATION:-1}"
CJ_BRIGHTNESS="${CJ_BRIGHTNESS:-0.3}"
CJ_CONTRAST="${CJ_CONTRAST:-0.4}"
CJ_SATURATION="${CJ_SATURATION:-0.5}"
CJ_HUE="${CJ_HUE:-0.08}"
RANDOM_ROTATION_ANGLE="${RANDOM_ROTATION_ANGLE:-}"
STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.0}"

SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-1}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

# ── Tokens aus Dateien einlesen ───────────────────────────────────────────────
# KISSKI setzt auf dem Login-Node SBATCH_EXPORT=none. Das überstimmt das
# #SBATCH --export=ALL oben, sodass inline vor sbatch übergebene Variablen NICHT
# im Job ankommen. Deshalb Tokens robust aus Dateien lesen (mode 600). Eine
# bereits gesetzte Umgebungsvariable hat Vorrang (z. B. sbatch --export=ALL,...).
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.hf_token}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$HOME/.wandb_key}"

if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
    if [[ -n "$HF_TOKEN" && "$HF_TOKEN" != REPLACE_* ]]; then
        export HF_TOKEN
        echo "    HF_TOKEN aus $HF_TOKEN_FILE gelesen."
    else
        unset HF_TOKEN   # Platzhalter/leer -> als nicht gesetzt behandeln
    fi
fi

if [[ -z "${WANDB_API_KEY:-}" && -f "$WANDB_KEY_FILE" ]]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_KEY_FILE")"
    if [[ -n "$WANDB_API_KEY" && "$WANDB_API_KEY" != REPLACE_* ]]; then
        export WANDB_API_KEY
        echo "    WANDB_API_KEY aus $WANDB_KEY_FILE gelesen."
    else
        unset WANDB_API_KEY   # Platzhalter/leer -> als nicht gesetzt behandeln
    fi
fi

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
echo "    GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE  (per_device = $((GLOBAL_BATCH_SIZE / NUM_GPUS)))"
echo "    NUM_GPUS:          $NUM_GPUS"
echo "    DATALOADER_WORKERS: $DATALOADER_WORKERS  (× $NUM_GPUS Ranks = $((DATALOADER_WORKERS * NUM_GPUS)) Prozesse)"
echo "    LEARNING_RATE:     $LEARNING_RATE"
echo "    WARMUP_RATIO:      $WARMUP_RATIO"
echo "    WANDB_PROJECT:     $WANDB_PROJECT"
echo "    TUNE_VISUAL:       $TUNE_VISUAL"
echo "    USE_COTRAIN:       $USE_COTRAIN$([[ "$USE_COTRAIN" == "1" ]] && echo "  (Mischung ${COTRAIN_MIX_RATIO} gerendert)")"
echo "    TRAIN_TEST_SPLIT:  $TRAIN_TEST_SPLIT  (Ratio=$TRAIN_SPLIT_RATIO)"
echo "    USE_AUGMENTATION:  $USE_AUGMENTATION"
echo "    OUTPUT_DIR:        ${OUTPUT_DIR:-<Default des Trainings-Skripts>}"
echo "    EXPERIMENT_NAME:   ${EXPERIMENT_NAME:-<Default des Trainings-Skripts>}"
echo "    RESUME:            $RESUME"
if [[ "$TRAIN_TEST_SPLIT" != "1" ]]; then
    echo "    ! Ohne Split gibt es hinterher nichts zu validieren — die Checkpoint-Auswahl"
    echo "      bleibt blind (so lief es in Lauf 1 und Lauf 2)."
fi
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
    --env "DATALOADER_WORKERS=$DATALOADER_WORKERS"
    --env "LEARNING_RATE=$LEARNING_RATE"
    --env "WARMUP_RATIO=$WARMUP_RATIO"
    --env "WANDB_PROJECT=$WANDB_PROJECT"
    --env "SKIP_DOWNLOAD=$SKIP_DOWNLOAD"
    --env "SKIP_CONVERT=$SKIP_CONVERT"
    --env "SKIP_TRAIN=$SKIP_TRAIN"
    --env "TUNE_VISUAL=$TUNE_VISUAL"
    --env "USE_COTRAIN=$USE_COTRAIN"
    --env "COTRAIN_DATASET_PATH=$COTRAIN_DATASET_PATH"
    --env "COTRAIN_HF_REPO=$COTRAIN_HF_REPO"
    --env "COTRAIN_MIX_RATIO=$COTRAIN_MIX_RATIO"
    --env "TRAIN_TEST_SPLIT=$TRAIN_TEST_SPLIT"
    --env "TRAIN_SPLIT_RATIO=$TRAIN_SPLIT_RATIO"
    --env "USE_AUGMENTATION=$USE_AUGMENTATION"
    --env "CJ_BRIGHTNESS=$CJ_BRIGHTNESS"
    --env "CJ_CONTRAST=$CJ_CONTRAST"
    --env "CJ_SATURATION=$CJ_SATURATION"
    --env "CJ_HUE=$CJ_HUE"
    --env "RANDOM_ROTATION_ANGLE=$RANDOM_ROTATION_ANGLE"
    --env "STATE_DROPOUT_PROB=$STATE_DROPOUT_PROB"
    --env "RESUME=$RESUME"
)
# Nur setzen, wenn angegeben — sonst greifen die Defaults der Trainings-Skripte.
[[ -n "$OUTPUT_DIR"      ]] && APPTAINER_ARGS+=(--env "OUTPUT_DIR=$OUTPUT_DIR")
[[ -n "$EXPERIMENT_NAME" ]] && APPTAINER_ARGS+=(--env "EXPERIMENT_NAME=$EXPERIMENT_NAME")

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
