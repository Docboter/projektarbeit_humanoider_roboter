#!/usr/bin/env bash
# TL;DR: Läuft im Container — Co-Training auf echten UND gerenderten Bildern (Schritt 4, --tune_visual).
# run_finetuning_cotrain.sh — Co-Training auf ECHTEN + GERENDERTEN Bildern (Schritt 4).
#
# Verhältnis zu den anderen beiden Trainings-Skripten:
#   run_finetuning.sh          Vision-Encoder eingefroren, ein Datensatz
#   run_finetuning_vision.sh   --tune_visual, ein Datensatz            ← Lauf 3
#   run_finetuning_cotrain.sh  --tune_visual, ZWEI Datensätze          ← dieses hier
#
# Begründung (Stand 2026-08-14): Lauf 34 hat gezeigt, dass der aufgetaute Encoder die
# Fingerspanne in der Sim von 20,5 % auf 27,6 % der Demonstration hebt — vollständige
# Trennung gegen den alten Checkpoint, p = 3,3e-4 — und dass das trotzdem nicht reicht
# (`lifted` 0/10, auf ECHTEN Bildern liefert dieselbe Politik 100 %). Der Hebel wirkt,
# das Material fehlt: der Encoder hat nie ein Sim-Bild gesehen. Genau das liefert
# `Simulation/g1_dex3_sim/render_cotrain_dataset.py`.
#
# Der Mischbetrieb selbst steckt schon im Fork (SingleDatasetConfig.mix_ratio); nur der
# CLI-Einstieg kennt ihn nicht. Deshalb ruft dieses Skript /scripts/launch_cotrain.py
# statt gr00t/experiment/launch_finetune.py auf — kein Submodul-Eingriff, kein Rebuild.
#
# Verwendung (im Container):
#   COTRAIN_DATASET_PATH=/data/cotrain/g1_dex3_rendered bash /scripts/run_finetuning_cotrain.sh
#   COTRAIN_MIX_RATIO=0.25 bash /scripts/run_finetuning_cotrain.sh --dry-run
#
# Logs landen unter /data/logs/.

set -euo pipefail

# ── Trainings-Parameter (alle via Env-Var überschreibbar) ─────────────────────
GROOT_ROOT="${GROOT_ROOT:-/app/Groot-1.6}"
MODEL_PATH="${MODEL_PATH:-/data/models/GR00T-N1.6-3B}"
DATASET_PATH="${DATASET_PATH:-/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset}"
# Eigener Output-/Experiment-Namespace — überschreibt weder die Standard- noch die
# Vision-Läufe. Deren Checkpoints bleiben zum Vergleich erhalten.
OUTPUT_DIR="${OUTPUT_DIR:-/data/g1_dex3_finetune/blockstacking_cotrain}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_dex3_blockstacking_cotrain_v1}"
MODALITY_CONFIG="${MODALITY_CONFIG:-$GROOT_ROOT/examples/G1_DEX3/g1_dex3_config.py}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"

# ── Der gerenderte Datensatz ──────────────────────────────────────────────────
# Entsteht auf dem Sim-Server (RT-Core-GPU), trainiert wird woanders — deshalb der
# optionale HF-Zwischenschritt. Größenordnung: ~28 MB je Episode (4 Kameras, h264),
# also ~1,7 GB für 60 Episoden. Das passt bequem in ein HF-Dataset-Repo.
COTRAIN_DATASET_PATH="${COTRAIN_DATASET_PATH:-/data/cotrain/g1_dex3_rendered}"
COTRAIN_HF_REPO="${COTRAIN_HF_REPO:-}"
# Anteil der GERENDERTEN Stichproben. 0.5 = jede zweite. Siehe docs/training/co-training.md
# zur Begründung des Startwerts und zur Abbruchregel.
COTRAIN_MIX_RATIO="${COTRAIN_MIX_RATIO:-0.5}"

MAX_STEPS="${MAX_STEPS:-30000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"

# ── Train-Test-Split ──────────────────────────────────────────────────────────
# Hier NICHT optional gedacht: die Abbruchregel des Co-Trainings vergleicht die
# Validierungs-MSE auf zurückgehaltenen ECHTEN Episoden gegen Lauf 3 (0,00716). Ohne
# Split gibt es diese Zahl nicht, und ein Rückschritt auf der Realdomäne bliebe unsichtbar.
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-1}"
TRAIN_SPLIT_RATIO="${TRAIN_SPLIT_RATIO:-0.8}"

# ── Bild-Augmentierung / Domain Randomization ─────────────────────────────────
USE_AUGMENTATION="${USE_AUGMENTATION:-1}"
CJ_BRIGHTNESS="${CJ_BRIGHTNESS:-0.3}"
CJ_CONTRAST="${CJ_CONTRAST:-0.4}"
CJ_SATURATION="${CJ_SATURATION:-0.5}"
CJ_HUE="${CJ_HUE:-0.08}"
RANDOM_ROTATION_ANGLE="${RANDOM_ROTATION_ANGLE:-}"
STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.0}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-gr00t-g1-dex3}"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ── Helfer ────────────────────────────────────────────────────────────────────
log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m!! \033[0m%s\n' "$*" >&2; }
warn() { printf '\033[1;33m ! \033[0m%s\n' "$*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_split.sh
source "$SCRIPT_DIR/lib_split.sh"
# shellcheck source=lib_resume_guard.sh
source "$SCRIPT_DIR/lib_resume_guard.sh"

# ── 1. Laufumgebung verifizieren ──────────────────────────────────────────────
if [[ ! -f /.dockerenv ]] && [[ -z "${APPTAINER_NAME:-}" ]] && [[ -z "${SINGULARITY_NAME:-}" ]]; then
    err "Dieses Skript muss im Container laufen. Starte z. B.:"
    err "    docker compose exec groot-training bash /scripts/run_finetuning_cotrain.sh"
    err "    apptainer run --nv projekt-humanoider-roboter.sif"
    exit 1
fi

[[ -d "$GROOT_ROOT" ]] || { err "GROOT_ROOT existiert nicht: $GROOT_ROOT"; exit 1; }

# ── 2. GPU-Sichtbarkeit prüfen ────────────────────────────────────────────────
log "Prüfe GPU-Sichtbarkeit …"
nvidia-smi -L || { err "GPU nicht sichtbar — Container ohne --gpus all gestartet?"; exit 1; }

# ── 3. Modell und Datensätze prüfen ───────────────────────────────────────────
log "Prüfe Modell und Datensätze …"
[[ -d "$MODEL_PATH"   ]] || { err "FEHLT: $MODEL_PATH (siehe scripts/download_data.sh)";   exit 1; }
[[ -d "$DATASET_PATH" ]] || { err "FEHLT: $DATASET_PATH (siehe scripts/download_data.sh)"; exit 1; }

if [[ ! -f "$DATASET_PATH/meta/modality.json" ]]; then
    log "modality.json fehlt — kopiere aus examples/G1_DEX3/modality_4cam.json"
    cp "$GROOT_ROOT/examples/G1_DEX3/modality_4cam.json" "$DATASET_PATH/meta/modality.json"
fi

# Gerenderten Datensatz beschaffen, falls nur als HF-Repo vorhanden.
if [[ ! -f "$COTRAIN_DATASET_PATH/meta/info.json" && -n "$COTRAIN_HF_REPO" ]]; then
    log "Lade gerenderten Datensatz von HuggingFace: $COTRAIN_HF_REPO"
    cd "$GROOT_ROOT"
    uv run --no-sync python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${COTRAIN_HF_REPO}", repo_type="dataset",
                  local_dir="${COTRAIN_DATASET_PATH}")
PY
fi

if [[ ! -f "$COTRAIN_DATASET_PATH/meta/info.json" ]]; then
    err "Gerenderter Datensatz fehlt: $COTRAIN_DATASET_PATH/meta/info.json"
    err "  Erzeugen (auf dem Sim-Server, RT-Core-GPU):"
    err "      ./Simulation/server_rl_run.sh render"
    err "  Oder COTRAIN_HF_REPO=<user>/<repo> setzen, damit er hier geladen wird."
    err "  Anleitung: docs/training/co-training.md"
    exit 1
fi

# modality.json des gerenderten Datensatzes muss dieselbe Aufteilung tragen wie der echte —
# sonst zerfallen dieselben 28 Dimensionen in zwei verschiedene Gruppierungen und das
# Modell bekommt stumm inkonsistente Eingaben.
if ! cmp -s "$DATASET_PATH/meta/modality.json" "$COTRAIN_DATASET_PATH/meta/modality.json"; then
    err "modality.json der beiden Datensätze unterscheidet sich:"
    err "    echt:      $DATASET_PATH/meta/modality.json"
    err "    gerendert: $COTRAIN_DATASET_PATH/meta/modality.json"
    err "Der Renderer kopiert sie eigentlich 1:1 — hier stimmt etwas nicht. Abbruch."
    exit 1
fi

RENDERED_EPS=$(python - "$COTRAIN_DATASET_PATH/meta/info.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("total_episodes", "?"))
PY
)
log "Gerenderter Datensatz: $COTRAIN_DATASET_PATH ($RENDERED_EPS Episoden)"
log "Mischungsverhältnis: $COTRAIN_MIX_RATIO gerendert / $(python -c "print(round(1-$COTRAIN_MIX_RATIO,3))") echt"

mkdir -p "$OUTPUT_DIR"

# ── 3a. Gegen stilles Fortsetzen eines alten Laufs absichern ──────────────────
resume_guard "$OUTPUT_DIR" "$EXPERIMENT_NAME" "$MAX_STEPS" || exit 1

# ── 3b. Train-Test-Split scharf schalten ──────────────────────────────────────
# NUR auf dem echten Datensatz: der gerenderte enthält per Konstruktion ausschließlich
# Trainings-Episoden (render_cotrain_dataset.py weigert sich, Test-Episoden zu rendern).
INFO_JSON="$DATASET_PATH/meta/info.json"
split_apply "$INFO_JSON" "$TRAIN_TEST_SPLIT" "$TRAIN_SPLIT_RATIO" "$OUTPUT_DIR"

# ── 4. W&B-Login ──────────────────────────────────────────────────────────────
if [[ "$USE_WANDB" == "1" ]]; then
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        err "USE_WANDB=1, aber WANDB_API_KEY ist nicht gesetzt."
        err "Setze WANDB_API_KEY per Env-Var oder USE_WANDB=0 für Training ohne W&B."
        exit 1
    fi
    if [[ "${WANDB_MODE:-}" == "offline" ]]; then
        log "W&B Offline-Modus — Login übersprungen (kein Internet auf Compute-Node)."
    else
        cd "$GROOT_ROOT"
        if uv run --no-sync wandb login --relogin "$WANDB_API_KEY" &>/dev/null; then
            log "W&B-Login OK"
        else
            err "W&B-Login fehlgeschlagen — API-Key ungültig?"
            exit 1
        fi
    fi
fi

# ── 5. Trainings-Befehl bauen ─────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-1}"
if [[ "$NUM_GPUS" -gt 1 ]]; then
    LAUNCHER=(torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS")
    log "Multi-GPU-Modus: torchrun mit $NUM_GPUS Prozessen (DeepSpeed ZeRO-2, per_device = $GLOBAL_BATCH_SIZE / $NUM_GPUS)."
else
    LAUNCHER=(python)
fi

TRAIN_CMD=(
    uv run --no-sync "${LAUNCHER[@]}" "$SCRIPT_DIR/launch_cotrain.py"
    --cotrain_dataset_path   "$COTRAIN_DATASET_PATH"
    --cotrain_mix_ratio      "$COTRAIN_MIX_RATIO"
    --base_model_path        "$MODEL_PATH"
    --dataset_path           "$DATASET_PATH"
    --embodiment_tag         "$EMBODIMENT_TAG"
    --modality_config_path   "$MODALITY_CONFIG"
    --output_dir             "$OUTPUT_DIR"
    --experiment_name        "$EXPERIMENT_NAME"
    --num_gpus               "$NUM_GPUS"
    --max_steps              "$MAX_STEPS"
    --save_steps             "$SAVE_STEPS"
    --save_total_limit       "$SAVE_TOTAL_LIMIT"
    --global_batch_size      "$GLOBAL_BATCH_SIZE"
    --learning_rate          "$LEARNING_RATE"
    --weight_decay           "$WEIGHT_DECAY"
    --warmup_ratio           "$WARMUP_RATIO"
    --dataloader_num_workers "$DATALOADER_WORKERS"
    --tune_visual            # ohne den Encoder wäre das gerenderte Material wirkungslos
)

if [[ "$USE_AUGMENTATION" == "1" ]]; then
    log "Augmentierung AN — Color-Jitter (b=$CJ_BRIGHTNESS c=$CJ_CONTRAST s=$CJ_SATURATION h=$CJ_HUE)"
    TRAIN_CMD+=(--color_jitter_params brightness "$CJ_BRIGHTNESS" contrast "$CJ_CONTRAST" saturation "$CJ_SATURATION" hue "$CJ_HUE")
    [[ -n "$RANDOM_ROTATION_ANGLE" ]] && TRAIN_CMD+=(--random_rotation_angle "$RANDOM_ROTATION_ANGLE")
    if [[ "$STATE_DROPOUT_PROB" != "0.0" && "$STATE_DROPOUT_PROB" != "0" ]]; then
        TRAIN_CMD+=(--state_dropout_prob "$STATE_DROPOUT_PROB")
    fi
else
    warn "Augmentierung AUS. Beim Co-Training weniger kritisch als bei Lauf 2 (die"
    warn "Sim-Bilder liefern die Varianz jetzt selbst), aber es gibt keinen Grund dafür."
    TRAIN_CMD+=(--color_jitter_params brightness 0 contrast 0 saturation 0 hue 0)
fi

[[ "$USE_WANDB" == "1" ]] && TRAIN_CMD+=(--use_wandb --wandb_project "$WANDB_PROJECT")

log "Co-Training AKTIV: echt + gerendert, Vision-Encoder auf (--tune_visual), LLM eingefroren."
log "Trainings-Befehl:"
printf '    %s\n' "${TRAIN_CMD[*]}"

if [[ "$DRY_RUN" == "1" ]]; then
    log "Dry-Run — nicht ausgeführt."
    exit 0
fi

# ── 6. Training ausführen ─────────────────────────────────────────────────────
LOG_DIR="/data/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/finetune-cotrain-$(date +%Y%m%d-%H%M%S).log"
log "Logs: $LOG_FILE"
log "Training startet — abbrechen mit Ctrl+C."
echo

cd "$GROOT_ROOT"
"${TRAIN_CMD[@]}" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

echo
if [[ "$EXIT_CODE" == "0" ]]; then
    log "Training erfolgreich beendet."
    LAST_CKPT=$(find "$OUTPUT_DIR" -maxdepth 3 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -1 || true)
    [[ -n "$LAST_CKPT" ]] && log "Letzter Checkpoint: $LAST_CKPT"
    log "Checkpoint-Auswahl (NICHT blind den letzten nehmen — Lauf 3: der letzte war 25 % schlechter):"
    log "    python /scripts/checkpoint_sweep.py --run-dir $OUTPUT_DIR --dataset-path $DATASET_PATH"
    log "Danach das Gate auf der Sim-Seite (auf dem Sim-Server):"
    log "    NUM_EPISODES=10 EPISODE_LENGTH_S=40 ./Simulation/server_rl_run.sh eval"
else
    err "Training mit Exit-Code $EXIT_CODE beendet — siehe $LOG_FILE"
fi
exit "$EXIT_CODE"
