#!/usr/bin/env bash
# Startet das GR00T-Finetuning *innerhalb des Containers*.
#
# Verwendung (im Container):
#   bash /scripts/run_finetuning.sh                      # Defaults
#   MAX_STEPS=10000 bash /scripts/run_finetuning.sh      # einzelne Werte overriden
#   bash /scripts/run_finetuning.sh --dry-run            # nur Befehl anzeigen
#
# Vom Host aus aufrufen:
#   docker compose exec groot-training bash /scripts/run_finetuning.sh
#
# Logs landen unter /data/logs/ — und damit über den Volume-Mount auch auf dem Host.

set -euo pipefail

# ── Trainings-Parameter (alle via Env-Var überschreibbar) ─────────────────────
GROOT_ROOT="${GROOT_ROOT:-/app/Groot-1.6}"
MODEL_PATH="${MODEL_PATH:-/data/models/GR00T-N1.6-3B}"
DATASET_PATH="${DATASET_PATH:-/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/g1_dex3_finetune/blockstacking}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_dex3_blockstacking_v1}"
MODALITY_CONFIG="${MODALITY_CONFIG:-$GROOT_ROOT/examples/G1_DEX3/g1_dex3_config.py}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"

MAX_STEPS="${MAX_STEPS:-30000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}" #8
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}" #2
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-gr00t-g1-dex3}"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ── Helfer ────────────────────────────────────────────────────────────────────
log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m!! \033[0m%s\n' "$*" >&2; }

# ── 1. Laufumgebung verifizieren ──────────────────────────────────────────────
if [[ ! -f /.dockerenv ]] && [[ -z "${APPTAINER_NAME:-}" ]] && [[ -z "${SINGULARITY_NAME:-}" ]]; then
    err "Dieses Skript muss im Container laufen. Starte z. B.:"
    err "    docker compose exec groot-training bash /scripts/run_finetuning.sh"
    err "    apptainer run --nv projekt-humanoider-roboter.sif"
    exit 1
fi

[[ -d "$GROOT_ROOT" ]] || { err "GROOT_ROOT existiert nicht: $GROOT_ROOT"; exit 1; }

# ── 2. GPU-Sichtbarkeit prüfen ────────────────────────────────────────────────
log "Prüfe GPU-Sichtbarkeit …"
nvidia-smi -L || { err "GPU nicht sichtbar — Container ohne --gpus all gestartet?"; exit 1; }

# ── 3. Modell und Datensatz prüfen ────────────────────────────────────────────
log "Prüfe Modell und Datensatz …"
[[ -d "$MODEL_PATH"   ]] || { err "FEHLT: $MODEL_PATH (siehe scripts/download_data.sh)";   exit 1; }
[[ -d "$DATASET_PATH" ]] || { err "FEHLT: $DATASET_PATH (siehe scripts/download_data.sh)"; exit 1; }

if [[ ! -f "$DATASET_PATH/meta/modality.json" ]]; then
    log "modality.json fehlt — kopiere aus examples/G1_DEX3/modality_4cam.json"
    cp "$GROOT_ROOT/examples/G1_DEX3/modality_4cam.json" "$DATASET_PATH/meta/modality.json"
fi

mkdir -p "$OUTPUT_DIR"

# ── 4. W&B-Login (nur via WANDB_API_KEY-Env-Var, kein interaktiver Fallback) ──
if [[ "$USE_WANDB" == "1" ]]; then
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        err "USE_WANDB=1, aber WANDB_API_KEY ist nicht gesetzt."
        err "Setze WANDB_API_KEY per Env-Var oder USE_WANDB=0 für Training ohne W&B."
        exit 1
    fi
    cd "$GROOT_ROOT"
    # wandb liest WANDB_API_KEY automatisch — kein expliziter Login nötig.
    if uv run --no-sync wandb login --relogin "$WANDB_API_KEY" &>/dev/null; then
        log "W&B-Login OK"
    else
        err "W&B-Login fehlgeschlagen — API-Key ungültig?"
        exit 1
    fi
fi

# ── 5. Trainings-Befehl bauen ─────────────────────────────────────────────────
TRAIN_CMD=(
    uv run --no-sync python "$GROOT_ROOT/gr00t/experiment/launch_finetune.py"
    --base_model_path        "$MODEL_PATH"
    --dataset_path           "$DATASET_PATH"
    --embodiment_tag         "$EMBODIMENT_TAG"
    --modality_config_path   "$MODALITY_CONFIG"
    --output_dir             "$OUTPUT_DIR"
    --experiment_name        "$EXPERIMENT_NAME"
    --num_gpus               "${NUM_GPUS:-1}"
    --max_steps              "$MAX_STEPS"
    --save_steps             "$SAVE_STEPS"
    --save_total_limit       "$SAVE_TOTAL_LIMIT"
    --global_batch_size      "$GLOBAL_BATCH_SIZE"
    --learning_rate          "$LEARNING_RATE"
    --weight_decay           "$WEIGHT_DECAY"
    --warmup_ratio           "$WARMUP_RATIO"
    --dataloader_num_workers "$DATALOADER_WORKERS"
    --color_jitter_params    brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
)
[[ "$USE_WANDB" == "1" ]] && TRAIN_CMD+=(--use_wandb --wandb_project "$WANDB_PROJECT")

log "Trainings-Befehl:"
printf '    %s\n' "${TRAIN_CMD[*]}"

if [[ "$DRY_RUN" == "1" ]]; then
    log "Dry-Run — nicht ausgeführt."
    exit 0
fi

# ── 6. Training ausführen, Logs nach /data/logs (= Host-sichtbar) ─────────────
LOG_DIR="/data/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/finetune-$(date +%Y%m%d-%H%M%S).log"
log "Logs: $LOG_FILE  (auf dem Host: ./data/logs/$(basename "$LOG_FILE"))"
log "Training startet — abbrechen mit Ctrl+C."
echo

cd "$GROOT_ROOT"
"${TRAIN_CMD[@]}" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

echo
if [[ "$EXIT_CODE" == "0" ]]; then
    log "Training erfolgreich beendet."
    LAST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    [[ -n "$LAST_CKPT" ]] && log "Letzter Checkpoint: $LAST_CKPT"
else
    err "Training mit Exit-Code $EXIT_CODE beendet — siehe $LOG_FILE"
fi
exit "$EXIT_CODE"
