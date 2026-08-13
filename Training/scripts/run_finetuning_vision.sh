#!/usr/bin/env bash
# run_finetuning_vision.sh — Variante von run_finetuning.sh, die ZUSÄTZLICH den
# Vision-Encoder (Eagle-Backbone) mittrainiert (--tune_visual).
#
# Unterschied zum Standard-Skript (run_finetuning.sh):
#   • --tune_visual            → Visual-Backbone wird NICHT eingefroren
#   • eigener OUTPUT_DIR + EXPERIMENT_NAME (kein Überschreiben der Standard-Läufe)
#   Alles Übrige (LLM eingefroren, Projector + Diffusion an) bleibt identisch.
#
# Das Standard-Skript run_finetuning.sh bleibt davon vollständig unberührt.
#
# Verwendung (im Container):
#   bash /scripts/run_finetuning_vision.sh                  # Defaults
#   MAX_STEPS=10000 bash /scripts/run_finetuning_vision.sh  # einzelne Werte overriden
#   bash /scripts/run_finetuning_vision.sh --dry-run        # nur Befehl anzeigen
#
# ⚠️  VRAM: Das Mittrainieren des Vision-Encoders erhöht den Speicherbedarf
#     spürbar. Auf <40 GB ggf. GLOBAL_BATCH_SIZE reduzieren. NVIDIA empfiehlt für
#     volles Visual-Tuning ≥40 GB VRAM (siehe FINETUNING_GUIDE.md).
#
# Logs landen unter /data/logs/.

set -euo pipefail

# ── Trainings-Parameter (alle via Env-Var überschreibbar) ─────────────────────
GROOT_ROOT="${GROOT_ROOT:-/app/Groot-1.6}"
MODEL_PATH="${MODEL_PATH:-/data/models/GR00T-N1.6-3B}"
DATASET_PATH="${DATASET_PATH:-/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset}"
# Eigener Output-/Experiment-Namespace, damit Vision-Läufe nicht die Standard-Läufe überschreiben.
OUTPUT_DIR="${OUTPUT_DIR:-/data/g1_dex3_finetune/blockstacking_vision}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_dex3_blockstacking_vision_v1}"
MODALITY_CONFIG="${MODALITY_CONFIG:-$GROOT_ROOT/examples/G1_DEX3/g1_dex3_config.py}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"

MAX_STEPS="${MAX_STEPS:-30000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}" #8
DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"

# ── Train-Test-Split (optional, Standard AUS) ─────────────────────────────────
# Ohne Split gibt es keine zurückgehaltenen Episoden — und damit hinterher keine
# Validierungs-Zahl für die Checkpoint-Auswahl. Genau das war die strukturelle
# Lücke von Lauf 1 und Lauf 2 (beide haben blind den letzten Step genommen).
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-0}"
TRAIN_SPLIT_RATIO="${TRAIN_SPLIT_RATIO:-0.8}"

# ── Bild-Augmentierung / Domain Randomization (optional, Standard AN) ──────────
# Identisch zu run_finetuning.sh. Vorher war der Color-Jitter hier fest verdrahtet,
# obwohl die Doku USE_AUGMENTATION als Schalter auswies — für den Vision-Lauf ist
# genau dieser Schalter der entscheidende Unterschied zu Lauf 2 (der lief noch ganz
# ohne Jitter; USE_AUGMENTATION kam erst danach dazu).
USE_AUGMENTATION="${USE_AUGMENTATION:-1}"
CJ_BRIGHTNESS="${CJ_BRIGHTNESS:-0.3}"
CJ_CONTRAST="${CJ_CONTRAST:-0.4}"
CJ_SATURATION="${CJ_SATURATION:-0.5}"
CJ_HUE="${CJ_HUE:-0.08}"
RANDOM_ROTATION_ANGLE="${RANDOM_ROTATION_ANGLE:-}"   # leer = keine Rotation
STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.0}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-gr00t-g1-dex3}"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ── Helfer ────────────────────────────────────────────────────────────────────
log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m!! \033[0m%s\n' "$*" >&2; }
warn() { printf '\033[1;33m ! \033[0m%s\n' "$*"; }

# Split-Logik teilen sich beide Trainings-Skripte.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_split.sh
source "$SCRIPT_DIR/lib_split.sh"

# ── 1. Laufumgebung verifizieren ──────────────────────────────────────────────
if [[ ! -f /.dockerenv ]] && [[ -z "${APPTAINER_NAME:-}" ]] && [[ -z "${SINGULARITY_NAME:-}" ]]; then
    err "Dieses Skript muss im Container laufen. Starte z. B.:"
    err "    docker compose exec groot-training bash /scripts/run_finetuning_vision.sh"
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

# ── 3b. Train-Test-Split scharf schalten (optional) ───────────────────────────
# Details und Begründung in lib_split.sh.
INFO_JSON="$DATASET_PATH/meta/info.json"
split_apply "$INFO_JSON" "$TRAIN_TEST_SPLIT" "$TRAIN_SPLIT_RATIO" "$OUTPUT_DIR"

# ── 4. W&B-Login (nur via WANDB_API_KEY-Env-Var, kein interaktiver Fallback) ──
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
        # wandb liest WANDB_API_KEY automatisch — kein expliziter Login nötig.
        if uv run --no-sync wandb login --relogin "$WANDB_API_KEY" &>/dev/null; then
            log "W&B-Login OK"
        else
            err "W&B-Login fehlgeschlagen — API-Key ungültig?"
            exit 1
        fi
    fi
fi

# ── 5. Trainings-Befehl bauen ─────────────────────────────────────────────────
# Launcher: torchrun nur bei Multi-GPU, sonst plain python (rückwärtskompatibel).
NUM_GPUS="${NUM_GPUS:-1}"
if [[ "$NUM_GPUS" -gt 1 ]]; then
    LAUNCHER=(torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS")
    log "Multi-GPU-Modus: torchrun mit $NUM_GPUS Prozessen (DeepSpeed ZeRO-2, per_device = $GLOBAL_BATCH_SIZE / $NUM_GPUS)."
else
    LAUNCHER=(python)
fi

TRAIN_CMD=(
    uv run --no-sync "${LAUNCHER[@]}" "$GROOT_ROOT/gr00t/experiment/launch_finetune.py"
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
    --tune_visual            # ← einziger funktionaler Unterschied zu run_finetuning.sh
)

# ── Bild-Augmentierung / Domain Randomization (optional) ──────────────────────
# Für den Vision-Lauf ist das der entscheidende Schalter: ein aufgetauter Encoder,
# der im Training nur unveränderte Realbilder sieht, spezialisiert sich noch stärker
# auf sie (Lauf 2, Politik-Kollaps). Erst variierte Bilder geben dem Auftauen
# überhaupt eine Chance gegen den Sim-Domain-Gap.
if [[ "$USE_AUGMENTATION" == "1" ]]; then
    log "Augmentierung AN — Color-Jitter (b=$CJ_BRIGHTNESS c=$CJ_CONTRAST s=$CJ_SATURATION h=$CJ_HUE)"
    TRAIN_CMD+=(--color_jitter_params brightness "$CJ_BRIGHTNESS" contrast "$CJ_CONTRAST" saturation "$CJ_SATURATION" hue "$CJ_HUE")
    [[ -n "$RANDOM_ROTATION_ANGLE" ]] && TRAIN_CMD+=(--random_rotation_angle "$RANDOM_ROTATION_ANGLE")
    if [[ "$STATE_DROPOUT_PROB" != "0.0" && "$STATE_DROPOUT_PROB" != "0" ]]; then
        TRAIN_CMD+=(--state_dropout_prob "$STATE_DROPOUT_PROB")
    fi
else
    warn "Augmentierung AUS bei --tune_visual — das ist die Konfiguration von Lauf 2,"
    warn "die zum Politik-Kollaps geführt hat (siehe docs/ergebnisse/lauf2-vision-auswertung.md)."
    TRAIN_CMD+=(--color_jitter_params brightness 0 contrast 0 saturation 0 hue 0)
fi

[[ "$USE_WANDB" == "1" ]] && TRAIN_CMD+=(--use_wandb --wandb_project "$WANDB_PROJECT")

log "Vision-Encoder-Tuning AKTIV (--tune_visual). LLM bleibt eingefroren."
log "Trainings-Befehl:"
printf '    %s\n' "${TRAIN_CMD[*]}"

if [[ "$DRY_RUN" == "1" ]]; then
    log "Dry-Run — nicht ausgeführt."
    exit 0
fi

# ── 6. Training ausführen, Logs nach /data/logs (= Host-sichtbar) ─────────────
LOG_DIR="/data/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/finetune-vision-$(date +%Y%m%d-%H%M%S).log"
log "Logs: $LOG_FILE  (auf dem Host: ./data/logs/$(basename "$LOG_FILE"))"
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
    if [[ "$TRAIN_TEST_SPLIT" == "1" ]]; then
        log "Checkpoint-Auswahl (nicht blind den letzten nehmen):"
        log "    python /scripts/checkpoint_sweep.py --run-dir $OUTPUT_DIR --dataset-path $DATASET_PATH"
    fi
else
    err "Training mit Exit-Code $EXIT_CODE beendet — siehe $LOG_FILE"
fi
exit "$EXIT_CODE"
