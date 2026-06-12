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

# Steps: bei ~241–301 Episoden (eine Aufgabe) sättigt Behavior Cloning früh; 20k Schritte
# reichen für gesunde Konvergenz (Lauf 1 war bei 30k bereits sauber konvergiert), sparen Zeit
# und reduzieren Memorierung, die den Sim-Domain-Gap verschärft. save_steps=2000 + limit=10
# → 10 gleichmäßig verteilte Checkpoints über den ganzen Lauf (gute Basis für die Open-Loop-
# Checkpoint-Auswahl), statt nur der letzten 5 wie zuvor.
MAX_STEPS="${MAX_STEPS:-20000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}" #8
DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"

# ── Train-Test-Split (optional, Standard AUS) ─────────────────────────────────
# TRAIN_TEST_SPLIT=1 schaltet den 80/20-Split scharf: patcht meta/info.json so, dass
# nur die ersten TRAIN_SPLIT_RATIO der Episoden als "train" geladen werden; der Rest
# steht als "test" für die Open-Loop-Eval auf ungesehenen Episoden bereit.
# Standard (0) → kompletter Datensatz wird trainiert (bisheriges Verhalten).
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-0}"
TRAIN_SPLIT_RATIO="${TRAIN_SPLIT_RATIO:-0.8}"

# ── Bild-Augmentierung / Domain Randomization (optional, Standard AN) ──────────
# USE_AUGMENTATION=1 → Color-Jitter + optionale Rotation/State-Dropout werden ans
# Training übergeben (adressiert den Sim-Real-Domain-Gap des eingefrorenen Encoders).
# USE_AUGMENTATION=0 → Color-Jitter explizit auf 0 gesetzt (Augmentierung effektiv aus;
# ein Weglassen der Flags würde sonst die Default-Augmentierung des Modells ziehen).
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

# ── 3b. Train-Test-Split scharf schalten (optional) ───────────────────────────
# Der Split-Code (_apply_split_filter) liest den Bereich aus meta/info.json; der
# Trainings-Datensatz fragt fest "train" ab (factory.py). Wir müssen also nur den
# splits-Eintrag in info.json setzen. Default (TRAIN_TEST_SPLIT=0) lässt info.json
# unangetastet → kompletter Datensatz wie bisher.
INFO_JSON="$DATASET_PATH/meta/info.json"
if [[ "$TRAIN_TEST_SPLIT" == "1" ]]; then
    [[ -f "$INFO_JSON" ]] || { err "TRAIN_TEST_SPLIT=1, aber info.json fehlt: $INFO_JSON"; exit 1; }
    log "TRAIN_TEST_SPLIT=1 — setze 80/20-Split (Ratio=$TRAIN_SPLIT_RATIO) in info.json"
    python - "$INFO_JSON" "$TRAIN_SPLIT_RATIO" <<'PY'
import json, sys
info_path, ratio = sys.argv[1], float(sys.argv[2])
with open(info_path) as f:
    info = json.load(f)
total = int(info.get("total_episodes") or 0)
if total <= 0:
    sys.exit(f"info.json hat kein gueltiges total_episodes ({total}).")
n_train = int(total * ratio)
if not (0 < n_train < total):
    sys.exit(f"Ungueltiger Split: ratio={ratio} -> n_train={n_train} von {total}.")
info["splits"] = {"train": f"0:{n_train}", "test": f"{n_train}:{total}"}
with open(info_path, "w") as f:
    json.dump(info, f, indent=4)
print(f"[split] train=0:{n_train}  test={n_train}:{total}  (gesamt {total} Episoden)")
PY
    log "Split aktiv: Test-Episoden werden NICHT mittrainiert (Eval danach mit split=\"test\")."
else
    # Sicherstellen, dass ein evtl. zuvor gesetzter Split wieder auf den vollen Datensatz
    # zurückfällt, damit ein Folge-Lauf ohne Split reproduzierbar alle Episoden sieht.
    if [[ -f "$INFO_JSON" ]] && grep -q '"test"' "$INFO_JSON" 2>/dev/null; then
        warn "TRAIN_TEST_SPLIT=0, aber info.json enthält einen test-Split — setze auf vollen Datensatz zurück."
        python - "$INFO_JSON" <<'PY'
import json, sys
info_path = sys.argv[1]
with open(info_path) as f:
    info = json.load(f)
total = int(info.get("total_episodes") or 0)
info["splits"] = {"train": f"0:{total}"}
with open(info_path, "w") as f:
    json.dump(info, f, indent=4)
print(f"[split] zurückgesetzt: train=0:{total} (voller Datensatz)")
PY
    fi
fi

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
# torchrun spawnt NUM_GPUS Prozesse und setzt WORLD_SIZE/LOCAL_RANK — genau die
# Env-Vars, die experiment.py für die nccl-Init erwartet. Ohne torchrun bleibt
# der Lauf Single-GPU, egal welcher Wert in --num_gpus steht.
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
)

# ── Bild-Augmentierung / Domain Randomization (optional) ──────────────────────
if [[ "$USE_AUGMENTATION" == "1" ]]; then
    log "Augmentierung AN — Color-Jitter (b=$CJ_BRIGHTNESS c=$CJ_CONTRAST s=$CJ_SATURATION h=$CJ_HUE)"
    TRAIN_CMD+=(--color_jitter_params brightness "$CJ_BRIGHTNESS" contrast "$CJ_CONTRAST" saturation "$CJ_SATURATION" hue "$CJ_HUE")
    [[ -n "$RANDOM_ROTATION_ANGLE" ]] && TRAIN_CMD+=(--random_rotation_angle "$RANDOM_ROTATION_ANGLE")
    # state_dropout_prob nur übergeben, wenn > 0 (Default 0.0 = aus)
    if [[ "$STATE_DROPOUT_PROB" != "0.0" && "$STATE_DROPOUT_PROB" != "0" ]]; then
        TRAIN_CMD+=(--state_dropout_prob "$STATE_DROPOUT_PROB")
    fi
else
    log "Augmentierung AUS — Color-Jitter explizit auf 0 (kein Modell-Default-Jitter)."
    TRAIN_CMD+=(--color_jitter_params brightness 0 contrast 0 saturation 0 hue 0)
fi

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
