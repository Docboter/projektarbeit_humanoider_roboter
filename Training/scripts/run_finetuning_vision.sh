#!/usr/bin/env bash
# TL;DR: Läuft im Container — wie run_finetuning.sh, trainiert aber zusätzlich den Vision-Encoder mit.
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

# ── GR00T-Version auflösen (N1.6 | N1.7) ───────────────────────────────────────
# Muss vor jeder Verwendung von GROOT_ROOT/MODEL_PATH/OUTPUT_DIR laufen.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_groot_version.sh
source "$SCRIPT_DIR/lib_groot_version.sh"
groot_resolve || { echo "FEHLER: GR00T-Versionsauflösung fehlgeschlagen (GROOT_VERSION=${GROOT_VERSION:-})." >&2; exit 1; }

# ── Trainings-Parameter (alle via Env-Var überschreibbar) ─────────────────────
# DATA_DIR: Container-seitige Datenwurzel (Default /data wie Dockerfile-ENV und entrypoint.sh).
# Bis 2026-10-13 waren die Pfade unten fest auf /data codiert (Review T5).
DATA_DIR="${DATA_DIR:-/data}"
MODEL_PATH="${MODEL_PATH:-$DATA_DIR/models/$GROOT_MODEL_NAME}"
DATASET_PATH="${DATASET_PATH:-$DATA_DIR/unitreerobotics/G1_Dex3_BlockStacking_Dataset}"
# Eigener Output-/Experiment-Namespace, damit Vision-Läufe nicht die Standard-Läufe überschreiben.
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/g1_dex3_finetune/blockstacking_vision${GROOT_NS_SUFFIX}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_dex3_blockstacking_vision_v1}"
MODALITY_CONFIG="${MODALITY_CONFIG:-$GROOT_ROOT/examples/G1_DEX3/g1_dex3_config.py}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"

# 20000 wie Dockerfile-ENV und entrypoint.sh — im Image wirkt ohnehin das ENV; der Wert hier
# griff nur bei einem Aufruf außerhalb des Images und stand bis 2026-10-13 auf 30000 (Review 4.6).
# Plattform-Defaults setzen die Launcher: lokal 30000, KISSKI 44000.
MAX_STEPS="${MAX_STEPS:-20000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}" #8
DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
# Entkoppelt die effektive Batchgroesse vom Speicherbedarf. experiment.py rechnet
# per_device = global_batch_size / num_gpus (Zeile 188); die Akkumulation kommt oben drauf.
# Auf einer Karte mit knappem VRAM — etwa wenn andere Dienste die Haelfte belegen — haelt
# man damit die Batchgroesse des Referenzlaufs, statt sie senken und das Optimierungs-
# regime mitaendern zu muessen. 1 = aus, verhaelt sich exakt wie bisher.
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
# Checkpoints: einheitlich in allen drei Trainings-Skripten (Review T5). Bis 2026-10-13 standen
# vision/cotrain auf 1000/5 — bei einem 44k-Lauf blieben nur die letzten fuenf Checkpoints, zu
# wenig fuer checkpoint_sweep.py (der beste muss nicht am Ende liegen). 2000/15 haelt jeden
# Checkpoint eines Laufs bis 30000 Schritte (Default des Pull-Launchers), bei 44k die letzten
# 15 ab 16000; Plattenbedarf hoechstens ~330 GB (~22 GB je Checkpoint inkl. optimizer.pt).
# Knapper Platz: SAVE_TOTAL_LIMIT senken. KISSKI setzt eigene Werte (kisski_submit.sh: 5000/40).
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-15}"
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
# obwohl die Doku USE_AUGMENTATION als Schalter auswies. Ob Lauf 2 mit oder ohne
# Jitter lief, ist nicht belegt: die früheste versionierte Fassung (caabf76, ~2 h nach
# dem Start von Lauf 2) hat ihn fest drin, und die W&B-Config zeigt bei allen Läufen
# nur den Modell-Default (0,1), nicht die per CLI gesetzten Werte.
USE_AUGMENTATION="${USE_AUGMENTATION:-1}"
CJ_BRIGHTNESS="${CJ_BRIGHTNESS:-0.3}"
CJ_CONTRAST="${CJ_CONTRAST:-0.4}"
CJ_SATURATION="${CJ_SATURATION:-0.5}"
CJ_HUE="${CJ_HUE:-0.08}"
RANDOM_ROTATION_ANGLE="${RANDOM_ROTATION_ANGLE:-}"   # leer = keine Rotation
STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-}"   # leer = Modell-Default (N1.6: 0.0, N1.7: 0.2)

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-gr00t-g1-dex3}"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ── Helfer ────────────────────────────────────────────────────────────────────
log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m!! \033[0m%s\n' "$*" >&2; }
warn() { printf '\033[1;33m ! \033[0m%s\n' "$*"; }

# Split-Logik teilen sich beide Trainings-Skripte. SCRIPT_DIR wurde bereits oben
# für lib_groot_version.sh gesetzt.
# shellcheck source=lib_split.sh
source "$SCRIPT_DIR/lib_split.sh"
# shellcheck source=lib_resume_guard.sh
source "$SCRIPT_DIR/lib_resume_guard.sh"

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

# ── 3a. Gegen stilles Fortsetzen eines alten Laufs absichern ──────────────────
resume_guard "$OUTPUT_DIR" "$EXPERIMENT_NAME" "$MAX_STEPS" || exit 1

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
        # wandb.init() liest WANDB_API_KEY selbst aus der Umgebung — ein Login ist nicht noetig.
        # Bis 2026-10-13 stand hier `wandb login --relogin "$WANDB_API_KEY"`: der Key lag damit
        # in der Prozessliste (auf geteilten Knoten fuer andere lesbar) und in ~/.netrc (Review
        # T4). Jetzt nur noch eine Vorab-Pruefung des Keys gegen den Server, ohne ihn in argv.
        if uv run --no-sync python -c "import sys, wandb; sys.exit(0 if wandb.login(verify=True) else 1)" &>/dev/null; then
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
    # Sieht der Prozess mehrere Karten, packt der HF-Trainer das Modell ohne torchrun in
    # nn.DataParallel — N1.7 stirbt daran mit StopIteration (self.parameters() der Replika
    # ist leer), N1.6 verteilt still. Einzel-GPU heisst hier deshalb: genau eine Karte zeigen.
    # Ein vorgegebenes CUDA_VISIBLE_DEVICES (SLURM, Hand) bleibt unangetastet.
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] \
        && [[ "$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')" -gt 1 ]]; then
        export CUDA_VISIBLE_DEVICES=0
        warn "NUM_GPUS=1, aber mehrere GPUs sichtbar — nutze nur GPU 0 (CUDA_VISIBLE_DEVICES=0)."
    fi
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
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
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
    # state_dropout_prob: leer = NVIDIA-Default der Generation (N1.6 0.0, N1.7 0.2 — bewusst
    # uebernommen, siehe docs/weiterfuehrend/groot-n17-migration.md); gesetzt = immer uebergeben.
    if [[ -n "$STATE_DROPOUT_PROB" ]]; then
        TRAIN_CMD+=(--state_dropout_prob "$STATE_DROPOUT_PROB")
    fi
else
    warn "Augmentierung AUS bei --tune_visual — vermutlich die Konfiguration von Lauf 2 (nicht belegt),"
    warn "die zum Politik-Kollaps geführt hat (siehe docs/ergebnisse/lauf2-vision-auswertung.md)."
    TRAIN_CMD+=(--color_jitter_params brightness 0 contrast 0 saturation 0 hue 0)
    # Auch State-Dropout aus: "Aug aus" heißt keine Regularisierung durch Modell-Defaults.
    # Ohne Flag trainierte N1.7 hier mit seinem Default 0,2, und ein gesetztes
    # STATE_DROPOUT_PROB wurde ignoriert (Review 5.2d). Gesetzt = gilt auch hier.
    TRAIN_CMD+=(--state_dropout_prob "${STATE_DROPOUT_PROB:-0}")
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
LOG_DIR="$DATA_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/finetune-vision-$(date +%Y%m%d-%H%M%S).log"
log "Logs: $LOG_FILE  (auf dem Host: ./data/logs/$(basename "$LOG_FILE"))"
log "Training startet — abbrechen mit Ctrl+C."
echo

cd "$GROOT_ROOT"
# set +e: unter "set -euo pipefail" beendet eine scheiternde Pipe das Skript sofort, die
# Fehlermeldung unten mit Exit-Code und Log-Pfad erschiene dann nie (Review 4.2).
set +e
"${TRAIN_CMD[@]}" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

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
