#!/usr/bin/env bash
# TL;DR: Login-Node-Helfer — reiht mehrere Trainingsläufe per SLURM-Abhängigkeit hintereinander, je mit Checkpoint-Sweep dahinter.
# kisski_chain.sh — Trainings-Warteschlange auf KISSKI.
#
# Für jeden genannten Lauf werden zwei Jobs eingereicht:
#
#   Training  (kisski_submit.sh)         --dependency=afterany:<Training davor>
#   Sweep     (kisski_open_loop_eval.sh) --dependency=afterok:<eigenes Training>
#
# Die Trainings laufen also streng nacheinander (afterany: ein abgebrochener Lauf hält die
# Kette nicht an), jeder Sweep direkt nach seinem Training — und nur, wenn es sauber
# durchlief. Die Läufe sind unten als Voreinstellungen hinterlegt, damit Namespace,
# Datensatz und Hyperparameter nicht bei jedem Einreichen neu getippt werden müssen.
#
# NUTZUNG (auf dem Login-Node, im Repo-Verzeichnis):
#   ./Training/kisski_chain.sh --list                               # Voreinstellungen zeigen
#   ./Training/kisski_chain.sh --dry-run synth_jointspace synth_v22 # nur anzeigen
#   ./Training/kisski_chain.sh synth_jointspace synth_v22           # einreichen
#   AFTER=1234567 ./Training/kisski_chain.sh synth_jointspace       # erst nach Job 1234567
#
# `--export=ALL` steht immer auf der sbatch-Zeile: KISSKI setzt SBATCH_EXPORT=none auf dem
# Login-Node, inline gesetzte Variablen kämen sonst nicht im Job an (siehe kisski_menu.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
KISSKI_PROJECT_DIR="${KISSKI_PROJECT_DIR:-/mnt/vast-kisski/projects/kisski-humrob}"
DATA_DIR="${DATA_DIR:-$KISSKI_PROJECT_DIR/data}"
REAL_DATASET="/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset"

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m ! \033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m!! \033[0m %s\n' "$*" >&2; }

# ── Voreinstellungen ──────────────────────────────────────────────────────────
# Alle synth-Läufe übernehmen die Parameter von cotrain_synth_v1
# (docs/training/synth-datensatz.md § 6.2/§ 7), damit sie gegen ihn vergleichbar bleiben.
# Ein Lauf = eine Zeile im case + eine Zeile in PRESETS.
PRESETS=(synth_jointspace synth_v22)

preset() {
    unset COTRAIN_DATASET_PATH OUTPUT_DIR EXPERIMENT_NAME DESCRIPTION
    USE_COTRAIN=1
    COTRAIN_MIX_RATIO="${COTRAIN_MIX_RATIO_OVERRIDE:-0.10}"
    TRAIN_TEST_SPLIT=1
    TRAIN_SPLIT_RATIO=0.8
    case "$1" in
        synth_jointspace)
            DESCRIPTION="Co-Training echt + Cube_Stacking_synth_jointspace (Aktion = state[t+1])"
            COTRAIN_DATASET_PATH=/data/cotrain/cube_stacking_synth_jointspace
            OUTPUT_DIR=/data/g1_dex3_finetune/blockstacking_cotrain_synth_jointspace
            EXPERIMENT_NAME=g1_dex3_blockstacking_cotrain_synth_jointspace_v1 ;;
        synth_v22)
            DESCRIPTION="Wiederholung von cotrain_synth_v1 mit korrigierter Handaktion (v22)"
            COTRAIN_DATASET_PATH=/data/cotrain/cube_stacking_synth_v22
            OUTPUT_DIR=/data/g1_dex3_finetune/blockstacking_cotrain_synth_v22
            EXPERIMENT_NAME=g1_dex3_blockstacking_cotrain_synth_v22 ;;
        *) return 1 ;;
    esac
}

host_path() { printf '%s\n' "${1/#\/data/$DATA_DIR}"; }

# ── Argumente ─────────────────────────────────────────────────────────────────
DRY_RUN=0; RUNS=()
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=1 ;;
        --list)
            for p in "${PRESETS[@]}"; do
                preset "$p"
                printf '  %-18s %s\n  %-18s → %s\n' "$p" "$DESCRIPTION" "" "$OUTPUT_DIR"
            done
            exit 0 ;;
        -h|--help) awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print; next } exit }' "$0"; exit 0 ;;
        -*) err "Unbekannte Option: $a"; exit 2 ;;
        *)  preset "$a" || { err "Unbekannter Lauf: '$a' (verfügbar: ${PRESETS[*]})"; exit 2; }
            RUNS+=("$a") ;;
    esac
done
[[ ${#RUNS[@]} -gt 0 ]] || { err "Keinen Lauf genannt. Verfügbar: ${PRESETS[*]}"; exit 2; }

# ── Vorabprüfung, bevor irgendetwas eingereicht wird ──────────────────────────
# Compute-Nodes haben kein Internet: ein fehlender Datensatz ist ein toter Job nach
# Stunden Warteschlange. Deshalb hier prüfen, nicht erst im Job.
fail=0
real_mod="$(host_path "$REAL_DATASET")/meta/modality.json"
for r in "${RUNS[@]}"; do
    preset "$r"
    ds="$(host_path "$COTRAIN_DATASET_PATH")"
    if [[ ! -f "$ds/meta/info.json" ]]; then
        err "$r: Datensatz fehlt: $ds  (rsync vom Laptop, siehe synth-datensatz.md § 6.1)"; fail=1
    elif [[ -f "$real_mod" ]] && ! cmp -s "$real_mod" "$ds/meta/modality.json"; then
        err "$r: modality.json weicht vom echten Datensatz ab — run_finetuning_cotrain.sh bricht ab"; fail=1
    fi
    out="$(host_path "$OUTPUT_DIR")"
    if compgen -G "$out/checkpoint-*" >/dev/null || compgen -G "$out/*/checkpoint-*" >/dev/null; then
        err "$r: $out enthält schon Checkpoints — der Fork würde FORTSETZEN statt neu trainieren."
        err "    Neuen EXPERIMENT_NAME/OUTPUT_DIR vergeben oder das Verzeichnis wegräumen."; fail=1
    fi
done
[[ -f "$real_mod" ]] || warn "Echter Datensatz nicht gefunden ($real_mod) — modality-Vergleich übersprungen."
[[ "$fail" == 0 ]] || { [[ "$DRY_RUN" == 1 ]] && warn "Dry-Run: trotzdem anzeigen." || exit 1; }

# ── Einreichen ────────────────────────────────────────────────────────────────
cd "$REPO_DIR"
mkdir -p logs
command -v sbatch >/dev/null 2>&1 || { [[ "$DRY_RUN" == 1 ]] || { err "sbatch fehlt — nicht auf dem Login-Node?"; exit 1; }; }

submit() {   # submit <dependency|""> <job-skript> ; gibt die Job-ID aus
    local dep="$1" job="$2" args=(sbatch --parsable --export=ALL)
    [[ -n "$dep" ]] && args+=("--dependency=$dep")
    args+=("$job")
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '    %s\n' "${args[*]}" >&2
        printf 'DRY%s\n' "$RANDOM"
    else
        "${args[@]}" | cut -d';' -f1
    fi
}

prev="${AFTER:-}"
SUMMARY=()
for r in "${RUNS[@]}"; do
    preset "$r"
    log "$r — $DESCRIPTION"
    echo "    COTRAIN_DATASET_PATH=$COTRAIN_DATASET_PATH  MIX=$COTRAIN_MIX_RATIO"
    echo "    OUTPUT_DIR=$OUTPUT_DIR"
    echo "    EXPERIMENT_NAME=$EXPERIMENT_NAME"
    train_id="$(
        export USE_COTRAIN COTRAIN_DATASET_PATH COTRAIN_MIX_RATIO TRAIN_TEST_SPLIT \
               TRAIN_SPLIT_RATIO OUTPUT_DIR EXPERIMENT_NAME
        submit "${prev:+afterany:$prev}" Training/kisski_submit.sh
    )"
    sweep_id="$(
        export RUN_DIR="$OUTPUT_DIR" DATASET_PATH="$REAL_DATASET"
        submit "afterok:$train_id" Training/kisski_open_loop_eval.sh
    )"
    echo "    Training: $train_id   Sweep: $sweep_id (afterok)"
    SUMMARY+=("$r  train=$train_id  sweep=$sweep_id  → $(host_path "$OUTPUT_DIR")/checkpoint_sweep.json")
    prev="$train_id"
done

echo
log "Kette eingereicht$([[ "$DRY_RUN" == 1 ]] && echo ' (DRY-RUN, nichts abgeschickt)'):"
printf '    %s\n' "${SUMMARY[@]}"
echo "    Beobachten:  squeue -u \$USER -o '%.10i %.20j %.8T %.20E %.10M'"
echo "    Abbrechen :  scancel <jobid>   (abhängige Jobs bleiben dann als DependencyNeverSatisfied stehen)"
