#!/usr/bin/env bash
# lib_resume_guard.sh — Schutz gegen stilles Fortsetzen eines alten Laufs.
#
# Wird von run_finetuning.sh und run_finetuning_vision.sh gesourct:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/lib_resume_guard.sh"
#     resume_guard "$OUTPUT_DIR" "$EXPERIMENT_NAME" "$MAX_STEPS"
#
# ── Warum ──────────────────────────────────────────────────────────────────────
# gr00t/experiment/experiment.py:288 ruft UNBEDINGT
#
#     trainer.train(resume_from_checkpoint=True)
#
# Der HuggingFace-Trainer sucht dann den letzten Checkpoint in
# `OUTPUT_DIR/EXPERIMENT_NAME` und stellt dessen `global_step` wieder her. Es gibt
# keinen Schalter dagegen — weder in `FinetuneConfig` noch als CLI-Flag.
#
# Folge: Wer einen neuen Lauf in ein bereits belegtes Verzeichnis startet, trainiert
# NICHT neu, sondern setzt fort. Liegt der vorhandene Checkpoint schon bei MAX_STEPS,
# ist die Trainingsschleife sofort zu Ende — der Job meldet „Training completed",
# speichert einen Checkpoint mit den ALTEN Gewichten und sieht erfolgreich aus.
#
# Genau das ist am 2026-08-13 passiert (Job 15271760): Lauf 3 lief in den Namespace
# von Lauf 2 (`blockstacking_vision/g1_dex3_blockstacking_vision_v1`, checkpoint-44000),
# war nach 255 s „fertig", protokollierte keinen einzigen Loss-Wert und legte
# `checkpoint-44001` mit den Gewichten von Lauf 2 ab. Der Train/Test-Split und die
# Augmentierung dieses Laufs hatten damit auf kein einziges Gewicht Wirkung.

declare -F log  >/dev/null || log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
declare -F err  >/dev/null || err()  { printf '\033[1;31m!! \033[0m%s\n' "$*" >&2; }
declare -F warn >/dev/null || warn() { printf '\033[1;33m ! \033[0m%s\n' "$*"; }

# resume_guard <output_dir> <experiment_name> <max_steps>
#
# RESUME=1 → Fortsetzen ist gewollt (z. B. nach Walltime-Abbruch), nur Hinweis.
# sonst    → Abbruch mit Erklärung, wenn schon Checkpoints da sind.
resume_guard() {
    local output_dir="$1" experiment_name="$2" max_steps="$3"
    local trainer_dir="$output_dir/$experiment_name"

    local last_ckpt last_step
    last_ckpt="$(find "$trainer_dir" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
        | sed 's/.*checkpoint-//' | grep -E '^[0-9]+$' | sort -n | tail -1)"

    if [[ -z "$last_ckpt" ]]; then
        return 0                      # sauberes Verzeichnis, nichts zu tun
    fi
    last_step="$last_ckpt"

    if [[ "${RESUME:-0}" == "1" ]]; then
        log "RESUME=1 — setze bei checkpoint-$last_step fort (von $max_steps Schritten)."
        if [[ "$last_step" -ge "$max_steps" ]]; then
            warn "Achtung: checkpoint-$last_step liegt bereits bei/über MAX_STEPS=$max_steps."
            warn "Der Trainer wird sofort fertig sein, ohne einen Schritt zu rechnen."
        fi
        return 0
    fi

    err "In $trainer_dir liegt bereits checkpoint-$last_step."
    err ""
    err "Der GR00T-Fork ruft trainer.train(resume_from_checkpoint=True) fest verdrahtet auf"
    err "(experiment.py:288). Dieser Lauf würde daher NICHT neu trainieren, sondern bei"
    err "Schritt $last_step fortsetzen — bei MAX_STEPS=$max_steps"
    if [[ "$last_step" -ge "$max_steps" ]]; then
        err "wäre er SOFORT fertig und würde die alten Gewichte als neuen Checkpoint ablegen."
        err "Der Job sähe erfolgreich aus, hätte aber nichts gelernt."
    else
        err "würden nur noch $((max_steps - last_step)) Schritte gerechnet."
    fi
    err ""
    err "Gewollt? Dann RESUME=1 setzen."
    err "Neuer Lauf? Dann einen eigenen Namespace wählen, z. B.:"
    err "    EXPERIMENT_NAME=${experiment_name%_v*}_v$(( ${experiment_name##*_v} + 1 ))"
    err "  oder OUTPUT_DIR=$output_dir-neu"
    return 1
}
