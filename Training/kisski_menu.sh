#!/usr/bin/env bash
# TL;DR: Host-Menue auf dem KISSKI-Login-Node — fragt Partition/Walltime, baut die sbatch-Zeile.
# kisski_menu.sh — gefuehrtes Einreichen auf dem KISSKI-Login-Node.
#
# Baut die sbatch-Zeile und zeigt sie an, bevor sie laeuft. Gedacht fuer genau die zwei
# Werte, die man ohne docs/training/kisski-hpc.md nicht raet — Partition und Walltime —
# und fuer die eine Falle, die dieses Cluster stellt:
#
#   KISSKI setzt SBATCH_EXPORT=none auf dem Login-Node. Das ueberstimmt das
#   `#SBATCH --export=ALL` IM Job-Skript, sodass inline uebergebene Variablen
#   (MAX_STEPS=... sbatch ...) NICHT im Job ankommen — stillschweigend, mit den
#   Defaults statt der eigenen Werte. `--export=ALL` MUSS deshalb auf der
#   Kommandozeile stehen. Dieses Skript setzt es immer.
#
# BEWUSST EIGENSTAENDIG: kisski_submit.sh soll weiterhin allein auf den Cluster kopiert
# werden koennen (`scp kisski_submit.sh <user>@glogin-gpu...`). Deshalb bekommt es KEINE
# Abhaengigkeit auf tools/ — dieses Menue ist ein Zusatz, kein Ersatz. Wer das ganze Repo
# auf dem Cluster hat, kann es nutzen; wer nur das eine Skript kopiert, merkt nichts.
#
# NUTZUNG (auf dem Login-Node, im Repo-Verzeichnis):
#   ./Training/kisski_menu.sh              # fragt und reicht ein
#   ./Training/kisski_menu.sh --dry-run    # zeigt die Zeile nur an
#
# IM BATCH-JOB laeuft das hier nie: menu_enabled prueft SLURM_JOB_ID (siehe lib_menu.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m ! \033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m!! \033[0m %s\n' "$*" >&2; }

source "$REPO_DIR/tools/lib_env_local.sh"; env_local_load "$REPO_DIR"
source "$REPO_DIR/tools/lib_menu.sh"
_MENU_LAUNCHER="sbatch --export=ALL"

DRY_RUN=0; ACTION=""
for a in "$@"; do
  case "$a" in
    --dry-run) DRY_RUN=1 ;;
    --menu)    MENU=1 ;;
    --profile=*) MENU_PROFILE="${a#*=}"; MENU=1 ;;
    -h|--help) awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print; next } if (/^[[:space:]]*$/) { print ""; next } exit }' "$0"; exit 0 ;;
    *)         ACTION="$a" ;;
  esac
done

command -v sbatch >/dev/null 2>&1 || warn "sbatch nicht gefunden — laeuft dieses Skript wirklich auf dem Login-Node?"

if [[ -z "$ACTION" ]]; then
  menu_enabled || { err "Keine Aktion angegeben und kein Terminal zum Fragen."; \
                    err "  Verfuegbar: $(menu_list_actions "$REPO_DIR/tools/menu" kisski | tr '\n' ' ')"; exit 2; }
  ACTION="$(menu_pick_action "$REPO_DIR/tools/menu" kisski)" || { echo; exit 0; }
fi
menu_enabled && { menu_ask "$REPO_DIR/tools/menu" kisski "$ACTION" || exit 0; }

# Job-Skript zur Aktion. Die Zuordnung steht auch als --cli in den Specs (fuer die
# Anzeige des Ein-Zeilers); hier braucht es sie noch einmal ausfuehrbar.
case "$ACTION" in
  train)    JOB="Training/kisski_submit.sh" ;;
  openloop) JOB="Training/kisski_open_loop_eval.sh" ;;
  rl)       JOB="Training/kisski_rl_submit.sh" ;;
  sim)      JOB="Simulation/kisski_sim_submit.sh" ;;
  *) err "Unbekannte Aktion: '$ACTION'"; exit 2 ;;
esac
[[ -f "$REPO_DIR/$JOB" ]] || { err "Job-Skript fehlt: $JOB"; exit 1; }

# Partition und Walltime sind #SBATCH-Direktiven, keine Env-Vars — sie muessen als
# sbatch-Argumente kommen, sonst gewinnt die Zeile im Skript.
SB=( sbatch --export=ALL )
[[ -n "${KISSKI_PARTITION:-}" ]] && SB+=( --partition="$KISSKI_PARTITION" )
[[ -n "${KISSKI_TIME:-}" ]]      && SB+=( --time="$KISSKI_TIME" )
SB+=( "$REPO_DIR/$JOB" )

# Die Env-Vars, die der Job wirklich liest — nur gesetzte weiterreichen.
ENVLINE=""
for v in MAX_STEPS GLOBAL_BATCH_SIZE NUM_GPUS TUNE_VISUAL USE_COTRAIN \
         COTRAIN_HF_REPO COTRAIN_MIX_RATIO TRAIN_TEST_SPLIT TRAIN_SPLIT_RATIO \
         SKIP_DOWNLOAD SKIP_CONVERT SKIP_GIT_PULL RESUME KISSKI_PROJECT_DIR \
         RL_NUM_ENVS RL_ITERATIONS NUM_EPISODES EPISODE_LENGTH_S WANDB_PROJECT; do
  [[ -n "${!v:-}" ]] && { export "$v"; ENVLINE+="$v=${!v} "; }
done

echo ""
log "Einreichen als:"
echo "    ${ENVLINE}${SB[*]}"
echo ""
if (( DRY_RUN )); then
  warn "--dry-run: nicht eingereicht."
  exit 0
fi
"${SB[@]}"
echo ""
log "Fortschritt:  squeue -u \$USER"
log "Ausgabe:      tail -f logs/slurm-<jobid>.out"
