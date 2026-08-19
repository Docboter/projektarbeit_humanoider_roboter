#!/usr/bin/env bash
# TL;DR: Schlanker Host-Launcher — zieht das Image von Docker Hub, startet den autonomen Container.
# setup_and_train_DockerHub-pull.sh
#
# Schlankes Host-Skript: zieht das Image von Docker Hub und startet den
# autonomen Container-Entrypoint. Alle eigentliche Arbeit (Download, Konvertierung,
# Training) passiert IM Container — das gleiche Image laeuft so auch auf vast.ai
# oder anderen Cloud-GPU-Plattformen ohne dieses Skript.
#
# Konzept: KEIN persistenter Storage auf dem Host.
#   * Kein `-v`-Mount nach /data
#   * Kein `--rm` — der Container bleibt nach `stop` bestehen
#   * Daten und Checkpoints leben im Container-Filesystem
#   * Bei `--destroy` (oder docker rm) ist alles weg
#
# Verwendung:
#   HF_TOKEN=hf_... WANDB_API_KEY=... ./setup_and_train_DockerHub-pull.sh
#   ./setup_and_train_DockerHub-pull.sh --skip-pull           # Image schon lokal
#   ./setup_and_train_DockerHub-pull.sh --interactive         # Shell statt Training
#   ./setup_and_train_DockerHub-pull.sh --resume              # Bestehenden Container weiterlaufen lassen
#   ./setup_and_train_DockerHub-pull.sh --destroy             # Alten Container loeschen + neu starten
#   ./setup_and_train_DockerHub-pull.sh --dry-run             # Nur Befehle anzeigen
#
#   Ohne Parameter aufgerufen fuehrt das Skript durch die noetigen Werte (gefuehrtes
#   Menue, docs/weiterfuehrend/cli-menuefuehrung.md). --no-menu bzw. MENU=0 schaltet
#   das ab; jede Aktion bleibt vollstaendig per Flag und Env-Var aufrufbar.
#   --profile=<name> laedt ein zuvor gesichertes Profil.
#
# Umgebungsvariablen:
#   HF_TOKEN           (Pflicht)  HuggingFace-Token
#   WANDB_API_KEY      (optional) W&B-Key — ohne laeuft Training ohne W&B
#   MAX_STEPS          (default 30000)
#   GLOBAL_BATCH_SIZE  (default 8)
#   NUM_GPUS           (default 1)
#   WANDB_PROJECT      (default gr00t-g1-dex3)
#   GROOT_VERSION      (default 1.6)      1.6 | 1.7 — waehlt Code-Baum/Modell/venv im
#                                         Container (siehe Training/scripts/lib_groot_version.sh)
#   CONTAINER_NAME     (default groot-train)
#   DOCKER_HUB_IMAGE   (default lucam03/projekt-humanoider-roboter:latest)

set -euo pipefail

# ── Flags ─────────────────────────────────────────────────────────────────────
SKIP_PULL=false
INTERACTIVE=false
RESUME=false
DESTROY=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --skip-pull)   SKIP_PULL=true ;;
        --interactive) INTERACTIVE=true ;;
        --resume)      RESUME=true ;;
        --destroy)     DESTROY=true ;;
        --dry-run)     DRY_RUN=true ;;
        --menu)        MENU=1 ;;
        --no-menu)     MENU=0 ;;
        --profile=*)   MENU_PROFILE="${arg#*=}"; MENU=1 ;;
        --help|-h)
            awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print; next } if (/^[[:space:]]*$/) { print ""; next } exit }' "$0"
            exit 0
            ;;
        *)
            echo "Unbekannter Parameter: $arg" >&2
            echo "Verwende --help fuer Hilfe." >&2
            exit 1
            ;;
    esac
done

# ── Logging ───────────────────────────────────────────────────────────────────
log()   { echo -e "\033[36m==> $1\033[0m"; }
ok()    { echo -e "\033[32m v  $1\033[0m"; }
warn()  { echo -e "\033[33m  ! $1\033[0m"; }
err()   { echo -e "\033[31m!! $1\033[0m"; }
fatal() { err "$1"; exit 1; }

invoke_cmd() {
    if $DRY_RUN; then
        echo -e "\033[33m[dry-run] $*\033[0m"
        return 0
    fi
    "$@"
}

# ── Repo-Wurzel + lokale Host-Konfiguration ───────────────────────────────────
# Bis 2026-08 las NUR server_rl_run.sh die gitignorierte .env.local — die in
# docs/portabilitaet.md beschriebene Vorrangregel galt hier also gar nicht, und ein
# dort hinterlegter HF_TOKEN wurde trotzdem abgefragt. Jetzt teilen sich beide Seiten
# dieselbe Fassung.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# shellcheck source=../tools/lib_env_local.sh
source "$REPO_DIR/tools/lib_env_local.sh"
env_local_load "$REPO_DIR"

# ── Gefuehrte Menuefuehrung ───────────────────────────────────────────────────
# Das Menue erzeugt NUR Umgebungsvariablen und laeuft VOR allem anderen. Es meldet
# sich ausschliesslich, wenn wirklich ein Mensch davorsitzt.
# shellcheck source=../tools/lib_menu.sh
source "$REPO_DIR/tools/lib_menu.sh"
_MENU_LAUNCHER="./Training/setup_and_train_DockerHub-pull.sh"

MENU_ACTION=""
if $RESUME;      then MENU_ACTION=resume
elif $DESTROY;   then MENU_ACTION=destroy
elif $INTERACTIVE; then MENU_ACTION=interactive
fi
if menu_enabled; then
    if [[ -z "$MENU_ACTION" ]]; then
        # Die Aktionsliste ersetzt die frueher handgestrickte resume/destroy-Abfrage
        # weiter unten — sie kommt jetzt VOR der Arbeit statt mitten hinein, und sie
        # zeigt gleich mit an, ob ueberhaupt ein Container existiert.
        MENU_ACTION="$(menu_pick_action "$REPO_DIR/tools/menu" train)" || { echo; exit 0; }
    fi
    menu_ask "$REPO_DIR/tools/menu" train "$MENU_ACTION" || exit 0
    case "$MENU_ACTION" in
        resume)      RESUME=true ;;
        destroy)     DESTROY=true ;;
        interactive) INTERACTIVE=true ;;
    esac
fi

# ── Konfiguration ─────────────────────────────────────────────────────────────
DOCKER_HUB_IMAGE="${DOCKER_HUB_IMAGE:-lucam03/projekt-humanoider-roboter:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-groot-train}"

MAX_STEPS="${MAX_STEPS:-30000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-gr00t-g1-dex3}"
GROOT_VERSION="${GROOT_VERSION:-1.6}"

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "\033[35m╔══════════════════════════════════════════════════════════════════╗\033[0m"
echo -e "\033[35m║   GR00T N1.6 Fine-tuning — Host-Launcher (Container ist autonom) ║\033[0m"
echo -e "\033[35m╚══════════════════════════════════════════════════════════════════╝\033[0m"
echo ""
$DRY_RUN && warn "DRY-RUN aktiv — es werden keine Befehle ausgefuehrt."

# ── 1. Voraussetzungen ────────────────────────────────────────────────────────
log "Schritt 1/3 — Voraussetzungen pruefen"

if ! command -v docker &>/dev/null; then
    fatal "docker nicht gefunden. Installation: https://docs.docker.com/get-docker/"
fi
if ! docker info &>/dev/null; then
    fatal "Docker-Daemon nicht erreichbar."
fi
ok "Docker-Daemon laeuft"

# NVIDIA Container Toolkit (nur informativ)
if docker run --rm --gpus all --entrypoint nvidia-smi \
       "nvidia/cuda:12.8.0-base-ubuntu22.04" -L &>/dev/null 2>&1; then
    ok "NVIDIA Container Toolkit funktioniert"
else
    warn "NVIDIA Container Toolkit nicht verfuegbar oder keine GPU erkannt."
    warn "Training ohne GPU nicht moeglich."
    if [[ -t 0 ]]; then
        ans=""
        read -rp "  Trotzdem fortfahren? [j/N] " ans || true
        [[ "${ans,,}" == "j" ]] || fatal "Abgebrochen."
    else
        fatal "Keine GPU erkannt und kein Terminal zum Nachfragen. Abgebrochen."
    fi
fi
echo ""

# ── 2. Bestehenden Container behandeln ────────────────────────────────────────
log "Schritt 2/3 — Container-Status pruefen ($CONTAINER_NAME)"

CONTAINER_EXISTS=false
CONTAINER_RUNNING=false
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    CONTAINER_EXISTS=true
    if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        CONTAINER_RUNNING=true
    fi
fi

if $DESTROY && $CONTAINER_EXISTS; then
    warn "Loesche bestehenden Container '$CONTAINER_NAME' (--destroy)."
    invoke_cmd docker rm -f "$CONTAINER_NAME"
    CONTAINER_EXISTS=false
    CONTAINER_RUNNING=false
fi

if $CONTAINER_RUNNING; then
    warn "Container '$CONTAINER_NAME' laeuft bereits."
    log "Haenge an die laufende Konsole an (Ctrl+P, Ctrl+Q zum Loesen ohne Stop)…"
    invoke_cmd docker attach "$CONTAINER_NAME"
    exit 0
fi

if $CONTAINER_EXISTS; then
    if $RESUME; then
        log "Starte bestehenden Container '$CONTAINER_NAME' (--resume)…"
        invoke_cmd docker start -ai "$CONTAINER_NAME"
        exit 0
    else
        warn "Container '$CONTAINER_NAME' existiert bereits (gestoppt)."
        warn "Optionen:"
        warn "  --resume   den Container weiterlaufen lassen (Daten + Checkpoints bleiben)"
        warn "  --destroy  Container loeschen, alles verwerfen und neu starten"
        # Ohne Terminal hier nicht fragen, sondern abbrechen: `read` wuerde unter
        # `set -euo pipefail` bei EOF das Skript stumm beenden.
        if [[ ! -t 0 ]]; then
            fatal "Kein Terminal — bitte --resume oder --destroy angeben."
        fi
        ans=""
        read -rp "  Was tun? [r=resume / d=destroy / a=abbrechen] " ans || true
        case "${ans,,}" in
            r) invoke_cmd docker start -ai "$CONTAINER_NAME"; exit 0 ;;
            d) invoke_cmd docker rm -f "$CONTAINER_NAME"; CONTAINER_EXISTS=false ;;
            *) fatal "Abgebrochen." ;;
        esac
    fi
fi
echo ""

# ── 3. Pflicht-Env pruefen ────────────────────────────────────────────────────
# Die frueheren zwei handgestrickten `read`-Abfragen standen hier. Sie hatten zwei
# Probleme: sie kannten nur diese beiden Variablen (alle Trainingsparameter musste man
# vorher wissen), und ohne Terminal riss `read` unter `set -euo pipefail` das Skript
# kommentarlos mit — `./setup_and_train_… < /dev/null` starb genau an dieser Stelle.
# Das Fragen erledigt jetzt das Menue weiter oben; hier bleibt nur die Pruefung.
if ! $INTERACTIVE; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        err "Kein HF_TOKEN gesetzt — er ist Pflicht."
        err "  Dauerhaft hinterlegen:  echo ': \"\${HF_TOKEN:=hf_...}\"' >> $REPO_DIR/.env.local"
        err "  Oder pro Aufruf:        HF_TOKEN=hf_... $0"
        menu_enabled || err "  Oder das gefuehrte Menue nutzen:  $0 --menu"
        exit 1
    fi
    ok "HF_TOKEN gesetzt"

    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        warn "Kein WANDB_API_KEY gesetzt — Training laeuft ohne W&B-Logging."
    fi
fi
echo ""

# ── 4. Image ziehen ───────────────────────────────────────────────────────────
log "Schritt 3/3 — Docker-Image laden und Container starten"
if $SKIP_PULL; then
    warn "Pull uebersprungen (--skip-pull)."
else
    invoke_cmd docker pull "$DOCKER_HUB_IMAGE"
fi
ok "Image bereit: $DOCKER_HUB_IMAGE"
echo ""

# ── 5. Container starten ──────────────────────────────────────────────────────
# Bewusst KEIN --rm und KEIN -v:
#   * --rm waere fatal — wir wollen den Container nach Stop behalten (Daten!)
#   * -v entfaellt — Daten leben im Container-Filesystem (vast.ai-Modell)
run_args=(
    "docker" "run"
    "--name" "$CONTAINER_NAME"
    "--gpus" "all"
    "--ipc=host"
    "--shm-size=16g"
)

if $INTERACTIVE; then
    log "Interaktive Shell — kein automatisches Training."
    run_args+=("-it" "$DOCKER_HUB_IMAGE" "bash")
else
    echo "  Trainings-Konfiguration:"
    printf "    %-25s %s\n" "MAX_STEPS"         "$MAX_STEPS"
    printf "    %-25s %s\n" "GLOBAL_BATCH_SIZE" "$GLOBAL_BATCH_SIZE"
    printf "    %-25s %s\n" "NUM_GPUS"          "$NUM_GPUS"
    printf "    %-25s %s\n" "WANDB_PROJECT"     "$WANDB_PROJECT"
    printf "    %-25s %s\n" "GROOT_VERSION"     "$GROOT_VERSION"
    printf "    %-25s %s\n" "CONTAINER_NAME"    "$CONTAINER_NAME"
    # Auch die durchgereichten Schalter anzeigen — sonst faellt nicht auf, wenn einer fehlt.
    for _v in TUNE_VISUAL USE_COTRAIN COTRAIN_MIX_RATIO TRAIN_TEST_SPLIT \
              USE_AUGMENTATION SKIP_DOWNLOAD SKIP_CONVERT SKIP_TRAIN \
              SHELL_ON_ERROR WANDB_MODE; do
        [[ -n "${!_v:-}" ]] && printf "    %-25s %s\n" "$_v" "${!_v}"
    done
    echo ""

    run_args+=(
        "-e" "HF_TOKEN=$HF_TOKEN"
        "-e" "MAX_STEPS=$MAX_STEPS"
        "-e" "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
        "-e" "NUM_GPUS=$NUM_GPUS"
        "-e" "WANDB_PROJECT=$WANDB_PROJECT"
        "-e" "GROOT_VERSION=$GROOT_VERSION"
    )
    [[ -n "${WANDB_API_KEY:-}" ]] && run_args+=("-e" "WANDB_API_KEY=$WANDB_API_KEY")

    # Bis 2026-08 endete die Liste hier — die Feature-Schalter des Entrypoints waren vom
    # Host aus also gar nicht erreichbar. Wer TUNE_VISUAL=1 ./setup_and_train_… aufrief,
    # bekam still ein normales Training: die Variable stand in der Host-Shell und kam nie
    # im Container an. entrypoint.sh liest 19 Variablen, weitergereicht wurden 6.
    # Weitergereicht wird nur, was auch gesetzt ist — sonst ueberschriebe ein leeres
    # "-e VAR=" die ENV-Defaults aus dem Dockerfile.
    for _v in TUNE_VISUAL USE_COTRAIN COTRAIN_MIX_RATIO COTRAIN_DATASET_PATH \
              COTRAIN_HF_REPO TRAIN_TEST_SPLIT TRAIN_SPLIT_RATIO USE_AUGMENTATION \
              USE_RL SKIP_DOWNLOAD SKIP_CONVERT SKIP_TRAIN SHELL_ON_ERROR \
              WANDB_MODE WANDB_DIR DATA_DIR; do
        [[ -n "${!_v:-}" ]] && run_args+=("-e" "$_v=${!_v}")
    done
    # -it sorgt fuer farbiges Log + Ctrl+C; bei reinem Headless waere -d sinnvoll.
    run_args+=("-it" "$DOCKER_HUB_IMAGE")
fi

invoke_cmd "${run_args[@]}"

echo ""
ok "Container beendet (nicht geloescht)."
echo ""
echo "  Naechste Schritte:"
echo "    * Container fortsetzen:  ./setup_and_train_DockerHub-pull.sh --resume"
echo "    * Checkpoints sichern:   docker cp $CONTAINER_NAME:/data/g1_dex3_finetune ./checkpoints"
echo "    * Logs sichern:          docker cp $CONTAINER_NAME:/data/logs ./logs"
echo "    * Alles loeschen:        ./setup_and_train_DockerHub-pull.sh --destroy"
echo ""
echo "  Hinweis: Auf vast.ai brauchst du dieses Skript NICHT — dort uebernimmt"
echo "  der vast.ai-Orchestrator die Container-Verwaltung. Du gibst nur das"
echo "  Image '$DOCKER_HUB_IMAGE' und die Env-Vars an."
echo ""
