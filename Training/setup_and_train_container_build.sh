#!/usr/bin/env bash
# TL;DR: Host-Launcher — baut das Image lokal und startet es wie der Pull-Launcher (ein dauerhafter Container).
# setup_and_train_container_build.sh
#
# Baut das Trainings-Image aus Training/ lokal und übergibt dann an
# setup_and_train_dockerhub_pull.sh (--skip-pull, DOCKER_HUB_IMAGE=<lokaler Tag>). Damit gilt
# dasselbe Speichermodell wie dort: EIN dauerhafter Container ohne --rm; Download,
# Konvertierung und Training laufen im Entrypoint, samt TUNE_VISUAL-/USE_COTRAIN-Weiche.
# Der einzige Unterschied zur Pull-Fassung ist die Herkunft des Images.
#
# Bis 2026-10-13 lief hier jeder Schritt in einem eigenen `docker compose run --rm` ohne
# Volume: die Downloads fehlten beim Konvertieren, die Konvertierung beim Training, die
# Checkpoints gingen mit dem Container verloren; dazu rief er run_finetuning.sh direkt auf
# (an der Entrypoint-Weiche vorbei) und erzwang USE_WANDB=1 (Review 4.1).
#
# Läuft das Skript außerhalb eines Checkouts (Bootstrap), klont es zuerst das Repo nach
# REPO_DIR und startet sich von dort neu.
#
# Verwendung:
#   HF_TOKEN=hf_... ./Training/setup_and_train_container_build.sh    # bauen + trainieren
#   ./Training/setup_and_train_container_build.sh --skip-build       # vorhandenes lokales Image
#   ./Training/setup_and_train_container_build.sh --dry-run          # nur anzeigen
#   Alle übrigen Optionen (--interactive, --resume, --destroy, --no-menu, …) und alle
#   Env-Vars gehen unverändert an setup_and_train_dockerhub_pull.sh — siehe dort.
#
# Ältere Optionen, weiter angenommen:
#   --skip-download / --skip-convert  → SKIP_DOWNLOAD=1 / SKIP_CONVERT=1 im Container
#   --only-train                      → --skip-build + beide obigen
#   --skip-clone                      → ohne Wirkung (geklont wird nur außerhalb eines Checkouts)
#
# Umgebungsvariablen (nur dieses Skript):
#   GROOT_VERSION (default 1.6) — 1.6 | 1.7: welche Generation ins Image gebaut wird
#                (--build-arg GROOT_VERSIONS, Label de.humrob.groot-versions wie bei
#                update_image.sh). Geht danach unverändert an den Pull-Launcher, der Image
#                und Generation gegeneinander prüft. Vor dem Aufruf setzen — das Menü des
#                Pull-Launchers kommt erst NACH dem Build.
#   LOCAL_IMAGE  (default projektarbeit-humanoider-roboter:local, bei 1.7 :local-n17)
#   BUILD_ARGS   (leer) — zusätzliche Argumente für `docker build`, z. B. "--no-cache"
#   REPO_DIR     (default ./phr) — Ziel des Bootstrap-Klons
#   REPO_BRANCH  (default: Branch des umgebenden Checkouts, sonst main) — Branch des Bootstrap-Klons

set -euo pipefail

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  !\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m!! \033[0m%s\n' "$*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_IMAGE_SET="${LOCAL_IMAGE:-}"
LOCAL_IMAGE="${LOCAL_IMAGE:-projektarbeit-humanoider-roboter:local}"
REPO_URL="https://github.com/Docboter/projektarbeit_humanoider_roboter.git"
# Branch des Bootstrap-Klons: REPO_BRANCH=… schlaegt alles; sonst der Branch des Checkouts, in
# dem dieses Skript liegt (falls es in einem liegt), sonst main. Bis 2026-10-13 stand hier fest
# training-luca-IKR-IS6.0 (Review R5). Detached HEAD zaehlt nicht als Branch.
_repo_branch="$(git -C "$SCRIPT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
[[ "$_repo_branch" == "HEAD" ]] && _repo_branch=""
REPO_BRANCH="${REPO_BRANCH:-${_repo_branch:-main}}"

SKIP_BUILD=0; DRY_RUN=0; PASS=()
for arg in "$@"; do
    case "$arg" in
        --skip-build)    SKIP_BUILD=1 ;;
        --skip-download) export SKIP_DOWNLOAD=1 ;;
        --skip-convert)  export SKIP_CONVERT=1 ;;
        --only-train)    SKIP_BUILD=1; export SKIP_DOWNLOAD=1 SKIP_CONVERT=1 ;;
        --skip-clone|--skip-pull) ;;
        --dry-run)       DRY_RUN=1; PASS+=("$arg") ;;
        -h|--help)
            awk 'NR>2 { if (/^#/) { sub(/^# ?/, ""); print; next } exit }' "$0"
            exit 0 ;;
        *)               PASS+=("$arg") ;;
    esac
done

case "${GROOT_VERSION:-1.6}" in
    1.6|n16|N1.6|16) GROOT_VERSION=1.6 ;;
    1.7|n17|N1.7|17) GROOT_VERSION=1.7 ;;
    *) err "Unbekannte GROOT_VERSION '$GROOT_VERSION' (erlaubt: 1.6, 1.7)"; exit 1 ;;
esac
export GROOT_VERSION
[[ "$GROOT_VERSION" == 1.7 ]] && LOCAL_IMAGE="${LOCAL_IMAGE_SET:-projektarbeit-humanoider-roboter:local-n17}"

# ── Bootstrap: außerhalb eines Checkouts erst klonen ──────────────────────────
if [[ ! -x "$SCRIPT_DIR/setup_and_train_dockerhub_pull.sh" || ! -f "$SCRIPT_DIR/Dockerfile" ]]; then
    REPO_DIR="${REPO_DIR:-$(pwd)/phr}"
    log "Kein Checkout gefunden — klone $REPO_BRANCH nach $REPO_DIR"
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "    git clone --branch $REPO_BRANCH --recurse-submodules $REPO_URL $REPO_DIR"
        exit 0
    fi
    [[ -d "$REPO_DIR/.git" ]] \
        || git clone --branch "$REPO_BRANCH" --recurse-submodules "$REPO_URL" "$REPO_DIR"
    exec "$REPO_DIR/Training/setup_and_train_container_build.sh" "$@"
fi

# ── Image bauen ───────────────────────────────────────────────────────────────
# Build-Kontext ist Training/ (dort liegt das Dockerfile, .dockerignore hält ihn bei
# scripts/). Das Dockerfile klont den GR00T-Fork selbst, Submodule sind nicht nötig.
if [[ "$SKIP_BUILD" == 1 ]]; then
    warn "Build übersprungen — nutze vorhandenes $LOCAL_IMAGE."
else
    command -v docker >/dev/null 2>&1 || { err "docker fehlt."; exit 1; }
    # shellcheck disable=SC2206  # BUILD_ARGS ist bewusst eine Wortliste
    # Gleiche Build-Args und Generations-Label wie update_image.sh — der Pull-Launcher liest
    # das Label und lehnt ein Image ab, dem die gewählte Generation fehlt.
    build_cmd=(docker build -t "$LOCAL_IMAGE"
               --build-arg "GROOT_VERSIONS=$GROOT_VERSION"
               --build-arg "GROOT_VERSION_DEFAULT=$GROOT_VERSION"
               --label "de.humrob.groot-versions=$GROOT_VERSION"
               ${BUILD_ARGS:-} "$SCRIPT_DIR")
    log "Baue $LOCAL_IMAGE für GR00T N$GROOT_VERSION (erstes Mal ~30 min: PyTorch, flash-attn)"
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "    ${build_cmd[*]}"
    else
        "${build_cmd[@]}"
    fi
fi

# ── Übergabe an den Pull-Launcher ─────────────────────────────────────────────
log "Starte über setup_and_train_dockerhub_pull.sh mit dem lokalen Image"
DOCKER_HUB_IMAGE="$LOCAL_IMAGE" \
    exec "$SCRIPT_DIR/setup_and_train_dockerhub_pull.sh" --skip-pull "${PASS[@]}"
