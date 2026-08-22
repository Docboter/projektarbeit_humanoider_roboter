#!/usr/bin/env bash
# TL;DR: Schlanker Host-Launcher — zieht das Augmentierungs-Image von Docker Hub, startet den autonomen Container.
# setup_and_augment_DockerHub-pull.sh
#
# Schlankes Host-Skript: zieht das Image von Docker Hub und startet den autonomen
# Container-Entrypoint. Alle eigentliche Arbeit (Download, Control-Videos, Cosmos-Inferenz,
# Datensatz-Zusammenbau) passiert IM Container — dasselbe Prinzip wie
# Training/setup_and_train_DockerHub-pull.sh.
#
# Konzept: KEIN persistenter Storage auf dem Host.
#   * Kein `-v`-Mount nach /data
#   * Kein `--rm` — der Container bleibt nach `stop` bestehen
#   * Daten, Control-Videos und der fertige Datensatz leben im Container-Filesystem
#   * Bei `--destroy` (oder docker rm) ist alles weg
#
# Verwendung:
#   HF_TOKEN=hf_... ./setup_and_augment_DockerHub-pull.sh
#   ./setup_and_augment_DockerHub-pull.sh --skip-pull           # Image schon lokal
#   ./setup_and_augment_DockerHub-pull.sh --interactive         # Shell statt Augmentierung
#   ./setup_and_augment_DockerHub-pull.sh --resume              # Bestehenden Container weiterlaufen lassen
#   ./setup_and_augment_DockerHub-pull.sh --destroy              # Alten Container loeschen + neu starten
#   ./setup_and_augment_DockerHub-pull.sh --dry-run             # Nur Befehle anzeigen
#
# Umgebungsvariablen (vollstaendige Referenz: docs/augmentation/anleitung.md):
#   HF_TOKEN                 (Pflicht)  HuggingFace-Token — braucht Zugriff auf den Datensatz
#                             UND das gated Modell nvidia/Cosmos-Transfer2.5-2B (Lizenz vorher
#                             manuell auf der HF-Modellseite akzeptieren)
#   NUM_GPU                  (default 1)
#   AUGMENT_EPISODE_LIMIT    (default 2)     — 0 = alle Train-Episoden
#   AUGMENT_CAMERAS          (default alle 4 Policy-Kameras)
#   AUGMENT_VARIANTS         (default 1)
#   AUGMENT_EDGE_THRESHOLD   (default medium) — very_low/low/medium/high/very_high
#   AUGMENT_MODEL_VARIANT    (default edge) — "edge/distilled" braucht COSMOS_EXPERIMENTAL_CHECKPOINTS
#   AUGMENT_HF_REPO          (optional)      — Upload-Ziel nach dem Zusammenbau
#   SKIP_INFERENCE           (optional)      — nur Specs generieren, dann stoppen
#   CONTAINER_NAME            (default groot-augment)
#   DOCKER_HUB_IMAGE          (default lucam03/projekt-humanoider-roboter-augmentation:latest)

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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# shellcheck source=../tools/lib_env_local.sh
source "$REPO_DIR/tools/lib_env_local.sh"
env_local_load "$REPO_DIR"

# ── Konfiguration ─────────────────────────────────────────────────────────────
DOCKER_HUB_IMAGE="${DOCKER_HUB_IMAGE:-lucam03/projekt-humanoider-roboter-augmentation:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-groot-augment}"

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "\033[35m╔══════════════════════════════════════════════════════════════════╗\033[0m"
echo -e "\033[35m║   Cosmos-Transfer2.5 Augmentierung — Host-Launcher               ║\033[0m"
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

if docker run --rm --gpus all --entrypoint nvidia-smi \
       "nvidia/cuda:12.8.0-base-ubuntu22.04" -L &>/dev/null 2>&1; then
    ok "NVIDIA Container Toolkit funktioniert"
else
    warn "NVIDIA Container Toolkit nicht verfuegbar oder keine GPU erkannt."
    warn "Cosmos-Transfer2.5 braucht ~65 GB VRAM — ohne GPU nicht moeglich."
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
        warn "  --resume   den Container weiterlaufen lassen (Daten bleiben)"
        warn "  --destroy  Container loeschen, alles verwerfen und neu starten"
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
if ! $INTERACTIVE; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        err "Kein HF_TOKEN gesetzt — er ist Pflicht."
        err "  Dauerhaft hinterlegen:  echo ': \"\${HF_TOKEN:=hf_...}\"' >> $REPO_DIR/.env.local"
        err "  Oder pro Aufruf:        HF_TOKEN=hf_... $0"
        exit 1
    fi
    ok "HF_TOKEN gesetzt"
    warn "Lizenz fuer nvidia/Cosmos-Transfer2.5-2B vorher akzeptiert?"
    warn "  https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B — der Container prueft das"
    warn "  selbst und bricht mit klarer Meldung ab, falls nicht."
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
# Bewusst KEIN --rm und KEIN -v — dasselbe Prinzip wie Training/setup_and_train_DockerHub-pull.sh.
run_args=(
    "docker" "run"
    "--name" "$CONTAINER_NAME"
    "--gpus" "all"
    "--ipc=host"
    "--shm-size=16g"
)

if $INTERACTIVE; then
    log "Interaktive Shell — keine automatische Augmentierung."
    run_args+=("-it" "$DOCKER_HUB_IMAGE" "bash")
else
    echo "  Augmentierungs-Konfiguration:"
    printf "    %-25s %s\n" "CONTAINER_NAME" "$CONTAINER_NAME"
    for _v in NUM_GPU SOURCE_DATASET_REPO AUGMENT_TRAIN_RATIO AUGMENT_EPISODE_LIMIT \
              AUGMENT_EPISODE_IDS AUGMENT_CAMERAS AUGMENT_VARIANTS AUGMENT_PROMPT_TEMPLATE \
              AUGMENT_EDGE_THRESHOLD AUGMENT_MODEL_VARIANT AUGMENT_NUM_STEPS AUGMENT_STRICT_FRAME_CHECK \
              AUGMENT_OUT_DIR AUGMENT_HF_REPO SKIP_DOWNLOAD SKIP_CONVERT SKIP_INFERENCE \
              SKIP_ASSEMBLE AUGMENT_OVERWRITE SHELL_ON_ERROR; do
        [[ -n "${!_v:-}" ]] && printf "    %-25s %s\n" "$_v" "${!_v}"
    done
    echo ""

    run_args+=("-e" "HF_TOKEN=$HF_TOKEN")
    for _v in NUM_GPU SOURCE_DATASET_REPO AUGMENT_TRAIN_RATIO AUGMENT_EPISODE_LIMIT \
              AUGMENT_EPISODE_IDS AUGMENT_CAMERAS AUGMENT_VARIANTS AUGMENT_PROMPT_TEMPLATE \
              AUGMENT_EDGE_THRESHOLD AUGMENT_MODEL_VARIANT AUGMENT_NUM_STEPS AUGMENT_STRICT_FRAME_CHECK \
              AUGMENT_OUT_DIR AUGMENT_HF_REPO SKIP_DOWNLOAD SKIP_CONVERT SKIP_INFERENCE \
              SKIP_ASSEMBLE AUGMENT_OVERWRITE SHELL_ON_ERROR DATA_DIR HF_HOME; do
        [[ -n "${!_v:-}" ]] && run_args+=("-e" "$_v=${!_v}")
    done
    run_args+=("-it" "$DOCKER_HUB_IMAGE")
fi

invoke_cmd "${run_args[@]}"

echo ""
ok "Container beendet (nicht geloescht)."
echo ""
echo "  Naechste Schritte:"
echo "    * Container fortsetzen:  ./setup_and_augment_DockerHub-pull.sh --resume"
echo "    * Datensatz sichern:     docker cp $CONTAINER_NAME:/data/augmentation ./cosmos_augmented"
echo "    * Logs sichern:          docker cp $CONTAINER_NAME:/data/logs ./logs"
echo "    * Alles loeschen:        ./setup_and_augment_DockerHub-pull.sh --destroy"
echo ""
echo "  Ins Training einbinden: docs/training/co-training.md (USE_COTRAIN=1,"
echo "  COTRAIN_DATASET_PATH=<Pfad zum augmentierten Datensatz>)."
echo ""
