#!/usr/bin/env bash
# TL;DR: Host-Skript: baut das Sim-Docker-Image (KISSKI oder --vastai) und pusht es zu Docker Hub.
# update_sim_image.sh — Linux/WSL-Port von update_sim_image.ps1
#
# Baut das Sim-Docker-Image und pusht es nach Docker Hub.
#
# Verwendung:
#   ./update_sim_image.sh              # KISSKI-Image bauen + pushen (Dockerfile)
#   ./update_sim_image.sh --vastai     # vast.ai-Image bauen + pushen (Dockerfile.vastai)
#   ./update_sim_image.sh --no-cache   # Build ohne Docker-Cache
#   ./update_sim_image.sh --skip-push  # Nur bauen, nicht pushen
#   ./update_sim_image.sh --dry-run    # Befehle anzeigen, nichts ausführen
#   ./update_sim_image.sh --help
#
# Voraussetzungen:
#   docker login nvcr.io   (Username: $oauthtoken, Password: NGC-API-Key)
#   docker login           (Docker Hub, für den Push)
#
# Umgebungsvariablen (optional):
#   DOCKER_IMAGE   (default abhängig von --vastai)

set -euo pipefail

# ── Logging ─────────────────────────────────────────────────────────────────
log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m v \033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m ! \033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m!! \033[0m %s\n' "$*" >&2; }
fatal(){ err "$*"; exit 1; }

# ── Argumente ─────────────────────────────────────────────────────────────────
NO_CACHE=0; SKIP_PUSH=0; DRY_RUN=0; VASTAI=0
for arg in "$@"; do
    case "$arg" in
        --no-cache)  NO_CACHE=1 ;;
        --skip-push) SKIP_PUSH=1 ;;
        --dry-run)   DRY_RUN=1 ;;
        --vastai|-VastAI) VASTAI=1 ;;
        --help|-h)   awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print; next } exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)           fatal "Unbekanntes Argument: $arg (--help für Hilfe)" ;;
    esac
done

run() {
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '\033[1;33m[dry-run]\033[0m %s\n' "$*"
        return 0
    fi
    "$@" || fatal "Befehl fehlgeschlagen (Exit $?): $*"
}

# ── Konfiguration ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$VASTAI" == "1" ]]; then
    DOCKER_IMAGE="${DOCKER_IMAGE:-lucam03/projekt-humanoider-roboter-sim-vastai}"
    DOCKERFILE="$SCRIPT_DIR/Dockerfile.vastai"
    TARGET="vast.ai (Isaac Sim + GR00T, Dockerfile.vastai)"
else
    DOCKER_IMAGE="${DOCKER_IMAGE:-lucam03/projekt-humanoider-roboter-sim}"
    DOCKERFILE="$SCRIPT_DIR/Dockerfile"
    TARGET="KISSKI/Apptainer (nur Sim-Client)"
fi

[[ -f "$DOCKERFILE" ]] || fatal "Dockerfile nicht gefunden: $DOCKERFILE"

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
printf '\033[1;35mGR00T N1.6 — Sim-Image Update (%s)\033[0m\n' "$TARGET"
echo ""
[[ "$DRY_RUN" == "1" ]] && warn "DRY-RUN aktiv — es werden keine Befehle ausgeführt."
if [[ "$VASTAI" == "1" ]]; then
    warn "Basis-Image: nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 (Isaac Sim 6.0, ~20–30 GB)"
else
    warn "Basis-Image: nvcr.io/nvidia/isaac-lab:2.3.2 (~20–30 GB)"
fi
warn "Bitte 'docker login nvcr.io' vorab ausführen (Username: \$oauthtoken)."
[[ "$VASTAI" == "1" ]] && warn "GPU-Anforderung: ≥24 GB VRAM, Ampere+, RT-Cores (RTX 3090/4090/A6000/L40)"
echo ""

# ── Schritt 1: Voraussetzungen prüfen ──────────────────────────────────────────
log "Schritt 1/3 — Voraussetzungen prüfen"
command -v docker >/dev/null 2>&1 || fatal "docker nicht gefunden. https://docs.docker.com/get-docker/"
docker info >/dev/null 2>&1 || fatal "Docker-Daemon nicht erreichbar. Läuft Docker?"
ok "Docker-Daemon läuft"

if [[ "$SKIP_PUSH" != "1" && "$DRY_RUN" != "1" ]]; then
    if docker info 2>/dev/null | grep -q "Username"; then
        ok "Bereits bei Docker Hub angemeldet"
    else
        warn "Nicht bei Docker Hub angemeldet."
        log "Starte 'docker login' …"
        docker login || fatal "Docker-Login fehlgeschlagen."
    fi
fi
echo ""

# ── Schritt 2: Image bauen ──────────────────────────────────────────────────────
log "Schritt 2/3 — Image bauen"
BUILD_TS="$(date +%Y%m%d-%H%M%S)"
IMAGE_LATEST="${DOCKER_IMAGE}:latest"
IMAGE_DATED="${DOCKER_IMAGE}:${BUILD_TS}"

BUILD_CMD=(docker build --platform linux/amd64 -f "$DOCKERFILE")
[[ "$NO_CACHE" == "1" ]] && BUILD_CMD+=(--no-cache)
BUILD_CMD+=(-t "$IMAGE_LATEST" -t "$IMAGE_DATED" "$SCRIPT_DIR")

log "Baue: ${BUILD_CMD[*]}"
[[ "$NO_CACHE" == "1" ]] && warn "Build ohne Cache — Isaac-Lab-Basis ist groß, kann 60+ Minuten dauern."
echo ""
run "${BUILD_CMD[@]}"
ok "Image gebaut: $IMAGE_LATEST"
ok "Image gebaut: $IMAGE_DATED"
echo ""

# ── Schritt 3: Push ─────────────────────────────────────────────────────────────
log "Schritt 3/3 — Nach Docker Hub pushen"
if [[ "$SKIP_PUSH" == "1" ]]; then
    warn "Push übersprungen (--skip-push)."
else
    run docker push "$IMAGE_LATEST"
    run docker push "$IMAGE_DATED"
    ok "Gepusht: $IMAGE_LATEST"
    ok "Gepusht: $IMAGE_DATED"
fi
echo ""

ok "Fertig."
echo ""
echo "  Zusammenfassung:"
printf "    %-20s %s\n" "Image (latest):" "$IMAGE_LATEST"
printf "    %-20s %s\n" "Image (dated):"  "$IMAGE_DATED"
echo ""
echo "  Nächste Schritte:"
if [[ "$VASTAI" == "1" ]]; then
    echo "    1. Auf vast.ai starten:"
    echo "         Image:          $IMAGE_LATEST"
    echo "         GPU:            L40 / RTX 4090 (≥24 GB, Ampere+, RT-Cores)"
    echo "         Docker Options: --ipc=host --shm-size=16g"
    echo "         Env:            CHECKPOINT_PATH=/data/checkpoints/checkpoint-XXXX  HF_TOKEN=hf_..."
    echo "    2. Checkpoint per Volume/SCP oder HF_CHECKPOINT_REPO bereitstellen"
    echo "    3. Ergebnisse sichern (vor Destroy!):"
    echo "         docker cp CONTAINER:/data/sim_results ./sim_results"
    echo "         docker cp CONTAINER:/data/sim_videos  ./sim_videos"
    echo "    Anleitung: docs/simulation/vastai-anleitung.md"
else
    echo "    1. SIF auf KISSKI ziehen (Login-Node):"
    echo "         module load apptainer"
    echo "         apptainer pull .project/images/projekt-humanoider-roboter-sim.sif docker://$IMAGE_LATEST"
    echo "    2. Eval-Job einreichen: sbatch Simulation/kisski_sim_submit.sh"
fi
echo ""
