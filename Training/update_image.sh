#!/usr/bin/env bash
# TL;DR: Host-Werkzeug — baut das Docker-Image neu und pusht es nach Docker Hub.
# update_image.sh
#
# Bringt das Docker-Image lucam03/projekt-humanoider-roboter:latest auf den
# neuesten Stand:
#
#   1. Neuesten Commit der Isaac-GR00T-Repo ermitteln — fuer BEIDE Fork-Branches
#      (N1.6: luca/g1-dex3 -> ARG GROOT16_COMMIT, N1.7: luca/g1-dex3-n17 -> ARG GROOT17_COMMIT)
#   2. Dockerfile aktualisieren (falls --update-commit gesetzt und Commit neuer)
#   3. Image bauen (docker build) — Default: beide Modellgenerationen im Image
#      (GROOT_VERSIONS="1.6 1.7"; nur eine: GROOT_VERSIONS=1.6 bzw. 1.7, siehe Dockerfile)
#   4. Nach Docker Hub pushen
#
# Verwendung:
#   ./update_image.sh                    # Build + Push (aktueller Dockerfile)
#   ./update_image.sh --update-commit    # Neuesten GR00T-Commit eintragen + bauen
#   ./update_image.sh --no-cache         # Build ohne Docker-Cache
#   ./update_image.sh --skip-push        # Nur bauen, nicht pushen
#   ./update_image.sh --dry-run          # Befehle anzeigen, nichts ausfuehren
#
# Voraussetzungen:
#   docker login   (einmalig; Token wird in ~/.docker/config.json gespeichert)
#   git (im PATH)
#
# Umgebungsvariablen (optional, vor dem Aufruf setzen):
#   DOCKER_IMAGE     (default: lucam03/projekt-humanoider-roboter)
#   GROOT_REPO       (default: https://github.com/lucam06/Isaac-GR00T.git)
#   GROOT_BRANCH     (default: luca/g1-dex3)      — N1.6-Fork-Branch -> ARG GROOT16_COMMIT
#   GROOT17_BRANCH   (default: luca/g1-dex3-n17)  — N1.7-Fork-Branch -> ARG GROOT17_COMMIT
#   GROOT_VERSIONS   (default: "1.6 1.7")         — Build-Arg: welche Baeume ins Image kommen

set -euo pipefail

# ── Argumente parsen ─────────────────────────────────────────────────────────
UPDATE_COMMIT=0
NO_CACHE=0
SKIP_PUSH=0
DRY_RUN=0
SHOW_HELP=0

for arg in "$@"; do
    case "$arg" in
        --update-commit) UPDATE_COMMIT=1 ;;
        --no-cache)      NO_CACHE=1 ;;
        --skip-push)     SKIP_PUSH=1 ;;
        --dry-run)       DRY_RUN=1 ;;
        -h|--help)       SHOW_HELP=1 ;;
        *)
            echo "Unbekanntes Argument: $arg" >&2
            echo "Hilfe: $0 --help" >&2
            exit 1
            ;;
    esac
done

# ── Farben / Logging ─────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    C_CYAN=$'\033[36m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'
    C_RED=$'\033[31m';  C_MAGENTA=$'\033[35m'; C_RESET=$'\033[0m'
else
    C_CYAN=''; C_GREEN=''; C_YELLOW=''; C_RED=''; C_MAGENTA=''; C_RESET=''
fi

log()        { echo "${C_CYAN}==> $*${C_RESET}"; }
ok()         { echo "${C_GREEN} v  $*${C_RESET}"; }
warn()       { echo "${C_YELLOW}  ! $*${C_RESET}"; }
err()        { echo "${C_RED}!! $*${C_RESET}" >&2; }
exit_fatal() { err "$*"; exit 1; }

# ── Hilfe ────────────────────────────────────────────────────────────────────
if [[ $SHOW_HELP -eq 1 ]]; then
    # Kopfkommentar (Zeilen 3..26) ausgeben, fuehrendes "# " entfernen
    awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print; next } exit }' "$0"
    exit 0
fi

# ── Invoke-Cmd: fuehrt Befehl aus oder zeigt ihn im Dry-Run an ───────────────
invoke_cmd() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "${C_YELLOW}[dry-run] $*${C_RESET}"
        return
    fi
    if ! "$@"; then
        exit_fatal "Befehl fehlgeschlagen (Exit-Code $?): $*"
    fi
}

# ── Konfiguration ────────────────────────────────────────────────────────────
DOCKER_IMAGE="${DOCKER_IMAGE:-lucam03/projekt-humanoider-roboter}"
GROOT_REPO="${GROOT_REPO:-https://github.com/lucam06/Isaac-GR00T.git}"
GROOT_BRANCH="${GROOT_BRANCH:-luca/g1-dex3}"
GROOT17_BRANCH="${GROOT17_BRANCH:-luca/g1-dex3-n17}"
GROOT_VERSIONS="${GROOT_VERSIONS:-1.6 1.7}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="$SCRIPT_DIR/Dockerfile"

if [[ ! -f "$DOCKERFILE" ]]; then
    exit_fatal "Dockerfile nicht gefunden: $DOCKERFILE"
fi

# ── Banner ───────────────────────────────────────────────────────────────────
echo ""
echo "${C_MAGENTA}╔══════════════════════════════════════════════════════════════════╗${C_RESET}"
echo "${C_MAGENTA}║   GR00T N1.6 + N1.7 — Docker-Image Update                        ║${C_RESET}"
echo "${C_MAGENTA}╚══════════════════════════════════════════════════════════════════╝${C_RESET}"
echo ""
if [[ $DRY_RUN -eq 1 ]]; then warn "DRY-RUN aktiv — es werden keine Befehle ausgefuehrt."; fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 1 — Voraussetzungen pruefen
# ══════════════════════════════════════════════════════════════════════════════
log "Schritt 1/4 — Voraussetzungen pruefen"

if ! command -v docker >/dev/null 2>&1; then
    exit_fatal "docker nicht gefunden. Installation: https://docs.docker.com/get-docker/"
fi
if ! command -v git >/dev/null 2>&1; then
    exit_fatal "git nicht gefunden. Installation: https://git-scm.com"
fi

if ! docker info >/dev/null 2>&1; then
    exit_fatal "Docker-Daemon nicht erreichbar. Laeuft der Docker-Dienst?"
fi
ok "Docker-Daemon laeuft"

if [[ $SKIP_PUSH -eq 0 && $DRY_RUN -eq 0 ]]; then
    if ! docker info 2>/dev/null | grep -q "Username"; then
        warn "Nicht bei Docker Hub angemeldet."
        log "Starte 'docker login' ..."
        if ! docker login; then exit_fatal "Docker-Login fehlgeschlagen."; fi
    else
        ok "Bereits bei Docker Hub angemeldet"
    fi
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 2 — Commit-Stand pruefen (und ggf. Dockerfile aktualisieren)
# ══════════════════════════════════════════════════════════════════════════════
log "Schritt 2/4 — Isaac-GR00T Commit pruefen"

# Gepinnte Commits aus dem Dockerfile lesen: ARG GROOT16_COMMIT=<sha> / ARG GROOT17_COMMIT=<sha>
# (seit dem N1.7-Pfad zwei Pins; die alte Form `checkout <sha>` gibt es nicht mehr).
read_pin() {  # $1 = ARG-Name
    grep -oE "^ARG $1=[0-9a-f]{40}" "$DOCKERFILE" | head -n1 | sed "s/^ARG $1=//"
}
check_pin() {  # $1 = Label, $2 = ARG-Name, $3 = Branch
    local label="$1" arg="$2" branch="$3" current latest
    current="$(read_pin "$arg")"
    if [[ -z "$current" ]]; then
        warn "$label: kein gepinnter Commit (ARG $arg) im Dockerfile gefunden — ueberspringe."
        return 0
    fi
    ok "$label: gepinnter Commit ${current:0:12}... (Branch $branch)"
    latest="$(git ls-remote "$GROOT_REPO" "refs/heads/$branch" 2>/dev/null | awk '{print $1}' | head -n1 || true)"
    if [[ -z "$latest" ]]; then
        warn "$label: Remote-Commit nicht ermittelbar (kein Netzwerk oder Branch $branch fehlt?)."
    elif [[ "$latest" == "$current" ]]; then
        ok "$label: Dockerfile ist bereits auf dem neuesten Stand."
    else
        warn "$label: Neuer Commit verfuegbar!"
        warn "  Aktuell: ${current:0:12}..."
        warn "  Neu:     ${latest:0:12}..."
        if [[ $UPDATE_COMMIT -eq 1 ]]; then
            log "$label: Aktualisiere Dockerfile (--update-commit) ..."
            if [[ $DRY_RUN -eq 0 ]]; then
                sed -i "s/^ARG $arg=$current/ARG $arg=$latest/" "$DOCKERFILE"
                ok "$label: Dockerfile aktualisiert: $current -> $latest"
            else
                echo "${C_YELLOW}[dry-run] Ersetze ARG $arg=$current -> $latest in Dockerfile${C_RESET}"
            fi
        else
            warn "$label: Dockerfile wird NICHT aktualisiert (kein --update-commit)."
            warn "  Zum Aktualisieren: ./update_image.sh --update-commit"
        fi
    fi
}
[[ " $GROOT_VERSIONS " == *" 1.6 "* ]] && check_pin "N1.6" GROOT16_COMMIT "$GROOT_BRANCH"
[[ " $GROOT_VERSIONS " == *" 1.7 "* ]] && check_pin "N1.7" GROOT17_COMMIT "$GROOT17_BRANCH"
CURRENT_COMMIT="$(read_pin GROOT16_COMMIT)"
CURRENT_COMMIT17="$(read_pin GROOT17_COMMIT)"
echo ""


BUILD_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
IMAGE_LATEST="${DOCKER_IMAGE}:latest"
IMAGE_DATED="${DOCKER_IMAGE}:${BUILD_TIMESTAMP}"

BUILD_CMD=(docker build --platform linux/amd64 --build-arg "GROOT_VERSIONS=$GROOT_VERSIONS")
if [[ $NO_CACHE -eq 1 ]]; then BUILD_CMD+=(--no-cache); fi
BUILD_CMD+=(-t "$IMAGE_LATEST" -t "$IMAGE_DATED" "$SCRIPT_DIR")

log "Baue: ${BUILD_CMD[*]}"
if [[ $NO_CACHE -eq 1 ]]; then warn "Build ohne Cache — das kann 30-60 Minuten dauern (flash-attn)."; fi
echo ""

invoke_cmd "${BUILD_CMD[@]}"

ok "Image gebaut: $IMAGE_LATEST"
ok "Image gebaut: $IMAGE_DATED"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 4 — Nach Docker Hub pushen
# ══════════════════════════════════════════════════════════════════════════════
log "Schritt 4/4 — Nach Docker Hub pushen"

if [[ $SKIP_PUSH -eq 1 ]]; then
    warn "Push uebersprungen (--skip-push)."
else
    invoke_cmd docker push "$IMAGE_LATEST"
    invoke_cmd docker push "$IMAGE_DATED"
    ok "Gepusht: $IMAGE_LATEST"
    ok "Gepusht: $IMAGE_DATED"
fi
echo ""

ok "Fertig."
echo ""
echo "  Zusammenfassung:"
printf "    %-20s %s\n" "Image (latest):" "$IMAGE_LATEST"
printf "    %-20s %s\n" "Image (dated):"  "$IMAGE_DATED"
printf "    %-20s %s\n" "GROOT_VERSIONS:" "$GROOT_VERSIONS"
if [[ -n "$CURRENT_COMMIT" ]]; then
    printf "    %-20s %s\n" "N1.6-Commit:" "${CURRENT_COMMIT:0:12}..."
fi
if [[ -n "${CURRENT_COMMIT17:-}" ]]; then
    printf "    %-20s %s\n" "N1.7-Commit:" "${CURRENT_COMMIT17:0:12}..."
fi
echo ""
echo "  Naechste Schritte:"
echo "    * Testen:  ./setup_and_train_DockerHub-pull.sh"
echo "    * Ziehen:  docker pull $IMAGE_LATEST"
echo ""
