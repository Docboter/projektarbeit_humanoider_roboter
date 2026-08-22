#!/usr/bin/env bash
# TL;DR: Host-Werkzeug — baut das Augmentierungs-Image neu und pusht es nach Docker Hub.
# update_image.sh
#
# Bringt das Docker-Image auf den neuesten Stand:
#
#   1. Neuesten TAG von cosmos-transfer2.5 ermitteln (bewusst kein Branch-HEAD, s. u.)
#   2. Dockerfile aktualisieren (falls --update-commit gesetzt und Tag neuer)
#   3. Image bauen (docker build)
#   4. Nach Docker Hub pushen
#
# Verwendung:
#   ./update_image.sh                    # Build + Push (aktueller Dockerfile-Stand)
#   ./update_image.sh --update-commit    # Neuesten cosmos-transfer2.5-Tag eintragen + bauen
#   ./update_image.sh --no-cache         # Build ohne Docker-Cache
#   ./update_image.sh --skip-push        # Nur bauen, nicht pushen
#   ./update_image.sh --dry-run          # Befehle anzeigen, nichts ausfuehren
#
# WARUM TAGS STATT main-HEAD: main hat .python-version zwischenzeitlich von 3.10 auf 3.13
# angehoben, waehrend der custom flash-attn-Wheel-Index (cu128_torch27) noch nur cp310
# liefert — `just install cu128` schlaegt dort mit "doesn't have a source distribution or
# wheel for the current platform" fehl. Der aktuell gepinnte Commit ist Tag v1.5.0 (letzter
# Tag vor dem Python-Bump). Tags sind kein Garant gegen sowas (Tag v1.5.4 hat denselben
# Bump schon mitgemacht) — nach --update-commit den Build TROTZDEM verifizieren.
#
# ACHTUNG: aktualisiert NUR den cosmos-transfer2.5-Commit-Pin. Der zweite Pin im Dockerfile
# (GROOT_FORK_COMMIT, fuer den vendorten v3->v2.1-Konverter) muss manuell mit dem Pin in
# Training/Dockerfile Schritt gehalten werden — siehe Kommentar dort im Dockerfile.
#
# Voraussetzungen:
#   docker login   (einmalig; Token wird in ~/.docker/config.json gespeichert)
#   git (im PATH)
#
# Umgebungsvariablen (optional, vor dem Aufruf setzen):
#   DOCKER_IMAGE     (default: lucam03/projekt-humanoider-roboter-augmentation)
#   COSMOS_REPO      (default: https://github.com/nvidia-cosmos/cosmos-transfer2.5.git)
#   COSMOS_TAG_GLOB  (default: v*)

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
DOCKER_IMAGE="${DOCKER_IMAGE:-lucam03/projekt-humanoider-roboter-augmentation}"
COSMOS_REPO="${COSMOS_REPO:-https://github.com/nvidia-cosmos/cosmos-transfer2.5.git}"
COSMOS_TAG_GLOB="${COSMOS_TAG_GLOB:-v*}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="$SCRIPT_DIR/Dockerfile"

if [[ ! -f "$DOCKERFILE" ]]; then
    exit_fatal "Dockerfile nicht gefunden: $DOCKERFILE"
fi

# ── Banner ───────────────────────────────────────────────────────────────────
echo ""
echo "${C_MAGENTA}╔══════════════════════════════════════════════════════════════════╗${C_RESET}"
echo "${C_MAGENTA}║   Cosmos-Transfer2.5 Augmentierung — Docker-Image Update          ║${C_RESET}"
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
log "Schritt 2/4 — cosmos-transfer2.5-Commit pruefen"

# Direkt an der ARG-Zeile extrahieren, nicht am `checkout`-Aufruf: die RUN-Zeile
# referenziert nur "$COSMOS_COMMIT" (eine Variable), der tatsaechliche 40-Zeichen-Hex
# steht im ARG-Default. Name ist eindeutig genug, um GROOT_FORK_COMMIT nicht zu treffen.
CURRENT_COMMIT="$(grep -oE '^ARG COSMOS_COMMIT=[0-9a-f]{40}' "$DOCKERFILE" | grep -oE '[0-9a-f]{40}')"

if [[ -z "$CURRENT_COMMIT" ]]; then
    warn "Kein gepinnter cosmos-transfer2.5-Commit im Dockerfile gefunden — ueberspringe Commit-Check."
else
    ok "Aktuell gepinnter Commit: ${CURRENT_COMMIT:0:12}..."

    # Bewusst der neueste TAG (nicht main's beweglicher HEAD): main hat .python-version
    # zwischenzeitlich von 3.10 auf 3.13 angehoben, waehrend der custom flash-attn-Wheel-
    # Index noch nur cp310 liefert — ein main-Commit kann jederzeit wieder in genau diese
    # Inkonsistenz laufen. Tags sind kein Garant gegen sowas (s. v1.5.4), aber naeher an
    # einem getesteten Release-Stand. Nach --update-commit den Build TROTZDEM verifizieren.
    LATEST_TAG_LINE="$(git ls-remote --tags --sort=-v:refname "$COSMOS_REPO" "$COSMOS_TAG_GLOB" 2>/dev/null | head -n1)"
    LATEST_COMMIT="$(awk '{print $1}' <<<"$LATEST_TAG_LINE")"
    LATEST_TAG="$(awk '{print $2}' <<<"$LATEST_TAG_LINE" | sed 's#refs/tags/##')"

    if [[ -z "$LATEST_COMMIT" ]]; then
        warn "Konnte Remote-Tag nicht ermitteln (kein Netzwerk, kein Tag passend zu '$COSMOS_TAG_GLOB', oder Repo nicht erreichbar?)."
    elif [[ "$LATEST_COMMIT" == "$CURRENT_COMMIT" ]]; then
        ok "Dockerfile ist bereits auf dem neuesten Tag ($LATEST_TAG, ${LATEST_COMMIT:0:12}...)."
    else
        warn "Neuerer Tag verfuegbar: $LATEST_TAG"
        warn "  Aktuell: ${CURRENT_COMMIT:0:12}..."
        warn "  Neu:     ${LATEST_COMMIT:0:12}... ($LATEST_TAG)"

        if [[ $UPDATE_COMMIT -eq 1 ]]; then
            log "Aktualisiere Dockerfile (--update-commit) ..."
            if [[ $DRY_RUN -eq 0 ]]; then
                sed -i "s/$CURRENT_COMMIT/$LATEST_COMMIT/g" "$DOCKERFILE"
                ok "Dockerfile aktualisiert: $CURRENT_COMMIT -> $LATEST_COMMIT ($LATEST_TAG)"
                warn "Build danach UNBEDINGT verifizieren — ein neuerer Tag kann eigene,"
                warn "neue Inkompatibilitaeten mitbringen (siehe Kommentar im Dockerfile)."
            else
                echo "${C_YELLOW}[dry-run] Ersetze '$CURRENT_COMMIT' -> '$LATEST_COMMIT' in Dockerfile${C_RESET}"
            fi
            CURRENT_COMMIT="$LATEST_COMMIT"
        else
            warn "Dockerfile wird NICHT aktualisiert (kein --update-commit)."
            warn "Zum Aktualisieren: ./update_image.sh --update-commit"
        fi
    fi
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 3 — Docker-Image bauen
# ══════════════════════════════════════════════════════════════════════════════
log "Schritt 3/4 — Docker-Image bauen"

BUILD_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
IMAGE_LATEST="${DOCKER_IMAGE}:latest"
IMAGE_DATED="${DOCKER_IMAGE}:${BUILD_TIMESTAMP}"

BUILD_CMD=(docker build --platform linux/amd64)
if [[ $NO_CACHE -eq 1 ]]; then BUILD_CMD+=(--no-cache); fi
BUILD_CMD+=(-t "$IMAGE_LATEST" -t "$IMAGE_DATED" "$SCRIPT_DIR")

log "Baue: ${BUILD_CMD[*]}"
if [[ $NO_CACHE -eq 1 ]]; then warn "Build ohne Cache — kann sehr lange dauern (Cosmos-Abhaengigkeiten)."; fi
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
if [[ -n "$CURRENT_COMMIT" ]]; then
    printf "    %-20s %s\n" "cosmos-Commit:" "${CURRENT_COMMIT:0:12}..."
fi
echo ""
echo "  Naechste Schritte:"
echo "    * Testen:  ./setup_and_augment_DockerHub-pull.sh"
echo "    * Ziehen:  docker pull $IMAGE_LATEST"
echo ""
