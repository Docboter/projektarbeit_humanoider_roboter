#!/usr/bin/env bash
# TL;DR: Host-Setup-Skript — baut das Image lokal und startet danach Download, Konvertierung, Training.
# setup_and_train.sh
#
# Vollständiges Setup-Skript für das GR00T N1.6 Fine-tuning Projekt.
# Dieses Skript läuft auf dem HOST (nicht im Container) und übernimmt:
#
#   1. Voraussetzungen prüfen  (Docker, NVIDIA, Git, Git LFS)
#   2. Repository klonen        (mit Submodulen)
#   3. Docker-Image bauen
#   4. HuggingFace-Login
#   5. Modell & Datensatz laden (~25 GB)
#   6. Datensatz konvertieren   (LeRobot v3.0 → v2.1)
#   7. Fine-tuning starten
#
# Verwendung:
#   bash setup_and_train.sh                          # Interaktiv, alle Schritte
#   bash setup_and_train.sh --skip-clone             # Repo existiert bereits
#   bash setup_and_train.sh --skip-download          # Daten bereits vorhanden
#   bash setup_and_train.sh --only-train             # Nur Training starten
#   bash setup_and_train.sh --dry-run                # Befehle anzeigen, nichts ausführen
#
# Umgebungsvariablen (optional, vor dem Aufruf setzen):
#   HF_TOKEN=hf_...          HuggingFace-Token (überspringt interaktiven Login)
#   WANDB_API_KEY=...        WandB-Token        (überspringt interaktiven Login)
#   REPO_DIR=/pfad/zum/repo  Zielverzeichnis für den Clone (Standard: ./projektarbeit)
#   MAX_STEPS=30000
#   GLOBAL_BATCH_SIZE=8
#   NUM_GPUS=1

set -euo pipefail

# ── Farben ────────────────────────────────────────────────────────────────────
RED='\033[1;31m'; GREEN='\033[1;32m'; BLUE='\033[1;34m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { printf "${BLUE}==>${NC} %s\n" "$*"; }
ok()   { printf "${GREEN} ✓${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}  !${NC} %s\n" "$*"; }
err()  { printf "${RED}!! ${NC}%s\n" "$*" >&2; }
die()  { err "$*"; exit 1; }

# ── Flags ─────────────────────────────────────────────────────────────────────
SKIP_CLONE=0
SKIP_BUILD=0
SKIP_DOWNLOAD=0
SKIP_CONVERT=0
ONLY_TRAIN=0
DRY_RUN=0

for arg in "$@"; do
    case "$arg" in
        --skip-clone)    SKIP_CLONE=1 ;;
        --skip-build)    SKIP_BUILD=1 ;;
        --skip-download) SKIP_DOWNLOAD=1 ;;
        --skip-convert)  SKIP_CONVERT=1 ;;
        --only-train)    ONLY_TRAIN=1; SKIP_CLONE=1; SKIP_BUILD=1; SKIP_DOWNLOAD=1; SKIP_CONVERT=1 ;;
        --dry-run)       DRY_RUN=1 ;;
        --help|-h)
            awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print; next } if (/^[[:space:]]*$/) { print ""; next } exit }' "$0"
            exit 0 ;;
        *) warn "Unbekannter Parameter: $arg — ignoriert" ;;
    esac
done

run() {
    if [[ "$DRY_RUN" == "1" ]]; then
        printf "${YELLOW}[dry-run]${NC} %s\n" "$*"
    else
        "$@"
    fi
}

# ── Konfiguration ─────────────────────────────────────────────────────────────
REPO_URL="https://github.com/Docboter/projektarbeit_humanoider_roboter.git"
REPO_BRANCH="training-luca"
REPO_DIR="${REPO_DIR:-$(pwd)/phr}"

MAX_STEPS="${MAX_STEPS:-30000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-gr00t-g1-dex3}"

# ── Banner ────────────────────────────────────────────────────────────────────
echo
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   GR00T N1.6 Fine-tuning — Unitree G1 DEX3 — Setup & Training   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo
[[ "$DRY_RUN" == "1" ]] && warn "DRY-RUN aktiv — es werden keine Befehle ausgeführt."
echo

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 1 — Voraussetzungen prüfen
# ══════════════════════════════════════════════════════════════════════════════
log "Schritt 1/7 — Voraussetzungen prüfen"

check_cmd() {
    if command -v "$1" &>/dev/null; then
        ok "$1 gefunden: $(command -v "$1")"
    else
        die "$1 nicht gefunden. Bitte installieren: $2"
    fi
}

check_cmd docker        "https://docs.docker.com/get-docker/"
check_cmd git           "https://git-scm.com"
check_cmd git-lfs       "https://git-lfs.com  →  dann: git lfs install"

# Docker-Daemon erreichbar?
if ! docker info &>/dev/null; then
    die "Docker-Daemon nicht erreichbar. Ist Docker Desktop gestartet?"
fi
ok "Docker-Daemon läuft"

# NVIDIA Container Toolkit
if docker run --rm --gpus all --entrypoint nvidia-smi nvidia/cuda:12.8.0-base-ubuntu22.04 -L &>/dev/null 2>&1; then
    ok "NVIDIA Container Toolkit funktioniert"
else
    warn "NVIDIA Container Toolkit nicht verfügbar oder keine GPU."
    warn "Training ohne GPU nicht möglich."
    warn "Installation: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    read -rp "Trotzdem fortfahren? (nur für Tests ohne GPU) [j/N] " ans
    [[ "${ans,,}" == "j" ]] || die "Abgebrochen."
fi
echo

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 2 — Repository klonen
# ══════════════════════════════════════════════════════════════════════════════
log "Schritt 2/7 — Repository klonen"

if [[ "$SKIP_CLONE" == "1" ]]; then
    warn "Clone übersprungen (--skip-clone)."
    [[ -d "$REPO_DIR" ]] || die "REPO_DIR existiert nicht: $REPO_DIR"
else
    if [[ -d "$REPO_DIR/.git" ]]; then
        warn "Verzeichnis existiert bereits: $REPO_DIR"
        warn "Submodule werden aktualisiert statt neu geklont."
        run git -C "$REPO_DIR" fetch origin
        run git -C "$REPO_DIR" checkout "$REPO_BRANCH"
        run git -C "$REPO_DIR" pull origin "$REPO_BRANCH"
        run git -C "$REPO_DIR" submodule update --init --recursive
    else
        log "Klone $REPO_URL  →  $REPO_DIR"
        git config --global core.longpaths true
        GIT_CLONE_PROTECTION_ACTIVE=false run git clone --recurse-submodules --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
    fi
fi

ok "Repository bereit: $REPO_DIR"
echo

# Ab hier immer im Repo-Verzeichnis arbeiten (CWD = Repo-Root, damit die
# ./data-Pfade unten stimmen). Die docker-compose.yml liegt seit dem Umbau
# unter Training/ — über COMPOSE_FILE finden alle `docker compose`-Aufrufe sie.
cd "$REPO_DIR"
export COMPOSE_FILE="Training/docker-compose.yml"

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 3 — Docker-Image bauen
# ══════════════════════════════════════════════════════════════════════════════
log "Schritt 3/7 — Docker-Image bauen"

if [[ "$SKIP_BUILD" == "1" ]]; then
    warn "Build übersprungen (--skip-build)."
else
    log "Baue Image (beim ersten Mal ~30 Minuten wegen flash-attn) …"
    run docker compose build
fi

ok "Image bereit"
echo

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 4 — HuggingFace-Login
# ══════════════════════════════════════════════════════════════════════════════
log "Schritt 4/7 — HuggingFace-Login"

if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    warn "Login-Check übersprungen (Daten werden nicht heruntergeladen)."
else
    HF_LOGIN_OK=0

    # Token über Env-Variable?
    if [[ -n "${HF_TOKEN:-}" ]]; then
        ok "HF_TOKEN gesetzt — überspringe interaktiven Login."
        HF_LOGIN_OK=1
    fi

    if [[ "$HF_LOGIN_OK" == "0" ]]; then
        warn "Kein HF_TOKEN gesetzt."
        warn "Du benötigst einen HuggingFace-Account mit Zugriff auf:"
        warn "  • nvidia/GR00T-N1.6-3B    (Lizenz auf HF akzeptieren!)"
        warn "  • unitreerobotics/G1_Dex3_BlockStacking_Dataset"
        echo
        echo -n "  HuggingFace-Token jetzt eingeben (leer lassen für interaktiven Login im Container): "
        read -rs HF_TOKEN_INPUT
        echo
        if [[ -n "$HF_TOKEN_INPUT" ]]; then
            export HF_TOKEN="$HF_TOKEN_INPUT"
            ok "Token gespeichert (nur für diese Sitzung)."
        else
            warn "Kein Token eingegeben — du wirst später im Container nach dem Token gefragt."
        fi
    fi
fi
echo

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 5 — Modell & Datensatz herunterladen (~25 GB)
# ══════════════════════════════════════════════════════════════════════════════
log "Schritt 5/7 — Modell & Datensatz herunterladen"

if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    warn "Download übersprungen (--skip-download)."
else
    MODALITY_FILE="./data/unitreerobotics/G1_Dex3_BlockStacking_Dataset/meta/modality.json"
    if [[ -f "$MODALITY_FILE" ]]; then
        warn "Daten scheinen bereits vorhanden zu sein (modality.json gefunden)."
        read -rp "Erneut herunterladen? [j/N] " ans
        [[ "${ans,,}" == "j" ]] || { ok "Download übersprungen."; SKIP_DOWNLOAD=1; }
    fi

    if [[ "$SKIP_DOWNLOAD" == "0" ]]; then
        log "Download startet (~25 GB, kann lange dauern) …"

        HF_ENV_FLAG=""
        [[ -n "${HF_TOKEN:-}" ]] && HF_ENV_FLAG="-e HF_TOKEN=${HF_TOKEN}"

        if [[ "$DRY_RUN" == "1" ]]; then
            warn "[dry-run] docker compose run --rm ${HF_ENV_FLAG} groot-training bash /scripts/download_data.sh"
        else
            # shellcheck disable=SC2086
            docker compose run --rm ${HF_ENV_FLAG} groot-training bash /scripts/download_data.sh
        fi
        ok "Download abgeschlossen."
    fi
fi
echo

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 6 — Datensatz konvertieren (LeRobot v3.0 → v2.1)
# ══════════════════════════════════════════════════════════════════════════════
log "Schritt 6/7 — Datensatz konvertieren (v3.0 → v2.1)"

MODALITY_FILE="./data/unitreerobotics/G1_Dex3_BlockStacking_Dataset/meta/modality.json"

if [[ "$SKIP_CONVERT" == "1" ]]; then
    warn "Konvertierung übersprungen (--skip-convert)."
elif [[ -f "$MODALITY_FILE" ]]; then
    ok "modality.json bereits vorhanden — Konvertierung wird übersprungen."
else
    log "Konvertiere Datensatz und kopiere modality.json …"
    run docker compose run --rm groot-training bash -c "
        set -e
        cd /app/Groot-1.6

        echo '==> Konvertiere LeRobot v3.0 → v2.1 …'
        python scripts/lerobot_conversion/convert_v3_to_v2_standalone.py \
            --repo-id unitreerobotics/G1_Dex3_BlockStacking_Dataset \
            --root /data

        echo '==> Kopiere modality_4cam.json …'
        cp examples/G1_DEX3/modality_4cam.json \
           /data/unitreerobotics/G1_Dex3_BlockStacking_Dataset/meta/modality.json

        echo '==> Konvertierung fertig.'
    "
    ok "Datensatz konvertiert."
fi
echo

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 7 — Fine-tuning starten
# ══════════════════════════════════════════════════════════════════════════════
log "Schritt 7/7 — Fine-tuning starten"

echo
echo "  Trainings-Konfiguration:"
printf "    %-25s %s\n" "MAX_STEPS"          "$MAX_STEPS"
printf "    %-25s %s\n" "GLOBAL_BATCH_SIZE"  "$GLOBAL_BATCH_SIZE"
printf "    %-25s %s\n" "NUM_GPUS"           "$NUM_GPUS"
printf "    %-25s %s\n" "WANDB_PROJECT"      "$WANDB_PROJECT"
printf "    %-25s %s\n" "Logs (Host)"        "./data/logs/"
echo
warn "Hinweis: Standard GLOBAL_BATCH_SIZE=8 (sicher für ~40 GB VRAM)"
warn "         Mit 40+ GB VRAM → GLOBAL_BATCH_SIZE=8 oder höher"
echo

# WandB-Key
WANDB_ENV_FLAG=""
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_ENV_FLAG="-e WANDB_API_KEY=${WANDB_API_KEY}"
    ok "WANDB_API_KEY gesetzt."
else
    warn "Kein WANDB_API_KEY gesetzt — du wirst im Container nach dem Key gefragt."
    echo -n "  WandB API-Key jetzt eingeben (leer lassen für Login im Container): "
    read -rs WANDB_KEY_INPUT
    echo
    if [[ -n "$WANDB_KEY_INPUT" ]]; then
        WANDB_ENV_FLAG="-e WANDB_API_KEY=${WANDB_KEY_INPUT}"
        ok "WandB-Key gespeichert (nur für diese Sitzung)."
    fi
fi

echo
log "Starte Training …"
log "Training-Logs landen in ./data/logs/ (Host-sichtbar über Volume-Mount)"
echo

TRAIN_CMD=(
    docker compose run --rm
    -e "MAX_STEPS=${MAX_STEPS}"
    -e "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
    -e "NUM_GPUS=${NUM_GPUS}"
    -e "WANDB_PROJECT=${WANDB_PROJECT}"
    -e "USE_WANDB=1"
)

[[ -n "$WANDB_ENV_FLAG" ]] && TRAIN_CMD+=($WANDB_ENV_FLAG)

TRAIN_CMD+=(groot-training bash /scripts/run_finetuning.sh)

if [[ "$DRY_RUN" == "1" ]]; then
    warn "[dry-run] ${TRAIN_CMD[*]}"
else
    "${TRAIN_CMD[@]}"
fi

echo
ok "Alle Schritte abgeschlossen."
echo
echo "  Nächste Schritte:"
echo "    • Training live verfolgen:  https://wandb.ai → Projekt '${WANDB_PROJECT}'"
echo "    • Checkpoints prüfen:       ls ./data/g1_dex3_finetune/blockstacking/"
echo "    • Logs lesen:               tail -f ./data/logs/finetune-*.log"
echo
