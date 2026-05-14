#!/usr/bin/env bash
# setup_and_train_DockerHub-pull.sh
#
# Vollstaendiges Setup-Skript fuer das GR00T N1.6 Fine-tuning Projekt.
# Dieses Skript laeuft auf dem HOST (nicht im Container) und uebernimmt:
#
#   1. Voraussetzungen pruefen  (Docker, NVIDIA, Git, Git LFS)
#   2. Repository klonen        (mit Submodulen)
#   3. Docker-Image laden       (von Docker Hub: lucam03/projekt-humanoider-roboter:latest)
#   4. HuggingFace-Login
#   5. Modell & Datensatz laden (~25 GB)
#   6. Datensatz konvertieren   (LeRobot v3.0 -> v2.1)
#   7. Fine-tuning starten
#
# Verwendung:
#   ./setup_and_train_DockerHub-pull.sh                    # Interaktiv, alle Schritte
#   ./setup_and_train_DockerHub-pull.sh --skip-clone       # Repo existiert bereits
#   ./setup_and_train_DockerHub-pull.sh --skip-pull        # Image bereits vorhanden
#   ./setup_and_train_DockerHub-pull.sh --skip-download    # Daten bereits vorhanden
#   ./setup_and_train_DockerHub-pull.sh --only-train       # Nur Training starten
#   ./setup_and_train_DockerHub-pull.sh --dry-run          # Befehle anzeigen, nichts ausfuehren
#
# Umgebungsvariablen (optional, vor dem Aufruf setzen):
#   export HF_TOKEN="hf_..."        HuggingFace-Token
#   export WANDB_API_KEY="..."      WandB-Token
#   export REPO_DIR="/pfad/..."     Zielverzeichnis (Standard: ./phr)
#   export MAX_STEPS="30000"
#   export GLOBAL_BATCH_SIZE="8"
#   export NUM_GPUS="1"

set -euo pipefail

# ── Flags ─────────────────────────────────────────────────────────────────────
SKIP_CLONE=false
SKIP_PULL=false
SKIP_DOWNLOAD=false
SKIP_CONVERT=false
ONLY_TRAIN=false
DRY_RUN=false

# ── Argument-Parsing ──────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --skip-clone)    SKIP_CLONE=true ;;
        --skip-pull)     SKIP_PULL=true ;;
        --skip-download) SKIP_DOWNLOAD=true ;;
        --skip-convert)  SKIP_CONVERT=true ;;
        --only-train)    ONLY_TRAIN=true ;;
        --dry-run)       DRY_RUN=true ;;
        --help|-h)
            sed -n '2,29p' "$0" | sed 's/^# \?//'
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
log()  { echo -e "\033[36m==> $1\033[0m"; }
ok()   { echo -e "\033[32m v  $1\033[0m"; }
warn() { echo -e "\033[33m  ! $1\033[0m"; }
err()  { echo -e "\033[31m!! $1\033[0m"; }
fatal() { err "$1"; exit 1; }

# ── Flags vererben ────────────────────────────────────────────────────────────
if $ONLY_TRAIN; then
    SKIP_CLONE=true
    SKIP_PULL=true
    SKIP_DOWNLOAD=true
    SKIP_CONVERT=true
fi

# ── Invoke-Cmd: fuehrt Befehl aus oder zeigt ihn im Dry-Run an ───────────────
invoke_cmd() {
    if $DRY_RUN; then
        echo -e "\033[33m[dry-run] $*\033[0m"
        return 0
    fi
    "$@"
    local code=$?
    if [ $code -ne 0 ]; then
        fatal "Befehl fehlgeschlagen (Exit-Code $code): $*"
    fi
}

# ── Konfiguration ─────────────────────────────────────────────────────────────
REPO_URL="https://github.com/Docboter/projektarbeit_humanoider_roboter.git"
REPO_BRANCH="training-luca"
REPO_DIR="${REPO_DIR:-$(pwd)/phr}"

DOCKER_HUB_IMAGE="lucam03/projekt-humanoider-roboter:latest"
LOCAL_IMAGE_NAME="projektarbeit-humanoider-roboter:latest"

MAX_STEPS="${MAX_STEPS:-30000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-gr00t-g1-dex3}"

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "\033[35m╔══════════════════════════════════════════════════════════════════╗\033[0m"
echo -e "\033[35m║   GR00T N1.6 Fine-tuning — Unitree G1 DEX3 — Setup & Training    ║\033[0m"
echo -e "\033[35m╚══════════════════════════════════════════════════════════════════╝\033[0m"
echo ""
if $DRY_RUN; then warn "DRY-RUN aktiv — es werden keine Befehle ausgefuehrt."; fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 1 — Voraussetzungen pruefen
# ══════════════════════════════════════════════════════════════════════════════
log "Schritt 1/7 — Voraussetzungen pruefen"

assert_command() {
    local name="$1"
    local hint="$2"
    if command -v "$name" &>/dev/null; then
        ok "$name gefunden: $(command -v "$name")"
    else
        fatal "$name nicht gefunden. Bitte installieren: $hint"
    fi
}

assert_command "docker"   "https://docs.docker.com/get-docker/"
assert_command "git"      "https://git-scm.com"
assert_command "git-lfs"  "https://git-lfs.com — danach: git lfs install"

if ! docker info &>/dev/null; then
    fatal "Docker-Daemon nicht erreichbar. Ist Docker gestartet? Ggf. 'sudo usermod -aG docker \$USER' noetig."
fi
ok "Docker-Daemon laeuft"

# NVIDIA Container Toolkit — leichtgewichtiger Test ohne Image-Pull
NVIDIA_OK=false
if docker run --rm --gpus all --entrypoint nvidia-smi \
       "nvidia/cuda:12.8.0-base-ubuntu22.04" -L &>/dev/null 2>&1; then
    NVIDIA_OK=true
fi

if $NVIDIA_OK; then
    ok "NVIDIA Container Toolkit funktioniert"
else
    warn "NVIDIA Container Toolkit nicht verfuegbar oder keine GPU erkannt."
    warn "Training ohne GPU nicht moeglich."
    warn "Installation: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    read -rp "  Trotzdem fortfahren? (nur fuer Tests ohne GPU) [j/N] " ans
    if [[ "${ans,,}" != "j" ]]; then fatal "Abgebrochen."; fi
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 2 — Repository klonen
# ══════════════════════════════════════════════════════════════════════════════
log "Schritt 2/7 — Repository klonen"

if $SKIP_CLONE; then
    warn "Clone uebersprungen (--skip-clone)."
    if [ ! -d "$REPO_DIR" ]; then
        fatal "REPO_DIR existiert nicht: $REPO_DIR"
    fi
elif [ -d "$REPO_DIR/.git" ]; then
    warn "Verzeichnis existiert bereits: $REPO_DIR"
    warn "Submodule werden aktualisiert statt neu geklont."
    invoke_cmd git -C "$REPO_DIR" fetch origin
    invoke_cmd git -C "$REPO_DIR" checkout "$REPO_BRANCH"
    invoke_cmd git -C "$REPO_DIR" pull origin "$REPO_BRANCH"
    invoke_cmd git -C "$REPO_DIR" submodule update --init --recursive
else
    log "Klone $REPO_URL -> $REPO_DIR"
    git config --global core.longpaths true
    invoke_cmd git clone --recurse-submodules --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
fi

ok "Repository bereit: $REPO_DIR"
echo ""

# Ab hier immer im Repo-Verzeichnis arbeiten
cd "$REPO_DIR"

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 3 — Docker-Image von Docker Hub laden
# ══════════════════════════════════════════════════════════════════════════════
log "Schritt 3/7 — Docker-Image von Docker Hub laden"

if $SKIP_PULL; then
    warn "Pull uebersprungen (--skip-pull)."
else
    log "Lade Image von Docker Hub: $DOCKER_HUB_IMAGE ..."
    invoke_cmd docker pull "$DOCKER_HUB_IMAGE"

    log "Tagge Image als '$LOCAL_IMAGE_NAME' fuer docker compose ..."
    invoke_cmd docker tag "$DOCKER_HUB_IMAGE" "$LOCAL_IMAGE_NAME"
fi

ok "Image bereit: $LOCAL_IMAGE_NAME"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 4 — HuggingFace-Login
# ══════════════════════════════════════════════════════════════════════════════
log "Schritt 4/7 — HuggingFace-Login"

if $SKIP_DOWNLOAD; then
    warn "Login-Check uebersprungen (Daten werden nicht heruntergeladen)."
else
    if [ -n "${HF_TOKEN:-}" ]; then
        ok "HF_TOKEN gesetzt — ueberspringe interaktiven Login."
    else
        warn "Kein HF_TOKEN gesetzt."
        warn "Du benoenigst einen HuggingFace-Account mit Zugriff auf:"
        warn "  * nvidia/GR00T-N1.6-3B    (Lizenz auf HF akzeptieren!)"
        warn "  * unitreerobotics/G1_Dex3_BlockStacking_Dataset"
        echo ""
        read -rp "  HuggingFace-Token eingeben (leer lassen fuer Login im Container): " hf_input
        if [ -n "$hf_input" ]; then
            export HF_TOKEN="$hf_input"
            ok "Token gespeichert (nur fuer diese Sitzung)."
        else
            warn "Kein Token eingegeben — du wirst spaeter im Container gefragt."
        fi
    fi
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 5 — Modell & Datensatz herunterladen (~25 GB)
# ══════════════════════════════════════════════════════════════════════════════
log "Schritt 5/7 — Modell & Datensatz herunterladen"

MODALITY_FILE="$REPO_DIR/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset/meta/modality.json"

if $SKIP_DOWNLOAD; then
    warn "Download uebersprungen (--skip-download)."
else
    if [ -f "$MODALITY_FILE" ]; then
        warn "Daten scheinen bereits vorhanden zu sein (modality.json gefunden)."
        read -rp "  Erneut herunterladen? [j/N] " ans
        if [[ "${ans,,}" != "j" ]]; then
            ok "Download uebersprungen."
            SKIP_DOWNLOAD=true
        fi
    fi

    if ! $SKIP_DOWNLOAD; then
        log "Download startet (~25 GB, kann lange dauern) ..."

        docker_args=("compose" "run" "--rm")
        if [ -n "${HF_TOKEN:-}" ]; then
            docker_args+=("-e" "HF_TOKEN=$HF_TOKEN")
        fi
        docker_args+=("groot-training" "bash" "/scripts/download_data.sh")

        invoke_cmd docker "${docker_args[@]}"
        ok "Download abgeschlossen."
    fi
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 6 — Datensatz konvertieren (LeRobot v3.0 -> v2.1)
# ══════════════════════════════════════════════════════════════════════════════
log "Schritt 6/7 — Datensatz konvertieren (v3.0 -> v2.1)"

if $SKIP_CONVERT; then
    warn "Konvertierung uebersprungen (--skip-convert)."
elif [ -f "$MODALITY_FILE" ]; then
    ok "modality.json bereits vorhanden — Konvertierung wird uebersprungen."
else
    log "Konvertiere Datensatz und kopiere modality.json ..."

    CONVERT_SCRIPT='set -e
cd /app/Groot-1.6
echo "==> Konvertiere LeRobot v3.0 -> v2.1 ..."
python scripts/lerobot_conversion/convert_v3_to_v2_standalone.py \
    --repo-id unitreerobotics/G1_Dex3_BlockStacking_Dataset \
    --root /data
echo "==> Kopiere modality_4cam.json ..."
cp examples/G1_DEX3/modality_4cam.json \
   /data/unitreerobotics/G1_Dex3_BlockStacking_Dataset/meta/modality.json
echo "==> Konvertierung fertig."'

    invoke_cmd docker compose run --rm groot-training bash -c "$CONVERT_SCRIPT"
    ok "Datensatz konvertiert."
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 7 — Fine-tuning starten
# ══════════════════════════════════════════════════════════════════════════════
log "Schritt 7/7 — Fine-tuning starten"

echo ""
echo "  Trainings-Konfiguration:"
printf "    %-25s %s\n" "MAX_STEPS"         "$MAX_STEPS"
printf "    %-25s %s\n" "GLOBAL_BATCH_SIZE" "$GLOBAL_BATCH_SIZE"
printf "    %-25s %s\n" "NUM_GPUS"          "$NUM_GPUS"
printf "    %-25s %s\n" "WANDB_PROJECT"     "$WANDB_PROJECT"
printf "    %-25s %s\n" "Logs (Host)"       "./data/logs/"
echo ""
warn "Hinweis: Mit 8 GB VRAM  -> GLOBAL_BATCH_SIZE=8  (oder kleiner bei OOM)"
warn "         Mit 16 GB VRAM -> GLOBAL_BATCH_SIZE=16"
echo ""

# WandB-Key
wandb_env_args=()
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb_env_args=("-e" "WANDB_API_KEY=$WANDB_API_KEY")
    ok "WANDB_API_KEY gesetzt."
else
    warn "Kein WANDB_API_KEY gesetzt — du wirst im Container nach dem Key gefragt."
    read -rp "  WandB API-Key eingeben (leer lassen fuer Login im Container): " wandb_input
    if [ -n "$wandb_input" ]; then
        wandb_env_args=("-e" "WANDB_API_KEY=$wandb_input")
        ok "WandB-Key gespeichert (nur fuer diese Sitzung)."
    fi
fi

echo ""
log "Starte Training ..."
log "Training-Logs landen in ./data/logs/ (Host-sichtbar ueber Volume-Mount)"
echo ""

train_cmd=(
    "docker" "compose" "run" "--rm"
    "-e" "MAX_STEPS=$MAX_STEPS"
    "-e" "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
    "-e" "NUM_GPUS=$NUM_GPUS"
    "-e" "WANDB_PROJECT=$WANDB_PROJECT"
    "-e" "USE_WANDB=1"
)
if [ ${#wandb_env_args[@]} -gt 0 ]; then
    train_cmd+=("${wandb_env_args[@]}")
fi
train_cmd+=("groot-training" "bash" "/scripts/run_finetuning.sh")

invoke_cmd "${train_cmd[@]}"

echo ""
ok "Alle Schritte abgeschlossen."
echo ""
echo "  Naechste Schritte:"
echo "    * Training live verfolgen:  https://wandb.ai -> Projekt '$WANDB_PROJECT'"
echo "    * Checkpoints pruefen:      ls ./data/g1_dex3_finetune/blockstacking/"
echo "    * Logs lesen:               tail -f ./data/logs/finetune-*.log"
echo ""
