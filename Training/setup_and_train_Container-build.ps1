#!/usr/bin/env pwsh
# setup_and_train.ps1
#
# Vollstaendiges Setup-Skript fuer das GR00T N1.6 Fine-tuning Projekt.
# Dieses Skript laeuft auf dem HOST (nicht im Container) und uebernimmt:
#
#   1. Voraussetzungen pruefen  (Docker, NVIDIA, Git, Git LFS)
#   2. Repository klonen        (mit Submodulen)
#   3. Docker-Image bauen
#   4. HuggingFace-Login
#   5. Modell & Datensatz laden (~25 GB)
#   6. Datensatz konvertieren   (LeRobot v3.0 -> v2.1)
#   7. Fine-tuning starten
#
# Verwendung:
#   .\setup_and_train.ps1                    # Interaktiv, alle Schritte
#   .\setup_and_train.ps1 -SkipClone        # Repo existiert bereits
#   .\setup_and_train.ps1 -SkipDownload     # Daten bereits vorhanden
#   .\setup_and_train.ps1 -OnlyTrain        # Nur Training starten
#   .\setup_and_train.ps1 -DryRun           # Befehle anzeigen, nichts ausfuehren
#
# Umgebungsvariablen (optional, vor dem Aufruf setzen):
#   $env:HF_TOKEN          = "hf_..."       HuggingFace-Token
#   $env:WANDB_API_KEY     = "..."          WandB-Token
#   $env:REPO_DIR          = "C:\pfad\..."  Zielverzeichnis (Standard: .\projektarbeit_humanoider_roboter)
#   $env:MAX_STEPS         = "30000"
#   $env:GLOBAL_BATCH_SIZE = "8"
#   $env:NUM_GPUS          = "1"

param(
    [switch]$SkipClone,
    [switch]$SkipBuild,
    [switch]$SkipDownload,
    [switch]$SkipConvert,
    [switch]$OnlyTrain,
    [switch]$DryRun,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# ── Logging ───────────────────────────────────────────────────────────────────
function Write-Log  { param([string]$Msg) Write-Host "==> $Msg" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Msg) Write-Host " v  $Msg" -ForegroundColor Green }
function Write-Warn { param([string]$Msg) Write-Host "  ! $Msg" -ForegroundColor Yellow }
function Write-Err  { param([string]$Msg) Write-Host "!! $Msg"  -ForegroundColor Red }
function Exit-Fatal { param([string]$Msg) Write-Err $Msg; exit 1 }

# ── Hilfe ─────────────────────────────────────────────────────────────────────
if ($Help) {
    Get-Content $MyInvocation.MyCommand.Path | Select-Object -Skip 2 -First 26 | ForEach-Object { $_ -replace '^# ?', '' }
    exit 0
}

# ── Flags vererben ────────────────────────────────────────────────────────────
if ($OnlyTrain) {
    $SkipClone    = $true
    $SkipBuild    = $true
    $SkipDownload = $true
    $SkipConvert  = $true
}

# ── Invoke-Cmd: fuehrt Befehl aus oder zeigt ihn im Dry-Run an ───────────────
function Invoke-Cmd {
    param([string[]]$Cmd)
    if ($DryRun) {
        Write-Host "[dry-run] $($Cmd -join ' ')" -ForegroundColor Yellow
        return
    }
    & $Cmd[0] $Cmd[1..($Cmd.Count - 1)]
    if ($LASTEXITCODE -ne 0) {
        Exit-Fatal "Befehl fehlgeschlagen (Exit-Code $LASTEXITCODE): $($Cmd -join ' ')"
    }
}

# ── Konfiguration ─────────────────────────────────────────────────────────────
$REPO_URL    = "https://github.com/Docboter/projektarbeit_humanoider_roboter.git"
$REPO_BRANCH = "training-luca"
$REPO_DIR    = if ($env:REPO_DIR) { $env:REPO_DIR }
               else { Join-Path (Get-Location).Path "phr" }

$MAX_STEPS         = if ($env:MAX_STEPS)         { $env:MAX_STEPS }         else { "30000" }
$GLOBAL_BATCH_SIZE = if ($env:GLOBAL_BATCH_SIZE) { $env:GLOBAL_BATCH_SIZE } else { "8" }
$NUM_GPUS          = if ($env:NUM_GPUS)          { $env:NUM_GPUS }          else { "1" }
$WANDB_PROJECT     = if ($env:WANDB_PROJECT)     { $env:WANDB_PROJECT }     else { "gr00t-g1-dex3" }

# ── Banner ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║   GR00T N1.6 Fine-tuning — Unitree G1 DEX3 — Setup & Training    ║" -ForegroundColor Magenta
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Magenta
Write-Host ""
if ($DryRun) { Write-Warn "DRY-RUN aktiv — es werden keine Befehle ausgefuehrt." }
Write-Host ""

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 1 — Voraussetzungen pruefen
# ══════════════════════════════════════════════════════════════════════════════
Write-Log "Schritt 1/7 — Voraussetzungen pruefen"

function Assert-Command {
    param([string]$Name, [string]$InstallHint)
    if (Get-Command $Name -ErrorAction SilentlyContinue) {
        Write-Ok "$Name gefunden: $((Get-Command $Name).Source)"
    } else {
        Exit-Fatal "$Name nicht gefunden. Bitte installieren: $InstallHint"
    }
}

Assert-Command "docker"   "https://docs.docker.com/get-docker/"
Assert-Command "git"      "https://git-scm.com"
Assert-Command "git-lfs"  "https://git-lfs.com — danach: git lfs install"

# Docker-Daemon erreichbar?
$dockerInfo = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Exit-Fatal "Docker-Daemon nicht erreichbar. Ist Docker Desktop gestartet?"
}
Write-Ok "Docker-Daemon laeuft"

# NVIDIA Container Toolkit — leichtgewichtiger Test ohne Image-Pull
$nvidiaOk = $false
try {
    $smiOut = docker run --rm --gpus all --entrypoint nvidia-smi `
        "nvidia/cuda:12.8.0-base-ubuntu22.04" -L 2>&1
    if ($LASTEXITCODE -eq 0) { $nvidiaOk = $true }
} catch { }

if ($nvidiaOk) {
    Write-Ok "NVIDIA Container Toolkit funktioniert"
} else {
    Write-Warn "NVIDIA Container Toolkit nicht verfuegbar oder keine GPU erkannt."
    Write-Warn "Training ohne GPU nicht moeglich."
    Write-Warn "Installation: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    $ans = Read-Host "  Trotzdem fortfahren? (nur fuer Tests ohne GPU) [j/N]"
    if ($ans.ToLower() -ne "j") { Exit-Fatal "Abgebrochen." }
}
Write-Host ""

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 2 — Repository klonen
# ══════════════════════════════════════════════════════════════════════════════
Write-Log "Schritt 2/7 — Repository klonen"

if ($SkipClone) {
    Write-Warn "Clone uebersprungen (-SkipClone)."
    if (-not (Test-Path $REPO_DIR -PathType Container)) {
        Exit-Fatal "REPO_DIR existiert nicht: $REPO_DIR"
    }
} elseif (Test-Path (Join-Path $REPO_DIR ".git") -PathType Container) {
    Write-Warn "Verzeichnis existiert bereits: $REPO_DIR"
    Write-Warn "Submodule werden aktualisiert statt neu geklont."
    Invoke-Cmd @("git", "-C", $REPO_DIR, "fetch", "origin")
    Invoke-Cmd @("git", "-C", $REPO_DIR, "checkout", $REPO_BRANCH)
    Invoke-Cmd @("git", "-C", $REPO_DIR, "pull", "origin", $REPO_BRANCH)
    Invoke-Cmd @("git", "-C", $REPO_DIR, "submodule", "update", "--init", "--recursive")
} else {
    Write-Log "Klone $REPO_URL -> $REPO_DIR"
    git config --global core.longpaths true
    $env:GIT_CLONE_PROTECTION_ACTIVE = "false"
    Invoke-Cmd @("git", "clone", "--recurse-submodules", "--branch", $REPO_BRANCH, $REPO_URL, $REPO_DIR)
    Remove-Item Env:\GIT_CLONE_PROTECTION_ACTIVE -ErrorAction SilentlyContinue
}

Write-Ok "Repository bereit: $REPO_DIR"
Write-Host ""

# Ab hier immer im Repo-Verzeichnis arbeiten (CWD = Repo-Root, damit die
# .\data-Pfade unten stimmen). Die docker-compose.yml liegt seit dem Umbau
# unter Training\ — ueber COMPOSE_FILE finden alle `docker compose`-Aufrufe sie.
Set-Location $REPO_DIR
$env:COMPOSE_FILE = "Training/docker-compose.yml"

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 3 — Docker-Image bauen
# ══════════════════════════════════════════════════════════════════════════════
Write-Log "Schritt 3/7 — Docker-Image bauen"

if ($SkipBuild) {
    Write-Warn "Build uebersprungen (-SkipBuild)."
} else {
    Write-Log "Baue Image (beim ersten Mal ~30 Minuten wegen flash-attn) ..."
    Invoke-Cmd @("docker", "compose", "build")
}

Write-Ok "Image bereit"
Write-Host ""

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 4 — HuggingFace-Login
# ══════════════════════════════════════════════════════════════════════════════
Write-Log "Schritt 4/7 — HuggingFace-Login"

if ($SkipDownload) {
    Write-Warn "Login-Check uebersprungen (Daten werden nicht heruntergeladen)."
} else {
    if ($env:HF_TOKEN) {
        Write-Ok "HF_TOKEN gesetzt — ueberspringe interaktiven Login."
    } else {
        Write-Warn "Kein HF_TOKEN gesetzt."
        Write-Warn "Du benoenigst einen HuggingFace-Account mit Zugriff auf:"
        Write-Warn "  * nvidia/GR00T-N1.6-3B    (Lizenz auf HF akzeptieren!)"
        Write-Warn "  * unitreerobotics/G1_Dex3_BlockStacking_Dataset"
        Write-Host ""
        $hfInput = Read-Host "  HuggingFace-Token eingeben (leer lassen fuer Login im Container)"
        if ($hfInput) {
            $env:HF_TOKEN = $hfInput
            Write-Ok "Token gespeichert (nur fuer diese Sitzung)."
        } else {
            Write-Warn "Kein Token eingegeben — du wirst spaeter im Container gefragt."
        }
    }
}
Write-Host ""

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 5 — Modell & Datensatz herunterladen (~25 GB)
# ══════════════════════════════════════════════════════════════════════════════
Write-Log "Schritt 5/7 — Modell & Datensatz herunterladen"

$modalityFile = Join-Path $REPO_DIR "data\unitreerobotics\G1_Dex3_BlockStacking_Dataset\meta\modality.json"

if ($SkipDownload) {
    Write-Warn "Download uebersprungen (-SkipDownload)."
} else {
    if (Test-Path $modalityFile -PathType Leaf) {
        Write-Warn "Daten scheinen bereits vorhanden zu sein (modality.json gefunden)."
        $ans = Read-Host "  Erneut herunterladen? [j/N]"
        if ($ans.ToLower() -ne "j") {
            Write-Ok "Download uebersprungen."
            $SkipDownload = $true
        }
    }

    if (-not $SkipDownload) {
        Write-Log "Download startet (~25 GB, kann lange dauern) ..."

        $dockerArgs = @("compose", "run", "--rm")
        if ($env:HF_TOKEN) { $dockerArgs += @("-e", "HF_TOKEN=$env:HF_TOKEN") }
        $dockerArgs += @("groot-training", "bash", "/scripts/download_data.sh")

        Invoke-Cmd (@("docker") + $dockerArgs)
        Write-Ok "Download abgeschlossen."
    }
}
Write-Host ""

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 6 — Datensatz konvertieren (LeRobot v3.0 -> v2.1)
# ══════════════════════════════════════════════════════════════════════════════
Write-Log "Schritt 6/7 — Datensatz konvertieren (v3.0 -> v2.1)"

if ($SkipConvert) {
    Write-Warn "Konvertierung uebersprungen (-SkipConvert)."
} elseif (Test-Path $modalityFile -PathType Leaf) {
    Write-Ok "modality.json bereits vorhanden — Konvertierung wird uebersprungen."
} else {
    Write-Log "Konvertiere Datensatz und kopiere modality.json ..."

    $convertScript = @"
set -e
cd /app/Groot-1.6
echo '==> Konvertiere LeRobot v3.0 -> v2.1 ...'
python scripts/lerobot_conversion/convert_v3_to_v2_standalone.py \
    --repo-id unitreerobotics/G1_Dex3_BlockStacking_Dataset \
    --root /data
echo '==> Kopiere modality_4cam.json ...'
cp examples/G1_DEX3/modality_4cam.json \
   /data/unitreerobotics/G1_Dex3_BlockStacking_Dataset/meta/modality.json
echo '==> Konvertierung fertig.'
"@

    Invoke-Cmd @("docker", "compose", "run", "--rm", "groot-training", "bash", "-c", $convertScript)
    Write-Ok "Datensatz konvertiert."
}
Write-Host ""

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 7 — Fine-tuning starten
# ══════════════════════════════════════════════════════════════════════════════
Write-Log "Schritt 7/7 — Fine-tuning starten"

Write-Host ""
Write-Host "  Trainings-Konfiguration:"
Write-Host ("    {0,-25} {1}" -f "MAX_STEPS",         $MAX_STEPS)
Write-Host ("    {0,-25} {1}" -f "GLOBAL_BATCH_SIZE", $GLOBAL_BATCH_SIZE)
Write-Host ("    {0,-25} {1}" -f "NUM_GPUS",          $NUM_GPUS)
Write-Host ("    {0,-25} {1}" -f "WANDB_PROJECT",     $WANDB_PROJECT)
Write-Host ("    {0,-25} {1}" -f "Logs (Host)",       ".\data\logs\")
Write-Host ""
Write-Warn "Hinweis: Mit 8 GB VRAM  -> GLOBAL_BATCH_SIZE=8  (oder kleiner bei OOM)"
Write-Warn "         Mit 16 GB VRAM -> GLOBAL_BATCH_SIZE=16"
Write-Host ""

# WandB-Key
$wandbEnvArgs = @()
if ($env:WANDB_API_KEY) {
    $wandbEnvArgs = @("-e", "WANDB_API_KEY=$env:WANDB_API_KEY")
    Write-Ok "WANDB_API_KEY gesetzt."
} else {
    Write-Warn "Kein WANDB_API_KEY gesetzt — du wirst im Container nach dem Key gefragt."
    $wandbInput = Read-Host "  WandB API-Key eingeben (leer lassen fuer Login im Container)"
    if ($wandbInput) {
        $wandbEnvArgs = @("-e", "WANDB_API_KEY=$wandbInput")
        Write-Ok "WandB-Key gespeichert (nur fuer diese Sitzung)."
    }
}

Write-Host ""
Write-Log "Starte Training ..."
Write-Log "Training-Logs landen in .\data\logs\ (Host-sichtbar ueber Volume-Mount)"
Write-Host ""

$trainCmd = @(
    "docker", "compose", "run", "--rm",
    "-e", "MAX_STEPS=$MAX_STEPS",
    "-e", "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE",
    "-e", "NUM_GPUS=$NUM_GPUS",
    "-e", "WANDB_PROJECT=$WANDB_PROJECT",
    "-e", "USE_WANDB=1"
)
if ($wandbEnvArgs) { $trainCmd += $wandbEnvArgs }
$trainCmd += @("groot-training", "bash", "/scripts/run_finetuning.sh")

Invoke-Cmd $trainCmd

Write-Host ""
Write-Ok "Alle Schritte abgeschlossen."
Write-Host ""
Write-Host "  Naechste Schritte:"
Write-Host "    * Training live verfolgen:  https://wandb.ai -> Projekt '$WANDB_PROJECT'"
Write-Host "    * Checkpoints pruefen:      Get-ChildItem .\data\g1_dex3_finetune\blockstacking\"
Write-Host "    * Logs lesen:               Get-Content .\data\logs\finetune-*.log -Wait"
Write-Host ""
