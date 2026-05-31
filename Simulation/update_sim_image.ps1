#!/usr/bin/env pwsh
# update_sim_image.ps1
#
# Baut das Sim-Docker-Image und pusht es nach Docker Hub.
#
# Verwendung:
#   .\update_sim_image.ps1              # KISSKI-Image bauen + pushen
#   .\update_sim_image.ps1 -VastAI     # vast.ai-Image bauen + pushen
#   .\update_sim_image.ps1 -NoCache    # Build ohne Docker-Cache
#   .\update_sim_image.ps1 -SkipPush   # Nur bauen, nicht pushen
#   .\update_sim_image.ps1 -DryRun     # Befehle anzeigen, nichts ausfuehren
#
# Voraussetzungen:
#   docker login nvcr.io   (Username: $oauthtoken, Password: NGC-API-Key)
#   docker login           (Docker Hub, fuer den Push)
#
# Umgebungsvariablen (optional):
#   $env:DOCKER_IMAGE   (default abhaengig von -VastAI)

param(
    [switch]$NoCache,
    [switch]$SkipPush,
    [switch]$DryRun,
    [switch]$VastAI,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

function Write-Log  { param([string]$Msg) Write-Host "==> $Msg" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Msg) Write-Host " v  $Msg" -ForegroundColor Green }
function Write-Warn { param([string]$Msg) Write-Host "  ! $Msg" -ForegroundColor Yellow }
function Write-Err  { param([string]$Msg) Write-Host "!! $Msg"  -ForegroundColor Red }
function Exit-Fatal { param([string]$Msg) Write-Err $Msg; exit 1 }

if ($Help) {
    Get-Content $MyInvocation.MyCommand.Path | Select-Object -Skip 2 -First 20 | ForEach-Object { $_ -replace '^# ?', '' }
    exit 0
}

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
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($VastAI) {
    $DockerImage = if ($env:DOCKER_IMAGE) { $env:DOCKER_IMAGE } else { "lucam03/projekt-humanoider-roboter-sim-vastai" }
    $Dockerfile  = Join-Path $ScriptDir "Dockerfile.vastai"
    $Target      = "vast.ai (Isaac Sim + GR00T, Dockerfile.vastai)"
} else {
    $DockerImage = if ($env:DOCKER_IMAGE) { $env:DOCKER_IMAGE } else { "lucam03/projekt-humanoider-roboter-sim" }
    $Dockerfile  = Join-Path $ScriptDir "Dockerfile"
    $Target      = "KISSKI/Apptainer (nur Sim-Client)"
}

if (-not (Test-Path $Dockerfile -PathType Leaf)) {
    Exit-Fatal "Dockerfile nicht gefunden: $Dockerfile"
}

# ── Banner ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "GR00T N1.6 -- Sim-Image Update ($Target)" -ForegroundColor Magenta
Write-Host ""
if ($DryRun) { Write-Warn "DRY-RUN aktiv -- es werden keine Befehle ausgefuehrt." }
Write-Warn "Basis-Image: nvcr.io/nvidia/isaac-lab:2.3.2 (~20-30 GB)"
Write-Warn "Bitte 'docker login nvcr.io' vorab ausfuehren (Username: oauthtoken)."
if ($VastAI) {
    Write-Warn "GPU-Anforderung: >=24 GB VRAM, Ampere+, RT-Cores (RTX 3090/4090/A6000/L40)"
}
Write-Host ""

# ── Schritt 1: Voraussetzungen pruefen ────────────────────────────────────────
Write-Log "Schritt 1/3 -- Voraussetzungen pruefen"

if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
    Exit-Fatal "docker nicht gefunden. Installation: https://docs.docker.com/get-docker/"
}
$dockerInfo = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Exit-Fatal "Docker-Daemon nicht erreichbar. Ist Docker Desktop gestartet?"
}
Write-Ok "Docker-Daemon laeuft"

if (-not $SkipPush -and -not $DryRun) {
    $loginCheck = docker info 2>$null | Select-String "Username"
    if (-not $loginCheck) {
        Write-Warn "Nicht bei Docker Hub angemeldet."
        Write-Log "Starte 'docker login' ..."
        docker login
        if ($LASTEXITCODE -ne 0) { Exit-Fatal "Docker-Login fehlgeschlagen." }
    } else {
        Write-Ok "Bereits bei Docker Hub angemeldet"
    }
}
Write-Host ""

# ── Schritt 2: Image bauen ────────────────────────────────────────────────────
Write-Log "Schritt 2/3 -- Image bauen"

$BuildTimestamp = (Get-Date -Format "yyyyMMdd-HHmmss")
$ImageLatest    = "${DockerImage}:latest"
$ImageDated     = "${DockerImage}:${BuildTimestamp}"

$BuildCmd = @("docker", "build", "--platform", "linux/amd64", "-f", $Dockerfile)
if ($NoCache) { $BuildCmd += "--no-cache" }
$BuildCmd += @("-t", $ImageLatest, "-t", $ImageDated, $ScriptDir)

Write-Log "Baue: $($BuildCmd -join ' ')"
if ($NoCache) { Write-Warn "Build ohne Cache -- Isaac-Lab-Basis ist gross, kann 60+ Minuten dauern." }
Write-Host ""

Invoke-Cmd $BuildCmd

Write-Ok "Image gebaut: $ImageLatest"
Write-Ok "Image gebaut: $ImageDated"
Write-Host ""

# ── Schritt 3: Push ───────────────────────────────────────────────────────────
Write-Log "Schritt 3/3 -- Nach Docker Hub pushen"

if ($SkipPush) {
    Write-Warn "Push uebersprungen (-SkipPush)."
} else {
    Invoke-Cmd @("docker", "push", $ImageLatest)
    Invoke-Cmd @("docker", "push", $ImageDated)
    Write-Ok "Gepusht: $ImageLatest"
    Write-Ok "Gepusht: $ImageDated"
}
Write-Host ""

Write-Ok "Fertig."
Write-Host ""
Write-Host "  Zusammenfassung:"
Write-Host ("    {0,-20} {1}" -f "Image (latest):", $ImageLatest)
Write-Host ("    {0,-20} {1}" -f "Image (dated):",  $ImageDated)
Write-Host ""
Write-Host "  Naechste Schritte:"
if ($VastAI) {
    Write-Host "    1. Auf vast.ai starten:"
    Write-Host "         Image:          $ImageLatest"
    Write-Host "         GPU:            L40 / RTX 4090 (>=24 GB, Ampere+, RT-Cores)"
    Write-Host "         Docker Options: --ipc=host --shm-size=16g"
    Write-Host "         Env:            CHECKPOINT_PATH=/data/checkpoints/checkpoint-XXXX"
    Write-Host "                         HF_TOKEN=hf_..."
    Write-Host "    2. Checkpoint per Volume/SCP oder HF_CHECKPOINT_REPO bereitstellen"
    Write-Host "    3. Ergebnisse sichern (vor Destroy!):"
    Write-Host "         docker cp CONTAINER:/data/sim_results ./sim_results"
    Write-Host "         docker cp CONTAINER:/data/sim_videos  ./sim_videos"
    Write-Host "    Anleitung: Simulation/VASTAI_SIM_ANLEITUNG.md"
} else {
    Write-Host "    1. SIF auf KISSKI ziehen (Login-Node):"
    Write-Host "         module load apptainer"
    Write-Host "         apptainer pull .project/images/projekt-humanoider-roboter-sim.sif docker://$ImageLatest"
    Write-Host "    2. Phase-A-Test + Eval:"
    Write-Host "         Siehe Simulation/KISSKI_SIM_DESKTOP_ANLEITUNG.md"
    Write-Host "    3. Eval-Job einreichen: sbatch Simulation/kisski_sim_submit.sh"
}
Write-Host ""
