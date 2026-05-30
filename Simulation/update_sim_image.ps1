#!/usr/bin/env pwsh
# update_sim_image.ps1
#
# Baut das Sim-Client-Docker-Image und pusht es nach Docker Hub.
# Analog zu Training/update_image.ps1, aber ohne GR00T-Commit-Check.
#
# Verwendung:
#   .\update_sim_image.ps1                  # Build + Push
#   .\update_sim_image.ps1 -NoCache         # Build ohne Docker-Cache (langsam!)
#   .\update_sim_image.ps1 -SkipPush        # Nur bauen, nicht pushen
#   .\update_sim_image.ps1 -DryRun          # Befehle anzeigen, nichts ausfuehren
#
# Voraussetzungen:
#   docker login nvcr.io   (Username: $oauthtoken, Password: NGC-API-Key)
#   docker login           (Docker Hub, für den Push)
#
# Umgebungsvariablen (optional):
#   $env:DOCKER_IMAGE   (default: lucam03/projekt-humanoider-roboter-sim)
#
# Wichtig:
#   Das Basis-Image nvcr.io/nvidia/isaac-lab:2.3.2 ist ~20-30 GB.
#   Rechnet mit 50-80 GB freiem Plattenplatz und langer Download-Zeit.
#   Auf GPUs mit RT-Cores beschraenkt (kein A100/H100) — siehe SIM_GPU_COMPATIBILITY.md.

param(
    [switch]$NoCache,
    [switch]$SkipPush,
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
$DockerImage = if ($env:DOCKER_IMAGE) { $env:DOCKER_IMAGE } else { "lucam03/projekt-humanoider-roboter-sim" }
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$Dockerfile  = Join-Path $ScriptDir "Dockerfile"

if (-not (Test-Path $Dockerfile -PathType Leaf)) {
    Exit-Fatal "Dockerfile nicht gefunden: $Dockerfile"
}

# ── Banner ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║   GR00T N1.6 — Sim-Client-Image Update                           ║" -ForegroundColor Magenta
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Magenta
Write-Host ""
if ($DryRun) { Write-Warn "DRY-RUN aktiv — es werden keine Befehle ausgefuehrt." }
Write-Warn "Basis-Image: nvcr.io/nvidia/isaac-lab:2.3.2 (~20-30 GB)"
Write-Warn "Bitte docker login nvcr.io vorab ausfuehren (Username: `$oauthtoken)."
Write-Host ""

# ── Schritt 1: Voraussetzungen pruefen ────────────────────────────────────────
Write-Log "Schritt 1/3 — Voraussetzungen pruefen"

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
Write-Log "Schritt 2/3 — Sim-Client-Image bauen"

$BuildTimestamp = (Get-Date -Format "yyyyMMdd-HHmmss")
$ImageLatest    = "${DockerImage}:latest"
$ImageDated     = "${DockerImage}:${BuildTimestamp}"

$BuildCmd = @("docker", "build", "--platform", "linux/amd64")
if ($NoCache) { $BuildCmd += "--no-cache" }
$BuildCmd += @("-t", $ImageLatest, "-t", $ImageDated, $ScriptDir)

Write-Log "Baue: $($BuildCmd -join ' ')"
if ($NoCache) { Write-Warn "Build ohne Cache — Isaac-Lab-Basis ist groß, kann 60+ Minuten dauern." }
Write-Host ""

Invoke-Cmd $BuildCmd

Write-Ok "Image gebaut: $ImageLatest"
Write-Ok "Image gebaut: $ImageDated"
Write-Host ""

# ── Schritt 3: Push ───────────────────────────────────────────────────────────
Write-Log "Schritt 3/3 — Nach Docker Hub pushen"

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
Write-Host "    1. SIF auf KISSKI ziehen:"
Write-Host "         module load apptainer"
Write-Host "         apptainer pull ~/.project/dir.project/images/projekt-humanoider-roboter-sim.sif \"
Write-Host "             docker://$ImageLatest"
Write-Host "    2. Phase-A-Test (leere Szene, headless EGL):"
Write-Host "         apptainer exec --nv <sim.sif> bash -c '"
Write-Host '             ${ISAACLAB_PATH}/isaaclab.sh -p -c "print(1)"'"'"
Write-Host "    3. Eval-Job einreichen: sbatch Simulation/kisski_sim_submit.sh"
Write-Host ""
