#!/usr/bin/env pwsh
# setup_and_train_DockerHub-pull.ps1
#
# Schlankes Host-Skript: zieht das Image von Docker Hub und startet den
# autonomen Container-Entrypoint. Alle eigentliche Arbeit (Download, Konvertierung,
# Training) passiert IM Container — das gleiche Image laeuft so auch auf vast.ai
# oder anderen Cloud-GPU-Plattformen ohne dieses Skript.
#
# Konzept: KEIN persistenter Storage auf dem Host.
#   * Kein -v-Mount nach /data
#   * Kein --rm — der Container bleibt nach `stop` bestehen
#   * Daten und Checkpoints leben im Container-Filesystem
#   * Bei -Destroy (oder docker rm) ist alles weg
#
# Verwendung:
#   $env:HF_TOKEN = "hf_..."; .\setup_and_train_DockerHub-pull.ps1
#   .\setup_and_train_DockerHub-pull.ps1 -SkipPull            # Image schon lokal
#   .\setup_and_train_DockerHub-pull.ps1 -Interactive         # Shell statt Training
#   .\setup_and_train_DockerHub-pull.ps1 -Resume              # Bestehenden Container weiterlaufen lassen
#   .\setup_and_train_DockerHub-pull.ps1 -Destroy             # Alten Container loeschen + neu starten
#   .\setup_and_train_DockerHub-pull.ps1 -DryRun              # Nur Befehle anzeigen
#
# Umgebungsvariablen:
#   $env:HF_TOKEN          = "hf_..."       HuggingFace-Token (Pflicht)
#   $env:WANDB_API_KEY     = "..."          W&B-Key (optional)
#   $env:MAX_STEPS         = "30000"
#   $env:GLOBAL_BATCH_SIZE = "8"
#   $env:NUM_GPUS          = "1"
#   $env:WANDB_PROJECT     = "gr00t-g1-dex3"
#   $env:CONTAINER_NAME    = "groot-train"
#   $env:DOCKER_HUB_IMAGE  = "lucam03/projekt-humanoider-roboter:latest"

param(
    [switch]$SkipPull,
    [switch]$Interactive,
    [switch]$Resume,
    [switch]$Destroy,
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
    Get-Content $PSCommandPath | Select-Object -Skip 1 -First 30 | ForEach-Object { $_ -replace '^# ?','' }
    exit 0
}

function Invoke-Cmd {
    param([string[]]$Cmd)
    if ($DryRun) {
        Write-Host "[dry-run] $($Cmd -join ' ')" -ForegroundColor Yellow
        return
    }
    & $Cmd[0] $Cmd[1..($Cmd.Length-1)]
    if ($LASTEXITCODE -ne 0) { Exit-Fatal "Befehl fehlgeschlagen: $($Cmd -join ' ')" }
}

# ── Konfiguration ─────────────────────────────────────────────────────────────
$DockerHubImage = if ($env:DOCKER_HUB_IMAGE) { $env:DOCKER_HUB_IMAGE } else { "lucam03/projekt-humanoider-roboter:latest" }
$ContainerName  = if ($env:CONTAINER_NAME)   { $env:CONTAINER_NAME }   else { "groot-train" }
$MaxSteps         = if ($env:MAX_STEPS)         { $env:MAX_STEPS }         else { "30000" }
$GlobalBatchSize  = if ($env:GLOBAL_BATCH_SIZE) { $env:GLOBAL_BATCH_SIZE } else { "8" }
$NumGpus          = if ($env:NUM_GPUS)          { $env:NUM_GPUS }          else { "1" }
$WandbProject     = if ($env:WANDB_PROJECT)     { $env:WANDB_PROJECT }     else { "gr00t-g1-dex3" }

# ── Banner ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║   GR00T N1.6 Fine-tuning — Host-Launcher (Container ist autonom) ║" -ForegroundColor Magenta
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Magenta
Write-Host ""
if ($DryRun) { Write-Warn "DRY-RUN aktiv — es werden keine Befehle ausgefuehrt." }

# ── 1. Voraussetzungen ────────────────────────────────────────────────────────
Write-Log "Schritt 1/3 — Voraussetzungen pruefen"
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Exit-Fatal "docker nicht gefunden. Installation: https://docs.docker.com/get-docker/"
}
docker info *> $null
if ($LASTEXITCODE -ne 0) { Exit-Fatal "Docker-Daemon nicht erreichbar." }
Write-Ok "Docker-Daemon laeuft"

docker run --rm --gpus all --entrypoint nvidia-smi "nvidia/cuda:12.8.0-base-ubuntu22.04" -L *> $null
if ($LASTEXITCODE -eq 0) {
    Write-Ok "NVIDIA Container Toolkit funktioniert"
} else {
    Write-Warn "NVIDIA Container Toolkit nicht verfuegbar oder keine GPU erkannt."
    $ans = Read-Host "  Trotzdem fortfahren? [j/N]"
    if ($ans.ToLower() -ne "j") { Exit-Fatal "Abgebrochen." }
}
Write-Host ""

# ── 2. Bestehenden Container behandeln ────────────────────────────────────────
Write-Log "Schritt 2/3 — Container-Status pruefen ($ContainerName)"

$existing = docker ps -a --format '{{.Names}}' | Where-Object { $_ -eq $ContainerName }
$running  = docker ps    --format '{{.Names}}' | Where-Object { $_ -eq $ContainerName }

if ($Destroy -and $existing) {
    Write-Warn "Loesche bestehenden Container '$ContainerName' (-Destroy)."
    Invoke-Cmd @("docker","rm","-f",$ContainerName)
    $existing = $null
    $running  = $null
}

if ($running) {
    Write-Warn "Container '$ContainerName' laeuft bereits."
    Write-Log "Haenge an die laufende Konsole an (Ctrl+P, Ctrl+Q zum Loesen ohne Stop)…"
    Invoke-Cmd @("docker","attach",$ContainerName)
    exit 0
}

if ($existing) {
    if ($Resume) {
        Write-Log "Starte bestehenden Container '$ContainerName' (-Resume)…"
        Invoke-Cmd @("docker","start","-ai",$ContainerName)
        exit 0
    } else {
        Write-Warn "Container '$ContainerName' existiert bereits (gestoppt)."
        Write-Warn "Optionen:"
        Write-Warn "  -Resume   den Container weiterlaufen lassen (Daten + Checkpoints bleiben)"
        Write-Warn "  -Destroy  Container loeschen, alles verwerfen und neu starten"
        $ans = Read-Host "  Was tun? [r=resume / d=destroy / a=abbrechen]"
        switch ($ans.ToLower()) {
            "r" { Invoke-Cmd @("docker","start","-ai",$ContainerName); exit 0 }
            "d" { Invoke-Cmd @("docker","rm","-f",$ContainerName); $existing = $null }
            default { Exit-Fatal "Abgebrochen." }
        }
    }
}
Write-Host ""

# ── 3. Pflicht-Env pruefen ────────────────────────────────────────────────────
if (-not $Interactive) {
    if (-not $env:HF_TOKEN) {
        Write-Warn "Kein HF_TOKEN gesetzt."
        $hfInput = Read-Host "  HuggingFace-Token eingeben"
        if (-not $hfInput) { Exit-Fatal "HF_TOKEN ist Pflicht." }
        $env:HF_TOKEN = $hfInput
    }
    Write-Ok "HF_TOKEN gesetzt"

    if (-not $env:WANDB_API_KEY) {
        Write-Warn "Kein WANDB_API_KEY gesetzt — Training laeuft ohne W&B-Logging."
        $wandbInput = Read-Host "  WandB API-Key eingeben (leer lassen fuer ohne W&B)"
        if ($wandbInput) { $env:WANDB_API_KEY = $wandbInput }
    }
}
Write-Host ""

# ── 4. Image ziehen ───────────────────────────────────────────────────────────
Write-Log "Schritt 3/3 — Docker-Image laden und Container starten"
if ($SkipPull) {
    Write-Warn "Pull uebersprungen (-SkipPull)."
} else {
    Invoke-Cmd @("docker","pull",$DockerHubImage)
}
Write-Ok "Image bereit: $DockerHubImage"
Write-Host ""

# ── 5. Container starten ──────────────────────────────────────────────────────
# Bewusst KEIN --rm und KEIN -v.
$runArgs = @("docker","run","--name",$ContainerName,"--gpus","all","--ipc=host","--shm-size=16g")

if ($Interactive) {
    Write-Log "Interaktive Shell — kein automatisches Training."
    $runArgs += @("-it",$DockerHubImage,"bash")
} else {
    Write-Host "  Trainings-Konfiguration:"
    Write-Host ("    {0,-25} {1}" -f "MAX_STEPS",         $MaxSteps)
    Write-Host ("    {0,-25} {1}" -f "GLOBAL_BATCH_SIZE", $GlobalBatchSize)
    Write-Host ("    {0,-25} {1}" -f "NUM_GPUS",          $NumGpus)
    Write-Host ("    {0,-25} {1}" -f "WANDB_PROJECT",     $WandbProject)
    Write-Host ("    {0,-25} {1}" -f "CONTAINER_NAME",    $ContainerName)
    Write-Host ""

    $runArgs += @(
        "-e","HF_TOKEN=$($env:HF_TOKEN)",
        "-e","MAX_STEPS=$MaxSteps",
        "-e","GLOBAL_BATCH_SIZE=$GlobalBatchSize",
        "-e","NUM_GPUS=$NumGpus",
        "-e","WANDB_PROJECT=$WandbProject"
    )
    if ($env:WANDB_API_KEY) { $runArgs += @("-e","WANDB_API_KEY=$($env:WANDB_API_KEY)") }
    $runArgs += @("-it",$DockerHubImage)
}

Invoke-Cmd $runArgs

Write-Host ""
Write-Ok "Container beendet (nicht geloescht)."
Write-Host ""
Write-Host "  Naechste Schritte:"
Write-Host "    * Container fortsetzen:  .\setup_and_train_DockerHub-pull.ps1 -Resume"
Write-Host "    * Checkpoints sichern:   docker cp $ContainerName`:/data/g1_dex3_finetune .\checkpoints"
Write-Host "    * Logs sichern:          docker cp $ContainerName`:/data/logs .\logs"
Write-Host "    * Alles loeschen:        .\setup_and_train_DockerHub-pull.ps1 -Destroy"
Write-Host ""
Write-Host "  Hinweis: Auf vast.ai brauchst du dieses Skript NICHT — dort uebernimmt"
Write-Host "  der vast.ai-Orchestrator die Container-Verwaltung. Du gibst nur das"
Write-Host "  Image '$DockerHubImage' und die Env-Vars an."
Write-Host ""
