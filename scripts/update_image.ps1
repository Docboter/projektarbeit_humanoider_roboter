#!/usr/bin/env pwsh
# update_image.ps1
#
# Bringt das Docker-Image lucam03/projekt-humanoider-roboter:latest auf den
# neuesten Stand:
#
#   1. Neuesten Commit der Isaac-GR00T-Repo ermitteln
#   2. Dockerfile aktualisieren (falls -UpdateCommit gesetzt und Commit neuer)
#   3. Image bauen (docker build)
#   4. Nach Docker Hub pushen
#
# Verwendung:
#   .\update_image.ps1                    # Build + Push (aktueller Dockerfile)
#   .\update_image.ps1 -UpdateCommit      # Neuesten GR00T-Commit eintragen + bauen
#   .\update_image.ps1 -NoCache           # Build ohne Docker-Cache
#   .\update_image.ps1 -SkipPush         # Nur bauen, nicht pushen
#   .\update_image.ps1 -DryRun           # Befehle anzeigen, nichts ausfuehren
#
# Voraussetzungen:
#   docker login   (einmalig; Token wird in %USERPROFILE%\.docker\config.json gespeichert)
#   git (im PATH)
#
# Umgebungsvariablen (optional, vor dem Aufruf setzen):
#   $env:DOCKER_IMAGE   (default: lucam03/projekt-humanoider-roboter)
#   $env:GROOT_REPO     (default: https://github.com/lucam06/Isaac-GR00T.git)
#   $env:GROOT_BRANCH   (default: luca/g1-dex3)

param(
    [switch]$UpdateCommit,
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

# ── Hilfe ─────────────────────────────────────────────────────────────────────
if ($Help) {
    Get-Content $MyInvocation.MyCommand.Path | Select-Object -Skip 2 -First 24 | ForEach-Object { $_ -replace '^# ?', '' }
    exit 0
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
$DockerImage  = if ($env:DOCKER_IMAGE)  { $env:DOCKER_IMAGE }  else { "lucam03/projekt-humanoider-roboter" }
$GrootRepo    = if ($env:GROOT_REPO)    { $env:GROOT_REPO }    else { "https://github.com/lucam06/Isaac-GR00T.git" }
$GrootBranch  = if ($env:GROOT_BRANCH)  { $env:GROOT_BRANCH }  else { "luca/g1-dex3" }

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$Dockerfile = Join-Path $ScriptDir "Dockerfile"

if (-not (Test-Path $Dockerfile -PathType Leaf)) {
    Exit-Fatal "Dockerfile nicht gefunden: $Dockerfile"
}

# ── Banner ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║   GR00T N1.6 — Docker-Image Update                               ║" -ForegroundColor Magenta
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Magenta
Write-Host ""
if ($DryRun) { Write-Warn "DRY-RUN aktiv — es werden keine Befehle ausgefuehrt." }
Write-Host ""

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 1 — Voraussetzungen pruefen
# ══════════════════════════════════════════════════════════════════════════════
Write-Log "Schritt 1/4 — Voraussetzungen pruefen"

if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
    Exit-Fatal "docker nicht gefunden. Installation: https://docs.docker.com/get-docker/"
}
if (-not (Get-Command "git" -ErrorAction SilentlyContinue)) {
    Exit-Fatal "git nicht gefunden. Installation: https://git-scm.com"
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

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 2 — Commit-Stand pruefen (und ggf. Dockerfile aktualisieren)
# ══════════════════════════════════════════════════════════════════════════════
Write-Log "Schritt 2/4 — Isaac-GR00T Commit pruefen"

# Aktuell gepinnten Commit aus dem Dockerfile lesen
$DockerfileContent = Get-Content $Dockerfile -Raw
$CommitMatch = [regex]::Match($DockerfileContent, '(?<=checkout )[0-9a-f]{40}')
$CurrentCommit = if ($CommitMatch.Success) { $CommitMatch.Value } else { "" }

if ([string]::IsNullOrEmpty($CurrentCommit)) {
    Write-Warn "Kein gepinnter Commit im Dockerfile gefunden — ueberspringe Commit-Check."
} else {
    Write-Ok "Aktuell gepinnter Commit: $($CurrentCommit.Substring(0,12))..."

    # Neuesten Remote-Commit ermitteln (ohne vollstaendigen Clone)
    $LatestCommit = ""
    try {
        $lsRemoteOutput = git ls-remote $GrootRepo "refs/heads/$GrootBranch" 2>$null
        if ($LASTEXITCODE -eq 0 -and $lsRemoteOutput) {
            $LatestCommit = ($lsRemoteOutput -split '\s+')[0]
        }
    } catch { }

    if ([string]::IsNullOrEmpty($LatestCommit)) {
        Write-Warn "Konnte Remote-Commit nicht ermitteln (kein Netzwerk oder Repo nicht erreichbar?)."
    } elseif ($LatestCommit -eq $CurrentCommit) {
        Write-Ok "Dockerfile ist bereits auf dem neuesten Stand ($($LatestCommit.Substring(0,12))...)."
    } else {
        Write-Warn "Neuer Commit verfuegbar!"
        Write-Warn "  Aktuell: $($CurrentCommit.Substring(0,12))..."
        Write-Warn "  Neu:     $($LatestCommit.Substring(0,12))..."

        if ($UpdateCommit) {
            Write-Log "Aktualisiere Dockerfile (-UpdateCommit) ..."
            if (-not $DryRun) {
                $DockerfileContent = $DockerfileContent -replace [regex]::Escape($CurrentCommit), $LatestCommit
                Set-Content -Path $Dockerfile -Value $DockerfileContent -NoNewline
                Write-Ok "Dockerfile aktualisiert: $CurrentCommit -> $LatestCommit"
            } else {
                Write-Host "[dry-run] Ersetze '$CurrentCommit' -> '$LatestCommit' in Dockerfile" -ForegroundColor Yellow
            }
            $CurrentCommit = $LatestCommit
        } else {
            Write-Warn "Dockerfile wird NICHT aktualisiert (kein -UpdateCommit)."
            Write-Warn "Zum Aktualisieren: .\update_image.ps1 -UpdateCommit"
        }
    }
}
Write-Host ""

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 3 — Docker-Image bauen
# ══════════════════════════════════════════════════════════════════════════════
Write-Log "Schritt 3/4 — Docker-Image bauen"

$BuildTimestamp = (Get-Date -Format "yyyyMMdd-HHmmss")
$ImageLatest    = "${DockerImage}:latest"
$ImageDated     = "${DockerImage}:${BuildTimestamp}"

$BuildCmd = @("docker", "build", "--platform", "linux/amd64")
if ($NoCache) { $BuildCmd += "--no-cache" }
$BuildCmd += @("-t", $ImageLatest, "-t", $ImageDated, $ScriptDir)

Write-Log "Baue: $($BuildCmd -join ' ')"
if ($NoCache) { Write-Warn "Build ohne Cache — das kann 30-60 Minuten dauern (flash-attn)." }
Write-Host ""

Invoke-Cmd $BuildCmd

Write-Ok "Image gebaut: $ImageLatest"
Write-Ok "Image gebaut: $ImageDated"
Write-Host ""

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 4 — Nach Docker Hub pushen
# ══════════════════════════════════════════════════════════════════════════════
Write-Log "Schritt 4/4 — Nach Docker Hub pushen"

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
if (-not [string]::IsNullOrEmpty($CurrentCommit)) {
    Write-Host ("    {0,-20} {1}" -f "GR00T-Commit:", "$($CurrentCommit.Substring(0,12))...")
}
Write-Host ""
Write-Host "  Naechste Schritte:"
Write-Host "    * Testen:  .\setup_and_train_DockerHub-pull.ps1 -Interactive"
Write-Host "    * Ziehen:  docker pull $ImageLatest"
Write-Host ""
