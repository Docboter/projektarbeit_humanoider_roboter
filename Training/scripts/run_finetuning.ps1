# TL;DR: Windows-Host-Skript — startet das Finetuning im Docker-Container via docker compose exec.
# Startet das GR00T-Finetuning *vom Host aus* im Docker-Container (Windows-Variante).
#
# Voraussetzungen auf dem Host:
#   - Docker Desktop mit WSL2-Backend
#   - NVIDIA Container Toolkit (über Docker Desktop)
#   - Repo geklont, .\data Volume gefüllt (siehe scripts/download_data.sh)
#
# Verwendung:
#   .\scripts\run_finetuning.ps1                          # Defaults (N1.6)
#   .\scripts\run_finetuning.ps1 -MaxSteps 10000          # einzeln overriden
#   .\scripts\run_finetuning.ps1 -GrootVersion 1.7        # N1.7-Pfad (Code-Baum, Modell,
#                                                          #   Output-Namespace passen sich an)
#   .\scripts\run_finetuning.ps1 -DryRun                  # nur Befehl anzeigen
#
# GrootVersion steuert nur die Defaults von GrootRoot/ModelPath/OutputDir — anders als die
# Bash-Skripte (lib_groot_version.sh) gibt es hier keine venv-Aktivierung/HF_HOME-Logik;
# die läuft ausschließlich im Container über den Entrypoint.
#
# Logs landen unter .\logs\finetune-<timestamp>.log auf dem Host.

[CmdletBinding()]
param(
    [string]$Service          = "groot-training",
    [string]$ComposeFile      = "",
    [ValidateSet("1.6", "1.7")]
    [string]$GrootVersion     = "1.6",
    [string]$GrootRoot        = "",
    [string]$ModelPath        = "",
    [string]$DatasetPath      = "/data/unitreerobotics/G1_Dex3_BlockStacking_Dataset",
    [string]$OutputDir        = "",
    [string]$ExperimentName   = "g1_dex3_blockstacking_v1",
    [string]$ModalityConfig   = "",
    [string]$EmbodimentTag    = "NEW_EMBODIMENT",
    [int]   $MaxSteps         = 30000,
    [int]   $GlobalBatchSize  = 8,
    [int]   $DataloaderWorkers= 2,
    [int]   $SaveSteps        = 1000,
    [int]   $SaveTotalLimit   = 5,
    [string]$LearningRate     = "1e-4",
    [string]$WeightDecay      = "1e-5",
    [string]$WarmupRatio      = "0.05",
    [bool]  $UseWandb         = $true,
    [string]$WandbProject     = "gr00t-g1-dex3",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# ── GR00T-Version: Defaults für GrootRoot/ModelPath/OutputDir ──────────────────
# Mirror von lib_groot_version.sh (nur die Namens-/Pfad-Zuordnung, kein venv/HF_HOME).
if ($GrootVersion -eq "1.7") {
    $GrootModelName = "GR00T-N1.7-3B"
    $GrootNsSuffix  = "_n17"
} else {
    $GrootModelName = "GR00T-N1.6-3B"
    $GrootNsSuffix  = ""
}
if (-not $GrootRoot)  { $GrootRoot  = "/app/Groot-$GrootVersion" }
if (-not $ModelPath)  { $ModelPath  = "/data/models/$GrootModelName" }
if (-not $OutputDir)  { $OutputDir  = "/data/g1_dex3_finetune/blockstacking$GrootNsSuffix" }

# ── Pfade ─────────────────────────────────────────────────────────────────────
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $ComposeFile)    { $ComposeFile    = Join-Path $RepoRoot "docker-compose.yml" }
if (-not $ModalityConfig) { $ModalityConfig = "$GrootRoot/examples/G1_DEX3/g1_dex3_config.py" }

function Log([string]$msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Err([string]$msg) { Write-Host "!! $msg"  -ForegroundColor Red }

function Dc { docker compose -f $ComposeFile @Args }

# ── 1. Container-Status ───────────────────────────────────────────────────────
Log "Prüfe Container-Status …"
$running = (Dc ps --status running --services 2>$null) -split "`n" | Where-Object { $_ -eq $Service }
if (-not $running) {
    Log "Container '$Service' läuft nicht — starte ihn (detached) …"
    Dc up -d $Service | Out-Null
} else {
    Log "Container '$Service' läuft bereits."
}

# ── 2. Sanity-Checks ──────────────────────────────────────────────────────────
Log "Prüfe GPU-Sichtbarkeit im Container …"
Dc exec -T $Service nvidia-smi -L
if ($LASTEXITCODE -ne 0) { Err "GPU im Container nicht sichtbar"; exit 1 }

Log "Prüfe Modell und Datensatz …"
$check = @"
[[ -d '$ModelPath'   ]] || { echo 'FEHLT: $ModelPath'   >&2; exit 1; }
[[ -d '$DatasetPath' ]] || { echo 'FEHLT: $DatasetPath' >&2; exit 1; }
[[ -f '$DatasetPath/meta/modality.json' ]] || {
    echo 'modality.json fehlt — kopiere aus examples/G1_DEX3/modality_4cam.json'
    cp '$GrootRoot/examples/G1_DEX3/modality_4cam.json' '$DatasetPath/meta/modality.json'
}
mkdir -p '$OutputDir'
"@
Dc exec -T $Service bash -c $check
if ($LASTEXITCODE -ne 0) { exit 1 }

# ── 3. W&B-Login ──────────────────────────────────────────────────────────────
if ($UseWandb) {
    Dc exec -T $Service bash -c "uv run wandb whoami" 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Log "W&B nicht eingeloggt — interaktiver Login (API-Key bereithalten):"
        Dc exec $Service bash -c "cd '$GrootRoot' && uv run wandb login"
    } else {
        Log "W&B-Login OK."
    }
}

# ── 4. Trainings-Befehl bauen ─────────────────────────────────────────────────
$TrainCmd = @(
    "uv","run","python","$GrootRoot/gr00t/experiment/launch_finetune.py",
    "--base_model_path",        $ModelPath,
    "--dataset_path",           $DatasetPath,
    "--embodiment_tag",         $EmbodimentTag,
    "--modality_config_path",   $ModalityConfig,
    "--output_dir",             $OutputDir,
    "--experiment_name",        $ExperimentName,
    "--num_gpus",               "1",
    "--max_steps",              "$MaxSteps",
    "--save_steps",             "$SaveSteps",
    "--save_total_limit",       "$SaveTotalLimit",
    "--global_batch_size",      "$GlobalBatchSize",
    "--learning_rate",          $LearningRate,
    "--weight_decay",           $WeightDecay,
    "--warmup_ratio",           $WarmupRatio,
    "--dataloader_num_workers", "$DataloaderWorkers",
    "--color_jitter_params",    "brightness","0.3","contrast","0.4","saturation","0.5","hue","0.08"
)
if ($UseWandb) { $TrainCmd += @("--use_wandb","--wandb_project",$WandbProject) }

Log "Trainings-Befehl:"
Write-Host ("    " + ($TrainCmd -join " "))

if ($DryRun) { Log "Dry-Run — nicht ausgeführt."; exit 0 }

# ── 5. Training ausführen ─────────────────────────────────────────────────────
$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("finetune-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
Log "Logs: $LogFile"
Log "Training startet — abbrechen mit Ctrl+C (Container läuft weiter)."
""

Dc exec -T --workdir $GrootRoot $Service @TrainCmd 2>&1 | Tee-Object -FilePath $LogFile
$ExitCode = $LASTEXITCODE

""
if ($ExitCode -eq 0) {
    Log "Training erfolgreich beendet."
    $last = Dc exec -T $Service bash -c "ls -d $OutputDir/checkpoint-* 2>/dev/null | sort -V | tail -1"
    Log "Letzter Checkpoint: $last"
} else {
    Err "Training mit Exit-Code $ExitCode beendet — siehe $LogFile"
}
exit $ExitCode
