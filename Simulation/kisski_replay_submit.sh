#!/usr/bin/env bash
# kisski_replay_submit.sh — SLURM-Skript für die OPEN-LOOP-DATASET-REPLAY-Diagnose auf GWDG
#
# ⚠️  ACHTUNG — GPU-INKOMPATIBILITÄT (Stand: docs/simulation/implementation-notes.md):
#     Die KISSKI jupyter-Partition hat NUR die Quadro RTX 5000 (Turing, SM 7.5). Isaac Sim 4.x
#     setzt Ampere (RTX 30xx) als Minimum voraus → das Kamera-Rendering scheitert auf der
#     RTX 5000 (createDLSSContext-Fehler). A100/H100 fehlen RT-Cores → ebenfalls untauglich.
#     => Auf KISSKI läuft dieses Skript mit dem aktuellen Sim-Image VORAUSSICHTLICH NICHT durch.
#        Praktischer Weg fürs Replay: vast.ai mit L40 / RTX 4090 / A6000 (entrypoint_replay.sh).
#     Dieses Skript bleibt nur für den Fall, dass ein Turing-fähiges, älteres Isaac-Lab-Image
#     (1.x / frühes 2.x) gebaut wird (siehe implementation-notes.md).
#
# Zweck: trennt eindeutig "Sim/Config-Fehler" von "Modell-Problem".
#   Statt das GR00T-Modell zu befragen, werden die ECHTEN aufgezeichneten Dataset-Aktionen
#   (gebündelte Episode 0 von unitreerobotics/G1_Dex3_BlockStacking_Dataset) direkt in
#   dieselbe Isaac-Lab-Env gespeist. KEIN GR00T-Server, KEIN Modell, KEIN Checkpoint nötig.
#
# Interpretation des Ergebnis-Videos (/data/sim_videos_replay/replay_episode0.mp4):
#   - Arme fahren NACH VORNE zum Tisch + Finger schließen greif-artig  → Sim/Config korrekt,
#     d.h. das Wegdriften im Closed-Loop liegt am MODELL (Training), nicht an der Config.
#   - Roboter driftet auch beim Replay weg / bewegt sich unsinnig       → es steckt ein
#     SIM-Problem drin (Reachability, Achsen-Konvention, Skalierung), das gezielt zu fixen ist.
#
# Unterschied zu kisski_sim_submit.sh: dort Closed-Loop mit GR00T-Server; HIER nur der
# Sim-Client mit aufgezeichneten Aktionen. Daher KEIN Server-SIF, KEIN Port-Warten, schneller.
#
# Voraussetzungen (einmalig auf dem Login-Node — identisch zu kisski_sim_submit.sh):
#   module load apptainer
#   export APPTAINER_CACHEDIR=/mnt/vast-kisski/projects/kisski-humrob/apptainer-cache
#   export APPTAINER_TMPDIR=/mnt/vast-kisski/projects/kisski-humrob/apptainer-tmp
#   apptainer pull ~/.project/dir.project/images/projekt-humanoider-roboter-sim.sif \
#       docker://lucam03/projekt-humanoider-roboter-sim:latest
#
#   # Job einreichen:
#   sbatch Simulation/kisski_replay_submit.sh
#   # Greif-Physik-Test (Würfel exakt an die aufgezeichneten Greifpunkte):
#   GRASP_TEST=1 sbatch Simulation/kisski_replay_submit.sh
#
# Optionale Überschreibungen (vor sbatch als export setzen):
#   SIM_SIF, DATA_DIR, ASSET_PATH, GRASP_TEST, MAX_STEPS

# ── SLURM-Direktiven ──────────────────────────────────────────────────────────
# WICHTIG: jupyter-Partition wegen RT-Cores (Isaac-Sim-Kamera-Rendering).
#          A100/H100 fehlen RT-Cores → Rendering scheitert (s. docs/simulation/).
#SBATCH --job-name=groot-sim-replay
#SBATCH -p jupyter
#SBATCH --gres=gpu:RTX5000:1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --output=/user/luca.muecke/u28320/.project/dir.project/logs/slurm-replay-%j.out
#SBATCH --error=/user/luca.muecke/u28320/.project/dir.project/logs/slurm-replay-%j.err
#SBATCH --export=ALL

set -euo pipefail

# ── Konfiguration ─────────────────────────────────────────────────────────────
SIM_SIF="${SIM_SIF:-/user/luca.muecke/u28320/.project/dir.project/images/projekt-humanoider-roboter-sim.sif}"
DATA_DIR="${DATA_DIR:-/mnt/vast-kisski/projects/kisski-humrob/data}"

# G1+Dex3 USD-Asset (muss erzeugt worden sein — identisch zum Closed-Loop-Eval)
ASSET_PATH="${ASSET_PATH:-/data/assets/g1_dex3.usd}"

# 1 = Würfel exakt an die aufgezeichneten Greifpunkte setzen (Greif-Physik prüfen),
# 0 = Standard-Würfelpositionen der Env.
GRASP_TEST="${GRASP_TEST:-0}"
# 0 = alle aufgezeichneten Frames abspielen.
MAX_STEPS="${MAX_STEPS:-0}"

# Code-/Asset-Pfade (analog zu kisski_sim_submit.sh)
ASSETS_DIR="${ASSETS_DIR:-/mnt/vast-kisski/projects/kisski-humrob/assets}"
SIM_CODE="${SIM_CODE:-/user/luca.muecke/u28320/.project/dir.project/repo/Simulation}"

# Isaac-Sim-Cache (SIF read-only → auf beschreibbaren Projektspeicher umlenken)
ISAAC_CACHE="${ISAAC_CACHE:-/mnt/vast-kisski/projects/kisski-humrob/isaac-cache}"

# ── Voraussetzungen prüfen ────────────────────────────────────────────────────
if [[ ! -f "$SIM_SIF" ]]; then
    echo "FEHLER: Sim-SIF nicht gefunden: $SIM_SIF" >&2
    echo "Einmalig auf dem Login-Node erstellen:" >&2
    echo "    module load apptainer && apptainer pull $SIM_SIF docker://lucam03/projekt-humanoider-roboter-sim:latest" >&2
    exit 1
fi

mkdir -p "$ISAAC_CACHE"/{kit,kit_data,ov,pip,nv,logs}
mkdir -p "$DATA_DIR/sim_videos_replay" "$DATA_DIR/sim_results_replay"

# ── Laufumgebung anzeigen ─────────────────────────────────────────────────────
echo "==> SLURM Job: ${SLURM_JOB_ID:-local} auf $(hostname)"
echo "    GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unbekannt')"
echo "    Sim-SIF:       $SIM_SIF"
echo "    DATA_DIR:      $DATA_DIR"
echo "    Asset:         $ASSET_PATH"
echo "    Grasp-Test:    $GRASP_TEST"
echo "    Max-Steps:     $MAX_STEPS (0 = alle)"
echo ""

module load apptainer

export APPTAINER_CACHEDIR="/mnt/vast-kisski/projects/kisski-humrob/apptainer-cache"
export APPTAINER_TMPDIR="/mnt/vast-kisski/projects/kisski-humrob/apptainer-tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# ── Isaac-Lab-Replay-Client (headless EGL) ────────────────────────────────────
echo "==> Starte Open-Loop-Dataset-Replay (headless) …"

SIM_APPTAINER_ARGS=(
    --nv
    --bind "$DATA_DIR:/data"
    --bind "$ASSETS_DIR:/data/assets"
    --bind "$SIM_CODE/g1_dex3_sim:/workspace/g1_dex3_sim"
    --bind "$ISAAC_CACHE/kit:/isaac-sim/kit/cache"
    --bind "$ISAAC_CACHE/kit_data:/isaac-sim/kit/data"
    --bind "$ISAAC_CACHE/ov:/root/.cache/ov"
    --bind "$ISAAC_CACHE/nv:/root/.cache/nvidia"
    --bind "$ISAAC_CACHE/pip:/root/.cache/pip"
    --bind "$ISAAC_CACHE/logs:/root/.local/share/ov/data/Kit/logs"
    --env "ACCEPT_EULA=Y"
    --env "PRIVACY_CONSENT=Y"
    --env "OMNI_KIT_ACCEPT_EULA=yes"
)

# Optionale Flags zusammenbauen
REPLAY_FLAGS=""
[[ "$GRASP_TEST" == "1" ]] && REPLAY_FLAGS="$REPLAY_FLAGS --grasp-test"
[[ "$MAX_STEPS" != "0" ]] && REPLAY_FLAGS="$REPLAY_FLAGS --max-steps $MAX_STEPS"

# VIRTUAL_ENV muss ungesetzt sein, damit isaaclab.sh sein eigenes Python nutzt.
apptainer exec "${SIM_APPTAINER_ARGS[@]}" "$SIM_SIF" \
    bash -lc 'unset VIRTUAL_ENV; export PYTHONUNBUFFERED=1; \
        ${ISAACLAB_PATH}/isaaclab.sh -p \
        /workspace/g1_dex3_sim/run_g1_dex3_replay.py \
        --headless \
        --enable_cameras \
        --asset-path '"$ASSET_PATH"' \
        --video-dir /data/sim_videos_replay \
        --results-file /data/sim_results_replay/results.json'"$REPLAY_FLAGS"

SIM_EXIT=$?
echo ""
echo "==> Replay-Client beendet (Exit-Code: $SIM_EXIT)"
echo ""

# Ergebnisse ausgeben
if [[ -f "$DATA_DIR/sim_results_replay/results.json" ]]; then
    echo "==> Replay-Ergebnisse:"
    cat "$DATA_DIR/sim_results_replay/results.json"
    echo ""
fi

echo "==> Artefakte abrufen (vom Laptop):"
echo "    rsync -avz $(whoami)@transfer.hpc.gwdg.de:$DATA_DIR/sim_videos_replay ./sim_videos_replay/"
echo "    rsync -avz $(whoami)@transfer.hpc.gwdg.de:$DATA_DIR/sim_results_replay ./sim_results_replay/"

exit $SIM_EXIT
