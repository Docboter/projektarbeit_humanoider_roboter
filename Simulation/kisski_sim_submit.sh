#!/usr/bin/env bash
# kisski_sim_submit.sh — SLURM-Skript für die G1+Dex3 Closed-Loop-Sim-Eval auf GWDG
#
# Startet GR00T-Policy-Server + Isaac-Lab-Sim-Client im selben SLURM-Job.
#
# GPU-Anforderung: Quadro RTX 5000 (jupyter-Partition) — A100/H100 fehlen RT-Cores!
#   → Isaac-Sim-Kamera-Rendering scheitert auf A100/H100.
#   Hintergrund: siehe Simulation/SIM_GPU_COMPATIBILITY.md
#
# Voraussetzungen (einmalig auf dem Login-Node):
#   # 1. Training-SIF (GR00T-Policy-Server) — falls noch nicht vorhanden:
#   module load apptainer
#   apptainer pull ~/.project/dir.project/images/projekt-humanoider-roboter.sif \
#       docker://lucam03/projekt-humanoider-roboter:latest
#
#   # 2. Sim-Client-SIF:
#   export APPTAINER_CACHEDIR=/mnt/vast-kisski/projects/kisski-humrob/apptainer-cache
#   export APPTAINER_TMPDIR=/mnt/vast-kisski/projects/kisski-humrob/apptainer-tmp
#   mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
#   apptainer pull ~/.project/dir.project/images/projekt-humanoider-roboter-sim.sif \
#       docker://lucam03/projekt-humanoider-roboter-sim:latest
#
#   # 3. Job einreichen:
#   export CHECKPOINT_DIR=/mnt/vast-kisski/projects/kisski-humrob/data/g1_dex3_finetune
#   sbatch Simulation/kisski_sim_submit.sh
#
# Optionale Überschreibungen (vor sbatch als export setzen):
#   SERVER_SIF, SIM_SIF, CHECKPOINT_DIR, DATA_DIR, NUM_EPISODES,
#   EXECUTION_HORIZON, TASK_DESCRIPTION, SERVER_PORT, ASSET_PATH

# ── SLURM-Direktiven ──────────────────────────────────────────────────────────
# WICHTIG: jupyter-Partition wegen RT-Cores (Isaac-Sim-Rendering)
#SBATCH --job-name=groot-sim-eval
#SBATCH -p jupyter
#SBATCH --gres=gpu:RTX5000:1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --output=/user/luca.muecke/u28320/.project/dir.project/logs/slurm-sim-%j.out
#SBATCH --error=/user/luca.muecke/u28320/.project/dir.project/logs/slurm-sim-%j.err
#SBATCH --export=ALL

set -euo pipefail

# ── Konfiguration ─────────────────────────────────────────────────────────────
SERVER_SIF="${SERVER_SIF:-/user/luca.muecke/u28320/.project/dir.project/images/projekt-humanoider-roboter.sif}"
SIM_SIF="${SIM_SIF:-/user/luca.muecke/u28320/.project/dir.project/images/projekt-humanoider-roboter-sim.sif}"

# Checkpoint-Verzeichnis (Ergebnis des Fine-tunings)
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/vast-kisski/projects/kisski-humrob/data/g1_dex3_finetune}"
DATA_DIR="${DATA_DIR:-/mnt/vast-kisski/projects/kisski-humrob/data}"

# Eval-Parameter
NUM_EPISODES="${NUM_EPISODES:-20}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-8}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-stack the blocks}"
SERVER_PORT="${SERVER_PORT:-5555}"

# G1+Dex3 USD-Asset (muss erzeugt worden sein — siehe ISAAC_LAB_SIM_PLAN.md Abschnitt 3)
# Default = schwarzhändiges Asset (Domain-Gap-Fix). Wird bei Bedarf automatisch aus
# g1_dex3.usd erzeugt (s.u.). BLACK_HANDS=0 → Original-USD ohne Recolor.
ASSET_PATH="${ASSET_PATH:-/data/assets/g1_dex3_blackhands.usd}"
BLACK_HANDS="${BLACK_HANDS:-1}"

# Isaac-Sim Cache-Verzeichnisse (SIF ist read-only → auf Projektspeicher umlenken,
# analog zu DATA_DIR im Training-Job)
ISAAC_CACHE="/mnt/vast-kisski/projects/kisski-humrob/isaac-cache"

# ── Voraussetzungen prüfen ────────────────────────────────────────────────────
for SIF in "$SERVER_SIF" "$SIM_SIF"; do
    if [[ ! -f "$SIF" ]]; then
        echo "FEHLER: SIF-Image nicht gefunden: $SIF" >&2
        echo "Einmalig auf dem Login-Node erstellen:" >&2
        echo "    module load apptainer && apptainer pull $SIF docker://lucam03/..." >&2
        exit 1
    fi
done

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "WARNUNG: Checkpoint-Verzeichnis nicht gefunden: $CHECKPOINT_DIR" >&2
    echo "         GR00T-Server startet, wird aber fehlschlagen." >&2
fi

mkdir -p "$ISAAC_CACHE"/{kit,kit_data,ov,pip,nv,logs}
mkdir -p "$DATA_DIR/sim_videos"

# ── Laufumgebung anzeigen ─────────────────────────────────────────────────────
echo "==> SLURM Job: ${SLURM_JOB_ID:-local} auf $(hostname)"
echo "    GPU:              $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unbekannt')"
echo "    Server-SIF:       $SERVER_SIF"
echo "    Sim-SIF:          $SIM_SIF"
echo "    Checkpoint:       $CHECKPOINT_DIR"
echo "    DATA_DIR:         $DATA_DIR"
echo "    Episoden:         $NUM_EPISODES"
echo "    Exec-Horizon:     $EXECUTION_HORIZON"
echo "    Task:             $TASK_DESCRIPTION"
echo "    Server-Port:      $SERVER_PORT"
echo "    Asset:            $ASSET_PATH"
echo ""

module load apptainer

export APPTAINER_CACHEDIR="/mnt/vast-kisski/projects/kisski-humrob/apptainer-cache"
export APPTAINER_TMPDIR="/mnt/vast-kisski/projects/kisski-humrob/apptainer-tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# ── 1) GR00T-Policy-Server im Hintergrund starten ────────────────────────────
echo "==> Starte GR00T-Policy-Server (Hintergrund) …"

GROOT_FORK_DIR="${GROOT_FORK_DIR:-/mnt/vast-kisski/projects/kisski-humrob/repo-groot}"
ASSETS_DIR="${ASSETS_DIR:-/mnt/vast-kisski/projects/kisski-humrob/assets}"
SIM_CODE="${SIM_CODE:-/user/luca.muecke/u28320/.project/dir.project/repo/Simulation}"

# Spezifischer Checkpoint (nicht das übergeordnete Verzeichnis!)
CHECKPOINT="${CHECKPOINT:-/data/g1_dex3_finetune/blockstacking/g1_dex3_blockstacking_v1/checkpoints/20260602/checkpoint-110000}"

GROOT_APPTAINER_ARGS=(
    --nv
    --bind "$DATA_DIR:/data"
    --env "TMPDIR=/tmp"
    --env "UV_OFFLINE=1"
    --env "PYTHONUNBUFFERED=1"
)

# gr00t-Modul aus Fork einbinden (analog zu kisski_submit.sh)
if [[ -d "$GROOT_FORK_DIR/gr00t" ]]; then
    GROOT_APPTAINER_ARGS+=(--bind "$GROOT_FORK_DIR/gr00t:/app/Groot-1.6/gr00t")
    GROOT_APPTAINER_ARGS+=(--bind "$GROOT_FORK_DIR/examples/G1_DEX3:/app/Groot-1.6/examples/G1_DEX3")
    echo "    gr00t-Modul aus Fork: $GROOT_FORK_DIR/gr00t"
fi

apptainer exec "${GROOT_APPTAINER_ARGS[@]}" "$SERVER_SIF" \
    bash -lc "cd /app/Groot-1.6 && \
        .venv/bin/python gr00t/eval/run_gr00t_server.py \
            --model-path $CHECKPOINT \
            --embodiment-tag NEW_EMBODIMENT \
            --use-sim-policy-wrapper \
            --no-flash-attn \
            --port $SERVER_PORT" \
    &
SERVER_PID=$!
echo "    Server PID: $SERVER_PID"

# Auf Server-Bereitschaft warten (TCP-Port aktiv pollen)
echo "==> Warte auf GR00T-Server (Port $SERVER_PORT) …"
MAX_WAIT=300
WAITED=0
while ! timeout 1 bash -c "echo > /dev/tcp/localhost/$SERVER_PORT" 2>/dev/null; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        echo "FEHLER: Server nicht bereit nach ${MAX_WAIT}s. Abbruch." >&2
        kill "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi
    echo "    … warte (${WAITED}s / ${MAX_WAIT}s)"
done
echo "    Server bereit."
echo ""

# ── 2) Isaac-Lab-Sim-Client (headless EGL) ───────────────────────────────────
# Hinweis zu den Cache-Bind-Mounts:
#   Isaac Sim schreibt beim Start viel in die Kit-/OmniVerse-Caches.
#   Das SIF ist read-only, daher müssen diese Pfade auf beschreibbaren Projektspeicher zeigen.
#   Die exakten Pfade können je nach Isaac-Sim-Version variieren — beim ersten
#   interaktiven Run verifizieren: `apptainer shell --nv <sim.sif>`
# ── Domain-Gap-Fix: schwarzhändiges Asset sicherstellen (BLACK_HANDS=0 deaktiviert) ──
# Im Dataset sind die DEX3-Hände schwarz, im USD weiß → für den eingefrorenen Vision-Encoder
# angleichen. Erzeugt g1_dex3_blackhands.usd bei Bedarf aus g1_dex3.usd (Recolor via pxr,
# reines USD-Authoring). Fallback aufs Original, falls der Recolor fehlschlägt.
if [[ "$BLACK_HANDS" == "1" ]]; then
    if [[ ! -f "$ASSETS_DIR/g1_dex3_blackhands.usd" && -f "$ASSETS_DIR/g1_dex3.usd" ]]; then
        echo "==> Domain-Gap-Fix: erzeuge schwarzhändiges Asset (Recolor) …"
        apptainer exec --nv \
            --bind "$DATA_DIR:/data" --bind "$ASSETS_DIR:/data/assets" \
            --bind "$SIM_CODE/g1_dex3_sim:/workspace/g1_dex3_sim" \
            "$SIM_SIF" bash -lc 'unset VIRTUAL_ENV; ${ISAACLAB_PATH}/isaaclab.sh -p \
                /workspace/g1_dex3_sim/recolor_hands_black.py \
                --in /data/assets/g1_dex3.usd --out /data/assets/g1_dex3_blackhands.usd' \
            || echo "WARN: Recolor fehlgeschlagen."
    fi
    if [[ "$ASSET_PATH" == *g1_dex3_blackhands.usd && ! -f "$ASSETS_DIR/g1_dex3_blackhands.usd" ]]; then
        echo "WARN: schwarzhändiges Asset fehlt — Fallback auf /data/assets/g1_dex3.usd."
        ASSET_PATH="/data/assets/g1_dex3.usd"
    fi
fi

echo "==> Starte Isaac-Lab-Sim-Client (headless) …"

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
    # Headless-EGL: auf RTX-5000 ist das im Isaac-Lab-Image vorkonfiguriert.
    # Falls Fehler: DISPLAY="" und VK_ICD_FILENAMES setzen (vgl. check_sim_eval_ready.py).
)

apptainer exec "${SIM_APPTAINER_ARGS[@]}" "$SIM_SIF" \
    bash -lc '${ISAACLAB_PATH}/isaaclab.sh -p \
        /workspace/g1_dex3_sim/run_g1_dex3_sim_eval.py \
        --headless \
        --enable_cameras \
        --server tcp://localhost:'"$SERVER_PORT"' \
        --num-episodes '"$NUM_EPISODES"' \
        --execution-horizon '"$EXECUTION_HORIZON"' \
        --task-description "'"$TASK_DESCRIPTION"'" \
        --video-dir /data/sim_videos \
        --results-file /data/sim_results.json \
        --asset-path '"$ASSET_PATH"

SIM_EXIT=$?
echo ""
echo "==> Sim-Client beendet (Exit-Code: $SIM_EXIT)"

# ── Aufräumen ─────────────────────────────────────────────────────────────────
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
echo "==> Server gestoppt."
echo ""

# Ergebnisse ausgeben
if [[ -f "$DATA_DIR/sim_results.json" ]]; then
    echo "==> Eval-Ergebnisse:"
    cat "$DATA_DIR/sim_results.json"
fi

echo ""
echo "==> Videos abrufen (vom Laptop):"
echo "    rsync -avz $(whoami)@transfer.hpc.gwdg.de:$DATA_DIR/sim_videos ./sim_videos/"
echo "    rsync -avz $(whoami)@transfer.hpc.gwdg.de:$DATA_DIR/sim_results.json ./sim_results.json"

exit $SIM_EXIT
