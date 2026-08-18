#!/usr/bin/env bash
# entrypoint_sim.sh — Sim-Container-Entrypoint für vast.ai
#
# Startet autonom:
#   1. GPU-Check
#   2. Checkpoint prüfen / von HuggingFace herunterladen
#   3. GR00T-Policy-Server im Hintergrund starten
#   4. Auf Server-Bereitschaft warten
#   5. Isaac-Lab-Sim-Client starten (Vordergrund)
#
# Pflicht-Env-Vars:
#   CHECKPOINT_PATH      — Pfad zum feingetunten Checkpoint-Verzeichnis
#                          (z. B. /data/checkpoints/checkpoint-3000)
#                          ODER HF_CHECKPOINT_REPO setzen für automatischen Download.
#
# Optionale Env-Vars (Defaults im Dockerfile gesetzt):
#   HF_TOKEN             — HuggingFace-Token (Pflicht wenn HF_CHECKPOINT_REPO gesetzt)
#   HF_CHECKPOINT_REPO   — HF-Modell-Repo für Checkpoint-Download (z. B. user/my-model)
#   NUM_EPISODES         — Anzahl Eval-Episoden           (default 20)
#   EXECUTION_HORIZON    — Steps pro Action-Chunk          (default 8)
#   TASK_DESCRIPTION     — Language-Prompt                 (default "stack the blocks")
#   ZMQ_PORT             — ZMQ-Port für Server-Kommunikation (default 5555)
#   ASSET_PATH           — Pfad zum USD-Asset              (default /workspace/assets/g1_dex3_blackhands.usd)
#   BLACK_HANDS          — 1=Hände schwarz einfärben (Domain-Gap-Fix, default), 0=Original-USD
#                          Bei 1 wird das schwarzhändige USD bei Bedarf automatisch erzeugt
#                          (Recolor aus g1_dex3.usd); schlägt das fehl, wird das Original genutzt.
#   NO_FLASH_ATTN        — IGNORIERT (Eagle-Block2A-2B-v2 erzwingt flash_attention_2)
#   SKIP_DOWNLOAD        — 1=Checkpoint-Download überspringen (default 0)
#   SHELL_ON_ERROR       — 1=bei Fehler in Shell fallen    (default 0)
#   LIVESTREAM           — LIVE-Variante statt Videos: 0=aus (default),
#                          1=WebRTC öffentlich, 2=WebRTC privat/lokal.
#                          Bei !=0 streamt Isaac Sim seinen 3D-Viewport per WebRTC; auf
#                          dem Arbeitsrechner öffnet ihn der native „Isaac Sim WebRTC
#                          Streaming Client". Details + Client-Anleitung:
#                          docs/simulation/live-ansicht.md
#   LIVESTREAM_PORT      — WebRTC-Signaling-Port, TCP (default 49100).
#                          ⚠️ vast.ai: auf den EXTERN gemappten Port setzen (intern==extern),
#                          sonst stimmt der in der SDP eingebettete Port nicht.
#   LIVESTREAM_MEDIA_PORT— WebRTC-Medien-Port, UDP (default 47998). Muss offen sein.
#   LIVE_KEEP_VIDEO      — 1 = trotz Live-Variante zusätzlich MP4s schreiben (default 0:
#                          live STATT Video — spart das Frame-Sammeln je Step)
#   PUBLIC_IP            — Öffentliche Instanz-IP für den Remote-Endpunkt
#                          (auto via ifconfig.me, nur wenn LIVESTREAM=1)
#   GROOT_INFERENCE_BACKEND — eager (Default, historischer PyTorch-Pfad), compile
#                             (DiT via torch.compile) oder tensorrt (DiT-Engine; kein
#                             stiller Fallback bei fehlender/inkompatibler Engine)
#   GROOT_TRT_ENGINE_PATH   — TensorRT-Engine; leer = automatisch per
#                             Checkpoint-Fingerprint unter /data/optimized suchen
#   CAMERA_RENDER_EVERY_N   — Policy-Schritte pro Render; 1 = bisheriges Verhalten,
#                             schneller Modus = EXECUTION_HORIZON (normalerweise 8)
#
# Auf vast.ai:
#   Image:          lucam03/projekt-humanoider-roboter-sim-vastai:latest
#   GPU:            RTX 3090 / RTX 4090 / A6000 (≥24 GB, Ampere+, RT-Cores!)
#   Docker Options: --ipc=host --shm-size=16g
#   Disk:           Checkpoint-Volume nach /data mounten
#   Environment:    CHECKPOINT_PATH=/data/checkpoints/checkpoint-3000

set -euo pipefail

# ── Instance-Log (SSH-Zugriff) ────────────────────────────────────────────────
# Alle Ausgaben in /data/logs/entrypoint.log spiegeln (zusätzlich zu stdout).
# Per SSH erreichbar: tail -f /data/logs/entrypoint.log
mkdir -p "${DATA_DIR:-/data}/logs"
exec > >(tee -a "${DATA_DIR:-/data}/logs/entrypoint.log") 2>&1

# ── Logging ───────────────────────────────────────────────────────────────────
log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m v \033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m ! \033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m!! \033[0m %s\n' "$*" >&2; }

trap_err() {
    err "Entrypoint mit Fehler beendet (Zeile $1)."
    if [[ "${SHELL_ON_ERROR:-0}" == "1" ]]; then
        warn "SHELL_ON_ERROR=1 — falle in interaktive Shell zur Diagnose."
        exec /bin/bash
    fi
    exit 1
}
trap 'trap_err $LINENO' ERR

# Argumente durchreichen (z. B. 'docker run … bash')
if [[ $# -gt 0 ]]; then
    log "Entrypoint: führe übergebenen Befehl aus: $*"
    exec "$@"
fi

# ── Modus-Dispatch ────────────────────────────────────────────────────────────
# SIM_MODE=baseline → Baseline-Test: un-finetuntes GR00T-N1.6-3B + stock G1 mit
# Dex1-Greifer (Embodiment UNITREE_G1). Default (dex3, leer) lässt das bestehende
# DEX3-Closed-Loop-Verhalten unverändert.
if [[ "${SIM_MODE:-dex3}" == "baseline" ]]; then
    log "SIM_MODE=baseline — wechsle zu /scripts/entrypoint_baseline.sh"
    exec /scripts/entrypoint_baseline.sh
fi

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "\033[1;35m╔══════════════════════════════════════════════════════════════════╗\033[0m"
echo -e "\033[1;35m║   GR00T N1.6 Closed-Loop Sim — vast.ai Container                  ║\033[0m"
echo -e "\033[1;35m╚══════════════════════════════════════════════════════════════════╝\033[0m"
echo ""

# ── Konfiguration ─────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/data}"
GROOT_ROOT="${GROOT_ROOT:-/app/Groot-1.6}"
ZMQ_PORT="${ZMQ_PORT:-5555}"
NUM_EPISODES="${NUM_EPISODES:-20}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-8}"
# Zeitbudget je Episode; 0 = cfg-Default (300 s = 9000 Steps à 4 gerenderte Kameras).
# Der Wert bestimmt die Laufzeit direkt und ist deshalb der erste Hebel, wenn eine Eval
# nicht in ein Zeitfenster passt. Anker: die menschliche Teleop-Demo braucht 39 s.
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-0}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-stack the blocks}"
ASSET_PATH="${ASSET_PATH:-/workspace/assets/g1_dex3_blackhands.usd}"
BLACK_HANDS="${BLACK_HANDS:-1}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$DATA_DIR/checkpoints}"
HF_CHECKPOINT_REPO="${HF_CHECKPOINT_REPO:-}"
NO_FLASH_ATTN="${NO_FLASH_ATTN:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
GROOT_INFERENCE_BACKEND="${GROOT_INFERENCE_BACKEND:-eager}"
GROOT_TRT_ENGINE_PATH="${GROOT_TRT_ENGINE_PATH:-}"
CAMERA_RENDER_EVERY_N="${CAMERA_RENDER_EVERY_N:-1}"

# LIVE-Variante (WebRTC-Viewport, „Spur A"). Die gesamte Logik — Kit-Settings je
# Isaac-Sim-Version, PUBLIC_IP nur im Modus 1, Verbindungshinweis, Video-an/aus —
# steckt in der gemeinsamen Lib, damit die Zwillinge sim/baseline/replay nicht
# auseinanderlaufen. Fehlt sie (sehr altes Image), bleibt der Lauf headless.
if [[ -r /scripts/lib_livestream.sh ]]; then
    # shellcheck source=lib_livestream.sh
    source /scripts/lib_livestream.sh
    livestream_init
else
    warn "lib_livestream.sh fehlt im Image — LIVE-Variante nicht verfügbar (headless)."
    LIVESTREAM=0
    livestream_active()    { return 1; }
    livestream_app_flags() { LS_APP_FLAGS=( --headless ); }
    livestream_video_dir() { printf '%s' "$1"; }
    livestream_banner()    { return 0; }
fi

# Video-Verzeichnis: bei aktiver LIVE-Variante leer -> run_g1_dex3_sim_eval.py schaltet
# Frame-Sammeln, MP4-Ausgabe und rgb_array-Render-Mode ab (`record = bool(args.video_dir)`).
VIDEO_DIR="$(livestream_video_dir "$DATA_DIR/sim_videos")"

mkdir -p "$DATA_DIR/sim_videos" "$DATA_DIR/sim_results" "$DATA_DIR/logs"

# ── SSH-Server ─────────────────────────────────────────────────────────────────
# vast.ai injiziert den User-SSH-Key als $PUBLIC_KEY — muss in authorized_keys stehen.
# Ohne diesen Schritt findet `vastai ssh-url` keinen gemappten SSH-Port.
mkdir -p /root/.ssh && chmod 700 /root/.ssh
[[ -n "${PUBLIC_KEY:-}" ]] && echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
/usr/sbin/sshd 2>/dev/null || true

# ── GPU-Check ──────────────────────────────────────────────────────────────────
log "GPU-Check"
if ! nvidia-smi -L &>/dev/null; then
    err "Keine GPU sichtbar. Container ohne --gpus all gestartet?"
    exit 1
fi
nvidia-smi -L | sed 's/^/    /'
ok "GPU verfügbar"
echo ""

# ── HuggingFace-Token ──────────────────────────────────────────────────────────
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    ok "HF_TOKEN gesetzt"
elif [[ "$SKIP_DOWNLOAD" != "1" && -n "$HF_CHECKPOINT_REPO" ]]; then
    err "HF_TOKEN ist Pflicht wenn HF_CHECKPOINT_REPO gesetzt und SKIP_DOWNLOAD=0."
    exit 1
fi
echo ""

# ── Checkpoint prüfen / herunterladen ─────────────────────────────────────────
log "Schritt 1/3 — Checkpoint"

if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    warn "SKIP_DOWNLOAD=1 — Download übersprungen."
    if [[ ! -d "$CHECKPOINT_PATH" ]]; then
        err "CHECKPOINT_PATH '$CHECKPOINT_PATH' existiert nicht."
        exit 1
    fi
    ok "Checkpoint: $CHECKPOINT_PATH"
elif [[ -n "$HF_CHECKPOINT_REPO" ]]; then
    CHECKPOINT_PATH="$DATA_DIR/checkpoints/$(basename "$HF_CHECKPOINT_REPO")"
    if [[ -d "$CHECKPOINT_PATH" && -n "$(ls -A "$CHECKPOINT_PATH" 2>/dev/null)" ]]; then
        ok "Checkpoint bereits vorhanden: $CHECKPOINT_PATH"
    else
        log "Lade Checkpoint von HuggingFace: $HF_CHECKPOINT_REPO"
        "$GROOT_ROOT/.venv/bin/python" - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${HF_CHECKPOINT_REPO}",
    local_dir="${CHECKPOINT_PATH}",
    token="${HF_TOKEN:-None}",
)
print("Download abgeschlossen.")
EOF
        ok "Checkpoint heruntergeladen: $CHECKPOINT_PATH"
    fi
else
    if [[ ! -d "$CHECKPOINT_PATH" ]] || [[ -z "$(ls -A "$CHECKPOINT_PATH" 2>/dev/null)" ]]; then
        err "CHECKPOINT_PATH '$CHECKPOINT_PATH' ist leer oder existiert nicht."
        err ""
        err "Optionen:"
        err "  1. Volume mit Checkpoint nach /data mounten und CHECKPOINT_PATH setzen:"
        err "       -v /host/checkpoint:/data/checkpoints"
        err "       -e CHECKPOINT_PATH=/data/checkpoints/checkpoint-3000"
        err "  2. Checkpoint von HuggingFace Hub herunterladen:"
        err "       -e HF_TOKEN=hf_... -e HF_CHECKPOINT_REPO=user/my-model"
        exit 1
    fi
    ok "Checkpoint gefunden: $CHECKPOINT_PATH"
fi
echo ""

# ── GR00T-Policy-Server starten ───────────────────────────────────────────────
log "Schritt 2/3 — GR00T-Policy-Server (Port $ZMQ_PORT)"
# Triton ruft gcc -lcuda auf wenn transformers importiert wird.
# libcuda.so.1 wird vom NVIDIA-Container-Runtime injiziert, aber libcuda.so (unversioniert,
# den gcc für -lcuda braucht) fehlt im isaac-lab-Image. Symlink erstellen damit der Linker
# ihn findet. (|| true: grep liefert exit 1 bei keinem Treffer — würde sonst set -e auslösen.)
LIBCUDA_SO1=$(ldconfig -p 2>/dev/null | grep "libcuda\.so\.1" | awk '{print $NF}' | head -1 || true)
if [[ -n "$LIBCUDA_SO1" ]]; then
    ln -sf "$LIBCUDA_SO1" /usr/lib/x86_64-linux-gnu/libcuda.so 2>/dev/null || true
    ldconfig 2>/dev/null || true
    ok "libcuda.so Symlink: $LIBCUDA_SO1 -> /usr/lib/x86_64-linux-gnu/libcuda.so"
else
    warn "libcuda.so.1 nicht gefunden — Triton-Kompilierung könnte fehlschlagen"
fi

GROOT_SERVER_LOG="$DATA_DIR/logs/groot_server.log"

# Hinweis: Flash-Attention 2 ist für nvidia/Eagle-Block2A-2B-v2 PFLICHT (hartes assert im
# Modell-Backbone) und flash_attn ist im venv installiert. Es gibt kein --no-flash-attn-Flag
# am Server (tyro ServerConfig kennt es nicht). NO_FLASH_ATTN wird daher ignoriert.
if [[ "$NO_FLASH_ATTN" == "1" ]]; then
    warn "NO_FLASH_ATTN=1 wird ignoriert — Eagle-Block2A-2B-v2 erfordert flash_attention_2 zwingend."
fi

case "$GROOT_INFERENCE_BACKEND" in
    eager)
        SERVER_CMD=(
            "$GROOT_ROOT/.venv/bin/python" "$GROOT_ROOT/gr00t/eval/run_gr00t_server.py"
            --model-path "$CHECKPOINT_PATH"
            --embodiment-tag NEW_EMBODIMENT
            --use-sim-policy-wrapper
            --port "$ZMQ_PORT"
        )
        ;;
    compile|tensorrt)
        SERVER_CMD=(
            "$GROOT_ROOT/.venv/bin/python" /scripts/run_groot_optimized_server.py
            --model-path "$CHECKPOINT_PATH"
            --embodiment-tag new_embodiment
            --backend "$GROOT_INFERENCE_BACKEND"
            --port "$ZMQ_PORT"
            --optimized-root "$DATA_DIR/optimized"
        )
        [[ -n "$GROOT_TRT_ENGINE_PATH" ]] \
            && SERVER_CMD+=( --engine-path "$GROOT_TRT_ENGINE_PATH" )
        ;;
    *)
        err "GROOT_INFERENCE_BACKEND='$GROOT_INFERENCE_BACKEND' ungueltig (eager|compile|tensorrt)."
        exit 1
        ;;
esac
log "Policy-Backend: $GROOT_INFERENCE_BACKEND"
"${SERVER_CMD[@]}" > "$GROOT_SERVER_LOG" 2>&1 &
GROOT_SERVER_PID=$!
ok "GR00T-Server gestartet (PID $GROOT_SERVER_PID)"
ok "Server-Log: $GROOT_SERVER_LOG"

cleanup() {
    warn "Stoppe GR00T-Server (PID $GROOT_SERVER_PID) …"
    kill "$GROOT_SERVER_PID" 2>/dev/null || true
    wait "$GROOT_SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Auf TCP-Port warten
echo "    Warte auf Server-Bereitschaft (Port $ZMQ_PORT) …"
MAX_WAIT=300
WAITED=0
while ! timeout 1 bash -c "echo > /dev/tcp/localhost/$ZMQ_PORT" 2>/dev/null; do
    # Server-Prozess noch am Leben?
    if ! kill -0 "$GROOT_SERVER_PID" 2>/dev/null; then
        err "GR00T-Server-Prozess ist unerwartet beendet. Log:"
        tail -20 "$GROOT_SERVER_LOG" >&2
        exit 1
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        err "Server nicht bereit nach ${MAX_WAIT}s. Letzte Log-Zeilen:"
        tail -20 "$GROOT_SERVER_LOG" >&2
        exit 1
    fi
    printf "    … %ds\n" "$WAITED"
done
ok "Server bereit (nach ${WAITED}s)"
echo ""

# ── Isaac-Lab-Sim-Client starten ──────────────────────────────────────────────
log "Schritt 3/3 — Isaac-Lab-Sim-Client"
# VIRTUAL_ENV muss ungesetzt sein, damit isaaclab.sh sein eigenes Python-Bundle
# nutzt und nicht das GR00T-venv (das kein 'isaaclab'-Modul enthält).
unset VIRTUAL_ENV
# PYTHONUNBUFFERED=1: isaaclab.sh führt Pythons stdout als Pipe → print() würde sonst block-
# gepuffert und der Episoden-Fortschritt erschiene erst am Ende. Unbuffered = Live-Ausgabe.
export PYTHONUNBUFFERED=1
printf "    %-22s %s\n" "Server:"         "tcp://localhost:$ZMQ_PORT"
printf "    %-22s %s\n" "Episoden:"       "$NUM_EPISODES"
printf "    %-22s %s\n" "Exec-Horizon:"   "$EXECUTION_HORIZON"
printf "    %-22s %s\n" "Policy-Backend:" "$GROOT_INFERENCE_BACKEND"
printf "    %-22s %s\n" "Render-Stride:"  "$CAMERA_RENDER_EVERY_N Policy-Schritt(e)"
if [[ "$EPISODE_LENGTH_S" != "0" ]]; then
    printf "    %-22s %s\n" "Episodenlaenge:" "${EPISODE_LENGTH_S}s (EPISODE_LENGTH_S)"
else
    printf "    %-22s %s\n" "Episodenlaenge:" "cfg-Default (300s = 9000 Steps)"
fi
printf "    %-22s %s\n" "Task:"           "$TASK_DESCRIPTION"
printf "    %-22s %s\n" "Checkpoint:"     "$CHECKPOINT_PATH"
printf "    %-22s %s\n" "Asset:"          "$ASSET_PATH"
printf "    %-22s %s\n" "Videos:"         "${VIDEO_DIR:-(aus — LIVE-Variante)}"

# ── App-Flags: headless (default) ODER LIVE-Variante (WebRTC-Viewport) ─────────
# livestream_app_flags liefert entweder ( --headless ) oder ( --livestream N [--kit_args …] ).
# Bei LIVESTREAM=0 bleibt das alte Verhalten exakt erhalten.
APP_FLAGS=( --enable_cameras )
livestream_app_flags
APP_FLAGS+=( "${LS_APP_FLAGS[@]}" )
livestream_banner
echo ""

# ── Domain-Gap-Fix: schwarzhändiges Asset sicherstellen (BLACK_HANDS=0 deaktiviert) ──
# Im Dataset sind die DEX3-Hände schwarz, im USD weiß. Da der Vision-Encoder eingefroren
# ist, hilft die Angleichung. Erzeugt g1_dex3_blackhands.usd bei Bedarf aus dem Original
# (Recolor via pxr; reines USD-Authoring, kein Sim-Start). Fallback aufs Original bei Fehler.
if [[ "$BLACK_HANDS" == "1" ]]; then
    case "$ASSET_PATH" in
        *_blackhands.usd) BASE_USD="${ASSET_PATH/_blackhands/}" ;;
        *)                BASE_USD="$ASSET_PATH" ;;
    esac
    BH_USD="${BASE_USD%.usd}_blackhands.usd"
    if [[ -f "$BASE_USD" && ! -f "$BH_USD" ]]; then
        log "Domain-Gap-Fix: erzeuge schwarzhändiges Asset → $BH_USD"
        # Recolor über das GR00T-venv (hat usd-core/pxr). Isaacs Python exponiert pxr NICHT
        # für standalone-Skripte → isaaclab.sh -p würde mit ModuleNotFoundError: pxr scheitern.
        "$GROOT_ROOT/.venv/bin/python" /workspace/g1_dex3_sim/recolor_hands_black.py \
            --in "$BASE_USD" --out "$BH_USD" || warn "Recolor fehlgeschlagen — nutze Original."
    fi
    if [[ -f "$BH_USD" ]]; then
        ASSET_PATH="$BH_USD"; ok "Asset (schwarze Hände): $ASSET_PATH"
    else
        ASSET_PATH="$BASE_USD"; warn "Schwarzhändiges Asset fehlt — nutze $ASSET_PATH"
    fi
fi

${ISAACLAB_PATH}/isaaclab.sh -p /workspace/g1_dex3_sim/run_g1_dex3_sim_eval.py \
    "${APP_FLAGS[@]}" \
    --server "tcp://localhost:$ZMQ_PORT" \
    --num-episodes   "$NUM_EPISODES" \
    --execution-horizon "$EXECUTION_HORIZON" \
    --task-description  "$TASK_DESCRIPTION" \
    --video-dir      "$VIDEO_DIR" \
    --results-file   "$DATA_DIR/sim_results/results.json" \
    --asset-path     "$ASSET_PATH" \
    --episode-length-s "$EPISODE_LENGTH_S" \
    `# Nur zur Protokollierung in results.json — geladen hat den Checkpoint der` \
    `# Policy-Server oben (Z. 235) aus derselben Variablen.` \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --ping-retries   20

# isaaclab.sh schluckt den Exit-Code des Python-Prozesses (ein Crash im Sim-Client liefert
# trotzdem Exit 0). Daher explizit prüfen, ob die Ergebnis-Datei erzeugt wurde, statt blind
# "abgeschlossen" zu melden.
RESULTS_FILE="$DATA_DIR/sim_results/results.json"
if [[ ! -s "$RESULTS_FILE" ]]; then
    err "Sim-Eval hat keine Ergebnis-Datei erzeugt ($RESULTS_FILE)."
    err "Der Sim-Client ist vermutlich abgestürzt — letzte Container-Ausgaben oben prüfen."
    exit 1
fi

ok "Sim-Eval abgeschlossen."
echo ""
echo "  Ergebnisse: $RESULTS_FILE"
if [[ -n "$VIDEO_DIR" ]]; then
    echo "  Videos:     $VIDEO_DIR/"
else
    echo "  Videos:     keine (LIVE-Variante lief; LIVE_KEEP_VIDEO=1 schreibt sie zusätzlich)"
fi
echo ""
echo "  Daten sichern (vom Host):"
echo "    docker cp <container_id>:/data/sim_results ./sim_results"
# Bewusst als if-Block, nicht als `[[ … ]] && echo`: das wäre der LETZTE Befehl des
# Skripts, gäbe bei leerem VIDEO_DIR 1 zurück und der ERR-Trap meldete einen Fehlschlag
# für einen erfolgreichen Lauf (dieselbe Falle wie in server_rl_run.sh/build_rl_env).
if [[ -n "$VIDEO_DIR" ]]; then
    echo "    docker cp <container_id>:/data/sim_videos  ./sim_videos"
fi
echo ""
