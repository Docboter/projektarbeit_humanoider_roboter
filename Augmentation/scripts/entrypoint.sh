#!/usr/bin/env bash
# TL;DR: Container-Entrypoint des Augmentierungs-Images: Download → Specs → Cosmos → Datensatz.
# entrypoint.sh — Container-Entrypoint für den eigenen Docker-Server (Dauerbetrieb).
#
# Laeuft beim Containerstart komplett autonom durch:
#   1. GPU + HF-Token + Cosmos-Lizenz pruefen
#   2. Echten Trainingsdatensatz nach /data herunterladen + v3.0→v2.1 konvertieren
#      (falls noch nicht vorhanden)
#   3. Cosmos-Transfer2.5-Specs bauen (nur Train-Split-Episoden, siehe AUGMENT_TRAIN_RATIO) —
#      OHNE eigenen Control-Video-Schritt: Cosmos erzeugt die Edge-Kontrolle selbst
#      on-the-fly aus dem Quellvideo (AUGMENT_EDGE_THRESHOLD steuert die Empfindlichkeit)
#   4. Cosmos-Transfer2.5-Inferenz pro Episode/Kamera/Variante fahren
#   5. Ergebnis zu einem COTRAIN_DATASET_PATH-kompatiblen LeRobot-v2.1-Datensatz zusammenbauen
#
# Steuerung ueber Env-Vars — vollstaendige Referenz: docs/augmentation/anleitung.md
#   HF_TOKEN                (Pflicht) HuggingFace-Token mit Zugriff auf den Datensatz UND die
#                            gated Modelle nvidia/Cosmos-Transfer2.5-2B UND
#                            nvidia/Cosmos-Predict2.5-2B (Lizenz muss VORHER manuell auf BEIDEN
#                            HF-Modellseiten akzeptiert werden — das kann dieses Skript nicht
#                            automatisieren)
#   HF_HOME                 (default /data/hf_cache) Cosmos-Checkpoint-Cache
#   DATA_DIR                (default /data)
#   NUM_GPU                 (default 1) — Cosmos' eigener Variablenname (torchrun --nproc_per_node),
#                            bewusst NICHT NUM_GPUS wie im Training-Image
#   SKIP_DOWNLOAD            (default 0)
#   SKIP_CONVERT              (default 0)
#   SOURCE_DATASET_REPO      (default unitreerobotics/G1_Dex3_BlockStacking_Dataset)
#   AUGMENT_TRAIN_RATIO      (default 0.8)  — MUSS mit TRAIN_SPLIT_RATIO im Training-Image
#                            uebereinstimmen, sonst drohen Test-Episoden ins Training zu leaken
#   AUGMENT_EPISODE_LIMIT    (default 2)    — 0 = alle Train-Episoden
#   AUGMENT_EPISODE_IDS      (default "")   — explizite Liste, ueberschreibt das Limit
#   AUGMENT_CAMERAS          (default alle 4 Policy-Kameras)
#   AUGMENT_VARIANTS         (default 1)
#   AUGMENT_EDGE_THRESHOLD   (default medium) — very_low/low/medium/high/very_high, steuert
#                            Cosmos' eigene on-the-fly-Kantenerkennung
#   AUGMENT_MODEL_VARIANT    (default edge) — "edge/distilled" braucht zusaetzlich
#                            COSMOS_EXPERIMENTAL_CHECKPOINTS=1 (run_inference.sh setzt das
#                            automatisch, wenn der Name "distilled" enthaelt)
#   AUGMENT_NUM_STEPS        (default 35, Cosmos' eigener Standard fuer das Vollmodell —
#                            bei "edge/distilled" reichen 4)
#   AUGMENT_DISABLE_GUARDRAILS (default 1) — NVIDIAs Content-Safety-Filter aus (laedt sonst
#                            ein zweites, separat gated HF-Repo: nvidia/Cosmos-Guardrail1)
#   AUGMENT_STRICT_FRAME_CHECK (default 1)
#   AUGMENT_OUT_DIR          (default $DATA_DIR/augmentation/g1_dex3_cosmos_augmented)
#   AUGMENT_HF_REPO          (default "")   — falls gesetzt: Upload nach dem Zusammenbau
#   SKIP_INFERENCE           (default 0)    — nach Schritt 4 stoppen (Prompts erst pruefen)
#   SKIP_ASSEMBLE            (default 0)    — nach Schritt 5 stoppen (Rohvideos erst pruefen)
#   AUGMENT_OVERWRITE        (default 0)
#   SHELL_ON_ERROR           (default 0)    — bei Fehler in eine Shell fallen
#
# Bei Aufruf mit Argumenten wird das Skript nicht aktiv — stattdessen wird das
# Argument direkt ausgefuehrt (nuetzlich fuer `docker run … bash`).

set -euo pipefail

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

# ── Manuelle Override-Befehle durchreichen ────────────────────────────────────
if [[ $# -gt 0 ]]; then
    log "Entrypoint: fuehre uebergebenen Befehl aus: $*"
    exec "$@"
fi

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "\033[1;35m╔══════════════════════════════════════════════════════════════════╗\033[0m"
echo -e "\033[1;35m║   Cosmos-Transfer2.5 Video-Augmentierung — Container-Entrypoint  ║\033[0m"
echo -e "\033[1;35m╚══════════════════════════════════════════════════════════════════╝\033[0m"
echo ""

# ── Konfiguration ─────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/data}"
export DATA_DIR
HF_HOME="${HF_HOME:-/data/hf_cache}"
export HF_HOME
mkdir -p "$HF_HOME"

NUM_GPU="${NUM_GPU:-1}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
SOURCE_DATASET_REPO="${SOURCE_DATASET_REPO:-unitreerobotics/G1_Dex3_BlockStacking_Dataset}"
AUGMENT_TRAIN_RATIO="${AUGMENT_TRAIN_RATIO:-0.8}"
AUGMENT_EPISODE_LIMIT="${AUGMENT_EPISODE_LIMIT:-2}"
AUGMENT_EPISODE_IDS="${AUGMENT_EPISODE_IDS:-}"
AUGMENT_CAMERAS="${AUGMENT_CAMERAS:-cam_left_high,cam_right_high,cam_left_wrist,cam_right_wrist}"
AUGMENT_VARIANTS="${AUGMENT_VARIANTS:-1}"
AUGMENT_EDGE_THRESHOLD="${AUGMENT_EDGE_THRESHOLD:-medium}"
AUGMENT_MODEL_VARIANT="${AUGMENT_MODEL_VARIANT:-edge}"
AUGMENT_NUM_STEPS="${AUGMENT_NUM_STEPS:-35}"
AUGMENT_DISABLE_GUARDRAILS="${AUGMENT_DISABLE_GUARDRAILS:-1}"
AUGMENT_STRICT_FRAME_CHECK="${AUGMENT_STRICT_FRAME_CHECK:-1}"
AUGMENT_OUT_DIR="${AUGMENT_OUT_DIR:-$DATA_DIR/augmentation/g1_dex3_cosmos_augmented}"
AUGMENT_HF_REPO="${AUGMENT_HF_REPO:-}"
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"
SKIP_ASSEMBLE="${SKIP_ASSEMBLE:-0}"
AUGMENT_OVERWRITE="${AUGMENT_OVERWRITE:-0}"

export NUM_GPU SOURCE_DATASET_REPO AUGMENT_TRAIN_RATIO AUGMENT_EPISODE_LIMIT \
       AUGMENT_EPISODE_IDS AUGMENT_CAMERAS AUGMENT_VARIANTS AUGMENT_EDGE_THRESHOLD \
       AUGMENT_MODEL_VARIANT AUGMENT_NUM_STEPS AUGMENT_DISABLE_GUARDRAILS \
       AUGMENT_STRICT_FRAME_CHECK AUGMENT_OUT_DIR AUGMENT_OVERWRITE

DATASET_DIR="$DATA_DIR/unitreerobotics/G1_Dex3_BlockStacking_Dataset"
SPECS_DIR="$DATA_DIR/augmentation/specs"
RAW_OUTPUT_DIR="$DATA_DIR/augmentation/raw_output"
WORK_LIST="$SPECS_DIR/work_list.json"

echo "  Konfiguration:"
printf "    %-24s %s\n" "DATA_DIR"                "$DATA_DIR"
printf "    %-24s %s\n" "NUM_GPU"                  "$NUM_GPU"
printf "    %-24s %s\n" "SOURCE_DATASET_REPO"      "$SOURCE_DATASET_REPO"
printf "    %-24s %s\n" "AUGMENT_TRAIN_RATIO"      "$AUGMENT_TRAIN_RATIO"
printf "    %-24s %s\n" "AUGMENT_EPISODE_LIMIT"    "$AUGMENT_EPISODE_LIMIT"
printf "    %-24s %s\n" "AUGMENT_EPISODE_IDS"      "${AUGMENT_EPISODE_IDS:-<keine>}"
printf "    %-24s %s\n" "AUGMENT_CAMERAS"          "$AUGMENT_CAMERAS"
printf "    %-24s %s\n" "AUGMENT_VARIANTS"         "$AUGMENT_VARIANTS"
printf "    %-24s %s\n" "AUGMENT_EDGE_THRESHOLD"   "$AUGMENT_EDGE_THRESHOLD"
printf "    %-24s %s\n" "AUGMENT_MODEL_VARIANT"    "$AUGMENT_MODEL_VARIANT"
printf "    %-24s %s\n" "AUGMENT_DISABLE_GUARDRAILS" "$AUGMENT_DISABLE_GUARDRAILS"
printf "    %-24s %s\n" "AUGMENT_OUT_DIR"          "$AUGMENT_OUT_DIR"
echo ""

# ── GPU-Check ─────────────────────────────────────────────────────────────────
log "GPU-Check"
if ! nvidia-smi -L &>/dev/null; then
    err "Keine GPU sichtbar. Container ohne --gpus all gestartet oder kein NVIDIA-Treiber?"
    exit 1
fi
nvidia-smi -L | sed 's/^/    /'
ok "GPU verfuegbar"
echo ""

# ── HuggingFace-Token + Cosmos-Lizenz-Preflight ────────────────────────────────
log "HuggingFace-Token pruefen"
if [[ -z "${HF_TOKEN:-}" ]]; then
    err "HF_TOKEN ist nicht gesetzt."
    err "  -e HF_TOKEN=hf_... beim docker run setzen."
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_TOKEN
ok "HF_TOKEN gesetzt (${#HF_TOKEN} Zeichen)"

# cosmos-transfer2.5 laedt Checkpoints ueber `uvx 'hf>=1.3.5' download ...`
# (checkpoint_db.py) — dieser Weg verlaesst sich primaer auf den LOKAL GESPEICHERTEN
# Login-Token (siehe `hf auth login`), nicht zuverlaessig auf die automatische
# HF_TOKEN-Erkennung ueber Umgebungsvariablen. Beobachtet 2026-08-24: der Python-seitige
# Lizenz-Preflight unten funktioniert (uebergibt token= explizit), aber `uvx hf download`
# schlug trotzdem mit "Access denied" fehl. Fix: Token EINMALIG explizit einloggen, damit
# er in $HF_HOME/token landet und jede spaetere `uvx hf ...`-Instanz (jedes Mal eine frische
# Umgebung) ihn zuverlaessig findet.
log "HF-Token bei 'hf' CLI einloggen (fuer uvx-basierte Checkpoint-Downloads)"
uvx 'hf>=1.3.5' auth login --token "$HF_TOKEN"
ok "Bei HuggingFace eingeloggt"

# Lieber HIER scheitern als nach Stunden Inferenz mit einem kryptischen 403 mitten im Lauf:
# die "NVIDIA Open Model License Agreement" muss VORHER manuell auf JEDER der folgenden
# HF-Modellseiten akzeptiert werden — das kann kein Skript automatisieren. cosmos-transfer2.5
# laedt zur Laufzeit MEHRERE separat gated Repos nach (per HF-Gating gilt Zustimmung nur
# pro Repo, nicht projektweit) — bislang zwei entdeckt:
#   1. nvidia/Cosmos-Transfer2.5-2B — die eigentlichen Modellgewichte
#   2. nvidia/Cosmos-Predict2.5-2B  — geteilte Basiskomponenten (u. a. Wan2.1-VAE/Tokenizer),
#      Absturz kam beim ersten echten Inferenzlauf (2026-08-24), erst dort sichtbar geworden
# nvidia/Cosmos-Guardrail1 wird bewusst NICHT geprueft — AUGMENT_DISABLE_GUARDRAILS=1
# (Default) laedt es gar nicht erst. Falls doch ein WEITERES gated Repo aus einer noch
# unentdeckten Ecke des Modells auftaucht, bricht der Lauf mit genau dieser Fehlerform
# ("Access denied. This repository requires approval.") wieder mitten in der Inferenz ab —
# dann hier ergaenzen.
#
# WICHTIG (entdeckt 2026-08-24): model_info() prueft nur Metadaten-Zugriff, NICHT
# Datei-Download-Zugriff — es liefert fuer ein gated Repo auch dann Erfolg, wenn der
# Account die Lizenz nie akzeptiert hat (Modellkarten-Metadaten sind oeffentlich lesbar,
# die eigentlichen Gewichts-Dateien nicht). Dieser Preflight kann DAHER faelschlich
# "OK" melden, obwohl der spaetere Datei-Download mit "Access denied. This repository
# requires approval." abbricht. Tatsaechliche Ursache in diesem Fall: die Lizenz war auf
# huggingface.co/<repo> nie erfolgreich akzeptiert (z. B. unter einem anderen Account als
# dem, der den Token besitzt — mit dem Token-Account einloggen und den "Agree and access
# repository"-Button auf der Modellseite pruefen/klicken). Zusaetzlich moeglich, aber
# seltener: ein "fine-grained" Token ohne das jeweilige Repo in seiner Freigabeliste.
# Direkter Test, unabhaengig vom Preflight: `curl -sI -H "Authorization: Bearer $HF_TOKEN"
# https://huggingface.co/<repo>/resolve/main/<eine_datei>` — 302/200 = Zugriff da, 403 = nicht.
log "Cosmos-Lizenzen pruefen (nvidia/Cosmos-Transfer2.5-2B, nvidia/Cosmos-Predict2.5-2B)"
if ! "$TOOLS_PYTHON" - <<'PYEOF'
import os
import sys

from huggingface_hub import HfApi
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

REQUIRED_REPOS = ["nvidia/Cosmos-Transfer2.5-2B", "nvidia/Cosmos-Predict2.5-2B"]

token = os.environ["HF_TOKEN"]
failed = []
for repo in REQUIRED_REPOS:
    try:
        HfApi().model_info(repo, token=token)
    except (GatedRepoError, HfHubHTTPError) as exc:
        print(f"Lizenz-Check fehlgeschlagen fuer {repo}: {exc}", file=sys.stderr)
        failed.append(repo)

if failed:
    sys.exit(1)
PYEOF
then
    err "Kein Zugriff auf mindestens eines der benoetigten gated HF-Repos (siehe oben)."
    err "  1. Fuer JEDES oben gemeldete Repo die HF-Seite oeffnen (huggingface.co/<repo>)"
    err "  2. 'NVIDIA Open Model License Agreement' akzeptieren (einmalig, mit demselben"
    err "     HF-Account, dessen Token hier als HF_TOKEN gesetzt ist)"
    err "  3. Container neu starten"
    exit 1
fi
ok "Lizenz-Zugriff bestaetigt"
echo ""

# ── Schritt 1/4 — Download + Konvertierung ─────────────────────────────────────
log "Schritt 1/4 — Datensatz-Download + v3.0→v2.1-Konvertierung"
if [[ "$SKIP_DOWNLOAD" == "1" && "$SKIP_CONVERT" == "1" ]]; then
    warn "SKIP_DOWNLOAD=1 und SKIP_CONVERT=1 — Schritt 1 uebersprungen."
else
    bash /scripts/download_source_dataset.sh
fi
if [[ ! -f "$DATASET_DIR/meta/modality.json" ]]; then
    err "meta/modality.json fehlt unter $DATASET_DIR — Download/Konvertierung unvollstaendig?"
    exit 1
fi
ok "Datensatz bereit: $DATASET_DIR"
echo ""

# ── Schritt 2/4 — Cosmos-Specs generieren ──────────────────────────────────────
# Kein eigener Control-Video-Schritt: Cosmos erzeugt die Edge-Kontrolle selbst on-the-fly
# aus dem Quellvideo (AUGMENT_EDGE_THRESHOLD), siehe Kommentar in build_controlnet_specs.py.
log "Schritt 2/4 — Cosmos-Specs generieren"
mkdir -p "$SPECS_DIR" "$RAW_OUTPUT_DIR"
"$TOOLS_PYTHON" /scripts/build_controlnet_specs.py \
    --dataset-dir "$DATASET_DIR" \
    --specs-dir "$SPECS_DIR" \
    --raw-output-dir "$RAW_OUTPUT_DIR" \
    --work-list "$WORK_LIST"
ok "Specs geschrieben: $SPECS_DIR (Arbeitsliste: $WORK_LIST)"
echo ""

if [[ "$SKIP_INFERENCE" == "1" ]]; then
    warn "SKIP_INFERENCE=1 — stoppe hier. Specs unter $SPECS_DIR pruefen, dann ohne"
    warn "SKIP_INFERENCE neu starten."
    exit 0
fi

# ── Schritt 3/4 — Cosmos-Inferenz ──────────────────────────────────────────────
log "Schritt 3/4 — Cosmos-Transfer2.5-Inferenz"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$DATA_DIR/logs/augment-$TIMESTAMP.log"
mkdir -p "$DATA_DIR/logs"

N_ITEMS="$("$TOOLS_PYTHON" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$WORK_LIST")"
log "Arbeitsliste: $N_ITEMS Eintraege (Episode x Kamera x Variante)"

"$TOOLS_PYTHON" - "$WORK_LIST" "$AUGMENT_OVERWRITE" <<'PYEOF' | tee -a "$LOG_FILE"
import json
import subprocess
import sys
from pathlib import Path

work_list_path, overwrite = sys.argv[1], sys.argv[2] == "1"
items = json.loads(Path(work_list_path).read_text())

for i, item in enumerate(items, 1):
    out_dir = Path(item["raw_output_dir"])
    done_marker = out_dir / ".done"
    if done_marker.exists() and not overwrite:
        print(f"[{i}/{len(items)}] uebersprungen (bereits vorhanden): {out_dir}")
        continue
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{i}/{len(items)}] Cosmos-Inferenz: {item['spec_path']} -> {out_dir}")
    subprocess.run(
        ["bash", "/scripts/run_inference.sh", item["spec_path"], str(out_dir)],
        check=True,
    )
    done_marker.touch()
PYEOF
ok "Inferenz abgeschlossen. Log: $LOG_FILE"
echo ""

if [[ "$SKIP_ASSEMBLE" == "1" ]]; then
    warn "SKIP_ASSEMBLE=1 — stoppe hier. Rohvideos unter $RAW_OUTPUT_DIR pruefen, dann ohne"
    warn "SKIP_ASSEMBLE neu starten."
    exit 0
fi

# ── Schritt 4/4 — Datensatz zusammenbauen ──────────────────────────────────────
log "Schritt 4/4 — LeRobot-v2.1-Datensatz zusammenbauen"
"$TOOLS_PYTHON" /scripts/assemble_dataset.py \
    --dataset-dir "$DATASET_DIR" \
    --work-list "$WORK_LIST" \
    --out-dir "$AUGMENT_OUT_DIR" \
    --strict-frame-check "$AUGMENT_STRICT_FRAME_CHECK"
ok "Datensatz geschrieben: $AUGMENT_OUT_DIR"

if [[ -n "$AUGMENT_HF_REPO" ]]; then
    log "Lade Datensatz nach HuggingFace hoch: $AUGMENT_HF_REPO"
    /opt/venv-tools/bin/huggingface-cli upload \
        "$AUGMENT_HF_REPO" "$AUGMENT_OUT_DIR" . --repo-type dataset
    ok "Hochgeladen: https://huggingface.co/datasets/$AUGMENT_HF_REPO"
fi
echo ""

log "Fertig."
echo "  Naechster Schritt — im Training-Image (Training/) als Co-Training-Datensatz nutzen:"
echo "    USE_COTRAIN=1 COTRAIN_DATASET_PATH=$AUGMENT_OUT_DIR COTRAIN_MIX_RATIO=0.25 \\"
echo "        ./Training/setup_and_train_DockerHub-pull.sh"
echo "  (oder COTRAIN_HF_REPO=$AUGMENT_HF_REPO, falls hochgeladen)"
echo "  Details: docs/training/co-training.md"
echo ""
