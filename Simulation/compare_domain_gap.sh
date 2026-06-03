#!/usr/bin/env bash
# compare_domain_gap.sh — montiert Dataset- vs. Sim-Policy-Kamera-Frames nebeneinander.
#
# Domain-Gap-Diagnose (Schritt 1): legt je Policy-Kamera das ECHTE Dataset-Referenzbild
# (Simulation/camera_reference/dataset_cam_*.png) links neben das SIM-Renderbild
# (sim_cam_*.png, erzeugt mit g1_dex3_sim/dump_policy_cams.py) rechts — beschriftet,
# als ein Vergleichsbild über alle vier Kameras.
#
# Ablauf:
#   1) Im Sim-Container (RT-Core-GPU) die Sim-Frames erzeugen:
#        unset VIRTUAL_ENV
#        ${ISAACLAB_PATH}/isaaclab.sh -p /workspace/g1_dex3_sim/dump_policy_cams.py \
#            --headless --enable_cameras \
#            --asset-path /data/checkpoints/<repo>/g1_dex3.usd \
#            --out-dir /data/sim_cam_frames
#   2) Die vier sim_cam_*.png herunterladen (z. B. nach ./sim_cam_frames).
#   3) Lokal montieren:
#        Simulation/compare_domain_gap.sh ./sim_cam_frames
#
# Argumente:
#   $1  SIM_DIR  (Verzeichnis mit sim_cam_*.png; Default: ./sim_cam_frames)
#   $2  OUT      (Ausgabe-PNG; Default: ./domain_gap_compare.png)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF_DIR="${REF_DIR:-$SCRIPT_DIR/camera_reference}"
SIM_DIR="${1:-./sim_cam_frames}"
OUT="${2:-./domain_gap_compare.png}"
FONT="${FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf}"

CAMS=(cam_left_high cam_right_high cam_left_wrist cam_right_wrist)

# ── Vorbedingungen ─────────────────────────────────────────────────────────────
missing=0
for cam in "${CAMS[@]}"; do
    [[ -f "$REF_DIR/dataset_${cam}.png" ]] || { echo "FEHLER: fehlt $REF_DIR/dataset_${cam}.png" >&2; missing=1; }
    [[ -f "$SIM_DIR/sim_${cam}.png"     ]] || { echo "FEHLT (Sim noch nicht erzeugt?): $SIM_DIR/sim_${cam}.png" >&2; missing=1; }
done
if [[ "$missing" == "1" ]]; then
    echo "" >&2
    echo "Sim-Frames zuerst im Sim-Container erzeugen (g1_dex3_sim/dump_policy_cams.py) und" >&2
    echo "die vier sim_cam_*.png nach '$SIM_DIR' legen. Siehe Kopf dieses Skripts." >&2
    exit 1
fi

drawtext() { echo "drawtext=fontfile=${FONT}:text='$1':x=10:y=10:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8"; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# ── Pro Kamera: Dataset (links) | Sim (rechts), beschriftet, gleiche Größe ─────
rows=()
for cam in "${CAMS[@]}"; do
    ffmpeg -y -loglevel error \
        -i "$REF_DIR/dataset_${cam}.png" \
        -i "$SIM_DIR/sim_${cam}.png" \
        -filter_complex "
            [0:v]scale=640:480,$(drawtext "DATASET (real) — ${cam}")[l];
            [1:v]scale=640:480,$(drawtext "SIM — ${cam}")[r];
            [l][r]hstack=inputs=2[o]" \
        -map "[o]" "$TMP/row_${cam}.png"
    rows+=("$TMP/row_${cam}.png")
done

# ── Vier Zeilen vertikal stapeln ───────────────────────────────────────────────
in_args=(); for r in "${rows[@]}"; do in_args+=(-i "$r"); done
ffmpeg -y -loglevel error "${in_args[@]}" \
    -filter_complex "[0:v][1:v][2:v][3:v]vstack=inputs=4[o]" -map "[o]" "$OUT"

echo "==> Vergleichsbild geschrieben: $OUT"
echo "    Links = echtes Dataset-Footage, rechts = Sim-Rendering — je Policy-Kamera."
