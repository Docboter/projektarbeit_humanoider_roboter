#!/usr/bin/env python3
"""
Overlay-Verfahren für Wrist-Kamera-Kalibrierung.

Legt Real-Frames halbtransparent über Sim-Frames, um die Kamera-Ausrichtung
visuell zu überprüfen. Wenn die Finger des simulierten Grippers exakt über
den realen Fingern liegen, stimmt die Kamera-Pose.

Verwendung:
    python Simulation/scripts/overlay_camera_check.py \
        --sim-dir Simulation/runs/202606002/new_run/

Ausgabe: /tmp/overlay_check/panel_cam_*.png  (Real | Overlay | Sim)

Iteration:
    1. overlay_camera_check.py → sieht Misalignment
    2. g1_dex3_cfg.py anpassen (wrist_eye / left_wrist_target / right_wrist_target)
    3. g1_dex3_cfg.py + g1_dex3_blockstack_env.py per scp auf vast.ai
    4. capture_camera_frames.sh auf vast.ai → neue _debug_obs_cam_*.png
    5. Frames herunterladen → overlay_camera_check.py erneut ausführen
    6. Wiederholen bis Finger exakt übereinander liegen
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Kamera-Paare: (real_filename, sim_filename, label)
# ---------------------------------------------------------------------------

CAMERA_PAIRS = [
    ("dataset_cam_left_wrist.png",  "_debug_obs_cam_left_wrist.png",  "cam_left_wrist"),
    ("dataset_cam_right_wrist.png", "_debug_obs_cam_right_wrist.png", "cam_right_wrist"),
    ("dataset_cam_left_high.png",   "_debug_obs_cam_left_high.png",   "cam_left_high"),
    ("dataset_cam_right_high.png",  "_debug_obs_cam_right_high.png",  "cam_right_high"),
]

PANEL_SIZE = (640, 480)
GAP = 8


def tint(img: Image.Image, r: float, g: float, b: float) -> Image.Image:
    """Multipliziert die RGB-Kanäle mit den angegebenen Faktoren (0–1 dämpft, >1 verstärkt)."""
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    arr[:, :, 0] *= r
    arr[:, :, 1] *= g
    arr[:, :, 2] *= b
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def blend(real: Image.Image, sim: Image.Image, alpha: float) -> Image.Image:
    """
    Alpha-Blend: real (grün-getönt) über sim (rot-getönt).
    Perfekte Überlappung → gelblich. Misalignment → grüner oder roter Geist.
    """
    size = sim.size
    real_r = real.resize(size, Image.LANCZOS)

    real_tinted = tint(real_r, r=0.5, g=1.0, b=0.5)   # grün = Real
    sim_tinted  = tint(sim,    r=1.0, g=0.5, b=0.5)   # rot  = Sim

    r_arr = np.array(real_tinted, dtype=np.float32)
    s_arr = np.array(sim_tinted,  dtype=np.float32)
    blended = (alpha * r_arr + (1.0 - alpha) * s_arr).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def make_panel(real: Image.Image, sim: Image.Image, overlay: Image.Image, label: str) -> Image.Image:
    """3-Panel: Real | Overlay (50 %) | Sim — mit Beschriftung."""
    w, h = PANEL_SIZE
    total_w = w * 3 + GAP * 2
    total_h = h + 28

    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    canvas.paste(real.resize(PANEL_SIZE, Image.LANCZOS),    (0,              28))
    canvas.paste(overlay.resize(PANEL_SIZE, Image.LANCZOS), (w + GAP,        28))
    canvas.paste(sim.resize(PANEL_SIZE, Image.LANCZOS),     (w * 2 + GAP * 2, 28))

    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6),
              f"{label}   ←Real (grün)  |  Overlay 50%  |  Sim (rot)→   "
              f"Ziel: Finger übereinanderlegend = gelb",
              fill=(220, 220, 220))
    # Panel-Beschriftungen
    for x, txt in [(4, "REAL"), (w + GAP + 4, "OVERLAY"), (w * 2 + GAP * 2 + 4, "SIM")]:
        draw.text((x, 30), txt, fill=(180, 180, 180))

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Kamera-Overlay für visuelle Kalibrierung")
    parser.add_argument(
        "--real-dir", default="Simulation/camera_reference/",
        help="Verzeichnis mit den Real-Referenzbildern (dataset_cam_*.png)"
    )
    parser.add_argument(
        "--sim-dir", required=True,
        help="Verzeichnis mit den Sim-Debug-Frames (_debug_obs_cam_*.png)"
    )
    parser.add_argument(
        "--out-dir", default="/tmp/overlay_check/",
        help="Ausgabeverzeichnis für die Panel-Bilder"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Gewicht des Real-Bildes im Overlay (0=nur Sim, 1=nur Real, default=0.5)"
    )
    args = parser.parse_args()

    real_dir = Path(args.real_dir)
    sim_dir  = Path(args.sim_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Real-Dir:  {real_dir}")
    print(f"Sim-Dir:   {sim_dir}")
    print(f"Out-Dir:   {out_dir}")
    print(f"Alpha:     {args.alpha}  (Real-Gewicht)")
    print()

    for real_name, sim_name, label in CAMERA_PAIRS:
        real_path = real_dir / real_name
        sim_path  = sim_dir  / sim_name

        if not real_path.exists():
            print(f"  [SKIP] Real-Frame nicht gefunden: {real_path}")
            continue
        if not sim_path.exists():
            print(f"  [SKIP] Sim-Frame nicht gefunden: {sim_path}")
            continue

        real_img = Image.open(real_path).convert("RGB")
        sim_img  = Image.open(sim_path).convert("RGB")

        overlay = blend(real_img, sim_img, args.alpha)
        panel   = make_panel(real_img, sim_img, overlay, label)

        panel_path   = out_dir / f"panel_{label}.png"
        overlay_path = out_dir / f"overlay_{label}.png"
        panel.save(panel_path)
        overlay.save(overlay_path)
        print(f"  {label:25s} → {panel_path}")

    print(f"\nFertig. Panel-Bilder unter: {out_dir}")
    print("Tipp: Im Overlay sollten die Finger GELB erscheinen (rot+grün), wenn die Pose stimmt.")


if __name__ == "__main__":
    main()
