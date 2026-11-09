#!/usr/bin/env python3
# TL;DR: Bettet die Bilder (latex/figures, Simulation/camera_reference) und Messdaten (CSV) der
# Präsentation als base64 in presentation/assets/images.js und data.js ein, damit file:// genügt.
"""Aufruf (aus dem Repo-Wurzelverzeichnis):  python3 presentation/tools/build_assets.py

Braucht nur Pillow. Die erzeugten Dateien werden eingecheckt; erneut ausführen, wenn sich ein
Quellbild oder eine CSV unter latex/figures/daten/ ändert.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "presentation/assets"
FIG = ROOT / "latex/figures"
CAM = ROOT / "Simulation/camera_reference"

# Schlüssel -> (Quelle, maximale Breite, Format)
IMAGES = {
    "logo": (FIG / "logo_hochschule.jpg", 900, "PNG"),
    "cam_left_high": (CAM / "dataset_cam_left_high.png", 640, "JPEG"),
    "cam_right_high": (CAM / "dataset_cam_right_high.png", 640, "JPEG"),
    "cam_left_wrist": (CAM / "dataset_cam_left_wrist.png", 640, "JPEG"),
    "cam_right_wrist": (CAM / "dataset_cam_right_wrist.png", 640, "JPEG"),
    "real_left_high": (FIG / "sim/real_cam_left_high.png", 960, "JPEG"),
    "sim_left_high": (FIG / "sim/sim_cam_left_high.png", 960, "JPEG"),
    "sim70_left_high": (FIG / "sim/sim_scene70_cam_left_high.png", 960, "JPEG"),
    "griff": (FIG / "sim/griff_lauf88_ep05_80s.png", 960, "JPEG"),
    "griff_crop": (FIG / "sim/griff_lauf88_ep05_80s_ausschnitt.png", 960, "JPEG"),
    "cosmos_orig": (FIG / "cosmos/ep000_f450_original.png", 960, "JPEG"),
    "cosmos_var": (FIG / "cosmos/ep000_f450_variante0.png", 960, "JPEG"),
}

DATA = ROOT / "latex/figures/daten"
CSVS = [
    "closed_loop.csv",
    "gate_mse_haende.csv",
    "orakel_versatz_lauf62.csv",
    "val_mse_lauf3_lauf4.csv",
]
LOSS_CSV = "lernkurven_lauf3_lauf4.csv"
LOSS_EVERY = 10  # jede 10. Zeile reicht für eine Linie auf der Folie


def encode(path: Path, max_w: int, fmt: str) -> str:
    im = Image.open(path)
    im = im.convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    if fmt == "JPEG":
        im.save(buf, "JPEG", quality=86, optimize=True, progressive=True)
        mime = "image/jpeg"
    else:
        im.save(buf, "PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    imgs = {k: encode(*v) for k, v in IMAGES.items()}
    (ASSETS / "images.js").write_text(
        "// Generiert von presentation/tools/build_assets.py, nicht von Hand ändern.\n"
        "window.PRES_IMAGES = " + json.dumps(imgs, indent=0) + ";\n",
        encoding="utf-8",
    )
    csv = {name: (DATA / name).read_text(encoding="utf-8") for name in CSVS}
    lines = (DATA / LOSS_CSV).read_text(encoding="utf-8").splitlines()
    csv[LOSS_CSV] = "\n".join([lines[0]] + lines[1::LOSS_EVERY]) + "\n"
    (ASSETS / "data.js").write_text(
        "// Generiert von presentation/tools/build_assets.py aus latex/figures/daten/*.csv.\n"
        f"// {LOSS_CSV}: jede {LOSS_EVERY}. Zeile.\n"
        "window.PRES_CSV = " + json.dumps(csv, ensure_ascii=False, indent=0) + ";\n",
        encoding="utf-8",
    )
    for f in ("images.js", "data.js"):
        print(f, f"{(ASSETS / f).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
