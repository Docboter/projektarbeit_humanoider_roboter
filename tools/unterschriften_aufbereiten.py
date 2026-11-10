#!/usr/bin/env python3
# TL;DR: Macht die eingescannten Unterschriften (latex/figures/signes/*.png) druckfertig: weißer Hintergrund wird transparent, schwarze Hilfslinien/gedruckte Namen fallen weg, Rand wird beschnitten; Ausgabe *_frei.png im selben, git-ignorierten Ordner.
"""Unterschriften für die ehrenwörtliche Erklärung aufbereiten.

Die Bilder selbst liegen NUR lokal in latex/figures/signes/ und sind über latex/.gitignore
ausgeschlossen (das Repo ist öffentlich). Dieses Skript enthält keine Unterschrift.

Aufruf (vom Repo-Wurzelverzeichnis):  python3 tools/unterschriften_aufbereiten.py
"""

from pathlib import Path

from PIL import Image

ORDNER = Path(__file__).resolve().parent.parent / "latex" / "figures" / "signes"
GRAU_MAX = 60  # max(R,G,B) - min(R,G,B) darunter gilt ein Pixel als grau/schwarz
RAND = 4  # px Rand nach dem Beschnitt
WEISS = 232  # ab dieser Helligkeit gilt ein Pixel als Papier (Scans sind nicht reinweiß)
# Bilder mit eingezeichneter schwarzer Linie und gedrucktem Namen: hier zählt nur farbige Tinte.
# Bei allen anderen bleibt jede Tinte erhalten (Scans mit fast grauer Tinte).
NUR_FARBIG = {"Matthias"}


def aufbereiten(quelle: Path, nur_farbig: bool) -> Path:
    bild = Image.open(quelle).convert("RGBA")
    px = bild.load()
    breite, hoehe = bild.size
    for y in range(hoehe):
        for x in range(breite):
            r, g, b, _ = px[x, y]
            hell = min(r, g, b)
            deckung = 255 - min(255, round(hell * 255 / WEISS))
            # Nur farbige Tinte behalten: graue/schwarze Pixel (Hilfslinie, Name) fallen weg.
            if nur_farbig and max(r, g, b) - hell < GRAU_MAX:
                deckung = 0
            if deckung == 0:
                px[x, y] = (255, 255, 255, 0)
                continue
            # Weiß herausrechnen, damit Kanten nicht blass werden
            a = deckung / 255
            px[x, y] = tuple(max(0, min(255, round((c - (1 - a) * 255) / a))) for c in (r, g, b)) + (
                deckung,
            )
    box = bild.getchannel("A").point(lambda v: 255 if v > 24 else 0).getbbox()
    if box:
        box = (
            max(0, box[0] - RAND),
            max(0, box[1] - RAND),
            min(breite, box[2] + RAND),
            min(hoehe, box[3] + RAND),
        )
        bild = bild.crop(box)
    ziel = quelle.with_name(quelle.stem + "_frei.png")
    bild.save(ziel)
    return ziel


if __name__ == "__main__":
    quellen = sorted(p for p in ORDNER.glob("*.png") if not p.stem.endswith("_frei"))
    if not quellen:
        raise SystemExit(f"Keine Unterschriften in {ORDNER}")
    for q in quellen:
        print(q.name, "->", aufbereiten(q, q.stem in NUR_FARBIG).name)
