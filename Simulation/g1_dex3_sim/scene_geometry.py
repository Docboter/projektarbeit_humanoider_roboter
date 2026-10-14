# TL;DR: Eine Quelle für Tisch-, Würfel- und Basisgeometrie (Dex3-Env, Gripper-Baseline, Layout).
"""Szenengeometrie, die mehrere Envs und die Bild-Rückprojektion teilen.

Bis 2026-10-13 standen diese Zahlen einzeln in jeder Datei. Die Baseline-Gripper-Sim
behielt dabei die Geometrie von vor dem Fix vom 2026-08-24 (Tisch 0,89, Würfel 0,915,
Becken 0,85), Baseline- und Dex3-Zahlen waren damit nicht vergleichbar (Review §3.5).

Isaac-frei, damit auch die CPU-Werkzeuge (extract_block_layout.py, Tests) es importieren.
Die Kamerapose steht bewusst NICHT hier, sondern in camera_geometry.py — sie ist gegen
gerenderte Bilder kalibriert und wandert nicht automatisch mit (s. dort, OFFEN / §3.4).
"""

from __future__ import annotations

import os


# Tischplatte: Oberfläche bei z = TABLE_HEIGHT, Box mit Mittelpunkt auf halber Höhe.
TABLE_HEIGHT = 0.87
TABLE_SIZE = (0.8, 0.6, TABLE_HEIGHT)
TABLE_POS = (0.5, 0.0, TABLE_HEIGHT / 2)

# 5-cm-Würfel, Mittelpunkt dort, wo er auf dem Tisch ruht (nachgemessen 2026-08-22).
# SIM_CUBE_EDGE nur für Diagnose-Sweeps (Würfelgröße im Orakel, Chronik Lauf 57); die
# Bild-Rückprojektion (extract_block_layout.py) rechnet weiter mit 5 cm.
CUBE_EDGE = float(os.environ.get("SIM_CUBE_EDGE", "0.05"))
CUBE_Z_CENTER = round(TABLE_HEIGHT + CUBE_EDGE / 2, 4)   # 0.895

# Roboterbasis (Becken), env-lokal. Beide Werte gemessen (Fingerkuppen-Projektion ins
# Realbild, Läufe 50/51); per Umgebungsvariable auf die alten Werte (0 / 0.85) umstellbar,
# um ältere Läufe nachzustellen.
ROBOT_BASE_X = float(os.environ.get("ROBOT_BASE_X", "-0.057"))
ROBOT_BASE_Z = float(os.environ.get("ROBOT_BASE_Z", "0.764"))
ROBOT_BASE_POS = (ROBOT_BASE_X, 0.0, ROBOT_BASE_Z)

# Alte Basishöhe (Becken env-lokal z = 0,85, vor dem Fix vom 2026-08-24). Kein Szenenwert
# mehr, sondern der Bezug, gegen den die Kopfkamera in camera_geometry.py kalibriert ist:
# deren Augpunkt/Blickziel stammen aus der URDF-Kette bei Becken-z 0,85. Nur
# HEAD_CAM_FOLLOWS_PELVIS=1 verschiebt die Kamera um ROBOT_BASE_Z − diesen Wert.
# NICHT ändern, solange die Kamerapose nicht neu kalibriert ist.
CAMERA_CALIB_BASE_Z = 0.85
