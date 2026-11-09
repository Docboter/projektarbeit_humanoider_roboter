#!/usr/bin/env python3
# TL;DR: Verschiebt die Würfel einer Layout-JSON gespiegelt nach außen (y += dy·sign(y), x += dx); erzeugt z. B. layout_heldout_dy3.json aus layout_heldout.json (Korrektur aus Lauf 62).
"""Würfellagen eines Layouts verschieben (Korrektur Würfel↔Hand, Chronik Lauf 62).

Jeder Würfel jeder Episode wird um ``dx`` (vom Roboter weg) und um ``dy`` **nach außen**
verschoben: Würfel links der Mitte (y > 0) nach +y, rechts (y < 0) nach −y. Ein Würfel mit
y == 0 bleibt seitlich, wo er ist. ``LAYOUT_BIAS`` kann das nicht, es verschiebt gleichsinnig.

Beispiele (auf dem Sim-Server, Datenverzeichnis ~/groot-rl-data = /data im Container):

    python3 tools/layout_shift.py ~/groot-rl-data/eval/layout_heldout.json \\
        ~/groot-rl-data/eval/layout_heldout_dy3.json --dy 0.03
    python3 tools/layout_shift.py <layout_aller_trainings_ep>.json \\
        ~/groot-rl-data/cotrain/layout_all240_dy3.json --dy 0.03

Nur ``episodes.<ep>.cubes`` wird geändert; alle anderen Felder bleiben, der Kopf bekommt
``shift_m = {"dx": .., "dy": ..}``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys


def shift_layout(layout: dict, dx: float, dy: float) -> int:
    """Verschiebt in place, gibt die Zahl verschobener Würfel zurück."""
    n = 0
    for ep in layout.get("episodes", {}).values():
        new = []
        for x, y in ep.get("cubes", []):
            side = math.copysign(1.0, y) if y != 0 else 0.0
            new.append([round(x + dx, 4), round(y + dy * side, 4)])
            n += 1
        ep["cubes"] = new
    prev = layout.get("shift_m") or {"dx": 0.0, "dy": 0.0}
    layout["shift_m"] = {"dx": round(prev["dx"] + dx, 4), "dy": round(prev["dy"] + dy, 4)}
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="Eingangs-Layout (JSON)")
    ap.add_argument("dst", help="Ausgabe-Layout (JSON)")
    ap.add_argument("--dy", type=float, default=0.03, help="nach außen, m (Default 0.03)")
    ap.add_argument("--dx", type=float, default=0.0, help="vom Roboter weg, m (Default 0)")
    a = ap.parse_args()
    if a.src == a.dst:
        print("Fehler: src und dst sind gleich; Original nicht überschreiben.", file=sys.stderr)
        return 2
    with open(a.src) as f:
        layout = json.load(f)
    n = shift_layout(layout, a.dx, a.dy)
    with open(a.dst, "w") as f:
        json.dump(layout, f, indent=2, ensure_ascii=False)
    print(f"{n} Würfel in {len(layout.get('episodes', {}))} Episoden verschoben → {a.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
