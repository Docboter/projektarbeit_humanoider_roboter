#!/usr/bin/env python3
# TL;DR: Host-Werkzeug — Gate-Kennzahl je Checkpoint: Mittel der Seed-Mediane MSE Hände (Bereich über die Seeds) für Echtbild und beide Prüfsets.
"""Fasst die ``visgate``-JSONs mehrerer Seeds je Checkpoint zusammen (Protokoll Lauf 80).

Je Seed der echte Median der MSE Hände über die Episoden (bei 4 Episoden der Mittelwert der
beiden mittleren, wie ``Simulation/scripts/visgate_summary.py``), dann Mittel und Bereich über
die Seeds. Dazu der Arm-Fehler zur Kontrolle.

Aufruf::

    python3 tools/sim_visgate_seeds_summary.py ~/groot-rl-data/sim_results \\
        cotrain-v3-cube4-ckpt44000 cotrain-v3-cube4-n17-ckpt44000
"""

import glob
import json
import statistics
import sys

SETS = ["real", "vis_gate_v3_cube5", "vis_gate_v3_cube4"]


def median_of(path: str, key: str) -> float:
    return statistics.median(e[key] for e in json.load(open(path))["episodes"])


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    d = sys.argv[1]
    for tag in sys.argv[2:]:
        print(f"\n{tag}")
        for s in SETS:
            fs = sorted(glob.glob(f"{d}/visgate_{tag}_s[0-9]_{s}.json"))
            if not fs:
                print(f"  {s:18s} —")
                continue
            h = [median_of(f, "mse_hand") for f in fs]
            a = [median_of(f, "mse_arm") for f in fs]
            print(
                f"  {s:18s} MSE Hände {sum(h) / len(h):.4f} ({min(h):.4f}–{max(h):.4f}), "
                f"Arme {sum(a) / len(a):.4f}, Seeds {len(fs)}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
