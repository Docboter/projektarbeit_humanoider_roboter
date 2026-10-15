#!/usr/bin/env python3
# TL;DR: Host-Auswertung von 'server_rl_run.sh visgate': Tabelle MSE Hände je Prüfset gegen die Ausgangswerte.
"""Fasst die JSONs einer ``visgate``-Messung zu einer Tabelle zusammen.

Kennzahl wie Lauf 66/70/72: **Median der MSE Hände** über die Episoden, und zwar der echte
Median (bei 4 Episoden der Mittelwert der beiden mittleren). ``finger_span_openloop.py``
schreibt in ``median_mse_hand`` den *oberen* Median — deshalb hier aus den Episoden neu.

Ausgangswerte = die JSONs, mit denen Lauf 72 die heutigen Modelle gemessen hat
(``vis_gate_v3_cube{5,4}_n16.json`` / ``_n17.json`` im selben Ordner). Fehlen sie, steht
in der Spalte ein Strich.

Aufruf (macht ``server_rl_run.sh visgate``)::

    python3 visgate_summary.py --results-dir ~/groot-rl-data/sim_results \\
        --tag cotrain_cube5_ckpt44000 --sets vis_gate_v3_cube5 vis_gate_v3_cube4
"""

import argparse
import json
from pathlib import Path
import statistics
import sys

BASELINES = (("N1.6 @30000", "n16"), ("N1.7 @44000", "n17"))


def median_hand(path: Path) -> tuple[float, int] | None:
    if not path.is_file():
        return None
    rows = json.loads(path.read_text())["episodes"]
    vals = [r["mse_hand"] for r in rows if r.get("mse_hand") is not None]
    return (statistics.median(vals), len(vals)) if vals else None


def fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--tag", required=True, help="Kennung der Messung (Dateinamen-Teil)")
    p.add_argument("--sets", nargs="+", required=True, help="Prüfset-Namen unter /data/cotrain")
    args = p.parse_args()

    d = args.results_dir
    real = median_hand(d / f"visgate_{args.tag}_real.json")
    print("\n" + "=" * 72)
    print(f"Visuelles Gate — {args.tag}   (Median MSE Hände, teacher-forced Open Loop)")
    print("=" * 72)
    print(f"Echtbild, dieselben Episoden: {fmt(real[0] if real else None)}"
          + (f"  ({real[1]} Ep.)" if real else ""))
    print()
    head = f"{'Prüfset':<22}{'gemessen':>10}" + "".join(f"{n:>14}" for n, _ in BASELINES)
    print(head + f"{'× Echtbild':>12}")
    missing = False
    for s in args.sets:
        got = median_hand(d / f"visgate_{args.tag}_{s}.json")
        missing |= got is None
        base = [median_hand(d / f"{s}_{suffix}.json") for _, suffix in BASELINES]
        cells = "".join(
            f"{fmt(b[0] if b else None):>8}"
            + (f" {got[0] / b[0]:>4.0%}" if got and b else "     ")
            for b in base
        )
        factor = f"×{got[0] / real[0]:.1f}" if got and real else "—"
        print(f"{s:<22}{fmt(got[0] if got else None):>10}{cells}{factor:>12}")
    print()
    print("Prozent = gemessen relativ zum Ausgangswert. Ein Closed Loop lohnt erst, wenn")
    print("das Modell die Ausgangswerte DEUTLICH unterbietet (Auftrag v3, Lauf 72).")
    print("=" * 72)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
