#!/usr/bin/env python3
# TL;DR: Host-Werkzeug — fasst Closed-Loop-JSONs (sim_results/results_*.json) zur Tabelle wie Chronik Lauf 82/88 zusammen, mit exaktem 95-%-KI.
"""Closed-Loop-Zusammenfassung je Ergebnis-JSON.

Gibt je Datei aus: Stufen ``reached`` … ``stacked`` (k/n, Clopper-Pearson-KI 95 %),
Fingerspanne Median (Bereich), kleinster Abstand Kuppe↔Würfel (Median, bester), größte
Anhebung, Episoden mit ≥ 4 cm Verschiebung und die Episoden mit Anhebung (für den Videobefund).

Artefakt-Verdacht (Würfel weggeschleudert): Verschiebung > 0,5 m oder Anhebung > 10 cm. Diese
Episoden zählen nicht als ``lifted``; ob es wirklich ein Wurf ist, entscheidet das Video.
``lifted_clear`` > 0 braucht ebenfalls den Videobefund: Die Metrik zählt auch einen Würfel,
der in der Hand auf einem anderen Würfel liegt (Chronik Lauf 94).

Aufruf::

    python3 tools/sim_cl_summary.py ~/groot-rl-data/sim_results/results_<…>.json [...]
"""

import json
import math
import statistics as st
import sys


STAGES = ["reached", "touched", "lifted", "lifted_clear", "stacked"]


def _cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def _bisect(pred, lo: float = 0.0, hi: float = 1.0) -> float:
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if pred(m) else (lo, m)
    return (lo + hi) / 2


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    """Exaktes zweiseitiges 95-%-KI in Prozent."""
    lo = 0.0 if k == 0 else _bisect(lambda p: 1 - _cdf(k - 1, n, p) < 0.025)
    hi = 1.0 if k == n else _bisect(lambda p: _cdf(k, n, p) > 0.025)
    return lo * 100, hi * 100


def is_artifact(e: dict) -> bool:
    return e["block_shift_max_m"] > 0.5 or e["max_cube_lift_cm"] > 10


def summarize(eps: list[dict], title: str) -> None:
    n = len(eps)
    art = {e["episode"] for e in eps if is_artifact(e)}
    ok = [e for e in eps if e["episode"] not in art]
    print(f"\n### {title}  (n={n}, Artefakt-Verdacht: {len(art)})")
    for m in STAGES:
        raw = sum(e["milestones"][m] for e in eps)
        k = sum(e["milestones"][m] for e in ok)
        lo, hi = clopper_pearson(k, n)
        print(f"  {m:13s} {k:2d}/{n} ohne Artefakte (roh {raw:2d})  KI {lo:.1f}–{hi:.1f} %")
    sp = [e["finger_span_max_rad"] for e in eps]
    mr = [e["min_reach_m"] * 100 for e in eps]
    print(f"  Fingerspanne Median {st.median(sp):.2f} rad ({min(sp):.2f}–{max(sp):.2f})")
    print(
        f"  kleinster Abstand Kuppe↔Würfel Median {st.median(mr):.1f} cm (bester {min(mr):.1f})"
    )
    if ok:
        print(
            f"  größte Anhebung ohne Artefakte {max(e['max_cube_lift_cm'] for e in ok):.1f} cm, "
            f"Unterseite mit Hand max. {max(e['max_clear_lift_cm'] for e in ok):.1f} cm"
        )
    print(f"  Verschiebung ≥ 4 cm in {sum(e['block_shift_max_m'] >= 0.04 for e in eps)} Ep.")
    for e in eps:
        flag = "ARTEFAKT? " if e["episode"] in art else ""
        if e["episode"] in art or e["milestones"]["lifted"] or e["max_clear_lift_cm"] >= 1.5:
            print(
                f"  {flag}Ep. {e['episode']:2d} Layout {e['layout_episode']} "
                f"Anhebung {e['max_cube_lift_cm']:.2f} cm, Unterseite mit Hand "
                f"{e['max_clear_lift_cm']:.2f} cm, lifted_clear={e['milestones']['lifted_clear']}, "
                f"Versatz max {e['block_shift_max_m'] * 100:.1f} cm"
            )


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    pooled: list[dict] = []
    for f in sys.argv[1:]:
        d = json.load(open(f))
        ck = d["checkpoint"]
        eps = d["episodes"]
        title = (
            f"{f.split('/')[-1]}  N{d['groot_version']} {ck['run_id']} @{ck['global_step']}, "
            f"{d['num_valid']}/{d['num_episodes_requested']} gültig, "
            f"{d['num_aborted']} abgebrochen"
        )
        summarize(eps, title)
        pooled += eps
    if len(sys.argv) > 2:
        summarize(pooled, "GEPOOLT über alle Dateien (nur bei identischem Protokoll sinnvoll)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
