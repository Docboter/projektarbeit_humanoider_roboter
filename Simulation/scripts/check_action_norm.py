#!/usr/bin/env python3
"""Statischer Check der Aktions-Normalisierungs-Statistiken eines GR00T-Checkpoints.

Wozu
----
Die BC-Policy kommandiert im Closed Loop nur ~0,39 rad Fingerspanne, wo die
Demonstration 2,09 rad enthält (Lauf 30). Eine mögliche Erklärung wäre, dass
nicht das Modell, sondern die **De-Normalisierung** staucht — falsche oder
vertauschte Statistiken im Checkpoint. Dieses Skript schließt das aus oder
weist es nach, bevor ein TUNE_VISUAL-Lauf über ~48 GPU-Stunden gestartet wird.

Geprüft wird:
  1. Welches Schema gilt (min/max gegen q01/q99, relativ gegen absolut)?
  2. Wie groß ist die maximal de-normalisierbare Spanne je Aktionsgruppe?
  3. Stimmen die Checkpoint-Statistiken mit `meta/stats.json` des Datensatzes?

**Nur Standardbibliothek** — kein numpy, kein torch, kein gr00t-Import, keine
GPU. Damit läuft es auch dort, wo nichts installiert werden kann: direkt auf dem
Host, ohne Container. Genau dafür ist es gedacht.

Usage
-----
    python3 check_action_norm.py <checkpoint_dir> [<dataset_dir>] [--embodiment new_embodiment]

Auf dem Server liegen die Container-Pfade unter $RL_HOST_DATA_DIR, z. B.
    python3 check_action_norm.py \\
        ~/project/data/RL/checkpoints/groot-g1dex3-checkpoint \\
        ~/project/data/RL/unitreerobotics/G1_Dex3_BlockStacking_Dataset
"""

import argparse
import json
from pathlib import Path

# Reihenfolge/Slices wie in examples/G1_DEX3/modality_4cam.json
G1_DEX3_SLICES = {
    "left_arm": (0, 7),
    "right_arm": (7, 14),
    "left_dex3": (14, 21),
    "right_dex3": (21, 28),
}

# Referenz aus der menschlichen Demonstration (replay_episode0.npz, Lauf 29).
DEMO_FINGER_SPAN_RAD = 2.094
# Was die Policy im Closed Loop tatsächlich kommandiert (Lauf 30, Median über 5 Episoden).
OBSERVED_FINGER_SPAN_RAD = 0.43


def first_row(values):
    """Erste Zeile einer evtl. verschachtelten Liste — relative Stats sind (T, D)."""
    return values[0] if values and isinstance(values[0], list) else values


def close(a, b, atol=1e-5):
    return len(a) == len(b) and all(abs(x - y) <= atol for x, y in zip(a, b))


def fmt(row, nd=4):
    return "[" + " ".join(f"{v:.{nd}f}" for v in row) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint")
    ap.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="optional: LeRobot-Dataset-Root (enthaelt meta/stats.json)",
    )
    ap.add_argument("--embodiment", default="new_embodiment")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    stats = json.loads((ckpt / "statistics.json").read_text())
    cfg = json.loads((ckpt / "processor_config.json").read_text())["processor_kwargs"]

    print(f"checkpoint            : {ckpt}")
    print(f"embodiments in stats  : {list(stats)}")
    print(f"use_percentiles       : {cfg['use_percentiles']}   (False -> min/max, True -> q01/q99)")
    print(f"clip_outliers         : {cfg['clip_outliers']}   (clippt auf [-1,1], staucht nicht)")
    print(f"use_relative_action   : {cfg['use_relative_action']}")
    print()

    emb = stats[args.embodiment]
    lo_key, hi_key = ("q01", "q99") if cfg["use_percentiles"] else ("min", "max")

    mcfg = cfg["modality_configs"][args.embodiment]["action"]
    reps = {k: c["rep"] for k, c in zip(mcfg["modality_keys"], mcfg["action_configs"] or [])}

    print(f"=== action-Normalisierungs-Bounds ({lo_key}/{hi_key}) ===")
    for group in mcfg["modality_keys"]:
        rep = reps.get(group, "?")
        # RELATIVE Gruppen werden gegen relative_action normalisiert, ABSOLUTE gegen action.
        # Bei G1_DEX3 sind die Arme RELATIVE, die Finger ABSOLUTE — für die Finger-Frage
        # zählen also die absoluten Stats.
        used_rel = rep == "RELATIVE" and cfg["use_relative_action"]
        src = "relative_action" if used_rel else "action"
        s = emb[src][group]
        lo, hi = first_row(s[lo_key]), first_row(s[hi_key])
        span = [h - a for a, h in zip(lo, hi)]
        print(f"\n{group}  rep={rep}  -> Norm-Quelle: {src}   dims={len(lo)}")
        print(f"  {lo_key:>4s}  = {fmt(lo)}")
        print(f"  {hi_key:>4s}  = {fmt(hi)}")
        print(f"  span  = {fmt(span)}")
        print(f"  -> max. de-normalisierbare Spanne (norm. -1..+1): {max(span):.4f} rad")

    if args.dataset:
        ds = json.loads((Path(args.dataset) / "meta" / "stats.json").read_text())["action"]
        print("\n=== Abgleich Checkpoint-Stats <-> Dataset meta/stats.json ===")
        for group, (a, b) in G1_DEX3_SLICES.items():
            s = emb["action"][group]
            ok_lo = close(first_row(s[lo_key]), ds[lo_key][a:b])
            ok_hi = close(first_row(s[hi_key]), ds[hi_key][a:b])
            print(
                f"  {group:11s} {lo_key}: {'OK' if ok_lo else 'ABWEICHUNG'}   "
                f"{hi_key}: {'OK' if ok_hi else 'ABWEICHUNG'}"
            )

    # Der eigentliche Punkt: reicht der de-normalisierbare Bereich für einen Griff?
    print("\n=== Finger-Check ===")
    print(
        f"Demonstration: {DEMO_FINGER_SPAN_RAD:.2f} rad | "
        f"Policy im Closed Loop: {OBSERVED_FINGER_SPAN_RAD:.2f} rad"
    )
    for group in ("left_dex3", "right_dex3"):
        s = emb["action"][group]
        lo, hi = first_row(s[lo_key]), first_row(s[hi_key])
        widest = max(h - a for a, h in zip(lo, hi))
        pct = 100 * OBSERVED_FINGER_SPAN_RAD / widest
        print(
            f"  {group}: max span {widest:.4f} rad -> die beobachteten "
            f"{OBSERVED_FINGER_SPAN_RAD:.2f} rad sind {pct:.1f} % des Wertebereichs "
            f"({2 * OBSERVED_FINGER_SPAN_RAD / widest:.3f} von 2.0 normierten Einheiten)"
        )
    print()
    print("Lesart: deckt sich 'max span' mit der Demonstration und stimmen die Stats mit")
    print("  dem Datensatz, kann die De-Normalisierung nicht die Ursache sein — dann ist")
    print("  der Modellausgang selbst gestaucht (-> Aktion 'span' auf echten Bildern).")


if __name__ == "__main__":
    main()
