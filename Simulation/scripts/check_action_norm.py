#!/usr/bin/env python3
"""Statischer Check der Aktions-Normalisierungs-Statistiken eines GR00T-Checkpoints.

Kein GPU, kein torch, kein gr00t-Import - nur json + numpy.

Usage:
    python check_action_norm.py <checkpoint_dir> [<dataset_dir>] [--embodiment new_embodiment]
"""

import argparse
import json
from pathlib import Path

import numpy as np

# Reihenfolge/Slices wie in examples/G1_DEX3/modality_4cam.json
G1_DEX3_SLICES = {
    "left_arm": (0, 7),
    "right_arm": (7, 14),
    "left_dex3": (14, 21),
    "right_dex3": (21, 28),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("dataset", nargs="?", default=None,
                    help="optional: LeRobot-Dataset-Root (enthaelt meta/stats.json)")
    ap.add_argument("--embodiment", default="new_embodiment")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    stats = json.loads((ckpt / "statistics.json").read_text())
    cfg = json.loads((ckpt / "processor_config.json").read_text())["processor_kwargs"]

    print(f"checkpoint            : {ckpt}")
    print(f"embodiments in stats  : {list(stats)}")
    print(f"use_percentiles       : {cfg['use_percentiles']}   (False -> min/max, True -> q01/q99)")
    print(f"clip_outliers         : {cfg['clip_outliers']}")
    print(f"use_relative_action   : {cfg['use_relative_action']}")
    print()

    emb = stats[args.embodiment]
    lo_key, hi_key = ("q01", "q99") if cfg["use_percentiles"] else ("min", "max")

    mcfg = cfg["modality_configs"][args.embodiment]["action"]
    reps = {k: c["rep"] for k, c in zip(mcfg["modality_keys"], mcfg["action_configs"] or [])}

    print(f"=== action-Normalisierungs-Bounds ({lo_key}/{hi_key}) ===")
    for group in mcfg["modality_keys"]:
        s = emb["action"][group]
        lo, hi = np.asarray(s[lo_key]), np.asarray(s[hi_key])
        span = hi - lo
        rep = reps.get(group, "?")
        used_rel = rep == "RELATIVE" and cfg["use_relative_action"]
        src = "relative_action" if used_rel else "action"
        if used_rel:
            r = emb["relative_action"][group]
            lo, hi = np.asarray(r[lo_key]), np.asarray(r[hi_key])
            span = hi - lo  # (T, D) bei relativen Stats
        print(f"\n{group}  rep={rep}  -> Norm-Quelle: {src}   shape={lo.shape}")
        print(f"  {lo_key:>4s}  = {np.round(np.atleast_2d(lo)[0], 4)}")
        print(f"  {hi_key:>4s}  = {np.round(np.atleast_2d(hi)[0], 4)}")
        print(f"  span  = {np.round(np.atleast_2d(span)[0], 4)}")
        print(f"  -> max. de-normalisierbare Spanne pro Dim (norm. -1..+1): "
              f"{np.round(np.atleast_2d(span)[0].max(), 4)} rad")

    # Optional: gegen die Dataset-Stats gegenpruefen
    if args.dataset:
        ds = json.loads((Path(args.dataset) / "meta" / "stats.json").read_text())["action"]
        print("\n=== Abgleich Checkpoint-Stats <-> Dataset meta/stats.json ===")
        for group, (a, b) in G1_DEX3_SLICES.items():
            s = emb["action"][group]
            ok_lo = np.allclose(np.asarray(s[lo_key]), np.asarray(ds[lo_key])[a:b], atol=1e-5)
            ok_hi = np.allclose(np.asarray(s[hi_key]), np.asarray(ds[hi_key])[a:b], atol=1e-5)
            print(f"  {group:11s} {lo_key}: {'OK' if ok_lo else 'ABWEICHUNG'}   "
                  f"{hi_key}: {'OK' if ok_hi else 'ABWEICHUNG'}")

    # Konkreter Sanity-Check fuer die Finger-Stauchung
    print("\n=== Finger-Check: welche norm. Amplitude entspraeche 0.39 rad? ===")
    for group in ("left_dex3", "right_dex3"):
        s = emb["action"][group]
        span = np.asarray(s[hi_key]) - np.asarray(s[lo_key])
        widest = span.max()
        print(f"  {group}: max span = {widest:.4f} rad -> 0.39 rad entspricht "
              f"{2 * 0.39 / widest:.3f} von 2.0 normierten Einheiten "
              f"({100 * 0.39 / widest:.1f} % des Wertebereichs)")


if __name__ == "__main__":
    main()
