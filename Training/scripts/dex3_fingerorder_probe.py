#!/usr/bin/env python3
# TL;DR: Belegt, dass die Reihenfolge der DEX3-Handgelenke NICHT aus Kinematik bestimmbar ist —
# Rückgewinnungstest am echten Datensatz mit bekannter Wahrheit.
"""dex3_fingerorder_probe.py — kann Vorwärtskinematik die Fingerreihenfolge bestimmen?

Warum es das gibt
─────────────────
``Fichtl00/Cube_Stacking_synth`` liefert 14 Handgelenke, deren Zuordnung zu den 14 Achsen
des echten Datensatzes offen ist (siehe docs/training/synth-datensatz.md). Naheliegender
Gedanke: die DEX3-Geometrie kennt man ja — man rechnet für jede Kandidaten-Reihenfolge die
Fingerkuppen aus und behält die, die eine plausible Greifgeometrie ergibt.

Dieses Skript prüft, ob dieser Gedanke trägt — und zwar **am echten Datensatz, wo die
Antwort bekannt ist**. Ein Verfahren, das die bekannte Wahrheit nicht zurückgewinnt, kann
sie beim fremden Datensatz erst recht nicht finden.

Ergebnis (2026-08-28, `Simulation/g1_dex3_sim/replay_episode0.npz`, 1173 Frames):

    Filterstufe                          links          rechts
    URDF-Gelenkgrenzen (Toleranz 0,40)   144 von 5040    72 von 5040
    + keine Fingerdurchdringung          130             70
    + plausible Öffnung und Schließung    49             34
    wahre Reihenfolge dabei              ja              ja
    unterscheidbar                       NEIN (48 Gleichstände)  NEIN (33)

**Das Verfahren trennt nicht.** Der Grund ist kinematisch und nicht behebbar: Zeige- und
Mittelfingerkette der DEX3 sind bis auf ±2,85 cm Palm-Versatz identisch (gleiche Achsen,
gleiche Gliedlängen). Ihre Kuppendistanz ist deshalb für *jede* Gelenkbelegung
√(5,70 cm² + Δ²) ≥ 5,70 cm — die Geometrie trägt die Information schlicht nicht.

Die Toleranz von 0,40 rad ist kein Schlendrian, sondern eine Messung: der echte Datensatz
überschreitet die URDF-Grenzen um bis zu 0,34 rad (Kommandowerte am realen Roboter). Mit
einer engeren Toleranz fällt die wahre Reihenfolge selbst durch das erste Sieb.

Konsequenz: die Reihenfolge muss beim Ersteller des Datensatzes erfragt und dann per
``harmonize_synth_dataset.py inspect --joint-names …`` eingespeist werden. Dort wird sie
gegen die unabhängig gemessene Armzuordnung und gegen die Wertebereiche des echten
Datensatzes gegengeprüft.

Aufruf:
    uv run --no-project --with numpy python Training/scripts/dex3_fingerorder_probe.py
    … --npz <pfad>   anderer Referenzdatensatz (Feld ``state``, (T,28))
"""

from __future__ import annotations

import argparse
import itertools
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF = {
    "left": "data/unitree_ros/robots/dexterous_hand_description/dex3_1/dex3_1_l.urdf",
    "right": "data/unitree_ros/robots/dexterous_hand_description/dex3_1/dex3_1_r.urdf",
}

# Rollen in der Reihenfolge, in der dieses Skript sie durchprobiert.
ROLES = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]

# Achsenreihenfolge des ECHTEN Datensatzes. Links Middle vor Index, rechts Index vor
# Middle — die Asymmetrie steht so in dessen meta/info.json.
REAL_ORDER = {
    "left": (14, ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1",
                  "index_0", "index_1"]),
    "right": (21, ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1",
                   "middle_0", "middle_1"]),
}

CHAINS = {"thumb": ["thumb_0", "thumb_1", "thumb_2"],
          "middle": ["middle_0", "middle_1"],
          "index": ["index_0", "index_1"]}


# ─────────────────────────────────────────────────────────────────────────────
# Vorwärtskinematik, Isaac-frei — damit sie außerhalb des Containers prüfbar ist
# ─────────────────────────────────────────────────────────────────────────────
def parse_hand(side: str) -> tuple[dict, dict]:
    """Gelenke (Ursprung, Achse, Grenzen) und Link-Schwerpunkte aus der offiziellen URDF."""
    root = ET.parse(URDF[side]).getroot()
    joints, coms = {}, {}
    for j in root.findall("joint"):
        if j.get("type") in ("fixed", None):
            continue
        origin, axis, limit = j.find("origin"), j.find("axis"), j.find("limit")
        rpy = [float(v) for v in ((origin.get("rpy") if origin is not None else None)
                                  or "0 0 0").split()]
        if max(abs(v) for v in rpy) > 1e-9:
            raise ValueError(f"{j.get('name')} hat rpy != 0 — die FK hier setzt rpy=0 voraus.")
        joints[j.get("name")] = {
            "xyz": np.array([float(v) for v in
                             ((origin.get("xyz") if origin is not None else None)
                              or "0 0 0").split()]),
            "axis": np.array([float(v) for v in axis.get("xyz").split()]),
            "limit": (float(limit.get("lower")), float(limit.get("upper"))),
        }
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None and inertial.find("origin") is not None:
            coms[link.get("name")] = np.array(
                [float(v) for v in inertial.find("origin").get("xyz").split()]
            )
    return joints, coms


def _rot(axis: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rodrigues um eine feste Achse, vektorisiert über die Frames → (T,3,3)."""
    k = axis / np.linalg.norm(axis)
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    cos, sin = np.cos(q)[:, None, None], np.sin(q)[:, None, None]
    return np.eye(3)[None] + sin * skew[None] + (1 - cos) * (skew @ skew)[None]


def fingertips(side: str, angles: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Kuppenpositionen im Palm-Frame.

    Die Kuppe wird als das Doppelte des Schwerpunkts des distalen Glieds geschätzt: bei
    einem näherungsweise homogenen Stab liegt der Schwerpunkt in dessen Mitte.
    """
    joints, coms = parse_hand(side)
    prefix = f"{side}_hand_"
    tips = {}
    for finger, chain in CHAINS.items():
        frames = len(angles[chain[0]])
        rot = np.repeat(np.eye(3)[None], frames, axis=0)
        pos = np.zeros((frames, 3))
        for name in chain:
            joint = joints[prefix + name + "_joint"]
            pos = pos + np.einsum("tij,j->ti", rot, joint["xyz"])
            rot = rot @ _rot(joint["axis"], np.asarray(angles[name], dtype=float))
        tips[finger] = pos + np.einsum(
            "tij,j->ti", rot, 2.0 * coms[prefix + chain[-1] + "_link"]
        )
    return tips


def apertures(tips: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "thumb-index": np.linalg.norm(tips["thumb"] - tips["index"], axis=1),
        "thumb-middle": np.linalg.norm(tips["thumb"] - tips["middle"], axis=1),
        "index-middle": np.linalg.norm(tips["index"] - tips["middle"], axis=1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rückgewinnungstest
# ─────────────────────────────────────────────────────────────────────────────
def probe(state: np.ndarray, side: str, args: argparse.Namespace) -> None:
    offset, true_order = REAL_ORDER[side]
    joints, _ = parse_hand(side)
    prefix = f"{side}_hand_"
    columns = [offset + k for k in range(7)]
    ranges = {c: (float(state[:, c].min()), float(state[:, c].max())) for c in columns}
    truth = tuple(true_order.index(role) for role in ROLES)

    counts = {"urdf": 0, "no_overlap": 0, "all": 0}
    survivors = []
    for perm in itertools.permutations(range(7)):
        violation = 0.0
        for k, role in enumerate(ROLES):
            low, high = ranges[columns[perm[k]]]
            lo_lim, hi_lim = joints[prefix + role + "_joint"]["limit"]
            violation += max(0.0, lo_lim - low - args.limit_tolerance)
            violation += max(0.0, high - hi_lim - args.limit_tolerance)
        if violation > 1e-9:
            continue
        counts["urdf"] += 1

        ap = apertures(fingertips(side, {role: state[:, columns[perm[k]]]
                                         for k, role in enumerate(ROLES)}))
        ti, tm = ap["thumb-index"], ap["thumb-middle"]
        if not (ti.min() > args.min_gap and tm.min() > args.min_gap):
            continue
        counts["no_overlap"] += 1
        if not (ti.max() < args.max_open and tm.max() < args.max_open):
            continue
        if not (ti.min() < args.max_closed and tm.min() < args.max_closed + 0.02):
            continue
        counts["all"] += 1
        survivors.append(perm)

    print(f"\n=== {side.upper()} — Rückgewinnung der BEKANNTEN Reihenfolge ===")
    print(f"    URDF-Grenzen (Toleranz {args.limit_tolerance} rad) : "
          f"{counts['urdf']:>5} von 5040")
    print(f"    + keine Fingerdurchdringung (> {args.min_gap * 100:.1f} cm) : "
          f"{counts['no_overlap']:>5}")
    print(f"    + plausible Öffnung und Schließung          : {counts['all']:>5}")
    found = truth in survivors
    print(f"    wahre Reihenfolge unter den Überlebenden    : {'ja' if found else 'NEIN'}")
    if found and len(survivors) > 1:
        print(f"    → NICHT unterscheidbar von {len(survivors) - 1} weiteren Reihenfolgen.")
        print("      Das Verfahren trennt nicht — die Fingerreihenfolge muss beim Ersteller")
        print("      des Datensatzes erfragt werden.")
    elif found:
        print("    → eindeutig. Das Verfahren trägt (unerwartet — bitte nachrechnen).")
    else:
        print("    → die Filter sind zu streng, sie schließen die Wahrheit aus.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", default="Simulation/g1_dex3_sim/replay_episode0.npz",
                   help="Referenz mit bekannter Reihenfolge; Feld 'state' der Form (T,28)")
    p.add_argument("--limit-tolerance", type=float, default=0.40,
                   help="Zulässige Überschreitung der URDF-Grenzen (rad). Der echte "
                        "Datensatz überschreitet sie um bis zu 0,34 rad")
    p.add_argument("--min-gap", type=float, default=0.015, help="Kuppenabstand (m), unter "
                   "dem sich Finger durchdringen würden")
    p.add_argument("--max-open", type=float, default=0.17, help="Größte plausible Öffnung (m)")
    p.add_argument("--max-closed", type=float, default=0.07,
                   help="Der Griff muss mindestens so weit schließen (m)")
    args = p.parse_args()

    path = Path(args.npz)
    if not path.exists():
        print(f"Referenzdatei fehlt: {path}")
        return 1
    state = np.load(path)["state"].astype(float)
    if state.ndim != 2 or state.shape[1] != 28:
        print(f"'state' muss (T,28) sein, ist {state.shape}")
        return 1
    print(f"Referenz: {path}  ({len(state)} Frames, Reihenfolge bekannt)")
    for side in ("left", "right"):
        probe(state, side, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
