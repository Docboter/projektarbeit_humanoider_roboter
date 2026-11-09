#!/usr/bin/env python3
# TL;DR: Wandelt das G1-URDF (unitree_ros, BSD-3) einmalig in presentation/assets/robot.js um:
# vereinfachte Meshes (quantisiert, base64) plus Gelenkbaum, ladbar per <script> auch über file://.
"""Aufruf (aus dem Repo-Wurzelverzeichnis):

    uv run --no-project --with trimesh --with fast-simplification --with numpy \
        python presentation/tools/build_robot.py

Quelle: data/unitree_ros/robots/g1_description/g1_29dof_with_hand.urdf (Submodul).
Das Ergebnis wird eingecheckt; das Skript muss nur bei Änderungen erneut laufen.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "data/unitree_ros/robots/g1_description/g1_29dof_with_hand.urdf"
OUT = ROOT / "presentation/assets/robot.js"

# Zielanzahl Dreiecke je Mesh: Anteil am Original, mit Unter- und Obergrenze.
RATIO = 0.06
MIN_FACES = 120
MAX_FACES = 2600


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


def floats(s: str | None, n: int = 3) -> list[float]:
    if not s:
        return [0.0] * n
    return [float(v) for v in s.split()]


def simplify(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.merge_vertices()
    n = len(mesh.faces)
    target = int(min(MAX_FACES, max(MIN_FACES, n * RATIO)))
    if n > target:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=target)
        except TypeError:  # ältere trimesh-Signatur
            mesh = mesh.simplify_quadric_decimation(target)
    mesh.merge_vertices()
    return mesh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=Path, default=URDF)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    root = ET.parse(args.urdf).getroot()
    base = args.urdf.parent
    meshes, links, joints = [], {}, []
    total = 0
    for link in root.findall("link"):
        name = link.get("name")
        vis = link.find("visual")
        entry: dict = {}
        if vis is not None and vis.find("geometry/mesh") is not None:
            fn = vis.find("geometry/mesh").get("filename")
            m = simplify(trimesh.load(base / fn, force="mesh"))
            v = np.asarray(m.vertices, dtype=np.float64)
            lo, hi = v.min(0), v.max(0)
            scale = np.where(hi - lo > 0, (hi - lo) / 65535.0, 1.0)
            q = np.round((v - lo) / scale).astype(np.uint16)
            idx = np.asarray(m.faces, dtype=np.uint16 if len(v) < 65535 else np.uint32)
            meshes.append(
                {
                    "min": lo.round(6).tolist(),
                    "scale": scale.tolist(),
                    "pos": b64(q),
                    "idx": b64(idx),
                    "i32": bool(idx.dtype == np.uint32),
                }
            )
            o = vis.find("origin")
            entry = {
                "mesh": len(meshes) - 1,
                "xyz": floats(o.get("xyz") if o is not None else None),
                "rpy": floats(o.get("rpy") if o is not None else None),
            }
            total += len(m.faces)
        links[name] = entry
    for j in root.findall("joint"):
        o, a, lim = j.find("origin"), j.find("axis"), j.find("limit")
        joints.append(
            {
                "name": j.get("name"),
                "type": j.get("type"),
                "parent": j.find("parent").get("link"),
                "child": j.find("child").get("link"),
                "xyz": floats(o.get("xyz") if o is not None else None),
                "rpy": floats(o.get("rpy") if o is not None else None),
                "axis": floats(a.get("xyz") if a is not None else "1 0 0"),
                "lower": float(lim.get("lower")) if lim is not None else 0.0,
                "upper": float(lim.get("upper")) if lim is not None else 0.0,
            }
        )
    data = {"root": "pelvis", "links": links, "joints": joints, "meshes": meshes}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "// Generiert von presentation/tools/build_robot.py aus unitree_ros (BSD-3-Clause).\n"
        "window.G1_ROBOT = " + json.dumps(data, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"{len(meshes)} Meshes, {total} Dreiecke, {args.out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
