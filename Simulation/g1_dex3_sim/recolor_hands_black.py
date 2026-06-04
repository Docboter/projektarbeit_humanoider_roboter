"""
Offline-USD-Recolor: färbt die DEX3-Hände im g1_dex3.usd schwarz.

Domain-Gap-Fix: Im Dataset sind die Hände schwarz, im Sim-Asset weiß. Da der
Vision-Encoder eingefroren ist, sieht er in den (von den Händen dominierten) Wrist-
Kameras einen großen falschfarbenen Bereich.

WICHTIG — zwei Eigenheiten dieses Assets (verifiziert mit usd-core auf g1_dex3.usd):
  1. Der Roboter ist INSTANZIERT (instanceable=True auf den .../<link>/visuals-Prims).
     Eine Material-Bindung am Instance-ROOT wird NICHT ins Prototype (die eigentlichen
     Meshes) komponiert und rendert daher nicht — die Hände blieben weiß. Lösung: die
     Hand-/visuals-Prims werden DE-INSTANZIERT (instanceable=False), und das Material wird
     DIREKT auf jedes darunterliegende Mesh-Gprim gebunden (überschreibt sicher auch eine
     bereits am Mesh vorhandene weiße Direktbindung, die eine reine Eltern-Vererbung gewinnt).
  2. Der Roboter-Root heißt "g1_29dof_with_hand_rev_1_0" — enthält selbst "hand". Ein
     naiver "hand"-Substring-Match würde den GANZEN Roboter treffen. Deshalb wird gezielt
     auf die Link-Präfixe ``left_hand_`` / ``right_hand_`` gematcht (16 /visuals-Prims:
     je 8 pro Hand — palm, thumb_0..2, index_0/1, middle_0/1).

Reines USD-Authoring (pxr) — KEIN Isaac-Sim-Start, KEINE GPU nötig. Läuft lokal
(``pip install usd-core``) oder im Sim-Container:

    python recolor_hands_black.py --in data/g1_dex3.usd --out data/g1_dex3_blackhands.usd

Hinweis: g1_dex3.usd ist ein dünner Wrapper mit RELATIVEN Referenzen auf configuration/.
Die Ausgabe ins selbe Verzeichnis schreiben (Default), damit die Referenzen gültig bleiben.
Die Warnungen über unaufgelöste imu_/d435_/mid360_-Referenzen sind vorbestehend und
betreffen die Hände nicht.
"""

from __future__ import annotations

import argparse

from pxr import Gf, Sdf, Usd, UsdShade


def main() -> None:
    ap = argparse.ArgumentParser(description="DEX3-Hände im USD schwarz einfärben")
    ap.add_argument("--in", dest="in_path", default="/data/assets/g1_dex3.usd")
    ap.add_argument("--out", dest="out_path", default=None,
                    help="Ziel-USD (Default: <in>_blackhands.usd, gleiches Verzeichnis)")
    ap.add_argument("--in-place", action="store_true", help="Eingabe-USD direkt überschreiben")
    ap.add_argument("--markers", default="left_hand_,right_hand_",
                    help="Komma-Liste: ein Prim-Pfad muss einen dieser Marker enthalten")
    ap.add_argument("--include-collisions", action="store_true",
                    help="Auch /collisions-Prims binden (Default: nur /visuals)")
    ap.add_argument("--color", nargs=3, type=float, default=[0.02, 0.02, 0.02],
                    help="RGB der Hand-Farbe (Default: nahezu schwarz)")
    args = ap.parse_args()

    out_path = args.in_path if args.in_place else (
        args.out_path or args.in_path.rsplit(".", 1)[0] + "_blackhands.usd"
    )
    markers = [m for m in args.markers.split(",") if m]

    stage = Usd.Stage.Open(args.in_path)
    if stage is None:
        raise SystemExit(f"Konnte USD nicht öffnen: {args.in_path}")

    # Schwarzes UsdPreviewSurface-Material anlegen (unter dem DefaultPrim, sonst top-level).
    default_prim = stage.GetDefaultPrim()
    looks_root = (default_prim.GetPath().pathString if default_prim and default_prim.IsValid()
                  else "")
    mat_path = f"{looks_root}/Looks/HandBlack"
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*[float(c) for c in args.color])
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Hand-/visuals-Roots finden, de-instanzieren und das Material direkt auf jedes
    # darunterliegende Mesh-Gprim binden. (Bindung am Instance-Root allein rendert nicht —
    # s. Modul-Docstring Punkt 1.) Erst alle passenden Roots sammeln, dann verarbeiten:
    # SetInstanceable(False) verändert die Komposition, daher die Traverse nicht gleichzeitig
    # mutieren.
    roots = []
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not any(m in path for m in markers):
            continue
        if not prim.IsInstance():
            continue
        if not args.include_collisions and not path.endswith("/visuals"):
            continue
        roots.append(path)

    bound = []
    for root_path in roots:
        root = stage.GetPrimAtPath(root_path)
        # De-instanzieren, damit der /visuals-Root im Stage editierbar wird und die
        # Bindung in den (sonst geteilten) Mesh-Subbaum komponiert.
        root.SetInstanceable(False)
        # Auf den /visuals-Root mit bindingStrength=strongerThanDescendants binden:
        # die referenzierten Meshes tragen eine eigene weiße Direktbindung
        # (material_white), die eine schwächere Vorfahr-Bindung sonst gewinnt. Eine
        # stärkere Vorfahr-Bindung überschreibt jede Bindung darunter zuverlässig.
        UsdShade.MaterialBindingAPI.Apply(root).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )
        bound.append(root_path)

    if not bound:
        print(f"WARN: keine Hand-/visuals-Roots für Marker {markers} gefunden — nichts geändert.")
        print("      Prim-Struktur prüfen (Instancing!) und --markers anpassen.")
    else:
        print(f"OK: schwarzes Material an {len(bound)} Hand-/visuals-Roots gebunden "
              f"(de-instanziert, strongerThanDescendants):")
        for p in bound:
            print(f"    {p}")

    stage.GetRootLayer().Export(out_path)
    print(f"Geschrieben: {out_path}")


if __name__ == "__main__":
    main()
