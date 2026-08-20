#!/usr/bin/env python3
# TL;DR: Liest Würfellagen aus Realbildern (Blob-Detektion + Rückprojektion) für den Renderer.
"""
extract_block_layout.py — Würfellage aus dem REALBILD lesen, statt sie aus der Fingerbewegung zu raten.

WARUM
─────
Der Renderer platzierte die Würfel bisher am Greifpunkt aus ``scan.json``, und der ist das
Minimum der Fingeröffnung über die ganze Episode. Bei einem Pick-and-Place bleibt die Hand
aber von der Aufnahme bis zum Ablegen geschlossen — ein breites Tal, kein Ausschlag. Wo
innerhalb dieses Tals das Minimum liegt, entscheidet minimales Nachdrücken: 48 von 116
Griffen des Laufs vom 2026-08-17 lagen jenseits von 60 % der Episode, 21 jenseits von 80 %.
Der Würfel landete damit irgendwo auf dem Transportweg, und der Arm griff beim echten Pick
ins Leere — in x, y UND z.

Die Information, wo die Würfel lagen, steckt aber im Datensatz: **im Bild**. Drei gesättigte
Würfel (rot = block_0, grün = block_1, gelb = block_2) auf weißem Tisch, und die Sim-Kameras
sind gegen genau diese Frames eingerichtet. Also: Blob segmentieren, Strahl durch den
Schwerpunkt auf die Würfelebene schneiden, fertig — Grundwahrheit statt Rekonstruktion, und
für alle drei Würfel statt für einen pro Hand.

WAS DIESES WERKZEUG BEWEIST — UND WAS NICHT
───────────────────────────────────────────
Die Rückprojektion ist nur so gut wie das Kameramodell, und das ist aus den Dataset-Frames
**rekonstruiert**, nicht kalibriert. Schlimmer: ob Isaac mit der konfigurierten Pose auch
rendert, ist eine eigene Frage — in Lauf 13 (2026-08-08) lagen USD-Stage und ``cam.data``
95,6° auseinander und drei Läufe waren umsonst. Deshalb hat der Modus ``detect`` ein
``--expect``:

    1. ``detect`` auf einem GERENDERTEN Sim-Bild mit bekannten Würfelpositionen
       (``render_manifest.json`` → ``cubes_xyz``). Das Residuum ist der Fehler des
       Schätzers gegen Grundwahrheit — Kameramodell und Blob-Schwerpunkt zusammen.
    2. Ist es klein und systematisch, ist es der Bias des Schätzers (der Blob-Schwerpunkt
       ist der Schwerpunkt der SICHTBAREN FLÄCHEN, nicht die Projektion des Würfel-
       mittelpunkts) und wird per ``--bias`` herausgerechnet.
    3. Erst dann ``extract`` auf den Realbildern.

Schritt 1 ist nicht optional. Ohne ihn ist jede Zahl hier plausibel und unbelegt.

VERWENDUNG
──────────
    # 1. Modell gegen den Renderer prüfen (Sim-Bild + bekannte Würfelpositionen)
    python3 extract_block_layout.py detect sim_frame0.png \\
        --expect '[[0.34,-0.15,0.915],[0.36,0.0,0.915],[0.34,0.15,0.915]]' --debug-dir /tmp/dbg

    # 2. Realbilder → layout.json
    python3 extract_block_layout.py extract \\
        --dataset-path /data/unitreerobotics/G1_Dex3_BlockStacking_Dataset \\
        --out /data/cotrain/layout.json --num-episodes 60 --debug-dir /data/cotrain/layout_dbg

Braucht kein Isaac Lab und keine GPU: numpy + Pillow (PNG) bzw. imageio (MP4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_geometry import CAMERA_CFG, PinholeCamera  # noqa: E402

# Würfelreihenfolge = block_0/1/2 aus g1_dex3_blockstack_env.py. Die Farben dort sind an die
# Realdaten angeglichen (rot/grün/gelb), deshalb ist die Zuordnung eindeutig.
CUBE_COLORS = ("rot", "gruen", "gelb")

# Farbfenster in HSV (H in Grad, S/V in 0…1). Weit genug für Beleuchtungsunterschiede
# zwischen Real und Sim, eng genug gegen weißen Tisch (S niedrig), schwarze Hände (V
# niedrig) und graue Arme (S niedrig).
HSV_WINDOWS = {
    "rot":   {"h": ((345.0, 360.0), (0.0, 15.0)), "s": 0.40, "v": 0.18},
    "gruen": {"h": ((75.0, 175.0),),              "s": 0.28, "v": 0.15},
    "gelb":  {"h": ((35.0, 70.0),),               "s": 0.35, "v": 0.28},
}

# Würfelmittelpunkt-Ebene. Muss zu block_z_surface in g1_dex3_blockstack_env.py passen.
Z_CUBE_CENTER = 0.915
CUBE_EDGE_M = 0.05

VIDEO_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CHUNK_SIZE = 1000


# ---------------------------------------------------------------------------
# Bild laden
# ---------------------------------------------------------------------------

def load_png(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_video_frame(path: Path, index: int = 0) -> np.ndarray:
    """Einen Frame aus einem MP4 holen. Sequenziell, weil Seeking bei h264 unzuverlässig ist."""
    import imageio.v2 as imageio
    with imageio.get_reader(str(path), format="FFMPEG") as rd:
        for i, frame in enumerate(rd):
            if i == index:
                return np.asarray(frame, dtype=np.uint8)[..., :3]
    raise IndexError(f"{path} hat keinen Frame {index}.")


# ---------------------------------------------------------------------------
# Segmentierung
# ---------------------------------------------------------------------------

def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vektorisiert, ohne cv2/matplotlib — H in Grad, S und V in 0…1."""
    r, g, b = (rgb[..., i].astype(np.float32) / 255.0 for i in range(3))
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    d = mx - mn
    h = np.zeros_like(mx)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(mx == r, ((g - b) / d) % 6.0, h)
        h = np.where(mx == g, (b - r) / d + 2.0, h)
        h = np.where(mx == b, (r - g) / d + 4.0, h)
    h = np.where(d < 1e-6, 0.0, h) * 60.0
    s = np.where(mx < 1e-6, 0.0, d / np.maximum(mx, 1e-6))
    return h, s, mx


def color_mask(rgb: np.ndarray, color: str) -> np.ndarray:
    w = HSV_WINDOWS[color]
    h, s, v = rgb_to_hsv(rgb)
    hue = np.zeros(h.shape, dtype=bool)
    for lo, hi in w["h"]:
        hue |= (h >= lo) & (h <= hi)
    return hue & (s >= w["s"]) & (v >= w["v"])


def largest_blob(mask: np.ndarray, min_area: int = 120) -> dict | None:
    """Größte zusammenhängende Fläche der Maske (4-Nachbarschaft).

    scipy, wenn vorhanden — sonst ein iterativer Flood-Fill. Der Fallback existiert, weil
    dieses Werkzeug auch in einem Container ohne scipy laufen soll und ein fehlendes Paket
    kein Grund ist, die Messung nicht zu machen.
    """
    if not mask.any():
        return None
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        if n == 0:
            return None
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        sel = lab == k
    except ImportError:
        H, W = mask.shape
        seen = np.zeros_like(mask)
        best: list[tuple[int, int]] = []
        for y0, x0 in zip(*np.nonzero(mask)):
            if seen[y0, x0]:
                continue
            stack, pix = [(int(y0), int(x0))], []
            seen[y0, x0] = True
            while stack:
                y, x = stack.pop()
                pix.append((y, x))
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(pix) > len(best):
                best = pix
        if not best:
            return None
        sel = np.zeros_like(mask)
        idx = np.array(best)
        sel[idx[:, 0], idx[:, 1]] = True

    ys, xs = np.nonzero(sel)
    area = int(ys.size)
    if area < min_area:
        return None
    return {
        "u": float(xs.mean()), "v": float(ys.mean()), "area": area,
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "extent": float(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1),
    }


# ---------------------------------------------------------------------------
# Rückprojektion
# ---------------------------------------------------------------------------

def locate_cubes(rgb: np.ndarray, cam: PinholeCamera, bias_xy=(0.0, 0.0),
                 min_area: int = 120) -> list[dict | None]:
    """Je Farbe: Blob suchen, Schwerpunkt auf die Würfelebene schneiden.

    ``dist_plane_m`` kommt aus der Geometrie, ``dist_size_m`` aus der scheinbaren Größe des
    Blobs. Beide sollten grob übereinstimmen — die zweite Zahl ist grob (die Silhouette
    eines schräg gesehenen Würfels ist breiter als seine Kante, gut 0–30 %), fängt aber
    einen Strahl, der auf der falschen Ebene landet.
    """
    out: list[dict | None] = []
    for color in CUBE_COLORS:
        blob = largest_blob(color_mask(rgb, color), min_area=min_area)
        if blob is None:
            out.append(None)
            continue
        try:
            p = cam.backproject_to_plane(blob["u"], blob["v"], Z_CUBE_CENTER)
        except ValueError as exc:
            out.append({"color": color, "error": str(exc), **blob})
            continue
        p = p + np.array([bias_xy[0], bias_xy[1], 0.0])
        out.append({
            "color": color,
            "xy": [round(float(p[0]), 4), round(float(p[1]), 4)],
            "u": round(blob["u"], 2), "v": round(blob["v"], 2),
            "area": blob["area"], "extent_px": round(blob["extent"], 1),
            "dist_plane_m": round(float(np.linalg.norm(p - cam.eye)), 4),
            "dist_size_m": round(float(cam.f_px * CUBE_EDGE_M / blob["extent"]), 4),
            "cm_per_px": round(cam.plane_scale_cm_per_px(blob["u"], blob["v"],
                                                         Z_CUBE_CENTER), 4),
        })
    return out


def draw_markers(rgb: np.ndarray, found: list[dict | None],
                 expect_uv: np.ndarray | None = None) -> np.ndarray:
    """Kreuz auf jeden gefundenen Schwerpunkt, Kästchen auf die erwartete Projektion."""
    img = rgb.copy()
    H, W = img.shape[:2]
    for f in found:
        if not f or "u" not in f:
            continue
        u, v = int(round(f["u"])), int(round(f["v"]))
        img[max(0, v - 9):v + 10, max(0, u - 1):u + 2] = (255, 255, 255)
        img[max(0, v - 1):v + 2, max(0, u - 9):u + 10] = (255, 255, 255)
    if expect_uv is not None:
        for uv in np.atleast_2d(expect_uv):
            if not np.all(np.isfinite(uv)):
                continue
            u, v = int(round(uv[0])), int(round(uv[1]))
            for du in range(-11, 12):
                for dv in (-11, 11):
                    if 0 <= v + dv < H and 0 <= u + du < W:
                        img[v + dv, u + du] = (0, 0, 0)
                    if 0 <= v + du < H and 0 <= u + dv < W:
                        img[v + du, u + dv] = (0, 0, 0)
    return img


def save_png(img: np.ndarray, path: Path) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


# ---------------------------------------------------------------------------
# Modus: detect
# ---------------------------------------------------------------------------

def run_detect(args) -> int:
    cam = PinholeCamera.from_cfg(args.camera)
    expect = np.asarray(json.loads(args.expect), dtype=float) if args.expect else None
    expect_uv = cam.project(expect) if expect is not None else None
    residuals: list[np.ndarray] = []

    for path in args.images:
        path = Path(path)
        rgb = load_png(path) if path.suffix.lower() == ".png" else load_video_frame(
            path, args.frame)
        if rgb.shape[:2] != (cam.height, cam.width):
            print(f"[warn] {path.name}: {rgb.shape[1]}×{rgb.shape[0]} statt "
                  f"{cam.width}×{cam.height} — Modell passt nicht zum Bild.")
        found = locate_cubes(rgb, cam, bias_xy=args.bias, min_area=args.min_area)
        print(f"\n{path.name}  ({args.camera})")
        for i, f in enumerate(found):
            if f is None:
                print(f"  {i} {CUBE_COLORS[i]:6}  NICHT GEFUNDEN")
                continue
            if "error" in f:
                print(f"  {i} {CUBE_COLORS[i]:6}  {f['error']}")
                continue
            line = (f"  {i} {CUBE_COLORS[i]:6}  Pixel ({f['u']:6.1f},{f['v']:6.1f})  "
                    f"xy ({f['xy'][0]:+.3f},{f['xy'][1]:+.3f})  Kante {f['extent_px']:5.1f}px  "
                    f"Abstand Ebene {f['dist_plane_m']:.3f} / Größe {f['dist_size_m']:.3f} m")
            if expect is not None and i < len(expect):
                d = np.array(f["xy"]) - expect[i][:2]
                residuals.append(d)
                line += f"  Δ ({d[0] * 100:+5.1f},{d[1] * 100:+5.1f}) cm"
            print(line)
        if args.debug_dir:
            out = Path(args.debug_dir) / f"{path.stem}_{args.camera}.png"
            save_png(draw_markers(rgb, found, expect_uv), out)
            print(f"  → {out}")

    if residuals:
        r = np.array(residuals)
        print(f"\nResiduum gegen Grundwahrheit über {len(r)} Würfel:")
        print(f"  Mittel  ({r[:, 0].mean() * 100:+.2f}, {r[:, 1].mean() * 100:+.2f}) cm"
              f"   ← das ist der BIAS des Schätzers, per --bias herausrechnen")
        print(f"  Streuung ({r[:, 0].std() * 100:.2f}, {r[:, 1].std() * 100:.2f}) cm"
              f"   ← das ist der Rest-Fehler, der bleibt")
        print(f"  Betrag  Median {np.median(np.linalg.norm(r, axis=1)) * 100:.2f} cm, "
              f"max {np.linalg.norm(r, axis=1).max() * 100:.2f} cm")
    return 0


# ---------------------------------------------------------------------------
# Modus: extract
# ---------------------------------------------------------------------------

def select_episodes(root: Path, args) -> list[int]:
    """Dieselbe Auswahl wie render_cotrain_dataset.py, inklusive Test-Episoden-Sperre."""
    with open(root / "meta" / "info.json") as fh:
        info = json.load(fh)
    total = int(info["total_episodes"])
    n_train = int(total * args.train_ratio)
    if args.episode_ids:
        bad = [e for e in args.episode_ids if e >= n_train]
        if bad:
            raise SystemExit(
                f"Zurückgehaltene Test-Episoden angefordert: {bad} (Grenze {n_train}). "
                "Gerendert werden dürfen sie nicht — die Validierungs-MSE wäre kontaminiert."
            )
        return sorted(args.episode_ids)
    n = min(args.num_episodes, n_train)
    return sorted({int(round(i)) for i in np.linspace(0, n_train - 1, n)})


def run_extract(args) -> int:
    root = Path(args.dataset_path)
    episodes = select_episodes(root, args)
    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    models = {c: PinholeCamera.from_cfg(c) for c in cams}

    out_path = Path(args.out)
    layout = {"episodes": {}, "cameras": cams, "bias_cm": [args.bias[0] * 100,
                                                           args.bias[1] * 100],
              "z_plane": Z_CUBE_CENTER, "source_dataset": str(root)}
    if out_path.exists() and not args.overwrite:
        layout = json.loads(out_path.read_text())
        layout.setdefault("episodes", {})

    for k, ep in enumerate(episodes, 1):
        head = f"[layout] ({k}/{len(episodes)}) Episode {ep}"
        if str(ep) in layout["episodes"] and not args.overwrite:
            print(f"{head}: schon vorhanden.", flush=True)
            continue
        per_cam: dict[str, list] = {}
        for cam_name in cams:
            video = root / VIDEO_TEMPLATE.format(
                episode_chunk=ep // CHUNK_SIZE,
                video_key=f"observation.images.{cam_name}", episode_index=ep)
            if not video.exists():
                print(f"{head}: {video} fehlt.", flush=True)
                continue
            rgb = load_video_frame(video, args.frame)
            per_cam[cam_name] = locate_cubes(rgb, models[cam_name], bias_xy=args.bias,
                                             min_area=args.min_area)
            if args.debug_dir:
                save_png(draw_markers(rgb, per_cam[cam_name]),
                         Path(args.debug_dir) / f"ep{ep:06d}_{cam_name}.png")

        cubes, spread = [], []
        for i in range(len(CUBE_COLORS)):
            picks = [per_cam[c][i]["xy"] for c in per_cam
                     if per_cam[c][i] and "xy" in per_cam[c][i]]
            if not picks:
                cubes.append(None)
                continue
            arr = np.array(picks, dtype=float)
            cubes.append([round(float(arr[:, 0].mean()), 4), round(float(arr[:, 1].mean()), 4)])
            if len(arr) > 1:
                spread.append(float(np.linalg.norm(arr[0] - arr[1])))

        layout["episodes"][str(ep)] = {
            "frame": args.frame,
            "cubes": cubes,
            "per_camera": per_cam,
            # Der Abgleich beider Kameras prüft die Detektion, NICHT das Kameramodell:
            # bei 4,7 cm Stereobasis auf 0,5 m verschiebt ein Modellfehler beide Strahlen
            # fast gleich. Eine kleine Zahl hier heißt also „beide Kameras sehen dasselbe",
            # nicht „die Pose stimmt".
            "camera_spread_cm": [round(s * 100, 2) for s in spread],
        }
        n_ok = sum(1 for c in cubes if c)
        print(f"{head}: {n_ok}/3 Würfel"
              + (f", Kameras uneins um {max(spread) * 100:.1f} cm" if spread else ""),
              flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(layout, indent=2))

    recs = layout["episodes"]
    full = sum(1 for r in recs.values() if all(r["cubes"]))
    spreads = [s for r in recs.values() for s in r.get("camera_spread_cm", [])]
    print(f"\n[layout] {len(recs)} Episoden, {full} mit allen drei Würfeln.")
    if spreads:
        print(f"[layout] Kamera-Uneinigkeit: Median {np.median(spreads):.2f} cm, "
              f"max {max(spreads):.2f} cm")
    print(f"[layout] {out_path}")
    print("[layout] fertig.", flush=True)
    return 0


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--frame", type=int, default=0, help="Welcher Frame (Default 0)")
    common.add_argument("--min-area", type=int, default=120,
                        help="Kleinere Blobs gelten als Rauschen (Pixel)")
    common.add_argument("--bias", type=float, nargs=2, default=(0.0, 0.0),
                        metavar=("DX", "DY"),
                        help="Korrektur in METERN, aus `detect --expect` gewonnen. Der "
                             "Blob-Schwerpunkt ist der Schwerpunkt der sichtbaren Flächen, "
                             "nicht die Projektion des Würfelmittelpunkts.")
    common.add_argument("--debug-dir", type=str, default="",
                        help="Markierte PNGs hierhin schreiben (Kreuz = gefunden, "
                             "Kästchen = erwartet)")

    d = sub.add_parser("detect", parents=[common],
                       help="Blobs in gegebenen Bildern; mit --expect gegen Grundwahrheit")
    d.add_argument("images", nargs="+")
    d.add_argument("--camera", default="cam_left_high",
                   choices=("cam_left_high", "cam_right_high", "cam_scene"))
    d.add_argument("--expect", type=str, default="",
                   help='Bekannte Würfelpositionen als JSON, z. B. '
                        '"[[0.34,-0.15,0.915],[0.36,0,0.915],[0.34,0.15,0.915]]" '
                        '(render_manifest.json → cubes_xyz)')

    e = sub.add_parser("extract", parents=[common],
                       help="Realbilder eines Datensatzes → layout.json")
    e.add_argument("--dataset-path", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--num-episodes", type=int, default=60)
    e.add_argument("--episode-ids", type=int, nargs="*", default=None)
    e.add_argument("--train-ratio", type=float, default=0.8)
    e.add_argument("--cameras", default="cam_left_high,cam_right_high")
    e.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()
    if CAMERA_CFG.width != 640 or CAMERA_CFG.height != 480:
        print(f"[warn] Kamerakonfiguration steht auf {CAMERA_CFG.width}×{CAMERA_CFG.height}.")
    return run_detect(args) if args.mode == "detect" else run_extract(args)


if __name__ == "__main__":
    raise SystemExit(main())
