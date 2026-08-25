#!/usr/bin/env python3
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
        --expect '[[0.34,-0.15,0.895],[0.36,0.0,0.895],[0.34,0.15,0.895]]' --debug-dir /tmp/dbg

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

# Würfelmittelpunkt-Ebene — die RUHELAGE, nicht die Spawnhöhe. Der Tisch ist 0,87 m hoch
# (Box 0.87, Mittelpunkt z=0.435), ein 5-cm-Würfel ruht also mit dem Mittelpunkt auf 0,895.
# Bis 2026-08-22 standen hier 0,915, der Spawnwert der Env, aus dem die Würfel zwei
# Zentimeter herunterfallen. Gegen die `cams`-Grundwahrheit gemessen kostete das rund 1 cm:
# Restfehler 1,11 cm (links) / 0,89 cm (rechts) mit 0,915 gegen 0,30 / 0,22 cm mit 0,895.
Z_CUBE_CENTER = 0.895
CUBE_EDGE_M = 0.05
# Deckflächen-Ebene. Der Gierwinkel wird auf IHR gemessen, nicht auf der Mittelpunktsebene:
# die sichtbare Deckfläche liegt physisch hier, und nur auf ihrer eigenen Ebene ist ihre
# Rückprojektion wieder ein echtes Quadrat.
Z_CUBE_TOP = Z_CUBE_CENTER + CUBE_EDGE_M / 2.0     # 0.920

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


def largest_component(mask: np.ndarray) -> np.ndarray | None:
    """Größte zusammenhängende Fläche der Maske (4-Nachbarschaft) als Auswahl-Array.

    scipy, wenn vorhanden — sonst ein iterativer Flood-Fill. Der Fallback existiert, weil
    dieses Werkzeug auch in einem Container ohne scipy laufen soll und ein fehlendes Paket
    kein Grund ist, die Messung nicht zu machen.

    Getrennt von ``largest_blob``, weil der Gierwinkel die PIXEL der Komponente braucht und
    nicht ihre Kennzahlen. In das Blob-Dict dürfen sie nicht: das landet in ``layout.json``.
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
    return sel


def largest_blob(mask: np.ndarray, min_area: int = 120) -> dict | None:
    """Kennzahlen der größten zusammenhängenden Fläche — Schwerpunkt, Fläche, Bounding-Box."""
    sel = largest_component(mask)
    if sel is None:
        return None
    ys, xs = np.nonzero(sel)
    area = int(ys.size)
    if area < min_area:
        return None
    return {
        "u": float(xs.mean()), "v": float(ys.mean()), "area": area,
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "extent": float(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1),
    }


def track_colors(path, min_area: int = 80) -> dict[str, list[list[float] | None]]:
    """Track the largest color blob per frame; this is used only for motion timing.

    Der Schwerpunkt der GANZEN Farbmaske wäre billiger, ist aber unbrauchbar: die Maske
    enthält neben dem Würfel regelmäßig Streupixel (Abnahmelauf 2026-08-22, Episode 0,
    ``cam_left_high``: rot 4 Komponenten / 47 % Würfel, gruen 27 / 64 %, gelb 16 / 64 %).
    Bei Gelb lag der Gesamtschwerpunkt 14 px neben dem Blobschwerpunkt — mehr als die
    8-px-Schwelle von ``find_motion_onset``. Folge: 33 von 40 Gelb-Onsets feuerten bis
    Frame 30, also bevor der Roboter den Würfel überhaupt berührt, und Gelb lieferte
    keinen einzigen Anker. Der Messpfad (``top_face_blob``) benutzt ohnehin
    ``largest_blob``; hier dieselbe Quelle zu nehmen ist die eigentliche Korrektur.
    """
    import imageio.v2 as imageio

    scale = 2
    reduced_min_area = max(20, min_area // (scale * scale))
    tracks = {color: [] for color in CUBE_COLORS}
    with imageio.get_reader(str(path), format="FFMPEG") as reader:
        for frame in reader:
            rgb = np.asarray(frame, dtype=np.uint8)[::scale, ::scale, :3]
            hue, saturation, value = rgb_to_hsv(rgb)
            for color in CUBE_COLORS:
                window = HSV_WINDOWS[color]
                hue_mask = np.zeros(hue.shape, dtype=bool)
                for low, high in window["h"]:
                    hue_mask |= (hue >= low) & (hue <= high)
                mask = hue_mask & (saturation >= window["s"]) & (value >= window["v"])
                blob = largest_blob(mask, min_area=reduced_min_area)
                tracks[color].append(
                    None if blob is None
                    else [float(blob["u"] * scale), float(blob["v"] * scale)]
                )
    return tracks


def find_motion_onset(
    track: list[list[float] | None], threshold_px: float = 8.0, stable_frames: int = 5
) -> int | None:
    """Return the first run that stays displaced from the first ten valid samples."""
    valid = [(index, point) for index, point in enumerate(track[:30]) if point is not None]
    if len(valid) < 10:
        return None
    baseline = np.median(np.asarray([point for _, point in valid[:10]], dtype=float), axis=0)
    search_start = valid[9][0] + 1
    run = 0
    for index, point in enumerate(track[search_start:], search_start):
        moved = point is not None and np.linalg.norm(np.asarray(point) - baseline) >= threshold_px
        run = run + 1 if moved else 0
        if run >= stable_frames:
            return index - stable_frames + 1
    return None

MOTION_ONSET_TOLERANCE_FRAMES = 12


def episode_motion_onsets(root: Path, ep: int, cams: list[str], min_area: int = 80,
                          tolerance: int = MOTION_ONSET_TOLERANCE_FRAMES) -> dict:
    """Ab welchem Frame sich jeder Würfel im Realvideo bewegt.

    Das ist die Grenze, bis zu der ein gerendertes Bild zur echten Aktion passt: davor liegt
    der Würfel dort, wo ihn das Layout hinsetzt, danach hat ihn die reale Hand bewegt,
    während der simulierte liegen bleibt.

    Bis 2026-08-22 kam diese Grenze aus ``scan.json`` (``close_step``, das Minimum der
    Fingeröffnung). Dieser Detektor funktioniert für die DEX3 nicht — bei 101 von 116
    Griffen bleibt die engste Kuppenöffnung über 6 cm bei 5 cm Würfelkante —, und er lag in
    Episode 0 achtundzwanzig Frames zu spät, also 21 % der Episode falsch beschriftet.

    Eine Farbe zählt nur, wenn **beide** Kopfkameras einen Bewegungsbeginn finden und
    höchstens ``tolerance`` Frames auseinanderliegen. Ohne diese Prüfung schlagen einzelne
    Fehlauslösungen durch: in Episode 0 meldet Grün 10 gegen 194 und Gelb 165 gegen 244,
    beide unbrauchbar, während Rot mit 108/107 trägt.
    """
    tracks = {}
    for cam in cams:
        video = root / VIDEO_TEMPLATE.format(
            episode_chunk=ep // CHUNK_SIZE,
            video_key=f"observation.images.{cam}", episode_index=ep)
        if video.exists():
            tracks[cam] = track_colors(video, min_area=min_area)

    per_color: dict[str, int] = {}
    rejected: dict[str, str] = {}
    for color in CUBE_COLORS:
        found = {}
        for cam, track in tracks.items():
            onset = find_motion_onset(track[color])
            if onset is not None:
                found[cam] = int(onset)
        if not tracks or len(found) < len(tracks):
            rejected[color] = "Bewegungsbeginn fehlt in einer Kopfkamera"
            continue
        spread = max(found.values()) - min(found.values())
        if spread > tolerance:
            rejected[color] = f"Kopfkameras uneins um {spread} Frames"
            continue
        per_color[color] = int(round(float(np.median(list(found.values())))))

    return {"per_color": per_color,
            "first": min(per_color.values()) if per_color else None,
            "rejected": rejected,
            "tolerance_frames": int(tolerance)}


# ---------------------------------------------------------------------------
# Rückprojektion
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Gierwinkel der Würfel um die eigene z-Achse
# ---------------------------------------------------------------------------

def top_face_mask(rgb: np.ndarray, mask: np.ndarray, bbox: list[int],
                  quantile: float = 70.0) -> np.ndarray | None:
    """Aus der Farbmaske die helle DECKFLÄCHE herausschneiden.

    Die Seitenflächen liegen im Schatten; ein Helligkeitsquantil innerhalb des Blobs
    trennt sie zuverlässig ab. Das ist nötig, weil nur die Deckfläche auf ``Z_CUBE_TOP``
    liegt — Seitenflächen-Pixel auf diese Ebene zurückzuprojizieren zieht sie zu einem
    Schweif von der Kamera weg und verdirbt jede Winkelmessung.
    """
    x0, y0, x1, y1 = bbox
    _, _, value = rgb_to_hsv(rgb)
    crop = mask[y0:y1 + 1, x0:x1 + 1]
    vals = value[y0:y1 + 1, x0:x1 + 1][crop]
    if vals.size < 12:
        return None
    bright = np.zeros_like(mask)
    bright[y0:y1 + 1, x0:x1 + 1] = crop & (
        value[y0:y1 + 1, x0:x1 + 1] >= float(np.percentile(vals, quantile)))
    return bright


def yaw_from_top_face(cam: PinholeCamera, uv: np.ndarray,
                      z_top: float = Z_CUBE_TOP) -> tuple[float, float] | None:
    """Gierwinkel aus den Pixeln der Deckfläche → (Grad in [0,90), Güte, Breite, Alternative).

    Zwei Entscheidungen, und beide sind der Grund, warum es überhaupt funktioniert:

    1. **Erst zurückprojizieren, dann messen.** Im Bild ist die Deckfläche ein perspektivisch
       verzerrtes Viereck; auf ihrer eigenen Ebene ist sie wieder ein Quadrat. Wer im
       Pixelraum misst, misst die Verzerrung mit.
    2. **Das 4. Winkelmoment statt einer Min-Area-Box.** Ein Quadrat ist 4-zählig, also
       steckt seine Richtung in genau dieser Harmonischen: ``Σ (dx + i·dy)^4`` hat die Phase
       4·ψ. Das benutzt jedes Pixel mit stetigem Gewicht und rastet darum nicht auf das
       Pixelraster ein — der Min-Area-Ansatz in ``replay_calibration.top_face_blob`` tut
       genau das und lieferte auf den 120 Realframes vom 2026-08-22 in 101 Fällen exakt 0,0°.

    Der Versatz von 45°: ``r^4`` gewichtet die ECKEN des Quadrats am stärksten, nicht die
    Kanten, und die liegen um 45° gedreht.

    Die Güte ist ``|Σ w z^4| / Σ w |z|^4`` ∈ [0, 1] — 1 heißt sauber 4-zählig, nahe 0 heißt
    „keine Vorzugsrichtung", also kein verwertbarer Winkel. Sie ersetzt keine Abnahme gegen
    Grundwahrheit, aber sie erkennt den entarteten Fall.
    """
    p = cam.backproject_many_to_plane(np.asarray(uv, dtype=float), z_top)
    p = p[np.isfinite(p).all(axis=-1)]
    if len(p) < 12:
        return None
    d = (p[:, 0] - p[:, 0].mean()) + 1j * (p[:, 1] - p[:, 1].mean())
    m = np.sum(d ** 4)
    scale = np.sum(np.abs(d) ** 4)
    if scale < 1e-12:
        return None
    # ``r^4`` gewichtet die ECKEN am stärksten, die liegen 45° neben den Kanten.
    yaw = (np.degrees(np.angle(m)) / 4.0 - 45.0) % 90.0

    # Formprobe. Ein 5-cm-Quadrat ist quer zur Kante 5 cm breit und quer zur Diagonale
    # 7,07 cm — das Verhältnis ist √2, und zwar unabhängig von Größe und Lage. Damit
    # fallen zwei Fehler auf, die die Phase allein nicht sieht:
    #   * Verhältnis < 1  → die Phase ist um 45° umgeschlagen (Ecke statt Kante). Das
    #     passiert bei verunreinigter Deckflächenmaske und ist bei einem 4-zähligen
    #     Würfel der größtmögliche Fehler; hier wird es zurückgedreht.
    #   * Verhältnis ≈ 1  → die Punktwolke ist gar kein Quadrat. Dann gibt es keine
    #     Kantenrichtung, und der Aufrufer soll den Winkel verwerfen statt ihn zu glauben.
    # Spannweite, nicht Perzentile: bei gefüllten Flächen ist die 5.–95.-Spanne für Kante
    # und Diagonale fast gleich (gemessen 1,08 statt 1,41) — die Unterscheidung stirbt.
    w_edge, w_diag = _extent_across(d, yaw), _extent_across(d, yaw + 45.0)
    if w_edge > 1e-9 and w_diag / w_edge < 1.0:
        yaw = (yaw + 45.0) % 90.0
        w_edge, w_diag = w_diag, w_edge
    squareness = float(w_diag / w_edge) if w_edge > 1e-9 else 0.0
    return float(yaw), float(np.abs(m) / scale), float(w_edge), squareness


def _extent_across(d: np.ndarray, yaw_deg: float) -> float:
    """Volle Spannweite der Punktwolke quer zu ``yaw_deg``, in Metern."""
    q = np.imag(d * np.exp(-1j * np.radians(float(yaw_deg))))
    return float(q.max() - q.min())


def accept_yaw(records: list[dict], tolerance_deg: float, min_squareness: float) -> bool:
    """Ist der Gierwinkel dieses Würfels belastbar? Zwei Tore, beide gemessen begründet.

    1. **Formprobe** je Kamera. Diagonale/Kante muss nahe √2 liegen; nahe 1 heißt, dass die
       Deckflächenmaske kein Quadrat mehr ist und die Kantenrichtung nicht existiert.
    2. **Beide Kameras einig.** Ihre Fehler sind weitgehend unabhängig — sie sehen den
       Würfel aus verschiedenen Richtungen, und die Verschmierung durch Seitenflächen zeigt
       jeweils anderswohin. Ein selbstbewusst falscher Wert überlebt das selten.

    Abnahme vom 2026-08-25 (synthetisch, 450 Würfel über fünf Rauschstufen bis σ=0,05):
    ohne Tore Median 3,15° / p90 26,3°; mit Formprobe 1,25 und Toleranz 8° bleiben 36 %
    der Würfel übrig, und für die gilt Median 0,12° / p90 1,68°, ein Ausreißer auf 160.
    Der Rest bekommt KEINEN Winkel — 0° ist ein ehrlicher Rückfall, ein geratener nicht.
    """
    if len(records) < 2:
        return False
    if any(r.get("top_squareness", 0.0) < min_squareness for r in records):
        return False
    yaws = [r["yaw_deg"] for r in records]
    return max(yaw_delta_deg(a, b) for a in yaws for b in yaws) <= tolerance_deg


def mean_yaw_deg(yaws) -> float | None:
    """Mittelwert von Winkeln mod 90° — über die 4-fache Phase, sonst mittelt 1° und 89° zu 45°."""
    a = np.asarray([y for y in yaws if y is not None], dtype=float)
    if a.size == 0:
        return None
    m = np.mean(np.exp(1j * np.radians(4.0 * a)))
    mean = (np.degrees(np.angle(m)) / 4.0) % 90.0
    # 89,999…° ist numerisch dasselbe wie 0° — ohne diesen Schnapp liest sich der
    # Mittelwert von 1° und 89° als „90°" und damit wie der größtmögliche Winkel.
    return float(0.0 if mean > 90.0 - 1e-6 else mean)


def yaw_delta_deg(a: float, b: float) -> float:
    """Abstand zweier Gierwinkel mod 90°, in [0, 45]."""
    d = (float(a) - float(b)) % 90.0
    return float(min(d, 90.0 - d))


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
        mask = color_mask(rgb, color)
        blob = largest_blob(mask, min_area=min_area)
        if blob is None:
            out.append(None)
            continue
        try:
            p = cam.backproject_to_plane(blob["u"], blob["v"], Z_CUBE_CENTER)
        except ValueError as exc:
            out.append({"color": color, "error": str(exc), **blob})
            continue
        p = p + np.array([bias_xy[0], bias_xy[1], 0.0])
        rec = {
            "color": color,
            "xy": [round(float(p[0]), 4), round(float(p[1]), 4)],
            "u": round(blob["u"], 2), "v": round(blob["v"], 2),
            "area": blob["area"], "extent_px": round(blob["extent"], 1),
            "dist_plane_m": round(float(np.linalg.norm(p - cam.eye)), 4),
            "dist_size_m": round(float(cam.f_px * CUBE_EDGE_M / blob["extent"]), 4),
            "cm_per_px": round(cam.plane_scale_cm_per_px(blob["u"], blob["v"],
                                                         Z_CUBE_CENTER), 4),
            # Füllgrad der Bounding-Box. Eine löchrige Farbmaske verzieht BEIDES —
            # den Schwerpunkt und den Gierwinkel. Auf den Realframes vom 2026-08-22
            # lag der Median bei 0,44, und genau daher kommt der Winkelfehler.
            "fill": round(blob["area"] / max(
                (blob["bbox"][2] - blob["bbox"][0] + 1)
                * (blob["bbox"][3] - blob["bbox"][1] + 1), 1), 3),
        }
        rec.update(measure_yaw(rgb, mask, blob, cam, min_area=min_area))
        out.append(rec)
    return out


def measure_yaw(rgb: np.ndarray, mask: np.ndarray, blob: dict, cam: PinholeCamera,
                min_area: int = 120, quantile: float = 70.0) -> dict:
    """Deckfläche isolieren und ihren Gierwinkel messen — Diagnosefelder inklusive.

    Getrennt von ``locate_cubes``, damit die Abnahme (``selftest``) genau diesen Weg
    aufrufen kann und nicht einen nachgebauten.
    """
    bright = top_face_mask(rgb, mask, blob["bbox"], quantile=quantile)
    if bright is None:
        return {"yaw_deg": None, "yaw_note": "keine Helligkeitswerte im Blob"}
    sel = largest_component(bright)
    if sel is None or int(sel.sum()) < max(25, min_area // 4):
        n = 0 if sel is None else int(sel.sum())
        return {"yaw_deg": None, "top_px": n,
                "yaw_note": f"Deckfläche zu klein ({n} px)"}
    ys, xs = np.nonzero(sel)
    got = yaw_from_top_face(cam, np.column_stack([xs, ys]).astype(float))
    if got is None:
        return {"yaw_deg": None, "top_px": int(sel.sum()),
                "yaw_note": "Rückprojektion entartet"}
    yaw, coherence, width_m, squareness = got
    return {"yaw_deg": round(yaw, 2), "yaw_coherence": round(coherence, 3),
            "top_px": int(sel.sum()),
            # Breite quer zur Kante — beim 5-cm-Würfel ~5 cm. Deutlich weniger heißt,
            # dass die Deckflächenmaske nur ein Bruchstück erwischt hat.
            "top_width_cm": round(width_m * 100, 2),
            # Diagonale/Kante. Ideal √2 = 1,41; nahe 1 heißt „kein Quadrat, kein Winkel".
            "top_squareness": round(squareness, 3)}


# ---------------------------------------------------------------------------
# Abnahme: synthetischer Würfel mit BEKANNTEM Gierwinkel
# ---------------------------------------------------------------------------

def _fill_polygon(uu: np.ndarray, vv: np.ndarray, poly: np.ndarray,
                  supersample: int = 5) -> np.ndarray:
    """Deckungsgrad je Pixel für ein Bildpolygon, 0…1 (Even-Odd-Regel, supersampelt)."""
    off = (np.arange(supersample) + 0.5) / supersample - 0.5
    cov = np.zeros(uu.shape, dtype=float)
    for du in off:
        for dv in off:
            px, py = uu + du, vv + dv
            inside = np.zeros(px.shape, dtype=bool)
            for i in range(len(poly)):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % len(poly)]
                with np.errstate(divide="ignore", invalid="ignore"):
                    xint = (x2 - x1) * (py - y1) / (y2 - y1) + x1
                inside ^= ((y1 > py) != (y2 > py)) & (px < xint)
            cov += inside
    return cov / (supersample * supersample)


def synthetic_cube_image(cam: PinholeCamera, center_xy, yaw_deg: float,
                         noise: float = 0.0, rng=None) -> np.ndarray:
    """Ein roter Würfel bekannten Gierwinkels, durch DIESES Kameramodell gerendert.

    Bewusst kein Isaac: die Abnahme soll den Schätzer prüfen, nicht den Renderer. Deck- und
    Seitenflächen bekommen verschiedene Helligkeit, weil ``top_face_mask`` genau davon lebt;
    ``noise`` verrauscht die Pixel VOR der Farbmaske und erzeugt so ausgefranste Masken.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    a = np.radians(float(yaw_deg))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    h = CUBE_EDGE_M / 2.0
    base = np.array([[-h, -h], [h, -h], [h, h], [-h, h]]) @ rot.T + np.asarray(center_xy)
    top = np.column_stack([base, np.full(4, Z_CUBE_TOP)])
    bottom = np.column_stack([base, np.full(4, Z_CUBE_TOP - CUBE_EDGE_M)])

    img = np.full((cam.height, cam.width, 3), 250.0)   # weißer Tisch
    faces = [(np.array([top[i], top[(i + 1) % 4], bottom[(i + 1) % 4], bottom[i]]),
              (130.0, 15.0, 15.0)) for i in range(4)]
    faces.append((top, (230.0, 20.0, 20.0)))      # Deckfläche zuletzt: sie liegt oben
    polys = [(cam.project(corners), colour) for corners, colour in faces]
    # Nur um den Würfel herum rastern. Über das ganze Bild zu laufen kostet bei 25
    # Supersamples je Fläche Sekunden statt Millisekunden — und die Abnahme soll
    # oft genug laufen, dass sie niemand wegen der Laufzeit überspringt.
    allp = np.concatenate([q for q, _ in polys])
    u0, v0 = np.clip(np.floor(allp.min(0)).astype(int) - 2, 0, [cam.width, cam.height])
    u1, v1 = np.clip(np.ceil(allp.max(0)).astype(int) + 2, 0, [cam.width, cam.height])
    if u1 <= u0 or v1 <= v0:
        return img.astype(np.uint8)
    uu, vv = np.meshgrid(np.arange(u0, u1), np.arange(v0, v1))
    tile = img[v0:v1, u0:u1]
    for poly, colour in polys:
        cov = _fill_polygon(uu, vv, poly)[..., None]
        tile = tile * (1.0 - cov) + np.asarray(colour) * cov
    img[v0:v1, u0:u1] = tile
    if noise > 0.0:
        img = img + rng.normal(0.0, noise * 255.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def run_selftest(args) -> int:
    """Bekannter Winkel rein, gemessener Winkel raus — über den vollen Messpfad.

    Deckt für den Gierwinkel ab, was ``detect --expect`` für die Position tut: ohne diese
    Zahl ist jeder Winkel plausibel und unbelegt. Gemessen wird der Weg, den ``extract``
    auch geht — beide Kameras, Mittel über die 4-fache Phase, Tor auf ihre Uneinigkeit —
    und nicht die nackte Formel: Farbmaske, Blobwahl und Deckflächenschnitt gehören zum
    Fehler dazu.

    Zwei Zahlen entscheiden. Der **Fehler der behaltenen** Würfel sagt, wie genau ein
    Winkel ist, der das Tor passiert. Die **Ausreißerquote** (>20°) sagt, wie oft beide
    Kameras gemeinsam auf die Diagonale statt die Kante einrasten — ein 45°-Fehler ist bei
    einem 4-zähligen Würfel der größtmögliche und setzt die Ecke dorthin, wo die Fläche
    hingehört. Diese Fälle sind selten, aber sie sind nicht harmlos.
    """
    cams = {name: PinholeCamera.from_cfg(name)
            for name in ("cam_left_high", "cam_right_high")}
    rng = np.random.default_rng(args.seed)
    truth = np.arange(0.0, 90.0, float(args.step))
    centers = [(0.30, -0.18), (0.35, 0.0), (0.40, 0.18), (0.32, 0.10), (0.38, -0.08)]
    print(f"Abnahme Gierwinkel: {len(truth)} Winkel × {len(centers)} Orte × "
          f"{len(cams)} Kameras, Tor: Formprobe ≥ {args.min_squareness:.2f} und "
          f"Kameras einig ≤ {args.yaw_tolerance:.0f}°")
    verdict = True
    for noise in args.noise:
        kept, errs, gross, total = 0, [], 0, 0
        for center in centers:
            for yaw in truth:
                est = []
                for name, cam in cams.items():
                    rgb = synthetic_cube_image(cam, center, yaw, noise=noise, rng=rng)
                    rec = locate_cubes(rgb, cam, min_area=args.min_area)[0]
                    if rec is not None and rec.get("yaw_deg") is not None:
                        est.append(rec)
                if len(est) < 2:
                    continue
                total += 1
                if not accept_yaw(est, args.yaw_tolerance, args.min_squareness):
                    continue
                kept += 1
                err = yaw_delta_deg(mean_yaw_deg([e["yaw_deg"] for e in est]), yaw)
                errs.append(err)
                gross += err > 20.0
        if not errs:
            print(f"  Rauschen {noise:.2f}: kein Würfel überlebt das Tor.")
            verdict = False
            continue
        e = np.array(errs)
        p90 = float(np.percentile(e, 90))
        rate = gross / len(e)
        # Unter zehn Überlebenden ist eine Ausreißerquote selbst Rauschen — ein Fall von
        # vieren sind „25 %". Solche Stufen werden gezeigt, aber nicht gewertet: der
        # Ertragseinbruch IST hier das Ergebnis, der Schätzer gibt sauber auf.
        judged = len(e) >= 10
        ok = p90 <= args.tolerance and rate <= args.max_gross
        verdict &= ok or not judged
        mark = ("ok" if ok else "FEHLER") if judged else "zu wenige für ein Urteil"
        print(f"  Rauschen {noise:.2f}: Tor behält {kept:3d}/{total:3d} "
              f"({100 * kept / total:3.0f} %)"
              f"  Fehler Median {np.median(e):5.2f}  p90 {p90:5.2f}  max {e.max():5.2f}°"
              f"  Ausreißer>20° {gross}/{len(e)} ({100 * rate:.1f} %)   {mark}")
    print(f"\n{'BESTANDEN' if verdict else 'DURCHGEFALLEN'}: verlangt p90 ≤ "
          f"{args.tolerance:.1f}° und ≤ {100 * args.max_gross:.0f} % Ausreißer je gewerteter "
          f"Rauschstufe (≥ 10 Überlebende).")
    return 0 if verdict else 1


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
    expect_yaw = json.loads(args.expect_yaw) if args.expect_yaw else None
    residuals: list[np.ndarray] = []
    yaw_residuals: list[float] = []

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
            if f.get("yaw_deg") is None:
                print(f"       Gierwinkel: — ({f.get('yaw_note', 'nicht gemessen')})")
                continue
            yaw_line = (f"       Gierwinkel {f['yaw_deg']:5.1f}°  Deckfläche {f['top_px']:4d} px"
                        f"  Breite {f['top_width_cm']:4.1f} cm"
                        f"  Formprobe {f['top_squareness']:.2f}"
                        f"  Güte {f['yaw_coherence']:.2f}"
                        f"  Maskenfüllung {f['fill']:.2f}")
            if expect_yaw is not None and i < len(expect_yaw):
                d = yaw_delta_deg(f["yaw_deg"], float(expect_yaw[i]))
                yaw_residuals.append(d)
                yaw_line += f"  Δ {d:4.1f}°"
            print(yaw_line)
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
    if yaw_residuals:
        y = np.array(yaw_residuals)
        print(f"\nGierwinkel gegen Grundwahrheit über {len(y)} Würfel (je Kamera einzeln, "
              f"ohne das Zwei-Kamera-Tor):")
        print(f"  Median {np.median(y):.1f}°, p90 {np.percentile(y, 90):.1f}°, "
              f"max {y.max():.1f}°")
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

        cubes, spread, yaws, yaw_spread = [], [], [], []
        for i in range(len(CUBE_COLORS)):
            recs = [per_cam[c][i] for c in per_cam if per_cam[c][i]]
            picks = [r["xy"] for r in recs if "xy" in r]
            if not picks:
                cubes.append(None)
                yaws.append(None)
                continue
            arr = np.array(picks, dtype=float)
            cubes.append([round(float(arr[:, 0].mean()), 4), round(float(arr[:, 1].mean()), 4)])
            if len(arr) > 1:
                spread.append(float(np.linalg.norm(arr[0] - arr[1])))

            # Gierwinkel nur, wenn er beide Tore passiert — sonst None. Der Renderer setzt
            # dann 0°, und das ist ein ehrlicher Rückfall: ein geratener Winkel dreht den
            # Würfel unter einer Realaktion weg, die für eine andere Lage aufgenommen wurde.
            turned = [r for r in recs if r.get("yaw_deg") is not None]
            if accept_yaw(turned, args.yaw_tolerance, args.min_squareness):
                yaws.append(round(mean_yaw_deg([r["yaw_deg"] for r in turned]), 2))
                yaw_spread.append(max(yaw_delta_deg(a["yaw_deg"], b["yaw_deg"])
                                      for a in turned for b in turned))
            else:
                yaws.append(None)

        onsets = ({"per_color": {}, "first": None, "rejected": {}, "tolerance_frames": 0}
                  if args.no_motion_onset
                  else episode_motion_onsets(root, ep, cams, min_area=args.min_area))
        layout["episodes"][str(ep)] = {
            "frame": args.frame,
            "cubes": cubes,
            # Gierwinkel um die eigene z-Achse, in Grad, mod 90° (der Würfel ist 4-zählig).
            # None heißt „nicht belastbar gemessen" — NICHT „liegt gerade". Wer den Schlüssel
            # gar nicht findet, liest ein layout.json von vor dem 2026-08-25.
            "cubes_yaw_deg": yaws,
            "yaw_gate": {"tolerance_deg": args.yaw_tolerance,
                         "min_squareness": args.min_squareness},
            # Fenstergrenze für den Renderer, siehe episode_motion_onsets.
            "motion_onset": onsets,
            "per_camera": per_cam,
            # Der Abgleich beider Kameras prüft die Detektion, NICHT das Kameramodell:
            # bei 4,7 cm Stereobasis auf 0,5 m verschiebt ein Modellfehler beide Strahlen
            # fast gleich. Eine kleine Zahl hier heißt also „beide Kameras sehen dasselbe",
            # nicht „die Pose stimmt".
            "camera_spread_cm": [round(s * 100, 2) for s in spread],
            "camera_yaw_spread_deg": [round(d, 2) for d in yaw_spread],
        }
        n_ok = sum(1 for c in cubes if c)
        first = onsets["first"]
        onset_txt = (f", Bewegung ab Frame {first} ({len(onsets['per_color'])}/3 Farben)"
                     if first is not None
                     else (", KEIN belastbarer Bewegungsbeginn" if not args.no_motion_onset
                           else ""))
        n_yaw = sum(1 for y in yaws if y is not None)
        print(f"{head}: {n_ok}/3 Würfel, {n_yaw}/3 mit Gierwinkel{onset_txt}"
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
    all_yaw = [y for r in recs.values() for y in r.get("cubes_yaw_deg", [])]
    got = [y for y in all_yaw if y is not None]
    if all_yaw:
        print(f"[layout] Gierwinkel: {len(got)}/{len(all_yaw)} Würfel durch das Tor "
              f"({100 * len(got) / len(all_yaw):.0f} %). Der Rest bleibt bei 0°.")
    if got:
        folded = np.abs(np.where(np.array(got) > 45.0, np.array(got) - 90.0, got))
        print(f"[layout] gemessene Schräglage gegen die Tischkante: Median "
              f"{np.median(folded):.1f}°, p90 {np.percentile(folded, 90):.1f}°, "
              f"max {folded.max():.1f}°")
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

    yaw_opts = argparse.ArgumentParser(add_help=False)
    yaw_opts.add_argument("--yaw-tolerance", type=float, default=8.0,
                          help="Wieviel Grad die beiden Kameras beim Gierwinkel "
                               "auseinanderliegen dürfen (Default 8)")
    yaw_opts.add_argument("--min-squareness", type=float, default=1.25,
                          help="Verlangte Diagonale/Kante der Deckfläche. Ideal √2=1,41, "
                               "nahe 1 heißt 'kein Quadrat' (Default 1,25)")

    d = sub.add_parser("detect", parents=[common, yaw_opts],
                       help="Blobs in gegebenen Bildern; mit --expect gegen Grundwahrheit")
    d.add_argument("images", nargs="+")
    d.add_argument("--camera", default="cam_left_high",
                   choices=("cam_left_high", "cam_right_high", "cam_scene"))
    d.add_argument("--expect", type=str, default="",
                   help='Bekannte Würfelpositionen als JSON, z. B. '
                        '"[[0.34,-0.15,0.895],[0.36,0,0.895],[0.34,0.15,0.895]]" '
                        '(render_manifest.json → cubes_xyz)')
    d.add_argument("--expect-yaw", type=str, default="",
                   help='Bekannte Gierwinkel in Grad als JSON, z. B. "[0,20,40]" '
                        '(render_manifest.json → cubes_yaw_deg). Dieselbe Abnahme wie '
                        '--expect, nur für die Drehung.')

    sub.add_parser("selftest", parents=[yaw_opts],
                   help="Gierwinkel-Schätzer gegen synthetische Grundwahrheit prüfen "
                        "— ohne Isaac, ohne Datensatz").set_defaults(
        seed=0, step=5.0, noise=[0.0, 0.01, 0.02, 0.03, 0.05], min_area=120,
        tolerance=5.0, max_gross=0.02)

    e = sub.add_parser("extract", parents=[common, yaw_opts],
                       help="Realbilder eines Datensatzes → layout.json")
    e.add_argument("--dataset-path", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--num-episodes", type=int, default=60)
    e.add_argument("--episode-ids", type=int, nargs="*", default=None)
    e.add_argument("--train-ratio", type=float, default=0.8)
    e.add_argument("--cameras", default="cam_left_high,cam_right_high")
    e.add_argument("--overwrite", action="store_true")
    e.add_argument("--no-motion-onset", action="store_true",
                   help="Bewegungsbeginn NICHT bestimmen. Spart Zeit (sonst wird jedes Video "
                        "einmal in halber Auflösung dekodiert, grob 4 s je Video und Kamera), "
                        "aber render_cotrain_dataset.py kann dann kein Fenster setzen.")

    args = ap.parse_args()
    if CAMERA_CFG.width != 640 or CAMERA_CFG.height != 480:
        print(f"[warn] Kamerakonfiguration steht auf {CAMERA_CFG.width}×{CAMERA_CFG.height}.")
    return {"detect": run_detect, "extract": run_extract,
            "selftest": run_selftest}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
