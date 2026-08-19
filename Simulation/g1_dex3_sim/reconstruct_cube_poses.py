#!/usr/bin/env python3
"""Kalibrierung und Würfelposen für physikbasierte Dataset-Replay-Videos.

Dieses Werkzeug läuft ohne Isaac Lab. Es liest den nach LeRobot v2.1 konvertierten
G1-DEX3-Datensatz, segmentiert die drei farbigen Würfel in den beiden festen
Kopfkameras und schreibt ausschließlich Arbeitsartefakte für ``run_dataset_replay_videos``.

Die Würfelpose wird nur am Episodenanfang bestimmt. Der spätere Replay setzt jeden
Würfel genau einmal; dieses Skript erzeugt keine Trajektorie und kein Tracking.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_geometry import (  # noqa: E402
    CAMERA_CFG,
    PinholeCamera,
    look_at_world_quat,
    quat_to_matrix,
)
from extract_block_layout import (  # noqa: E402
    CUBE_COLORS,
    Z_CUBE_CENTER,
    color_mask,
    largest_blob,
)
from replay_calibration import (  # noqa: E402
    apply_homography,
    best_top_face_detection,
    homography_report,
    read_json,
)

HEAD_CAMS = ("cam_left_high", "cam_right_high")
COLOR_TO_BLOCK = {"rot": "block_0", "gruen": "block_1", "gelb": "block_2"}
DEFAULT_VIDEO_TEMPLATE = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)
DEFAULT_DATA_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"


def read_info(root: Path) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    if not path.is_file():
        raise SystemExit(f"Datensatz-Metadaten fehlen: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def select_episodes(info: dict[str, Any], args: argparse.Namespace) -> list[int]:
    total = int(info["total_episodes"])
    train_stop = int(total * float(args.train_ratio))
    if args.episode_ids:
        episodes = sorted(set(int(ep) for ep in args.episode_ids))
    else:
        start = int(args.start_episode)
        episodes = list(range(start, min(start + int(args.num_episodes), train_stop)))
    bad = [ep for ep in episodes if ep < 0 or ep >= train_stop]
    if bad:
        raise SystemExit(
            f"Episoden außerhalb des Trainingsbereichs 0:{train_stop}: {bad}. "
            "Testepisoden werden für eine spätere Trainingsnutzung nicht verarbeitet."
        )
    return episodes


def format_dataset_path(root: Path, template: str, info: dict[str, Any], ep: int,
                        video_key: str | None = None) -> Path:
    values = {
        "episode_chunk": ep // int(info.get("chunks_size", 1000)),
        "episode_index": ep,
        "video_key": video_key,
    }
    try:
        return root / template.format(**values)
    except KeyError as exc:
        raise SystemExit(
            "Der Datensatz ist noch LeRobot v3.0. Zuerst `server_rl_run.sh "
            "replay-prepare` ausführen (v3→v2.1)."
        ) from exc


def video_path(root: Path, info: dict[str, Any], ep: int, cam: str) -> Path:
    template = info.get("video_path") or DEFAULT_VIDEO_TEMPLATE
    return format_dataset_path(root, template, info, ep, f"observation.images.{cam}")


def data_path(root: Path, info: dict[str, Any], ep: int) -> Path:
    return format_dataset_path(root, info.get("data_path", DEFAULT_DATA_TEMPLATE), info, ep)


def load_video_frames(path: Path, limit: int) -> list[np.ndarray]:
    import imageio.v2 as imageio

    if not path.is_file():
        raise FileNotFoundError(path)
    frames: list[np.ndarray] = []
    with imageio.get_reader(str(path), format="FFMPEG") as reader:
        for idx, frame in enumerate(reader):
            if idx >= limit:
                break
            frames.append(np.asarray(frame, dtype=np.uint8)[..., :3])
    if not frames:
        raise RuntimeError(f"Video enthält keine Frames: {path}")
    return frames


def find_all_cubes(rgb: np.ndarray, min_area: int) -> list[dict[str, Any] | None]:
    return [largest_blob(color_mask(rgb, color), min_area=min_area) for color in CUBE_COLORS]


def best_detection(frames: list[np.ndarray], min_area: int) -> tuple[int, np.ndarray, list]:
    """Wählt den frühen Frame mit den meisten und größten vollständigen Farbblobs."""
    best: tuple[tuple[int, int], int, np.ndarray, list] | None = None
    for idx, rgb in enumerate(frames):
        found = find_all_cubes(rgb, min_area)
        count = sum(blob is not None for blob in found)
        area = sum(int(blob["area"]) for blob in found if blob is not None)
        candidate = ((count, area), idx, rgb, found)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2], best[3]


def camera_record(name: str) -> dict[str, Any]:
    pose_name = name if not name.endswith("_wrist") else f"{name}_local"
    pose = getattr(CAMERA_CFG, pose_name)
    focal = CAMERA_CFG.focal_wrist if name.endswith("_wrist") else CAMERA_CFG.focal_high
    return {
        "eye": [float(v) for v in pose["pos"]],
        "quat_wxyz": [float(v) for v in pose["rot"]],
        "focal_mm": float(focal),
        "aperture_mm": float(CAMERA_CFG.horizontal_aperture_mm),
        "width": int(CAMERA_CFG.width),
        "height": int(CAMERA_CFG.height),
    }


def camera_from_record(rec: dict[str, Any]) -> PinholeCamera:
    return PinholeCamera(
        rec["eye"], rec["quat_wxyz"], rec["focal_mm"], rec["aperture_mm"],
        rec["width"], rec["height"],
    )


def save_detection_overlay(rgb: np.ndarray, found: list, path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    colors = ("red", "green", "yellow")
    for color, blob in zip(colors, found):
        if blob is None:
            continue
        x0, y0, x1, y1 = blob["bbox"]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        u, v = blob["u"], blob["v"]
        draw.line((u - 8, v, u + 8, v), fill="white", width=2)
        draw.line((u, v - 8, u, v + 8), fill="white", width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def run_inspect(args: argparse.Namespace) -> int:
    root = Path(args.dataset_path)
    info = read_info(root)
    errors: list[str] = []
    if info.get("codebase_version") != "v2.1":
        errors.append(f"codebase_version={info.get('codebase_version')!r}, erwartet 'v2.1'")
    if abs(float(info.get("fps", 0.0)) - 30.0) > 1e-6:
        errors.append(f"fps={info.get('fps')}, erwartet 30")
    for key in ("observation.state", "action"):
        shape = info.get("features", {}).get(key, {}).get("shape")
        if shape != [28]:
            errors.append(f"{key}.shape={shape}, erwartet [28]")
    for cam in HEAD_CAMS + ("cam_left_wrist", "cam_right_wrist"):
        feature = info.get("features", {}).get(f"observation.images.{cam}", {})
        if feature.get("shape") != [3, 480, 640]:
            errors.append(f"{cam}.shape={feature.get('shape')}, erwartet [3, 480, 640]")

    episodes = select_episodes(info, args)
    for ep in episodes[:1]:
        if not data_path(root, info, ep).is_file():
            errors.append(f"Episode-Parquet fehlt: {data_path(root, info, ep)}")
        for cam in HEAD_CAMS + ("cam_left_wrist", "cam_right_wrist"):
            path = video_path(root, info, ep, cam)
            if not path.is_file():
                errors.append(f"Video fehlt: {path}")
    if errors:
        for error in errors:
            print(f"[replay-prepare] FEHLER: {error}")
        return 1
    print(f"[replay-prepare] Datensatz bereit: {root}")
    print(f"[replay-prepare] {info['total_episodes']} Episoden, 30 Hz, 28 DoF, vier Kameras.")
    print("[replay-prepare] fertig.", flush=True)
    return 0


def run_calibrate(args: argparse.Namespace) -> int:
    if args.anchors:
        return run_anchor_calibrate(args)
    root = Path(args.dataset_path)
    info = read_info(root)
    episodes = select_episodes(info, args)
    out = Path(args.out)
    debug = Path(args.debug_dir)
    cameras = {name: PinholeCamera.from_cfg(name) for name in HEAD_CAMS}
    edge_estimates: list[float] = []
    samples: list[dict[str, Any]] = []

    for ep in episodes:
        for cam_name in HEAD_CAMS:
            path = video_path(root, info, ep, cam_name)
            try:
                frames = load_video_frames(path, args.max_frames)
            except (FileNotFoundError, RuntimeError) as exc:
                print(f"[replay-calibrate] Episode {ep}, {cam_name}: {exc}")
                continue
            frame_idx, rgb, found = best_detection(frames, args.min_area)
            save_detection_overlay(rgb, found, debug / f"ep{ep:06d}_{cam_name}.png")
            for color, blob in zip(CUBE_COLORS, found):
                if blob is None:
                    continue
                point = cameras[cam_name].backproject_to_plane(
                    float(blob["u"]), float(blob["v"]), Z_CUBE_CENTER
                )
                distance = float(np.linalg.norm(point - cameras[cam_name].eye))
                x0, _, x1, _ = blob["bbox"]
                width_px = float(x1 - x0 + 1)
                edge_m = width_px * distance / cameras[cam_name].f_px
                edge_estimates.append(edge_m)
                samples.append({
                    "episode": ep,
                    "camera": cam_name,
                    "frame": frame_idx,
                    "color": color,
                    "u": round(float(blob["u"]), 3),
                    "v": round(float(blob["v"]), 3),
                    "width_px": round(width_px, 2),
                    "bbox": [int(v) for v in blob["bbox"]],
                    "distance_m": round(distance, 4),
                    "edge_m": round(edge_m, 5),
                })

    if len(edge_estimates) < 6:
        raise SystemExit(
            f"Nur {len(edge_estimates)} gültige Größenmessungen; mindestens 6 erforderlich."
        )
    raw_edge = float(np.median(edge_estimates))
    # Die horizontale Silhouettenbreite enthält bei gedrehten Würfeln einen yaw-abhängigen
    # Anteil. Der physische Würfel ist laut Asset und Datensatzaufbau 5 cm groß; die
    # Bildmessung ist deshalb eine Abnahme der Skala, nicht ein Grund, jede Episode anders
    # zu skalieren.
    if not args.edge_min <= raw_edge <= args.edge_max:
        raise SystemExit(
            f"Bildbasierte Würfelkante {raw_edge * 100:.2f} cm außerhalb "
            f"{args.edge_min * 100:.1f}–{args.edge_max * 100:.1f} cm. "
            "Kamera/Skala prüfen; keine Kalibrierung geschrieben."
        )
    edge_m = float(args.cube_edge)

    # Kleine, getrennte FOV-Korrektur je Kopfkamera: Wenn ein bekannter 5-cm-Würfel mit
    # dem aktuellen Pinhole-Modell z. B. als 5,2 cm erscheint, muss die Brennweite um
    # denselben Faktor steigen. Die Korrektur bleibt in den Arbeitsartefakten und ändert
    # die globale Eval-Kamera nicht.
    camera_records = {name: camera_record(name) for name in HEAD_CAMS}
    for cam_name in HEAD_CAMS:
        cam_edges = [
            float(sample["edge_m"]) for sample in samples if sample["camera"] == cam_name
        ]
        if not cam_edges:
            raise SystemExit(f"Keine Größenmessung für {cam_name}.")
        focal_scale = float(np.median(cam_edges)) / edge_m
        if not 0.8 <= focal_scale <= 1.3:
            raise SystemExit(
                f"Erforderliche FOV-Korrektur für {cam_name} ist unplausibel: "
                f"Faktor {focal_scale:.3f}."
            )
        camera_records[cam_name]["focal_mm"] *= focal_scale
        camera_records[cam_name]["focal_scale"] = focal_scale
    calibrated_cameras = {
        name: camera_from_record(record) for name, record in camera_records.items()
    }

    # Die reine Sehstrahl-Triangulation ist bei der nur ungefähr bekannten realen
    # Stereo-Basis zu empfindlich: schon wenige Millimeter Basisfehler verschieben die
    # geschätzte Tischhöhe um Dezimeter. Sie bleibt deshalb ein Diagnosewert. Die Höhe
    # wird stattdessen aus dem gemeinsamen 3D-Würfelmodell und seiner bekannten Kantenlänge
    # geschätzt. Dabei müssen beide Kameras gleichzeitig zu allen vier Bounding-Box-Kanten
    # passen.
    by_key = {(s["episode"], s["color"], s["camera"]): s for s in samples}
    stereo_centers: list[np.ndarray] = []
    stereo_gaps: list[float] = []
    model_centers: list[np.ndarray] = []
    model_errors: list[float] = []
    for ep in episodes:
        for color in CUBE_COLORS:
            left = by_key.get((ep, color, "cam_left_high"))
            right = by_key.get((ep, color, "cam_right_high"))
            if left is None or right is None:
                continue
            c1, c2 = calibrated_cameras["cam_left_high"], calibrated_cameras["cam_right_high"]
            d1 = c1.ray(float(left["u"]), float(left["v"]))
            d2 = c2.ray(float(right["u"]), float(right["v"]))
            w0 = c1.eye - c2.eye
            a, b, c = float(d1 @ d1), float(d1 @ d2), float(d2 @ d2)
            d, e = float(d1 @ w0), float(d2 @ w0)
            denominator = a * c - b * b
            if abs(denominator) < 1e-8:
                continue
            t1 = (b * e - c * d) / denominator
            t2 = (a * e - b * d) / denominator
            if t1 <= 0.0 or t2 <= 0.0:
                continue
            p1, p2 = c1.eye + t1 * d1, c2.eye + t2 * d2
            stereo_centers.append((p1 + p2) / 2.0)
            stereo_gaps.append(float(np.linalg.norm(p1 - p2)))

            blobs = {
                "cam_left_high": {
                    "u": float(left["u"]), "v": float(left["v"]), "bbox": left["bbox"],
                },
                "cam_right_high": {
                    "u": float(right["u"]), "v": float(right["v"]), "bbox": right["bbox"],
                },
            }
            center, _, fit_error = fit_cube_pose_3d(
                blobs, calibrated_cameras, Z_CUBE_CENTER, edge_m,
            )
            if fit_error <= args.max_fit_error_px:
                model_centers.append(center)
                model_errors.append(fit_error)
    if len(model_centers) < 6:
        raise SystemExit(
            f"Nur {len(model_centers)} gültige gemeinsame 3D-Würfel-Fits; mindestens 6 "
            f"mit höchstens {args.max_fit_error_px:.1f} px Fehler nötig."
        )
    center_z = float(np.median(np.asarray(model_centers)[:, 2]))
    if not 0.85 <= center_z <= 1.00:
        raise SystemExit(
            f"Triangulierter Würfelmittelpunkt z={center_z:.3f} m ist unplausibel."
        )
    table_top_z = center_z - edge_m / 2.0

    known = np.array([
        [0.34, -0.15, center_z],
        [0.36, 0.00, center_z],
        [0.34, 0.15, center_z],
    ])
    roundtrip_errors = []
    for camera in calibrated_cameras.values():
        for point, uv in zip(known, camera.project(known)):
            back = camera.backproject_to_plane(float(uv[0]), float(uv[1]), center_z)
            roundtrip_errors.append(float(np.linalg.norm(back - point)))

    payload = {
        "version": 1,
        "source_dataset": str(root),
        "episodes": episodes,
        "cameras": camera_records,
        "cube_edge_m": edge_m,
        "table_top_z_m": table_top_z,
        "cube_center_z_m": center_z,
        "quality": {
            "num_edge_samples": len(edge_estimates),
            "observed_edge_median_m": raw_edge,
            "observed_edge_p10_m": float(np.percentile(edge_estimates, 10)),
            "observed_edge_p90_m": float(np.percentile(edge_estimates, 90)),
            "stereo_samples": len(stereo_centers),
            "stereo_center_z_median_m": (
                float(np.median(np.asarray(stereo_centers)[:, 2]))
                if stereo_centers else None
            ),
            "stereo_ray_gap_median_m": (
                float(np.median(stereo_gaps)) if stereo_gaps else None
            ),
            "stereo_ray_gap_p90_m": (
                float(np.percentile(stereo_gaps, 90)) if stereo_gaps else None
            ),
            "cube_model_samples": len(model_centers),
            "cube_model_fit_error_median_px": float(np.median(model_errors)),
            "cube_model_fit_error_p90_px": float(np.percentile(model_errors, 90)),
            "projection_roundtrip_max_m": max(roundtrip_errors),
        },
        "samples": samples,
        "notes": [
            "Kopfkameraposen stammen aus camera_geometry.py und werden gegen die reale "
            "Würfelgröße geprüft.",
            "Die Würfelkante ist global 0.05 m; je Kopfkamera wird nur die Brennweite "
            "an der beobachteten Silhouettenskala korrigiert.",
            "Würfelmittelpunkt und Tischhöhe stammen aus einem gemeinsamen 3D-Würfel-Fit "
            "beider Kopfkameras; reine Stereo-Triangulation bleibt nur Diagnosewert.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[replay-calibrate] Bildkante Median {raw_edge * 100:.2f} cm; verwendet 5.00 cm.")
    print(f"[replay-calibrate] 3D-Würfelmodell: Mittelpunkt z={center_z:.3f} m; "
          f"Tischoberkante z={table_top_z:.3f} m.")
    if stereo_centers:
        stereo_z = float(np.median(np.asarray(stereo_centers)[:, 2]))
        if abs(stereo_z - center_z) > 0.05:
            print(
                f"[replay-calibrate] Hinweis: reine Stereo-Triangulation ergibt "
                f"z={stereo_z:.3f} m. Wegen der unsicheren realen Kamerabasis wird dieser "
                "Wert nur diagnostisch gespeichert.",
                flush=True,
            )
    print(f"[replay-calibrate] {out}")
    print("[replay-calibrate] fertig.", flush=True)
    return 0


def optimize_head_camera(camera_name: str, anchors: list[dict[str, Any]],
                         z_top: float) -> tuple[dict[str, Any], dict[str, float]]:
    usable = [a for a in anchors if camera_name in a.get("pixels", {})]
    world = np.asarray([[*a["world_xy_m"], z_top] for a in usable], dtype=float)
    observed = np.asarray(
        [a["pixels"][camera_name]["top_uv"] for a in usable], dtype=float
    )
    base = camera_record(camera_name)
    camera = camera_from_record(base)
    direction = camera.R[:, 0]
    distance = (z_top - camera.eye[2]) / direction[2]
    target = camera.eye + distance * direction
    params = np.array([
        *camera.eye, target[0], target[1], float(base["focal_mm"]),
    ])
    lower = np.array([-0.10, -0.15, 1.10, 0.15, -0.25, 7.0])
    upper = np.array([0.25, 0.15, 1.55, 0.75, 0.25, 25.0])

    def record(values: np.ndarray) -> dict[str, Any]:
        eye = values[:3]
        target_point = (values[3], values[4], z_top)
        quat = look_at_world_quat(eye, target_point)
        return {
            **base,
            "eye": [float(v) for v in eye],
            "quat_wxyz": [float(v) for v in quat],
            "focal_mm": float(values[5]),
        }

    def loss(values: np.ndarray) -> float:
        model = camera_from_record(record(values))
        projected = model.project(world)
        return float(np.sqrt(np.mean(np.square(projected - observed))))

    best = (loss(params), params.copy())
    steps = np.array([0.025, 0.025, 0.025, 0.035, 0.035, 1.5])
    while float(np.max(steps)) > 0.0005:
        improved = False
        for index in range(6):
            for sign in (-1.0, 1.0):
                candidate = best[1].copy()
                candidate[index] = np.clip(
                    candidate[index] + sign * steps[index], lower[index], upper[index]
                )
                score = loss(candidate)
                if score < best[0]:
                    best, improved = (score, candidate), True
        if not improved:
            steps[:6] *= 0.5
    fitted = record(best[1])
    errors = np.linalg.norm(camera_from_record(fitted).project(world) - observed, axis=1)
    return fitted, {
        "median_pixel_error": float(np.median(errors)),
        "p90_pixel_error": float(np.percentile(errors, 90)),
        "rmse_pixel": float(best[0]),
    }


def matrix_to_quat(matrix: np.ndarray) -> list[float]:
    matrix = np.asarray(matrix, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(np.array([
        [matrix[0, 0] - matrix[1, 1] - matrix[2, 2],
         matrix[1, 0] + matrix[0, 1], matrix[2, 0] + matrix[0, 2],
         matrix[1, 2] - matrix[2, 1]],
        [matrix[1, 0] + matrix[0, 1], matrix[1, 1] - matrix[0, 0] - matrix[2, 2],
         matrix[2, 1] + matrix[1, 2], matrix[2, 0] - matrix[0, 2]],
        [matrix[2, 0] + matrix[0, 2], matrix[2, 1] + matrix[1, 2],
         matrix[2, 2] - matrix[0, 0] - matrix[1, 1],
         matrix[0, 1] - matrix[1, 0]],
        [matrix[1, 2] - matrix[2, 1], matrix[2, 0] - matrix[0, 2],
         matrix[0, 1] - matrix[1, 0], matrix[0, 0] + matrix[1, 1] + matrix[2, 2]],
    ]) / 3.0)
    quaternion = eigenvectors[:, np.argmax(eigenvalues)][[3, 0, 1, 2]]
    if quaternion[0] < 0:
        quaternion *= -1
    return [float(v) for v in quaternion]


def rotation_xyz(angles: np.ndarray) -> np.ndarray:
    x, y, z = angles
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    return np.array([
        [cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy],
        [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
        [-sy, cy * sx, cx * cy],
    ])


def optimize_wrist_camera(camera_name: str, observations: list[dict[str, Any]],
                          z_top: float) -> tuple[dict[str, Any], dict[str, Any]]:
    usable = [item for item in observations if item["camera"] == camera_name]
    base = camera_record(camera_name)
    if len(usable) < 12 or len({item["episode"] for item in usable}) < 2:
        return base, {
            "status": "insufficient_observations",
            "samples": len(usable),
            "episodes": len({item["episode"] for item in usable}),
        }
    base_eye = np.asarray(base["eye"], dtype=float)
    base_rotation = quat_to_matrix(base["quat_wxyz"])
    params = np.r_[base_eye, np.zeros(3), float(base["focal_mm"])]
    lower = np.r_[base_eye - 0.05, [-0.45] * 3, 7.0]
    upper = np.r_[base_eye + 0.05, [0.45] * 3, 30.0]

    def projected(values: np.ndarray) -> np.ndarray:
        local_eye = values[:3]
        local_rotation = base_rotation @ rotation_xyz(values[3:6])
        focal_px = values[6] / float(base["aperture_mm"]) * int(base["width"])
        pixels = []
        for item in usable:
            link_rotation = quat_to_matrix(item["link_quat_wxyz"])
            eye = np.asarray(item["link_position_m"]) + link_rotation @ local_eye
            camera_rotation = link_rotation @ local_rotation
            point = np.array([*item["world_xy_m"], z_top], dtype=float)
            camera_point = (point - eye) @ camera_rotation
            if camera_point[0] <= 1e-6:
                pixels.append([np.nan, np.nan])
                continue
            pixels.append([
                base["width"] / 2.0 - 0.5 - focal_px * camera_point[1] / camera_point[0],
                base["height"] / 2.0 - 0.5 - focal_px * camera_point[2] / camera_point[0],
            ])
        return np.asarray(pixels)

    observed = np.asarray([item["top_uv"] for item in usable], dtype=float)

    def loss(values: np.ndarray) -> float:
        prediction = projected(values)
        if not np.isfinite(prediction).all():
            return 1e9
        return float(np.sqrt(np.mean(np.square(prediction - observed))))

    best = (loss(params), params.copy())
    steps = np.array([0.01, 0.01, 0.01, 0.06, 0.06, 0.06, 1.5])
    while float(np.max(steps)) > 0.0005:
        improved = False
        for index in range(len(params)):
            for sign in (-1.0, 1.0):
                candidate = best[1].copy()
                candidate[index] = np.clip(
                    candidate[index] + sign * steps[index], lower[index], upper[index]
                )
                score = loss(candidate)
                if score < best[0]:
                    best, improved = (score, candidate), True
        if not improved:
            steps *= 0.5
    fitted_values = best[1]
    fitted = {
        **base,
        "eye": [float(v) for v in fitted_values[:3]],
        "quat_wxyz": matrix_to_quat(
            base_rotation @ rotation_xyz(fitted_values[3:6])
        ),
        "focal_mm": float(fitted_values[6]),
    }
    errors = np.linalg.norm(projected(fitted_values) - observed, axis=1)
    return fitted, {
        "status": "ok",
        "samples": len(usable),
        "episodes": len({item["episode"] for item in usable}),
        "median_pixel_error": float(np.median(errors)),
        "p90_pixel_error": float(np.percentile(errors, 90)),
        "rmse_pixel": float(best[0]),
    }


def run_anchor_calibrate(args: argparse.Namespace) -> int:
    anchor_doc = read_json(Path(args.anchors))
    anchors = anchor_doc.get("anchors", [])
    if len(anchors) < args.min_anchors:
        raise SystemExit(
            f"Nur {len(anchors)} Bewegungsanker; mindestens {args.min_anchors} erforderlich. "
            "REPLAY_NUM_EPISODES erhöhen."
        )
    homographies = {}
    for camera_name in HEAD_CAMS:
        try:
            report = homography_report(anchors, camera_name, args.min_anchors)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if report["median_error_m"] > args.max_median_error_m \
                or report["p90_error_m"] > args.max_p90_error_m:
            raise SystemExit(
                f"{camera_name}: Homographiefehler Median/P90 "
                f"{report['median_error_m']:.3f}/{report['p90_error_m']:.3f} m zu groß."
            )
        homographies[camera_name] = report

    left_h = np.asarray(homographies[HEAD_CAMS[0]]["pixel_to_world"])
    right_h = np.asarray(homographies[HEAD_CAMS[1]]["pixel_to_world"])
    both = [a for a in anchors if all(cam in a.get("pixels", {}) for cam in HEAD_CAMS)]
    left_xy = apply_homography(
        left_h, np.asarray([a["pixels"][HEAD_CAMS[0]]["top_uv"] for a in both])
    )
    right_xy = apply_homography(
        right_h, np.asarray([a["pixels"][HEAD_CAMS[1]]["top_uv"] for a in both])
    )
    disagreement = np.linalg.norm(left_xy - right_xy, axis=1)
    median_disagreement = float(np.median(disagreement))
    if median_disagreement > args.max_camera_disagreement_m:
        raise SystemExit(
            f"Kopfkameras widersprechen sich im Median um {median_disagreement:.3f} m."
        )

    edge_m = float(args.cube_edge)
    z_values = [float(a["world_xyz_m"][2]) for a in anchors if "world_xyz_m" in a]
    if not z_values:
        raise SystemExit("Bewegungsanker enthalten keine Fingerkuppenhöhe.")
    z_spread = float(np.percentile(z_values, 90) - np.percentile(z_values, 10))
    if z_spread > 0.04:
        raise SystemExit(
            f"Fingerkuppenhöhen streuen mit {z_spread:.3f} m zu stark; Anker mehrdeutig."
        )
    measured_center_z = float(np.median(z_values))
    center_z = measured_center_z if args.cube_center_z is None else float(args.cube_center_z)
    if not 0.85 <= center_z <= 1.00:
        raise SystemExit(f"Bewegungsanker ergeben unplausibles Würfel-z={center_z:.3f} m.")
    table_top_z = center_z - edge_m / 2.0
    cameras, camera_fit = {}, {}
    for camera_name in HEAD_CAMS:
        cameras[camera_name], camera_fit[camera_name] = optimize_head_camera(
            camera_name, anchors, table_top_z + edge_m
        )
        if camera_fit[camera_name]["median_pixel_error"] > 5.0 \
                or camera_fit[camera_name]["p90_pixel_error"] > 10.0:
            raise SystemExit(
                f"{camera_name}: analytischer Pixel-Fit Median/P90 "
                f"{camera_fit[camera_name]['median_pixel_error']:.1f}/"
                f"{camera_fit[camera_name]['p90_pixel_error']:.1f} px zu groß."
            )
    wrist_fit = {}
    for camera_name in ("cam_left_wrist", "cam_right_wrist"):
        cameras[camera_name], wrist_fit[camera_name] = optimize_wrist_camera(
            camera_name, anchor_doc.get("wrist_observations", []), table_top_z + edge_m
        )
    wrist_ready = all(
        record.get("status") == "ok"
        and record["median_pixel_error"] <= 10.0
        and record["p90_pixel_error"] <= 20.0
        for record in wrist_fit.values()
    )

    payload = {
        "version": 2,
        "method": "trajectory_anchored_homography",
        "source_dataset": str(args.dataset_path),
        "anchor_file": str(args.anchors),
        "episodes": anchor_doc.get("episodes", []),
        "cube_edge_m": edge_m,
        "table_top_z_m": table_top_z,
        "cube_center_z_m": center_z,
        "homographies": homographies,
        "cameras": cameras,
        "camera_fit": camera_fit,
        "wrist_calibration": {
            "status": "ok" if wrist_ready else "warning",
            "dataset_ready": wrist_ready,
            "fits": wrist_fit,
            "note": "Im Videomodus Warnung; im Dataset-Modus verpflichtend.",
        },
        "quality": {
            "num_anchors": len(anchors),
            "head_camera_disagreement_median_m": median_disagreement,
            "head_camera_disagreement_p90_m": float(np.percentile(disagreement, 90)),
            "action_hashes": anchor_doc.get("action_hashes", {}),
            "actual_camera_diagnostics": anchor_doc.get("actual_camera_diagnostics", {}),
            "fingertip_anchor_z_median_m": (
                float(np.median(z_values)) if z_values else None
            ),
            "fingertip_anchor_z_p90_p10_m": z_spread,
            "cube_center_z_source": (
                "fingertip_anchor_median" if args.cube_center_z is None else "cli_override"
            ),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_dir = Path(args.debug_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "calibration_report.json").write_text(
        json.dumps(payload["quality"] | {"camera_fit": camera_fit}, indent=2),
        encoding="utf-8",
    )
    from PIL import Image, ImageDraw

    root = Path(args.dataset_path)
    info = read_info(root)
    frame_cache: dict[tuple[int, str, int], np.ndarray] = {}
    for anchor in anchors:
        for camera_name, pixel in anchor.get("pixels", {}).items():
            frame_index = int(pixel.get("frame", 0))
            key = (int(anchor["episode"]), camera_name, frame_index)
            if key not in frame_cache:
                frames = load_video_frames(
                    video_path(root, info, key[0], camera_name), frame_index + 1
                )
                frame_cache[key] = frames[frame_index]
            image = Image.fromarray(frame_cache[key].copy())
            draw = ImageDraw.Draw(image)
            u, v = pixel["top_uv"]
            draw.ellipse((u - 7, v - 7, u + 7, v + 7), outline="black", width=3)
            draw.text((u + 9, v - 9), f"{anchor['color']} {anchor['world_xy_m']}", fill="black")
            image.save(
                report_dir /
                f"ep{int(anchor['episode']):06d}_{anchor['color']}_{camera_name}.png"
            )
    print(
        f"[replay-calibrate] {len(anchors)} Bewegungsanker; Kameradifferenz "
        f"Median {median_disagreement * 100:.2f} cm."
    )
    print(f"[replay-calibrate] {out}")
    print("[replay-calibrate] fertig.", flush=True)
    return 0


def cube_vertices(center: np.ndarray, edge: float, yaw: float) -> np.ndarray:
    half = edge / 2.0
    local = np.array([
        [sx * half, sy * half, sz * half]
        for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
    ])
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return local @ rotation.T + center


def projected_bbox(camera: PinholeCamera, center: np.ndarray, edge: float,
                   yaw: float) -> np.ndarray:
    uv = camera.project(cube_vertices(center, edge, yaw))
    if not np.all(np.isfinite(uv)):
        return np.full(4, np.nan)
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])


def fit_cube_pose_3d(blobs: dict[str, dict], cameras: dict[str, PinholeCamera],
                     z_initial: float, edge: float) -> tuple[np.ndarray, float, float]:
    """Fit x/y/z/yaw of one known-size cube jointly to both camera bounding boxes."""
    starts = [
        cameras[name].backproject_to_plane(blob["u"], blob["v"], z_initial)
        for name, blob in blobs.items()
    ]
    xyz0 = np.mean(np.asarray(starts), axis=0)

    def loss(x: float, y: float, z: float, yaw: float) -> float:
        if not 0.10 <= x <= 0.90 or not -0.30 <= y <= 0.30 or not 0.85 <= z <= 1.00:
            return float("inf")
        residuals = []
        for name, blob in blobs.items():
            pred = projected_bbox(cameras[name], np.array([x, y, z]), edge, yaw)
            if not np.all(np.isfinite(pred)):
                return float("inf")
            residuals.extend((pred - np.asarray(blob["bbox"], dtype=float)).tolist())
        return float(np.sqrt(np.mean(np.square(residuals))))

    best = (float("inf"), float(xyz0[0]), float(xyz0[1]), float(z_initial), 0.0)
    for yaw in np.linspace(0.0, math.pi / 2.0, 13, endpoint=False):
        candidate = (
            loss(xyz0[0], xyz0[1], z_initial, yaw),
            float(xyz0[0]), float(xyz0[1]), float(z_initial), float(yaw),
        )
        if candidate[0] < best[0]:
            best = candidate

    step_xy, step_z, step_yaw = 0.015, 0.015, math.radians(8.0)
    while max(step_xy, step_z) > 0.00025:
        improved = False
        _, bx, by, bz, byaw = best
        for dx, dy, dz, da in (
            (step_xy, 0, 0, 0), (-step_xy, 0, 0, 0),
            (0, step_xy, 0, 0), (0, -step_xy, 0, 0),
            (0, 0, step_z, 0), (0, 0, -step_z, 0),
            (0, 0, 0, step_yaw), (0, 0, 0, -step_yaw),
        ):
            yaw = (byaw + da) % (math.pi / 2.0)
            candidate = (
                loss(bx + dx, by + dy, bz + dz, yaw),
                bx + dx, by + dy, bz + dz, yaw,
            )
            if candidate[0] < best[0]:
                best, improved = candidate, True
        if not improved:
            step_xy *= 0.5
            step_z *= 0.5
            step_yaw *= 0.5
    fit_error, x, y, z, yaw = best
    return np.array([x, y, z]), yaw, fit_error


def fit_cube_pose(blobs: dict[str, dict], cameras: dict[str, PinholeCamera],
                  z_center: float, edge: float) -> tuple[np.ndarray, float, float]:
    starts = []
    for cam_name, blob in blobs.items():
        starts.append(cameras[cam_name].backproject_to_plane(blob["u"], blob["v"], z_center))
    xy0 = np.mean(np.asarray(starts)[:, :2], axis=0)

    def loss(x: float, y: float, yaw: float) -> float:
        center = np.array([x, y, z_center])
        residuals = []
        for cam_name, blob in blobs.items():
            pred = projected_bbox(cameras[cam_name], center, edge, yaw)
            obs = np.asarray(blob["bbox"], dtype=float)
            if not np.all(np.isfinite(pred)):
                return float("inf")
            residuals.extend((pred - obs).tolist())
        return float(np.sqrt(np.mean(np.square(residuals))))

    best = (float("inf"), float(xy0[0]), float(xy0[1]), 0.0)
    for yaw in np.linspace(0.0, math.pi / 2.0, 13, endpoint=False):
        candidate = (loss(xy0[0], xy0[1], yaw), float(xy0[0]), float(xy0[1]), float(yaw))
        if candidate[0] < best[0]:
            best = candidate

    step_xy, step_yaw = 0.015, math.radians(8.0)
    while step_xy > 0.00025:
        improved = False
        _, bx, by, byaw = best
        for dx, dy, da in (
            (step_xy, 0, 0), (-step_xy, 0, 0), (0, step_xy, 0), (0, -step_xy, 0),
            (0, 0, step_yaw), (0, 0, -step_yaw),
        ):
            yaw = (byaw + da) % (math.pi / 2.0)
            candidate = (loss(bx + dx, by + dy, yaw), bx + dx, by + dy, yaw)
            if candidate[0] < best[0]:
                best, improved = candidate, True
        if not improved:
            step_xy *= 0.5
            step_yaw *= 0.5
    fit_error, x, y, yaw = best
    return np.array([x, y, z_center]), yaw, fit_error


def save_pose_overlay(rgb: np.ndarray, found: list, camera: PinholeCamera,
                      blocks: list[dict[str, Any]], path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for blob in found:
        if blob is not None:
            draw.rectangle(tuple(blob["bbox"]), outline="white", width=2)
    for block in blocks:
        if block.get("status") != "ok":
            continue
        bbox = projected_bbox(
            camera, np.asarray(block["position_m"]), float(block["edge_m"]),
            float(block["yaw_rad"]),
        )
        if np.all(np.isfinite(bbox)):
            draw.rectangle(tuple(float(v) for v in bbox), outline="black", width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_homography_overlay(rgb: np.ndarray, blocks: list[dict[str, Any]], camera: str,
                            matrix: np.ndarray, path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    inverse = np.linalg.inv(matrix)
    colors = {"rot": "red", "gruen": "green", "gelb": "yellow"}
    for block in blocks:
        estimate = block.get("camera_estimates", {}).get(camera)
        if estimate:
            u, v = estimate["top_uv"]
            draw.ellipse((u - 6, v - 6, u + 6, v + 6), outline=colors[block["color"]], width=3)
        if block.get("status") == "ok":
            xy = np.asarray([block["position_m"][:2]], dtype=float)
            uv = apply_homography(inverse, xy)[0]
            draw.line((uv[0] - 8, uv[1], uv[0] + 8, uv[1]), fill="black", width=3)
            draw.line((uv[0], uv[1] - 8, uv[0], uv[1] + 8), fill="black", width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def run_homography_poses(args: argparse.Namespace, root: Path, info: dict[str, Any],
                          episodes: list[int], calibration: dict[str, Any]) -> int:
    matrices = {
        camera: np.asarray(calibration["homographies"][camera]["pixel_to_world"], dtype=float)
        for camera in HEAD_CAMS
    }
    anchor_doc = read_json(Path(calibration["anchor_file"]))
    anchor_map = {
        (int(anchor["episode"]), anchor["color"]): anchor for anchor in anchor_doc["anchors"]
    }
    edge = float(calibration["cube_edge_m"])
    center_z = float(calibration["cube_center_z_m"])
    out_path = Path(args.out)
    result = {
        "version": 2,
        "method": "trajectory_anchored_homography",
        "source_dataset": str(root),
        "calibration": str(args.calibration),
        "cube_edge_m": edge,
        "table_top_z_m": float(calibration["table_top_z_m"]),
        "episodes": {},
    }
    if out_path.exists() and not args.overwrite:
        old = read_json(out_path)
        if old.get("calibration") == str(args.calibration) and old.get("version") == 2:
            result["episodes"].update(old.get("episodes", {}))

    for ordinal, episode in enumerate(episodes, 1):
        if str(episode) in result["episodes"] and not args.overwrite:
            print(f"[replay-poses] ({ordinal}/{len(episodes)}) Episode {episode}: vorhanden.")
            continue
        frames_by_camera: dict[str, np.ndarray] = {}
        detections: dict[str, dict[str, dict]] = {}
        for camera in HEAD_CAMS:
            frames = load_video_frames(video_path(root, info, episode, camera), args.max_frames)
            detections[camera] = {}
            for color in CUBE_COLORS:
                frame_index, rgb, blob = best_top_face_detection(
                    frames, color, min_area=max(60, args.min_area // 2)
                )
                detections[camera][color] = blob
                if camera not in frames_by_camera or frame_index == 0:
                    frames_by_camera[camera] = rgb
        wrist_detections: dict[str, dict[str, dict]] = {}
        for camera in ("cam_left_wrist", "cam_right_wrist"):
            frames = load_video_frames(video_path(root, info, episode, camera), args.max_frames)
            wrist_detections[camera] = {}
            for color in CUBE_COLORS:
                frame_index, _, blob = best_top_face_detection(
                    frames, color, min_area=max(40, args.min_area // 3)
                )
                wrist_detections[camera][color] = (frame_index, blob)

        blocks = []
        for color in CUBE_COLORS:
            estimates = {}
            yaw_values = []
            for camera in HEAD_CAMS:
                blob = detections[camera][color]
                if blob is None:
                    continue
                uv = np.asarray([[blob["u"], blob["v"]]], dtype=float)
                xy = apply_homography(matrices[camera], uv)[0]
                x0, y0, x1, y1 = blob["bbox"]
                axis_uv = np.asarray(blob.get(
                    "axis_uv", [[x0, (y0 + y1) / 2], [x1, (y0 + y1) / 2]]
                ))
                axis_xy = apply_homography(matrices[camera], axis_uv)
                delta = axis_xy[1] - axis_xy[0]
                yaw_values.append(float(math.atan2(delta[1], delta[0]) % (math.pi / 2.0)))
                estimates[camera] = {
                    "top_uv": [float(blob["u"]), float(blob["v"])],
                    "world_xy_m": [float(xy[0]), float(xy[1])],
                    "bbox": [int(v) for v in blob["bbox"]],
                    "detection_source": blob.get("source", "full_blob"),
                }
            block = {"name": COLOR_TO_BLOCK[color], "color": color,
                     "camera_estimates": estimates}
            block["wrist_observations"] = {
                camera: {
                    "top_uv": [float(blob["u"]), float(blob["v"])],
                    "bbox": [int(v) for v in blob["bbox"]],
                    "frame": int(frame_index),
                }
                for camera, per_color in wrist_detections.items()
                if (frame_index := per_color[color][0]) >= 0
                and (blob := per_color[color][1]) is not None
            }
            if len(estimates) != len(HEAD_CAMS):
                block.update(status="missing", reason="nicht in beiden Kopfkameras erkannt")
                blocks.append(block)
                continue
            xy_values = np.asarray([estimates[c]["world_xy_m"] for c in HEAD_CAMS])
            camera_delta = float(np.linalg.norm(xy_values[0] - xy_values[1]))
            image_xy = np.median(xy_values, axis=0)
            anchor = anchor_map.get((episode, color))
            source = "head_homography"
            if anchor is not None:
                anchor_xy = np.asarray(anchor["world_xy_m"], dtype=float)
                anchor_delta = float(np.linalg.norm(image_xy - anchor_xy))
                if anchor_delta > args.max_anchor_disagreement_m:
                    block.update(
                        status="rejected",
                        reason=f"Bild/Griffanker widersprechen sich um {anchor_delta:.3f} m",
                        image_anchor_disagreement_m=anchor_delta,
                    )
                    blocks.append(block)
                    continue
                image_xy, source = anchor_xy, "trajectory_anchor"
                block["trajectory_anchor"] = anchor
                block["image_anchor_disagreement_m"] = anchor_delta
            if camera_delta > args.max_camera_disagreement_m:
                block.update(
                    status="rejected",
                    reason=f"Kopfkameras widersprechen sich um {camera_delta:.3f} m",
                )
                blocks.append(block)
                continue
            inside = 0.10 <= image_xy[0] <= 0.90 and -0.35 <= image_xy[1] <= 0.35
            yaw = float(np.median(yaw_values)) if yaw_values else 0.0
            block.update({
                "status": "ok" if inside else "rejected",
                "reason": "" if inside else "Position außerhalb des Tischbereichs",
                "source": source,
                "position_m": [float(image_xy[0]), float(image_xy[1]), center_z],
                "yaw_rad": yaw,
                "orientation_wxyz": [math.cos(yaw / 2.0), 0.0, 0.0,
                                      math.sin(yaw / 2.0)],
                "edge_m": edge,
                "camera_disagreement_m": camera_delta,
            })
            blocks.append(block)

        status = "ok" if len(blocks) == 3 and all(b["status"] == "ok" for b in blocks) \
            else "skipped"
        result["episodes"][str(episode)] = {"status": status, "blocks": blocks}
        for camera in HEAD_CAMS:
            save_homography_overlay(
                frames_by_camera[camera], blocks, camera, matrices[camera],
                Path(args.debug_dir) / f"ep{episode:06d}_{camera}.png",
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[replay-poses] ({ordinal}/{len(episodes)}) Episode {episode}: {status}.")
    print(f"[replay-poses] {out_path}")
    print("[replay-poses] fertig.", flush=True)
    return 0


def run_poses(args: argparse.Namespace) -> int:
    root = Path(args.dataset_path)
    info = read_info(root)
    episodes = select_episodes(info, args)
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    if calibration.get("method") == "trajectory_anchored_homography":
        return run_homography_poses(args, root, info, episodes, calibration)
    cameras = {
        name: camera_from_record(calibration["cameras"][name]) for name in HEAD_CAMS
    }
    edge = float(calibration["cube_edge_m"])
    z_center = float(calibration["cube_center_z_m"])
    out_path = Path(args.out)
    result = {
        "version": 1,
        "source_dataset": str(root),
        "calibration": str(args.calibration),
        "cube_edge_m": edge,
        "table_top_z_m": float(calibration["table_top_z_m"]),
        "episodes": {},
    }
    if out_path.exists() and not args.overwrite:
        old = json.loads(out_path.read_text(encoding="utf-8"))
        result["episodes"].update(old.get("episodes", {}))

    for ordinal, ep in enumerate(episodes, 1):
        if str(ep) in result["episodes"] and not args.overwrite:
            print(f"[replay-poses] ({ordinal}/{len(episodes)}) Episode {ep}: vorhanden.")
            continue
        per_camera: dict[str, dict[str, Any]] = {}
        debug_frames: dict[str, tuple[np.ndarray, list]] = {}
        for cam_name in HEAD_CAMS:
            try:
                frames = load_video_frames(video_path(root, info, ep, cam_name), args.max_frames)
                frame_idx, rgb, found = best_detection(frames, args.min_area)
            except (FileNotFoundError, RuntimeError) as exc:
                print(f"[replay-poses] Episode {ep}, {cam_name}: {exc}")
                continue
            per_camera[cam_name] = {"frame": frame_idx, "found": found}
            debug_frames[cam_name] = (rgb, found)

        blocks: list[dict[str, Any]] = []
        for index, color in enumerate(CUBE_COLORS):
            blobs = {
                cam: rec["found"][index] for cam, rec in per_camera.items()
                if rec["found"][index] is not None
            }
            if len(blobs) != len(HEAD_CAMS):
                blocks.append({
                    "name": COLOR_TO_BLOCK[color], "color": color, "status": "missing",
                    "reason": f"nur in {len(blobs)}/2 Kopfkameras gefunden",
                })
                continue
            center, yaw, fit_error = fit_cube_pose(blobs, cameras, z_center, edge)
            inside = 0.10 <= center[0] <= 0.90 and -0.30 <= center[1] <= 0.30
            status = "ok" if inside and fit_error <= args.max_fit_error_px else "rejected"
            blocks.append({
                "name": COLOR_TO_BLOCK[color],
                "color": color,
                "status": status,
                "position_m": [round(float(v), 5) for v in center],
                "yaw_rad": round(float(yaw), 6),
                "orientation_wxyz": [round(math.cos(yaw / 2.0), 7), 0.0, 0.0,
                                      round(math.sin(yaw / 2.0), 7)],
                "edge_m": edge,
                "fit_error_px": round(float(fit_error), 3),
                "frames": {cam: int(per_camera[cam]["frame"]) for cam in blobs},
            })

        status = "ok" if len(blocks) == 3 and all(b["status"] == "ok" for b in blocks) \
            else "skipped"
        result["episodes"][str(ep)] = {"status": status, "blocks": blocks}
        for cam_name, (rgb, found) in debug_frames.items():
            save_pose_overlay(
                rgb, found, cameras[cam_name], blocks,
                Path(args.debug_dir) / f"ep{ep:06d}_{cam_name}.png",
            )
        print(f"[replay-poses] ({ordinal}/{len(episodes)}) Episode {ep}: {status} "
              f"({sum(b['status'] == 'ok' for b in blocks)}/3 Würfel).", flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"[replay-poses] {out_path}")
    print("[replay-poses] fertig.", flush=True)
    return 0


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--episode-ids", type=int, nargs="*", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="v2.1-Datensatz und Beispielpfade prüfen")
    inspect.add_argument("--dataset-path", required=True)
    add_selection_args(inspect)

    calibrate = sub.add_parser("calibrate", help="Kameraskala, Tisch und Würfelgröße prüfen")
    calibrate.add_argument("--dataset-path", required=True)
    calibrate.add_argument("--out", required=True)
    calibrate.add_argument("--debug-dir", required=True)
    calibrate.add_argument("--anchors", default="")
    calibrate.add_argument("--max-frames", type=int, default=30)
    calibrate.add_argument("--min-area", type=int, default=120)
    calibrate.add_argument("--cube-edge", type=float, default=0.05)
    calibrate.add_argument("--cube-center-z", type=float, default=None)
    calibrate.add_argument("--edge-min", type=float, default=0.04)
    calibrate.add_argument("--edge-max", type=float, default=0.065)
    calibrate.add_argument("--max-fit-error-px", type=float, default=25.0)
    calibrate.add_argument("--min-anchors", type=int, default=8)
    calibrate.add_argument("--max-median-error-m", type=float, default=0.015)
    calibrate.add_argument("--max-p90-error-m", type=float, default=0.03)
    calibrate.add_argument("--max-camera-disagreement-m", type=float, default=0.02)
    add_selection_args(calibrate)

    poses = sub.add_parser("poses", help="Würfelpose je Episode aus zwei Kopfkameras fitten")
    poses.add_argument("--dataset-path", required=True)
    poses.add_argument("--calibration", required=True)
    poses.add_argument("--out", required=True)
    poses.add_argument("--debug-dir", required=True)
    poses.add_argument("--max-frames", type=int, default=30)
    poses.add_argument("--min-area", type=int, default=120)
    poses.add_argument("--max-fit-error-px", type=float, default=20.0)
    poses.add_argument("--max-camera-disagreement-m", type=float, default=0.03)
    poses.add_argument("--max-anchor-disagreement-m", type=float, default=0.03)
    poses.add_argument("--overwrite", action="store_true")
    add_selection_args(poses)

    args = parser.parse_args()
    if args.num_episodes < 1:
        raise SystemExit("--num-episodes muss mindestens 1 sein.")
    return {"inspect": run_inspect, "calibrate": run_calibrate, "poses": run_poses}[
        args.command
    ](args)


if __name__ == "__main__":
    raise SystemExit(main())
