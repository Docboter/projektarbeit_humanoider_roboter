#!/usr/bin/env python3
"""Reconstruct initial cube poses for physical dataset replay from real RGB frames.

Only the first frames of the two fixed head cameras are inspected. Cube tracking and
trajectory-derived calibration anchors are deliberately not part of this tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_geometry import CAMERA_CFG, PinholeCamera  # noqa: E402
from extract_block_layout import CUBE_COLORS  # noqa: E402
from replay_calibration import best_top_face_detection  # noqa: E402

HEAD_CAMS = ("cam_left_high", "cam_right_high")
POLICY_CAMS = HEAD_CAMS + ("cam_left_wrist", "cam_right_wrist")
COLOR_TO_BLOCK = {"rot": "block_0", "gruen": "block_1", "gelb": "block_2"}
DEFAULT_VIDEO_TEMPLATE = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)
DEFAULT_DATA_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_CUBE_EDGE_M = 0.05
DEFAULT_CUBE_CENTER_Z_M = 0.915


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
            "Testepisoden werden nicht für spätere Trainingsdaten verarbeitet."
        )
    return episodes


def format_dataset_path(
    root: Path,
    template: str,
    info: dict[str, Any],
    episode: int,
    video_key: str | None = None,
) -> Path:
    values = {
        "episode_chunk": episode // int(info.get("chunks_size", 1000)),
        "episode_index": episode,
        "video_key": video_key,
    }
    try:
        return root / template.format(**values)
    except KeyError as exc:
        raise SystemExit(
            "Der Datensatz ist noch LeRobot v3.0. Zuerst `server_rl_run.sh "
            "replay-prepare` ausführen (v3→v2.1)."
        ) from exc


def video_path(root: Path, info: dict[str, Any], episode: int, camera: str) -> Path:
    template = info.get("video_path") or DEFAULT_VIDEO_TEMPLATE
    return format_dataset_path(
        root, template, info, episode, f"observation.images.{camera}"
    )


def data_path(root: Path, info: dict[str, Any], episode: int) -> Path:
    return format_dataset_path(
        root, info.get("data_path", DEFAULT_DATA_TEMPLATE), info, episode
    )


def load_video_frames(path: Path, limit: int) -> list[np.ndarray]:
    import imageio.v2 as imageio

    if not path.is_file():
        raise FileNotFoundError(path)
    frames = []
    with imageio.get_reader(str(path), format="FFMPEG") as reader:
        for index, frame in enumerate(reader):
            if index >= limit:
                break
            frames.append(np.asarray(frame, dtype=np.uint8)[..., :3])
    if not frames:
        raise RuntimeError(f"Video enthält keine Frames: {path}")
    return frames


def camera_record(name: str) -> dict[str, Any]:
    pose = getattr(CAMERA_CFG, name)
    return {
        "eye": [float(value) for value in pose["pos"]],
        "quat_wxyz": [float(value) for value in pose["rot"]],
        "focal_mm": float(CAMERA_CFG.focal_high),
        "aperture_mm": float(CAMERA_CFG.horizontal_aperture_mm),
        "width": int(CAMERA_CFG.width),
        "height": int(CAMERA_CFG.height),
    }


def camera_from_record(record: dict[str, Any]) -> PinholeCamera:
    return PinholeCamera(
        record["eye"],
        record["quat_wxyz"],
        record["focal_mm"],
        record["aperture_mm"],
        record["width"],
        record["height"],
    )


def save_overlay(
    rgb: np.ndarray,
    detections: dict[str, dict | None],
    estimates: dict[str, list[float]],
    camera: PinholeCamera,
    z_top: float,
    path: Path,
) -> None:
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    display = {"rot": "red", "gruen": "green", "gelb": "yellow"}
    for color, blob in detections.items():
        if blob is None:
            continue
        x0, y0, x1, y1 = blob["bbox"]
        draw.rectangle((x0, y0, x1, y1), outline=display[color], width=3)
        draw.ellipse(
            (blob["u"] - 5, blob["v"] - 5, blob["u"] + 5, blob["v"] + 5),
            outline="white",
            width=2,
        )
        if color in estimates:
            point = np.asarray([[*estimates[color], z_top]], dtype=float)
            u, v = camera.project(point)[0]
            draw.line((u - 8, v, u + 8, v), fill="black", width=3)
            draw.line((u, v - 8, u, v + 8), fill="black", width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def run_inspect(args: argparse.Namespace) -> int:
    root = Path(args.dataset_path)
    info = read_info(root)
    errors = []
    if info.get("codebase_version") != "v2.1":
        errors.append(f"codebase_version={info.get('codebase_version')!r}, erwartet 'v2.1'")
    if abs(float(info.get("fps", 0.0)) - 30.0) > 1e-6:
        errors.append(f"fps={info.get('fps')}, erwartet 30")
    for key in ("observation.state", "action"):
        shape = info.get("features", {}).get(key, {}).get("shape")
        if shape != [28]:
            errors.append(f"{key}.shape={shape}, erwartet [28]")
    for camera in POLICY_CAMS:
        shape = info.get("features", {}).get(
            f"observation.images.{camera}", {}
        ).get("shape")
        if shape != [3, 480, 640]:
            errors.append(f"{camera}.shape={shape}, erwartet [3, 480, 640]")
    episodes = select_episodes(info, args)
    for episode in episodes[:1]:
        if not data_path(root, info, episode).is_file():
            errors.append(f"Episode-Parquet fehlt: {data_path(root, info, episode)}")
        for camera in POLICY_CAMS:
            path = video_path(root, info, episode, camera)
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
    root = Path(args.dataset_path)
    info = read_info(root)
    episodes = select_episodes(info, args)
    records = {camera: camera_record(camera) for camera in HEAD_CAMS}
    samples: list[dict[str, Any]] = []
    edge_by_camera: dict[str, list[float]] = {camera: [] for camera in HEAD_CAMS}
    z_top = float(args.cube_center_z) + float(args.cube_edge) / 2.0

    for episode in episodes:
        for camera_name in HEAD_CAMS:
            frames = load_video_frames(
                video_path(root, info, episode, camera_name), args.max_frames
            )
            camera = camera_from_record(records[camera_name])
            overlay_detections = {}
            overlay_rgb = frames[0]
            for color in CUBE_COLORS:
                frame_index, rgb, blob = best_top_face_detection(
                    frames, color, min_area=args.min_area
                )
                overlay_rgb = rgb
                overlay_detections[color] = blob
                if blob is None:
                    continue
                point = camera.backproject_to_plane(float(blob["u"]), float(blob["v"]), z_top)
                distance = float(np.linalg.norm(point - camera.eye))
                x0, y0, x1, y1 = blob["bbox"]
                extent_px = float(max(x1 - x0 + 1, y1 - y0 + 1))
                edge_m = extent_px * distance / camera.f_px
                edge_by_camera[camera_name].append(edge_m)
                samples.append(
                    {
                        "episode": episode,
                        "camera": camera_name,
                        "color": color,
                        "frame": frame_index,
                        "top_uv": [float(blob["u"]), float(blob["v"])],
                        "extent_px": extent_px,
                        "observed_edge_m": edge_m,
                        "source": blob.get("source", "full_blob"),
                    }
                )
            save_overlay(
                overlay_rgb,
                overlay_detections,
                {},
                camera,
                z_top,
                Path(args.debug_dir) / f"ep{episode:06d}_{camera_name}.png",
            )

    if len(samples) < 6 or any(not edge_by_camera[camera] for camera in HEAD_CAMS):
        raise SystemExit(
            f"Nur {len(samples)} gültige Würfeloberseiten; mindestens sechs und beide "
            "Kopfkameras erforderlich."
        )
    all_edges = np.asarray([sample["observed_edge_m"] for sample in samples])
    raw_edge = float(np.median(all_edges))
    if not args.edge_min <= raw_edge <= args.edge_max:
        raise SystemExit(
            f"Bildbasierte Würfelkante {raw_edge * 100:.2f} cm außerhalb "
            f"{args.edge_min * 100:.1f}–{args.edge_max * 100:.1f} cm."
        )
    for camera_name in HEAD_CAMS:
        focal_scale = float(np.median(edge_by_camera[camera_name])) / args.cube_edge
        if not 0.8 <= focal_scale <= 1.3:
            raise SystemExit(
                f"FOV-Korrektur für {camera_name} unplausibel: {focal_scale:.3f}."
            )
        records[camera_name]["focal_mm"] *= focal_scale
        records[camera_name]["focal_scale"] = focal_scale

    payload = {
        "version": 2,
        "method": "cv_cube_scale",
        "source_dataset": str(root),
        "episodes": episodes,
        "cameras": records,
        "cube_edge_m": float(args.cube_edge),
        "cube_center_z_m": float(args.cube_center_z),
        "table_top_z_m": float(args.cube_center_z - args.cube_edge / 2.0),
        "quality": {
            "num_samples": len(samples),
            "observed_edge_median_m": raw_edge,
            "observed_edge_p10_m": float(np.percentile(all_edges, 10)),
            "observed_edge_p90_m": float(np.percentile(all_edges, 90)),
        },
        "samples": samples,
        "notes": [
            "Nur frühe RGB-Frames und die bekannte 5-cm-Würfelkante werden verwendet.",
            "Würfelmittelpunkt und Tischhöhe werden nicht trianguliert.",
            "Greifstützen werden erst episodenspezifisch in replay-poses ergänzt.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[replay-calibrate] {len(samples)} Oberseiten; Bildkante Median "
        f"{raw_edge * 100:.2f} cm; verwendet 5.00 cm."
    )
    print(f"[replay-calibrate] {out}")
    print("[replay-calibrate] fertig.", flush=True)
    return 0


def run_poses(args: argparse.Namespace) -> int:
    root = Path(args.dataset_path)
    info = read_info(root)
    episodes = select_episodes(info, args)
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    cameras = {
        name: camera_from_record(calibration["cameras"][name]) for name in HEAD_CAMS
    }
    edge = float(calibration["cube_edge_m"])
    z_center = float(calibration["cube_center_z_m"])
    z_top = z_center + edge / 2.0
    out_path = Path(args.out)
    result = {
        "version": 3,
        "method": "cv_top_face_stereo",
        "source_dataset": str(root),
        "calibration": str(args.calibration),
        "cube_edge_m": edge,
        "table_top_z_m": float(calibration["table_top_z_m"]),
        "episodes": {},
    }
    if out_path.exists() and not args.overwrite:
        old = json.loads(out_path.read_text(encoding="utf-8"))
        if old.get("version") == 3 and old.get("calibration") == str(args.calibration):
            result["episodes"].update(old.get("episodes", {}))

    for ordinal, episode in enumerate(episodes, 1):
        if str(episode) in result["episodes"] and not args.overwrite:
            print(f"[replay-poses] ({ordinal}/{len(episodes)}) Episode {episode}: vorhanden.")
            continue
        detections: dict[str, dict[str, dict | None]] = {}
        debug_frames = {}
        for camera_name in HEAD_CAMS:
            frames = load_video_frames(
                video_path(root, info, episode, camera_name), args.max_frames
            )
            detections[camera_name] = {}
            debug_frames[camera_name] = frames[0]
            for color in CUBE_COLORS:
                _, rgb, blob = best_top_face_detection(
                    frames, color, min_area=args.min_area
                )
                debug_frames[camera_name] = rgb
                detections[camera_name][color] = blob

        blocks = []
        overlay_estimates = {camera: {} for camera in HEAD_CAMS}
        for color in CUBE_COLORS:
            estimates = {}
            for camera_name in HEAD_CAMS:
                blob = detections[camera_name][color]
                if blob is None:
                    continue
                point = cameras[camera_name].backproject_to_plane(
                    float(blob["u"]), float(blob["v"]), z_top
                )
                xy = [float(point[0]), float(point[1])]
                estimates[camera_name] = {
                    "top_uv": [float(blob["u"]), float(blob["v"])],
                    "world_xy_m": xy,
                    "bbox": [int(value) for value in blob["bbox"]],
                    "source": blob.get("source", "full_blob"),
                }
                overlay_estimates[camera_name][color] = xy
            block = {
                "name": COLOR_TO_BLOCK[color],
                "color": color,
                "camera_estimates": estimates,
            }
            if len(estimates) != 2:
                block.update(
                    status="missing",
                    reason=f"nur in {len(estimates)}/2 Kopfkameras erkannt",
                )
                blocks.append(block)
                continue
            points = np.asarray(
                [estimates[camera]["world_xy_m"] for camera in HEAD_CAMS], dtype=float
            )
            disagreement = float(np.linalg.norm(points[0] - points[1]))
            xy = np.median(points, axis=0)
            inside = 0.25 <= xy[0] <= 0.45 and -0.25 <= xy[1] <= 0.25
            accepted = disagreement <= args.max_camera_disagreement_m and inside
            reason = ""
            if disagreement > args.max_camera_disagreement_m:
                reason = f"Kopfkameras widersprechen sich um {disagreement:.3f} m"
            elif not inside:
                reason = "Position außerhalb des Würfel-Arbeitsbereichs"
            position = [float(xy[0]), float(xy[1]), z_center]
            block.update(
                {
                    "status": "ok" if accepted else "rejected",
                    "reason": reason,
                    "cv_position_m": position,
                    "position_m": position.copy(),
                    "position_source": "cv",
                    "grasp_support": {
                        "accepted": False,
                        "reason": "noch nicht ausgewertet",
                    },
                    "yaw_rad": 0.0,
                    "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "edge_m": edge,
                    "camera_disagreement_m": disagreement,
                }
            )
            blocks.append(block)
        status = "ok" if len(blocks) == 3 and all(
            block["status"] == "ok" for block in blocks
        ) else "skipped"
        result["episodes"][str(episode)] = {
            "status": status,
            "blocks": blocks,
            "grasp_support": {"enabled": False, "accepted": 0, "fk_states": 0},
        }
        for camera_name in HEAD_CAMS:
            save_overlay(
                debug_frames[camera_name],
                detections[camera_name],
                overlay_estimates[camera_name],
                cameras[camera_name],
                z_top,
                Path(args.debug_dir) / f"ep{episode:06d}_{camera_name}.png",
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        cv_count = sum(block["status"] == "ok" for block in blocks)
        print(
            f"[replay-poses] ({ordinal}/{len(episodes)}) Episode {episode}: "
            f"{status}; CV {cv_count}/3.",
            flush=True,
        )
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
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--dataset-path", required=True)
    add_selection_args(inspect)

    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--dataset-path", required=True)
    calibrate.add_argument("--out", required=True)
    calibrate.add_argument("--debug-dir", required=True)
    calibrate.add_argument("--max-frames", type=int, default=30)
    calibrate.add_argument("--min-area", type=int, default=80)
    calibrate.add_argument("--cube-edge", type=float, default=DEFAULT_CUBE_EDGE_M)
    calibrate.add_argument(
        "--cube-center-z", type=float, default=DEFAULT_CUBE_CENTER_Z_M
    )
    calibrate.add_argument("--edge-min", type=float, default=0.04)
    calibrate.add_argument("--edge-max", type=float, default=0.065)
    add_selection_args(calibrate)

    poses = sub.add_parser("poses")
    poses.add_argument("--dataset-path", required=True)
    poses.add_argument("--calibration", required=True)
    poses.add_argument("--out", required=True)
    poses.add_argument("--debug-dir", required=True)
    poses.add_argument("--max-frames", type=int, default=30)
    poses.add_argument("--min-area", type=int, default=80)
    poses.add_argument("--max-camera-disagreement-m", type=float, default=0.03)
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
