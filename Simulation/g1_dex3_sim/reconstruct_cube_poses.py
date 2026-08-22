#!/usr/bin/env python3
"""Reconstruct initial cube poses for physical dataset replay from real RGB frames.

Calibration uses sparse pick anchors; pose reconstruction uses only stationary real frames.
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
from replay_calibration import (  # noqa: E402
    apply_homography,
    build_calibration,
    combine_camera_estimates,
    episode_diagnostic_pixels,
    file_sha256,
    find_motion_onset,
    pose_resume_matches,
    require_calibration_dataset,
    require_pick_homography_calibration,
    stable_top_face_measurement,
    track_colors,
    validate_anchor_document,
)

HEAD_CAMS = ("cam_left_high", "cam_right_high")
POLICY_CAMS = HEAD_CAMS + ("cam_left_wrist", "cam_right_wrist")
COLOR_TO_BLOCK = {"rot": "block_0", "gruen": "block_1", "gelb": "block_2"}
DEFAULT_VIDEO_TEMPLATE = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)
DEFAULT_DATA_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_CUBE_EDGE_M = 0.05
DEFAULT_CUBE_CENTER_Z_M = 0.895


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


def save_measurement_overlay(
    rgb: np.ndarray, measurement: dict[str, Any], world_xy: list[float], path: Path
) -> None:
    """Write one overlay on the exact frame from which its measurement came."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = measurement["bbox"]
    draw.rectangle((x0, y0, x1, y1), outline="white", width=3)
    u, v = measurement["top_uv"]
    draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline="black", width=3)
    draw.text((u + 7, v + 7), f"x={world_xy[0]:.3f} y={world_xy[1]:.3f}", fill="white")
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
    anchor_document = json.loads(Path(args.anchors).read_text(encoding="utf-8"))
    try:
        validate_anchor_document(anchor_document, root, episodes)
    except ValueError as exc:
        raise SystemExit(f"{exc} `replay-calibrate` sammelt die Anker neu.") from exc
    anchors = anchor_document.get("anchors", [])
    fit = build_calibration(anchors, episodes, holdout_ratio=args.holdout_ratio, seed=args.seed)
    raw_samples = []
    z_top = args.cube_center_z + args.cube_edge / 2.0
    episode_diagnostics = anchor_document.get("episode_diagnostics", {})
    for pixel in episode_diagnostic_pixels(episode_diagnostics):
        camera = camera_from_record(camera_record(pixel["camera"]))
        u, v = pixel["top_uv"]
        point = camera.backproject_to_plane(float(u), float(v), z_top)
        distance = float(np.linalg.norm(point - camera.eye))
        observed = float(pixel.get("extent_px", 0.0)) * distance / camera.f_px
        raw_samples.append({
            **pixel,
            "observed_edge_m": observed if np.isfinite(observed) else None,
        })
    observed_edges = np.asarray([
        sample["observed_edge_m"]
        for sample in raw_samples
        if sample["observed_edge_m"] is not None
    ])
    payload = {
        "version": 4,
        "method": "pick_anchored_homography",
        "valid": bool(fit["valid"]),
        "source_dataset": str(root.resolve()),
        "episodes": episodes,
        "fit_episodes": fit["fit_episodes"],
        "holdout_episodes": fit["holdout_episodes"],
        "cameras": {
            name: {**camera_record(name), **record}
            for name, record in fit["cameras"].items()
        },
        "cube_edge_m": float(args.cube_edge),
        "cube_center_z_m": float(args.cube_center_z),
        "table_top_z_m": float(args.cube_center_z - args.cube_edge / 2.0),
        "quality": {
            **fit["quality"],
            "failures": fit["failures"],
            "diagnostic_observed_edge_median_m": (
                float(np.median(observed_edges)) if len(observed_edges) else None
            ),
            "diagnostic_observed_edge_p10_m": (
                float(np.percentile(observed_edges, 10)) if len(observed_edges) else None
            ),
            "diagnostic_observed_edge_p90_m": (
                float(np.percentile(observed_edges, 90)) if len(observed_edges) else None
            ),
        },
        "samples": raw_samples,
        "anchors": anchors,
        "action_hashes": anchor_document.get("action_hashes", {}),
        "episode_diagnostics": episode_diagnostics,
        "notes": ["Die AABB-Kantenschätzung ist nur Diagnose und verändert keine Intrinsics."],
    }
    report = Path(args.debug_dir) / "calibration_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    out = Path(args.out)
    if fit["valid"]:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    elif out.exists():
        out.unlink()
    print(f"[replay-calibrate] Diagnosebericht: {report}")
    if not fit["valid"]:
        for failure in fit["failures"]:
            print(f"[replay-calibrate] QUALITY-GATE: {failure}")
        return 1
    print(f"[replay-calibrate] {len(anchors)} Anker; {out}")
    print("[replay-calibrate] fertig.", flush=True)
    return 0


def run_poses(args: argparse.Namespace) -> int:
    root = Path(args.dataset_path)
    info = read_info(root)
    episodes = select_episodes(info, args)
    calibration_path = Path(args.calibration)
    calibration_sha256 = file_sha256(calibration_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    try:
        require_pick_homography_calibration(calibration)
        require_calibration_dataset(calibration, root)
    except ValueError as exc:
        raise SystemExit(f"{exc} `replay-calibrate` erneut ausführen.") from exc
    matrices = {name: np.asarray(calibration["cameras"][name]["pixel_to_table_xy"], dtype=float)
                for name in HEAD_CAMS}
    edge = float(calibration["cube_edge_m"])
    z_center = float(calibration["cube_center_z_m"])
    out_path = Path(args.out)
    result = {
        "version": 4,
        "method": "stationary_top_face_homography",
        "source_dataset": str(root.resolve()),
        "calibration": str(args.calibration),
        "calibration_sha256": calibration_sha256,
        "cube_edge_m": edge,
        "table_top_z_m": float(calibration["table_top_z_m"]),
        "episodes": {},
    }
    if out_path.exists() and not args.overwrite:
        old = json.loads(out_path.read_text(encoding="utf-8"))
        if pose_resume_matches(old, root, calibration_sha256):
            result["episodes"].update(old.get("episodes", {}))

    for ordinal, episode in enumerate(episodes, 1):
        if str(episode) in result["episodes"] and not args.overwrite:
            print(f"[replay-poses] ({ordinal}/{len(episodes)}) Episode {episode}: vorhanden.")
            continue
        detections: dict[str, dict[str, dict | None]] = {camera: {} for camera in HEAD_CAMS}
        camera_frames = {}
        for camera_name in HEAD_CAMS:
            path = video_path(root, info, episode, camera_name)
            tracks = track_colors(path, min_area=args.min_area)
            onsets = {color: find_motion_onset(tracks[color]) for color in CUBE_COLORS}
            limit = max(
                [value for value in onsets.values() if value is not None]
                + [args.max_frames]
            )
            frames = load_video_frames(path, limit + 1)
            camera_frames[camera_name] = frames
            for color in CUBE_COLORS:
                stop = (
                    onsets[color]
                    if onsets[color] is not None
                    else min(len(frames), args.max_frames)
                )
                detections[camera_name][color] = stable_top_face_measurement(
                    frames, color, stop, min_area=args.min_area, min_samples=5,
                    allow_full_blob=True,
                )

        blocks = []
        for color in CUBE_COLORS:
            estimates = {}
            for camera_name in HEAD_CAMS:
                measurement = detections[camera_name][color]
                if measurement is None:
                    continue
                xy_array = apply_homography(matrices[camera_name], [measurement["top_uv"]])[0]
                xy = [float(value) for value in xy_array]
                top_face_xy = None
                if measurement["top_face_uv"] is not None:
                    top_face_array = apply_homography(
                        matrices[camera_name], [measurement["top_face_uv"]]
                    )[0]
                    top_face_xy = [float(value) for value in top_face_array]
                estimates[camera_name] = {
                    "top_uv": measurement["top_uv"],
                    "world_xy_m": xy,
                    "bbox": measurement["bbox"],
                    "source": measurement["source"],
                    "frame": measurement["frame"],
                    "frames": measurement["frames"],
                    "num_samples": measurement["num_samples"],
                    "top_face_count": measurement["top_face_count"],
                    "top_face_uv": measurement["top_face_uv"],
                    "top_face_frames": measurement["top_face_frames"],
                    "top_face_world_xy_m": top_face_xy,
                }
                frame_index = measurement["frame"]
                overlay_path = Path(args.debug_dir) / (
                    f"ep{episode:06d}_{camera_name}_{color}_f{frame_index:04d}.png"
                )
                save_measurement_overlay(
                    camera_frames[camera_name][frame_index], measurement, xy, overlay_path
                )
            block = {
                "name": COLOR_TO_BLOCK[color],
                "color": color,
                "camera_estimates": estimates,
            }
            if not estimates:
                block.update(
                    status="missing",
                    reason="in keiner Kopfkamera stabil erkannt",
                )
                blocks.append(block)
                continue
            xy, disagreement, reason = combine_camera_estimates(
                estimates, args.max_camera_disagreement_m
            )
            accepted = xy is not None
            position = ([float(xy[0]), float(xy[1]), z_center]
                        if xy is not None else [None, None, z_center])
            block.update(
                {
                    "status": "ok" if accepted else "rejected",
                    "reason": reason,
                    "position_m": position.copy(),
                    "position_source": "pick_anchored_homography",
                    "confidence": "stereo" if len(estimates) == 2 else "single_top_face",
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
            "pick_anchor_diagnostics": [anchor for anchor in calibration.get("anchors", [])
                if int(anchor["episode"]) == episode],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        cv_count = sum(block["status"] == "ok" for block in blocks)
        print(
            f"[replay-poses] ({ordinal}/{len(episodes)}) Episode {episode}: "
            f"{status}; Homographie {cv_count}/3.",
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
    calibrate.add_argument("--anchors", required=True)
    calibrate.add_argument("--out", required=True)
    calibrate.add_argument("--debug-dir", required=True)
    calibrate.add_argument("--cube-edge", type=float, default=DEFAULT_CUBE_EDGE_M)
    calibrate.add_argument(
        "--cube-center-z", type=float, default=DEFAULT_CUBE_CENTER_Z_M
    )
    calibrate.add_argument("--holdout-ratio", type=float, default=0.2)
    calibrate.add_argument("--seed", type=int, default=17)
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
    if args.command == "calibrate" and not 0.0 < args.holdout_ratio < 1.0:
        raise SystemExit("--holdout-ratio muss zwischen 0 und 1 liegen.")
    return {"inspect": run_inspect, "calibrate": run_calibrate, "poses": run_poses}[
        args.command
    ](args)


if __name__ == "__main__":
    raise SystemExit(main())
